import torch

@torch.inference_mode()
def strehl_after_gdd_noalloc(
    sim, A_out, D, *,
    t_mode="peak",              # "peak" | "centroid" | int(索引)
    search_defocus=True,        # 搜索最佳去焦项 e^{-i a r^2}
    n_alpha=81,
    alpha_span_factor=6.0,
    use_gpu=True,
    aperture_mask=None          # 可选：(Nx,Ny) 的 float 掩模（外光阑）
):
    """
    计算“压缩后”光束的 Strehl，比起先生成 A_comp：
      • 不在 CPU 保存 A_comp（CPU 不会因 A_comp 增长到 3 场）
      • GPU 峰值 ~ 两幅完整电场 + 少量 2D 缓冲
    返回: {"S": float, "tidx": int, "alpha": float}
    """
    dev = sim.T.device if (use_gpu and torch.cuda.is_available()) else torch.device("cpu")
    # 1) 搬到 GPU（不克隆，不改写原 CPU 张量）
    A = (A_out if torch.is_tensor(A_out) else torch.as_tensor(A_out)).to(
        dev, dtype=torch.complex64, non_blocking=True
    )
    Nx, Ny, Nt = A.shape
    dx, dy = float(sim.dx), float(sim.dy)

    # 2) 频域：一次 FFT（此刻内存峰值 ~ A + Â）
    _, Aw = sim.fft_t(A) 
    del A

    # 3) 乘 GDD 相位（频率向量采用中心偏移 sim.omega）
    w = sim.omega.to(dev, dtype=torch.float32)
    phase = torch.exp(0.5j * float(D) * (w ** 2)).to(Aw.dtype)
    Aw.mul_(phase)


    # 4) 逆变回时间域（此刻内存峰值 ~ Â + A^t）
    _, At = sim.ifft_t(Aw)  
    del Aw

    # 5) 选时间切片
    P_t = torch.sum(At.real * At.real + At.imag * At.imag, dim=(0, 1)) * dx * dy  # (Nt,)
    if t_mode == "peak":
        tidx = int(torch.argmax(P_t).item())
    elif t_mode == "centroid":
        T = sim.T.to(P_t.device)
        wgt = torch.trapz(P_t, T)
        t_cm = (torch.trapz(T * P_t, T) / (wgt + 1e-30)).item()
        dt = float(T[1] - T[0])
        tidx = int(max(0, min(Nt - 1, round((t_cm - float(T[0])) / dt))))
    else:
        tidx = int(t_mode)

    # 只保留该帧（2D），立即释放 3D 场，避免多占一幅
    Axy = At[..., tidx].clone()   # 克隆一帧 (Nx,Ny) 以便安全释放 At
    del At, P_t

    # 可选：外光阑
    if aperture_mask is not None:
        M = (aperture_mask if torch.is_tensor(aperture_mask) else torch.as_tensor(aperture_mask)).to(
            Axy.device, dtype=torch.float32, non_blocking=True
        )
        Axy = Axy * M  # 复数 × 实数，类型保持为 complex64

    # 6) Strehl（带去焦搜索）
    U_ideal = torch.sum(torch.abs(Axy)) * dx * dy  # 理想相位的归一化因子
    if U_ideal.abs().item() == 0.0:
        return {"S": 0.0, "tidx": tidx, "alpha": 0.0}

    if not search_defocus:
        U = torch.sum(Axy) * dx * dy
        S = (torch.abs(U) ** 2 / (U_ideal ** 2)).item()
        return {"S": S, "tidx": tidx, "alpha": 0.0}

    # 只建一个 2D 的 r^2；避免 Nx*Ny*n_alpha 的 3D 爆内存
    X = sim.X.to(Axy.device, dtype=torch.float32)
    Y = sim.Y.to(Axy.device, dtype=torch.float32)
    r2 = X * X + Y * Y
    r2max = float(torch.max(r2).item())
    alpha_max = alpha_span_factor / (r2max + 1e-30)
    alphas = torch.linspace(-alpha_max, alpha_max, n_alpha, device=Axy.device)

    best_val = torch.tensor(0.0, device=Axy.device)
    best_alpha = 0.0
    for a in alphas:
        U = torch.sum(Axy * torch.exp(-1j * a * r2)) * dx * dy
        val = torch.abs(U)
        if val > best_val:
            best_val = val
            best_alpha = float(a.item())

    S = (best_val ** 2 / (U_ideal ** 2)).item()
    return {"S": S, "tidx": tidx, "alpha": best_alpha}

import math
import torch

@torch.inference_mode()
def strehl_ratio_fullfield(
    sim,
    *,
    field: str = "output",     # 'output' | 'input' | 'current'
    t_mode: str | int = "peak",# 'peak' | 'centroid' | 指定时间索引
    search_defocus: bool = True,
    n_alpha: int = 81,
    alpha_span_factor: float = 6.0,
    use_gpu: bool = True,
    tile_x: int = 64,          # 分块宽度(沿 x)，控制显存峰值
):
    """
    直接对整幅口径(不需要 M0)计算 Strehl。内存友好：
      - CPU：不产生整场副本；保持 A_field 在 CPU。
      - GPU：流式分块 (tile_x, Ny)，只保留一个切片及临时相位，峰值 << 两个完整场。

    返回: dict = {'S': float, 'tidx': int, 'alpha': float}
    """
    # -------- 取场 --------
    if field == "output":
        A_field = sim.A_out
    elif field == "input":
        A_field = sim.A_in
    else:
        A_field = sim.A
    if A_field is None:
        raise ValueError(f"sim.{field} 为空")

    A = A_field if torch.is_tensor(A_field) else torch.as_tensor(A_field)
    assert A.ndim == 3, "A must be (Nx, Ny, Nt)"
    Nx, Ny, Nt = A.shape
    dx, dy = float(sim.dx), float(sim.dy)

    # 选择设备
    dev = (sim.T.device if (use_gpu and torch.cuda.is_available()) else torch.device("cpu"))

    # -------- 第一步：找时间切片 tidx（流式，不建整场临时）--------
    P_t = torch.zeros(Nt, dtype=torch.float32, device=dev)  # 仅 1xNt
    for xs in range(0, Nx, tile_x):
        xe = min(xs + tile_x, Nx)
        # 把一个 x-块搬到目标设备；dtype 统一成 complex64
        A_blk = A[xs:xe, :, :].to(dev, dtype=torch.complex64, non_blocking=True)
        # |A|^2 累加到时间功率：sum_{x,y} |A|^2
        # 不保留 I 全场，只对块求和后加到 P_t
        # pow_blk: (tx, Ny, Nt) -> sum over (0,1) -> (Nt,)
        pow_blk = (A_blk.real*A_blk.real + A_blk.imag*A_blk.imag).to(torch.float32)
        P_t += torch.sum(pow_blk, dim=(0,1)) * (dx * dy)
        del A_blk, pow_blk
    # 选时间索引
    if isinstance(t_mode, int):
        tidx = int(max(0, min(int(t_mode), Nt-1)))
    elif t_mode == "centroid":
        T = sim.T.to(P_t.device, dtype=torch.float32)
        w = torch.trapz(P_t, T)
        t_cm = (torch.trapz(T * P_t, T) / (w + 1e-30)).item()
        dt = float(T[1]-T[0])
        tidx = int(max(0, min(round((t_cm - float(T[0]))/dt), Nt-1)))
    else:  # 'peak'
        tidx = int(torch.argmax(P_t).item())

    # -------- 第二步：计算 U_ideal = ∫|A(x,y,t*)| dxdy（流式）--------
    U_ideal = 0.0
    for xs in range(0, Nx, tile_x):
        xe = min(xs + tile_x, Nx)
        Axy_blk = A[xs:xe, :, tidx].to(dev, dtype=torch.complex64, non_blocking=True)
        # 分块求 |A| 的和
        abs_blk = torch.sqrt(Axy_blk.real*Axy_blk.real + Axy_blk.imag*Axy_blk.imag)
        U_ideal += float(torch.sum(abs_blk).item())
        del Axy_blk, abs_blk
    U_ideal *= (dx * dy)
    if U_ideal == 0.0:
        return {"S": 0.0, "tidx": tidx, "alpha": 0.0}

    # 若不搜索去焦，直接算一次相干和
    if not search_defocus:
        U_real = 0.0
        U_imag = 0.0
        for xs in range(0, Nx, tile_x):
            xe = min(xs + tile_x, Nx)
            Axy_blk = A[xs:xe, :, tidx].to(dev, dtype=torch.complex64, non_blocking=True)
            U_real += float(torch.sum(Axy_blk.real).item())
            U_imag += float(torch.sum(Axy_blk.imag).item())
            del Axy_blk
        U_real *= (dx * dy); U_imag *= (dx * dy)
        S = (U_real*U_real + U_imag*U_imag) / (U_ideal*U_ideal)
        return {"S": float(S), "tidx": tidx, "alpha": 0.0}

    # -------- 第三步：去焦搜索  U(a) = ∫ A(x,y) e^{-i a r^2} dxdy  --------
    # 估计 r^2 最大值（不额外建场；直接用 sim.X/Y 的最大值）
    try:
        x_max = float(torch.max(torch.abs(sim.X)).detach().cpu().item())
        y_max = float(torch.max(torch.abs(sim.Y)).detach().cpu().item())
        r2max = x_max*x_max + y_max*y_max
    except Exception:
        # 兜底：用网格尺寸估计
        r2max = ((Nx-1)*dx/2)**2 + ((Ny-1)*dy/2)**2
    alpha_max = alpha_span_factor / (r2max + 1e-30)
    alphas = torch.linspace(-alpha_max, alpha_max, n_alpha, device=dev, dtype=torch.float32)

    best_absU = 0.0
    best_alpha = 0.0

    # 为了减少重复搬运，把几何 r^2 在“块作用域”内预计算一次，跨 alpha 循环复用
    for a in alphas:
        U_r = 0.0
        U_i = 0.0
        for xs in range(0, Nx, tile_x):
            xe = min(xs + tile_x, Nx)
            # 场块
            Axy_blk = A[xs:xe, :, tidx].to(dev, dtype=torch.complex64, non_blocking=True)
            # r^2 块（只保留在块作用域内）
            try:
                X_blk = sim.X[xs:xe, :].to(dev, dtype=torch.float32, non_blocking=True)
                Y_blk = sim.Y[xs:xe, :].to(dev, dtype=torch.float32, non_blocking=True)
                r2_blk = X_blk*X_blk + Y_blk*Y_blk
                del X_blk, Y_blk
            except Exception:
                # 兜底：如果没有 sim.X/Y，可用索引近似构造（很少用到）
                ix = torch.arange(xs, xe, device=dev, dtype=torch.float32) - (Nx-1)/2
                iy = torch.arange(0, Ny, device=dev, dtype=torch.float32) - (Ny-1)/2
                Xb = ix[:, None] * dx
                Yb = iy[None, :] * dy
                r2_blk = Xb*Xb + Yb*Yb

            phi = a * r2_blk                           # (tx,Ny), float32
            c = torch.cos(phi); s = torch.sin(phi)     # 两个块级临时
            Ar = Axy_blk.real; Ai = Axy_blk.imag       # 视图，不复制

            # (Ar + i Ai) * (cos - i sin):
            # real = Ar*cos + Ai*sin
            # imag = -Ar*sin + Ai*cos
            U_r += float(torch.sum(Ar*c + Ai*s).item())
            U_i += float(torch.sum(-Ar*s + Ai*c).item())

            del Axy_blk, r2_blk, phi, c, s

        U_r *= (dx * dy); U_i *= (dx * dy)
        absU = math.hypot(U_r, U_i)
        if absU > best_absU:
            best_absU = absU
            best_alpha = float(a.item())

    S = (best_absU * best_absU) / (U_ideal * U_ideal)

    # 清理 GPU 空闲块（不会影响 PyTorch 已在用的张量）
    if dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return {"S": float(S), "tidx": tidx, "alpha": best_alpha}

import numpy as np, torch, math
import matplotlib.pyplot as plt

@torch.inference_mode()
def analyze_phase_flatness(sim, *, field="output", t_mode="peak",
                           R_factor=3.0,w=0.004, fit_astig=True, plot=True):
    """
    分析在时间切片 t* 上的相位平整度（仅 3w 口径内）：
      - 拟合并去掉 piston, tip-tilt, defocus（和可选 astig）
      - 给出残余相位 RMS、Marechal 估计的 S，以及去除前/后真实 S
    返回: dict（含系数、RMS、S 等），并可绘图
    """
    # 1) 取场与时间切片
    if field == "output":   A = sim.A_out
    elif field == "input":  A = sim.A_in
    else:                   A = sim.A
    if A is None: raise ValueError(f"sim.{field} 为空")
    Nx, Ny, Nt = A.shape

    dx, dy = float(sim.dx), float(sim.dy)
    P_t = torch.sum(A.real*A.real + A.imag*A.imag, dim=(0,1)) * dx * dy
    if t_mode == "peak":
        tidx = int(torch.argmax(P_t).item())
    elif t_mode == "centroid":
        T = sim.T.to(P_t.device)
        wgt = torch.trapz(P_t, T)
        t_cm = (torch.trapz(T * P_t, T) / (wgt + 1e-30)).item()
        dt = float(T[1]-T[0]); tidx = int(max(0, min(round((t_cm - float(T[0]))/dt), Nt-1)))
    else:
        tidx = int(t_mode)

    Axy  = A[..., tidx].detach().to("cpu").numpy()
    X    = sim.X.detach().to("cpu").numpy()
    Y    = sim.Y.detach().to("cpu").numpy()

    # 2) 3w 圆孔径
    R = R_factor * float(w)           # w 已由你提前定义
    mask = (X*X + Y*Y) <= (R*R)

    amp   = np.abs(Axy)
    phase = np.angle(Axy)             # [-pi, pi]
    # 简单 2D 展相位（对高噪/强包络畸变时可失败；我们用能量权重来稳住拟合）
    phi_unw = np.unwrap(np.unwrap(phase, axis=0), axis=1)

    # 3) 加权最小二乘拟合低阶像差：1, x, y, (x^2+y^2), [x^2-y^2, 2xy]
    x = X[mask]; y = Y[mask]
    yvec = phi_unw[mask]
    wA   = (amp[mask]**2)             # 用强度作权重
    sw   = np.sqrt(np.maximum(wA, 1e-30))

    cols = [np.ones_like(x), x, y, (x*x + y*y)]
    names= ["piston", "tip_x", "tip_y", "defocus"]
    if fit_astig:
        cols += [(x*x - y*y), (2.0*x*y)]
        names+= ["astig_0/90", "astig_45"]

    M = np.vstack(cols).T                   # (Npix, nbasis)
    Mw = M * sw[:, None]; yw = yvec * sw
    coeffs, *_ = np.linalg.lstsq(Mw, yw, rcond=None)

    phi_fit = (M @ coeffs)                  # (Npix,)
    # 回填到全场，计算残差
    phi_fit_full = np.zeros_like(phi_unw); phi_fit_full[mask] = phi_fit
    phi_res = np.zeros_like(phi_unw);      phi_res[mask] = yvec - phi_fit

    # 去掉残差的加权均值（只剩高阶起伏）
    mean_res = np.average(phi_res[mask], weights=wA)
    phi_res[mask] -= mean_res

    # 4) 指标：残余 RMS、Marechal S、去除前/后的真实 Strehl
    sigma_phi = math.sqrt(np.average((phi_res[mask])**2, weights=wA))   # rad
    S_marechal = math.exp(-sigma_phi**2)

    # 真实 Strehl（只在口径内做相干和；比例里 dxdy 会相消，不必乘）
    U_ideal = np.sum(amp[mask])
    U_raw   = np.abs(np.sum(Axy[mask]))
    S_raw   = (U_raw / (U_ideal + 1e-30))**2

    A_corr = Axy * np.exp(-1j * phi_fit_full)  # 去掉低阶像差后的场
    S_corr = (np.abs(np.sum(A_corr[mask])) / (U_ideal + 1e-30))**2

    if plot:
        v = 3.0*sigma_phi
        fig, axs = plt.subplots(1,4, figsize=(18,4))
        im0 = axs[0].imshow(amp, origin='lower');               axs[0].set_title('|A|')
        im1 = axs[1].imshow(phi_unw, origin='lower');           axs[1].set_title('Unwrapped φ')
        im2 = axs[2].imshow(phi_fit_full, origin='lower');      axs[2].set_title('Low-order fit φ')
        im3 = axs[3].imshow(phi_res, origin='lower', vmin=-v, vmax=v); axs[3].set_title(f'Residual φ (RMS={sigma_phi:.2f} rad)')
        for im, ax in zip([im0,im1,im2,im3], axs):
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.suptitle(f'tidx={tidx} | S_raw={S_raw:.3f},  S_corr={S_corr:.3f},  S_Marechal≈{S_marechal:.3f}')
        plt.tight_layout(); plt.show()

    coeff_dict = {n: c for n, c in zip(names, coeffs)}
    return dict(tidx=tidx,
                sigma_phi=sigma_phi,
                S_raw=S_raw, S_corr=S_corr, S_marechal=S_marechal,
                coeffs=coeff_dict)

