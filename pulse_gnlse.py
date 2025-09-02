#!/usr/bin/env python3
# ==============================================================
#  Minimal working version of PulseGNLSESimulator
#  * keeps only the parts you need right now *
#    – stores t_fwhm correctly
#    – builds initial pulse that you can inspect
# ==============================================================

from __future__ import annotations

# 头部 import
import torch
from torch import fft as tfft     # torch.fft 别名
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
import psutil
import numpy as np
import matplotlib.pyplot as plt
from math import factorial, sqrt, log, pi
from typing import Optional, Callable, Dict, Sequence
from scipy.interpolate import interp1d
import os
from scipy import fft as sp_fft      # 后面如果切 CuPy 改这行即可
import pathlib
WORKERS = os.cpu_count()             # 也可改成你想用的核数
print(f'cpu counts: {WORKERS}')


class PulseGNLSESimulator:
    r"""
    Split-step (2 + 1)D + T Generalised NLSE simulator ― **initialisation only**

    Parameters
    ----------
    # --- temporal ---
    pulse_energy : float
        Pulse energy [J].
    t_fwhm : float
        Intensity FWHM of the input pulse [s].
    nt : int
        Temporal sample count.
    t_window_factor : float
        Time-window length = ``factor × t_fwhm``.

    # --- longitudinal ---
    L : float
        Physical propagation length [m].
    nz : int
        Split-step segments (uniform Δz).
    compression_steps : int
        Points in quadratic-phase scan for RMS-TBP compression.

    # --- carrier & dispersion ---
    lambda0 : float
        Carrier wavelength [m].
    dispersion : dict[int, float]
        Must include β₁ (order 1) and β₂ (order 2) at minimum, units sⁿ m⁻¹.

    # --- non-linear ---
    n2 : float | Sequence[tuple[float, float]]
        Kerr index - scalar or list of (λ, n₂) pairs (m, m²/W).
    raman : dict | None
        Raman parameters ``{'f_R', 'tau1', 'tau2'}``; ``None`` ⇒ off.

    # --- transverse grid ---
    Nx, Ny : int
        Sample points in x- & y-directions.
    x_window, y_window : float
        Physical window size (full width) in x & y [m].
    beam_profile : dict
        Initial transverse shape.
        • ``{'shape':'gaussian','waist':w0}``
        • ``{'shape':'flat','radius':R}``
        • ``{'shape':'custom','func': callable}``

    # --- misc / optional ---
    alpha : float, default 0.0
        Linear attenuation coefficient α [m⁻¹].
    n0 : float, default 1.0
        Linear refractive index (=> k₀ = n₀ ω₀/c).
    pml_width : float, default 0.0
        Absorbing boundary (PML) thickness [m]; 0 ⇒ none.
    device : {'cpu','cuda','auto'}, default 'cpu'
        Execution backend selector.
    retarded_frame : bool, default False
        ``False`` ⇒ laboratory frame (β₁ active);
        ``True``  ⇒ moving frame (β₁ dropped internally).
    """
    # ------------------------------------------------------------------
    def __init__(
        self,
        *,
        pulse_energy: float,
        t_fwhm: float,
        nt: int,
        t_window_factor: float,
        L: float,
        nz: int,
        compression_steps: int,
        lambda0: float,
        dispersion: dict[int, float],
        n2,
        Nx: int,
        Ny: int,
        x_window: float,
        y_window: float,
        beam_profile: dict,
        raman: dict | None = None,
        alpha: float = 0.0,
        noise: dict | None = None,
        n0: float = 1.0,
        pml_width: float = 0.0,
        device: str = "cpu",
        retarded_frame: bool = False,
        generate_A_in: bool = True,
        
    ) -> None:

        # -------- store basic inputs ----------
        Rc= None
        self.pulse_energy   = float(pulse_energy)
        self.t_fwhm         = float(t_fwhm)        # ★ stored for later use
        self.nt             = int(nt)
        self.t_window_factor= float(t_window_factor)
        self.beam_profile   = beam_profile
        self.noise_cfg = noise or {}

        
        # ---------- basic constants ----------------------------------
        self.c = 2.99792458e8                # [m s⁻¹]
        self.lambda0 = float(lambda0)
        self.omega0  = 2*pi*self.c/self.lambda0
        self.k0      = n0*self.omega0/self.c
        self.n0      = n0
        self.alpha   = float(alpha)
        self.device  = device.lower()
        

        
        # ---------- store propagation settings -----------------------
        self.L  = float(L)
        self.nz = int(nz)
        self.dz = self.L / self.nz
        self.retarded = bool(retarded_frame)

        # ---------- dispersion dict validation -----------------------
        self.dispersion = {}
        for order, beta in dispersion.items():
            self.dispersion[int(order)] = float(beta)
        if not self.retarded and 1 not in self.dispersion:
            raise ValueError("laboratory frame requires β₁ (order 1) in dispersion")
        if 2 not in self.dispersion:
            raise ValueError("dispersion must at least contain β₂ (order 2)")

        self.n0      = n0
        R = ((self.n0 - 1)/(self.n0 + 1))**2  # ~0.035
        self.T_surf = 1 - R

        # ---------- time grid ----------------------------------------
        self.nt = int(nt)
        T_lim   = t_window_factor * t_fwhm / 2          # ± limit
        self.T  = torch.linspace(-T_lim, +T_lim, self.nt, dtype=torch.float32, device=DEVICE)
        self.dT = float(self.T[1] - self.T[0])
        # frequency / angular-frequency grids
        f = tfft.fftfreq(self.nt, d=self.dT, device=DEVICE)   # ★ 不再 fftshift
        self.omega   = 2 * torch.pi * f              # [rad/s]
        self.domega  = float(self.omega[1] - self.omega[0])
        self.freq    = self.omega / (2*torch.pi)

        # ---------- transverse grid ----------------------------------
        self.Nx, self.Ny = int(Nx), int(Ny)
        self.x = torch.linspace(-x_window/2, +x_window/2, self.Nx,
                                dtype=torch.float32, device=DEVICE)
        self.y = torch.linspace(-y_window/2, +y_window/2, self.Ny,
                                dtype=torch.float32, device=DEVICE)
        self.dx = float(self.x[1] - self.x[0])
        self.dy = float(self.y[1] - self.y[0])
        # 2-D mesh & k-space grids
        self.X, self.Y = torch.meshgrid(self.x, self.y, indexing='ij')
        kx = 2 * torch.pi * tfft.fftfreq(self.Nx, d=self.dx, device=DEVICE)   # ★
        ky = 2 * torch.pi * tfft.fftfreq(self.Ny, d=self.dy, device=DEVICE)   # ★
        self.Kx, self.Ky = torch.meshgrid(kx, ky, indexing='ij')              # ★ 取消 fftshift
        self.K_perp2 = self.Kx**2 + self.Ky**2

        self._scale_fft_t   = (self.dT / np.sqrt(2*np.pi))
        self._scale_ifft_t  = (self.nt * self.domega / np.sqrt(2*np.pi))
        self._scale_fft_xy  = (self.dx * self.dy) / (2*np.pi)
        self._scale_ifft_xy = (2*np.pi) / (self.dx * self.dy)
        
        # 转成与数据同 dtype/device 的标量张量（避免每次构建）
        to_scalar = lambda v, dev=DEVICE: torch.tensor(v, dtype=torch.float32, device=dev)
        self._scale_fft_t   = to_scalar(self._scale_fft_t)
        self._scale_ifft_t  = to_scalar(self._scale_ifft_t)
        self._scale_fft_xy  = to_scalar(self._scale_fft_xy)
        self._scale_ifft_xy = to_scalar(self._scale_ifft_xy)


        # ---------- γ(ω) 全频矢量 ------------------------------------
        omega_abs = self.omega + self.omega0          # ← 提前放到 if 外

        if np.isscalar(n2):
            n2_vec = float(n2) * torch.ones_like(self.omega, device=DEVICE)
        else:
            lam_pts, n2_pts = np.asarray(n2).T
            n2_interp = interp1d(lam_pts, n2_pts, kind='linear',
                                  bounds_error=False,
                                  fill_value=(n2_pts[0], n2_pts[-1]))
            lam_grid = (2*torch.pi*self.c / omega_abs.cpu()).numpy()
            n2_vec = torch.from_numpy(n2_interp(lam_grid)).to(DEVICE)

        # γ(ω) — 与 fftshift 后的 ω 对齐，形状 (1,1,Nt)
        self.gamma_w = (omega_abs * n2_vec / self.c).to(torch.float32)[None, None, :].to(torch.complex64)
        
        # 中心频率那一点留作诊断 / B-integral
        self.gamma0  = float(self.gamma_w[0, 0, 0])

        # ---------- Raman preparation --------------------------------
        self.raman_on = raman is not None
        if self.raman_on:
            self.f_R  = float(raman["f_R"])
            self.tau1 = float(raman["tau1"])
            self.tau2 = float(raman["tau2"])
            # build normalised Raman kernel h_R(t)
            h_R   = torch.zeros_like(self.T) 
            mask = self.T >= 0
            coeff = (self.tau1**2 + self.tau2**2) / (self.tau1*self.tau2**2)
            h_R[mask] = coeff \
                        * torch.exp(-self.T[mask]/self.tau2) \
                        * torch.sin(self.T[mask]/self.tau1)
            norm = torch.trapz(h_R, self.T)
            h_R  = h_R / norm

            # ④ 频域表示 —— 直接调用已改好的 torch 版 self.fft_t
            _, H_R = self.fft_t(h_R)          # 返回 torch.complex64, cuda
            self.H_R_omega = H_R              # (Nt,) → 后续广播时会加维
        if self.raman_on:
            print('raman on')


        
        # ---- stepped film placeholder (route A geometry only at this step) ----
        self.stepped_film: SteppedFilm | None = None
        
        # === Route A: 定义参考介质(真空)与薄膜材料的参数 ===
        self.n_ref = 1.0                               # 参考介质 = 真空
        self.k0_ref = self.omega0 / self.c             # k_ref = ω0/c
        
        # 参考色散 β^ref（真空）：β1 = 1/c；m≥2 ~ 0（可按需扩展）
        beta_ref = {1: 1.0/self.c}
        # 若你想显式置零以清晰： for m in (2,3,4): beta_ref[m] = 0.0
        
        # 薄膜材料色散 β^film 使用现有 self.dispersion（已在上文校验）
        beta_film = dict(self.dispersion)
        
        self.materials = dict(
            n_ref=self.n_ref,
            n_film=self.n0,
            beta_ref=beta_ref,
            beta_film=beta_film,
            alpha_ref=0.0,
            alpha_film=self.alpha,
        )


        

        # ---------- pre-compute half-step linear operator -------------
        # build ω & k_perp^2 arrays with broadcast shape (Nx,Ny,Nt)
        # —— 预计算线性半步的两个因子（分离存储） ——
        # 空间因子：依赖 kx, ky
        self.Lxy_half = torch.exp(
            (-1j * self.K_perp2 / (2 * self.k0)) * (self.dz / 2)
        ).to(torch.complex64)                               # (Nx, Ny)
        
        # 时间因子：依赖 ω 与 β_m、α
        Om = self.omega.to(torch.complex128)               # (Nt,)
        disp_phase = torch.zeros_like(Om, dtype=torch.complex128)
        for order, beta in self.dispersion.items():
            if self.retarded and order == 1:
                continue
            disp_phase += beta * (Om**order) * ((-1)**order) / factorial(order)
        
        self.Lw_half = torch.exp(
            (1j * disp_phase - self.alpha/2) * (self.dz / 2)
        ).to(torch.complex64)                               # (Nt,)

        
        
        # —— 一次性预计算，后续重复使用（避免每步 dtype 转换 & 重建） ——
        self.gamma_w_even128 = (0.5*(self.gamma_w + torch.flip(self.gamma_w, dims=[-1]))).to(torch.complex128)
        if self.raman_on:
            self.H_R_omega128 = self.H_R_omega.to(torch.complex128)
        self.omega128 = self.omega.to(torch.complex128)
        
        # ---------- other bookkeeping --------------------------------
        if generate_A_in:
            self.A_in_gpu = self._generate_initial_pulse(
            chirp = 0.0,                             # 可选初始啁啾
            extra_spatial_phase = lambda X,Y:        # 可选波前曲率
                0.5 * self.k0 / Rc * (X**2 + Y**2) if Rc is not None else 0.0
            )
            self.A_in = self.A_in_gpu.cpu()                     # 只保留 CPU 副本
            del self.A_in_gpu                                    # 立刻释放 GPU 显存
            torch.cuda.empty_cache()
        else:
            self.A_in = None
            
        # ---- 记录初始 k-空间一维 FWHM（kx/ky），依赖现有 diagnose() ----
        # 说明：diagnose(domain='kspace') 在内部会调用 self.fft_xy(self.A)，
        # 因此这里临时把 self.A 指向 GPU 版本的 A_in，计算完再还原并清空缓存。
        try:
            A_saved = getattr(self, "A", None)
            self.A  = self.A_in.to(self.T.device, non_blocking=True)  # 临时 GPU 拷贝
            diag_k  = self.diagnose(domain='kspace', metrics=('fwhm',), save_raw=False)
            self.kx_fwhm0 = float(diag_k['metrics']['fwhm']['kx'])   # [m^-1]
            self.ky_fwhm0 = float(diag_k['metrics']['fwhm']['ky'])   # [m^-1]
            self.k_fwhm0  = 0.5 * (self.kx_fwhm0 + self.ky_fwhm0)    # 作为代表量
        finally:
            self.A = A_saved
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(self.k_fwhm0)
        self.k_fwhm0=600


        # === 2.1 空间参考系数：_coef_xy_ref (Nx,Ny) ===
        self._coef_xy_ref = (-1j * self.K_perp2 / (2.0 * self.k0_ref)).to(torch.complex64)
        
        # === 2.2 时间参考系数：_coef_w_ref (Nt,) ===
        Om128   = self.omega.to(torch.complex128)
        disp_ref = torch.zeros_like(Om128, dtype=torch.complex128)
        for m, beta in self.materials["beta_ref"].items():
            if self.retarded and m == 1:   # 共动坐标下跳过 β1_ref
                continue
            disp_ref += beta * ((-1)**m) * (Om128**m) / factorial(m)
        self._coef_w_ref = (1j * disp_ref - self.materials["alpha_ref"]/2.0).to(torch.complex64)


        
        # === 2.3 增量系数：_coef_w_inc (Nt,) —— 注意 Δβ1 必须保留！ ===
        # 增量频域多项式（含 m=1 在内的所有 Δβm）
        disp_inc = torch.zeros_like(Om128, dtype=torch.complex128)
        orders = set(self.materials["beta_film"].keys()) | set(self.materials["beta_ref"].keys())
        for m in orders:
            if m >= 1:  # m=1 自动包含 Δβ1；m>=2 为更高阶
                bf = float(self.materials["beta_film"].get(m, 0.0))
                br = float(self.materials["beta_ref"].get(m, 0.0))  # 真空：beta_ref[1]=1/c, 其余≈0
                disp_inc += (bf - br) * ((-1)**m) * (Om128**m) / factorial(m)
        
        # Δβ0 与 Δα
        delta_beta0 = (self.omega0 / self.c) * (self.materials["n_film"] - self.materials["n_ref"])

        delta_alpha = self.materials["alpha_film"] - self.materials["alpha_ref"]
        
        # 最终增量系数（complex64 以省显存/算力）
        self._coef_w_inc = (1j * (delta_beta0 + disp_inc) - delta_alpha/2.0).to(torch.complex64)

        # === 若时间参考算子恒等，就可以在传播时跳过两次 t-FFT ===
        with torch.no_grad():
            # 这里的阈值可以按数值需求微调
            self._time_ref_is_identity = (torch.max(torch.abs(self._coef_w_ref)).item() < 1e-14)


        
        # 工作副本只在 GPU
        self.A = self.A_in     # 传播前搬回 GPU
        self.A_out = None  
        self.B_running=0
        self.B_log= []
        self.spatial_w=[]
        self.spatial_freq_w=[]


    # ---------- temporal ----------
    @torch.inference_mode()
    def fft_t(self, A_xt):
        A_w = tfft.fft(A_xt, dim=-1)
        A_w.mul_(self._scale_fft_t)
        return self.omega, A_w
    
    @torch.inference_mode()
    def ifft_t(self, A_w):
        A_t = tfft.ifft(A_w, dim=-1)
        A_t.mul_(self._scale_ifft_t)
        return self.T, A_t

        
    # ---------- spatial ----------
    @torch.inference_mode()
    def fft_xy(self, A_xyt):
        A_k = tfft.fftn(A_xyt, dim=(0,1))
        A_k.mul_(self._scale_fft_xy)
        return self.Kx, self.Ky, A_k
    
    @torch.inference_mode()
    def ifft_xy(self, A_kxt):
        A_xy = tfft.ifftn(A_kxt, dim=(0,1))
        A_xy.mul_(self._scale_ifft_xy)
        return self.X, self.Y, A_xy


    def fft_xy_test(self, A_xyt: torch.Tensor):
        device = A_xyt.device
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    
        before_alloc = torch.cuda.memory_allocated(device)
        before_reserved = torch.cuda.memory_reserved(device)
        t0 = time.time()
    
        # 分步做，以便在关键环节打点
        A1 = tfft.ifftshift(A_xyt, dim=(0,1))
        mid1 = torch.cuda.memory_allocated(device)
        A2 = tfft.fftn(A1, dim=(0,1))
        del A1
        torch.cuda.empty_cache()
        mid2 = torch.cuda.memory_allocated(device)
        A3 = tfft.fftshift(A2, dim=(0,1))
        del A2
        torch.cuda.empty_cache()
        result = A3 * (self.dx * self.dy) / (2*torch.pi)
        del A3
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        after_alloc = torch.cuda.memory_allocated(device)
        peak_alloc = torch.cuda.max_memory_allocated(device)
        t1 = time.time()
    
        print(f"[FFT_xy]  time: {t1-t0:.3f}s")
        print(f"  before alloc:   {human(before_alloc)}")
        print(f"  after ifftshift: {human(mid1)}")
        print(f"  after fftn:     {human(mid2)}")
        print(f"  after fftshift: {human(after_alloc)}")
        print(f"  peak alloc:     {human(peak_alloc)}")
        print(f"  reserved:       {human(torch.cuda.memory_reserved(device))}")
        print("––––––––––––––––––––––––––––––––––––––––––")
    
        return self.Kx, self.Ky, result


    # =================================================================
    #  Build stepped-film geometry (ROUTE A, geometry only)
    # =================================================================
    def set_stepped_film_from_annuli(
        self,
        *,
        inner_radii: Sequence[float],     # len K, meters; inner_radii[0]=None for full disk
        dL_list: Sequence[float],         # len K, meters (layer thicknesses)
        nz_per_layer: Sequence[int] | int,  # len K or scalar
        dr_fwhm: float,                   # edge smoothing FWHM [m]
        # --- 新增：自动光阑参数 ---
        aperture_radius: float | None = None,   # 若 None 且 aperture_factor 不为 None → 自动设为 aperture_factor * w
        aperture_factor: float | None = 3.0,    # 缺省 = 3w；设为 None 则不加外光阑
        aperture_fwhm: float | None = None,     # 软边缘 FWHM；默认跟 dr_fwhm 一样
        names: Sequence[str] | None = None,
        to_device: torch.device | None = None,
    ) -> SteppedFilm:
        """
        Construct self.stepped_film with K layers.
        若 aperture_radius is None 且 aperture_factor 给出，则自动使用 aperture_factor * w 作为外光阑。
        """
        K = len(dL_list)
        assert len(inner_radii) == K, "inner_radii and dL_list must match length"
    
        if isinstance(nz_per_layer, int):
            nz_list = [int(nz_per_layer)] * K
        else:
            assert len(nz_per_layer) == K, "nz_per_layer must be int or length-K"
            nz_list = [int(z) for z in nz_per_layer]
    
        dev = to_device or self.X.device
    
        # ------- 自动外光阑（默认 3w） -------
        if aperture_radius is None and (aperture_factor is not None):
            w = self._infer_beam_radius()
            aperture_radius = float(aperture_factor) * float(w)
        # 不要超过网格边界
        if aperture_radius is not None:
            r_grid = float(min(self.x.abs().max(), self.y.abs().max()))
            aperture_radius = float(min(aperture_radius, r_grid * 0.98))
        if aperture_fwhm is None:
            aperture_fwhm = dr_fwhm
    
        # ------- 生成各层掩膜 -------
        masks = make_annular_masks(
            self.X.to(dev), self.Y.to(dev),
            inner_radii=inner_radii,
            dr_fwhm=dr_fwhm,
            aperture_radius=aperture_radius,
            aperture_fwhm=aperture_fwhm,
        )
    
        layers: List[LayerSpec] = []
        for j in range(K):
            nm = "" if names is None else str(names[j])
            layers.append(LayerSpec(dL=float(dL_list[j]), nz=int(nz_list[j]),
                                    mask=masks[j].to(torch.float32), name=nm))
        film = SteppedFilm(layers)
        self.stepped_film = film
        return film

    def _infer_beam_radius(self, fallback: float | None = None) -> float | None:
        """返回 1/e^2 半径 w（米）。优先从 beam_profile['waist']，否则回退到网格尺寸估计。"""
        try:
            shape = str(self.beam_profile.get("shape", "gaussian")).lower()
            if shape in ("gaussian", "supergauss", "supergaussian"):
                w = float(self.beam_profile["waist"])
                if w > 0:
                    return w
        except Exception:
            pass
        if fallback is not None:
            return fallback
        # 回退：取网格半宽的 0.4 倍，保证不会超过窗口
        r_grid = float(min(self.x.abs().max(), self.y.abs().max()))
        return 0.4 * r_grid
    

    
    # =================================================================
    #  GPU 版 _generate_initial_pulse
    # =================================================================
    @torch.inference_mode()
    def _generate_initial_pulse(
        self,
        *,
        chirp: float = 0.0,
        extra_spatial_phase: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
    ) -> torch.Tensor:
        """
        Build A(x,y,t) on GPU so that ∭|A|² dx dy dt = pulse_energy.
        Returns
        -------
        torch.Tensor
            dtype = complex64, device = DEVICE, shape (Nx, Ny, Nt)
        """
    
        # ---------- temporal envelope (normalised) ------------------
        t0  = self.t_fwhm / (2 * torch.sqrt(torch.log(torch.tensor(2.0))))
        A_t = torch.exp(-(self.T**2) / (2 * t0**2))          # float32
        if chirp != 0.0:
            A_t = A_t.to(torch.complex64) * torch.exp(1j * chirp * self.T**2)
        else:
            A_t = A_t.to(torch.complex64)
        int_t = torch.sqrt(torch.tensor(torch.pi)) * t0       # ∫|A_t|² dt  (float32)
    
        # ---------- spatial envelope (normalised) -------------------
        shape = self.beam_profile.get("shape", "gaussian").lower()
        if shape == "import":
            # 1) 取出现成场
            field = self.beam_profile["field"]          # numpy / torch / …
            if isinstance(field, np.ndarray):
                A0 = torch.from_numpy(field)
            else:
                A0 = field
            if not torch.is_tensor(A0):
                raise TypeError("beam_profile['field'] must be numpy array or torch tensor")
    
            # 2) 形状校验
            if A0.shape != (self.Nx, self.Ny, self.nt):
                raise ValueError(
                    f"Imported field shape {A0.shape} ≠ simulator grid {(self.Nx, self.Ny, self.nt)}"
                )
    
            # 3) 搬到目标设备 & dtype
            A0 = A0.to(dtype=torch.complex64, device=DEVICE).clone()
    
            # 4) 可选能量重标定（默认重标）
            if self.beam_profile.get("renorm", True):
                E_file = torch.trapz(
                    torch.trapz(torch.trapz(torch.abs(A0) ** 2, self.T), self.y), self.x
                )
                scale = torch.sqrt(self.pulse_energy / E_file)
                A0 *= scale
    
            return A0   # ← 直接返回，后面分支全部跳过    

        if shape == "flat":
            R = float(self.beam_profile["radius"])
            A_xy = torch.where(
                self.X**2 + self.Y**2 <= R**2, 1.0, 0.0
            ).to(torch.complex64)
    
        elif shape == "gaussian":
            w0   = float(self.beam_profile["waist"])
            A_xy = torch.exp(-(self.X**2 + self.Y**2) / w0**2).to(torch.complex64)
    
        elif shape in {"supergauss", "supergaussian"}:
            w0   = float(self.beam_profile["waist"])
            n    = int(self.beam_profile.get("order", 8))
            A_xy = torch.exp(
                -((self.X**2 + self.Y**2) / w0**2) ** (n / 2)
            ).to(torch.complex64)
    
        elif shape == "custom":
            if "func" in self.beam_profile:
                A_xy = self.beam_profile["func"](self.X, self.Y).to(torch.complex64)
            elif "amp" in self.beam_profile:
                amp_np = np.asarray(self.beam_profile["amp"], dtype=np.complex64)
                if amp_np.shape != self.X.shape:
                    raise ValueError("beam_profile['amp'] shape must match (Nx,Ny)")
                A_xy = torch.from_numpy(amp_np).to(self.X.device)
            else:
                raise ValueError("custom shape needs 'func' or 'amp' key")
    
        else:
            raise ValueError("shape must be flat | gaussian | supergauss | custom")
    
        # ---------- optional noise ----------------------------------
        noise = self.noise_cfg
        if noise:
            sig_amp   = float(noise.get("sigma_amp",   0.0))
            sig_phase = float(noise.get("sigma_phase", 0.0))
    
            if sig_amp > 0.0:
                A_xy *= 1.0 + torch.randn_like(A_xy.real) * sig_amp
            if sig_phase > 0.0:
                A_xy *= torch.exp(1j * torch.randn_like(A_xy.real) * sig_phase)
    
            # ---- coherent low‑order modes --------------------------
            if "coherent_modes" in noise and noise.get("coh_level", 0.0) > 0.0:
                Nm  = int(noise["coherent_modes"])
                lvl = float(noise["coh_level"])
                coh_map = torch.zeros_like(A_xy.real)
                rng = torch.Generator(device=DEVICE)
                for _ in range(Nm):
                    kx = torch.empty(1, device=DEVICE).uniform_(0.5, 3.0)
                    ky = torch.empty(1, device=DEVICE).uniform_(0.5, 3.0)
                    phi = torch.empty(1, device=DEVICE).uniform_(0, 2*torch.pi)
                    kx *= torch.pi / self.beam_profile["waist"]
                    ky *= torch.pi / self.beam_profile["waist"]
                    coh_map += torch.cos(kx * self.X + ky * self.Y + phi)
                coh_map -= coh_map.mean()
                coh_map  = lvl * coh_map / coh_map.std()
                A_xy *= 1.0 + coh_map
    
        # ---------- normalise spatial power -------------------------
        area_xy = torch.trapz(
            torch.trapz(torch.abs(A_xy)**2, self.y), self.x
        )                               # float32
        A_xy /= torch.sqrt(area_xy)     # now ∫∫|A_xy|² dx dy = 1
    
        if extra_spatial_phase is not None:
            phase = extra_spatial_phase(self.X, self.Y)
            if not torch.is_tensor(phase):                       # ← 新增
                phase = torch.zeros_like(self.X) + phase         # ← 新增
            A_xy *= torch.exp(1j * phase)

        # ---------- scale to desired pulse energy -------------------
        A_scale = torch.sqrt(self.pulse_energy / int_t)      # scalar (float32 / cpu ok)
        A0 = A_scale * (A_xy[..., None] * A_t[None, None, :])
        return A0.to(torch.complex64)        # shape (Nx,Ny,Nt), gpu



    def _Lxy_ref_half(self, dz_half: float):
        return torch.exp(self._coef_xy_ref * float(dz_half))          # (Nx,Ny)

    def _Lw_ref_half(self, dz_half: float):
        return torch.exp(self._coef_w_ref * float(dz_half))           # (Nt,)
    
    def _amp_increment_half(self, dz_half: float):
        return torch.exp(self._coef_w_inc * float(dz_half))           # (Nt,)

    @torch.inference_mode()
    def _space_ref_half(self, dz_half: float) -> None:
        """参考介质的空间半步：A(x,y,t) -> A(kx,ky,t) 乘以 exp(_coef_xy_ref*dz/2) -> 回来。"""
        # (x,y,t) → (kx,ky,t)
        _, _, A_k = self.fft_xy(self.A)
        # 乘以空间半步（与 t 无关，广播到 Nt）
        self.A = None
        torch.cuda.empty_cache()
        A_k *= self._Lxy_ref_half(dz_half)[..., None]  # (Nx,Ny,1)
        # (kx,ky,t) → (x,y,t)
        _, _, self.A = self.ifft_xy(A_k)
    

    @torch.inference_mode()
    def _time_ref_half(self, dz_half: float, *, eps: float = 1e-14) -> None:
        """参考介质的时间半步：A(x,y,t) -> A(x,y,ω) 乘以 exp(_coef_w_ref*dz/2) -> 回来。"""
        # 可选：如果本半步几乎恒等（例如真空 + retarded_frame 去掉 β1），就跳过
        max_mag = torch.max(torch.abs(self._coef_w_ref * float(dz_half))).item()
        if max_mag < eps:
            return
        # (x,y,t) → (x,y,ω)
        _, A_w = self.fft_t(self.A)
        # 乘以时间半步（广播到 Nx,Ny）
        self.A = None
        torch.cuda.empty_cache()
        A_w *= self._Lw_ref_half(dz_half)[None, None, :]  # (1,1,Nt)
        # (x,y,ω) → (x,y,t)
        _, self.A = self.ifft_t(A_w)


    @torch.inference_mode()
    def _apply_incremental_linear_half(
        self,
        Mj: torch.Tensor,        # (Nx,Ny) float32
        dz_half: float,
        *,
        eps: float = 0.0,        # 可选：算子近似恒等时跳过
    ) -> None:
        """
        Route-A：增量线性半步（不切块，一次性在整块 (Nx,Ny,Nt) 上执行）。
        频域更新公式（对每个 Ω）：
            Â_new = Â_old * [ (1-M) + M * amp(Ω) ]
                  = Â_old + M * (amp(Ω)-1) * Â_old
        其中 amp(Ω) = exp(_coef_w_inc(Ω) * dz_half)，M = Mj(x,y) ∈ [0,1] 软掩膜。
        """
    
        dev = self.T.device
    
        # 1) 预备：掩膜与“增量算子”
        Mj = Mj.to(device=dev, dtype=torch.float32, non_blocking=True)   # (Nx,Ny)
        amp = self._amp_increment_half(dz_half)                           # (Nt,) complex64
    
        # 可选：如果本半步几乎恒等，直接跳过（节省 FFT 与显存）
        if eps > 0.0:
            if torch.max(torch.abs((amp - 1.0))).item() < eps:
                return
    
        # 2) (x,y,t) → (x,y,ω)
        _, A_w = self.fft_t(self.A)             # A_w: (Nx,Ny,Nt) complex64
        # 释放时域 A，降低峰值显存（关键一步）
        self.A = None
        torch.cuda.empty_cache()
    
        # 3) 频域内“只对膜内区域乘 amp”，对膜外保持不变
        #    写成“拼回”形式以减少中间大张量：
        #    A_w_masked = A_w * amp
        #    A_w        = (1-M)*A_w + M*A_w_masked
        A_w_masked = A_w * amp[None, None, :]                         # (Nx,Ny,Nt)
        # 先把 A_w 变成 “外部份”： (1-M)*A_w
        A_w.mul_((1.0 - Mj)[..., None].to(torch.float32))             # in-place
        # 把膜内部分加回： M * A_w_masked
        A_w_masked.mul_(Mj[..., None].to(torch.float32))              # in-place
        A_w.add_(A_w_masked)                                          # in-place 累加
        del A_w_masked
        torch.cuda.empty_cache()
        # 4) (x,y,ω) → (x,y,t)
        _, self.A = self.ifft_t(A_w)                                     # (Nx,Ny,Nt)e()
 


    
    @torch.inference_mode()
    def _kerr_step_with_film(self, dz: float, Mj: torch.Tensor, *,
                             tile_x: int = 32, empty_cache_every: int = 0):
        """
        Kerr/Raman/shock 非线性步（按 x‑tile 流式）。本函数假定非线性仅在薄膜区域生效，
        因此始终使用传入的软掩膜 Mj(x,y) ∈ [0,1]：
            A *= exp(i * [Mj * φ(t)] * dz)         （SPM 纯相位）
            A -= Mj * (γ0/ω0) * dz * ∂t[A(t) N(t)] （shock 校正）
        """
        with torch.no_grad():
            # --- 常量/缓存 ---
            Nx, Ny, Nt = self.Nx, self.Ny, self.nt
            dev = self.T.device
    
            # Shock 系数：k_shock = (γ0/ω0) * dz
            k_shock = (self.gamma0 / self.omega0) * dz
    
            # 频域核（已在 __init__ 里准备好，复杂步骤里用 complex128 更稳）
            gamma_even = self.gamma_w_even128   # (1,1,Nt) complex128，偶对称 γ(ω)
            omega128   = self.omega128          # (Nt,)    float64/complex128
            H_R        = self.H_R_omega128 if self.raman_on else None
    
            # 掩膜：固定在 GPU&float32；每个 tile 取切片广播到时间轴
            Mj = Mj.to(device=dev, dtype=torch.float32, non_blocking=True)
    
            tcount = 0
            for x0 in range(0, Nx, tile_x):
                xs = slice(x0, min(x0 + tile_x, Nx))
    
                # ===== 取出一个 tile，并升精度 =====
                A_blk = self.A[xs, :, :].to(torch.complex128)      # (tx,Ny,Nt)
    
                # ===== (1) 频域里构造 N̂(ω) = 𝓕_t{|A|^2}（含 Raman）=====
                I_hat = self.fft_t(torch.abs(A_blk)**2)[1]         # (tx,Ny,Nt)
                if H_R is not None:
                    # ((1-fR) + fR H_R) * I_hat —— 原地写法，省一次临时
                    I_hat *= (1.0 - self.f_R)
                    I_hat += (self.f_R * H_R) * I_hat / (1.0 - self.f_R)
    
                # ===== (2) SPM：φ(t) = 𝓕⁻¹{γ_even · N̂}（实数），只在膜内生效 =====
                phi_t = self.ifft_t(I_hat * gamma_even)[1].real     # (tx,Ny,Nt) float64
                M_blk = Mj[xs, :].to(dtype=torch.float32)           # (tx,Ny)
                phase = (phi_t * dz) * M_blk[..., None]             # (tx,Ny,Nt) float32/64
                A_blk *= torch.exp(1j * phase)                      # 纯相位（膜外 phase=0）
                del phi_t, phase
    
                # ===== (3) Shock：dA_shock = (γ0/ω0) dz ∂t(A N)，再乘膜掩膜 =====
                N_t = self.ifft_t(I_hat)[1].real                    # (tx,Ny,Nt) float64
                del I_hat
    
                F_AN = self.fft_t(A_blk * N_t)[1]                   # (tx,Ny,Nt) complex128
                F_AN *= k_shock                                     # 乘 (γ0/ω0)·dz
                dA_shock = self.ifft_t(1j * omega128 * F_AN)[1]     # (tx,Ny,Nt)
                del F_AN, N_t
    
                A_blk -= dA_shock * M_blk[..., None]                # 只在膜内扣除 shock
                del dA_shock, M_blk
    
                # ===== (4) 回写（降回 complex64） =====
                self.A[xs, :, :] = A_blk.to(torch.complex64)
                del A_blk
    
                tcount += 1
                if empty_cache_every and (tcount % empty_cache_every == 0):
                    torch.cuda.empty_cache()
                    
    @torch.inference_mode()
    def _apply_global_time_shift(self, dt: float) -> None:
        """全局时间平移：A(t) ← A(t+dt)；频域乘 e^{+iΩ dt}。"""
        _, A_w = self.fft_t(self.A)
        self.A=None
        torch.cuda.empty_cache()
        phase = torch.exp(1j * self.omega.to(A_w.dtype) * float(dt))
        A_w *= phase[None, None, :]
        del phase
        torch.cuda.empty_cache()
        _, self.A = self.ifft_t(A_w)
    
    def _build_exclusive_region_masks(self) -> list[torch.Tensor]:
        """
        把 stepped_film 累积掩膜变成互斥分区：
          Region0=中心区，RegionK-1=最外区，中间为各环带。
        返回 float32, device=self.T.device 的列表。
        """
        assert self.stepped_film is not None and len(self.stepped_film.layers) > 0
        dev = self.T.device
        cum = [L.mask.to(device=dev, dtype=torch.float32, non_blocking=True) for L in self.stepped_film.layers]
        K = len(cum)
        excl = []
        if K == 1:
            excl = [cum[0]]
        else:
            for j in range(K - 1):
                excl.append(torch.clamp(cum[j] - cum[j + 1], 0.0, 1.0))
            excl.append(torch.clamp(cum[-1], 0.0, 1.0))
        return excl
    
    def _overlap_fraction(self, A: torch.Tensor, B: torch.Tensor) -> float:
        """
        计算两个软掩膜的重叠占比：⟨A·B⟩/⟨A⟩。A 视为“分区”，B 视为“本层”。
        """
        num = (A * B).sum().item()
        den = A.sum().item()
        return 0.0 if den == 0.0 else (num / den)
                        
    @torch.inference_mode()
    def propagate_with_film(self, *,
                            apply_surfaces: bool = True,
                            store_steps: tuple[int, ...] = (),
                            callback: Optional[Callable[[int, "PulseGNLSESimulator"], None]] = None,
                            tile_x_kerr: int = 32,
                            eps_inc: float = 0.0,
                            skip_empty_masks: bool = True,
                            center_camera_midpoint: bool = False):   # ← 新增参数
        """
        （…原文不变…）
        """
        assert self.stepped_film is not None, "stepped_film 未设置；先 set_stepped_film_from_annuli(...)"
    
        dev = self.T.device
        self.A = (self.A_in if self.A is None else self.A).to(dev, non_blocking=True)
    
        if apply_surfaces:
            self.A.mul_(torch.sqrt(torch.tensor(self.T_surf, device=dev, dtype=self.A.dtype)))
    
        snapshots: list[torch.Tensor] = []
        global_step = 0
    
        # —— 摄像机中点：预备 —— #
        if center_camera_midpoint:
            regions = self._build_exclusive_region_masks()
            M_center = regions[0]
            M_outer  = regions[-1]
            # Δβ1 = β1_film - β1_ref
            delta_beta1 = float(self.materials["beta_film"][1] - self.materials["beta_ref"][1])  # [s/m]
    
        for j, layer in enumerate(self.stepped_film.layers):
            dL = float(layer.dL); nz = int(layer.nz)
            if nz <= 0 or dL == 0.0: continue
    
            dz = dL / nz
            dz_half = 0.5 * dz
    
            Mj = layer.mask.to(device=dev, dtype=torch.float32, non_blocking=True)
            has_film = (Mj.max().item() > 0.0)
    
            for s in range(nz):
                print(global_step)
                # ---- ½ 参考(space) ----
                self._space_ref_half(dz_half)
                # ---- ½ 参考(time) ----
                self._time_ref_half(dz_half)
    
                # ---- ½ 增量线性（膜区）----
                if has_film:
                    self._apply_incremental_linear_half(Mj, dz_half, eps=eps_inc)
    
                # ---- 非线性（膜区）----
                if has_film:
                    self._kerr_step_with_film(dz, Mj, tile_x=tile_x_kerr)
    
                # ---- ½ 增量线性（膜区）----
                if has_film:
                    self._apply_incremental_linear_half(Mj, dz_half, eps=eps_inc)
    
                # ---- ½ 参考(time) ----
                self._time_ref_half(dz_half)
                # ---- ½ 参考(space) ----
                self._space_ref_half(dz_half)
    
                # === 摄像机对中：把“中心区 & 最外区”的本步 GD 取平均并反向平移 ===
                if center_camera_midpoint and has_film:
                    # 重叠占比（软边缘）：当前层是否作用到中心/外区
                    fc = self._overlap_fraction(M_center, Mj)   # ∈[0,1]
                    fo = self._overlap_fraction(M_outer,  Mj)
                    # 本步两端的增量群时延（相对真空）
                    tau_c = delta_beta1 * dz * fc               # [s]
                    tau_o = delta_beta1 * dz * fo               # [s]
                    dt_shift = 0.5 * (tau_c + tau_o)            # [s] —— 两端中点
                    if dt_shift != 0.0:
                        # 反向加入到“相机时间原点”上：A(t)←A(t+dt_shift)
                        self._apply_global_time_shift(dt_shift)
    
                # ---- 快照 / 回调 ----
                if global_step in store_steps:
                    snapshots.append(self.A.detach().cpu().clone())
                if callback is not None:
                    callback(global_step, self)
                global_step += 1
    
        if apply_surfaces:
            self.A.mul_(torch.sqrt(torch.tensor(self.T_surf, device=dev, dtype=self.A.dtype)))
    
        self.A_out = self.A.detach().cpu()
        del self.A
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
        return snapshots
    

        
    @torch.inference_mode()
    def propagate_in_vacuum(self, L: float, nz: int,
                            *,
                            apply_surfaces: bool = False,
                            store_steps: tuple[int, ...] = (),
                            callback: Optional[Callable[[int, "PulseGNLSESimulator"], None]] = None):
        """
        仅在参考介质=真空中传播 L（米），分 nz 步。
        不使用薄膜与非线性；每步做：½L_ref(space) → ½L_ref(time) → ½L_ref(time) → ½L_ref(space)
        注意：若初始化时 retarded_frame=True，则 β1_ref 会被去掉，相当于共动帧。
        """
        dev = self.T.device
        
        if self.A is None:
            self.A = self.A_out.to(dev, non_blocking=True)
        else:
            self.A = self.A.to(dev, non_blocking=True)
    
        if apply_surfaces:
            self.A.mul_(torch.sqrt(torch.tensor(self.T_surf, device=dev, dtype=self.A.dtype)))
    
        dz = float(L) / int(nz)
        dz_half = 0.5 * dz
    
        # 局部构造真空时间半步算子，避免 retarded_frame=True 时被跳过
        Om = self.omega.to(torch.complex64)                   # (Nt,)
        Lw_half_vac = torch.exp(-1j * (1.0/self.c) * Om * dz_half)  # exp{-i β1_ref Ω dz/2}
    
        snapshots: list[torch.Tensor] = []
        flag=0
        for step in range(int(nz)):
            print(flag)
            flag+=1
            # ½ space (vac)
            _, _, A_k = self.fft_xy(self.A)
            self.A = None
            torch.cuda.empty_cache()
            A_k *= torch.exp(self._coef_xy_ref * dz_half)[..., None]
            _, _, self.A = self.ifft_xy(A_k)
            del A_k
            torch.cuda.empty_cache()
            '''
            # ½ time (vac)
            _, A_w = self.fft_t(self.A)
            self.A = None
            torch.cuda.empty_cache()
            A_w *= Lw_half_vac[None, None, :]
            _, self.A = self.ifft_t(A_w)
            del A_w
            torch.cuda.empty_cache()
            
            # ½ time (vac)
            _, A_w = self.fft_t(self.A)
            self.A = None
            torch.cuda.empty_cache()
            A_w *= Lw_half_vac[None, None, :]
            _, self.A = self.ifft_t(A_w)
            del A_w
            torch.cuda.empty_cache()
            '''
            # ½ space (vac)
            _, _, A_k = self.fft_xy(self.A)
            self.A = None
            torch.cuda.empty_cache()
            A_k *= torch.exp(self._coef_xy_ref * dz_half)[..., None]
            _, _, self.A = self.ifft_xy(A_k)
            del A_k
            torch.cuda.empty_cache()
    
            if step in store_steps:
                snapshots.append(self.A.detach().cpu().clone())
            if callback is not None:
                callback(step, self)
    
        if apply_surfaces:
            self.A.mul_(torch.sqrt(torch.tensor(self.T_surf, device=dev, dtype=self.A.dtype)))
    
        self.A_out = self.A.detach().cpu()
        del self.A
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
        return snapshots


    @torch.inference_mode()
    def _linear_half_step(self):
        with torch.no_grad():
             # 1) (x,y,t) → (kx,ky,t)
            _, _, A = self.fft_xy(self.A)              # A: (Nx,Ny,Nt), complex64
            self.A = None
            torch.cuda.empty_cache()
            # 2) (kx,ky,t) → (kx,ky,ω)
            _, A = self.fft_t(A)                       # A: (Nx,Ny,Nt)
                
            # 3) 乘以分离的线性半步因子（广播，不会复制大张量）
            A *= self.Lxy_half[..., None]              # (Nx,Ny,1) 广播到 (Nx,Ny,Nt)
            A *= self.Lw_half[None, None, :]           # (1,1,Nt) 广播到 (Nx,Ny,Nt)
                
            # 4) (kx,ky,ω) → (kx,ky,t) → (x,y,t)
            _, A = self.ifft_t(A)
            _, _, A = self.ifft_xy(A)
                
            self.A = A                                 # 回写

    
    @torch.inference_mode()    
    def _kerr_step(self, dz: float, *, tile_x: int = 32, empty_cache_every: int = 0):
        """
        内存友好的 Kerr 步：按 x 方向分块 (tile_x × Ny × Nt) 流式处理。
        - 物理不变：SPM 仍为纯相位；shock 仍为 -(γ0/ω0) ∂t(A N)。
        - 数值：仅在本函数内部临时升到 complex128；每个 tile 就地更新回写 self.A（complex64）。
        - 参数：
            tile_x:   每次处理的 x 条数（32/64 经验上较好）
            empty_cache_every: 每处理多少个 tile 调一次 empty_cache；0 表示不调
        """
        with torch.no_grad():
            Nx, Ny, Nt = self.Nx, self.Ny, self.nt
            k_shock = (self.gamma0 / self.omega0) * dz
    
            gamma_even = self.gamma_w_even128                         # (1,1,Nt)
            omega128   = self.omega128                                # (Nt,)
            H_R = self.H_R_omega128 if self.raman_on else None
    
            # 主循环：按 x 方向切片
            tcount = 0
            for x0 in range(0, Nx, tile_x):
                xs = slice(x0, min(x0 + tile_x, Nx))
    
                # ===== (0) 取出一个 tile，并升精度到 complex128 =====
                A_blk = self.A[xs, :, :].to(torch.complex128)         # (tx,Ny,Nt)
    
                # ===== (1) N̂(ω) = 𝓕_t{|A|²}，乘上 Raman 响应 =====
                I_hat = self.fft_t(torch.abs(A_blk)**2)[1]            # (tx,Ny,Nt)
                if H_R is not None:
                    # ((1-fR) + fR H_R) * I_hat  —— 就地乘法减少新张量
                    I_hat *= (1.0 - self.f_R)
                    I_hat += (self.f_R * H_R) * I_hat / (1.0 - self.f_R)  # 复用缓冲
    
                # ===== (2) SPM：phi(t) = 𝓕⁻¹{ γ_even · N̂ } → 纯相位 =====
                phi_t = self.ifft_t(I_hat * gamma_even)[1].real        # (tx,Ny,Nt)
                # 就地相位更新：A_blk *= exp(i*phi*dz)
                A_blk *= torch.exp(1j * phi_t * dz)
                # 立刻释放 phi_t，降低峰值显存
                del phi_t
                
                # ===== (3) 自陡峭：N(t) = 𝓕⁻¹{N̂}；k_shock 先乘到频域 =====
                N_t = self.ifft_t(I_hat)[1].real                       # (tx,Ny,Nt)
                del I_hat                                              # N̂ 不再需要
    
                F_AN = self.fft_t(A_blk * N_t)[1]                      # (tx,Ny,Nt)
                # 先缩放，避免中间量放大溢出
                F_AN *= k_shock
                dA_shock = self.ifft_t(1j * omega128 * F_AN)[1]        # (tx,Ny,Nt)
                del F_AN, N_t
    
                A_blk -= dA_shock
                del dA_shock
                
                
    
                # ===== (4) 回写（降回 complex64），释放 tile 内存 =====
                self.A[xs, :, :] = A_blk.to(torch.complex64)
                del A_blk
    
                tcount += 1
                if empty_cache_every and (tcount % empty_cache_every == 0):
                    torch.cuda.empty_cache()

        
    

    # --------------------------------------------------------------
    #  在 PulseGNLSESimulator 内部
    # --------------------------------------------------------------
    @torch.inference_mode()    
    def propagate(
            self,
            n_steps: int | None = None,
            store_steps: tuple[int, ...] = (),
            callback: Optional[Callable[[int, "PulseGNLSESimulator"], None]] = None,
            report_every: int = 1,
    ):
        """
        Strang-split propagation: ½Linear → Kerr → ½Linear, with snapshot support.
    
        Parameters
        ----------
        n_steps : int | None
            传播步数；默认 = self.nz
        store_steps : tuple[int]
            需要保存快照的步号（0-based）。可传 range(...)
        callback : callable(step_idx:int, self), optional
            每步结束后执行，例如 tqdm 进度条或诊断

        Returns
        -------
        list[np.ndarray]
            保存的 A(x,y,t) 副本，顺序与 store_steps 一致
        """
        with torch.no_grad():
            self.A = self.A.to(DEVICE)     # 传播前搬回 GPU
            self.A = self.A * torch.sqrt(torch.tensor(self.T_surf, device=DEVICE))
    
            
            if n_steps is None:
                n_steps = self.nz
            dz = self.dz
            snapshots: list[torch.Tensor] = []                        # ← 存快照
            
            for step in range(n_steps):
                print(step)               
        
                # -------- ½ 线性 --------
                self._linear_half_step()
                torch.isnan(self.A).any() and print('NaN after ½lin', step)
                
                # -------- Kerr 非线性 --------
                self._kerr_step(dz)
                #print('Kerr')
                torch.isnan(self.A).any() and print('NaN after kerr', step)

                # -------- ½ 线性 --------
                self._linear_half_step()
                #print('2nd linear')
                torch.isnan(self.A).any() and print('NaN after ½lin', step)

                # -------- 记录 / 回调 --------
                diag = self.diagnose(domain='space', metrics=('fwhm',))
                self.spatial_w.append(diag['metrics']['fwhm']['x'])
                del diag
                torch.cuda.empty_cache()
                diag2 = self.diagnose(domain='kspace', metrics=('fwhm',))
                self.spatial_freq_w.append(diag2['metrics']['fwhm']['kx'])
                del diag2
                torch.cuda.empty_cache()
                #print('record')
                
                if step in store_steps:
                    snapshots.append(self.A.clone())      # 深拷贝
                if callback is not None:
                    callback(step, self)
                
    
    
            self.A = self.A * torch.sqrt(torch.tensor(self.T_surf, device=DEVICE))       
            beta2_mgf2 = 2.05e-26             # s^2/m
            L_win      = 5.1e-3               # m
            phase_win  = 0.5 * beta2_mgf2 * (self.omega**2) * L_win   # 因为 factorial(2)=2
            _, A_w = self.fft_t(self.A)
            #_, self.A = self.ifft_t(A_w * torch.exp(1j*phase_win))
            self.A_out = self.A.detach().cpu()
            del self.A
            return snapshots
                
    @torch.inference_mode()
    def apply_circular_lowpass_by_factor(
        self,
        factor: float,
        *,
        A: torch.Tensor | np.ndarray | None = None,
        field: str = "output",
        dk_fwhm: float | None = None,
        dk_rel: float = 0.0,
        tile_t: int = 32,
        preserve_energy: bool = False,
        to_cpu: bool = True,
    
        # —— 预览参数 ——
        preview: bool = True,
        preview_shift: bool = True,
        preview_logscale: bool = True,
        preview_db_floor: float = -60.0,
        preview_percentile: float = 99.9,
        preview_span_factor: float = 1.5
    ) -> torch.Tensor:
        """
        基于“初始 A_in 的空间频谱 FWHM”的圆形低通：
            k_cut = factor * (self.k_fwhm0 / 2)
        预览仅用于显示；滤波结果写 self.A_out（CPU）并返回。
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        from torch import fft as tfft
    
        # —— 兜底：没有 k_fwhm0 就临时测一次 ——
        if not hasattr(self, "k_fwhm0") or not np.isfinite(getattr(self, "k_fwhm0", np.nan)):
            A_saved = getattr(self, "A", None)
            try:
                Ain_gpu = (self.A_in if torch.is_tensor(self.A_in) else torch.as_tensor(self.A_in)).to(self.T.device)
                self.A  = Ain_gpu
                diag_k  = self.diagnose(domain='kspace', metrics=('fwhm',))
                self.kx_fwhm0 = float(diag_k['metrics']['fwhm']['kx'])
                self.ky_fwhm0 = float(diag_k['metrics']['fwhm']['ky'])
                self.k_fwhm0  = 0.5 * (self.kx_fwhm0 + self.ky_fwhm0)
            finally:
                self.A = A_saved
                if torch.cuda.is_available(): torch.cuda.empty_cache()
    
        # —— 选择输入场 ——
        if A is None:
            if field == "output":
                if self.A_out is None:
                    raise ValueError("apply_circular_lowpass_by_factor: self.A_out is None；先得到输出场或显式传入 A。")
                A_src = self.A_out
            elif field == "current":
                if self.A is None:
                    raise ValueError("apply_circular_lowpass_by_factor: self.A is None；传播过程中才有 current 场。")
                A_src = self.A
            elif field == "input":
                if self.A_in is None:
                    raise ValueError("apply_circular_lowpass_by_factor: self.A_in is None。")
                A_src = self.A_in
            else:
                raise ValueError("field 必须是 'output' | 'current' | 'input'")
        else:
            A_src = A
    
        A_src = (A_src if torch.is_tensor(A_src) else torch.as_tensor(A_src))
        if A_src.shape != (self.Nx, self.Ny, self.nt):
            raise ValueError(f"输入场形状 {tuple(A_src.shape)} ≠ {(self.Nx,self.Ny,self.nt)}")
    
        dev = self.T.device
        A_src = A_src.to(device=dev, dtype=torch.complex64, non_blocking=True)
    
        # —— 截止半径 & 软边 ——
        k_cut = float(factor) * (float(self.k_fwhm0) * 0.5)
        if dk_fwhm is None:
            dk_fwhm = float(dk_rel) * max(k_cut, 1e-12)
        dk_fwhm = float(dk_fwhm)
    
        # 低通核
        K_perp = torch.sqrt(self.K_perp2).to(dev)
        Hk     = soft_unit_step(k_cut - K_perp, dk_fwhm).to(torch.float32)  # (Nx,Ny)∈[0,1]
    
        # —— 分块沿 t 处理 ——
        Nt = int(self.nt)
        dx, dy, dT    = float(self.dx), float(self.dy), float(self.dT)
        scale_fft_xy  = float(self._scale_fft_xy.detach().cpu())
        scale_ifft_xy = float(self._scale_ifft_xy.detach().cpu())
    
        out = torch.empty_like(A_src, device=dev)
        E_in = 0.0
        E_ot = 0.0
    
        # 预览累计：滤波前的 I(kx,ky) = ∫ |Â|² dt （**梯形积分**）
        I_accum = torch.zeros((self.Nx, self.Ny), device=dev, dtype=torch.float32) if preview else None
    
        for t0 in range(0, Nt, int(tile_t)):
            t1   = min(t0 + int(tile_t), Nt)
            Ablk = A_src[:, :, t0:t1]
    
            # 入能（块）
            Pin = torch.sum(torch.abs(Ablk)**2, dim=(0,1)) * dx * dy
            E_in += float(torch.sum(Pin).item() * dT)
    
            # 2D FFT
            Ak = tfft.fftn(Ablk, dim=(0, 1))
            Ak.mul_(scale_fft_xy)
    
            # —— 预览积分：梯形权重（首末端 0.5）——
            if I_accum is not None:
                pow2 = Ak.real*Ak.real + Ak.imag*Ak.imag                  # (Nx,Ny,dt)
                w = torch.ones((t1 - t0), device=dev, dtype=pow2.dtype)
                if t0 == 0:  w[0]  = 0.5
                if t1 == Nt: w[-1] = 0.5
                I_accum.add_(torch.sum(pow2 * w[None, None, :], dim=2) * dT)
    
            # 滤波 → iFFT
            Ak *= Hk[..., None].to(Ak.dtype)
            Ablk = tfft.ifftn(Ak, dim=(0, 1))
            Ablk.mul_(scale_ifft_xy)
            del Ak
    
            out[:, :, t0:t1] = Ablk
    
            # 出能（块）
            Pout = torch.sum(torch.abs(Ablk)**2, dim=(0,1)) * dx * dy
            E_ot += float(torch.sum(Pout).item() * dT)
            del Ablk
    
        # 可选能量保持
        if preserve_energy and E_ot > 0.0:
            out.mul_(np.sqrt(E_in / E_ot))
    
        # —— 预览绘图（坐标/shift/线谱与 visualize_simulator 完全一致）——
        if I_accum is not None:
            with torch.no_grad():
                # 通过能量占比（基于未 shift 的核）
                num = float(torch.sum(I_accum * Hk).item())
                den = float(torch.sum(I_accum).item())
                pass_frac = (num / den) if den > 0 else 0.0
    
                # 原生坐标
                I2d = I_accum.detach().cpu().numpy()        # (Nx,Ny)
                Kx1 = self.Kx[:, 0].detach().cpu().numpy()  # (Nx,)
                Ky1 = self.Ky[0, :].detach().cpu().numpy()  # (Ny,)
    
                # 仅显示用的 shift
                if preview_shift:
                    I2d_show = np.fft.fftshift(I2d, axes=(0,1))
                    Kx_show  = np.fft.fftshift(Kx1)
                    Ky_show  = np.fft.fftshift(Ky1)
                else:
                    I2d_show = I2d
                    Kx_show  = Kx1
                    Ky_show  = Ky1
    
                # 动态范围
                pval = np.percentile(I2d_show, preview_percentile)
                pval = pval if pval > 0 else (I2d_show.max() if I2d_show.max()>0 else 1.0)
                if preview_logscale:
                    img = 10.0 * np.log10(np.maximum(I2d_show / pval, 10.0**(preview_db_floor/10.0)))
                    vmin, vmax = preview_db_floor, 0.0
                    cbar_label = f"dB (ref {preview_percentile:.1f}%-ile)"
                else:
                    img = np.clip(I2d_show, 0.0, pval)
                    vmin, vmax = 0.0, pval
                    cbar_label = "∫|Â|² dt (arb)"
    
                # 视野
                span = max(1e-6, float(preview_span_factor) * max(k_cut, 1e-12))
    
                # 画图
                fig, (ax2d, ax1d) = plt.subplots(1, 2, figsize=(11, 4.5))
    
                # 2D 频谱
                extent = [Kx_show.min(), Kx_show.max(), Ky_show.min(), Ky_show.max()]
                im = ax2d.imshow(img, origin="lower", extent=extent, aspect="equal",
                                 vmin=vmin, vmax=vmax)
                fig.colorbar(im, ax=ax2d, label=cbar_label)
                ax2d.set_xlabel("kₓ [m⁻¹]"); ax2d.set_ylabel("k_y [m⁻¹]")
                ax2d.set_title(f"Pre-filter spectrum  (predicted pass ~ {pass_frac*100:.1f}%)")
                ax2d.set_xlim(-span, +span); ax2d.set_ylim(-span, +span)
                ax2d.add_patch(Circle((0.0, 0.0), radius=k_cut, fill=False, lw=1.5))
    
                # 1D 线谱：ky≈0 切片（与 visualize_simulator 相同）
                idx_ky0  = int(np.argmin(np.abs(Ky_show)))
                Ikx_line = I2d_show[:, idx_ky0]
                ax1d.plot(Kx_show, Ikx_line, label="|Â|² at ky≈0")
                ax1d.axvline(+k_cut, ls="--", lw=1.0, label="± k_cut")
                ax1d.axvline(-k_cut, ls="--", lw=1.0)
                ax1d.set_xlim(-span, +span)
                ax1d.set_xlabel("kₓ [m⁻¹]"); ax1d.set_ylabel("Power [arb]")
                ax1d.set_title("Line spectrum (ky≈0)")
                ax1d.grid(True); ax1d.legend()
    
                fig.tight_layout(); plt.show()
    
                if pass_frac < 1e-3:
                    print("[WARN] 该参数下预计通过能量 < 0.1%。请检查 k_fwhm0 是否合理，或调小 factor / 放宽 dk。")
    
        # —— 写回 CPU ——
        A_cpu = out.detach().cpu()
        if to_cpu:
            self.A_out = A_cpu
        del out, A_src
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return A_cpu


    
    # ===========================================================
    #   Diagnostics:  fwhm / rms   in space / time / kspace / freq
    # ===========================================================
    @torch.inference_mode()
    def diagnose(self, A=None, *, domain='space',
                 metrics=('fwhm', 'rms'), save_raw=False,
                 align_time: str = 'none',   # 'none' | 'peak' | 'centroid'
                 subsample_align: bool = True,  # 用频域相移实现亚采样对齐
                 report_center: bool = True):   # 报告 t_peak / t_centroid

        """
        Quick diagnostic on the given 3-D envelope A(x,y,t).

        Parameters
        ----------
        A : ndarray | None
            If None, use self.A (current field).
        domain : str
            'space', 'time', 'kspace', or 'freq'.
        metrics : iterable[str]
            Any subset of {'fwhm','rms'}.
        save_raw : bool
            If True, return the 1-D projection curves.

        Returns
        -------
        dict
        """
        flag=1
        if A is None:
            A = self.A
            flag+=1
        if torch.is_tensor(A):
            A = A.detach().cpu().numpy()
            flag+=1

        def to_np(x):
            return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
        x     = to_np(self.x)
        y     = to_np(self.y)
        T     = to_np(self.T)
        Kx    = to_np(self.Kx)
        Ky    = to_np(self.Ky)
        omega = to_np(self.omega)

        metrics = set(metrics)
        out = {'domain': domain, 'metrics': {}}

        # ---------- build 1-D projections -----------------------
        if domain == 'space':
            I_xy = np.sum(np.abs(A)**2, axis=2) * self.dT
            proj = {'x': np.trapezoid(I_xy, y, axis=1),
                    'y': np.trapezoid(I_xy, x, axis=0)}
            grid = {'x': x, 'y': y}

        elif domain == 'time':
            # ---- 原始投影 P(t) --------------------------------------
            P_t  = np.sum(np.abs(A)**2, axis=(0,1)) * self.dx * self.dy
            T    = to_np(self.T)
            dT   = float(T[1] - T[0])

            # ---- 找中心（峰值/质心） --------------------------------
            t_peak     = float(T[np.argmax(P_t)]) if P_t.max() > 0 else 0.0
            t_centroid = self._temporal_centroid_np(T, P_t)

            # ---- 可选对齐（不改 self.A，只改诊断用的 1D 曲线） -------
            P_plot = P_t.copy()
            T_plot = T.copy()
            if align_time in ('peak', 'centroid'):
                t0 = t_peak if align_time == 'peak' else t_centroid
                if subsample_align:
                    # 频域相移（亚采样）
                    P_plot = self._shift_1d_np(T, P_plot, shift=-t0, d=dT)
                else:
                    # 整数采样 roll（快，但有 ±0.5 样本误差）
                    n = int(np.rint(t0 / dT))
                    P_plot = np.roll(P_plot, -n)
                # 坐标系参考也移到以 0 为中心（更直观）
                T_plot = T - t0

            proj = {'t': P_plot}
            grid = {'t': T_plot}


        elif domain == 'kspace':
            if flag==3:
                _, _, A_k = self.fft_xy(self.A)
                # back to CPU/numpy for projection
                A_k = A_k.detach().cpu().numpy()
                Ik_xy = np.sum(np.abs(A_k)**2, axis=2) * self.dT
                proj = {'kx': np.trapezoid(Ik_xy, Ky[0, :], axis=1),  # integrate over ky (axis=1)
                        'ky': np.trapezoid(Ik_xy, Kx[:,0], axis=0)}   # integrate over kx (axis=0)
                grid = {'kx': Kx[:,0], 'ky': Ky[0,:]}
            else:
                A_gpu = torch.from_numpy(A).to(self.A.device)
                # GPU FFT
                _, _, A_k = self.fft_xy(A_gpu)
                # back to CPU/numpy for projection
                A_k = A_k.detach().cpu().numpy()
                Ak_abs2 = (A_k.real**2 + A_k.imag**2)                  # → 实数、非负
                Ik_xy    = Ak_abs2.sum(axis=2) * float(self.dT)         # ∫|Â|^2 dt（矩形求积）

                proj = {'kx': np.trapezoid(Ik_xy, Ky[0, :], axis=1),  # integrate over ky (axis=1)
                        'ky': np.trapezoid(Ik_xy, Kx[:,0], axis=0)}   # integrate over kx (axis=0)
                grid = {'kx': Kx[:,0], 'ky': Ky[0,:]}


        elif domain == 'freq':
            if flag==3:
                _, A_w_gpu = self.fft_t(self.A)
                A_w = A_w_gpu.detach().cpu().numpy()
                S_w = np.sum(np.abs(A_w)**2, axis=(0,1)) * self.dx * self.dy
                proj = {'w': S_w}
                grid = {'w': omega}
            else:
                A_gpu = torch.from_numpy(A).to(self.A.device)
                _, A_w_gpu = self.fft_t(A_gpu)
                A_w = A_w_gpu.detach().cpu().numpy()
                S_w = np.sum(np.abs(A_w)**2, axis=(0,1)) * self.dx * self.dy
                proj = {'w': S_w}
                grid = {'w': omega}

        else:
            raise ValueError("domain must be 'space' | 'time' | 'kspace' | 'freq'")

        # ---------- metrics ------------------------------------
        # ---------- metrics ------------------------------------
        if 'rms' in metrics:
            out['metrics']['rms'] = {ax: self._rms_width(grid[ax], y)
                                     for ax, y in proj.items()}
        if 'fwhm' in metrics:
            out['metrics']['fwhm'] = {ax: self._fwhm(grid[ax], y)
                                      for ax, y in proj.items()}

        # ---------- optional raw data & centers -----------------
        if save_raw:
            out['raw']  = proj
            out['grid'] = grid

        if report_center and domain == 'time':
            out.setdefault('center', {})
            out['center']['t_peak']     = t_peak
            out['center']['t_centroid'] = t_centroid

        return out


    # ---------- helper functions ------------------------------
    @staticmethod
    def _fwhm(x, y):
        x = np.asarray(x)
        y = np.abs(np.asarray(y))
        if y.size == 0 or y.max() == 0:
            return 0.0
    
        half = 0.5 * y.max()
        mask = y >= half
        idx  = np.flatnonzero(mask)
        if idx.size < 2:
            return 0.0
    
        # 是否“跨端”（两端都≥half，FFT 原生顺序常见）
        wraps = mask[0] and mask[-1]
        if wraps:
            # 展开成两倍长度再取首尾
            x_step = x[1] - x[0]
            x = np.concatenate([x, x + x[-1] + x_step])
            y = np.concatenate([y, y])
            mask = np.concatenate([mask, mask])
            idx  = np.flatnonzero(mask)
    
        i0, i1 = idx[0], idx[-1]
    
        # --- 左边界：用 [j0, j0+1] 线性插值，j0 至少 0
        j0 = max(i0 - 1, 0)
        if j0 + 1 >= len(x):
            x_l = x[j0]
        else:
            x_l = np.interp(half, y[j0:j0+2], x[j0:j0+2])
    
        # --- 右边界：用 [j1, j1+1] 线性插值，j1 至多 len(x)-2
        j1 = min(i1, len(x) - 2)
        if j1 + 1 >= len(x):
            x_r = x[j1]
        else:
            x_r = np.interp(half, y[j1:j1+2], x[j1:j1+2])
    
        return float(x_r - x_l)

    
        # 线性插值
        def _interp(i_left):
            return np.interp(half, y[i_left:i_left+2], x[i_left:i_left+2])
        x_l = _interp(i0-1)
        x_r = _interp(i1)
        return x_r - x_l


    @staticmethod
    def _rms_width(x, y):
        y = np.abs(y)
        norm = np.trapz(y, x)
        if norm == 0:
            return 0.0
        mean = np.trapz(x * y, x) / norm
        var  = np.trapz((x - mean)**2 * y, x) / norm
        return np.sqrt(var)

    @staticmethod
    def _temporal_centroid_np(T: np.ndarray, P: np.ndarray) -> float:
        """质心 t_cm = ∫ t P(t) dt / ∫ P(t) dt；若能量为 0 则返回 0."""
        denom = np.trapz(P, T)
        if denom == 0:
            return 0.0
        return float(np.trapz(T * P, T) / denom)

    @staticmethod
    def _shift_1d_np(x: np.ndarray, y: np.ndarray, shift: float, *, d: float) -> np.ndarray:
        """
        把 y(x) 在 x 轴上平移 'shift'（可分数采样），返回移位后的 y。
        用频域相位坡度实现：Y(k) * exp(-i k shift)。
        """
        # 频率坐标（与 np.fft.fftfreq 一致）
        k = 2*np.pi*np.fft.fftfreq(y.size, d=d)
        Y = np.fft.fft(y)
        y_shift = np.fft.ifft(Y * np.exp(-1j * k * shift))
        # 对于功率曲线，虚部应为数值噪声
        return np.real(y_shift)

from dataclasses import dataclass
from typing import List, Sequence

# -------------------------------
# Stepped-film data structures
# -------------------------------

@dataclass
class LayerSpec:
    """A single physical layer in the stepped film."""
    dL: float                   # layer thickness ΔL_j  [m]
    nz: int                     # axial steps within this layer
    mask: torch.Tensor          # M_j(x,y) in [0,1], shape (Nx,Ny), float32, device=sim.X.device
    name: str = ""              # (optional) tag for debugging/plots


class SteppedFilm:
    """Holds all layers and convenience accessors."""
    def __init__(self, layers: Sequence[LayerSpec]):
        self.layers: List[LayerSpec] = list(layers)

    @property
    def K(self) -> int:
        return len(self.layers)

    def thickness_map(self) -> torch.Tensor:
        """L(x,y) = sum_j ΔL_j * M_j(x,y)  (float32, same device as masks)."""
        assert self.layers, "No layers"
        dev = self.layers[0].mask.device
        Lxy = torch.zeros_like(self.layers[0].mask, dtype=torch.float32, device=dev)
        for L in self.layers:
            Lxy = Lxy + float(L.dL) * L.mask
        return Lxy


import re
import numpy as np
import pandas as pd

def _parse_r_range_mm_to_m(s: str) -> tuple[float, float]:
    """
    把 '0.4–0.7' / '0.4-0.7' / '0.4 — 0.7' 这样的半径区间(单位mm)解析成 (lo_m, hi_m)。
    """
    parts = re.split(r"\s*[-–—]\s*", str(s).strip())
    if len(parts) != 2:
        raise ValueError(f"Bad r range string: {s!r}")
    lo_mm = float(parts[0]); hi_mm = float(parts[1])
    return lo_mm * 1e-3, hi_mm * 1e-3  # → m


def csv_to_layers_lists(
    csv_path: str,
    *,
    col_range: str = "r range (mm)",
    col_thickness: str = "t_final (mm)",   # 也可换成 't_flat (mm)'
    names_col: str | None = "Layer k",
    drop_const_out: bool = True,           # 末行 const(out) 且增量≈0 时丢弃
    eps_nm: float = 1.0                    # 小于该阈值的厚度增量视为0
) -> tuple[list[float | None], list[float], list[str]]:
    """
    从 CSV 生成 (inner_radii, dL_list, names)：
    - inner_radii[0] = None 表示第一层全片；
    - dL_list[0] = t[0]；dL_list[j] = t[j] - t[j-1]（把每层做成“台阶增量”）。
    - 单位：返回值均为“米”。
    """
    df = pd.read_csv(csv_path)

    # 1) 解析每行半径区间
    ranges_m = [ _parse_r_range_mm_to_m(s) for s in df[col_range].tolist() ]
    r_lo = [lo for lo, _ in ranges_m]
    r_hi = [hi for _, hi in ranges_m]

    # 2) 厚度（m）—— 默认使用 t_final
    t_m = (df[col_thickness].astype(float).to_numpy()) * 1e-3

    # 3) 增量列表：第一层是全片厚度，其后是相邻层厚度差
    dL = np.empty_like(t_m)
    dL[0] = t_m[0]
    dL[1:] = t_m[1:] - t_m[:-1]

    # 4) 清理：把“绝对值 < eps_nm”视为 0；如出现负增量，按 0 夹紧并提示
    eps_m = eps_nm * 1e-9
    tiny = np.abs(dL) < eps_m
    dL[tiny] = 0.0
    if np.any(dL < -eps_m):
        print("[csv_to_layers_lists] WARNING: non‑monotonic thickness detected; "
              "negative increments were clipped to 0.")
        dL = np.maximum(dL, 0.0)

    # 5) inner_radii：第一层 None（全片），之后用每行“区间下界”
    inner_radii: list[float | None] = [None] + [float(x) for x in r_lo[1:]]

    # 6) 可选丢弃：增量≈0 的末行（比如 'const(out)'）避免生成空层
    names_raw = df[names_col].astype(str).tolist() if (names_col in df.columns and names_col) else [f"L{j}" for j in range(len(dL))]
    if drop_const_out:
        keep = [(i == 0) or (dL[i] > 0.0) for i in range(len(dL))]
        inner_radii = [inner_radii[i] for i, k in enumerate(keep) if k]
        dL_list     = [float(dL[i])     for i, k in enumerate(keep) if k]
        names       = [names_raw[i]     for i, k in enumerate(keep) if k]
    else:
        dL_list = [float(x) for x in dL]
        names   = names_raw

    return inner_radii, dL_list, names


def build_film_from_csv(
    sim,
    csv_path: str = "film/film_layers_final_codesign.csv",
    *,
    nz_per_layer: int = 4,
    dr_fwhm: float = 40e-6,     # 软边缘FWHM（推荐 30–60 µm）
    aperture: str | float | None = "3w",  # '3w' | 'from_csv' | 数值(米) | None
    **csv_kwargs                   # 透传给 csv_to_layers_lists，例如 col_thickness='t_flat (mm)'
):
    """
    读取 CSV → inner_radii, dL_list → 构建 sim.stepped_film
    """
    inner_radii, dL_list, names = csv_to_layers_lists(csv_path, **csv_kwargs)

    # 选择外光阑
    aperture_radius = None
    aperture_factor = None
    if aperture == "3w":
        aperture_radius = None
        aperture_factor = 3.0
    elif aperture == "from_csv":
        # 用CSV最后一行的r上界；需要再解析一次
        df = pd.read_csv(csv_path)
        last_hi_m = _parse_r_range_mm_to_m(df[csv_kwargs.get("col_range", "r range (mm)")].iloc[-1])[1]
        aperture_radius = float(last_hi_m)
        aperture_factor = None
    elif isinstance(aperture, (int, float)):
        aperture_radius = float(aperture)
        aperture_factor = None
    else:
        aperture_radius = None
        aperture_factor = None

    film = sim.set_stepped_film_from_annuli(
        inner_radii=inner_radii,
        dL_list=dL_list,
        nz_per_layer=nz_per_layer,
        dr_fwhm=dr_fwhm,
        aperture_radius=aperture_radius,
        aperture_factor=aperture_factor,
        names=names,
    )
    return film, inner_radii, dL_list, names


def set_film_simple(sim,
                    r,              # 例如 [1,2,3]  —— 半径分界（默认 mm）
                    t,              # 例如 [0.1,0.2,0.3] —— 各区“绝对厚度”（默认 mm）
                    *,
                    units="mm",     # "mm" 或 "m"
                    nz_per_layer=4, # 每层的纵向步数
                    dr_fwhm=40e-6,  # 软边 FWHM（米），=0 则硬边
                    aperture_from_last=True,   # 用最后一个半径当做外光阑
                    names=None,     # 可选：每层名字
                    clip_negative=True,  # 绝对厚度若非单调，把负增量夹成0
                    interpret="absolute" # "absolute"=t是绝对厚度；"delta"=t就是增量
                   ):
    """
    0–r[0]、r[0]–r[1]、... 这些区域的厚度由 t[0], t[1], ... 给定。
    内部会自动把“绝对厚度”转换成增量 ΔL，以适配 set_stepped_film_from_annuli。
    """
    r = [float(x) for x in r]
    t = [float(x) for x in t]
    assert len(r) == len(t), "r 和 t 的长度必须一致（每个分界半径对应一个厚度值）"
    # 单位换算
    if units == "mm":
        r_m = [x * 1e-3 for x in r]
        t_m = [x * 1e-3 for x in t]
    elif units == "m":
        r_m = r
        t_m = t
    else:
        raise ValueError("units 必须是 'mm' 或 'm'")

    K = len(t_m)

    # 把绝对厚度 t → 增量 dL（累加式台阶）：
    if interpret == "absolute":
        dL = [t_m[0]] + [t_m[i] - t_m[i-1] for i in range(1, K)]
    elif interpret == "delta":
        dL = list(t_m)
    else:
        raise ValueError("interpret 只能是 'absolute' 或 'delta'")

    if clip_negative:
        # 和你 csv 版本一致：若厚度非单调，负增量夹到 0
        dL = [x if x >= 0.0 else 0.0 for x in dL]

    # inner_radii 的长度必须等于层数 K
    # 第一层用 None 表示“整片”，后面的从 r[0], r[1], ... 依次开始累加
    inner_radii = [None] + r_m[:-1] if K > 0 else []

    # 可选：把最后一个半径当做外光阑（避免最外层无限延伸）
    aperture_radius = r_m[-1] if aperture_from_last else None

    # 缺省名字
    if names is None:
        names = [f"step{j}" for j in range(K)]

    # 直接用你现有的入口构建
    film = sim.set_stepped_film_from_annuli(
        inner_radii=inner_radii,
        dL_list=dL,
        nz_per_layer=nz_per_layer,
        dr_fwhm=dr_fwhm,
        aperture_radius=aperture_radius,
        names=names,
    )
    return film

import pandas as pd, numpy as np, torch
import re

def _parse_r_range_mm_to_m(s: str) -> tuple[float, float]:
    parts = re.split(r"\s*[-–—]\s*", str(s).strip())
    if len(parts) != 2: raise ValueError(f"Bad r range string: {s!r}")
    lo_mm, hi_mm = float(parts[0]), float(parts[1])
    return lo_mm*1e-3, hi_mm*1e-3

def build_mirror_height_map(sim,
                            csv_path="film/mirror_echelon_final_codesign.csv",
                            *,
                            col_range="r range (mm)",
                            col_height="d_abs_final (µm)",
                            dr_fwhm=40e-6,
                            aperture="3w",            # '3w' | 'from_csv' | float(m) | None
                            eps_um=1e-3,              # 小于该阈值(µm)的增量视为0（去抖）
                            sort_rows=True            # 先按 r_lo 升序
                            ):
    """
    从镜面台阶 CSV 生成 H(x,y)（单位 m）。
    修正点：
      • aperture='3w' 现在会按 3*w 自动计算半径（并夹到网格内）
      • 增量 dH 允许正负，只去除极小抖动
      • 可选：按 r_lo 升序排序
    """
    import pandas as pd, numpy as np, torch

    df = pd.read_csv(csv_path)

    # 1) 解析半径区间 (mm→m)
    ranges_m = [ _parse_r_range_mm_to_m(s) for s in df[col_range].astype(str).tolist() ]
    r_lo = np.array([lo for lo, _ in ranges_m], dtype=float)
    r_hi = np.array([hi for _, hi in ranges_m], dtype=float)

    # 2) 绝对高度 (µm→m)
    h_abs = df[col_height].astype(float).to_numpy() * 1e-6

    # 3) 按 r_lo 升序排序（避免顺序错误）
    if sort_rows:
        order = np.argsort(r_lo)
        r_lo, r_hi, h_abs = r_lo[order], r_hi[order], h_abs[order]

    # 4) 绝对→增量（允许正负），只去超小抖动
    dH = np.empty_like(h_abs)
    dH[0]  = h_abs[0]
    dH[1:] = h_abs[1:] - h_abs[:-1]
    eps_m = float(eps_um) * 1e-6
    dH[np.abs(dH) < eps_m] = 0.0   # 去抖；不剪负号！

    # 5) 内半径：第一层用 None 表示整片，其后用每行下界
    inner_radii = [None] + [float(x) for x in r_lo[1:]]

    # 6) 外光阑半径
    if aperture == "3w":
        w = sim._infer_beam_radius() or 0.0
        if w > 0:
            aperture_radius = 3.0 * float(w)
        else:
            # 退化：取 CSV 最后一行上界
            aperture_radius = float(r_hi[-1])
    elif aperture == "from_csv":
        aperture_radius = float(r_hi[-1])
    elif isinstance(aperture, (int, float)):
        aperture_radius = float(aperture)
    else:
        aperture_radius = None

    # 限幅到网格
    if aperture_radius is not None:
        r_grid = float(min(sim.x.abs().max(), sim.y.abs().max()))
        aperture_radius = min(aperture_radius, 0.98 * r_grid)

    # 7) 掩膜（软边）
    masks = make_annular_masks(sim.X, sim.Y,
                               inner_radii=inner_radii,
                               dr_fwhm=dr_fwhm,
                               aperture_radius=aperture_radius,
                               aperture_fwhm=dr_fwhm)

    # 8) 叠加：H(x,y) = Σ dH_j · M_j
    Hxy = torch.zeros_like(sim.X, dtype=torch.float32, device=sim.X.device)
    for d, M in zip(dH, masks):
        Hxy.add_(M, alpha=float(d))
    return Hxy



# -------------------------------
# Mask utilities (soft edges)
# -------------------------------

def _sigma_from_fwhm(dr_fwhm: float) -> float:
    """FWHM (m) → Gaussian sigma (m) for erf-based soft step."""
    # FWHM = 2*sqrt(2*ln2)*sigma
    return dr_fwhm / (2.0 * (2.0 * np.log(2.0))**0.5)

def soft_unit_step(x: torch.Tensor, fwhm: float) -> torch.Tensor:
    """
    Smooth Heaviside H(x) with transition ~ FWHM.
    Returns value in [0,1]. fwhm in meters.
    """
    if fwhm <= 0:
        return (x >= 0).to(torch.float32)
    sigma = _sigma_from_fwhm(fwhm)
    t = x / (np.sqrt(2.0) * sigma)
    return 0.5 * (1.0 + torch.erf(t))



def make_annular_masks(
    X: torch.Tensor, Y: torch.Tensor,
    inner_radii: Sequence[float],         # r_edge for each layer; j=0 可用 None/负数 表示“整片为1”
    dr_fwhm: float,                        # edge smoothing FWHM [m], e.g. 30e-6
    aperture_radius: float | None = None,  # optional hard/soft outer stop
    aperture_fwhm: float | None = None,    # soft edge for aperture if given
) -> List[torch.Tensor]:
    """
    Generate M_j(x,y) for K layers where layer j occupies r >= inner_radii[j].
    inner_radii[j] can be None (or <0) to indicate a full disk (M=1 everywhere).
    Returns masks as float32 tensors on the same device as X/Y.
    """
    r = torch.sqrt(X**2 + Y**2)
    masks: List[torch.Tensor] = []
    for j, r_edge in enumerate(inner_radii):
        if (r_edge is None) or (r_edge < 0):
            Mj = torch.ones_like(r, dtype=torch.float32)
        else:
            Mj = soft_unit_step(r - float(r_edge), dr_fwhm).to(torch.float32)

        # optional outer aperture
        if aperture_radius is not None:
            if aperture_fwhm is None or aperture_fwhm <= 0:
                M_ap = (r <= aperture_radius).to(torch.float32)
            else:
                M_ap = soft_unit_step(aperture_radius - r, aperture_fwhm).to(torch.float32)
            Mj = Mj * M_ap

        masks.append(Mj)
    return masks



from typing import Literal, Optional
import time

def human(mem_bytes):
    for unit in ['B','KB','MB','GB']:
        if mem_bytes < 1024.0:
            return f"{mem_bytes:.1f}{unit}"
        mem_bytes /= 1024.0
    return f"{mem_bytes:.1f}TB"

# -----------------------------------------------------------------------------
# Helper functions —–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# -----------------------------------------------------------------------------

def _auto_limits(x: np.ndarray, y: np.ndarray, zoom: Optional[float] = 6.0):
    """Return (xmin, xmax) so that the shown span ≈ *zoom* × FWHM of *y*.
    If *zoom* is *None* the full axis is returned.
    """
    if zoom is None:
        return x.min(), x.max()

    y = np.abs(y)
    if y.max() == 0:
        return x.min(), x.max()

    half = 0.5 * y.max()
    idx = np.where(y >= half)[0]
    if len(idx) < 2:
        return x.min(), x.max()

    x0, x1 = x[idx[0]], x[idx[-1]]
    cx = 0.5 * (x0 + x1)
    w = (x1 - x0) * 0.5 * zoom
    return cx - w, cx + w

# -----------------------------------------------------------------------------
# Main visualisation routine —–––––––––––––––––––––––––––––––––––––––––––––––––
# -----------------------------------------------------------------------------
@torch.inference_mode()
def visualize_simulator(
    sim,  # PulseGNLSESimulator instance
    *,
    field: Literal["input", "current", "output"] = "current",
    zoom_time: Optional[float] = 3.0,
    zoom_space: Optional[float] = 12.0,
    log_scale_2d: bool = False,
):
    """Create a 3×2 Matplotlib figure visualising the given *sim* state.

    Parameters
    ----------
    sim : PulseGNLSESimulator
        The simulator object.
    field : {'input','current','output'}
        Which electric‑field snapshot to visualise.  *current* uses ``sim.A``.
    zoom_time, zoom_space : float | None
        Multiple of the FWHM/RMS width kept visible along the time and spatial axes.
        Set to *None* to show the full window.
    log_scale_2d : bool, default False
        If *True*, the 2‑D heat maps are coloured on a log10 scale.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure object (already populated).
    """
    # -------- choose which field ------------------------------------

    # -------- 2) 统‑一‑搬‑到 CPU (关键一步) --------------------------
    T   = sim.T.detach().cpu().numpy()
    x   = sim.x.detach().cpu().numpy()
    y   = sim.y.detach().cpu().numpy()
    omega = sim.omega.detach().cpu().numpy()
    Kx  = sim.Kx.detach().cpu().numpy()
    Ky  = sim.Ky.detach().cpu().numpy()
    dx, dy = float(sim.dx), float(sim.dy)    # python‑float
    
    if field == "input":
    
        torch.cuda.empty_cache()
        
        I_kxky = np.trapezoid(np.abs(sim.fft_xy((sim.A_in).to(sim.T.device))[2].detach().cpu().numpy())**2, T, axis=2)
        torch.cuda.empty_cache()
    
        S_w  = np.sum(np.abs(sim.fft_t((sim.A_in).to(sim.T.device))[1].detach().cpu().numpy())**2, axis=(0,1)) * dx * dy  # |A|^2 vs ω
        torch.cuda.empty_cache()
    
        I =np.abs( (sim.A_in).to(sim.T.device).detach().cpu().numpy() )**2       # (Nx,Ny,Nt), complex
   
    elif field == "output":
        if sim.A_out is None:
            raise ValueError("sim.A_out is None — run sim.propagate() first or choose another field")

        torch.cuda.empty_cache()
        
        I_kxky = np.trapezoid(np.abs(sim.fft_xy((sim.A_out).to(sim.T.device))[2].detach().cpu().numpy())**2, T, axis=2)
        torch.cuda.empty_cache()
    
        S_w  = np.sum(np.abs(sim.fft_t((sim.A_out).to(sim.T.device))[1].detach().cpu().numpy())**2, axis=(0,1)) * dx * dy  # |A|^2 vs ω
        
        torch.cuda.empty_cache()
    
        I =np.abs( (sim.A_out).to(sim.T.device).detach().cpu().numpy() )**2        # (Nx,Ny,Nt), complex
    else:
    
        torch.cuda.empty_cache()
        
        I_kxky = np.trapezoid(np.abs(sim.fft_xy(sim.A)[2].detach().cpu().numpy())**2, T, axis=2)
        torch.cuda.empty_cache()
    
        S_w  = np.sum(np.abs(sim.fft_t(sim.A)[1].detach().cpu().numpy())**2, axis=(0,1)) * dx * dy  # |A|^2 vs ω

        torch.cuda.empty_cache()
    
        I =np.abs( sim.A.detach().cpu().numpy() )**2
 
    
    # ---------- 频域中心化 ----------------------------------------
    omega = np.fft.fftshift(omega)          # (Nt,)
    S_w   = np.fft.fftshift(S_w)            # (Nt,)
    # --------------------------------------------------------------
    
    # ---------- k‑space 中心化 -------------------------------------
    I_kxky = np.fft.fftshift(I_kxky, axes=(0, 1))
    Kx     = np.fft.fftshift(Kx,   axes=0)
    Ky     = np.fft.fftshift(Ky,   axes=1)
    torch.cuda.empty_cache()
    

    # ===== 1D temporal power ========================================
    P_t = np.sum(I, axis=(0,1)) * dx * dy


    
    # ===== 1D spectral power (use direct FFT of full field) =========
    '''
    _,  A_w  = sim.fft_t(A_gpu)           # ← 这行仍在 GPU 做 FFT
    A_w  = sim.fft_t(A_gpu)[1].detach().cpu().numpy()
    '''
    #S_w  = np.sum(np.abs( A_w)**2, axis=(0,1)) * dx * dy  # |A|^2 vs ω
    
    
    # ===== 2D transverse intensity ==================================
    I_xy = np.trapezoid(I,T, axis=2)  # integrate over t

    torch.cuda.empty_cache()

    # ===== 2D k‑space intensity =====================================
    '''
    _, _, A_k = sim.fft_xy(A_gpu)
    A_k  = A_k.detach().cpu().numpy()
    I_kxky = np.trapezoid(np.abs(A_k)**2, T, axis=2)
    '''
    

    # ===== 1D transverse profile (x) ================================
    idx_y0=np.argmin(np.abs(y))
    I_x = I_xy[:,idx_y0]  # integrate over y

    # ===== 1D k‑space profile (kx) ==================================
    idx_ky0=np.argmin(np.abs(Ky[0, :]))
    I_kx = I_kxky[:,idx_ky0]

    # ----------------------------------------------------------------
    # Figure layout  ––––––––––––––––––––––––––––––––––––––––––––––––––
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    (ax_t, ax_w), (ax_xy, ax_k), (ax_x, ax_kx) = axes

    # ---- Time domain ----------------------------------------------
    ax_t.plot(T, P_t)
    ax_t.set_xlabel("t  [s]")
    ax_t.set_ylabel("∫|A|² dx dy  [J/s]")
    ax_t.set_title("Temporal power (integrated over x,y)")
    ax_t.set_xlim(_auto_limits(T, P_t, zoom_time))


    # ---- Frequency domain -----------------------------------------
    ax_w.plot(omega, S_w)
    ax_w.set_xlabel("ω  [rad/s]")
    ax_w.set_ylabel("∫|Â|² dx dy  [arb]")
    ax_w.set_title("Spectral power (integrated over x,y)")
    ax_w.set_xlim(_auto_limits(omega, S_w, zoom_time))

    # ---- 2‑D spatial map ------------------------------------------
    extent_xy = [x.min(), x.max(), y.min(), y.max()]
    if log_scale_2d:
        im_xy = ax_xy.imshow(
            np.log10(np.maximum(I_xy, I_xy.max() * 1e-12)),  # avoid log(0)
            origin="lower",
            extent=extent_xy,
            aspect="equal",
        )
        cbar_label = "log₁₀ I(x,y)"
    else:
        im_xy = ax_xy.imshow(
            I_xy,
            origin="lower",
            extent=extent_xy,
            aspect="equal",
        )
        cbar_label = "I(x,y)  [arb]"
    fig.colorbar(im_xy, ax=ax_xy, label=cbar_label)
    ax_xy.set_xlabel("x  [m]")
    ax_xy.set_ylabel("y  [m]")
    ax_xy.set_title("Spatial intensity (t‑integrated)")
    # zoom spatial axes
    x_lim = _auto_limits(x, I_x, zoom_space)
    y_proj = np.trapezoid(I_xy, x, axis=0)
    y_lim = _auto_limits(y, y_proj, zoom_space)
    ax_xy.set_xlim(x_lim)
    ax_xy.set_ylim(y_lim)

    # ---- 2‑D k‑space map -----------------------------------------
    extent_k = [Kx.min(), Kx.max(), Ky.min(), Ky.max()]
    if log_scale_2d:
        im_k = ax_k.imshow(
            np.log10(np.maximum(I_kxky, I_kxky.max() * 1e-12)),
            origin="lower",
            extent=extent_k,
            aspect="equal",
        )
        cbar_label_k = "log₁₀ I(kx,ky)"
    else:
        im_k = ax_k.imshow(
            I_kxky,
            origin="lower",
            extent=extent_k,
            aspect="equal",
        )
        cbar_label_k = "I(kx,ky)  [arb]"
    fig.colorbar(im_k, ax=ax_k, label=cbar_label_k)
    ax_k.set_xlabel("kₓ  [m⁻¹]")
    ax_k.set_ylabel("k_y  [m⁻¹]")
    ax_k.set_title("Spatial spectrum (t‑integrated)")
    Kx_lim = _auto_limits(Kx[:, 0], I_kx, zoom_space)
    ky_proj = np.trapezoid(I_kxky, Kx[:, 0], axis=0)
    Ky_lim = _auto_limits(Ky[0, :], ky_proj, zoom_space)
    ax_k.set_xlim(Kx_lim)
    ax_k.set_ylim(Ky_lim)


    # ---- 1‑D spatial profile (x) ----------------------------------
    ax_x.plot(x, I_x)
    ax_x.set_xlabel("x  [m]")
    ax_x.set_ylabel("|A|² at y≈0  [arb]")
    ax_x.set_title("Line profile at y≈0")
    ax_x.set_xlim(x_lim)

    # ---- 1‑D k‑space profile (kx) ---------------------------------
    ax_kx.plot(Kx[:, 0], I_kx)
    ax_kx.set_xlabel("kₓ  [m⁻¹]")
    ax_kx.set_ylabel("∫|Â|² dky dω  [arb]")
    ax_kx.set_title("k‑space line (integrated over ky,ω)")
    ax_kx.set_xlim(Kx_lim)

    fig.tight_layout()
    plt.show()
    return fig

@torch.inference_mode()
def film_thickness_map(sim) -> torch.Tensor:
    """
    Return L(x,y) = sum_j dL_j * M_j(x,y)
    dtype=float32, device=sim.X.device, shape=(Nx,Ny)
    """
    assert sim.stepped_film is not None, "sim.stepped_film is None"
    dev = sim.X.device
    Lxy = torch.zeros((sim.Nx, sim.Ny), dtype=torch.float32, device=dev)
    for layer in sim.stepped_film.layers:
        # 显式按层累加：L += ΔL_j * M_j
        Lxy.add_(layer.mask, alpha=float(layer.dL))
    return Lxy


@torch.inference_mode()
def visualize_film(sim, *, crop_factor: float = 2.0, line_axis: str = 'x'):
    """
    左：厚度图 L(x,y)
    中：中心截线厚度（默认 y=0；line_axis='y' 则画 x=0 的垂直截线）
    右：中心截线“相对全空气”的理论群时延 Δt(x) = [β1_film - β1_ref] * L(x)，单位 fs
    """
    assert sim.stepped_film is not None, "sim.stepped_film is None"

    # 1) 厚度图 = ∑ dL_j * M_j
    Lxy = film_thickness_map(sim)                 # (Nx,Ny) [m]

    # 2) 网格与截线索引
    x = sim.x.detach().cpu().numpy()
    y = sim.y.detach().cpu().numpy()
    L = Lxy.detach().cpu().numpy()

    if line_axis.lower() == 'x':
        # 取 y≈0 的那一行
        iy0 = int(torch.argmin(torch.abs(sim.y)).item())
        x_line = x
        L_line = L[:, iy0]
        xlabel = "x [m]"
    else:
        # 取 x≈0 的那一列
        ix0 = int(torch.argmin(torch.abs(sim.x)).item())
        x_line = y
        L_line = L[ix0, :]
        xlabel = "y [m]"

    # 3) 视野范围：±(crop_factor * w)；若取不到 w，则用 0.8×网格半宽
    try:
        w = sim._infer_beam_radius()
    except Exception:
        w = None
    r_grid = float(min(np.max(np.abs(x)), np.max(np.abs(y))))
    lim = float(min(crop_factor * w, 0.98 * r_grid)) if (w is not None and w > 0) else 0.8 * r_grid

    # === 新增：理论群时延（相对“全空气”）===========================
    # β1 = dβ/dω，单位 s/m；已在 sim.materials 中给出：
    #   beta_film[1] 来自你的 dispersion 字典；beta_ref[1]≈1/c
    beta1_film = float(sim.materials["beta_film"][1])   # s/m
    beta1_ref  = float(sim.materials["beta_ref" ][1])   # s/m
    delta_beta1 = beta1_film - beta1_ref                # s/m
    tau_line_fs = (delta_beta1 * L_line) * 1e15         # fs

    # 4) 画图：三联图
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    # 左：L(x,y)
    im = ax1.imshow(
        L, origin="lower",
        extent=[x.min(), x.max(), y.min(), y.max()],
        aspect="equal"
    )
    ax1.set_title("Film thickness $L(x,y)$ [m]")
    fig.colorbar(im, ax=ax1)
    ax1.set_xlabel("x [m]"); ax1.set_ylabel("y [m]")
    ax1.set_xlim(-lim, lim); ax1.set_ylim(-lim, lim)

    # 中：中心截线厚度（mm）
    ax2.plot(x_line, L_line * 1e3)
    ax2.set_xlabel(xlabel); ax2.set_ylabel("Thickness [mm]")
    ax2.set_title("Center-line thickness")
    ax2.set_xlim(-lim, lim)
    ax2.grid(True)

    # 右：中心截线“相对空气”的理论群时延（fs）
    ax3.plot(x_line, tau_line_fs)
    ax3.set_xlabel(xlabel); ax3.set_ylabel("Δt [fs]  (vs air)")
    ax3.set_title("Center-line group delay (theory)")
    ax3.set_xlim(-lim, lim)
    ax3.grid(True)

    fig.tight_layout(); plt.show()


import torch
import numpy as np
import matplotlib.pyplot as plt

@torch.inference_mode()
def visualize_time_and_freq_by_regions(
    sim,
    *,
    field: str = "output",   # 'output' | 'current' | 'input'
    t_unit: str = "fs",      # 'fs' | 'ps' | 's'
    f_unit: str = "THz",     # 'THz' | 'rad/s'
    normalize: bool = False, # 时间/频域分别各自归一化到最大值=1
    fftshift: bool = True,   # 频谱居中显示
    show_energy_fraction: bool = True,  # 图例显示各分区能量占比
    title: str = "Temporal & Spectral power by spatial regions (GPU, 2-field peak)",
):
    dev = sim.T.device
    if dev.type != "cuda":
        print("[Note] 当前设备非 CUDA，将在 CPU 上运行（功能正确但速度较慢）。")
    torch.cuda.empty_cache()
    check_mem()
    # -------- 取场并搬到 GPU ----------
    if field == "output":
        if sim.A_out is None:
            raise ValueError("sim.A_out is None — 请先完成传播（propagate_with_film 或 propagate）")
        A = (sim.A_out if torch.is_tensor(sim.A_out) else torch.as_tensor(sim.A_out)).to(
            dev, dtype=torch.complex64, non_blocking=True
        )
    elif field == "input":
        A = (sim.A_in if torch.is_tensor(sim.A_in) else torch.as_tensor(sim.A_in)).to(
            dev, dtype=torch.complex64, non_blocking=True
        )
    else:  # 'current'
        if sim.A is None:
            raise ValueError("sim.A is None — 只有传播过程中才有 current 场；否则请用 'output' 或 'input'")
        A = (sim.A if torch.is_tensor(sim.A) else torch.as_tensor(sim.A)).to(
            dev, dtype=torch.complex64, non_blocking=True
        )

    Nx, Ny, Nt = A.shape
    dx, dy = float(sim.dx), float(sim.dy)
    area_scale = dx * dy

    # -------- 构造互斥分区掩模（GPU，float32） ----------
    assert sim.stepped_film is not None and len(sim.stepped_film.layers) > 0, "stepped_film 未设置或为空"
    cum = [L.mask.to(device=dev, dtype=torch.float32, non_blocking=True) for L in sim.stepped_film.layers]
    if len(cum) == 1:
        excl = [cum[0]]
    else:
        excl = [torch.clamp(cum[j] - cum[j + 1], 0.0, 1.0) for j in range(len(cum) - 1)]
        excl.append(torch.clamp(cum[-1], 0.0, 1.0))
    M0 = cum[0]
    torch.cuda.empty_cache()

    # ==========================
    # 1) 时间域 — 峰值内存：A + I
    # ==========================
    P_total = torch.zeros(Nt, device=dev, dtype=torch.float32)
    P_regions = [torch.zeros(Nt, device=dev, dtype=torch.float32) for _ in excl]
    
    tile_x = 32  # 或 64/128，视显存调
    for xs in range(0, Nx, tile_x):
        xe = min(xs + tile_x, Nx)
        Ab = A[xs:xe, :, :]  # (tx, Ny, Nt), complex64
        Ib = (Ab.real * Ab.real + Ab.imag * Ab.imag).to(torch.float32)  # 只建小块
        P_total += torch.einsum('xyt,xy->t', Ib, M0[xs:xe, :]) * area_scale
        for j, Mj in enumerate(excl):
            P_regions[j] += torch.einsum('xyt,xy->t', Ib, Mj[xs:xe, :]) * area_scale
        del Ib
    # 这里 Ib 已全释放，不会触发 4GB 段
    torch.cuda.synchronize()  # 可选：让统计更稳定
    torch.cuda.empty_cache()

    # —— 立刻搬到 CPU，避免后续阶段这些小张量钉住大段 —— #
    T = sim.T.detach().cpu().numpy()
    omega = sim.omega.detach().cpu().numpy()
    P_total_np = P_total.detach().cpu().numpy()
    P_regions_np = [p.detach().cpu().numpy() for p in P_regions]

    # 如需最大限度清理，也可把掩膜先删，后面频域再重建
    del P_total, P_regions, M0, cum, excl
    torch.cuda.empty_cache()
    # ==========================
    # 2) 频域（分块流式；不建 4GB 缓冲）
    # ==========================

    scale_fft_t = float(sim._scale_fft_t.detach().cpu().item())
    A_w = torch.fft.fft(A, dim=-1)  # complex64
    del A
    torch.cuda.empty_cache()
    
    A_w.mul_(scale_fft_t)

    # 重新拿掩膜（GPU float32）
    cum = [L.mask.to(device=dev, dtype=torch.float32, non_blocking=True) for L in sim.stepped_film.layers]
    if len(cum) == 1:
        excl = [cum[0]]
    else:
        excl = [torch.clamp(cum[j] - cum[j + 1], 0.0, 1.0) for j in range(len(cum) - 1)]
        excl.append(torch.clamp(cum[-1], 0.0, 1.0))
    M0 = cum[0]

    S_total = torch.zeros(Nt, device=dev, dtype=torch.float32)
    S_regions = [torch.zeros(Nt, device=dev, dtype=torch.float32) for _ in excl]

    tile_x = 32  # 可调：32/64/128
    for xs in range(0, Nx, tile_x):
        xe = min(xs + tile_x, Nx)
        Aw_blk = A_w[xs:xe, :, :]  # (tx, Ny, Nt)

        pow_blk = (Aw_blk.real * Aw_blk.real + Aw_blk.imag * Aw_blk.imag).to(torch.float32)

        S_total += torch.sum(pow_blk, dim=(0, 1)) * area_scale
        for j, Mj in enumerate(excl):
            S_regions[j] += torch.einsum('xyt,xy->t', pow_blk, Mj[xs:xe, :]) * area_scale

        del pow_blk

    del A_w
    torch.cuda.empty_cache()
    
    # —— 搬到 CPU —— 
    S_total_np   = S_total.detach().cpu().numpy()
    S_regions_np = [s.detach().cpu().numpy() for s in S_regions]
    del S_total, S_regions, M0, cum, excl
    torch.cuda.empty_cache()

    # ===== 归一化（时间/频域各自） =====
    if normalize:
        denom_t = P_total_np.max() if P_total_np.max() > 0 else 1.0
        denom_w = S_total_np.max() if S_total_np.max() > 0 else 1.0
        P_total_np   = P_total_np / denom_t
        P_regions_np = [p / denom_t for p in P_regions_np]
        S_total_np   = S_total_np / denom_w
        S_regions_np = [s / denom_w for s in S_regions_np]
        ylab_t = "Normalized  ∫|A|² dx dy"
        ylab_w = "Normalized  ∫|Â|² dx dy"
    else:
        ylab_t = "∫|A|² dx dy  [J/s]"
        ylab_w = "∫|Â|² dx dy  [arb]"

    # ===== 能量占比 =====
    import numpy as _np
    E_total_t = _np.trapz(P_total_np, T)
    frac_t = [(_np.trapz(p, T) / E_total_t * 100.0) if E_total_t > 0 else 0.0 for p in P_regions_np]

    if f_unit.lower() == "thz":
        f = omega / (2 * np.pi * 1e12)
        xlab_w = "Frequency offset [THz]"
    else:
        f = omega
        xlab_w = "ω  [rad/s]"

    if fftshift:
        f_plot = np.fft.fftshift(f)
        S_total_plot   = np.fft.fftshift(S_total_np)
        S_regions_plot = [np.fft.fftshift(s) for s in S_regions_np]
    else:
        f_plot = f
        S_total_plot   = S_total_np
        S_regions_plot = S_regions_np

    E_total_w = np.trapz(S_total_plot, f_plot)
    frac_w = [(np.trapz(s, f_plot) / E_total_w * 100.0) if E_total_w > 0 else 0.0 for s in S_regions_plot]

    # ===== 横轴单位（时间） =====
    if t_unit == "fs":
        t = T * 1e15; xlab_t = "Time [fs]"
    elif t_unit == "ps":
        t = T * 1e12; xlab_t = "Time [ps]"
    else:
        t = T; xlab_t = "Time [s]"

    # ===== 区域标签（不再依赖已删除的 excl） =====
    K_excl = len(S_regions_np)
    if K_excl == 1:
        labels_t = [f"Region 0 — {frac_t[0]:.1f}% E"] if show_energy_fraction else ["Region 0"]
        labels_w = [f"Region 0 — {frac_w[0]:.1f}% E"] if show_energy_fraction else ["Region 0"]
    else:
        labels_t = [f"Region 0 (center) — {frac_t[0]:.1f}% E" if show_energy_fraction else "Region 0 (center)"]
        labels_w = [f"Region 0 (center) — {frac_w[0]:.1f}% E" if show_energy_fraction else "Region 0 (center)"]
        for j in range(1, K_excl - 1):
            labels_t.append(f"Region {j} (ring) — {frac_t[j]:.1f}% E" if show_energy_fraction else f"Region {j} (ring)")
            labels_w.append(f"Region {j} (ring) — {frac_w[j]:.1f}% E" if show_energy_fraction else f"Region {j} (ring)")
        labels_t.append(f"Region {K_excl-1} (outermost) — {frac_t[-1]:.1f}% E" if show_energy_fraction else f"Region {K_excl-1} (outermost)")
        labels_w.append(f"Region {K_excl-1} (outermost) — {frac_w[-1]:.1f}% E" if show_energy_fraction else f"Region {K_excl-1} (outermost)")

    # ---- 绘图 ----
    fig, (ax_t, ax_w) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title)

    ax_t.plot(t, P_total_np, label="Total (within aperture)")
    for Pj, lab in zip(P_regions_np, labels_t):
        ax_t.plot(t, Pj, linestyle="--", label=lab)
    ax_t.set_xlabel(xlab_t); ax_t.set_ylabel(ylab_t)
    ax_t.set_title("Temporal power P(t)")
    ax_t.grid(True); ax_t.legend()

    ax_w.plot(f_plot, S_total_plot, label="Total (within aperture)")
    for Sj, lab in zip(S_regions_plot, labels_w):
        ax_w.plot(f_plot, Sj, linestyle="--", label=lab)
    ax_w.set_xlabel(xlab_w); ax_w.set_ylabel(ylab_w)
    ax_w.set_title("Spectral power S(ω)")
    ax_w.grid(True); ax_w.legend()

    fig.tight_layout()
    plt.show()
    return 0


import math
import torch
import numpy as np
import matplotlib.pyplot as plt

@torch.inference_mode()
def visualize_mirror(
    sim,
    Hxy: torch.Tensor | None = None,
    *,
    # 若未传 Hxy，可直接从 CSV 构建：
    csv_path: str | None = None,
    col_range: str = "r range (mm)",
    col_height: str = "d_abs_final (µm)",
    dr_fwhm: float = 40e-6,
    aperture: str | float | None = "3w",
    # 光学参数
    theta_deg: float = 0.0,   # 入射角（度）
    n0: float = 1.0,          # 传播介质折射率（空气≈1）
    # 画图选项
    crop_factor: float = 2.0, # 视野 ~ ±(crop_factor × w)
    line_axis: str = "x"      # 'x' 表示取 y≈0 的中心线；'y' 取 x≈0
):
    """
    可视化反射镜台阶高度 H(x,y) 及由几何光程导致的群时延 Δt(x,y)。

    公式：
        L(x,y) = 2 cosθ · H(x,y)
        φ0(x,y) = (n0·ω0/c) · L(x,y)
        Δt(x,y) = (n0/c) · L(x,y) = (2 n0 cosθ / c) · H(x,y)

    参数：
      • 若已在外部构建了 Hxy（单位 m），可直接传入；
        否则提供 csv_path，本函数会调用 build_mirror_height_map() 读取并生成。
    """
    # 1) 准备 H(x,y)
    if Hxy is None:
        assert csv_path is not None, "未传 Hxy 时必须提供 csv_path 才能读取镜面台阶 CSV"
        Hxy = build_mirror_height_map(sim,
                                      csv_path=csv_path,
                                      col_range=col_range,
                                      col_height=col_height,
                                      dr_fwhm=dr_fwhm,
                                      aperture=aperture)
    Hxy = Hxy.to(sim.X.device, dtype=torch.float32)
    H   = Hxy.detach().cpu().numpy()  # [m]

    # 2) 网格坐标
    x = sim.x.detach().cpu().numpy()
    y = sim.y.detach().cpu().numpy()

    # 3) 选中心线
    if str(line_axis).lower() == "x":
        iy0    = int(torch.argmin(torch.abs(sim.y)).item())
        coord  = x
        H_line = H[:, iy0]
        xlabel = "x [m]"
    else:
        ix0    = int(torch.argmin(torch.abs(sim.x)).item())
        coord  = y
        H_line = H[ix0, :]
        xlabel = "y [m]"

    # 4) 计算几何光程与群时延（中心线）
    cos_theta   = math.cos(math.radians(theta_deg))
    L_line      = 2.0 * cos_theta * H_line                # [m]
    tau_line_s  = (n0 / sim.c) * L_line                   # [s]
    tau_line_fs = tau_line_s * 1e15                        # [fs]

    # 5) 视野范围（尽量与 visualize_film 一致）
    try:
        w = sim._infer_beam_radius()
    except Exception:
        w = None
    r_grid = float(min(np.max(np.abs(x)), np.max(np.abs(y))))
    lim = float(min(crop_factor * w, 0.98 * r_grid)) if (w is not None and w > 0) else 0.8 * r_grid

    # 6) 三联图：H(x,y)、中心线 H、中心线 Δt
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    # 左：H(x,y)（颜色条用微米单位直观一些）
    im = ax1.imshow(
        H * 1e6, origin="lower",
        extent=[x.min(), x.max(), y.min(), y.max()],
        aspect="equal"
    )
    ax1.set_title("Mirror height $H(x,y)$ [µm]")
    fig.colorbar(im, ax=ax1, label="H [µm]")
    ax1.set_xlabel("x [m]"); ax1.set_ylabel("y [m]")
    ax1.set_xlim(-lim, lim); ax1.set_ylim(-lim, lim)

    # 中：中心线高度（µm）
    ax2.plot(coord, H_line * 1e6)
    ax2.set_xlabel(xlabel); ax2.set_ylabel("Height [µm]")
    ax2.set_title("Center-line height")
    ax2.set_xlim(-lim, lim); ax2.grid(True)

    # 右：中心线群时延（fs）
    ax3.plot(coord, tau_line_fs)
    ax3.set_xlabel(xlabel); ax3.set_ylabel("Δt [fs]  (vs flat mirror)")
    ax3.set_title(f"Center-line group delay (n0={n0:g}, θ={theta_deg:.1f}°)")
    ax3.set_xlim(-lim, lim); ax3.grid(True)

    fig.tight_layout(); plt.show()

    # 可选返回，便于后续数值处理
    return {
        "Hxy": Hxy.detach().cpu(),
        "coord": coord,
        "H_line_um": H_line * 1e6,
        "tau_line_fs": tau_line_fs,
        "theta_deg": theta_deg,
        "n0": n0
    }

    
import psutil

def check_mem(gpu_id: int = 0):
    """Print GPU 显存 + CPU 内存使用情况."""
    # ---------------- GPU ----------------
    props      = torch.cuda.get_device_properties(gpu_id)
    total_mem  = props.total_memory
    reserved   = torch.cuda.memory_reserved(gpu_id)
    allocated  = torch.cuda.memory_allocated(gpu_id)

    free_in_pool = reserved - allocated
    gpu_avail    = total_mem - reserved

    print("======= GPU memory (device", gpu_id, ") =======")
    print(f"Total       : {total_mem/1024**2:8.1f} MB")
    print(f"Allocated   : {allocated/1024**2:8.1f} MB")
    print(f"Reserved    : {reserved/1024**2:8.1f} MB "
          f"(free in pool: {free_in_pool/1024**2:8.1f} MB)")
    print(f"Still avail : {gpu_avail/1024**2:8.1f} MB")

    # ---------------- CPU ----------------
    vm = psutil.virtual_memory()
    total_ram = vm.total
    avail_ram = vm.available

    proc = psutil.Process(os.getpid())
    rss  = proc.memory_info().rss          # Resident Set Size (bytes)

    print("\n======= CPU memory =========")
    print(f"Total RAM    : {total_ram/1024**3:6.2f} GB")
    print(f"Available RAM: {avail_ram/1024**3:6.2f} GB")
    print(f"Process RSS  : {rss/1024**2:6.1f} MB")
    print("-" * 45)

def print_gpu_usage(gpu_id: int = 0, where = None):
    props     = torch.cuda.get_device_properties(gpu_id)
    total     = props.total_memory
    reserved  = torch.cuda.memory_reserved(gpu_id)
    allocated = torch.cuda.memory_allocated(gpu_id)
    available = total - reserved

    if where != None:
        print(where,':')
    print(f"GPU {gpu_id} memory usage: {allocated/1024**2:.1f} MB, available: {available/1024**2:.1f} MB")

import pandas as pd

def inspect_simulator(sim):
    data = []
    for name, val in vars(sim).items():
        if isinstance(val, torch.Tensor):
            mem_bytes = val.numel() * val.element_size()
            device = str(val.device)
            dtype = str(val.dtype)
            shape = tuple(val.shape)
        elif hasattr(val, 'nbytes'):  # e.g., numpy arrays
            mem_bytes = val.nbytes
            device = 'CPU (numpy)'
            dtype = str(val.dtype)
            shape = tuple(val.shape)
        else:
            continue
        data.append({
            'Attribute': name,
            'Type': type(val).__name__,
            'Shape': shape,
            'Dtype': dtype,
            'Device': device,
            'Memory (MB)': mem_bytes / (1024**2)
        })
    df = pd.DataFrame(data)
    df = df.sort_values(by='Memory (MB)', ascending=False).reset_index(drop=True)
    return df


import numpy as np, torch

@torch.inference_mode()
def apply_reflective_echelon_beta01(
    A_in, sim, Hxy,
    *,
    n0=1.0,                  # 镜面所在介质折射率（空气=1）
    theta_deg=0.0,           # 入射角
    amplitude_r=1.0,         # 幅度反射系数（强度R=|r|^2）
    phi_const=np.pi,         # 常数反射相位
    omega_chunk: int = 1,    # 每次处理多少个频点(>=1)；1 最省显存，>1 更快
    tile_x: int | None = None  # 按 x 方向切 tile；None=不切，显存吃紧时可设 32/64/128
):
    """
    仅β0与β1作用：
      Â_out = r * Â_in * exp{i[(β0 + β1Ω) L(x,y) + φ_const]},
    其中 L(x,y) = 2 cosθ H(x,y), β0 = n0 ω0 / c, β1 = n0 / c.

    低内存实现要点：
      • 不构造 3D 相位；β0 用 2D 相位一次性乘上，
      • β1·Ω 用“频片循环”乘上，仅保留一张 2D 临时相位。
      • 峰值显存 ≈ A_w(1×场) + A_after(1×场) + O(Nx*Ny) 2D 临时。
    """
    import math, torch
    dev = sim.T.device

    # 0) 准备输入场（在 GPU / complex64）
    A = (A_in if torch.is_tensor(A_in) else torch.as_tensor(A_in))
    A = A.to(dev, dtype=torch.complex64, non_blocking=True)
    
    # 1) 计算光程 L(x,y)（2D float32）
    cos_theta = math.cos(math.radians(theta_deg))
    Lxy = (2.0 * cos_theta) * Hxy.to(dev, dtype=torch.float32, non_blocking=True)
    torch.cuda.empty_cache()
    # 2) 频域化（得到 A_w），立刻释放时域 A 以降低峰值
    _, A_w = sim.fft_t(A)     # A_w: (Nx,Ny,Nt) complex64
    del A
    # 不建议频繁 empty_cache，会打乱缓存分配；留到大步清理点

    # 3) 全局幅度与常数相位（标量）
    #    r * e^{i phi_const}
    r_phase = complex(float(amplitude_r * math.cos(phi_const)),
                      float(amplitude_r * math.sin(phi_const)))
    A_w.mul_(r_phase)

    # 4) β0 · L(x,y) —— 只是一张 2D 相位图
    beta0 = float(n0 * sim.omega0 / sim.c)   # [rad/m]
    phi0_xy = (beta0 * Lxy)                  # float32, (Nx,Ny)
    phase0_xy = torch.complex(torch.cos(phi0_xy), torch.sin(phi0_xy))  # complex64, (Nx,Ny)
    A_w.mul_(phase0_xy[..., None])           # 原地广播，不产生 3D 副本
    del phi0_xy, phase0_xy

    # 5) β1·L(x,y)·Ω —— 按频片/小块处理，避免 3D 相位
    beta1 = float(n0 / sim.c)                # [s/m]
    omega = sim.omega.to(dev)                # (Nt,)

    Nx = sim.Nx
    Ny = sim.Ny
    tx = Nx if tile_x is None else int(tile_x)
    oc = max(1, int(omega_chunk))            # 每次处理 oc 个频点
    torch.cuda.empty_cache()

    for xs in range(0, Nx, tx):              # 可选：按 x 切块，进一步压峰值
        xe = min(xs + tx, Nx)
        L_blk = Lxy[xs:xe, :]                # (tx,Ny) float32

        if oc == 1:
            # —— 最省显存：逐频点处理 —— #
            for k in range(omega.numel()):
                theta_xy = -(beta1 * float(omega[k])) * L_blk       # (tx,Ny)
                phase_xy  = torch.complex(torch.cos(theta_xy),
                                           torch.sin(theta_xy))    # (tx,Ny) complex64
                A_w[xs:xe, :, k].mul_(phase_xy)                    # 原地乘
        else:
            # —— 小块按频点批量处理（更快；显存≈ Nx*Ny*oc*复数） —— #
            for k0 in range(0, omega.numel(), oc):
                k1 = min(k0 + oc, omega.numel())
                om = omega[k0:k1]                                  # (oc,)
                # 构造 (tx,Ny,oc) 的小相位块
                theta = (beta1 * L_blk[..., None]) * om[None, None, :]     # float32
                phase = torch.complex(torch.cos(theta), torch.sin(theta))  # complex64
                A_w[xs:xe, :, k0:k1] *= phase
                del theta, phase

    del Lxy
    torch.cuda.empty_cache()

    # 6) 回到时域（此刻峰值≈ A_w + A_after；~2×完整场）
    _, A_after = sim.ifft_t(A_w)
    # 如需继续释放，可在调用处再 del A_w/empty_cache
    sim.A_out=None
    torch.cuda.empty_cache()
    return A_after.detach().cpu()



"""
 aperture_and_compress_full.py — GPU‑friendly *full‑pulse‑only* compression
 -----------------------------------------------------------------------------
 *2025‑07‑24 rewrite*
 
 ▸ **What changed?**  The original implementation kept a full 3‑D electric‑field
   tensor for *every* GDD point during the scan, which could blow up GPU memory
   when `N_scan` was large.  The new design:

   1.  Splits the task into two clear steps:
        • **`find_best_gdd()`** —— scan the GDD list, **never** stores the
          time‑domain field; only tracks the minimum TBP and its GDD.
        • **`compress_with_gdd()`** —— given a field & a single GDD, performs the
          compression once and *optionally* returns the compressed field.

   2.  `aperture_and_compress()` first calls `find_best_gdd()` (cheap & lean),
       then calls `compress_with_gdd()` *once* with `return_field=True` to get
       the final diagnostics / plots.

   3.  All helper maths (RMS width, FWHM, spectrum…) remain GPU‑native Torch.

 Usage
 -----
 >>> sim.propagate()
 >>> from aperture_and_compress_full import aperture_and_compress
 >>> res = aperture_and_compress(sim, D_min=-3e-27, D_max=0.0, N_scan=401)

 The returned dict contains the optimal GDD, TBP, FWHM values and, by default,
 the compressed field tensor `A_comp`.  Set `return_field=False` in the public
 API if you only need the numbers.
 """

# -----------------------------------------------------------------------------
# 1. Torch‑native helper metrics
# -----------------------------------------------------------------------------

def _rms_width(x: torch.Tensor, y: torch.Tensor):
    y = torch.abs(y)
    norm = torch.trapz(y, x)
    if norm == 0:
        return 0.0
    mean = torch.trapz(x * y, x) / norm
    var = torch.trapz((x - mean) ** 2 * y, x) / norm
    return torch.sqrt(var)



def _fwhm(x, y):
    """
    Full‑width at half‑maximum for a 1‑D profile.
    • Accepts torch.Tensor (CPU/CUDA) or np.ndarray.
    • 自动处理“峰值在数组两端”的 FFT 原生排布。
    """
    import torch, numpy as np

    # --------- 转成 torch(cpu) 统一后端 ----------
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    if not torch.is_tensor(y):
        y = torch.as_tensor(y)
    if x.is_cuda: x = x.cpu()
    if y.is_cuda: y = y.cpu()

    y = torch.abs(y)
    if y.max() == 0:
        return 0.0

    half = 0.5 * y.max()
    idx  = torch.nonzero(y >= half, as_tuple=False).flatten()
    if idx.numel() < 2:
        return 0.0

    # --------- 判断是否“跨端” ----------------------
    wraps = (idx[0] == 0) and (idx[-1] == len(y) - 1)

    # -- 若跨端，把数组 roll，让峰值到中心，再正常计算 --
    if wraps:
        peak = torch.argmax(y).item()          # 零频 or 最高点
        shift = len(y)//2 - peak
        y = torch.roll(y, shifts=shift, dims=0)
        x = torch.roll(x, shifts=shift, dims=0)
        idx  = torch.nonzero(y >= half, as_tuple=False).flatten()

    # --------- 线性插值左右半高点 ------------------
    def _interp(i_left):
        i_right = i_left + 1
        if i_right >= len(x):                  # 防止末尾越界
            return x[i_left].item()
        x0, x1 = x[i_left],  x[i_right]
        y0, y1 = y[i_left],  y[i_right]
        return (x0 + (half - y0) * (x1 - x0) / (y1 - y0)).item()

    i0, i1 = idx[0], idx[-1]
    x_l = _interp(i0-1 if i0 > 0 else i0)
    x_r = _interp(i1)
    return x_r - x_l





# -----------------------------------------------------------------------------
# 2. Low‑level helpers (time profile & spectrum)
# -----------------------------------------------------------------------------

def _ensure_dev(x: torch.Tensor, dev):
    return x.to(dev, non_blocking=True) if x.device != dev else x

def _time_profile(A, sim):
    # 时间投影只做 sum，不要求在 GPU 上
    A = _ensure_dev(A, sim.T.device) 
    I_t=torch.sum(torch.abs(A)**2, dim=(0,1)) * sim.dx * sim.dy
    return I_t

def _spectrum(A, sim):
    A = _ensure_dev(A, sim.T.device)     # 确保与 sim.fft_t 的缩放张量同设备

    _, A_w = sim.fft_t(A)
    del A
    torch.cuda.empty_cache()
    S_w = torch.sum(torch.abs(A_w)**2, dim=(0,1)) * sim.dx * sim.dy
    return S_w.detach().cpu().numpy()



# -----------------------------------------------------------------------------
# 3. Core utility – compress once with a given GDD
# -----------------------------------------------------------------------------

@torch.inference_mode()
def compress_with_gdd(A_out, sim, D, *, return_field=True):
    """Compress a field with second‑order dispersion *D*.

    Parameters
    ----------
    A_gpu        : (Nx,Ny,Nt) complex Torch tensor (on the same device as *sim*)
    sim          : PulseGNLSESimulator instance (provides grids & FFT helpers)
    D            : float, group‑delay dispersion [s²]
    return_field : bool, if *True* include the compressed field tensor in the
                   returned dict; otherwise memory‑cheap.

    Returns
    -------
    dict
        tbp        : time–bandwidth product
        fwhm_t     : FWHM in time (s)
        sigma_w    : RMS bandwidth (rad/s)
        A_comp     : compressed field (Torch tensor) – only if *return_field*
    """
    ω = sim.omega
    
    dx, dy = sim.dx, sim.dy

    A_gpu = A_out.to(sim.T.device)

    '''
    phase = torch.exp(0.5j * D * ω ** 2)
    _, A_w = sim.fft_t(A_gpu)
    print_gpu_usage(where='A_w')
    A_w_D = A_w * phase[None, None, :]
    print_gpu_usage(where='A_w_D')

    # --- metrics in the spectral domain ------------------------------------
    sigma_w = _rms_width(ω, torch.sum(torch.abs(A_w_D) ** 2, dim=(0, 1)) * dx * dy)
    print_gpu_usage(where='sigma_w')

    # --- back to time domain to get temporal metrics -----------------------
    _, A_t = sim.ifft_t(A_w_D)
    print_gpu_usage(where='A_t')
    P_t = _time_profile(A_t, sim)
    '''
    A_w_D =  sim.fft_t(A_gpu)[1]
    del A_gpu
    torch.cuda.empty_cache()
    A_w_D = A_w_D* torch.exp(0.5j * D *ω ** 2)[None, None, :]
    torch.cuda.empty_cache()
    I_w_D=torch.sum(torch.abs(A_w_D) ** 2, dim=(0, 1)) 

    I_w_D=I_w_D * dx * dy
    
    sigma_w = _rms_width(ω,I_w_D)

    A_t_D=sim.ifft_t(A_w_D)[1]
    del A_w_D
    torch.cuda.empty_cache()
    
    P_t = _time_profile(A_t_D, sim)

    sigma_t = _rms_width(sim.T, P_t)

    fwhm_t = _fwhm(sim.T, P_t)

    tbp = (sigma_t * sigma_w).item()

    out = dict(tbp=tbp, fwhm_t=fwhm_t, sigma_w=sigma_w.item())

    if return_field:
        out["A_comp"] = A_t_D.detach().cpu()   # ← 放到 CPU

    return out


# -----------------------------------------------------------------------------
# 4. Fast scan – find the best GDD without keeping fields
# -----------------------------------------------------------------------------

def find_best_gdd(A_out, sim, D_scan):
    """Return the GDD in *D_scan* that minimises the TBP.

    Only TBP & D are tracked – *no* field tensor is kept, saving memory.
    """
    best_tbp = float("inf")
    best_D = D_scan[0]
    tbp_curve = []
    flag=0
    for D in D_scan:
        #print(flag)
        flag+=1
        res = compress_with_gdd(A_out, sim, D, return_field=False)
        tbp_curve.append(res["tbp"])
        if res["tbp"] < best_tbp:
            best_tbp = res["tbp"]
            best_D = D

    return best_D, best_tbp, tbp_curve


# -----------------------------------------------------------------------------
# 5. Public API – high‑level helper (scan + one compression)
# -----------------------------------------------------------------------------
import math
import torch

@torch.inference_mode()
def design_circular_aperture_for_fraction(
    A_field: torch.Tensor, sim,
    *,
    pass_fraction: float,       # 0< f ≤1，如 0.85 表示通过 85% 能量
    fwhm: float = 40e-6,        # 软边 FWHM（米）
    tol: float = 1e-3,          # 通过能量的相对误差容忍
    max_iter: int = 40
):
    """
    依据目标通过能量比例，在 x-y 平面为 A_field 设计一个圆形软边光阑：
        M_R(x,y) = soft_unit_step(R - r, fwhm)
    其中 r = sqrt(x^2 + y^2)，R 由二分法自动求得，使：
        frac(R) = ∬ I_xy M_R dxdy / ∬ I_xy dxdy  ≈ pass_fraction

    返回：
      M_xy       : (Nx,Ny) float32 ∈[0,1] 的软边光阑掩膜（在 sim.T.device 上）
      R_opt      : 最终半径（米）
      frac_ach   : 实际通过比例
    """
    assert 0.0 < pass_fraction <= 1.0, "pass_fraction 必须在 (0,1]"

    dev = sim.T.device
    # —— I_xy = ∫ |A|^2 dt ——（dx·dy 会在比值中相消）
    A = (A_field if torch.is_tensor(A_field) else torch.as_tensor(A_field)).to(dev, dtype=torch.complex64, non_blocking=True)
    I_xy = torch.sum(A.real*A.real + A.imag*A.imag, dim=2) * float(sim.dT)   # (Nx,Ny) float32
    del A; torch.cuda.empty_cache()

    denom = torch.sum(I_xy)  # ∬ I_xy dxdy 中 dxdy 省略，做比例会相消
    if denom.item() <= 0:
        # 空能量兜底：不给光阑
        M_full = torch.ones((sim.Nx, sim.Ny), device=dev, dtype=torch.float32)
        return M_full, float(min(sim.x.abs().max(), sim.y.abs().max())), 1.0

    # 半径网格
    r = torch.sqrt(sim.X.to(dev)**2 + sim.Y.to(dev)**2)  # (Nx,Ny)

    # 目标函数：给定 R → 通过比例
    def frac_of_R(R: float) -> float:
        M = soft_unit_step(R - r, fwhm)      # 内部≈1，外部≈0
        num = torch.sum(I_xy * M)
        return float((num / denom).item())

    # 特例：≈1 就直接满片
    if pass_fraction >= 1.0 - 1e-6:
        M_full = soft_unit_step((r.max().item()*2) - r, fwhm)  # 基本上全 1
        return M_full, float(r.max().item()), 1.0

    # 二分法求 R
    lo = 0.0
    hi = float(r.max().item())
    frac_lo = frac_of_R(lo)  # ≈0
    frac_hi = frac_of_R(hi)  # ≤1

    # 保护：若 hi 仍 < 目标（数值边缘），就取 hi
    if frac_hi < pass_fraction:
        M_hi = soft_unit_step(hi - r, fwhm)
        return M_hi, hi, frac_hi

    R = hi
    frac_mid = frac_hi
    for _ in range(max_iter):
        mid = 0.5*(lo + hi)
        fmid = frac_of_R(mid)
        if fmid < pass_fraction:
            lo = mid
        else:
            hi = mid
        R = hi
        frac_mid = fmid
        if abs(fmid - pass_fraction) <= tol * pass_fraction:
            break

    M_xy = soft_unit_step(R - r, fwhm).to(torch.float32)
    return M_xy, R, frac_mid


def aperture_and_compress(
    sim,
    *,
    D_min=-3e-27,
    D_max=0.0,
    N_scan=401,
    plot=True,
    return_field=True,
    # ===== 新增：空间光阑选项 =====
    aperture_pass_fraction: float | None = None,   # 传 None/不传 ⇒ 不加光阑
    aperture_fwhm: float = 40e-6,                  # 软边宽度（米）
    return_aperture_mask: bool = False,            # 是否把光阑掩膜也放进结果
):
    """
    先（可选）加空间光阑，再扫描 GDD 压缩。
      • aperture_pass_fraction=None    ⇒ 不加光阑（与原版完全一致）
      • aperture_pass_fraction∈(0,1]  ⇒ 自动求光阑半径，使通过能量≈该比例
    """
    if sim.A_out is None:
        raise ValueError("sim.A_out is None — run sim.propagate() first.")
    A_out = sim.A_out   # CPU 张量
    sim.A_out=None
    torch.cuda.empty_cache()

    # ---------- (A) 可选：空间光阑 ----------
    aperture_info = None
    if aperture_pass_fraction is not None:
        # 设计光阑（在 GPU 上算），并应用到场
        M_xy, R_opt, frac_ach = design_circular_aperture_for_fraction(
            A_out, sim,
            pass_fraction=float(aperture_pass_fraction),
            fwhm=float(aperture_fwhm),
            tol=1e-3, max_iter=40,
        )
        A_base_gpu = (A_out if torch.is_tensor(A_out) else torch.as_tensor(A_out)).to(sim.T.device, dtype=torch.complex64)
        A_base_gpu = A_base_gpu * M_xy[..., None].to(A_base_gpu.dtype)   # 应用光阑
        A_base = A_base_gpu.detach().cpu()
        del A_base_gpu
        torch.cuda.empty_cache()

        aperture_info = dict(
            pass_target=float(aperture_pass_fraction),
            pass_achieved=float(frac_ach),
            radius_m=float(R_opt),
            fwhm_m=float(aperture_fwhm),
        )
    else:
        # 不加光阑 ⇒ 使用原来的输出场
        A_base = A_out

    # ---------- (B) 粗扫 GDD（对 A_base） ----------
    D_scan = np.linspace(D_min, D_max, N_scan)
    D_best, tbp_min, tbp_curve = find_best_gdd(A_base, sim, D_scan)
    
    sigma_w_init = _rms_width(
        sim.omega,
        torch.sum(torch.abs(sim.fft_t(sim.A_in.to(sim.T.device, non_blocking=True))[1]) ** 2, dim=(0, 1)) * sim.dx * sim.dy,
    )

    if plot:
        S_w_in   = _spectrum(sim.A_in, sim);     S_w_in_c   = np.fft.fftshift(S_w_in)
        P_t_in   = _time_profile(sim.A_in, sim).cpu().numpy()
        sim.A_in=None
    
    torch.cuda.empty_cache()
    # ---------- (C) 最终压缩 ----------
    final = compress_with_gdd(A_base, sim, D_best, return_field=return_field)
    torch.cuda.empty_cache()
    check_mem()
    # ---------- (D) 控制台摘要 ----------

    print(
        f"FWHM_t(best): {final['fwhm_t']*1e15:.2f} fs  "
        f"(GDD={D_best:.3e} s²,  TBP={final['tbp']:.3f})"
    )
    if aperture_info is not None:
        print(f"[Aperture] target={aperture_info['pass_target']*100:.1f}% "
              f"→ achieved={aperture_info['pass_achieved']*100:.1f}% , "
              f"R≈{aperture_info['radius_m']*1e3:.2f} mm,  FWHM={aperture_info['fwhm_m']*1e6:.1f} µm")

    # ---------- (E) 可选绘图（把“Output”基准换成 A_base） ----------
    if plot:
        omega_c = torch.fft.fftshift(sim.omega).cpu().numpy()
        freq_THz_c = omega_c / (2*np.pi*1e12)

        # 频谱：输入 vs（光阑后）输出 vs 压缩后
        S_w_base = _spectrum(A_base,   sim);     S_w_base_c = np.fft.fftshift(S_w_base)
        S_w_comp_c = None
        if return_field:
            S_w_comp = _spectrum(final["A_comp"], sim)
            S_w_comp_c = np.fft.fftshift(S_w_comp)

        # 时间域
        T_fs = (sim.T * 1e15).cpu().numpy()
        P_t_base = _time_profile(A_base,   sim).cpu().numpy()


        
        P_t_comp = None
        if return_field:
            P_t_comp = _time_profile(final["A_comp"], sim).cpu().numpy()

        fwhm_in = _fwhm(sim.T, torch.as_tensor(P_t_in))
        half_tr_fs = min(5 * fwhm_in * 1e15, abs(T_fs[-1]))

        # 频域视野
        half_fr = 2 * _fwhm(torch.as_tensor(omega_c), torch.as_tensor(S_w_in_c)) / (2*np.pi*1e12)

        fig, axes = plt.subplots(4, 1, figsize=(6, 12))

        # (0) 时间域：输入 / （光阑后）输出 / 压缩后
        axes[0].plot(T_fs, P_t_in,  label="Input")
        axes[0].plot(T_fs, P_t_base,label="Output (after aperture)" if aperture_info else "Output")
        if P_t_comp is not None:
            axes[0].plot(T_fs, P_t_comp, ls="--", label="Compressed")
        axes[0].set_xlim(-half_tr_fs, half_tr_fs)
        axes[0].set(xlabel="Time [fs]", ylabel="∫|A|² dx dy  [J/s]",
                    title="Temporal power")
        axes[0].legend(); axes[0].grid()

        # (1) 频谱：输入 / （光阑后）输出 / 压缩后（归一化对比）
        axes[1].plot(freq_THz_c, S_w_in_c   / max(S_w_in_c.max(), 1e-30), label="Before (input)")
        axes[1].plot(freq_THz_c, S_w_base_c / max(S_w_base_c.max(), 1e-30),
                     label="After aperture" if aperture_info else "Before (output)")
        if S_w_comp_c is not None:
            axes[1].plot(freq_THz_c, S_w_comp_c / max(S_w_comp_c.max(), 1e-30),
                         ls="--", label="After compression")
        axes[1].set_xlim(-half_fr, half_fr)
        axes[1].set(xlabel="Frequency offset [THz]", ylabel="Norm. power", title="Spectrum")
        axes[1].legend(); axes[1].grid()

        # (2) TBP-GDD 扫描
        axes[2].plot(D_scan*1e28, tbp_curve)
        axes[2].scatter(D_best*1e28, tbp_min, color="red")
        axes[2].axhline(0.5, ls="--", color="gray")
        axes[2].set(xlabel="GDD [×10⁻²⁸ s²]", ylabel="TBP", title="GDD scan")
        axes[2].grid()

        # (3) 压缩前后时间域（以“光阑后输出”为基准）
        P_t_before = _time_profile(A_base, sim).cpu().numpy()
        axes[3].plot(T_fs, P_t_before, label="Before (after aperture)" if aperture_info else "Before")
        if return_field:
            P_t_after = _time_profile(final["A_comp"], sim).cpu().numpy()
            axes[3].plot(T_fs, P_t_after, ls="--", label="After")
        axes[3].set(xlabel="Time [fs]", ylabel="∫|A|² dx dy  [J/s]",
                    title="Temporal profile (before vs after)")
        axes[3].legend(); axes[3].grid()

        fig.tight_layout(); plt.show()

    # ---------- (F) 结果 ----------
    result = dict(D_opt=D_best, tbp=final["tbp"], fwhm_t=final["fwhm_t"],
                  sigma_w=final["sigma_w"], tbp_curve=tbp_curve)
    if return_field:
        result["A_comp"] = final["A_comp"]
    if aperture_info is not None:
        result["aperture"] = aperture_info
        if return_aperture_mask:
            # 返回 CPU 版掩膜，便于外部复用/保存
            result["aperture"]["mask_xy"] = M_xy.detach().cpu()
    sim.A_out= A_out
    return result


import os, torch, gc

def save_A_every_step(step, sim, save_dir="A_dump", every=1):
    if step % every:                 # 只在 every 的倍数步保存
        return
    os.makedirs(save_dir, exist_ok=True)
    fname = os.path.join(save_dir, f"A_{step:05d}.pt")
    # 直接序列化 GPU tensor；torch 会在内部搬到 CPU 并写文件
    torch.save(sim.A.detach(), fname)
    # 显式清理 CPU buffer，减轻 host 内存占用
    gc.collect()

# ========= quick diagnostics to print before every visualization =========
@torch.inference_mode()
def diagnose_and_print(sim, *, field="current", label=None):
    """
    field: 'input' | 'current' | 'output'
    打印：Energy, Peak power, FWHM/σ_t, FWHM/σ_v, FWHM_x/y, t_peak, t_centroid, B-integral
    """
    dev = sim.T.device
    if field == "input":
        A_src = sim.A_in
    elif field == "output":
        if sim.A_out is None:
            raise ValueError("sim.A_out is None — run propagate() first")
        A_src = sim.A_out
    else:
        A_src = sim.A

    with torch.no_grad():
        A_gpu = (A_src if torch.is_tensor(A_src) else torch.from_numpy(A_src)).to(dev, non_blocking=True)
        # ---- Energy & temporal power ----
        torch.cuda.empty_cache()
        I=torch.abs(A_gpu)**2
        I_t=torch.trapz(I, sim.T)
        E = torch.trapz(torch.trapz(I_t, sim.y), sim.x).item()

        # 空间 FWHM（取 y≈0 / x≈0 的线）
        I_xy = torch.trapz(I, sim.T, dim=2)

        P_t = torch.sum(I, dim=(0, 1)) * sim.dx * sim.dy
        del I
        torch.cuda.empty_cache()
        P_peak = P_t.max().item()

        # 时间域指标
        sigma_t = _rms_width(sim.T, P_t).item()
        fwhm_t  = _fwhm(sim.T, P_t)
        t_peak  = sim.T[torch.argmax(P_t)].item() if P_peak > 0 else 0.0
        denom   = torch.trapz(P_t, sim.T)
        t_cm    = (torch.trapz(sim.T * P_t, sim.T) / denom).item() if float(denom) != 0.0 else 0.0

        # 频域指标
        _, A_w = sim.fft_t(A_gpu)
        A_w =A_w.detach().cpu().numpy()
        I_w= np.sum(np.abs(A_w)**2, axis=(0, 1))
        I_w=torch.from_numpy(I_w).to(dev)
        S_w = I_w * sim.dx * sim.dy
        sigma_w = _rms_width(sim.omega, S_w).item()
        fwhm_w  = _fwhm(sim.omega, S_w)

        # 空间 FWHM（取 y≈0 / x≈0 的线）
        iy0 = torch.argmin(torch.abs(sim.y)).item()
        ix0 = torch.argmin(torch.abs(sim.x)).item()
        fwhm_x = float(_fwhm(sim.x, I_xy[:, iy0]))
        fwhm_y = float(_fwhm(sim.y, I_xy[ix0,  :]))

        B = getattr(sim, "B_running", float("nan"))

    name = label or field.upper()
    print(f"\n===== Diagnostics [{name}] =====")
    print(f"Energy                : {E:.4e} J")
    print(f"Peak power            : {P_peak/1e6:.2f} MW")
    print(f"Time  — FWHM / σ_t    : {fwhm_t*1e15:.2f} fs / {sigma_t*1e15:.2f} fs")
    print(f"Freq  — FWHM / σ_v    : {fwhm_w/2/np.pi/1e12:.2f} THz / {sigma_w/2/np.pi/1e12:.2f} THz")
    print(f"Space — FWHM_x / FWHM_y: {fwhm_x*1e3:.2f} mm / {fwhm_y*1e3:.2f} mm")
    print(f"Centers (t_peak / t_cm): {t_peak*1e15:.2f} fs / {t_cm*1e15:.2f} fs")
    if field != "input":
        print(f"B-integral (running)  : {B:.3f} (approx.)")
    print("================================\n")

# ========= 2) 扫描：能量通过比例 100% → 10%，只记录压缩后的 FWHM =========
import numpy as np
import matplotlib.pyplot as plt

def sweep_energy_clip_and_plot(sim,
                               passes=None,             # 能量通过比例列表（0~1）
                               D_min=-3e-28, D_max=0.0, # GDD 扫描范围
                               N_scan=201,              # GDD 扫描点数
                               aperture_fwhm=30e-6,     # 软边宽度
                               save=True,               # 是否保存结果
                               save_dir=None,           # 保存目录；默认用全局 BASE_SAVE
                               filename_prefix="fwhm_vs_energy_after_mirror"  # 文件名前缀
                               ):
    """
    对每个能量通过比例 pass_fraction：
      1) 自动设计圆形软边光阑（通过比例=pass_fraction）
      2) 扫描 GDD，找到 TBP 最小处
      3) 记录对应的压缩后 FWHM_t（秒 → fs）
    不返回压缩场，仅返回数值并画图；可将 (x,y) 数据与图保存到 BASE_SAVE。
    """
    import os, time
    if passes is None:
        # 100%→10%，步长 2%
        passes = [p/100 for p in range(100, 9, -4)]

    fwhm_fs = []
    for pf in passes:
        res = aperture_and_compress(
            sim,
            D_min=D_min, D_max=D_max, N_scan=N_scan,
            plot=False,             # 不画每次压缩的图
            return_field=False,     # 不返回压缩后的场，节省内存
            aperture_pass_fraction=pf,
            aperture_fwhm=aperture_fwhm,
            return_aperture_mask=False
        )
        fwhm_fs.append(res["fwhm_t"] * 1e15)   # s → fs

        # 可选：释放 cuFFT 计划与缓存（如果你有 release_fft_pool）
        try:
            release_fft_pool()
        except Exception:
            torch.cuda.empty_cache()

    # ========= 3) 画图：能量通过比例(%) vs 压缩后 FWHM(fs) =========
    x_pct = [p*100 for p in passes]
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111)
    ax.plot(x_pct, fwhm_fs, marker='o')
    ax.set_xlabel("Energy pass [%]")            # 能量剪裁理解为“保留百分比”
    ax.set_ylabel("Compressed FWHM [fs]")
    ax.set_title("FWHM vs Energy Pass")
    ax.grid(True)
    # 如需“剪得越狠在右侧”，取消注释下一行
    # ax.invert_xaxis()
    fig.tight_layout()
    plt.show()

    # ========= 4) 保存到 BASE_SAVE（数据 + 图） =========
    if save:
        out_dir = save_dir or (BASE_SAVE if 'BASE_SAVE' in globals() else ".")
        os.makedirs(out_dir, exist_ok=True)
        # 时间戳避免覆盖
        ts = time.strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(out_dir, f"{filename_prefix}_{ts}.csv")
        npz_path = os.path.join(out_dir, f"{filename_prefix}_{ts}.npz")
        fig_path = os.path.join(out_dir, f"{filename_prefix}_{ts}.png")

        # 保存 CSV（两列：energy_pass_pct, fwhm_fs）
        arr = np.column_stack([np.array(x_pct, dtype=float),
                               np.array(fwhm_fs, dtype=float)])
        np.savetxt(csv_path, arr, delimiter=",",
                   header="energy_pass_pct,fwhm_fs", comments="")
        # 保存 NPZ（原始 pass ∈[0,1] 也一并放进去）
        np.savez(npz_path,
                 energy_pass=np.array(passes, dtype=float),
                 energy_pass_pct=np.array(x_pct, dtype=float),
                 fwhm_fs=np.array(fwhm_fs, dtype=float))
        # 保存图
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        print(f"[saved] CSV: {csv_path}\n[saved] NPZ: {npz_path}\n[saved] FIG: {fig_path}")

    return np.array(passes), np.array(fwhm_fs)
