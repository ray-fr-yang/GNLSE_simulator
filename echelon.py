##### from IPython.display import display
import os, gc, pathlib, torch, numpy as np
from datetime import datetime
import matplotlib.pyplot as plt 
from pulse_gnlse import (
    PulseGNLSESimulator,
    aperture_and_compress,
    save_A_every_step,
    visualize_simulator,
    _rms_width,
    _fwhm,
    diagnose_and_print,
    inspect_simulator,
    check_mem,    
    visualize_film,
    build_film_from_csv,
    set_film_simple,
    visualize_time_and_freq_by_regions,
    apply_reflective_echelon_beta01,
    build_mirror_height_map,
    visualize_mirror,
)
from sim_blueprint import save_sim_blueprint
from strehl_after_gdd_noalloc import strehl_after_gdd_noalloc,strehl_ratio_fullfield

def release_fft_pool():
    import torch
    torch.cuda.synchronize()
    # 查看/清理 cuFFT 计划缓存
    try:
        pc = torch.backends.cuda.cufft_plan_cache
    except AttributeError:
        # 旧版本兜底（一般仍叫 cufft_plan_cache）
        pc = torch.backends.cuda.cufft_plan_cache
    try:
        print("[cuFFT] plan cache size before:", pc.size())
    except Exception:
        pass

    pc.clear()                      # ★ 这一步才会真正归还 cuFFT 的大工作区
    # 可选：暂时禁用缓存，防止再次吃满（之后想恢复可以设回 >0）
    pc.max_size = 0                 # 或者设小一点，比如 4

    torch.cuda.empty_cache()        # 把 PyTorch allocator 的空闲块也还给驱动
    torch.cuda.ipc_collect()        # 处理跨进程句柄（一般无伤大雅）
    torch.cuda.synchronize()


# ---------- 工具函数 -------------------------------------------------
def build_simulator(beam_profile):
    """返回初始化后的 PulseGNLSESimulator 实例（GPU 上）"""
    return PulseGNLSESimulator(beam_profile=beam_profile, **COMMON_PARAMS)

def run_one_layer(layer_idx, beam_profile):
    layer_dir = os.path.join(BASE_SAVE, f"layer_{layer_idx:02d}")
    pathlib.Path(layer_dir).mkdir(parents=True, exist_ok=True)

    # 1) 初始化
    sim = build_simulator(beam_profile)

    film, inner_radii, dL_list, names = build_film_from_csv(
        sim,
        csv_path="film/film_layers_after_fine_tweak.csv",  # ← 新文件名
        nz_per_layer=2,
        dr_fwhm=0,
        aperture="3w",
        col_thickness="t_after_fine (mm)"                 # ← 使用细调后的厚度列
    )

    '''
    # 2) 读 CSV 构薄膜（建议给个软边，避免硬边振铃）
    film, inner_radii, dL_list, names = build_film_from_csv(
        sim,
        "film/film_layers_final_codesign.csv",
        nz_per_layer=2,
        dr_fwhm=40e-6,     # 你原来传 0e-6，这里建议 30–60 µm 更稳
        aperture="3w"
    )
    '''
    '''
    set_film_simple(sim,
                    r=[2, 4, 12],
                    t=[0.06, 0.125, 0.45],
                    units="mm",
                    nz_per_layer=2,
                    dr_fwhm=0.0,               # 想要硬边就设 0；建议 30–60 μm 更稳
                    aperture_from_last=True)   # r>3mm 直接光阑截掉
    '''
    # 3) 看一眼厚度图
    # 3) 先看一下厚度与输入场
    visualize_film(sim, crop_factor=1.2, line_axis='x')
    save_sim_blueprint(layer_dir, COMMON_PARAMS, beam_profile, sim=sim, save_A_in=True)

    df = inspect_simulator(sim); display(df)
    diagnose_and_print(sim, field="input", label=f"Layer {layer_idx} — Input")
    #visualize_simulator(sim, field="input"); plt.show()
    ret = strehl_ratio_fullfield(sim, field="input", t_mode="peak",
                                 search_defocus=True, n_alpha=81,
                                 alpha_span_factor=6.0, use_gpu=True, tile_x=64)
    print(ret)   # {'S': ..., 'tidx': ..., 'alpha': ...}
    # 4) 传播（阶梯薄膜）
    snapshots = sim.propagate_with_film(
        apply_surfaces=True,
        store_steps=(),         # 需要就传 (0,1,2,...) 保存快照
        callback=None,          # 或 lambda s,sm: save_A_every_step(s, sm, save_dir=layer_dir, every=1)
        tile_x_kerr=32,
        eps_inc=1e-6,           # 增量几乎恒等就跳过
        skip_empty_masks=True,
        center_camera_midpoint=True
    )

    torch.save(sim.A_out.cpu(), os.path.join(layer_dir, "A_out.pt"))

    # 5) 出口诊断 & 可视化
    diagnose_and_print(sim, field="output", label=f"Layer {layer_idx} — Output")
    visualize_simulator(sim, field="output", zoom_time=2.0); plt.show()
    
    visualize_time_and_freq_by_regions(
        sim,
        field="output",  # 或 'input' / 'current'
        t_unit="fs",
        f_unit="THz",
        normalize=False,
        fftshift=True,
    )

    
    '''
    sim.propagate_in_vacuum(0.20, 10)

    diagnose_and_print(sim, field="output", label=f"Layer {layer_idx} — Output")
    visualize_simulator(sim, field="output", zoom_time=2.0); plt.show()
    '''
    
    
    # 1) 读镜面 CSV → H(x,y)
    '''
    Hxy = build_mirror_height_map(sim, csv_path="film/mirror_echelon_final_codesign.csv",
                                  dr_fwhm=40e-6, aperture="3w")
    res = visualize_mirror(sim, Hxy, theta_deg=0.0, n0=1.0)
    '''

    Hxy = build_mirror_height_map(
        sim,
        csv_path="film/reflective_echelon_design_gdfirst.csv",  # ← 新文件名
        col_height="abs height (µm)",                           # ← 该CSV的高度列名
        dr_fwhm=0,
        aperture="3w"
    )
    res = visualize_mirror(sim, Hxy, theta_deg=0.0, n0=1.0)
    torch.cuda.synchronize(); torch.cuda.empty_cache()

    # 2) 在镜面处“反射一次”（A_out 作为输入）
    A_after_mirror = apply_reflective_echelon_beta01(
        sim.A_out, sim, Hxy,
        n0=1.0,          # 空气
        theta_deg=0,
        amplitude_r=1.0, # 或 amplitude_r=np.sqrt(0.995)
        phi_const=np.pi
    )
    release_fft_pool()
    sim.A_out = A_after_mirror
    del A_after_mirror
    release_fft_pool()
    ret = strehl_ratio_fullfield(sim, field="output", t_mode="peak",
                                 search_defocus=True, n_alpha=81,
                                 alpha_span_factor=6.0, use_gpu=True, tile_x=64)
    print(ret)   # {'S': ..., 'tidx': ..., 'alpha': ...}  
    visualize_time_and_freq_by_regions(
        sim,
        field="output",  # 或 'input' / 'current'
        t_unit="fs",
        f_unit="THz",
        normalize=False,
        fftshift=True,
    )
    
    release_fft_pool()
    torch.cuda.synchronize(); torch.cuda.empty_cache()


    '''
    # 传播结束后
    # 例如：以“初始 FWHM 半宽的 4 倍”为半径；软边 = 5%
    sim.apply_circular_lowpass_by_factor(
        factor=6.0,
        field="output",
        dk_rel=0.0,
        tile_t=32,
        preserve_energy=False,
        preview=True,            # 预览频谱 + 通带
        preview_shift=True,      # 仅用于显示 shift 到中心
        preview_logscale=False,   
        preview_db_floor=-50,    # dB 下限
        preview_span_factor=1.8  # 视野半径≈1.8*k_cut
    )
    '''
    
    # 已写入 sim.A_out（CPU），同时返回 A_filtered
    diagnose_and_print(sim, field="output", label=f"Layer {layer_idx} — Output")
    visualize_simulator(sim, field="output", zoom_time=2.0); plt.show()
    release_fft_pool()
    
    # —— 2) 粗扫 GDD（含画图），拿到最优值与压缩后的场 ——
    res_comp = aperture_and_compress(
        sim,
        D_min=-3e-28, D_max=0.0, N_scan=21,
        plot=True, return_field=True,
        aperture_pass_fraction=1.0,
        aperture_fwhm=30e-6,
        return_aperture_mask=True,   # 可拿到 mask
    )
    D_opt  = res_comp["D_opt"]
    #A_comp = res_comp["A_comp"]          # (Nx,Ny,Nt) on CPU
    
    print(f"[after mirror] best GDD = {D_opt:.3e} s^2; FWHM_t = {res_comp['fwhm_t']*1e15:.2f} fs")
    release_fft_pool()
    sim.A_out = res_comp["A_comp"]
    visualize_time_and_freq_by_regions(
        sim,
        field="output",  # 或 'input' / 'current'
        t_unit="fs",
        f_unit="THz",
        normalize=False,
        fftshift=True,
    )
    # 假设你已经有 D_opt（GDD），且 A_out 在 CPU（numpy 或 torch 都行）
    
    # 没有 stepped_film：用 3w 生成外光阑；有的话仍用第一层的 mask
    if getattr(sim, "stepped_film", None) is None or len(sim.stepped_film.layers) == 0:
        R = 3.0 * float(w)  # w 已定义（米）
        dev = sim.T.device
        X = sim.X.to(dev, dtype=torch.float32, non_blocking=True)
        Y = sim.Y.to(dev, dtype=torch.float32, non_blocking=True)
    
        dr_fwhm = 40e-6   # 软边 FWHM；若想要硬边就设 0
        if dr_fwhm > 0.0:
            a = dr_fwhm / (2.0 * math.log(3.0))  # logistic 软边宽度参数
            r = torch.sqrt(X*X + Y*Y)
            M0 = 1.0 / (1.0 + torch.exp((r - R)/a))
        else:
            M0 = ((X*X + Y*Y) <= (R*R)).to(torch.float32)
    
        # 可选：不再用 X/Y 时释放临时引用（避免额外占用）
        del X, Y
    else:
        M0 = sim.stepped_film.layers[0].mask.to(dtype=torch.float32, device=sim.T.device)

    ret = strehl_after_gdd_noalloc(sim, sim.A_out, res_comp['D_opt'],
                                   t_mode="peak",
                                   search_defocus=True,
                                   n_alpha=81,
                                   aperture_mask=M0)
    print("Strehl =", ret["S"], "  at tidx =", ret["tidx"], "  defocus α =", ret["alpha"])
    
    del sim
    torch.cuda.empty_cache()
    '''
    df = inspect_simulator(sim); display(df)
    
    A_out_cpu = sim.A_out.clone()
    
    return A_out_cpu
    '''
    return 0

    
s=1.31071
w=0.004

# ---------- 公共参数（与原脚本一致，可集中放配置文件） ----------
COMMON_PARAMS = dict(
    pulse_energy    = 300e-3,
    t_fwhm          = 40e-15,
    nt              = 2**10,
    t_window_factor = 50,
    lambda0         = 800e-9,
    dispersion      = {1: 1.463/2.99792458e8, 2: 36e-27, 3: 2.75e-41, 4: -2.79e-56},
    n0              = 1.4530,
    n2              = np.array([[532e-9, 2.24e-20],[1064e-9, 2.14e-20]]),
    L               = 0.04e-3,
    nz              = 50,
    compression_steps = 1000,
    Nx = 2**11, Ny = 2**11,
    x_window =  18*w*2,  y_window = 18*w*2,
    retarded_frame  = True,
    alpha           = 2.5e-4,
    raman           = {'f_R':0.18,'tau1':12.2e-15,'tau2':32e-15},
    noise           = None,
    device          = str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
)

INIT_BEAM = {           # 第一层用超高斯截面
    "shape":"supergauss",
    "waist":w,
    "order":2,
}

BASE_SAVE = "/rds/general/ephemeral/user/fy624/ephemeral/Aug_27th/singlepass/res1"
A_seed = None
N_LAYERS = 1

# 假设 w = 4 mm
inner_radii = [0, 3.0e-3, 3.6e-3]
dL_list     = [0.037e-3, 0.028e-3, 0.072e-3]
nz_per_layer= 10
dr_fwhm     = 40e-6

# 外圆光阑（可选）——若你的设计只到 4.0 mm，外面不关心可加一个软边
aperture_radius = 4.0e-3
aperture_fwhm   = 60e-6



# ---------- 主循环 ---------------------------------------------------
if __name__ == "__main__":
    for layer in range(1, N_LAYERS+1):
        if layer == 1:
            beam_prof = INIT_BEAM
        else:
            beam_prof = {"shape":"import", "field":A_seed, "renorm":False}
        A_seed = run_one_layer(layer, beam_prof)

    print("✅ 所有层已完成并保存")
