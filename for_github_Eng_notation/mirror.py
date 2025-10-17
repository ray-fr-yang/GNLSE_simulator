# ======== mirror.py —— drop-in patch (fully usable) ========
import os, json, math, re
import torch
import numpy as np
def _exclusive_from_cumulative(cum_masks: list[torch.Tensor]) -> list[torch.Tensor]:
    """Convert cumulative masks [M0, M1, ..., MK-1] into mutually exclusive regions [R0, R1, ..., RK-1].
       R0 = center region, R{K-1} = outermost region; intermediate rings are obtained by subtraction."""
    K = len(cum_masks)
    excl = []
    if K == 0:
        return excl
    if K == 1:
        return [cum_masks[0].clamp(0,1)]
    for j in range(K - 1):
        excl.append((cum_masks[j] - cum_masks[j+1]).clamp(0.0, 1.0))
    excl.append(cum_masks[-1].clamp(0.0, 1.0))
    return excl

# --- Missing utility: parse "r range (mm)" such as "0.4–0.7" / "0.4-0.7" ---
def _parse_r_range_mm_to_m(s: str) -> tuple[float, float]:
    """
    Parse '0.4–0.7' / '0.4-0.7' / '0.4 — 0.7' into (lo_m, hi_m) (units: meters).
    """
    s = str(s).strip()
    parts = re.split(r"\s*[-–—]\s*", s)
    if len(parts) != 2:
        raise ValueError(f"Bad r range string: {s!r}. Expected 'lo-hi' in mm.")
    lo_mm = float(parts[0]); hi_mm = float(parts[1])
    return lo_mm * 1e-3, hi_mm * 1e-3

# --- If the project doesn't already have make_annular_masks / soft_unit_step, provide fallback implementations ---
if 'make_annular_masks' not in globals():
    def _sigma_from_fwhm(dr_fwhm: float) -> float:
        # FWHM = 2*sqrt(2*ln2)*sigma
        return dr_fwhm / (2.0 * (2.0*np.log(2.0))**0.5)

    def soft_unit_step(x: torch.Tensor, fwhm: float) -> torch.Tensor:
        """
        Smooth step H(x); fwhm is the transition width (meters). Returns [0,1].
        """
        if fwhm <= 0:
            return (x >= 0).to(torch.float32)
        sigma = _sigma_from_fwhm(fwhm)
        t = x / (np.sqrt(2.0) * sigma)
        return 0.5 * (1.0 + torch.erf(t))

    def make_annular_masks(
        X: torch.Tensor, Y: torch.Tensor,
        inner_radii,                  # Sequence[float | None]
        dr_fwhm: float,
        aperture_radius: float | None = None,
        aperture_fwhm: float | None = None,
    ):
        """
        Generate the cumulative soft mask for each layer: layer j occupies r >= inner_radii[j] (j=0 may be None meaning full plate).
        """
        r = torch.sqrt(X**2 + Y**2)
        masks = []
        for r_edge in inner_radii:
            if (r_edge is None) or (r_edge < 0):
                Mj = torch.ones_like(r, dtype=torch.float32)
            else:
                Mj = soft_unit_step(r - float(r_edge), dr_fwhm).to(torch.float32)
            if aperture_radius is not None:
                if aperture_fwhm is None or aperture_fwhm <= 0:
                    M_ap = (r <= aperture_radius).to(torch.float32)
                else:
                    M_ap = soft_unit_step(aperture_radius - r, aperture_fwhm).to(torch.float32)
                Mj = Mj * M_ap
            masks.append(Mj)
        return masks

# ----------------------------------------------------------------------
# 1) Build “layer increments dH (meters) + cumulative soft masks” from CSV
# ----------------------------------------------------------------------
@torch.inference_mode()
def build_mirror_layers_from_csv(
    sim,
    csv_path="film/mirror_echelon_final_codesign.csv",
    *,
    col_range="r range (mm)",
    col_height="mirror_after_um",
    dr_fwhm=40e-6,
    aperture="3w",            # '3w' | 'from_csv' | float(m) | None
    eps_um=1e-3,              # Increments smaller than this threshold (µm) are treated as 0 (debounce)
    sort_rows=True,
    to_device=None,
):
    """
    Returns:
      dH_list: list[float], per-layer height “increment” [m] (can be positive or negative)
      masks  : list[Tensor], per-layer cumulative soft masks, float32, (Nx,Ny), device=sim.X.device
      meta   : {'inner_radii': [...], 'aperture_radius': float}
    """
    import pandas as pd
    dev = to_device or sim.X.device

    df = pd.read_csv(csv_path)

    if col_range not in df.columns:
        raise KeyError(f"Column {col_range!r} does not exist. CSV columns: {list(df.columns)}")

    # 1) Radius ranges (mm→m)
    ranges_m = [ _parse_r_range_mm_to_m(s) for s in df[col_range].astype(str).tolist() ]
    r_lo = np.array([lo for lo, _ in ranges_m], dtype=float)
    r_hi = np.array([hi for _, hi in ranges_m], dtype=float)

    # 2) Absolute heights (µm→m)
    if col_height not in df.columns:
        raise KeyError(f"Column {col_height!r} does not exist. CSV columns: {list(df.columns)}")
    h_abs = df[col_height].astype(float).to_numpy() * 1e-6

    # 3) Optional sort (ascending by inner radius)
    if sort_rows:
        order = np.argsort(r_lo)
        r_lo, r_hi, h_abs = r_lo[order], r_hi[order], h_abs[order]
    # Insert before step 4:
    h_abs = h_abs - h_abs[0]     # Use the first layer as zero baseline
    # Then convert absolute → increment
    dH = np.empty_like(h_abs)
    dH[0]  = h_abs[0]            # now equals 0
    dH[1:] = h_abs[1:] - h_abs[:-1]
    eps_m = float(eps_um) * 1e-6
    dH[np.abs(dH) < eps_m] = 0.0

    # 5) Inner radii (first layer None means the whole plate)
    inner_radii = [None] + [float(x) for x in r_lo[1:]]

    # 6) External aperture/stop
    if aperture == "3w":
        w = sim._infer_beam_radius() or 0.0
        aperture_radius = 3.0 * float(w) if w > 0 else float(r_hi[-1])
    elif aperture == "from_csv":
        aperture_radius = float(r_hi[-1])
    elif isinstance(aperture, (int, float)):
        aperture_radius = float(aperture)
    else:
        aperture_radius = None
    if aperture_radius is not None:
        r_grid = float(min(sim.x.abs().max(), sim.y.abs().max()))
        aperture_radius = min(aperture_radius, 0.98 * r_grid)

    # 7) Masks
    masks = make_annular_masks(
        sim.X.to(dev), sim.Y.to(dev),
        inner_radii=inner_radii,
        dr_fwhm=dr_fwhm,
        aperture_radius=aperture_radius,
        aperture_fwhm=dr_fwhm
    )
    masks = [m.to(device=dev, dtype=torch.float32) for m in masks]

    dH_list = [float(x) for x in dH.tolist()]
    meta = dict(inner_radii=inner_radii, aperture_radius=aperture_radius)
    return dH_list, masks, meta

# ----------------------------------------------------------------------
# 2) Apply per-layer + save to disk for each layer
# ----------------------------------------------------------------------
def _sanitize_name(s: str) -> str:
    keep = []
    for ch in str(s):
        if ch.isalnum() or ch in ('_', '-', '.'):
            keep.append(ch)
        elif ch.isspace():
            keep.append('_')
    out = ''.join(keep).strip('_')
    return out or 'layer'

@torch.inference_mode()
def apply_reflective_echelon_layers_beta01_save(
    A_in, sim, dH_list, masks, *,
    layer_dir: str | None = None,
    layer_names: list[str] | None = None,
    n0: float = 1.0,
    theta_deg: float = 0.0,
    amplitude_r: float = 1.0,
    phi_const: float = math.pi,
    omega_chunk: int = 1,
    tile_x: int | None = None,
    save_even_if_zero: bool = True,
    # —— New: final centering —— 
    center_camera_midpoint: bool = False,
    # —— New: center each saved layer (midpoint reference) ——
    center_each_save: bool = False,          # <—————— new
) -> dict:


    """
    Do a single t-FFT; for each layer multiply β0 and β1·Ω phases.
    After each layer (if layer_dir is provided): IFFT back to time domain → sim.A_out.cpu() → save to layer_dir/<index_name>/A_out.pt
    """
    dev = sim.T.device
    Nx, Ny, Nt = sim.Nx, sim.Ny, sim.nt
    tx = Nx if tile_x is None else int(tile_x)
    oc = max(1, int(omega_chunk))

    # 1) t-FFT (once)
    A = (A_in if torch.is_tensor(A_in) else torch.as_tensor(A_in)).to(
        dev, dtype=torch.complex64, non_blocking=True
    )
    _, A_w = sim.fft_t(A)       # (Nx,Ny,Nt)
    del A
    sim.A_out=None
    A_in=None
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # 2) Constants
    cos_theta = math.cos(math.radians(theta_deg))
    beta0 = float(n0 * sim.omega0 / sim.c)   # [rad/m]
    beta1 = float(n0 / sim.c)                # [s/m]
    omega = sim.omega.to(dev)                # (Nt,)
    # —— For “center-each-save”: center/outer masks and cumulative height ——
    _excl = _exclusive_from_cumulative(masks)
    M_center = _excl[0].to(device=dev, dtype=torch.float32)
    M_outer  = _excl[-1].to(device=dev, dtype=torch.float32)
    H_cum = torch.zeros_like(M_center, dtype=torch.float32, device=dev)
    
    def _dt_shift_midpoint_from_H(Hxy_accum: torch.Tensor) -> float:
        """Compute the midpoint group delay (seconds) between the center and outer regions based on current cumulative height H_cum."""
        if not center_each_save:
            return 0.0
        Lxy_cum = (2.0 * cos_theta) * Hxy_accum  # [m]
        den_c = torch.sum(M_center).item()
        den_o = torch.sum(M_outer ).item()
        Lc = 0.0 if den_c == 0.0 else (torch.sum(Lxy_cum * M_center).item() / den_c)
        Lo = 0.0 if den_o == 0.0 else (torch.sum(Lxy_cum * M_outer ).item() / den_o)
        return 0.5 * beta1 * (Lc + Lo)          # [s]

    # 3) Prepare save directory
    saved_dirs: list[str] = []
    if layer_dir is not None:
        os.makedirs(layer_dir, exist_ok=True)

    # 4) Per-layer
    K = len(dH_list)
    if layer_names is None:
        layer_names = [f"mirror_{j+1:02d}" for j in range(K)]

    for j, (dH, Mj) in enumerate(zip(dH_list, masks), start=1):
        # 4.1 L_j(x,y) = 2 cosθ · dH · M_j
        if dH == 0.0:
            if save_even_if_zero and (layer_dir is not None):
                dt_save = _dt_shift_midpoint_from_H(H_cum)
                if center_each_save and (dt_save != 0.0):
                    phase_t = torch.exp(1j * omega * float(dt_save))
                    A_w.mul_(phase_t[None, None, :])             # apply temporarily
                    _, A_t_tmp = sim.ifft_t(A_w)
                    A_w.mul_(torch.conj(phase_t)[None, None, :]) # immediately revert
                    del phase_t
                else:
                    _, A_t_tmp = sim.ifft_t(A_w)
        
                sim.A_out = A_t_tmp.detach().cpu()
                sub = os.path.join(layer_dir, f"{j:02d}_{_sanitize_name(layer_names[j-1])}")
                os.makedirs(sub, exist_ok=True)
                torch.save(sim.A_out, os.path.join(sub, "A_out.pt"))
                with open(os.path.join(sub, "meta.json"), "w", encoding="utf-8") as f:
                    json.dump({
                        "layer_index": j, "dH_m": float(dH),
                        "note": "zero increment",
                        "center_each_save": bool(center_each_save),
                        "dt_centered_s": float(dt_save)
                    }, f, indent=2)
                saved_dirs.append(sub)
                del A_t_tmp
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            continue


        Lxy = (2.0 * cos_theta * float(dH)) * Mj.to(
            device=dev, dtype=torch.float32, non_blocking=True
        )  # (Nx,Ny)

        # 4.2 β0: 2D phase broadcast
        phi0_xy   = (beta0 * Lxy)
        phase0_xy = torch.complex(torch.cos(phi0_xy), torch.sin(phi0_xy))  # complex64
        A_w.mul_(phase0_xy[..., None])
        del phi0_xy, phase0_xy
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # 4.3 β1·Ω: x-tile + small frequency batches (**note the minus sign**: positive delay)
        for xs in range(0, Nx, tx):
            xe = min(xs + tx, Nx)
            L_blk = Lxy[xs:xe, :]   # (tx,Ny)
            if oc == 1:
                for k in range(Nt):
                    theta_xy = -(beta1 * float(omega[k])) * L_blk
                    phase_xy = torch.complex(torch.cos(theta_xy), torch.sin(theta_xy))
                    A_w[xs:xe, :, k].mul_(phase_xy)
            else:
                for k0 in range(0, Nt, oc):
                    k1 = min(k0 + oc, Nt)
                    om = omega[k0:k1]  # (oc,)
                    theta = -(beta1 * L_blk[..., None]) * om[None, None, :]  # **minus sign**
                    phase = torch.complex(torch.cos(theta), torch.sin(theta))
                    A_w[xs:xe, :, k0:k1].mul_(phase)
                    del theta, phase

        del Lxy
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # —— Accumulate current layer's geometric height into H_cum (for centering reference) ——
        H_cum.add_(Mj.to(device=dev, dtype=torch.float32), alpha=float(dH))
        
        # 4.4 Save each layer (if layer_dir provided): optional centering
        if layer_dir is not None:
            dt_save = _dt_shift_midpoint_from_H(H_cum)
            if center_each_save and (dt_save != 0.0):
                phase_t = torch.exp(1j * omega * float(dt_save))
                A_w.mul_(phase_t[None, None, :])              # apply temporarily
                _, A_t_tmp = sim.ifft_t(A_w)
                A_w.mul_(torch.conj(phase_t)[None, None, :])  # immediately revert
                del phase_t
            else:
                _, A_t_tmp = sim.ifft_t(A_w)
        
            sub = os.path.join(layer_dir, f"{j:02d}_{_sanitize_name(layer_names[j-1])}")
            os.makedirs(sub, exist_ok=True)
            torch.save(A_t_tmp.detach().cpu(), os.path.join(sub, "A_out.pt"))
            with open(os.path.join(sub, "meta.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "layer_index": j,
                    "dH_m": float(dH),
                    "theta_deg": float(theta_deg),
                    "n0": float(n0),
                    "center_each_save": bool(center_each_save),
                    "dt_centered_s": float(dt_save)
                }, f, indent=2)
            saved_dirs.append(sub)
            del A_t_tmp
            if torch.cuda.is_available(): torch.cuda.empty_cache()


    # 5) After all layers: multiply reflection amplitude and constant phase once
    if not (amplitude_r == 1.0 and float(phi_const) == 0.0):
        r_phase = complex(float(amplitude_r * math.cos(phi_const)),
                          float(amplitude_r * math.sin(phi_const)))
        A_w.mul_(r_phase)

    # === New: one-shot “camera centering to midpoint” ===================================
    if center_camera_midpoint:
        dev = sim.T.device
        # (i) Build total geometric optical path L_total(x,y) = 2 cosθ * H_total(x,y)
        #     where H_total = Σ_j dH_j * M_j — same as your build_mirror_height_map
        Hxy = torch.zeros_like(masks[0], dtype=torch.float32, device=dev)
        for dH, Mj in zip(dH_list, masks):
            if dH != 0.0:
                Hxy.add_(Mj.to(device=dev, dtype=torch.float32), alpha=float(dH))
        Lxy = (2.0 * math.cos(math.radians(theta_deg))) * Hxy  # [m]

        # (ii) Mutually exclusive partitions (center / outermost)
        excl = _exclusive_from_cumulative(masks)
        M_center = excl[0].to(device=dev, dtype=torch.float32)
        M_outer  = excl[-1].to(device=dev, dtype=torch.float32)

        # (iii) Mean geometric group delay for the two regions: τ = β1 * ⟨L⟩_region
        beta1 = float(n0 / sim.c)   # [s/m]
        def _mean_over(M):
            num = torch.sum(Lxy * M).item()
            den = torch.sum(M).item()
            return 0.0 if den == 0.0 else (num / den)
        Lc = _mean_over(M_center)    # [m]
        Lo = _mean_over(M_outer)     # [m]
        tau_c = beta1 * Lc           # [s]
        tau_o = beta1 * Lo           # [s]
        dt_shift = 0.5 * (tau_c + tau_o)   # midpoint of the two regions

        # (iv) Multiply e^{+i Ω dt} in frequency domain, equivalent to A(t) ← A(t + dt) in time domain
        if dt_shift != 0.0:
            phase_t = torch.exp(1j * sim.omega.to(dev) * float(dt_shift))  # (Nt,)
            A_w.mul_(phase_t[None, None, :])

        # Record to meta (for later review)
        dt_applied = float(dt_shift)
    else:
        dt_applied = 0.0
    # ==================================================================

    # 6) Back to time domain and return; also write the final field to sim.A_out (CPU)
    _, A_final = sim.ifft_t(A_w)
    A_final_cpu = A_final.detach().cpu()
    del A_w, A_final
    sim.A_out = A_final_cpu

    
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    return {
        "A_final":  None,
        "saved_dirs": saved_dirs,
        "meta": {
            "K_layers": K,
            "applied_amplitude_r": float(amplitude_r),
            "applied_phi_const": float(phi_const),
            "theta_deg": float(theta_deg),
            "n0": float(n0)
        }
    }

# ----------------------------------------------------------------------
# 3) Convenience entry: apply sequentially from CSV and save each layer
# ----------------------------------------------------------------------
@torch.inference_mode()
def apply_reflective_echelon_from_csv_sequential_save(
    A_in, sim, csv_path="film/mirror_echelon_final_codesign.csv", *,
    layer_dir: str | None = None,
    col_range="r range (mm)",
    col_height="mirror_after_um",
    dr_fwhm=40e-6,
    aperture="3w",
    n0=1.0,
    theta_deg=0.0,
    amplitude_r=1.0,
    phi_const=math.pi,
    omega_chunk=1,
    tile_x=None,
    save_even_if_zero=True,
    # —— New —— 
    center_camera_midpoint: bool = False,
    center_each_save: bool = False,   
) -> dict:
    dH_list, masks, _ = build_mirror_layers_from_csv(
        sim, csv_path=csv_path, col_range=col_range, col_height=col_height,
        dr_fwhm=dr_fwhm, aperture=aperture, to_device=sim.T.device
    )
    return apply_reflective_echelon_layers_beta01_save(
        A_in, sim, dH_list, masks,
        layer_dir=layer_dir, layer_names=None,
        n0=n0, theta_deg=theta_deg,
        amplitude_r=amplitude_r, phi_const=phi_const,
        omega_chunk=omega_chunk, tile_x=tile_x,
        save_even_if_zero=save_even_if_zero,
        center_camera_midpoint=center_camera_midpoint,   # ← pass-through
        center_each_save=center_each_save
    )

# ======== end patch ========
