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
    apply_reflective_echelon_beta01,
    build_mirror_height_map,
    visualize_mirror,
)
from visualize_time_and_freq_by_regions import visualize_time_and_freq_by_regions
from analyze_spm_uniformity import analyze_freq_uniformity_wrms_delta
from measure_phase_flatness import measure_phase_flatness
from sim_blueprint import save_sim_blueprint
from strehl_after_gdd_noalloc import strehl_after_gdd_noalloc,strehl_ratio_fullfield
from mirror import apply_reflective_echelon_from_csv_sequential_save
@torch.inference_mode()
def compute_Ubar_wrms_delta_simple(
    sim,
    *,
    field_in:  str = "input",      # 'input' | 'current' | 'output' —— input field
    field_out: str = "output",     # 'input' | 'current' | 'output' —— output field
    weight_field: str = "output",  # Which field's I_xy is used as energy weight ('input' or 'output' are most common)
    center: str = "centroid",      # Beam center: 'centroid' (intensity centroid) or 'grid' (0,0)
    core_fraction: float = 0.01,   # Beam core: smallest disk whose cumulative energy reaches this fraction (default 1%)
    tile_x: int = 32,              # Tile size along x (reduce if OOM)
    amp_floor_rel: float = 1e-8,   # Spectral zero threshold (relative to this tile's S0.max), to avoid div-by-zero/instability
    deltas = None,                 # δ values (default generates 25 points from 0.02→0.5)
    hist_bins: int = 400,          # Number of bins for ρ histogram
    hist_rho_max: float = 0.6,     # Upper limit of ρ statistics (δ will be clipped to this limit)
    save_path: str | None = None,  # Save image and data (directory or file prefix)
    plot: bool = True,             # Whether to plot
    clear_cufft: bool = True,      # Clear cuFFT plan cache at the end (let reserved/free-in-pool drop)
):
    """
    Only compute and plot/save the first figure: \\overline{U}_σ(δ), plus CSV/NPZ data.
    Based on pixel-level local ω_RMS increment Δσ_ω for "input → output", strictly tiled, without keeping the whole 3D tensor on GPU.

    Output files:
      • Ubar_wrms_delta.png  —— single curve
      • Ubar_curve.csv       —— two columns: delta, Ubar
      • Ubar_curve.npz       —— delta, Ubar, and small metadata (meta)

    Return value (all CPU/numpy):
      dict(deltas, Ubar, core_radius_m, delta_sigma_core, center)
    """
    # —— Basic grid quantities —— #
    dev = sim.T.device   # compatible with CUDA / CPU
    Nx, Ny, Nt = int(sim.Nx), int(sim.Ny), int(sim.nt)
    dx, dy, dT = float(sim.dx), float(sim.dy), float(sim.dT)
    domega     = float(sim.domega)

    # —— Use CPU version of fields; do not move the entire block to GPU —— #
    def _get_field_cpu(tag: str) -> torch.Tensor:
        src = {"input": sim.A_in, "current": sim.A, "output": sim.A_out}.get(tag)
        if src is None:
            raise ValueError(f"field '{tag}' is empty; please ensure this field has been generated.")
        t = (src if torch.is_tensor(src) else torch.as_tensor(src))
        return t.detach().cpu().to(torch.complex64)  # (Nx,Ny,Nt) on CPU

    A_in_cpu   = _get_field_cpu(field_in)
    A_out_cpu  = _get_field_cpu(field_out)
    A_wgt_cpu  = _get_field_cpu(weight_field)

    # =========================
    # PASS-1: I_xy (weights) and beam center/core
    # =========================
    I_xy = np.zeros((Nx, Ny), dtype=np.float64)
    for xs in range(0, Nx, int(tile_x)):
        xe = min(xs + int(tile_x), Nx)
        Ab = A_wgt_cpu[xs:xe, :, :].to(dev, non_blocking=True)
        Ib = (Ab.real * Ab.real + Ab.imag * Ab.imag).to(torch.float32)
        I_blk = (torch.sum(Ib, dim=2, dtype=torch.float64) * dT).detach().cpu().numpy()
        I_xy[xs:xe, :] += I_blk
        del Ab, Ib, I_blk
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    X_cpu = sim.X.detach().cpu().numpy()
    Y_cpu = sim.Y.detach().cpu().numpy()

    # Beam center
    if center.lower() == "centroid":
        denom = float(I_xy.sum())
        if denom > 0.0:
            x0 = float((I_xy * X_cpu).sum() / denom)
            y0 = float((I_xy * Y_cpu).sum() / denom)
        else:
            x0 = y0 = 0.0
    else:
        x0 = y0 = 0.0

    # Beam core: smallest disk whose cumulative energy reaches core_fraction
    r_cpu = np.sqrt((X_cpu - x0)**2 + (Y_cpu - y0)**2).astype(np.float32)
    w_pix = (I_xy * dx * dy).astype(np.float64)
    W_tot = float(w_pix.sum())
    if W_tot <= 0.0:
        # Fallback for no energy
        if deltas is None:
            deltas = np.array([0.0], dtype=np.float64)
        else:
            deltas = np.asarray(deltas, dtype=np.float64)
        Ubar = np.zeros_like(deltas)
        _save_ubar_plot_and_data(deltas, Ubar, save_path, title=r"$\overline{U}_\sigma(\delta)$ — empty field")
        return {"deltas": deltas, "Ubar": Ubar, "core_radius_m": 0.0, "delta_sigma_core": 0.0, "center": (x0, y0)}

    sort_idx = np.argsort(r_cpu, axis=None)
    cum = np.cumsum(w_pix.flatten()[sort_idx])
    thr = core_fraction * W_tot
    k = int(np.searchsorted(cum, thr, side='left'))
    r_core = float(r_cpu.flatten()[sort_idx[min(k, sort_idx.size-1)]])
    core_mask_cpu = (r_cpu <= r_core)

    # =========================
    # Preparation: frequency vector & per-tile σ_ω helper
    # =========================
    omega  = sim.omega.to(dev, dtype=torch.float64)
    omega2 = omega * omega

    def _sigma_tile_from_cpu(A_cpu_tile: torch.Tensor) -> torch.Tensor:
        """
        Input: CPU (tx,Ny,Nt) complex field
        Output: GPU (tx,Ny) σ_ω (float64)
        Implementation: use sim.fft_t (preserve your simulator's scaling and non-shift convention)
        """
        A = A_cpu_tile.to(dev, non_blocking=True)
        _, Aw = sim.fft_t(A)   # complex
        Pw = (Aw.real * Aw.real + Aw.imag * Aw.imag).to(torch.float64)
        S0 = torch.sum(Pw, dim=2) * domega
        S1 = torch.sum(Pw * omega[None, None, :],  dim=2) * domega
        S2 = torch.sum(Pw * omega2[None, None, :], dim=2) * domega
        # Spectral energy floor (relative to S0.max of current tile)
        S0 = torch.clamp(S0, min=float(amp_floor_rel) * float(S0.max().item()) + 1e-30)
        var = (S2 / S0 - (S1 / S0) ** 2).clamp_min(0.0)
        sig = torch.sqrt(var)
        del A, Aw, Pw, S0, S1, S2
        return sig

    # =========================
    # PASS-2: compute Δσ_core only within the core
    # =========================
    core_num = 0.0
    core_den = 0.0

    for xs in range(0, Nx, int(tile_x)):
        xe = min(xs + int(tile_x), Nx)
        sig_in  = _sigma_tile_from_cpu(A_in_cpu [xs:xe, :, :])
        sig_out = _sigma_tile_from_cpu(A_out_cpu[xs:xe, :, :])
        d_sig   = sig_out - sig_in                             # GPU (tx,Ny)

        # Core tile weights/mask (move only this x-band)
        m_core = torch.from_numpy(core_mask_cpu[xs:xe, :]).to(dev)
        w_core = torch.from_numpy(w_pix        [xs:xe, :]).to(dev)

        core_num += float(torch.sum(d_sig * w_core * m_core).item())
        core_den += float(torch.sum(w_core * m_core).item())

        del sig_in, sig_out, d_sig, m_core, w_core
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    d_sigma_core = core_num / max(core_den, 1e-30)
    eps_core = 1e-30 * max(1.0, abs(d_sigma_core))

    # =========================
    # PASS-3: weighted histogram (ρ) → Ubar(δ)
    # =========================
    edges = np.linspace(0.0, float(hist_rho_max), int(hist_bins) + 1, dtype=np.float64)
    hist  = np.zeros(int(hist_bins), dtype=np.float64)

    for xs in range(0, Nx, int(tile_x)):
        xe = min(xs + int(tile_x), Nx)
        sig_in  = _sigma_tile_from_cpu(A_in_cpu [xs:xe, :, :]).detach().cpu().numpy()
        sig_out = _sigma_tile_from_cpu(A_out_cpu[xs:xe, :, :]).detach().cpu().numpy()
        d_sig   = sig_out - sig_in
        rho     = np.abs(d_sig - d_sigma_core) / (abs(d_sigma_core) + eps_core)   # (tx,Ny)
        w_tile  = w_pix[xs:xe, :].ravel()
        h, _    = np.histogram(rho.ravel(), bins=edges, weights=w_tile)
        hist   += h.astype(np.float64)

        del sig_in, sig_out, d_sig, rho
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    cdf = np.cumsum(hist) / max(hist.sum(), 1e-30)
    if deltas is None:
        deltas = np.linspace(0.02, 0.5, 25, dtype=np.float64)
    else:
        deltas = np.asarray(deltas, dtype=np.float64)
    # Ubar(δ) = CDF(ρ ≤ δ)
    Ubar = np.interp(np.clip(deltas, 0.0, float(hist_rho_max)), edges[1:], cdf)

    # —— Optional: clear cuFFT plan cache & return allocator free memory —— #
    if clear_cufft:
        try:
            pc = torch.backends.cuda.cufft_plan_cache
            # print("[cuFFT] plan cache size before:", pc.size())
            pc.clear()
            pc.max_size = 0   # set back >0 if you want to keep plans
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

    # =========================
    # Plot/save (first figure only)
    # =========================
    if plot or (save_path is not None):
        _save_ubar_plot_and_data(
            deltas, Ubar, save_path,
            title=r"$\overline{U}_\sigma(\delta)$",
            meta=dict(
                core_radius_m=r_core,
                delta_sigma_core=d_sigma_core,
                center=(float(x0), float(y0)),
                hist_edges=edges,
                hist_counts=hist,
            )
        )

    return {
        "deltas": deltas,
        "Ubar": Ubar,
        "core_radius_m": r_core,
        "delta_sigma_core": d_sigma_core,
        "center": (float(x0), float(y0)),
    }
from pathlib import Path
def _save_ubar_plot_and_data(deltas, Ubar, save_path, title="Ubar(δ)", meta=None):
    """Only draw/save the first figure and (δ, Ubar) data (CSV/NPZ)."""
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    ax.plot(deltas, Ubar,label='Echelon film')
    ax.plot([0.02,0.8],[0.02,0.8],linestyle='--',label='Uniform film')
    ax.set_xlabel("Relative tolerance δ")
    ax.set_ylabel("Energy fraction within tolerance")
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    plt.legend()
    if save_path is not None:
        p = Path(str(save_path))
        if (p.exists() and p.is_dir()) or str(save_path).endswith(("/", "\\")):
            p = p / "Ubar_wrms_delta.png"
        if p.suffix == "":
            p = p.with_suffix(".png")
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=600, bbox_inches='tight')

        # Save data in the same directory
        data_dir = p.parent
        np.savetxt(
            data_dir / "Ubar_curve.csv",
            np.column_stack([deltas, Ubar]),
            delimiter=",",
            header="delta,Ubar",
            comments=""
        )
        # NPZ contains the Ubar curve and metadata (core radius/center/histogram, etc.)
        np.savez(
            data_dir / "Ubar_curve.npz",
            delta=deltas,
            Ubar=Ubar,
            **({} if meta is None else meta)
        )

    plt.show()
