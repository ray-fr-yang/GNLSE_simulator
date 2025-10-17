import math, numpy as np, torch
import matplotlib.pyplot as plt
from pathlib import Path
import math, numpy as np, torch
import matplotlib.pyplot as plt
from pathlib import Path
import math, numpy as np, torch
import matplotlib.pyplot as plt
from pathlib import Path

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
    amp_floor_rel: float = 1e-8,   # Spectrum zero threshold (relative to S0.max of this tile), to avoid div-by-zero/instability
    deltas = None,                 # δ values (default generates 25 points from 0.02→0.5)
    hist_bins: int = 400,          # Number of bins for ρ histogram
    hist_rho_max: float = 0.6,     # Upper limit of ρ statistics (δ will be clipped to this limit)
    save_path: str | None = None,  # Save image and data (directory or file prefix)
    plot: bool = True,             # Whether to plot
    clear_cufft: bool = True,      # Clear cuFFT plan cache at the end (so reserved/free-in-pool drops)
):
    """
    Only compute and plot/save the first figure: \overline{U}_σ(δ), plus CSV/NPZ data.
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
            raise ValueError(f"field '{tag}' is empty; please ensure the field has been generated.")
        t = (src if torch.is_tensor(src) else torch.as_tensor(src))
        return t.detach().cpu().to(torch.complex64)  # (Nx,Ny,Nt) on CPU

    A_in_cpu   = _get_field_cpu(field_in)
    A_out_cpu  = _get_field_cpu(field_out)
    A_wgt_cpu  = _get_field_cpu(weight_field)

    # =========================
    # PASS-1: I_xy (weight) and beam center/core
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
    # Preparation: frequency vector & tile-wise σ_ω function
    # =========================
    omega  = sim.omega.to(dev, dtype=torch.float64)
    omega2 = omega * omega

    def _sigma_tile_from_cpu(A_cpu_tile: torch.Tensor) -> torch.Tensor:
        """
        Input: CPU (tx,Ny,Nt) complex field
        Output: GPU (tx,Ny) σ_ω (float64)
        Implementation: use sim.fft_t (keeping your simulator's scaling and non-shift convention)
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

        # Core tile weights/masks (move only this x-band)
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
    # Plot/save (only the first figure)
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


def _save_ubar_plot_and_data(deltas, Ubar, save_path, title="Ubar(δ)", meta=None):
    """Only draw/save the first figure and (δ, Ubar) data (CSV/NPZ)."""
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.0))
    ax.plot(deltas, Ubar, marker='o')
    ax.set_xlabel("Relative tolerance δ")
    ax.set_ylabel("Energy fraction within tolerance")
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()

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


@torch.inference_mode()
def analyze_freq_uniformity_wrms_delta(
    sim,
    *,
    field_in:  str = "input",     # 'input' | 'current' | 'output'
    field_out: str = "output",
    weight_field: str = "output", # Which field's I_xy is used as energy weight (default: output)
    center: str = "centroid",     # 'centroid' | 'grid' (definition of beam center)
    nbins_r: int | None = None,   # Radial bins; None→auto
    quantiles = (0.5, 0.8, 0.9),  # Power quantiles (cumulative power ratio up to that radius)
    core_fraction: float = 0.01,  # Beam-core cumulative power fraction (for reference Δσ_core)
    tile_x: int = 32,             # x-direction tiling, memory friendly; reduce to 16/8 if VRAM tight
    amp_floor_rel: float = 1e-8,  # Denominator floor for spectral second moment (avoid numerical issues at very low-energy pixels)
    # —— Ū(δ) curve (histogram method) ——
    deltas = None,
    hist_bins: int = 400,
    hist_rho_max: float = 0.6,
    # —— Display/save ——
    show_delta_mu: bool = False,  # Additionally plot Δμ ring profile/heatmap (replaces the two Δσ subplots)
    to_thz: bool = True,          # Show frequency-related quantities in THz on plots
    save_path: str | None = None,
    title: str = "Uniformity from regional ω_RMS delta (input → output)",
    return_values: bool = True
):
    """
    Using local spectral second-moment difference (ω_RMS increment) between input and output to assess spatial uniformity, with plotting/saving.
    """
    dev = sim.T.device
    Nx, Ny, Nt = int(sim.Nx), int(sim.Ny), int(sim.nt)
    dx, dy, dT = float(sim.dx), float(sim.dy), float(sim.dT)
    domega     = float(sim.domega)
    TWO_PI     = 2.0 * math.pi

    # ---------- 0) Fetch three fields ----------
    def _get_field(name: str):
        src = {"input": sim.A_in, "current": sim.A, "output": sim.A_out}.get(name)
        if src is None:
            raise ValueError(f"field '{name}' does not exist; please ensure it's generated in sim.")
        return (src if torch.is_tensor(src) else torch.as_tensor(src)).to(
            dev, dtype=torch.complex64, non_blocking=True
        )

    A_in   = _get_field(field_in)
    A_out  = _get_field(field_out)
    A_wgt  = _get_field(weight_field)

    # ---------- 1) Compute I_xy (energy weight) and determine beam center ----------
    I_xy = torch.zeros((Nx, Ny), device=dev, dtype=torch.float64)
    for xs in range(0, Nx, int(tile_x)):
        xe = min(xs + int(tile_x), Nx)
        Ab = A_wgt[xs:xe, :, :]
        Ib = (Ab.real*Ab.real + Ab.imag*Ab.imag).to(torch.float32)
        I_xy[xs:xe, :] += torch.sum(Ib, dim=2, dtype=torch.float64) * dT
        del Ab, Ib
    I_xy_cpu = I_xy.detach().cpu().numpy()

    if center.lower() == "centroid":
        denom = I_xy.sum().clamp_min(1e-30)
        x0 = float((I_xy * sim.X.to(torch.float64)).sum() / denom)
        y0 = float((I_xy * sim.Y.to(torch.float64)).sum() / denom)
    else:
        x0, y0 = 0.0, 0.0

    # Radius map (CPU)
    X_cpu = sim.X.detach().cpu().numpy()
    Y_cpu = sim.Y.detach().cpu().numpy()
    r_cpu = np.sqrt((X_cpu - x0)**2 + (Y_cpu - y0)**2).astype(np.float32)

    # ---------- 2) Radial binning & core radius ----------
    if nbins_r is None:
        nbins_r = max(8, min(Nx, Ny)//2)
    r_max = float(r_cpu.max())
    dr = r_max / nbins_r
    bin_idx = np.clip((r_cpu / dr).astype(np.int32), 0, nbins_r-1)
    r_bin_center = (np.arange(nbins_r) + 0.5) * dr

    # Core radius (where cumulative power reaches core_fraction)
    w_pix = I_xy_cpu * (dx * dy)                 # weight = energy density × pixel area
    sort_r = np.argsort(r_cpu, axis=None)
    cum = np.cumsum(w_pix.flatten()[sort_r])
    W_tot = float(w_pix.sum())
    thr = core_fraction * W_tot
    k = int(np.searchsorted(cum, thr, side='left'))
    r_core = float(r_cpu.flatten()[sort_r[min(k, sort_r.size-1)]])

    core_mask_cpu = (r_cpu <= r_core)

    # ---------- 3) Prepare output 2D arrays (CPU) ----------
    mu_in_xy   = np.zeros((Nx, Ny), dtype=np.float32)
    mu_out_xy  = np.zeros((Nx, Ny), dtype=np.float32)
    sig_in_xy  = np.zeros((Nx, Ny), dtype=np.float32)
    sig_out_xy = np.zeros((Nx, Ny), dtype=np.float32)

    # ---------- 4) tile-x streaming: compute local spectral moments for input/output in blocks ----------
    omega = sim.omega.to(dev, dtype=torch.float64)     # (Nt,)
    omega2 = omega*omega

    # First sweep input
    for xs in range(0, Nx, int(tile_x)):
        xe = min(xs + int(tile_x), Nx)

        # Input field
        Ab = A_in[xs:xe, :, :]                         # (tx,Ny,Nt)
        _, Aw = sim.fft_t(Ab)                          # consistent with sim scaling
        Pw = (Aw.real*Aw.real + Aw.imag*Aw.imag).to(torch.float64)

        S0 = torch.sum(Pw, dim=2) * domega             # (tx,Ny)
        S1 = torch.sum(Pw * omega[None,None,:], dim=2) * domega
        S2 = torch.sum(Pw * omega2[None,None,:], dim=2) * domega

        S0 = torch.clamp(S0, min=float(amp_floor_rel)*float(S0.max().item()) + 1e-30)

        mu  = (S1 / S0).detach().cpu().numpy()
        var = (S2 / S0 - (S1/S0)**2).clamp_min(0).detach().cpu().numpy()
        sig = np.sqrt(var, dtype=np.float64)

        mu_in_xy[xs:xe, :]  = mu.astype(np.float32)
        sig_in_xy[xs:xe, :] = sig.astype(np.float32)

        del Ab, Aw, Pw, S0, S1, S2

    # Then sweep output (optionally reconfirm I_xy_out — already have I_xy above)
    for xs in range(0, Nx, int(tile_x)):
        xe = min(xs + int(tile_x), Nx)

        Ab = A_out[xs:xe, :, :]
        _, Aw = sim.fft_t(Ab)
        Pw = (Aw.real*Aw.real + Aw.imag*Aw.imag).to(torch.float64)

        S0 = torch.sum(Pw, dim=2) * domega
        S1 = torch.sum(Pw * omega[None,None,:], dim=2) * domega
        S2 = torch.sum(Pw * omega2[None,None,:], dim=2) * domega

        S0 = torch.clamp(S0, min=float(amp_floor_rel)*float(S0.max().item()) + 1e-30)

        mu  = (S1 / S0).detach().cpu().numpy()
        var = (S2 / S0 - (S1/S0)**2).clamp_min(0).detach().cpu().numpy()
        sig = np.sqrt(var, dtype=np.float64)

        mu_out_xy[xs:xe, :]  = mu.astype(np.float32)
        sig_out_xy[xs:xe, :] = sig.astype(np.float32)

        del Ab, Aw, Pw, S0, S1, S2

    # ---------- 5) Compute Δσ/Δμ (CPU) ----------
    d_sigma = (sig_out_xy - sig_in_xy).astype(np.float64)  # rad/s
    d_mu    = (mu_out_xy  - mu_in_xy ).astype(np.float64)  # rad/s

    # Reference values (core-weighted average)
    w_core = w_pix[core_mask_cpu]
    d_sigma_core = float((d_sigma[core_mask_cpu]*w_core).sum() / max(w_core.sum(), 1e-30))
    d_mu_core    = float((d_mu[core_mask_cpu]*w_core).sum()    / max(w_core.sum(),  1e-30))

    # ---------- 6) Global weighted statistic: D_sigma ----------
    W = w_pix                                # energy weight
    mean_dsig = float((d_sigma*W).sum() / max(W.sum(), 1e-30))
    var_dsig  = float((((d_sigma-mean_dsig)**2)*W).sum() / max(W.sum(), 1e-30))
    rms_dsig  = math.sqrt(max(var_dsig, 0.0))
    D_sigma   = rms_dsig / (abs(mean_dsig) + 1e-30)

    # ---------- 7) Δσ/Δμ profile on power-quantile rings & Q_p ----------
    # Aggregate by bin
    Wb = np.bincount(bin_idx.ravel(), weights=W.ravel(), minlength=nbins_r)
    dS_b = np.bincount(bin_idx.ravel(), weights=(d_sigma*W).ravel(), minlength=nbins_r)
    dM_b = np.bincount(bin_idx.ravel(), weights=(d_mu*W).ravel(),    minlength=nbins_r)
    with np.errstate(divide='ignore', invalid='ignore'):
        dS_ring = np.where(Wb>0, dS_b/Wb, np.nan)   # ring-weighted mean Δσ
        dM_ring = np.where(Wb>0, dM_b/Wb, np.nan)   # ring-weighted mean Δμ

    # Cumulative power → find the ring index of each p
    cumW = np.cumsum(Wb)
    Q_sigma = {}
    Q_mu    = {}
    r_p     = {}
    for p in quantiles:
        thrW = float(p) * W_tot
        j = int(np.searchsorted(cumW, thrW, side='left'))
        j = max(0, min(nbins_r-1, j))
        r_p[p] = r_bin_center[j]
        Q_sigma[p] = (dS_ring[j] / d_sigma_core) if abs(d_sigma_core) > 0 else np.nan
        Q_mu[p]    = (dM_ring[j] / d_mu_core)    if abs(d_mu_core)    > 0 else np.nan

    # ---------- 8) Ū(δ) (CDF from histogram in one pass) ----------
    if deltas is None:
        deltas = np.linspace(0.02, 0.5, 25, dtype=np.float64)
    deltas = np.asarray(deltas, dtype=np.float64)

    eps_core = 1e-30 * max(1.0, abs(d_sigma_core))
    rho = np.abs(d_sigma - d_sigma_core) / (abs(d_sigma_core) + eps_core)   # ρ(x,y)
    # Only count valid weights (avoid extreme-edge noise)
    w_flat = W.ravel()
    r_flat = rho.ravel()
    hist, edges = np.histogram(r_flat, bins=int(hist_bins), range=(0.0, float(hist_rho_max)),
                               weights=w_flat)
    cdf = np.cumsum(hist) / max(hist.sum(), 1e-30)
    # Interpolate Ubar(δ)
    Ubar = np.interp(np.clip(deltas, 0, float(hist_rho_max)), edges[1:], cdf)

    # ---------- 9) Plot ----------
    # Unit conversion
    def _to_thz(x):
        return x/(TWO_PI*1e12)

    if to_thz:
        d_sigma_plot = _to_thz(d_sigma)
        d_mu_plot    = _to_thz(d_mu)
        d_sigma_core_plot = _to_thz(d_sigma_core)
        d_mu_core_plot    = _to_thz(d_mu_core)
        ylab_map   = "Δσ (THz)" if not show_delta_mu else "Δμ (THz)"
        prof_name  = "Δσ (THz)" if not show_delta_mu else "Δμ (THz)"
    else:
        d_sigma_plot = d_sigma
        d_mu_plot    = d_mu
        d_sigma_core_plot = d_sigma_core
        d_mu_core_plot    = d_mu_core
        ylab_map   = r"Δσ (rad/s)" if not show_delta_mu else r"Δμ (rad/s)"
        prof_name  = r"Δσ (rad/s)" if not show_delta_mu else r"Δμ (rad/s)"

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(title)

    # (1) Ū(δ) curve
    ax = axes[0,0]
    ax.plot(deltas, Ubar, marker='o', label=r"$\overline{U}_\sigma(\delta)$")
    ax.set_xlabel("Relative tolerance δ")
    ax.set_ylabel("Energy fraction within tolerance")
    ax.set_title("Energy kept vs tolerance (global)")
    ax.grid(True); ax.legend()
    # Annotate global D_sigma in the corner
    ax.text(0.98, 0.04, f"Dσ = {D_sigma:.3f}", ha="right", va="bottom", transform=ax.transAxes)

    # (2) Quantile ring profile (Δσ or Δμ)
    ax = axes[0,1]
    x_pct = 100.0 * np.asarray(quantiles, dtype=float)
    if not show_delta_mu:
        vals = np.asarray([dS_ring[max(0, min(int(np.searchsorted(cumW, p*W_tot, 'left')), nbins_r-1))] for p in quantiles], dtype=float)
        ax.plot(x_pct, _to_thz(vals) if to_thz else vals, marker='o', label=prof_name)
        ax.axhline(d_sigma_core_plot if to_thz else d_sigma_core, ls="--", color="gray", label="core")
    else:
        vals = np.asarray([dM_ring[max(0, min(int(np.searchsorted(cumW, p*W_tot, 'left')), nbins_r-1))] for p in quantiles], dtype=float)
        ax.plot(x_pct, _to_thz(vals) if to_thz else vals, marker='o', label=prof_name)
        ax.axhline(d_mu_core_plot if to_thz else d_mu_core, ls="--", color="gray", label="core")
    ax.set_xlabel("Cumulative power up to radius [%]")
    ax.set_ylabel(prof_name); ax.grid(True); ax.legend()
    ax.set_title("Quantile ring profile")

    # (3) Δσ or Δμ heatmap (display within ±some percentile range)
    ax = axes[1,0]
    if not show_delta_mu:
        z = d_sigma_plot
        c0 = np.nanpercentile(z, 1); c1 = np.nanpercentile(z, 99)
    else:
        z = d_mu_plot
        c0 = np.nanpercentile(z, 1); c1 = np.nanpercentile(z, 99)
    im = ax.imshow(z, origin="lower",
                   extent=[X_cpu.min(), X_cpu.max(), Y_cpu.min(), Y_cpu.max()],
                   aspect="equal", vmin=c0, vmax=c1)
    fig.colorbar(im, ax=ax, label=ylab_map)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("Map")
    # draw core radius
    circ = plt.Circle((x0, y0), r_core, fill=False, color='w', lw=1.0)
    ax.add_patch(circ)

    # (4) I_xy (output field) heatmap as weight reference
    ax = axes[1,1]
    im2 = ax.imshow(I_xy_cpu, origin="lower",
                    extent=[X_cpu.min(), X_cpu.max(), Y_cpu.min(), Y_cpu.max()],
                    aspect="equal")
    fig.colorbar(im2, ax=ax, label=r"$\int |A_\text{out}|^2 dt$")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("Time-integrated intensity (weight)")

    fig.tight_layout()
    if save_path is not None:
        p = Path(str(save_path))
        if (p.exists() and p.is_dir()) or str(save_path).endswith(("/", "\\")):
            p = p / ("wrms_uniformity_dsigma.png" if not show_delta_mu else "wrms_uniformity_dmu.png")
        if p.suffix == "":
            p = p.with_suffix(".png")
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=600, bbox_inches='tight')
    plt.show()

    if not return_values:
        return 0

    # ---------- 10) Return values ----------
    result = dict(
        # 2D map (rad/s)
        delta_sigma=d_sigma, delta_mu=d_mu,
        sigma_in=sig_in_xy.astype(np.float64),
        sigma_out=sig_out_xy.astype(np.float64),
        mu_in=mu_in_xy.astype(np.float64),
        mu_out=mu_out_xy.astype(np.float64),
        # Reference and global metrics
        center_xy=(x0, y0), core_radius_m=r_core,
        delta_sigma_core=d_sigma_core, delta_mu_core=d_mu_core,
        D_sigma=D_sigma, mean_delta_sigma=mean_dsig, rms_delta_sigma=rms_dsig,
        # Rings & quantiles
        nbins_r=nbins_r, r_bin_center=r_bin_center, bin_idx=bin_idx,
        quantiles=tuple(float(p) for p in quantiles),
        Q_sigma={float(k): float(v) for k, v in Q_sigma.items()},
        Q_mu={float(k): float(v) for k, v in Q_mu.items()},
        ring_profile_sigma=dS_ring, ring_profile_mu=dM_ring,
        # Tolerance curve
        deltas=deltas, Ubar=Ubar,
        # Weights and intensity
        I_xy=I_xy_cpu, weight_total=W_tot,
    )
    return result
