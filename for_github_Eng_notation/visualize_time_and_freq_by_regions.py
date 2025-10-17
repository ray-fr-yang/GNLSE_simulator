import torch
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import fft as tfft     # alias for torch.fft
import psutil
import numpy as np
import matplotlib.pyplot as plt
from math import factorial, sqrt, log, pi
from typing import Optional, Callable, Dict, Sequence
from scipy.interpolate import interp1d
import os
import math
from scipy import fft as sp_fft      # Change this line if switching to CuPy later
import pathlib
@torch.inference_mode()
def visualize_time_and_freq_by_regions(
    sim,
    *,
    field: str = "output",   # 'output' | 'current' | 'input'
    t_unit: str = "fs",      # 'fs' | 'ps' | 's'
    f_unit: str = "THz",     # 'THz' | 'rad/s'
    normalize: bool = False, # Normalize time/frequency domains separately to max=1
    fftshift: bool = True,   # Centered spectrum display
    show_energy_fraction: bool = True,  # Legend shows energy fraction of each region
    title: str = "Temporal & Spectral power by spatial regions (GPU, 2-field peak)",
    save_path=None,          # Image save path; None = do not save
    # ========= New parameters =========
    center_mode: str = "centroid",  # 'centroid' (default) or 'peak'
    report_centers: bool = True,    # Print pulse center of each region
    compute_dt_rms: bool = True,    # Compute and print Δt_rms
    return_values: bool = False,    # If True, return a result dict; by default keep old behavior returning 0
    tlim=None,
    wlim=None,
    legend=None,
):
    """
    Based on the original time–frequency visualization, record and print the temporal pulse center (centroid/peak)
    for each mutually exclusive spatial region (annulus), and compute energy-weighted Δt_rms.
    """
    dev = sim.T.device
    if dev.type != "cuda":
        print("[Note] Current device is not CUDA; will run on CPU (functionally correct but slower).")
    torch.cuda.empty_cache()
    
    # -------- Fetch field and move to GPU ----------
    if field == "output":
        if sim.A_out is None:
            raise ValueError("sim.A_out is None — please complete propagation first (propagate_with_film or propagate)")
        A = (sim.A_out if torch.is_tensor(sim.A_out) else torch.as_tensor(sim.A_out)).to(
            dev, dtype=torch.complex64, non_blocking=True
        )
    elif field == "input":
        A = (sim.A_in if torch.is_tensor(sim.A_in) else torch.as_tensor(sim.A_in)).to(
            dev, dtype=torch.complex64, non_blocking=True
        )
    else:  # 'current'
        if sim.A is None:
            raise ValueError("sim.A is None — 'current' field exists only during propagation; otherwise use 'output' or 'input'")
        A = (sim.A if torch.is_tensor(sim.A) else torch.as_tensor(sim.A)).to(
            dev, dtype=torch.complex64, non_blocking=True
        )

    Nx, Ny, Nt = A.shape
    dx, dy = float(sim.dx), float(sim.dy)
    dT = float(sim.dT)
    area_scale = dx * dy

    # -------- Construct mutually exclusive region masks (GPU, float32) ----------
    assert sim.stepped_film is not None and len(sim.stepped_film.layers) > 0, "stepped_film not set or empty"
    cum = [L.mask.to(device=dev, dtype=torch.float32, non_blocking=True) for L in sim.stepped_film.layers]
    if len(cum) == 1:
        excl = [cum[0]]
    else:
        excl = [torch.clamp(cum[j] - cum[j + 1], 0.0, 1.0) for j in range(len(cum) - 1)]
        excl.append(torch.clamp(cum[-1], 0.0, 1.0))
    M0 = cum[0]
    torch.cuda.empty_cache()

    # ==========================
    # 1) Time domain — peak memory: A + I
    # ==========================
    P_total = torch.zeros(Nt, device=dev, dtype=torch.float32)
    P_regions = [torch.zeros(Nt, device=dev, dtype=torch.float32) for _ in excl]

    # —— New: accumulation for intensity-weighted mean radius r̄_j (energy-weighted) ——
    r_num = [torch.zeros((), device=dev, dtype=torch.float64) for _ in excl]  # ∑ r * I_xy * M
    r_den = [torch.zeros((), device=dev, dtype=torch.float64) for _ in excl]  # ∑ I_xy * M

    tile_x = 32  # or 64/128, tune based on VRAM
    for xs in range(0, Nx, tile_x):
        xe = min(xs + tile_x, Nx)
        Ab = A[xs:xe, :, :]  # (tx, Ny, Nt), complex64
        Ib = (Ab.real * Ab.real + Ab.imag * Ab.imag).to(torch.float32)  # (tx, Ny, Nt)
        # Time projection (original logic)
        P_total += torch.einsum('xyt,xy->t', Ib, M0[xs:xe, :]) * area_scale
        for j, Mj in enumerate(excl):
            P_regions[j] += torch.einsum('xyt,xy->t', Ib, Mj[xs:xe, :]) * area_scale

        # ==== New: I_xy(x,y) = ∫ I dt, used for r̄_j accumulation ====
        I_xy_blk = torch.sum(Ib, dim=2) * dT  # (tx,Ny)
        # Radius block (do not build full r grid to avoid peak memory)
        r_blk = torch.sqrt(sim.X[xs:xe, :].to(dev)**2 + sim.Y[xs:xe, :].to(dev)**2).to(I_xy_blk.dtype)
        for j, Mj in enumerate(excl):
            w_blk = I_xy_blk * Mj[xs:xe, :]                # (tx,Ny)
            r_den[j] += torch.sum(w_blk, dtype=torch.float64)
            r_num[j] += torch.sum(w_blk * r_blk, dtype=torch.float64)

        del Ib, I_xy_blk, r_blk
    # Ib etc. already freed here; will not trigger 4GB segment
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # ====== New: per-region temporal centers and Δt_rms ======
    centers = []       # center of each region (seconds)
    energies = []      # energy of each region (joules)
    r_mean = []        # intensity-weighted mean radius of each region (meters)
    # Region energy: E_j = ∫ P_j(t) dt
    for j in range(len(excl)):
        Pj = P_regions[j]                           # (Nt,)
        Ej = torch.trapz(Pj, sim.T).item()          # J
        energies.append(Ej)

        if center_mode.lower() == "peak":
            if float(Pj.max().item()) > 0.0:
                tau_j = sim.T[torch.argmax(Pj)].item()
            else:
                tau_j = 0.0
        else:  # 'centroid'
            denom = torch.trapz(Pj, sim.T).item()
            if abs(denom) > 0.0:
                tau_j = (torch.trapz(sim.T * Pj, sim.T).item() / denom)
            else:
                tau_j = 0.0
        centers.append(tau_j)

        # Mean radius
        den = r_den[j].item()
        num = r_num[j].item()
        r_mean.append(num/den if den > 0.0 else float("nan"))

    E_total = float(sum(energies)) if len(energies) else 0.0
    tau_avg = (sum(Ej*tj for Ej, tj in zip(energies, centers)) / E_total) if E_total > 0.0 else 0.0
    if compute_dt_rms and E_total > 0.0:
        dt_rms = math.sqrt(sum(Ej*(tj - tau_avg)**2 for Ej, tj in zip(energies, centers)) / E_total)
    else:
        dt_rms = float("nan")

    # —— Print table (can be disabled) ——
    if report_centers:
        print("\n===== Region temporal centers =====")
        print(f"Mode = {center_mode}   (Units: time fs, radius mm; energy fraction %)")
        if E_total <= 0.0:
            print("Total energy is zero; centers not meaningful.")
        else:
            for j, (rj, Ej, tj) in enumerate(zip(r_mean, energies, centers)):
                frac = 100.0 * (Ej / E_total) if E_total > 0.0 else 0.0
                r_mm = rj * 1e3 if np.isfinite(rj) else float("nan")
                print(f"Region {j:2d} : r̄ = {r_mm:8.3f} mm ,  E = {frac:6.2f}% ,  t_center = {tj*1e15:10.5f} fs")
        print(f"⟨t_center⟩ (energy-weighted) = {tau_avg*1e15:.5f} fs")
        if compute_dt_rms:
            print(f"Δt_rms (energy-weighted)     = {dt_rms*1e15:.5f} fs")
        print("===================================\n")

    # —— Move to CPU immediately to keep small tensors from pinning large chunks —— #
    T = sim.T.detach().cpu().numpy()
    omega = sim.omega.detach().cpu().numpy()
    P_total_np = P_total.detach().cpu().numpy()
    P_regions_np = [p.detach().cpu().numpy() for p in P_regions]

    # For maximal cleanup, masks can be deleted now and rebuilt for frequency domain later
    del P_total, P_regions, M0, cum, excl
    torch.cuda.empty_cache()

    # ==========================
    # 2) Frequency domain (chunked streaming; no 4GB buffer)
    # ==========================
    scale_fft_t = float(sim._scale_fft_t.detach().cpu().item())
    A_w = torch.fft.fft(A, dim=-1)  # complex64
    del A
    torch.cuda.empty_cache()
    A_w.mul_(scale_fft_t)

    # Re-fetch masks (GPU float32)
    cum = [L.mask.to(device=dev, dtype=torch.float32, non_blocking=True) for L in sim.stepped_film.layers]
    if len(cum) == 1:
        excl = [cum[0]]
    else:
        excl = [torch.clamp(cum[j] - cum[j + 1], 0.0, 1.0) for j in range(len(cum) - 1)]
        excl.append(torch.clamp(cum[-1], 0.0, 1.0))
    M0 = cum[0]

    S_total = torch.zeros(Nt, device=dev, dtype=torch.float32)
    S_regions = [torch.zeros(Nt, device=dev, dtype=torch.float32) for _ in excl]

    tile_x = 32  # tunable: 32/64/128
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
    
    # —— Move to CPU ——
    S_total_np   = S_total.detach().cpu().numpy()
    S_regions_np = [s.detach().cpu().numpy() for s in S_regions]
    del S_total, S_regions, M0, cum, excl
    torch.cuda.empty_cache()

    # ===== Normalization (time/frequency separately) =====
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
        ylab_t = "Temporal power P(t)"
        ylab_w = "Spectral power S(ω)"

    # ===== Energy fraction =====
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

    # ===== X-axis unit (time) =====
    if t_unit == "fs":
        t = T * 1e15; xlab_t = "Time [fs]"
    elif t_unit == "ps":
        t = T * 1e12; xlab_t = "Time [ps]"
    else:
        t = T; xlab_t = "Time [s]"

    # ===== Region labels (no longer relying on deleted excl) =====
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

    # ---- Plotting ----
    fig, (ax_t, ax_w) = plt.subplots(1, 2, figsize=(6, 3))
    fig.suptitle(title)
    if tlim is not None:
        ax_t.set_xlim(-tlim, tlim)
    ax_t.plot(t, P_total_np, label="Total (within aperture)",linewidth=0.5)
    for Pj, lab in zip(P_regions_np, labels_t):
        ax_t.plot(t, Pj, linestyle="--", label=lab,linewidth=0.5)
        
    ax_t.set_xlabel(xlab_t); ax_t.set_ylabel(ylab_t)
    #ax_t.set_title("Temporal power P(t)")
    ax_t.grid(True)
    if legend is not None:
        ax_t.legend()
        
    if wlim is not None:
        ax_w.set_xlim(-wlim, wlim)
    ax_w.plot([0,1], [0,0],alpha=0)
    #ax_w.plot(f_plot, S_total_plot, label="Total (within aperture)")
    for Sj, lab in zip(S_regions_plot, labels_w):
        ax_w.plot(f_plot, Sj, linestyle="--", label=lab,linewidth=0.5)
    ax_w.set_xlabel(xlab_w); ax_w.set_ylabel(ylab_w)
    #ax_w.set_title("Spectral power S(ω)")
    ax_w.grid(True)      
    
    fig.tight_layout()

    # === Save (600 dpi) ===
    if save_path is not None:
        from pathlib import Path
        p = Path(str(save_path))
        if (p.exists() and p.is_dir()) or str(save_path).endswith(("/", "\\")):
            p = p / "visualize_time_and_freq_by_regions.png"
        if p.suffix == "":
            p = p.with_suffix(".png")
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=1200, bbox_inches='tight')

    plt.show()

    if return_values:
        return {
            "region_centers_s": centers,      # list[float]
            "region_r_mean_m": r_mean,        # list[float]
            "region_energies_J": energies,    # list[float]
            "tau_avg_s": tau_avg,
            "delta_t_rms_s": dt_rms,
        }
    return 0
