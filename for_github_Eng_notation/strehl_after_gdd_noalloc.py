import torch

@torch.inference_mode()
def strehl_after_gdd_noalloc(
    sim, A_out, D, *,
    t_mode="peak",              # "peak" | "centroid" | int(index)
    search_defocus=True,        # Search for best defocus term e^{-i a r^2}
    n_alpha=81,
    alpha_span_factor=12.0,
    use_gpu=True,
    aperture_mask=None          # Optional: (Nx,Ny) float mask (external aperture/stop)
):
    """
    Compute the Strehl of the “post-compression” beam, compared to generating A_comp first:
      • Do not store A_comp on CPU (CPU will not grow to 3 fields because of A_comp)
      • GPU peak memory ~ two full complex fields + a small 2D buffer
    Returns: {"S": float, "tidx": int, "alpha": float}
    """
    dev = sim.T.device if (use_gpu and torch.cuda.is_available()) else torch.device("cpu")
    # 1) Move to GPU (no clone, do not modify the original CPU tensor)
    A = (A_out if torch.is_tensor(A_out) else torch.as_tensor(A_out)).to(
        dev, dtype=torch.complex64, non_blocking=True
    )
    Nx, Ny, Nt = A.shape
    dx, dy = float(sim.dx), float(sim.dy)

    # 2) Frequency domain: one FFT (peak memory ~ A + Â)
    _, Aw = sim.fft_t(A) 
    del A

    # 3) Apply GDD phase (frequency vector uses sim.omega)
    w = sim.omega.to(dev, dtype=torch.float32)
    phase = torch.exp(0.5j * float(D) * (w ** 2)).to(Aw.dtype)
    Aw.mul_(phase)


    # 4) Inverse back to time domain (peak memory ~ Â + A^t)
    _, At = sim.ifft_t(Aw)  
    del Aw

    # 5) Choose time slice
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

    # Keep only this frame (2D), free the 3D field immediately to avoid extra copy
    Axy = At[..., tidx].clone()   # Clone one frame (Nx,Ny) to safely free At
    del At, P_t

    # Optional: external aperture
    if aperture_mask is not None:
        M = (aperture_mask if torch.is_tensor(aperture_mask) else torch.as_tensor(aperture_mask)).to(
            Axy.device, dtype=torch.float32, non_blocking=True
        )
        Axy = Axy * M  # complex × real; remains complex64

    # 6) Strehl (with defocus search)
    U_ideal = torch.sum(torch.abs(Axy)) * dx * dy  # Normalization factor for ideal phase
    if U_ideal.abs().item() == 0.0:
        return {"S": 0.0, "tidx": tidx, "alpha": 0.0}

    if not search_defocus:
        U = torch.sum(Axy) * dx * dy
        S = (torch.abs(U) ** 2 / (U_ideal ** 2)).item()
        return {"S": S, "tidx": tidx, "alpha": 0.0}

    # Build only a 2D r^2; avoid Nx*Ny*n_alpha 3D memory blowup
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
    t_mode: str | int = "peak",# 'peak' | 'centroid' | specified time index
    search_defocus: bool = True,
    n_alpha: int = 81,
    alpha_span_factor: float = 6.0,
    use_gpu: bool = True,
    tile_x: int = 64,          # Block width (along x), controls VRAM peak
):
    """
    Compute Strehl over the full aperture directly (no need for M0). Memory-friendly:
      - CPU: no full-field copies; keep A_field on CPU.
      - GPU: stream in tiles (tile_x, Ny); keep only one slice and temporary phase; peak << two full fields.

    Returns: dict = {'S': float, 'tidx': int, 'alpha': float}
    """
    # -------- Fetch field --------
    if field == "output":
        A_field = sim.A_out
    elif field == "input":
        A_field = sim.A_in
    else:
        A_field = sim.A
    if A_field is None:
        raise ValueError(f"sim.{field} is empty")

    A = A_field if torch.is_tensor(A_field) else torch.as_tensor(A_field)
    assert A.ndim == 3, "A must be (Nx, Ny, Nt)"
    Nx, Ny, Nt = A.shape
    dx, dy = float(sim.dx), float(sim.dy)

    # Choose device
    dev = (sim.T.device if (use_gpu and torch.cuda.is_available()) else torch.device("cpu"))

    # -------- Step 1: find time index tidx (streaming; no full-field temp) --------
    P_t = torch.zeros(Nt, dtype=torch.float32, device=dev)  # only 1×Nt
    for xs in range(0, Nx, tile_x):
        xe = min(xs + tile_x, Nx)
        # Move one x-block to the target device; unify dtype to complex64
        A_blk = A[xs:xe, :, :].to(dev, dtype=torch.complex64, non_blocking=True)
        # Accumulate |A|^2 into temporal power: sum_{x,y} |A|^2
        # Do not keep the full I field; sum per block then add to P_t
        # pow_blk: (tx, Ny, Nt) -> sum over (0,1) -> (Nt,)
        pow_blk = (A_blk.real*A_blk.real + A_blk.imag*A_blk.imag).to(torch.float32)
        P_t += torch.sum(pow_blk, dim=(0,1)) * (dx * dy)
        del A_blk, pow_blk
    # Choose time index
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

    # -------- Step 2: compute U_ideal = ∫|A(x,y,t*)| dxdy (streaming) --------
    U_ideal = 0.0
    for xs in range(0, Nx, tile_x):
        xe = min(xs + tile_x, Nx)
        Axy_blk = A[xs:xe, :, tidx].to(dev, dtype=torch.complex64, non_blocking=True)
        # Per-block sum of |A|
        abs_blk = torch.sqrt(Axy_blk.real*Axy_blk.real + Axy_blk.imag*Axy_blk.imag)
        U_ideal += float(torch.sum(abs_blk).item())
        del Axy_blk, abs_blk
    U_ideal *= (dx * dy)
    if U_ideal == 0.0:
        return {"S": 0.0, "tidx": tidx, "alpha": 0.0}

    # If not searching defocus, compute one coherent sum directly
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

    # -------- Step 3: defocus search  U(a) = ∫ A(x,y) e^{-i a r^2} dxdy  --------
    # Estimate max r^2 (no extra field; use max of sim.X/Y directly)
    try:
        x_max = float(torch.max(torch.abs(sim.X)).detach().cpu().item())
        y_max = float(torch.max(torch.abs(sim.Y)).detach().cpu().item())
        r2max = x_max*x_max + y_max*y_max
    except Exception:
        # Fallback: estimate from grid size
        r2max = ((Nx-1)*dx/2)**2 + ((Ny-1)*dy/2)**2
    alpha_max = alpha_span_factor / (r2max + 1e-30)
    alphas = torch.linspace(-alpha_max, alpha_max, n_alpha, device=dev, dtype=torch.float32)

    best_absU = 0.0
    best_alpha = 0.0

    # To reduce repeated transfers, precompute geometric r^2 within the “block scope” and reuse across alpha loop
    for a in alphas:
        U_r = 0.0
        U_i = 0.0
        for xs in range(0, Nx, tile_x):
            xe = min(xs + tile_x, Nx)
            # Field block
            Axy_blk = A[xs:xe, :, tidx].to(dev, dtype=torch.complex64, non_blocking=True)
            # r^2 block (kept only within block scope)
            try:
                X_blk = sim.X[xs:xe, :].to(dev, dtype=torch.float32, non_blocking=True)
                Y_blk = sim.Y[xs:xe, :].to(dev, dtype=torch.float32, non_blocking=True)
                r2_blk = X_blk*X_blk + Y_blk*Y_blk
                del X_blk, Y_blk
            except Exception:
                # Fallback: if sim.X/Y are missing, construct approximately from indices (rarely used)
                ix = torch.arange(xs, xe, device=dev, dtype=torch.float32) - (Nx-1)/2
                iy = torch.arange(0, Ny, device=dev, dtype=torch.float32) - (Ny-1)/2
                Xb = ix[:, None] * dx
                Yb = iy[None, :] * dy
                r2_blk = Xb*Xb + Yb*Yb

            phi = a * r2_blk                           # (tx,Ny), float32
            c = torch.cos(phi); s = torch.sin(phi)     # two block-level temporaries
            Ar = Axy_blk.real; Ai = Axy_blk.imag       # views, not copies

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

    # Clear GPU free blocks (does not affect tensors currently used by PyTorch)
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
    Analyze phase flatness at time slice t* (within 3w aperture only):
      - Fit and remove piston, tip-tilt, defocus (and optional astigmatism)
      - Report residual phase RMS, Marechal-estimated S, and true S before/after removal
    Returns: dict (with coefficients, RMS, S, etc.), and can plot
    """
    # 1) Fetch field and time slice
    if field == "output":   A = sim.A_out
    elif field == "input":  A = sim.A_in
    else:                   A = sim.A
    if A is None: raise ValueError(f"sim.{field} is empty")
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

    # 2) 3w circular aperture
    R = R_factor * float(w)           # w has been defined by you in advance
    mask = (X*X + Y*Y) <= (R*R)

    amp   = np.abs(Axy)
    phase = np.angle(Axy)             # [-pi, pi]
    # Simple 2D phase unwrapping (can fail under high noise/strong envelope distortion; we use energy weights to stabilize the fit)
    phi_unw = np.unwrap(np.unwrap(phase, axis=0), axis=1)

    # 3) Weighted least squares fit of low-order aberrations: 1, x, y, (x^2+y^2), [x^2-y^2, 2xy]
    x = X[mask]; y = Y[mask]
    yvec = phi_unw[mask]
    wA   = (amp[mask]**2)             # use intensity as weight
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
    # Fill back to full field, compute residual
    phi_fit_full = np.zeros_like(phi_unw); phi_fit_full[mask] = phi_fit
    phi_res = np.zeros_like(phi_unw);      phi_res[mask] = yvec - phi_fit

    # Remove weighted mean of residual (leave only higher-order ripples)
    mean_res = np.average(phi_res[mask], weights=wA)
    phi_res[mask] -= mean_res

    # 4) Metrics: residual RMS, Marechal S, true Strehl before/after removal
    sigma_phi = math.sqrt(np.average((phi_res[mask])**2, weights=wA))   # rad
    S_marechal = math.exp(-sigma_phi**2)

    # True Strehl (coherent sum within aperture only; dxdy cancels in the ratio, no need to multiply)
    U_ideal = np.sum(amp[mask])
    U_raw   = np.abs(np.sum(Axy[mask]))
    S_raw   = (U_raw / (U_ideal + 1e-30))**2

    A_corr = Axy * np.exp(-1j * phi_fit_full)  # field after removing low-order aberrations
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
