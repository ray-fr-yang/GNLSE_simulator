import torch, math
from typing import Literal, Optional, Tuple

@torch.inference_mode()
def measure_phase_flatness(
    sim,
    *,
    field: Literal["current","input","output"] = "current",
    # Aperture: choose one of three (priority: high → low)
    aperture_mask: Optional[torch.Tensor] = None,   # (Nx,Ny), float32 in [0,1]
    aperture_radius: Optional[float] = None,        # Circular aperture radius [m]; if provided, use soft edge fwhm
    aperture_fwhm: float = 0.0,                     # Soft-edge FWHM [m]; =0 for hard edge
    use_center_region_if_available: bool = True,    # If none of the above is provided and stepped_film exists, use the center-region mask

    # Time-slice selection
    t_mode: Literal["peak","centroid","index"] = "peak",
    t_index: Optional[int] = None,                  # Valid when t_mode='index'

    # Remove tilt/defocus
    remove_tilt: bool = True,
    search_defocus: bool = False,
    n_alpha: int = 101,
    alpha_span_factor: float = 6.0,                 # α scan range ~ alpha_span_factor / w^2
                                                    # (units: rad/m^2; order-of-magnitude is fine)
    # Result options
    return_phase_map: bool = False,                 # Return phase after removing tilt/defocus and piston (wrapped in [-π,π))
):
    """
    Evaluate phase flatness of arbitrary A(x,y,t) within a given aperture:
      - Select time t* (default: peak power within the aperture)
      - Remove tilt (find peak via 2D FFT) and optional defocus (scan α)
      - Compute coherence ratio R and σ_φ (radian RMS) inferred from R
    Returns a dict with keys:
      tidx, R_piston, R_corr, sigma_piston_rad, sigma_corr_rad, waves_rms,
      strehl_est, kx_tilt, ky_tilt, alpha_opt, (phase_map if requested)
    """
    dev = sim.T.device
    dxdy = float(sim.dx * sim.dy)

    # ===== 1) Fetch field =====
    if field == "current":
        assert sim.A is not None, "measure_phase_flatness: sim.A is None (the 'current' field exists only during propagation)"
        A = sim.A
    elif field == "input":
        A = sim.A_in
    else:
        assert sim.A_out is not None, "measure_phase_flatness: sim.A_out is None (please obtain the output field first)"
        A = sim.A_out
    A = (A if torch.is_tensor(A) else torch.as_tensor(A)).to(dev, dtype=torch.complex64, non_blocking=True)
    Nx, Ny, Nt = A.shape

    # ===== 2) Aperture mask M(x,y) =====
    if aperture_mask is not None:
        M = aperture_mask.to(dev, dtype=torch.float32, non_blocking=True)
        assert M.shape == (Nx,Ny)
    elif aperture_radius is not None:
        # Circular aperture (reuse your existing soft_unit_step)
        r = torch.sqrt(sim.X.to(dev)**2 + sim.Y.to(dev)**2)
        if aperture_fwhm > 0.0:
            from math import sqrt, log
            # Prefer your project's soft_unit_step for consistency; here is a fallback implementation
            sigma = aperture_fwhm / (2.0 * (2.0 * math.log(2.0))**0.5)
            t = (aperture_radius - r) / (math.sqrt(2.0) * sigma)
            M = 0.5 * (1.0 + torch.erf(t))
        else:
            M = (r <= aperture_radius).to(torch.float32)
    elif use_center_region_if_available and (getattr(sim, "stepped_film", None) is not None) and len(sim.stepped_film.layers) > 0:
        # Use the stepped_film's "center exclusive region" as the aperture
        cum = [L.mask.to(dev, dtype=torch.float32, non_blocking=True) for L in sim.stepped_film.layers]
        if len(cum) == 1:
            M = torch.clamp(cum[0], 0.0, 1.0)
        else:
            M = torch.clamp(cum[0] - cum[1], 0.0, 1.0)  # Region 0
    else:
        # Fallback: use a circular aperture of radius 3w
        w = sim._infer_beam_radius() or float(min(sim.x.abs().max(), sim.y.abs().max())*0.4)
        r = torch.sqrt(sim.X.to(dev)**2 + sim.Y.to(dev)**2)
        M = (r <= (3.0 * w)).to(torch.float32)

    # If the aperture is empty, return immediately
    if float(torch.sum(M).item()) == 0.0:
        return dict(error="empty_aperture")

    # ===== 3) Select time slice t* =====
    I = A.real*A.real + A.imag*A.imag  # |A|^2
    P_t_ap = torch.einsum('xyt,xy->t', I, M) * dxdy  # Power within aperture vs t
    if t_mode == "index":
        tidx = int(t_index if t_index is not None else 0)
    elif t_mode == "centroid":
        T = sim.T
        denom = torch.trapz(P_t_ap, T)
        t_cm = (torch.trapz(T * P_t_ap, T) / (denom + 1e-30)).item()
        dT = float(T[1]-T[0]); tidx = int(round((t_cm - float(T[0]))/dT))
        tidx = max(0, min(Nt-1, tidx))
    else:  # "peak"
        tidx = int(torch.argmax(P_t_ap).item())

    U = A[..., tidx]                # (Nx,Ny) complex64
    U = U * M                       # Apply aperture (zero outside)

    # ===== 4) Piston-only coherence (no unwrapping) =====
    # E_max = ∫ M |U| dxdy; E0 = ∫ M U dxdy; R = |E0|/E_max ∈ [0,1]
    E0 = torch.sum(U).abs() * dxdy
    Emax = torch.sum(torch.abs(U)) * dxdy + 1e-30
    R_piston = float((E0 / Emax).item())
    # Using circular statistics to convert R to phase RMS approximation (small phase): sigma = sqrt(-2 ln R)
    sigma_piston = float(math.sqrt(max(0.0, -2.0*math.log(max(R_piston, 1e-12)))))  # [rad]
    waves_rms_piston = sigma_piston / (2.0*math.pi)  # RMS wavefront in units of λ

    # ===== 5) Remove tilt =====
    kx_tilt = ky_tilt = 0.0
    if remove_tilt:
        U3 = U[..., None]  # (Nx,Ny,1)
        _, _, Ak = sim.fft_xy(U3)         # (Nx,Ny,1)
        Ak0 = Ak[..., 0]
        pow2 = (Ak0.real*Ak0.real + Ak0.imag*Ak0.imag)
        flat_idx = int(torch.argmax(pow2).item())
        ix = flat_idx // Ny
        iy = flat_idx - ix*Ny
        kx_tilt = float(sim.Kx[ix, 0].item())
        ky_tilt = float(sim.Ky[0, iy].item())
        phase_tilt = torch.exp(-1j*(kx_tilt*sim.X.to(dev) + ky_tilt*sim.Y.to(dev)))
        U = U * phase_tilt

    # ===== 6) Optional: remove defocus (scan α) =====
    alpha_opt = 0.0
    if search_defocus:
        # α scan range: ~ alpha_span_factor / w^2
        w_eff = sim._infer_beam_radius() or float(min(sim.x.abs().max(), sim.y.abs().max())*0.4)
        alpha_max = float(alpha_span_factor / (w_eff**2 + 1e-30))  # [rad/m^2]
        alphas = torch.linspace(-alpha_max, alpha_max, n_alpha, device=dev)
        r2 = (sim.X.to(dev)**2 + sim.Y.to(dev)**2)
        denom = torch.sum(torch.abs(U)) * dxdy + 1e-30
        Rbest = -1.0; kbest = 0
        for k, a in enumerate(alphas):
            ph = torch.exp(-1j * a * r2)
            E = torch.sum(U * ph) * dxdy
            R = float((torch.abs(E) / denom).item())
            if R > Rbest:
                Rbest = R; kbest = k
        alpha_opt = float(alphas[kbest].item())
        U = U * torch.exp(-1j * alpha_opt * r2)

    # ===== 7) Remove piston and compute final coherence/σφ =====
    E = torch.sum(U) * dxdy
    piston_phase = torch.angle(E)
    U_corr = U * torch.exp(-1j * piston_phase)   # Remove constant phase
    Emax = torch.sum(torch.abs(U_corr)) * dxdy + 1e-30
    R_corr = float((torch.abs(E) / Emax).item())
    sigma_corr = float(math.sqrt(max(0.0, -2.0*math.log(max(R_corr, 1e-12)))))  # [rad]
    waves_rms_corr = sigma_corr / (2.0*math.pi)
    strehl_est = R_corr**2                       # Approximate Strehl (Marechal approximation / coherence^2)

    out = dict(
        tidx = tidx,
        R_piston = R_piston,
        sigma_piston_rad = sigma_piston,
        waves_rms_piston = waves_rms_piston,
        R_corr = R_corr,
        sigma_corr_rad = sigma_corr,
        waves_rms_corr = waves_rms_corr,
        strehl_est = strehl_est,
        kx_tilt = kx_tilt,
        ky_tilt = ky_tilt,
        alpha_opt = alpha_opt,
    )

    if return_phase_map:
        # Return wrapped phase after removing tilt/defocus and piston (-π~π), for visualization only
        phi = torch.angle(U_corr).detach().cpu()
        out["phase_map_wrapped"] = phi  # (Nx,Ny)

    return out
