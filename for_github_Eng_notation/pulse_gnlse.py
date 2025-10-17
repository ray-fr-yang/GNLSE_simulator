#!/usr/bin/env python3
# ==============================================================
#  Minimal working version of PulseGNLSESimulator
#  * keeps only the parts you need right now *
#    – stores t_fwhm correctly
#    – builds initial pulse that you can inspect
# ==============================================================

from __future__ import annotations

# Header imports
import torch
from torch import fft as tfft     # alias for torch.fft
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
import psutil
import numpy as np
import matplotlib.pyplot as plt
from math import factorial, sqrt, log, pi
from typing import Optional, Callable, Dict, Sequence
from scipy.interpolate import interp1d
import os
from scipy import fft as sp_fft      # if switching to CuPy later, change this line
import pathlib
WORKERS = os.cpu_count()             # you can change this to the number of cores you want
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
        self.enable_shock = True
        self.enable_diffraction = True


        
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
        f = tfft.fftfreq(self.nt, d=self.dT, device=DEVICE)   # ★ no longer using fftshift
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
        self.Kx, self.Ky = torch.meshgrid(kx, ky, indexing='ij')              # ★ no fftshift
        self.K_perp2 = self.Kx**2 + self.Ky**2

        self._scale_fft_t   = (self.dT / np.sqrt(2*np.pi))
        self._scale_ifft_t  = (self.nt * self.domega / np.sqrt(2*np.pi))
        self._scale_fft_xy  = (self.dx * self.dy) / (2*np.pi)
        self._scale_ifft_xy = (2*np.pi) / (self.dx * self.dy)
        
        # Convert to scalar tensors on same dtype/device as data (avoid rebuilding each time)
        to_scalar = lambda v, dev=DEVICE: torch.tensor(v, dtype=torch.float32, device=dev)
        self._scale_fft_t   = to_scalar(self._scale_fft_t)
        self._scale_ifft_t  = to_scalar(self._scale_ifft_t)
        self._scale_fft_xy  = to_scalar(self._scale_fft_xy)
        self._scale_ifft_xy = to_scalar(self._scale_ifft_xy)


        # ---------- γ(ω) full-spectrum vector ------------------------------------
        omega_abs = self.omega + self.omega0          # ← moved outside the if

        if np.isscalar(n2):
            n2_vec = float(n2) * torch.ones_like(self.omega, device=DEVICE)
        else:
            lam_pts, n2_pts = np.asarray(n2).T
            n2_interp = interp1d(lam_pts, n2_pts, kind='linear',
                                  bounds_error=False,
                                  fill_value=(n2_pts[0], n2_pts[-1]))
            lam_grid = (2*torch.pi*self.c / omega_abs.cpu()).numpy()
            n2_vec = torch.from_numpy(n2_interp(lam_grid)).to(DEVICE)

        # γ(ω) — aligned with ω (after fftshift), shape (1,1,Nt)
        self.gamma_w = (omega_abs * n2_vec / self.c).to(torch.float32)[None, None, :].to(torch.complex64)
        
        # Keep the central-frequency point for diagnostics / B-integral
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

            # ④ Frequency-domain representation — directly call the torch version of self.fft_t
            _, H_R = self.fft_t(h_R)          # returns torch.complex64 on CUDA
            self.H_R_omega = H_R              # (Nt,) — extra dims will be added during broadcasting
        if self.raman_on:
            print('raman on')


        
        # ---- stepped film placeholder (route A geometry only at this step) ----
        self.stepped_film: SteppedFilm | None = None
        
        # === Route A: define reference medium (vacuum) and film material parameters ===
        self.n_ref = 1.0                               # reference medium = vacuum
        self.k0_ref = self.omega0 / self.c             # k_ref = ω0/c
        
        # Reference dispersion β^ref (vacuum): β1 = 1/c; m≥2 ~ 0 (extend as needed)
        beta_ref = {1: 1.0/self.c}
        # If you want to explicitly zero for clarity: for m in (2,3,4): beta_ref[m] = 0.0
        
        # Film material dispersion β^film uses existing self.dispersion (already validated above)
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
        # —— Precompute the two factors for the linear half step (stored separately) ——
        # Spatial factor: depends on kx, ky
        self.Lxy_half = torch.exp(
            (-1j * self.K_perp2 / (2 * self.k0)) * (self.dz / 2)
        ).to(torch.complex64)                               # (Nx, Ny)
        
        # Temporal factor: depends on ω and β_m, α
        Om = self.omega.to(torch.complex128)               # (Nt,)
        disp_phase = torch.zeros_like(Om, dtype=torch.complex128)
        for order, beta in self.dispersion.items():
            if self.retarded and order == 1:
                continue
            disp_phase += beta * (Om**order) * ((-1)**order) / factorial(order)
        
        self.Lw_half = torch.exp(
            (1j * disp_phase - self.alpha/2) * (self.dz / 2)
        ).to(torch.complex64)                               # (Nt,)

        
        
        # —— Precompute once to reuse later (avoid per-step dtype conversions & rebuilding) ——
        self.gamma_w_even128 = (0.5*(self.gamma_w + torch.flip(self.gamma_w, dims=[-1]))).to(torch.complex128)
        if self.raman_on:
            self.H_R_omega128 = self.H_R_omega.to(torch.complex128)
        self.omega128 = self.omega.to(torch.complex128)
        
        # ---------- other bookkeeping --------------------------------
        if generate_A_in:
            self.A_in_gpu = self._generate_initial_pulse(
            chirp = 0.0,                             # optional initial chirp
            extra_spatial_phase = lambda X,Y:        # optional wavefront curvature
                0.5 * self.k0 / Rc * (X**2 + Y**2) if Rc is not None else 0.0
            )
            self.A_in = self.A_in_gpu.cpu()                     # keep CPU copy only
            del self.A_in_gpu                                    # free GPU memory immediately
            torch.cuda.empty_cache()
        else:
            self.A_in = None
            
        # ---- Record initial 1D FWHM in k-space (kx/ky), relies on existing diagnose() ----
        # Note: diagnose(domain='kspace') internally calls self.fft_xy(self.A),
        # so we temporarily point self.A to the GPU version of A_in, compute, then restore and clear cache.
        try:
            A_saved = getattr(self, "A", None)
            self.A  = self.A_in.to(self.T.device, non_blocking=True)  # temporary GPU copy
            diag_k  = self.diagnose(domain='kspace', metrics=('fwhm',), save_raw=False)
            self.kx_fwhm0 = float(diag_k['metrics']['fwhm']['kx'])   # [m^-1]
            self.ky_fwhm0 = float(diag_k['metrics']['fwhm']['ky'])   # [m^-1]
            self.k_fwhm0  = 0.5 * (self.kx_fwhm0 + self.ky_fwhm0)    # representative value
        finally:
            self.A = A_saved
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(self.k_fwhm0)
        self.k_fwhm0=600


        # === 2.1 Spatial reference coefficient: _coef_xy_ref (Nx,Ny) ===
        self._coef_xy_ref = (-1j * self.K_perp2 / (2.0 * self.k0_ref)).to(torch.complex64)
        
        # === 2.2 Temporal reference coefficient: _coef_w_ref (Nt,) ===
        Om128   = self.omega.to(torch.complex128)
        disp_ref = torch.zeros_like(Om128, dtype=torch.complex128)
        for m, beta in self.materials["beta_ref"].items():
            if self.retarded and m == 1:   # skip β1_ref in retarded frame
                continue
            disp_ref += beta * ((-1)**m) * (Om128**m) / factorial(m)
        self._coef_w_ref = (1j * disp_ref - self.materials["alpha_ref"]/2.0).to(torch.complex64)


        
        # === 2.3 Incremental coefficients: _coef_w_inc (Nt,) — note Δβ1 must be kept! ===
        # Incremental frequency-domain polynomial (including all Δβm with m≥1)
        disp_inc = torch.zeros_like(Om128, dtype=torch.complex128)
        orders = set(self.materials["beta_film"].keys()) | set(self.materials["beta_ref"].keys())
        for m in orders:
            if m >= 1:  # m=1 automatically includes Δβ1; m≥2 are higher orders
                bf = float(self.materials["beta_film"].get(m, 0.0))
                br = float(self.materials["beta_ref"].get(m, 0.0))  # Vacuum: beta_ref[1]=1/c, others ≈ 0
                disp_inc += (bf - br) * ((-1)**m) * (Om128**m) / factorial(m)
        
        # Δβ0 and Δα
        delta_beta0 = (self.omega0 / self.c) * (self.materials["n_film"] - self.materials["n_ref"])

        delta_alpha = self.materials["alpha_film"] - self.materials["alpha_ref"]
        
        # Final incremental coefficient (complex64 to save memory/compute)
        self._coef_w_inc = (1j * (delta_beta0 + disp_inc) - delta_alpha/2.0).to(torch.complex64)

        # === If the time reference operator is (almost) identity, we can skip two t-FFTs during propagation ===
        with torch.no_grad():
            # Threshold can be tuned as needed
            self._time_ref_is_identity = (torch.max(torch.abs(self._coef_w_ref)).item() < 1e-14)


        
        # Working copy lives on GPU only
        self.A = self.A_in     # move back to GPU before propagation
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
    
        # Do it step-by-step to profile key points
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
        # --- New: automatic aperture parameters ---
        aperture_radius: float | None = None,   # If None and aperture_factor is given → automatically set to aperture_factor * w
        aperture_factor: float | None = 3.0,    # default = 3w; set to None to add no outer aperture
        aperture_fwhm: float | None = None,     # soft-edge FWHM; default same as dr_fwhm
        names: Sequence[str] | None = None,
        to_device: torch.device | None = None,
    ) -> SteppedFilm:
        """
        Construct self.stepped_film with K layers.
        If aperture_radius is None and aperture_factor is provided, automatically use aperture_factor * w as the outer aperture.
        """
        K = len(dL_list)
        assert len(inner_radii) == K, "inner_radii and dL_list must match length"
    
        if isinstance(nz_per_layer, int):
            nz_list = [int(nz_per_layer)] * K
        else:
            assert len(nz_per_layer) == K, "nz_per_layer must be int or length-K"
            nz_list = [int(z) for z in nz_per_layer]
    
        dev = to_device or self.X.device
    
        # ------- Automatic outer aperture (default 3w) -------
        if aperture_radius is None and (aperture_factor is not None):
            w = self._infer_beam_radius()
            aperture_radius = float(aperture_factor) * float(w)
        # Don't exceed grid boundary
        if aperture_radius is not None:
            r_grid = float(min(self.x.abs().max(), self.y.abs().max()))
            aperture_radius = float(min(aperture_radius, r_grid * 0.98))
        if aperture_fwhm is None:
            aperture_fwhm = dr_fwhm
    
        # ------- Generate masks for each layer -------
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
        """Return the 1/e^2 radius w (meters). Prefer beam_profile['waist']; otherwise fall back to a grid-size estimate."""
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
        # Fallback: take 0.4 × half of the grid width to ensure it stays within the window
        r_grid = float(min(self.x.abs().max(), self.y.abs().max()))
        return 0.4 * r_grid
    

    
    # =================================================================
    #  GPU version of _generate_initial_pulse
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
        Build A(x,y,t) on GPU so that ∭|A|² dx dy dt = pulse_energy.
        Returns
        -------
        torch.Tensor
            dtype = complex64, device = DEVICE, shape (Nx, Ny, Nt)
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
            # 1) take the existing field
            field = self.beam_profile["field"]          # numpy / torch / …
            if isinstance(field, np.ndarray):
                A0 = torch.from_numpy(field)
            else:
                A0 = field
            if not torch.is_tensor(A0):
                raise TypeError("beam_profile['field'] must be numpy array or torch tensor")
    
            # 2) shape check
            if A0.shape != (self.Nx, self.Ny, self.nt):
                raise ValueError(
                    f"Imported field shape {A0.shape} ≠ simulator grid {(self.Nx, self.Ny, self.nt)}"
                )
    
            # 3) move to target device & dtype
            A0 = A0.to(dtype=torch.complex64, device=DEVICE).clone()
    
            # 4) optional energy renormalisation (on by default)
            if self.beam_profile.get("renorm", True):
                E_file = torch.trapz(
                    torch.trapz(torch.trapz(torch.abs(A0) ** 2, self.T), self.y), self.x
                )
                scale = torch.sqrt(self.pulse_energy / E_file)
                A0 *= scale
    
            return A0   # ← return directly; skip everything below    

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
    
            # ---- coherent low-order modes --------------------------
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
        A_xy /= torch.sqrt(area_xy)     # now ∫∫|A_xy|² dx dy = 1
    
        if extra_spatial_phase is not None:
            phase = extra_spatial_phase(self.X, self.Y)
            if not torch.is_tensor(phase):                       # ← new
                phase = torch.zeros_like(self.X) + phase         # ← new
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
        if not getattr(self, "enable_diffraction", True):
            return
        """Spatial half-step in the reference medium: A(x,y,t) -> A(kx,ky,t), multiply by exp(_coef_xy_ref*dz/2) -> back."""
        # (x,y,t) → (kx,ky,t)
        _, _, A_k = self.fft_xy(self.A)
        # Multiply spatial half-step (independent of t, broadcast to Nt)
        self.A = None
        torch.cuda.empty_cache()
        A_k *= self._Lxy_ref_half(dz_half)[..., None]  # (Nx,Ny,1)
        # (kx,ky,t) → (x,y,t)
        _, _, self.A = self.ifft_xy(A_k)
    

    @torch.inference_mode()
    def _time_ref_half(self, dz_half: float, *, eps: float = 1e-14) -> None:
        """Temporal half-step in the reference medium: A(x,y,t) -> A(x,y,ω), multiply by exp(_coef_w_ref*dz/2) -> back."""
        # Optional: if this half-step is nearly identity (e.g., vacuum + retarded_frame dropping β1), skip it
        max_mag = torch.max(torch.abs(self._coef_w_ref * float(dz_half))).item()
        if max_mag < eps:
            return
        # (x,y,t) → (x,y,ω)
        _, A_w = self.fft_t(self.A)
        # Multiply temporal half-step (broadcast to Nx,Ny)
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
        eps: float = 0.0,        # Optional: skip when the operator is approximately identity
    ) -> None:
        """
        Route-A: incremental linear half-step (no chunking; apply at once over the whole (Nx,Ny,Nt) block).
        Frequency-domain update (for each Ω):
            Â_new = Â_old * [ (1-M) + M * amp(Ω) ]
                  = Â_old + M * (amp(Ω)-1) * Â_old
        where amp(Ω) = exp(_coef_w_inc(Ω) * dz_half), and M = Mj(x,y) ∈ [0,1] is a soft mask.
        """
    
        dev = self.T.device
    
        # 1) Prepare: mask and the "incremental operator"
        Mj = Mj.to(device=dev, dtype=torch.float32, non_blocking=True)   # (Nx,Ny)
        amp = self._amp_increment_half(dz_half)                           # (Nt,) complex64
    
        # Optional: if this half-step is nearly identity, skip it (save FFTs and memory)
        if eps > 0.0:
            if torch.max(torch.abs((amp - 1.0))).item() < eps:
                return
    
        # 2) (x,y,t) → (x,y,ω)
        _, A_w = self.fft_t(self.A)             # A_w: (Nx,Ny,Nt) complex64
        # Release time-domain A to reduce peak memory (critical step)
        self.A = None
        torch.cuda.empty_cache()
    
        # 3) In frequency domain, multiply by amp only inside the film; keep outside unchanged.
        #    Use a "stitch back" form to reduce large intermediate tensors:
        #    A_w_masked = A_w * amp
        #    A_w        = (1-M)*A_w + M*A_w_masked
        A_w_masked = A_w * amp[None, None, :]                         # (Nx,Ny,Nt)
        # First turn A_w into the "outside part": (1-M)*A_w
        A_w.mul_((1.0 - Mj)[..., None].to(torch.float32))             # in-place
        # Add back the film-inside part: M * A_w_masked
        A_w_masked.mul_(Mj[..., None].to(torch.float32))              # in-place
        A_w.add_(A_w_masked)                                          # in-place accumulate
        del A_w_masked
        torch.cuda.empty_cache()
        # 4) (x,y,ω) → (x,y,t)
        _, self.A = self.ifft_t(A_w)                                     # (Nx,Ny,Nt)e()
 

    
    @torch.inference_mode()
    def _kerr_step_with_film(self, dz: float, Mj: torch.Tensor, *,
                             tile_x: int = 32, empty_cache_every: int = 0):
        """
        Kerr/Raman/shock nonlinear step (streamed by x-tiles). This assumes nonlinearity is active
        only in the film region, so the provided soft mask Mj(x,y) ∈ [0,1] is always used:
            A *= exp(i * [Mj * φ(t)] * dz)         (SPM pure phase)
            A -= Mj * (γ0/ω0) * ∂t[A(t) N(t)] * dz (shock: conservative form)
        """
        with torch.no_grad():
            Nx, Ny, Nt = self.Nx, self.Ny, self.nt
            dev = self.T.device
    
            # Frequency-domain kernels (prepared in __init__)
            gamma_even = self.gamma_w_even128   # (1,1,Nt) complex128
            H_R        = self.H_R_omega128 if self.raman_on else None
    
            # Mask (x,y), float32 on GPU; slice each tile then broadcast
            Mj = Mj.to(device=dev, dtype=torch.float32, non_blocking=True)
    
            # Shock coefficient and CFL
            s       = float(self.gamma0 / self.omega0)
            dT      = float(self.dT)
            cfl_max = 0.5  # increase up to ~0.7 if you want to be more aggressive
    
            tcount = 0
            for x0 in range(0, Nx, tile_x):
                xs = slice(x0, min(x0 + tile_x, Nx))
    
                # ===== take one tile and promote precision to complex128 =====
                A_blk = self.A[xs, :, :].to(torch.complex128)           # (tx,Ny,Nt)
    
                # ===== (1) N̂(ω) = FFT_t{|A|²}, multiply Raman =====
                I_hat = self.fft_t(torch.abs(A_blk)**2)[1]              # (tx,Ny,Nt)
                if H_R is not None:
                    I_hat *= (1.0 - self.f_R)
                    I_hat += (self.f_R * H_R) * I_hat / (1.0 - self.f_R)
    
                # ===== (2) SPM: φ(t) = IFFT_t{ γ_even · N̂ } (real), active only in film =====
                phi_t = self.ifft_t(I_hat * gamma_even)[1].real         # (tx,Ny,Nt)
                M_blk = Mj[xs, :].to(dtype=torch.float32)               # (tx,Ny)
                phase = (phi_t * dz) * M_blk[..., None]
                A_blk *= torch.exp(1j * phase)
                del phi_t, phase
    
                # ===== (3) self-steepening (conservative, full step; no FFT) =====
                if getattr(self, "enable_shock", True):
                    # N(t) (including Raman). Release I_hat immediately to reduce peak memory.
                    N_t = self.ifft_t(I_hat)[1].real                     # (tx,Ny,Nt)
                    del I_hat
                    if N_t.dtype == torch.float64:                       # coefficients can be fp32
                        N_t = N_t.to(torch.float32)
    
                    # CFL substep estimate (global upper bound; avoid extra large tensors)
                    a_max = abs(s) * float(torch.max(torch.abs(N_t)).item())
                    n_sub = max(1, int(np.ceil((a_max * dz / dT) / max(cfl_max, 1e-9))))
                    dz_sub = dz / n_sub
    
                    # Expand film mask to (tx,Ny,1) and cast to complex for in-place updates
                    M = M_blk[..., None].to(dtype=A_blk.dtype)
    
                    for _ in range(n_sub):
                        # right neighbor (periodic)
                        X_R = torch.roll(A_blk, shifts=-1, dims=-1)     # complex
                        N_R = torch.roll(N_t,   shifts=-1, dims=-1)     # real
    
                        # average of physical fluxes F_L, F_R: F̂ ← 0.5(F_L+F_R)
                        F_hat_R = s * A_blk * N_t                       # complex
                        F_hat_R.add_(s * X_R * N_R)                     # += s A_R N_R
                        F_hat_R.mul_(0.5)
    
                        # viscous term: -0.5 a_face (A_R - A)
                        X_R.sub_(A_blk)                                 # X_R ← A_R - A
                        a_face = torch.maximum(torch.abs(s * N_t), torch.abs(s * N_R))  # real
                        F_hat_R.add_(a_face.to(F_hat_R.dtype) * X_R, alpha=-0.5)

    
                        # divergence: F̂_{j+1/2} - F̂_{j-1/2}
                        F_hat_L = torch.roll(F_hat_R, shifts=+1, dims=-1)
                        F_hat_R.sub_(F_hat_L)                            # → dF
                        F_hat_R.mul_(-dz_sub / dT)                       # → ΔA
    
                        # update only inside film
                        A_blk.add_(M * F_hat_R)
    
                        # release temporaries timely
                        del X_R, N_R, a_face, F_hat_L, F_hat_R
    
                    del M, N_t
                else:
                    del I_hat
                    del M_blk
    
                # ===== (4) write back (downcast to complex64) =====
                self.A[xs, :, :] = A_blk.to(torch.complex64)
                del A_blk
    
                tcount += 1
                if empty_cache_every and (tcount % empty_cache_every == 0):
                    torch.cuda.empty_cache()

                    
    @torch.inference_mode()
    def _apply_global_time_shift(self, dt: float) -> None:
        """Global time shift: A(t) ← A(t+dt); multiply frequency domain by e^{+iΩ dt}."""
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
        Turn cumulative stepped_film masks into mutually exclusive regions:
          Region0 = center region, RegionK-1 = outermost region, and rings in between.
        Returns a list of float32 tensors on device=self.T.device.
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
        Compute the overlap fraction of two soft masks: ⟨A·B⟩/⟨A⟩.
        Treat A as the "region" and B as "this layer".
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
                            center_camera_midpoint: bool = False,
                            # ===== New: per-layer save directory (None = do not save) =====
                            layer_dir: str | None = None):
        """
        (…same as original…)
        Additional parameter
        --------------------
        layer_dir : str | None
            If provided, after each layer completes, save the field as {j:02d}_film_{j:02d}.pt to that directory.
        """
        assert self.stepped_film is not None, "stepped_film not set; call set_stepped_film_from_annuli(...) first"
    
        dev = self.T.device
        self.A = (self.A_in if self.A is None else self.A).to(dev, non_blocking=True)
    
        # If saving per-layer results, create the directory first
        if layer_dir is not None:
            import os
            os.makedirs(layer_dir, exist_ok=True)
    
        if apply_surfaces:
            self.A.mul_(torch.sqrt(torch.tensor(self.T_surf, device=dev, dtype=self.A.dtype)))
    
        snapshots: list[torch.Tensor] = []
        global_step = 0
    
        # —— Camera midpoint: preparation —— #
        if center_camera_midpoint:
            regions = self._build_exclusive_region_masks()
            M_center = regions[0]
            M_outer  = regions[-1]
            # Δβ1 = β1_film - β1_ref
            delta_beta1 = float(self.materials["beta_film"][1] - self.materials["beta_ref"][1])  # [s/m]
    
        for j, layer in enumerate(self.stepped_film.layers):
            dL = float(layer.dL); nz = int(layer.nz)
            if nz <= 0 or dL == 0.0:
                continue
    
            dz = dL / nz
            dz_half = 0.5 * dz
    
            Mj = layer.mask.to(device=dev, dtype=torch.float32, non_blocking=True)
            has_film = (Mj.max().item() > 0.0)
    
            for s in range(nz):
                print(global_step)
                # ---- ½ reference (space) ----
                self._space_ref_half(dz_half)
                # ---- ½ reference (time) ----
                self._time_ref_half(dz_half)
    
                # ---- ½ incremental linear (film region) ----
                if has_film:
                    self._apply_incremental_linear_half(Mj, dz_half, eps=eps_inc)
    
                # ---- nonlinear (film region) ----
                if has_film:
                    self._kerr_step_with_film(dz, Mj, tile_x=tile_x_kerr)
    
                # ---- ½ incremental linear (film region) ----
                if has_film:
                    self._apply_incremental_linear_half(Mj, dz_half, eps=eps_inc)
    
                # ---- ½ reference (time) ----
                self._time_ref_half(dz_half)
                # ---- ½ reference (space) ----
                self._space_ref_half(dz_half)
    
                # === Camera centering: take the average GD of "center & outer" for this step and shift back ===
                if center_camera_midpoint and has_film:
                    # Overlap fraction (soft edges): whether this layer affects the center/outer regions
                    fc = self._overlap_fraction(M_center, Mj)   # ∈[0,1]
                    fo = self._overlap_fraction(M_outer,  Mj)
                    # Incremental group delay at the two ends for this step (relative to vacuum)
                    tau_c = delta_beta1 * dz * fc               # [s]
                    tau_o = delta_beta1 * dz * fo               # [s]
                    dt_shift = 0.5 * (tau_c + tau_o)            # [s] — midpoint of the two ends
                    if dt_shift != 0.0:
                        # Apply as a reverse shift to the "camera time origin": A(t) ← A(t+dt_shift)
                        self._apply_global_time_shift(dt_shift)
    
                # ---- snapshots / callback ----
                if global_step in store_steps:
                    snapshots.append(self.A.detach().cpu().clone())
                if callback is not None:
                    callback(global_step, self)
                global_step += 1
    
            # ======== After each layer: save the field after this layer ========
            if layer_dir is not None:
                # Copy GPU field to CPU (as in the example, write to self.A_out then save)
                self.A_out = self.A.detach().cpu()
                # Filename like "01_film_01.pt", "02_film_02.pt", ...
                j1 = j + 1  # 1-based
                # If you'd rather set the right-hand number to this layer's nz, change to: fname = f"{j1:02d}_film_{nz:02d}.pt"
                fname = f"{j1:02d}_film_{j1:02d}.pt"
                fpath = os.path.join(layer_dir, fname)
                torch.save(self.A_out, fpath)
            # ============================================
    
        if apply_surfaces:
            self.A.mul_(torch.sqrt(torch.tensor(self.T_surf, device=dev, dtype=self.A.dtype)))
    
        self.A_out = self.A.detach().cpu()
        self.A = None
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
        Propagate only in the reference medium = vacuum over length L (meters), split into nz steps.
        No film and no nonlinearity are used; each step does:
            ½L_ref(space) → ½L_ref(time) → ½L_ref(time) → ½L_ref(space)
        Note: if retarded_frame=True at initialisation, β1_ref is dropped internally (co-moving frame).
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
    
        # Build the vacuum temporal half-step operator locally to avoid being skipped when retarded_frame=True
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
            if not getattr(self, "enable_diffraction", True):
                _, A = self.fft_t(self.A)
                A *= self.Lw_half[None, None, :]
                _, A = self.ifft_t(A)
                self.A = A
                return

             # 1) (x,y,t) → (kx,ky,t)
            _, _, A = self.fft_xy(self.A)              # A: (Nx,Ny,Nt), complex64
            self.A = None
            torch.cuda.empty_cache()
            # 2) (kx,ky,t) → (kx,ky,ω)
            _, A = self.fft_t(A)                       # A: (Nx,Ny,Nt)
                
            # 3) Multiply the separated linear half-step factors (broadcast; does not copy large tensors)
            A *= self.Lxy_half[..., None]              # (Nx,Ny,1) broadcast to (Nx,Ny,Nt)
            A *= self.Lw_half[None, None, :]           # (1,1,Nt)  broadcast to (Nx,Ny,Nt)
                
            # 4) (kx,ky,ω) → (kx,ky,t) → (x,y,t)
            _, A = self.ifft_t(A)
            _, _, A = self.ifft_xy(A)
                
            self.A = A                                 # write back

        
    @torch.inference_mode()    
    def _kerr_step(self, dz: float, *, tile_x: int = 32, empty_cache_every: int = 0):
        """
        Memory-friendly Kerr step: stream blocks along x (tile_x × Ny × Nt).
        SPM is a pure phase; shock uses the conservative form -(γ0/ω0) ∂t(A N) with periodic boundaries.
        """
        with torch.no_grad():
            Nx, Ny, Nt = self.Nx, self.Ny, self.nt
    
            gamma_even = self.gamma_w_even128
            H_R        = self.H_R_omega128 if self.raman_on else None
    
            s       = float(self.gamma0 / self.omega0)
            dT      = float(self.dT)
            cfl_max = 0.5
    
            tcount = 0
            for x0 in range(0, Nx, tile_x):
                xs = slice(x0, min(x0 + tile_x, Nx))
    
                # ===== (0) Take one tile and promote precision =====
                A_blk = self.A[xs, :, :].to(torch.complex128)            # (tx,Ny,Nt)
    
                # ===== (1) N̂(ω) = FFT_t{|A|²}, multiply Raman =====
                I_hat = self.fft_t(torch.abs(A_blk)**2)[1]
                if H_R is not None:
                    I_hat *= (1.0 - self.f_R)
                    I_hat += (self.f_R * H_R) * I_hat / (1.0 - self.f_R)
    
                # ===== (2) SPM phase =====
                phi_t = self.ifft_t(I_hat * gamma_even)[1].real
                A_blk *= torch.exp(1j * phi_t * dz)
                del phi_t
    
                # ===== (3) Self-steepening (conservative, full-step; no FFT) =====
                if getattr(self, "enable_shock", True):
                    N_t = self.ifft_t(I_hat)[1].real
                    del I_hat
                    if N_t.dtype == torch.float64:
                        N_t = N_t.to(torch.float32)
    
                    a_max = abs(s) * float(torch.max(torch.abs(N_t)).item())
                    n_sub = max(1, int(np.ceil((a_max * dz / dT) / max(cfl_max, 1e-9))))
                    dz_sub = dz / n_sub
    
                    for _ in range(n_sub):
                        X_R = torch.roll(A_blk, shifts=-1, dims=-1)     # complex
                        N_R = torch.roll(N_t,   shifts=-1, dims=-1)     # real
    
                        F_hat_R = s * A_blk * N_t
                        F_hat_R.add_(s * X_R * N_R)
                        F_hat_R.mul_(0.5)
    
                        X_R.sub_(A_blk)
                        a_face = torch.maximum(torch.abs(s * N_t), torch.abs(s * N_R))
                        F_hat_R.add_(a_face.to(F_hat_R.dtype) * X_R, alpha=-0.5)

    
                        F_hat_L = torch.roll(F_hat_R, shifts=+1, dims=-1)
                        F_hat_R.sub_(F_hat_L)                           # → dF
                        F_hat_R.mul_(-dz_sub / dT)                      # → ΔA
    
                        A_blk.add_(F_hat_R)
    
                        del X_R, N_R, a_face, F_hat_L, F_hat_R
    
                    del N_t
                else:
                    del I_hat
    
                # ===== (4) Write back =====
                self.A[xs, :, :] = A_blk.to(torch.complex64)
                del A_blk
    
                tcount += 1
                if empty_cache_every and (tcount % empty_cache_every == 0):
                    torch.cuda.empty_cache()

        
    

    # --------------------------------------------------------------
    #  Inside PulseGNLSESimulator
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
            Number of propagation steps; default = self.nz.
        store_steps : tuple[int]
            Step indices (0-based) at which to save snapshots; range(...) is acceptable.
        callback : callable(step_idx:int, self), optional
            Executed after each step, e.g., for tqdm progress or diagnostics.

        Returns
        -------
        list[np.ndarray]
            Saved copies of A(x,y,t), in the same order as store_steps.
        """
        with torch.no_grad():
            self.A = self.A.to(DEVICE)     # move back to GPU before propagation
            self.A = self.A * torch.sqrt(torch.tensor(self.T_surf, device=DEVICE))
    
            
            if n_steps is None:
                n_steps = self.nz
            dz = self.dz
            snapshots: list[torch.Tensor] = []                        # ← store snapshots
            
            for step in range(n_steps):
                print(step)               
        
                # -------- ½ linear --------
                self._linear_half_step()
                torch.isnan(self.A).any() and print('NaN after ½lin', step)
                
                # -------- Kerr nonlinearity --------
                self._kerr_step(dz)
                #print('Kerr')
                torch.isnan(self.A).any() and print('NaN after kerr', step)

                # -------- ½ linear --------
                self._linear_half_step()
                #print('2nd linear')
                torch.isnan(self.A).any() and print('NaN after ½lin', step)

                # -------- record / callback --------
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
                    snapshots.append(self.A.clone())      # deep copy
                if callback is not None:
                    callback(step, self)
                
    
    
            self.A = self.A * torch.sqrt(torch.tensor(self.T_surf, device=DEVICE))       
            beta2_mgf2 = 2.05e-26             # s^2/m
            L_win      = 5.1e-3               # m
            phase_win  = 0.5 * beta2_mgf2 * (self.omega**2) * L_win   # because factorial(2)=2
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
    
        # —— Preview parameters ——
        preview: bool = True,
        preview_shift: bool = True,
        preview_logscale: bool = True,
        preview_db_floor: float = -60.0,
        preview_percentile: float = 99.9,
        preview_span_factor: float = 1.5
    ) -> torch.Tensor:
        """
        Circular low-pass based on the "spatial spectrum FWHM of the initial A_in":
            k_cut = factor * (self.k_fwhm0 / 2)
        Preview is for display only; the filtered result is written to self.A_out (CPU) and also returned.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        from torch import fft as tfft
    
        # —— Fallback: if no k_fwhm0, measure once on the fly ——
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
    
        # —— Choose input field ——
        if A is None:
            if field == "output":
                if self.A_out is None:
                    raise ValueError("apply_circular_lowpass_by_factor: self.A_out is None; obtain an output field first or pass A explicitly.")
                A_src = self.A_out
            elif field == "current":
                if self.A is None:
                    raise ValueError("apply_circular_lowpass_by_factor: self.A is None; a 'current' field exists only during propagation.")
                A_src = self.A
            elif field == "input":
                if self.A_in is None:
                    raise ValueError("apply_circular_lowpass_by_factor: self.A_in is None.")
                A_src = self.A_in
            else:
                raise ValueError("field must be 'output' | 'current' | 'input'")
        else:
            A_src = A
    
        A_src = (A_src if torch.is_tensor(A_src) else torch.as_tensor(A_src))
        if A_src.shape != (self.Nx, self.Ny, self.nt):
            raise ValueError(f"Input field shape {tuple(A_src.shape)} ≠ {(self.Nx,self.Ny,self.nt)}")
    
        dev = self.T.device
        A_src = A_src.to(device=dev, dtype=torch.complex64, non_blocking=True)
    
        # —— Cutoff radius & soft edge ——
        k_cut = float(factor) * (float(self.k_fwhm0) * 0.5)
        if dk_fwhm is None:
            dk_fwhm = float(dk_rel) * max(k_cut, 1e-12)
        dk_fwhm = float(dk_fwhm)
    
        # Low-pass kernel
        K_perp = torch.sqrt(self.K_perp2).to(dev)
        Hk     = soft_unit_step(k_cut - K_perp, dk_fwhm).to(torch.float32)  # (Nx,Ny)∈[0,1]
    
        # —— Process in chunks along t ——
        Nt = int(self.nt)
        dx, dy, dT    = float(self.dx), float(self.dy), float(self.dT)
        scale_fft_xy  = float(self._scale_fft_xy.detach().cpu())
        scale_ifft_xy = float(self._scale_ifft_xy.detach().cpu())
    
        out = torch.empty_like(A_src, device=dev)
        E_in = 0.0
        E_ot = 0.0
    
        # Preview accumulation: I(kx,ky) = ∫ |Â|² dt (trapezoidal integration)
        I_accum = torch.zeros((self.Nx, self.Ny), device=dev, dtype=torch.float32) if preview else None
    
        for t0 in range(0, Nt, int(tile_t)):
            t1   = min(t0 + int(tile_t), Nt)
            Ablk = A_src[:, :, t0:t1]
    
            # Input energy (block)
            Pin = torch.sum(torch.abs(Ablk)**2, dim=(0,1)) * dx * dy
            E_in += float(torch.sum(Pin).item() * dT)
    
            # 2D FFT
            Ak = tfft.fftn(Ablk, dim=(0, 1))
            Ak.mul_(scale_fft_xy)
    
            # —— Preview integral: trapezoidal weights (0.5 at the first/last ends) ——
            if I_accum is not None:
                pow2 = Ak.real*Ak.real + Ak.imag*Ak.imag                  # (Nx,Ny,dt)
                w = torch.ones((t1 - t0), device=dev, dtype=pow2.dtype)
                if t0 == 0:  w[0]  = 0.5
                if t1 == Nt: w[-1] = 0.5
                I_accum.add_(torch.sum(pow2 * w[None, None, :], dim=2) * dT)
    
            # Filter → iFFT
            Ak *= Hk[..., None].to(Ak.dtype)
            Ablk = tfft.ifftn(Ak, dim=(0, 1))
            Ablk.mul_(scale_ifft_xy)
            del Ak
    
            out[:, :, t0:t1] = Ablk
    
            # Output energy (block)
            Pout = torch.sum(torch.abs(Ablk)**2, dim=(0,1)) * dx * dy
            E_ot += float(torch.sum(Pout).item() * dT)
            del Ablk
    
        # Optional energy preservation
        if preserve_energy and E_ot > 0.0:
            out.mul_(np.sqrt(E_in / E_ot))
    
        # —— Preview plotting (coordinates/shift/line-spectrum consistent with visualize_simulator) ——
        if I_accum is not None:
            with torch.no_grad():
                # Passed energy fraction (based on the unshifted kernel)
                num = float(torch.sum(I_accum * Hk).item())
                den = float(torch.sum(I_accum).item())
                pass_frac = (num / den) if den > 0 else 0.0
    
                # Native coordinates
                I2d = I_accum.detach().cpu().numpy()        # (Nx,Ny)
                Kx1 = self.Kx[:, 0].detach().cpu().numpy()  # (Nx,)
                Ky1 = self.Ky[0, :].detach().cpu().numpy()  # (Ny,)
    
                # Shift for display only
                if preview_shift:
                    I2d_show = np.fft.fftshift(I2d, axes=(0,1))
                    Kx_show  = np.fft.fftshift(Kx1)
                    Ky_show  = np.fft.fftshift(Ky1)
                else:
                    I2d_show = I2d
                    Kx_show  = Kx1
                    Ky_show  = Ky1
    
                # Dynamic range
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
    
                # Field of view
                span = max(1e-6, float(preview_span_factor) * max(k_cut, 1e-12))
    
                # Plot
                fig, (ax2d, ax1d) = plt.subplots(1, 2, figsize=(11, 4.5))
    
                # 2D spectrum
                extent = [Kx_show.min(), Kx_show.max(), Ky_show.min(), Ky_show.max()]
                im = ax2d.imshow(img, origin="lower", extent=extent, aspect="equal",
                                 vmin=vmin, vmax=vmax)
                fig.colorbar(im, ax=ax2d, label=cbar_label)
                ax2d.set_xlabel("kₓ [m⁻¹]"); ax2d.set_ylabel("k_y [m⁻¹]")
                ax2d.set_title(f"Pre-filter spectrum  (predicted pass ~ {pass_frac*100:.1f}%)")
                ax2d.set_xlim(-span, +span); ax2d.set_ylim(-span, +span)
                ax2d.add_patch(Circle((0.0, 0.0), radius=k_cut, fill=False, lw=1.5))
    
                # 1D line spectrum: slice at ky≈0 (same as visualize_simulator)
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
                    print("[WARN] With these parameters the predicted passed energy is < 0.1%. "
                          "Please check whether k_fwhm0 is reasonable, or reduce 'factor' / relax 'dk'.")
    
        # —— Write back to CPU ——
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
                 subsample_align: bool = True,  # use frequency-domain phase shift for sub-sample alignment
                 report_center: bool = True):   # report t_peak / t_centroid

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
            # ---- raw projection P(t) --------------------------------------
            P_t  = np.sum(np.abs(A)**2, axis=(0,1)) * self.dx * self.dy
            T    = to_np(self.T)
            dT   = float(T[1] - T[0])

            # ---- find center (peak/centroid) ------------------------------
            t_peak     = float(T[np.argmax(P_t)]) if P_t.max() > 0 else 0.0
            t_centroid = self._temporal_centroid_np(T, P_t)

            # ---- optional alignment (does not change self.A; only the 1D curve for diagnostics) -------
            P_plot = P_t.copy()
            T_plot = T.copy()
            if align_time in ('peak', 'centroid'):
                t0 = t_peak if align_time == 'peak' else t_centroid
                if subsample_align:
                    # frequency-domain phase shift (sub-sample)
                    P_plot = self._shift_1d_np(T, P_plot, shift=-t0, d=dT)
                else:
                    # integer-sample roll (fast but ±0.5 sample error)
                    n = int(np.rint(t0 / dT))
                    P_plot = np.roll(P_plot, -n)
                # Also shift the coordinate reference so that 0 is centered (more intuitive)
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
                Ak_abs2 = (A_k.real**2 + A_k.imag**2)                  # → real, non-negative
                Ik_xy    = Ak_abs2.sum(axis=2) * float(self.dT)         # ∫|Â|^2 dt (rectangle rule)

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
    
        # Cross-boundary case? (both ends ≥ half; common in native FFT order)
        wraps = mask[0] and mask[-1]
        if wraps:
            # Unwrap to double length and then take first/last
            x_step = x[1] - x[0]
            x = np.concatenate([x, x + x[-1] + x_step])
            y = np.concatenate([y, y])
            mask = np.concatenate([mask, mask])
            idx  = np.flatnonzero(mask)
    
        i0, i1 = idx[0], idx[-1]
    
        # --- Left boundary: linear interp over [j0, j0+1], j0 ≥ 0
        j0 = max(i0 - 1, 0)
        if j0 + 1 >= len(x):
            x_l = x[j0]
        else:
            x_l = np.interp(half, y[j0:j0+2], x[j0:j0+2])
    
        # --- Right boundary: linear interp over [j1, j1+1], j1 ≤ len(x)-2
        j1 = min(i1, len(x) - 2)
        if j1 + 1 >= len(x):
            x_r = x[j1]
        else:
            x_r = np.interp(half, y[j1:j1+2], x[j1:j1+2])
    
        return float(x_r - x_l)

    
        # Linear interpolation (unreached duplicate kept as in original)
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
        """Centroid t_cm = ∫ t P(t) dt / ∫ P(t) dt; return 0 if total energy is 0."""
        denom = np.trapz(P, T)
        if denom == 0:
            return 0.0
        return float(np.trapz(T * P, T) / denom)

    @staticmethod
    def _shift_1d_np(x: np.ndarray, y: np.ndarray, shift: float, *, d: float) -> np.ndarray:
        """
        Shift y(x) along the x-axis by 'shift' (fractional samples allowed), and return the shifted y.
        Implemented via frequency-domain phase ramp: Y(k) * exp(-i k shift).
        """
        # Frequency coordinates (same as np.fft.fftfreq)
        k = 2*np.pi*np.fft.fftfreq(y.size, d=d)
        Y = np.fft.fft(y)
        y_shift = np.fft.ifft(Y * np.exp(-1j * k * shift))
        # For power curves, the imaginary part should be numerical noise
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
    Parse a radius interval in mm like '0.4–0.7' / '0.4-0.7' / '0.4 — 0.7'
    into (lo_m, hi_m) in meters.
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
    col_thickness: str = "t_final (mm)",   # can also use 't_flat (mm)'
    names_col: str | None = "Layer k",
    drop_const_out: bool = True,           # drop trailing const(out) row if increment ≈ 0
    eps_nm: float = 1.0                    # thickness increments below this (nm) treated as 0
) -> tuple[list[float | None], list[float], list[str]]:
    """
    Build (inner_radii, dL_list, names) from CSV:
    - inner_radii[0] = None means the first layer covers the full disk;
    - dL_list[0] = t[0]; dL_list[j] = t[j] - t[j-1] (turn absolute thickness into step increments).
    - Units: all returned values are in meters.
    """
    df = pd.read_csv(csv_path)

    # 1) Parse per-row radius intervals
    ranges_m = [ _parse_r_range_mm_to_m(s) for s in df[col_range].tolist() ]
    r_lo = [lo for lo, _ in ranges_m]
    r_hi = [hi for _, hi in ranges_m]

    # 2) Thickness (m) — by default use t_final
    t_m = (df[col_thickness].astype(float).to_numpy()) * 1e-3
    print(t_m * 1e3)

    # 3) Increment list: first layer is absolute thickness; others are differences
    dL = np.empty_like(t_m)
    dL[0] = t_m[0]
    dL[1:] = t_m[1:] - t_m[:-1]

    # 4) Clean: treat |Δ| < eps_nm as 0; clip negative increments to 0 and warn
    eps_m = eps_nm * 1e-9
    tiny = np.abs(dL) < eps_m
    dL[tiny] = 0.0
    if np.any(dL < -eps_m):
        print("[csv_to_layers_lists] WARNING: non-monotonic thickness detected; "
              "negative increments were clipped to 0.")
        dL = np.maximum(dL, 0.0)

    # 5) inner_radii: None for full disk on the first layer; then use each row's lower bound
    inner_radii: list[float | None] = [None] + [float(x) for x in r_lo[1:]]

    # 6) Optionally drop trailing rows with ≈0 increment (e.g., a 'const(out)' row)
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
    dr_fwhm: float = 40e-6,     # soft-edge FWHM (recommended 30–60 µm)
    aperture: str | float | None = "3w",  # '3w' | 'from_csv' | value in meters | None
    **csv_kwargs                   # forwarded to csv_to_layers_lists, e.g., col_thickness='t_flat (mm)'
):
    """
    Read CSV → inner_radii, dL_list → construct sim.stepped_film
    """
    inner_radii, dL_list, names = csv_to_layers_lists(csv_path, **csv_kwargs)
    
    # Choose outer aperture
    aperture_radius = None
    aperture_factor = None
    if aperture == "3w":
        aperture_radius = None
        aperture_factor = 3.0
    elif aperture == "from_csv":
        # Use the upper radius bound of the last CSV row; re-parse once
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
                    r,              # e.g. [1,2,3]  — radius boundaries (default in mm)
                    t,              # e.g. [0.1,0.2,0.3] — absolute thickness per zone (default in mm)
                    *,
                    units="mm",     # "mm" or "m"
                    nz_per_layer=4, # axial steps per layer
                    dr_fwhm=40e-6,  # soft-edge FWHM (m); 0 → hard edge
                    aperture_from_last=True,   # use the last radius as an outer aperture
                    names=None,     # optional: per-layer names
                    clip_negative=True,  # if absolute thickness is non-monotonic, clip negative increments to 0
                    interpret="absolute" # "absolute" = t are absolute thicknesses; "delta" = t are already increments
                   ):
    """
    Regions 0–r[0], r[0]–r[1], ... take thicknesses t[0], t[1], ... respectively.
    Internally converts absolute thicknesses to increments ΔL to match set_stepped_film_from_annuli.
    """
    r = [float(x) for x in r]
    t = [float(x) for x in t]
    assert len(r) == len(t), "Lengths of r and t must match (each boundary radius corresponds to one thickness value)"
    # Unit conversion
    if units == "mm":
        r_m = [x * 1e-3 for x in r]
        t_m = [x * 1e-3 for x in t]
    elif units == "m":
        r_m = r
        t_m = t
    else:
        raise ValueError("units must be 'mm' or 'm'")

    K = len(t_m)

    # Absolute t → incremental dL (cumulative steps):
    if interpret == "absolute":
        dL = [t_m[0]] + [t_m[i] - t_m[i-1] for i in range(1, K)]
    elif interpret == "delta":
        dL = list(t_m)
    else:
        raise ValueError("interpret must be 'absolute' or 'delta'")

    if clip_negative:
        # Same as the CSV variant: if thickness is non-monotonic, clip negative increments to 0
        dL = [x if x >= 0.0 else 0.0 for x in dL]

    # inner_radii length must equal the number of layers K
    # First layer uses None for “full disk”, subsequent ones start at r[0], r[1], ...
    inner_radii = [None] + r_m[:-1] if K > 0 else []

    # Optionally use the last radius as the outer aperture (to avoid infinite extent)
    aperture_radius = r_m[-1] if aperture_from_last else None

    # Default names
    if names is None:
        names = [f"step{j}" for j in range(K)]

    # Use your existing constructor
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
                            eps_um=1e-3,              # increments smaller than this (µm) are treated as 0 (de-noise)
                            sort_rows=True            # sort by r_lo ascending first
                            ):
    """
    Build H(x,y) (in meters) from a mirror-steps CSV.
    Fixes:
      • aperture='3w' now uses 3*w automatically (and clamps to grid)
      • Increment dH can be positive or negative; only tiny jitter is removed
      • Optional: sort by r_lo ascending
    """
    import pandas as pd, numpy as np, torch

    df = pd.read_csv(csv_path)

    # 1) Parse radius intervals (mm→m)
    ranges_m = [ _parse_r_range_mm_to_m(s) for s in df[col_range].astype(str).tolist() ]
    r_lo = np.array([lo for lo, _ in ranges_m], dtype=float)
    r_hi = np.array([hi for _, hi in ranges_m], dtype=float)

    # 2) Absolute height (µm→m)
    h_abs = df[col_height].astype(float).to_numpy() * 1e-6

    # 3) Sort by r_lo (avoid mis-ordered rows)
    if sort_rows:
        order = np.argsort(r_lo)
        r_lo, r_hi, h_abs = r_lo[order], r_hi[order], h_abs[order]
    print(r_hi)

    # 4) Absolute→incremental (allow ±), only remove ultra-small jitter
    dH = np.empty_like(h_abs)
    dH[0]  = h_abs[0]
    dH[1:] = h_abs[1:] - h_abs[:-1]
    eps_m = float(eps_um) * 1e-6
    dH[np.abs(dH) < eps_m] = 0.0   # de-jitter; do NOT clip sign!

    # 5) Inner radii: use None for full disk on the first layer; use each row's lower bound thereafter
    inner_radii = [None] + [float(x) for x in r_lo[1:]]

    # 6) Outer aperture radius
    if aperture == "3w":
        w = sim._infer_beam_radius() or 0.0
        if w > 0:
            aperture_radius = 3.0 * float(w)
        else:
            # Fallback: use upper bound of the last CSV row
            aperture_radius = float(r_hi[-1])
    elif aperture == "from_csv":
        aperture_radius = float(r_hi[-1])
    elif isinstance(aperture, (int, float)):
        aperture_radius = float(aperture)
    else:
        aperture_radius = None

    # Clamp to grid
    if aperture_radius is not None:
        r_grid = float(min(sim.x.abs().max(), sim.y.abs().max()))
        aperture_radius = min(aperture_radius, 0.98 * r_grid)

    # 7) Masks (soft edges)
    masks = make_annular_masks(sim.X, sim.Y,
                               inner_radii=inner_radii,
                               dr_fwhm=dr_fwhm,
                               aperture_radius=aperture_radius,
                               aperture_fwhm=dr_fwhm)

    # 8) Accumulate: H(x,y) = Σ dH_j · M_j
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
    inner_radii: Sequence[float],         # r_edge for each layer; j=0 may use None/negative to indicate “full disk = 1”
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
# Helper functions ————————————————————————————————————————————————————————
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
# Main visualisation routine ———————————————————————————————————————————————
# -----------------------------------------------------------------------------
@torch.inference_mode()
def visualize_simulator(
    sim,  # PulseGNLSESimulator instance
    *,
    field: Literal["input", "current", "output"] = "current",
    zoom_time: Optional[float] = 3.0,
    zoom_space: Optional[float] = 12.0,
    log_scale_2d: bool = False,
    save_path=None,  # NEW: path to save the figure; None = do not save
):
    """Create a 3×2 Matplotlib figure visualising the given *sim* state.

    Parameters
    ----------
    sim : PulseGNLSESimulator
        The simulator object.
    field : {'input','current','output'}
        Which electric-field snapshot to visualise.  *current* uses ``sim.A``.
    zoom_time, zoom_space : float | None
        Multiple of the FWHM/RMS width kept visible along the time and spatial axes.
        Set to *None* to show the full window.
    log_scale_2d : bool, default False
        If *True*, the 2-D heat maps are coloured on a log10 scale.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure object (already populated).
    """
    # -------- choose which field ------------------------------------

    # -------- 2) Unify: move to CPU (key step) ----------------------
    T   = sim.T.detach().cpu().numpy()
    x   = sim.x.detach().cpu().numpy()
    y   = sim.y.detach().cpu().numpy()
    omega = sim.omega.detach().cpu().numpy()
    Kx  = sim.Kx.detach().cpu().numpy()
    Ky  = sim.Ky.detach().cpu().numpy()
    dx, dy = float(sim.dx), float(sim.dy)    # python-float
    
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
 
    
    # ---------- Frequency-domain centering --------------------------
    omega = np.fft.fftshift(omega)          # (Nt,)
    S_w   = np.fft.fftshift(S_w)            # (Nt,)
    # --------------------------------------------------------------
    
    # ---------- k-space centering -----------------------------------
    I_kxky = np.fft.fftshift(I_kxky, axes=(0, 1))
    Kx     = np.fft.fftshift(Kx,   axes=0)
    Ky     = np.fft.fftshift(Ky,   axes=1)
    torch.cuda.empty_cache()
    

    # ===== 1D temporal power ========================================
    P_t = np.sum(I, axis=(0,1)) * dx * dy


    
    # ===== 1D spectral power (use direct FFT of full field) =========
    '''
    _,  A_w  = sim.fft_t(A_gpu)           # ← still compute FFT on GPU here
    A_w  = sim.fft_t(A_gpu)[1].detach().cpu().numpy()
    '''
    #S_w  = np.sum(np.abs( A_w)**2, axis=(0,1)) * dx * dy  # |A|^2 vs ω
    
    
    # ===== 2D transverse intensity ==================================
    I_xy = np.trapezoid(I,T, axis=2)  # integrate over t

    torch.cuda.empty_cache()

    # ===== 2D k-space intensity =====================================
    '''
    _, _, A_k = sim.fft_xy(A_gpu)
    A_k  = A_k.detach().cpu().numpy()
    I_kxky = np.trapezoid(np.abs(A_k)**2, T, axis=2)
    '''
    

    # ===== 1D transverse profile (x) ================================
    idx_y0=np.argmin(np.abs(y))
    I_x = I_xy[:,idx_y0]  # |A|^2 line at y≈0

    # ===== 1D k-space profile (kx) ==================================
    idx_ky0=np.argmin(np.abs(Ky[0, :]))
    I_kx = I_kxky[:,idx_ky0]

    # ----------------------------------------------------------------
    # Figure layout  --------------------------------------------------
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

    # ---- 2-D spatial map ------------------------------------------
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
    ax_xy.set_title("Spatial intensity (t-integrated)")
    # zoom spatial axes
    x_lim = _auto_limits(x, I_x, zoom_space)
    y_proj = np.trapezoid(I_xy, x, axis=0)
    y_lim = _auto_limits(y, y_proj, zoom_space)
    ax_xy.set_xlim(x_lim)
    ax_xy.set_ylim(y_lim)

    # ---- 2-D k-space map -----------------------------------------
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
    ax_k.set_title("Spatial spectrum (t-integrated)")
    Kx_lim = _auto_limits(Kx[:, 0], I_kx, zoom_space)
    ky_proj = np.trapezoid(I_kxky, Kx[:, 0], axis=0)
    Ky_lim = _auto_limits(Ky[0, :], ky_proj, zoom_space)
    ax_k.set_xlim(Kx_lim)
    ax_k.set_ylim(Ky_lim)


    # ---- 1-D spatial profile (x) ----------------------------------
    ax_x.plot(x, I_x)
    ax_x.set_xlabel("x  [m]")
    ax_x.set_ylabel("|A|² at y≈0  [arb]")
    ax_x.set_title("Line profile at y≈0")
    ax_x.set_xlim(x_lim)

    # ---- 1-D k-space profile (kx) ---------------------------------
    ax_kx.plot(Kx[:, 0], I_kx)
    ax_kx.set_xlabel("kₓ  [m⁻¹]")
    ax_kx.set_ylabel("∫|Â|² dky dω  [arb]")
    ax_kx.set_title("k-space line (integrated over ky,ω)")
    ax_kx.set_xlim(Kx_lim)

    fig.tight_layout()

    # === NEW: optionally save (600 dpi) =============================
    if save_path is not None:
        from pathlib import Path
        p = Path(str(save_path))
        # If a directory or ends with a separator, use a default filename
        if (p.exists() and p.is_dir()) or str(save_path).endswith(("/", "\\")):
            p = p / "visualize_simulator.png"
        # Add .png if no suffix
        if p.suffix == "":
            p = p.with_suffix(".png")
        # Create parent directory
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=600, bbox_inches='tight')

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
        # Explicit per-layer accumulation: L += ΔL_j * M_j
        Lxy.add_(layer.mask, alpha=float(layer.dL))
    return Lxy


import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

@torch.inference_mode()
def visualize_film(sim, *, crop_factor: float = 2.0, line_axis: str = 'x', save_path=None):
    """
    Left: thickness map L(x,y)
    Middle: center-line thickness (default y=0; line_axis='y' plots the vertical cut at x=0)
    Right-1: center-line theoretical group delay vs pure air
             Δt(r) = [β1_film - β1_ref] * L(r) (fs)
    Right-2: center-line phase vs pure air
             Δφ(r) = [β0_film - β0_ref] * L(r) (rad, wrapped to (-π, π])
    """
    assert sim.stepped_film is not None, "sim.stepped_film is None"

    # 1) Thickness map
    Lxy = film_thickness_map(sim)                 # (Nx,Ny) [m]

    # 2) Grid and center line
    x = sim.x.detach().cpu().numpy()
    y = sim.y.detach().cpu().numpy()
    L = Lxy.detach().cpu().numpy()

    if line_axis.lower() == 'x':
        iy0 = int(torch.argmin(torch.abs(sim.y)).item())   # y≈0
        x_line = x
        L_line = L[:, iy0]
        xlabel = "x [m]"
    else:
        ix0 = int(torch.argmin(torch.abs(sim.x)).item())   # x≈0
        x_line = y
        L_line = L[ix0, :]
        xlabel = "y [m]"

    # 3) View range
    try:
        w = sim._infer_beam_radius()
    except Exception:
        w = None
    r_grid = float(min(np.max(np.abs(x)), np.max(np.abs(y))))
    lim = float(min(crop_factor * w, 0.98 * r_grid)) if (w is not None and w > 0) else 0.8 * r_grid

    # === Theoretical group delay (vs pure air) ======================
    beta1_film = float(sim.materials["beta_film"][1])   # s/m
    beta1_ref  = float(sim.materials["beta_ref" ][1])   # s/m
    delta_beta1 = beta1_film - beta1_ref                # s/m
    tau_line_fs = (delta_beta1 * L_line) * 1e15         # fs

    # === Phase difference (vs pure air) =============================
    # Key fix: β0 comes directly from sim.k0 and sim.k0_ref (do not take from materials['beta_*'][0])
    delta_beta0 = float(sim.k0 - sim.k0_ref)            # [rad/m]
    dphi_line   = delta_beta0 * L_line                  # [rad]
    # Wrap to (-π, π] — using angle() is robust
    dphi_wrapped = np.angle(np.exp(1j * dphi_line))

    # 4) Plot: four panels
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 4))

    # Left: L(x,y)
    im = ax1.imshow(
        L, origin="lower",
        extent=[x.min(), x.max(), y.min(), y.max()],
        aspect="equal"
    )
    ax1.set_title("Film thickness $L(x,y)$ [m]")
    fig.colorbar(im, ax=ax1)
    ax1.set_xlabel("x [m]"); ax1.set_ylabel("y [m]")
    ax1.set_xlim(-lim, lim); ax1.set_ylim(-lim, lim)

    # Middle: center-line thickness (mm)
    ax2.plot(x_line, L_line * 1e3)
    ax2.set_xlabel(xlabel); ax2.set_ylabel("Thickness [mm]")
    ax2.set_title("Center-line thickness")
    ax2.set_xlim(-lim, lim)
    ax2.grid(True)

    # Right-1: center-line group delay (fs)
    ax3.plot(x_line, tau_line_fs)
    ax3.set_xlabel(xlabel); ax3.set_ylabel(r"$\Delta t$ [fs]  (vs air)")
    ax3.set_title("Center-line group delay (theory)")
    ax3.set_xlim(-lim, lim)
    ax3.grid(True)

    # Right-2: center-line phase (rad, wrapped) — label horizontal axis as r
    ax4.plot(x_line, dphi_wrapped)
    ax4.set_xlabel("r [m]")  # per your request label as r
    ax4.set_ylabel(r"$\Delta\phi$ [rad]  (vs air, wrapped)")
    ax4.set_title("Center-line phase (theory)")
    ax4.set_xlim(-lim, lim)
    ax4.grid(True)

    fig.tight_layout()

    # Save (600 dpi)
    if save_path is not None:
        p = Path(str(save_path))
        if (p.exists() and p.is_dir()) or str(save_path).endswith(("/", "\\")):
            p = p / "visualize_film.png"
        if p.suffix == "":
            p = p.with_suffix(".png")
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=600, bbox_inches='tight')

    plt.show()





import math
import torch
import numpy as np
import matplotlib.pyplot as plt

@torch.inference_mode()
def visualize_mirror(
    sim,
    Hxy: torch.Tensor | None = None,
    *,
    # If Hxy is not provided, you can build directly from CSV:
    csv_path: str | None = None,
    col_range: str = "r range (mm)",
    col_height: str = "d_abs_final (µm)",
    dr_fwhm: float = 40e-6,
    aperture: str | float | None = "3w",
    # Optical parameters
    theta_deg: float = 0.0,   # incidence angle (degrees)
    n0: float = 1.0,          # refractive index of propagation medium (air ≈ 1)
    # Plotting options
    crop_factor: float = 2.0, # view ~ ±(crop_factor × w)
    line_axis: str = "x",     # 'x' → center line at y≈0; 'y' → cut at x≈0
    save_path=None            # NEW: path to save the figure; None = do not save
):
    """
    Visualize the mirror step height H(x,y) and the group delay Δt(x,y) caused by geometric path length.

    Formulae:
        L(x,y) = 2 cosθ · H(x,y)
        φ0(x,y) = (n0·ω0/c) · L(x,y)
        Δt(x,y) = (n0/c) · L(x,y) = (2 n0 cosθ / c) · H(x,y)

    Notes:
      • If Hxy (in meters) is already built externally, pass it directly;
        otherwise provide csv_path and this function will call build_mirror_height_map().
    """
    # 1) Prepare H(x,y)
    if Hxy is None:
        assert csv_path is not None, "When Hxy is not provided, csv_path must be set to read the mirror-steps CSV."
        Hxy = build_mirror_height_map(sim,
                                      csv_path=csv_path,
                                      col_range=col_range,
                                      col_height=col_height,
                                      dr_fwhm=dr_fwhm,
                                      aperture=aperture)
    Hxy = Hxy.to(sim.X.device, dtype=torch.float32)
    H   = Hxy.detach().cpu().numpy()  # [m]

    # 2) Grid coordinates
    x = sim.x.detach().cpu().numpy()
    y = sim.y.detach().cpu().numpy()

    # 3) Choose center line
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

    # 4) Geometric path and group delay on the center line
    cos_theta   = math.cos(math.radians(theta_deg))
    L_line      = 2.0 * cos_theta * H_line                # [m]
    tau_line_s  = (n0 / sim.c) * L_line                   # [s]
    tau_line_fs = tau_line_s * 1e15                        # [fs]

    # 5) View range (keep consistent with visualize_film where possible)
    try:
        w = sim._infer_beam_radius()
    except Exception:
        w = None
    r_grid = float(min(np.max(np.abs(x)), np.max(np.abs(y))))
    lim = float(min(crop_factor * w, 0.98 * r_grid)) if (w is not None and w > 0) else 0.8 * r_grid

    # 6) Three-panel figure: H(x,y), center-line H, center-line Δt
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    # Left: H(x,y) (colorbar in µm is more intuitive)
    im = ax1.imshow(
        H * 1e6, origin="lower",
        extent=[x.min(), x.max(), y.min(), y.max()],
        aspect="equal"
    )
    ax1.set_title("Mirror height $H(x,y)$ [µm]")
    fig.colorbar(im, ax=ax1, label="H [µm]")
    ax1.set_xlabel("x [m]"); ax1.set_ylabel("y [m]")
    ax1.set_xlim(-lim, lim); ax1.set_ylim(-lim, lim)

    # Middle: center-line height (µm)
    ax2.plot(coord, H_line * 1e6)
    ax2.set_xlabel(xlabel); ax2.set_ylabel("Height [µm]")
    ax2.set_title("Center-line height")
    ax2.set_xlim(-lim, lim); ax2.grid(True)

    # Right: center-line group delay (fs)
    ax3.plot(coord, tau_line_fs)
    ax3.set_xlabel(xlabel); ax3.set_ylabel("Δt [fs]  (vs flat mirror)")
    ax3.set_title(f"Center-line group delay (n0={n0:g}, θ={theta_deg:.1f}°)")
    ax3.set_xlim(-lim, lim); ax3.grid(True)

    fig.tight_layout()

    # === NEW: optionally save (600 dpi) ===
    if save_path is not None:
        from pathlib import Path
        p = Path(str(save_path))
        # If a directory or ends with a separator, use a default filename
        if (p.exists() and p.is_dir()) or str(save_path).endswith(("/", "\\")):
            p = p / "visualize_mirror.png"
        # Add .png if no suffix
        if p.suffix == "":
            p = p.with_suffix(".png")
        # Create parent directory
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=600, bbox_inches='tight')

    plt.show()

    # Optional return for further processing
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
    """Print GPU VRAM + CPU RAM usage."""
    # ---------------- GPU ----------------
    props      = torch.cuda.get_device_properties(gpu_id)
    total_mem  = props.total_memory
    reserved   = torch.cuda.memory_reserved(gpu_id)
    allocated  = torch.cuda.memory_allocated(gpu_id)

    free_in_pool = reserved - allocated
    gpu_avail    = total_mem - reserved

    print("======= GPU memory (device", gpu_id, ") =======")
    print(f"Total       : {total_mem/1024**2:8.1f} MB")
    print(f"Allocated   : {allocated/1024**2:8.1f} MB")
    print(f"Reserved    : {reserved/1024**2:8.1f} MB "
          f"(free in pool: {free_in_pool/1024**2:8.1f} MB)")
    print(f"Still avail : {gpu_avail/1024**2:8.1f} MB")

    # ---------------- CPU ----------------
    vm = psutil.virtual_memory()
    total_ram = vm.total
    avail_ram = vm.available

    proc = psutil.Process(os.getpid())
    rss  = proc.memory_info().rss          # Resident Set Size (bytes)

    print("\n======= CPU memory =========")
    print(f"Total RAM    : {total_ram/1024**3:6.2f} GB")
    print(f"Available RAM: {avail_ram/1024**3:6.2f} GB")
    print(f"Process RSS  : {rss/1024**2:6.1f} MB")
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




# =====================================================================
# 1) Mirror: reflective echelon with β0 & β1 only (GPU-friendly)
# =====================================================================

@torch.inference_mode()
def apply_reflective_echelon_beta01(
    A_in, sim, Hxy,
    *,
    n0: float = 1.0,             # mirror medium (air=1)
    theta_deg: float = 0.0,      # incidence angle
    amplitude_r: float = 1.0,    # amplitude reflectivity (|r|^2 = R)
    phi_const: float = np.pi,    # constant reflection phase
    omega_chunk: int = 1,        # process this many freq bins at once (>=1)
    tile_x: int | None = None    # optional x-tiling for low VRAM
):
    """
    Apply reflective echelon with only β0 and β1 contributions:

        Â_out = r · Â_in · exp{i[(β0 + β1Ω) · L(x,y) + φ_const]},
        L(x,y) = 2 cosθ · H(x,y),  β0 = n0 ω0 / c,  β1 = n0 / c.

    Low-memory design:
      • β0 uses a single 2-D phase (broadcasted)
      • β1Ω uses freq-chunk loop to avoid building a 3-D phase
      • Peak VRAM ≈ one spectral field + one temp field + one 2-D phase
    """
    dev = sim.T.device

    # 0) ensure input on GPU (complex64)
    A = (A_in if torch.is_tensor(A_in) else torch.as_tensor(A_in))
    A = A.to(dev, dtype=torch.complex64, non_blocking=True)

    # 1) optical path L(x,y) (float32)
    cos_theta = math.cos(math.radians(theta_deg))
    Lxy = (2.0 * cos_theta) * Hxy.to(dev, dtype=torch.float32, non_blocking=True)

    # 2) FFT_t → A_w
    _, A_w = sim.fft_t(A)
    del A

    # 3) global amplitude & constant phase
    r_phase = complex(float(amplitude_r * math.cos(phi_const)),
                      float(amplitude_r * math.sin(phi_const)))
    A_w.mul_(r_phase)

    # 4) β0 · L(x,y) (single 2D phase)
    beta0 = float(n0 * sim.omega0 / sim.c)            # [rad/m]
    phi0_xy = beta0 * Lxy                             # (Nx,Ny) float32
    phase0_xy = torch.complex(torch.cos(phi0_xy), torch.sin(phi0_xy))
    A_w.mul_(phase0_xy[..., None])
    del phi0_xy, phase0_xy

    # 5) β1 · L(x,y) · Ω — chunk by frequency and (optionally) by x
    beta1 = float(n0 / sim.c)                         # [s/m]
    omega = sim.omega.to(dev)                         # (Nt,)  **ensure device**

    Nx = sim.Nx
    Ny = sim.Ny
    tx = Nx if tile_x is None else int(tile_x)
    oc = max(1, int(omega_chunk))

    for xs in range(0, Nx, tx):
        xe = min(xs + tx, Nx)
        L_blk = Lxy[xs:xe, :]                         # (tx,Ny) float32

        if oc == 1:
            # most memory-lean: one ω per multiply
            for k in range(omega.numel()):
                theta_xy = -(beta1 * float(omega[k])) * L_blk
                phase_xy = torch.complex(torch.cos(theta_xy), torch.sin(theta_xy))
                A_w[xs:xe, :, k].mul_(phase_xy)
        else:
            # faster: small ω-chunks
            for k0 in range(0, omega.numel(), oc):
                k1 = min(k0 + oc, omega.numel())
                om = omega[k0:k1]                                   # (oc,)
                theta = (beta1 * L_blk[..., None]) * om[None, None, :]
                phase = torch.complex(torch.cos(theta), torch.sin(theta))
                A_w[xs:xe, :, k0:k1] *= phase
                del theta, phase

    del Lxy

    # 6) back to time domain
    _, A_after = sim.ifft_t(A_w)
    del A_w

    # optional: free caller's output slot to lower peak mem in pipelines
    sim.A_out = None
    torch.cuda.empty_cache()

    return A_after.detach().cpu()


# =====================================================================
# 2) GDD compression utilities (lean scan + one final compress)
# =====================================================================

def _rms_width(x: torch.Tensor, y: torch.Tensor):
    y = torch.abs(y)
    norm = torch.trapz(y, x)
    if norm == 0:
        return 0.0
    mean = torch.trapz(x * y, x) / norm
    var  = torch.trapz((x - mean) ** 2 * y, x) / norm
    return torch.sqrt(var)


def _fwhm(x, y):
    """
    FWHM for 1-D profile (torch or numpy). Handles FFT wraparound peaks.
    """
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
    idx = torch.nonzero(y >= half, as_tuple=False).flatten()
    if idx.numel() < 2:
        return 0.0

    wraps = (idx[0] == 0) and (idx[-1] == len(y) - 1)
    if wraps:
        peak = torch.argmax(y).item()
        shift = len(y)//2 - peak
        y = torch.roll(y, shifts=shift, dims=0)
        x = torch.roll(x, shifts=shift, dims=0)
        idx = torch.nonzero(y >= half, as_tuple=False).flatten()

    def _interp(i_left: int):
        i_right = i_left + 1
        if i_right >= len(x):
            return x[i_left].item()
        x0, x1 = x[i_left], x[i_right]
        y0, y1 = y[i_left], y[i_right]
        return (x0 + (half - y0) * (x1 - x0) / (y1 - y0)).item()

    i0, i1 = idx[0], idx[-1]
    x_l = _interp(i0-1 if i0 > 0 else i0)
    x_r = _interp(i1)
    return x_r - x_l


def _ensure_dev(x: torch.Tensor, dev):
    return x.to(dev, non_blocking=True) if x.device != dev else x


def _time_profile(A, sim):
    A = _ensure_dev(A, sim.T.device)
    I_t = torch.sum(torch.abs(A)**2, dim=(0, 1)) * sim.dx * sim.dy
    return I_t


def _spectrum(A, sim):
    A = _ensure_dev(A, sim.T.device)
    _, A_w = sim.fft_t(A)
    S_w = torch.sum(torch.abs(A_w)**2, dim=(0, 1)) * sim.dx * sim.dy
    return S_w.detach().cpu().numpy()


@torch.inference_mode()
def compress_with_gdd(A_out, sim, D: float, *, return_field: bool = True):
    """
    Apply quadratic spectral phase exp{i·(D/2)·ω²} and compute metrics.
    """
    dev = sim.T.device
    ω = sim.omega.to(dev)                     # **ensure same device**

    A_gpu = (A_out if torch.is_tensor(A_out) else torch.as_tensor(A_out)).to(dev)

    # spectral domain
    _, A_w = sim.fft_t(A_gpu)
    del A_gpu
    phase = torch.exp(0.5j * D * ω**2)[None, None, :]
    A_w_D = A_w * phase
    del A_w

    # spectral metrics
    I_w_D = torch.sum(torch.abs(A_w_D)**2, dim=(0, 1)) * sim.dx * sim.dy
    sigma_w = _rms_width(ω, I_w_D)

    # back to time for temporal metrics
    _, A_t_D = sim.ifft_t(A_w_D)
    del A_w_D
    P_t = _time_profile(A_t_D, sim)

    sigma_t = _rms_width(sim.T, P_t)
    fwhm_t  = _fwhm(sim.T, P_t)
    tbp = (sigma_t * sigma_w).item()

    out = dict(tbp=tbp, fwhm_t=fwhm_t, sigma_w=sigma_w.item())
    if return_field:
        out["A_comp"] = A_t_D.detach().cpu()
    return out


def find_best_gdd(A_out, sim, D_scan: np.ndarray):
    """
    Return (D_best, tbp_min, tbp_curve) without keeping any time-domain fields.
    """
    best_tbp = float("inf")
    best_D = float(D_scan[0])
    tbp_curve: list[float] = []

    for D in D_scan:
        res = compress_with_gdd(A_out, sim, float(D), return_field=False)
        tbp_curve.append(res["tbp"])
        if res["tbp"] < best_tbp:
            best_tbp = res["tbp"]
            best_D = float(D)

    return best_D, best_tbp, tbp_curve


# =====================================================================
# 3) Aperture design (uses soft_unit_step from your utilities)
# =====================================================================

@torch.inference_mode()
def design_circular_aperture_for_fraction(
    A_field: torch.Tensor, sim,
    *,
    pass_fraction: float,       # e.g. 0.85
    fwhm: float = 40e-6,        # soft edge FWHM (m)
    tol: float = 1e-3,
    max_iter: int = 40
):
    """
    Find R so that soft circular stop M_R(x,y)=soft_unit_step(R-r,fwhm) passes
    target energy fraction. Returns (M_xy, R_opt, frac_achieved).
    """
    assert 0.0 < pass_fraction <= 1.0, "pass_fraction must be in (0,1]"

    dev = sim.T.device
    A = (A_field if torch.is_tensor(A_field) else torch.as_tensor(A_field)).to(
        dev, dtype=torch.complex64, non_blocking=True
    )
    I_xy = torch.sum(A.real*A.real + A.imag*A.imag, dim=2) * float(sim.dT)
    del A; torch.cuda.empty_cache()

    denom = torch.sum(I_xy)
    if denom.item() <= 0:
        M_full = torch.ones((sim.Nx, sim.Ny), device=dev, dtype=torch.float32)
        Rmax = float(min(sim.x.abs().max(), sim.y.abs().max()))
        return M_full, Rmax, 1.0

    r = torch.sqrt(sim.X.to(dev)**2 + sim.Y.to(dev)**2)

    def frac_of_R(R: float) -> float:
        M = soft_unit_step(R - r, fwhm)
        num = torch.sum(I_xy * M)
        return float((num / denom).item())

    if pass_fraction >= 1.0 - 1e-6:
        M_full = soft_unit_step((r.max().item()*2) - r, fwhm)
        return M_full, float(r.max().item()), 1.0

    lo, hi = 0.0, float(r.max().item())
    if frac_of_R(hi) < pass_fraction:
        M_hi = soft_unit_step(hi - r, fwhm)
        return M_hi, hi, frac_of_R(hi)

    R, frac_mid = hi, frac_of_R(hi)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = frac_of_R(mid)
        if fmid < pass_fraction:
            lo = mid
        else:
            hi = mid
        R, frac_mid = hi, fmid
        if abs(fmid - pass_fraction) <= tol * pass_fraction:
            break

    M_xy = soft_unit_step(R - r, fwhm).to(torch.float32)
    return M_xy, R, frac_mid


# =====================================================================
# 4) High-level: aperture + GDD scan + final compress (optional plots)
# =====================================================================

def aperture_and_compress(
    sim,
    *,
    D_min: float = -3e-27,
    D_max: float = 0.0,
    N_scan: int = 401,
    plot: bool = True,
    return_field: bool = True,
    # aperture options
    aperture_pass_fraction: float | None = None,
    aperture_fwhm: float = 40e-6,
    return_aperture_mask: bool = False,
    save_path: str | os.PathLike | None = None,
):
    """
    Optionally apply a soft circular aperture (by desired energy fraction),
    scan GDD to minimize TBP, then do one final compression at the optimum.
    """
    if sim.A_out is None:
        raise ValueError("sim.A_out is None — run sim.propagate() first.")

    A_out_cpu = sim.A_out
    sim.A_out = None
    torch.cuda.empty_cache()

    # (A) optional aperture
    aperture_info = None
    if aperture_pass_fraction is not None:
        M_xy, R_opt, frac_ach = design_circular_aperture_for_fraction(
            A_out_cpu, sim,
            pass_fraction=float(aperture_pass_fraction),
            fwhm=float(aperture_fwhm),
            tol=1e-3, max_iter=40,
        )
        A_base_gpu = (A_out_cpu if torch.is_tensor(A_out_cpu) else torch.as_tensor(A_out_cpu)).to(
            sim.T.device, dtype=torch.complex64
        )
        A_base_gpu = A_base_gpu * M_xy[..., None].to(A_base_gpu.dtype)
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
        A_base = A_out_cpu

    # (B) GDD scan
    D_scan = np.linspace(D_min, D_max, int(N_scan))
    D_best, tbp_min, tbp_curve = find_best_gdd(A_base, sim, D_scan)

    # diagnostics of input (optional, for plotting)
    if plot:
        S_w_in   = _spectrum(sim.A_in, sim);     S_w_in_c   = np.fft.fftshift(S_w_in)
        P_t_in   = _time_profile(sim.A_in, sim).cpu().numpy()

    # (C) final compress at best D
    final = compress_with_gdd(A_base, sim, D_best, return_field=return_field)
    torch.cuda.empty_cache()

    print(
        f"FWHM_t(best): {final['fwhm_t']*1e15:.2f} fs  "
        f"(GDD={D_best:.3e} s²,  TBP={final['tbp']:.3f})"
    )
    if aperture_info is not None:
        print(f"[Aperture] target={aperture_info['pass_target']*100:.1f}% "
              f"→ achieved={aperture_info['pass_achieved']*100:.1f}% , "
              f"R≈{aperture_info['radius_m']*1e3:.2f} mm,  FWHM={aperture_info['fwhm_m']*1e6:.1f} µm")

    # (D) plots
    if plot:
        omega_c = torch.fft.fftshift(sim.omega).cpu().numpy()
        freq_THz_c = omega_c / (2*np.pi*1e12)

        S_w_base   = _spectrum(A_base, sim);     S_w_base_c = np.fft.fftshift(S_w_base)
        S_w_comp_c = None
        if return_field:
            S_w_comp   = _spectrum(final["A_comp"], sim)
            S_w_comp_c = np.fft.fftshift(S_w_comp)

        T_fs   = (sim.T * 1e15).cpu().numpy()
        P_t_in = _time_profile(sim.A_in, sim).cpu().numpy()
        P_t_base = _time_profile(A_base, sim).cpu().numpy()
        P_t_comp = _time_profile(final["A_comp"], sim).cpu().numpy() if return_field else None

        fwhm_in  = _fwhm(sim.T, torch.as_tensor(P_t_in))
        half_tr  = min(5 * fwhm_in * 1e15, abs(T_fs[-1]))
        half_fr  = 2 * _fwhm(torch.as_tensor(omega_c), torch.as_tensor(np.fft.fftshift(S_w_in))) / (2*np.pi*1e12)

        fig, axes = plt.subplots(4, 1, figsize=(6, 12))

        # (0) time domain
        axes[0].plot(T_fs, P_t_in,   label="Input")
        axes[0].plot(T_fs, P_t_base, label="After aperture" if aperture_info else "Output")
        if P_t_comp is not None:
            axes[0].plot(T_fs, P_t_comp, ls="--", label="Compressed")
        axes[0].set_xlim(-half_tr, half_tr)
        axes[0].set(xlabel="Time [fs]", ylabel="∫|A|² dx dy  [J/s]", title="Temporal power")
        axes[0].legend(); axes[0].grid()

        # (1) spectrum (normalized)
        axes[1].plot(freq_THz_c, np.fft.fftshift(S_w_in)   / max(S_w_in.max(),   1e-30), label="Before (input)")
        axes[1].plot(freq_THz_c, S_w_base_c                / max(S_w_base_c.max(),1e-30),
                     label="After aperture" if aperture_info else "Before (output)")
        if S_w_comp_c is not None:
            axes[1].plot(freq_THz_c, S_w_comp_c / max(S_w_comp_c.max(), 1e-30), ls="--", label="After compression")
        axes[1].set_xlim(-half_fr, half_fr)
        axes[1].set(xlabel="Frequency offset [THz]", ylabel="Norm. power", title="Spectrum")
        axes[1].grid()

        # (2) TBP vs GDD
        axes[2].plot(D_scan*1e28, tbp_curve)
        axes[2].scatter(D_best*1e28, tbp_min, color="red")
        axes[2].axhline(0.5, ls="--", color="gray")
        axes[2].set(xlabel="GDD [×10⁻²⁸ s²]", ylabel="TBP", title="GDD scan")
        axes[2].grid()

        # (3) before vs after (time)
        axes[3].plot(T_fs, P_t_base, label="Before (after aperture)" if aperture_info else "Before")
        if return_field:
            axes[3].plot(T_fs, P_t_comp, ls="--", label="After")
        axes[3].set(xlabel="Time [fs]", ylabel="∫|A|² dx dy  [J/s]",
                    title="Temporal profile (before vs after)")
        axes[3].legend(); axes[3].grid()

        fig.tight_layout()

        if save_path is not None:
            p = Path(str(save_path))
            if (p.exists() and p.is_dir()) or str(save_path).endswith(("/", "\\")):
                p = p / "aperture_and_compress.png"
            if p.suffix == "":
                p = p.with_suffix(".png")
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=600, bbox_inches='tight')

        plt.show()

    # (E) results
    result = dict(D_opt=D_best, tbp=final["tbp"], fwhm_t=final["fwhm_t"],
                  sigma_w=final["sigma_w"], tbp_curve=tbp_curve)
    if return_field:
        result["A_comp"] = final["A_comp"]
    if aperture_info is not None:
        result["aperture"] = aperture_info
        if return_aperture_mask:
            result["aperture"]["mask_xy"] = M_xy.detach().cpu()  # from inner scope

    # restore
    sim.A_out = A_out_cpu
    return result


# =====================================================================
# 5) Quick saves / diagnostics / sweeps
# =====================================================================

def save_A_every_step(step, sim, save_dir="A_dump", every=1):
    if step % every:
        return
    os.makedirs(save_dir, exist_ok=True)
    fname = os.path.join(save_dir, f"A_{step:05d}.pt")
    torch.save(sim.A.detach(), fname)


@torch.inference_mode()
def diagnose_and_print(sim, *, field="current", label=None):
    """
    Print: Energy, Peak power, FWHM/σ_t, FWHM/σ_v, FWHM_x/y, centers, B-integral.
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

        I = torch.abs(A_gpu)**2
        I_t = torch.trapz(I, sim.T)                      # integrate over t
        E = torch.trapz(torch.trapz(I_t, sim.y), sim.x).item()

        I_xy = torch.trapz(I, sim.T, dim=2)
        P_t  = torch.sum(I, dim=(0, 1)) * sim.dx * sim.dy
        del I

        P_peak = P_t.max().item()
        sigma_t = _rms_width(sim.T, P_t).item()
        fwhm_t  = _fwhm(sim.T, P_t)
        t_peak  = sim.T[torch.argmax(P_t)].item() if P_peak > 0 else 0.0
        denom   = torch.trapz(P_t, sim.T)
        t_cm    = (torch.trapz(sim.T * P_t, sim.T) / denom).item() if float(denom) != 0.0 else 0.0

        _, A_w = sim.fft_t(A_gpu)
        A_w_np = A_w.detach().cpu().numpy()
        I_w = np.sum(np.abs(A_w_np)**2, axis=(0, 1))
        S_w = torch.from_numpy(I_w).to(dev) * sim.dx * sim.dy
        sigma_w = _rms_width(sim.omega.to(dev), S_w).item()
        fwhm_w  = _fwhm(sim.omega.to(dev), S_w)

        iy0 = torch.argmin(torch.abs(sim.y)).item()
        ix0 = torch.argmin(torch.abs(sim.x)).item()
        fwhm_x = float(_fwhm(sim.x, I_xy[:, iy0]))
        fwhm_y = float(_fwhm(sim.y, I_xy[ix0,  :]))

        B = getattr(sim, "B_running", float("nan"))

    name = label or field.upper()
    print(f"\n===== Diagnostics [{name}] =====")
    print(f"Energy                 : {E:.4e} J")
    print(f"Peak power             : {P_peak/1e6:.2f} MW")
    print(f"Time  — FWHM / σ_t     : {fwhm_t*1e15:.2f} fs / {sigma_t*1e15:.2f} fs")
    print(f"Freq  — FWHM / σ_v     : {fwhm_w/2/np.pi/1e12:.2f} THz / {sigma_w/2/np.pi/1e12:.2f} THz")
    print(f"Space — FWHM_x / FWHM_y: {fwhm_x*1e3:.2f} mm / {fwhm_y*1e3:.2f} mm")
    print(f"Centers (t_peak / t_cm): {t_peak*1e15:.2f} fs / {t_cm*1e15:.2f} fs")
    if field != "input":
        print(f"B-integral (running)   : {B:.3f} (approx.)")
    print("================================\n")


def sweep_energy_clip_and_plot(sim,
                               passes: Optional[Sequence[float]] = None,
                               D_min: float = -3e-28, D_max: float = 0.0,
                               N_scan: int = 201,
                               aperture_fwhm: float = 30e-6,
                               save: bool = True,
                               save_dir: str | os.PathLike | None = None,
                               filename_prefix: str = "fwhm_vs_energy_after_mirror"):
    """
    For each energy pass fraction:
      1) design soft circular aperture for that fraction
      2) scan GDD, find min TBP
      3) record compressed FWHM_t (fs)
    """
    if passes is None:
        passes = [p/100 for p in range(100, 10-1, -2)]   # 100% → 10% step -2%

    fwhm_fs = []
    for pf in passes:
        res = aperture_and_compress(
            sim,
            D_min=D_min, D_max=D_max, N_scan=N_scan,
            plot=False, return_field=False,
            aperture_pass_fraction=pf,
            aperture_fwhm=aperture_fwhm,
            return_aperture_mask=False
        )
        fwhm_fs.append(res["fwhm_t"] * 1e15)

        # optionally release FFT plans if you have such a helper
        try:
            release_fft_pool()  # noqa: F821
        except Exception:
            torch.cuda.empty_cache()

    # plot
    x_pct = [p*100 for p in passes]
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.plot(x_pct, fwhm_fs, marker='o')
    ax.set_xlabel("Energy pass [%]")
    ax.set_ylabel("Compressed FWHM [fs]")
    ax.set_title("FWHM vs Energy Pass")
    ax.grid(True)
    fig.tight_layout()
    plt.show()

    # save
    if save:
        out_dir = save_dir or "."
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(out_dir, f"{filename_prefix}_{ts}.csv")
        npz_path = os.path.join(out_dir, f"{filename_prefix}_{ts}.npz")
        fig_path = os.path.join(out_dir, f"{filename_prefix}_{ts}.png")

        arr = np.column_stack([np.array(x_pct, dtype=float),
                               np.array(fwhm_fs, dtype=float)])
        np.savetxt(csv_path, arr, delimiter=",",
                   header="energy_pass_pct,fwhm_fs", comments="")
        np.savez(npz_path,
                 energy_pass=np.array(passes, dtype=float),
                 energy_pass_pct=np.array(x_pct, dtype=float),
                 fwhm_fs=np.array(fwhm_fs, dtype=float))
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        print(f"[saved] CSV: {csv_path}\n[saved] NPZ: {npz_path}\n[saved] FIG: {fig_path}")

    return np.array(passes), np.array(fwhm_fs)
