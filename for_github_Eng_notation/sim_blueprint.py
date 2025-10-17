# sim_blueprint.py
import json, os, torch, numpy as np, pathlib

def _to_jsonable(obj):
    # Convert numpy/torch/dict keys etc. into JSON-serializable types
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            k2 = str(k)  # dispersion keys are int; convert to string for storage
            out[k2] = _to_jsonable(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj

def save_sim_blueprint(layer_dir, COMMON_PARAMS, beam_profile, sim=None, save_A_in=True):
    """
    Save the following into layer_dir:
      - common_params.json
      - beam_profile_meta.json (parameters only; do not store non-serializable func)
      - grids.pt (if sim is provided)
      - A_in.pt (optional, strongly recommended)
    """
    pathlib.Path(layer_dir).mkdir(parents=True, exist_ok=True)

    # 1) Save COMMON_PARAMS
    with open(os.path.join(layer_dir, "common_params.json"), "w") as f:
        json.dump(_to_jsonable(COMMON_PARAMS), f, indent=2)

    # 2) Save a "reproducible description (meta)" of beam_profile
    meta = dict(shape=beam_profile.get("shape", "gaussian"))
    if meta["shape"] == "custom":
        # Your custom profile is polynomial-exponential; include all parameters
        # If these parameters are not present in beam_profile, attach them when constructing beam_profile
        for key in ("waist","order","rho_edge","p0","p1","p2","p3","w","type"):
            if key in beam_profile:
                meta[key] = beam_profile[key]
        # Mark func as non-serializable—just a note
        meta["note"] = "custom.func not serialisable; prefer using saved A_in.pt to import."
    else:
        # gaussian / flat / supergaussian etc. can store numeric parameters directly
        for key in ("waist","radius","order"):
            if key in beam_profile:
                meta[key] = beam_profile[key]

    with open(os.path.join(layer_dir, "beam_profile_meta.json"), "w") as f:
        json.dump(_to_jsonable(meta), f, indent=2)

    # 3) Save grids (if sim is provided)
    if sim is not None:
        torch.save({"x": sim.x.cpu(), "y": sim.y.cpu(), "T": sim.T.cpu()},
                   os.path.join(layer_dir,"grids.pt"))

    # 4) Save A_in (strongly recommended)
    if save_A_in and sim is not None and hasattr(sim, "A_in") and sim.A_in is not None:
        torch.save(sim.A_in.cpu(), os.path.join(layer_dir,"A_in.pt"))
