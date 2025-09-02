# sim_blueprint.py
import json, os, torch, numpy as np, pathlib

def _to_jsonable(obj):
    # 把 numpy / torch / 字典键等转成 JSON 可存的类型
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            k2 = str(k)  # dispersion 的键是 int，这里转成字符串存
            out[k2] = _to_jsonable(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj

def save_sim_blueprint(layer_dir, COMMON_PARAMS, beam_profile, sim=None, save_A_in=True):
    """
    在 layer_dir 保存：
      - common_params.json
      - beam_profile_meta.json（只存参数，不存不可序列化的 func）
      - grids.pt（若提供 sim）
      - A_in.pt（可选，强烈推荐）
    """
    pathlib.Path(layer_dir).mkdir(parents=True, exist_ok=True)

    # 1) 保存 COMMON_PARAMS
    with open(os.path.join(layer_dir, "common_params.json"), "w") as f:
        json.dump(_to_jsonable(COMMON_PARAMS), f, indent=2)

    # 2) 保存 beam_profile 的“可复现描述(meta)”
    meta = dict(shape=beam_profile.get("shape", "gaussian"))
    if meta["shape"] == "custom":
        # 你的自定义是多项式指数型，建议把参数都放进去
        # 若 beam_profile 里没有这些可见参数，请在你构造 beam_profile 时附带上它们
        for key in ("waist","order","rho_edge","p0","p1","p2","p3","w","type"):
            if key in beam_profile:
                meta[key] = beam_profile[key]
        # 标记 func 不可序列化，仅作提示
        meta["note"] = "custom.func not serialisable; prefer using saved A_in.pt to import."
    else:
        # gaussian / flat / supergaussian 等都可直接存数值参数
        for key in ("waist","radius","order"):
            if key in beam_profile:
                meta[key] = beam_profile[key]

    with open(os.path.join(layer_dir, "beam_profile_meta.json"), "w") as f:
        json.dump(_to_jsonable(meta), f, indent=2)

    # 3) 保存网格（若给了 sim）
    if sim is not None:
        torch.save({"x": sim.x.cpu(), "y": sim.y.cpu(), "T": sim.T.cpu()},
                   os.path.join(layer_dir,"grids.pt"))

    # 4) 保存 A_in（强烈建议）
    if save_A_in and sim is not None and hasattr(sim, "A_in") and sim.A_in is not None:
        torch.save(sim.A_in.cpu(), os.path.join(layer_dir,"A_in.pt"))
