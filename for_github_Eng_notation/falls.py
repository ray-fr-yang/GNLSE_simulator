import os, re, gc, warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_ridge_from_numpy_minmax2(
    base_path,
    film_refs,              # 第一组数组（长度 8）：对应 film 的 01..08
    mirror_refs,            # 第二组数组（长度 8）：对应 mirror 的 01..08
    amp_scale=0.8,          # 归一后再乘的幅度系数
    time_unit='fs',         # 'fs' | 'ps' | 's'
    line_alpha=0.9          # 线条透明度
):
    base = Path(base_path)
    film_dir   = base / "film"
    mirror_dir = base / "mirror"

    # ---------- 时间轴 ----------
    T = None
    tfile = base / "time_axis_T.npy"
    if tfile.exists():
        T = np.load(tfile)  # seconds

    if T is not None:
        if time_unit == 'fs':
            x = T * 1e15; xlab = "Time [fs]"
        elif time_unit == 'ps':
            x = T * 1e12; xlab = "Time [ps]"
        else:
            x = T;         xlab = "Time [s]"
    else:
        x = None;          xlab = "Sample index"

    # ---------- Tmax（film/01.npy 的峰值） ----------
    f01 = film_dir / "01.npy"
    if not f01.exists():
        raise FileNotFoundError(f"未找到 {f01}，请确认路径与文件存在。")
    Pt_01 = np.load(f01)
    Tmax = float(Pt_01.max())
    if Tmax <= 0:
        raise ValueError("film/01.npy 的最大值 Tmax ≤ 0，无法做归一化。")

    # ---------- 生成“均匀分布”的基线位置（保持与原数组相同的大小顺序） ----------
    film_refs   = np.asarray(film_refs, dtype=float)
    mirror_refs = np.asarray(mirror_refs, dtype=float)
    if film_refs.size != 8 or mirror_refs.size != 8:
        raise ValueError("film_refs 与 mirror_refs 必须各包含 8 个数（对应 01..08）。")

    def uniform_offsets_by_rank(vals, lo, hi):
        """
        vals: 原数组（用于决定相对大小顺序/名次）
        返回:
          y_by_index: 与 vals 同长，按“vals 的升序名次”映射到 [lo, hi] 的均匀点
          ticks_pos : [lo, hi] 上的均匀点（升序）
          ticks_lbl : 与 ticks_pos 一一对应的原数组“升序值”（用于刻度标签）
        """
        N = len(vals)
        # 稳定升序名次（相等时保留原次序）
        order = np.argsort(vals, kind='mergesort')
        ranks = np.empty_like(order)
        ranks[order] = np.arange(N)

        # 在 [lo, hi] 上生成 N 个等间距位置（升序）
        ticks_pos = np.linspace(lo, hi, N)

        # 映射到每个索引（01..08）的 y 基线
        y_by_index = ticks_pos[ranks]

        # 刻度标签用“原数组的升序值”
        ticks_lbl = vals[order]
        return y_by_index, ticks_pos, ticks_lbl

    # film 基线均匀分布在 [1, 2]，mirror 基线均匀分布在 [0, 1]
    y_off_film,   film_ticks_pos,   film_ticks_lbl   = uniform_offsets_by_rank(film_refs,   2.0, 1.0)
    y_off_mirror, mirror_ticks_pos, mirror_ticks_lbl = uniform_offsets_by_rank(mirror_refs, 1.0, 0.0)

    # ---------- 开始画图 ----------
    plt.figure(figsize=(10, 6))

    # 画 film：01..08
    for i in range(1, 9):
        fpath = film_dir / f"{i:02d}.npy"
        if not fpath.exists():
            print(f"[skip] 缺少 {fpath}")
            continue
        Pt = np.load(fpath)
        y = (Pt / Tmax) * amp_scale + y_off_film[i-1]

        if x is not None:
            x_use = x
            if len(x_use) != len(y):
                n = min(len(x_use), len(y))
                x_use = x_use[:n]; y = y[:n]
        else:
            x_use = np.arange(len(y))

        plt.plot(x_use, y, lw=1.5, alpha=line_alpha, color='C0', label="film" if i==1 else None)

    # 画 mirror：01..08
    for i in range(1, 9):
        mpath = mirror_dir / f"{i:02d}.npy"
        if not mpath.exists():
            print(f"[skip] 缺少 {mpath}")
            continue
        Pt = np.load(mpath)
        y = (Pt / Tmax) * amp_scale + y_off_mirror[i-1]

        if x is not None:
            x_use = x
            if len(x_use) != len(y):
                n = min(len(x_use), len(y))
                x_use = x_use[:n]; y = y[:n]
        else:
            x_use = np.arange(len(y))

        plt.plot(x_use, y, lw=1.5, alpha=line_alpha, color='C1', label="mirror" if i==1 else None)

    # ---------- y 轴刻度 ----------
    # 位置用“均匀分布”的 ticks_pos；标签用“原数组（升序）”的数值
    yticks  = list(film_ticks_pos) + list(mirror_ticks_pos)
    ylabels = [f"{v:.6g}" for v in film_ticks_lbl] + [f"{v:.6g}" for v in mirror_ticks_lbl]
    plt.yticks(yticks, ylabels)

    # y 轴范围：覆盖两组基线 + 曲线高度（给一点余量）
    y_min = min(0.0, float(np.min(mirror_ticks_pos)) - 0.05)
    y_max = max(float(np.max(film_ticks_pos)), float(np.max(mirror_ticks_pos))) + float(amp_scale) + 0.05
    plt.ylim(y_min, y_max)

    plt.xlabel(xlab)
    plt.ylabel("Offset (uniform ridge levels)")
    plt.title("Ridgeline plot (uniform baselines; labels = original values)")
    plt.legend(loc="upper right")
    plt.grid(False)
    plt.tight_layout()
    plt.show()

def plot_ridge_from_numpy_minmax(
    base_path,
    film_refs,              # 第一组数组（长度 8）：对应 film 的 01..08
    mirror_refs,            # 第二组数组（长度 8）：对应 mirror 的 01..08
    amp_scale=0.8,          # 归一后再乘的幅度系数
    time_unit='fs',         # 'fs' | 'ps' | 's'
    line_alpha=0.9          # 线条透明度
):
    base = Path(base_path)
    film_dir   = base / "film"
    mirror_dir = base / "mirror"

    # ---------- 时间轴 ----------
    T = None
    tfile = base / "time_axis_T.npy"
    if tfile.exists():
        T = np.load(tfile)  # seconds

    if T is not None:
        if time_unit == 'fs':
            x = T * 1e15; xlab = "Time [fs]"
        elif time_unit == 'ps':
            x = T * 1e12; xlab = "Time [ps]"
        else:
            x = T;         xlab = "Time [s]"
    else:
        x = None;          xlab = "Sample index"

    # ---------- Tmax（film/01.npy 的峰值） ----------
    f01 = film_dir / "01.npy"
    if not f01.exists():
        raise FileNotFoundError(f"未找到 {f01}，请确认路径与文件存在。")
    Pt_01 = np.load(f01)
    Tmax = float(Pt_01.max())
    if Tmax <= 0:
        raise ValueError("film/01.npy 的最大值 Tmax ≤ 0，无法做归一化。")

    # ---------- min–max 归一化工具 ----------
    def minmax_norm(a):
        a = np.asarray(a, dtype=float)
        a_min, a_max = float(np.min(a)), float(np.max(a))
        if a_max == a_min:
            # 退化情况：全相等，返回全 0（也可返回全 1，看你的需求）
            return np.zeros_like(a, dtype=float)
        return (a - a_min) / (a_max - a_min)

    # ---------- 基线（两组各自 min–max 归一化，然后 ×(-1)+常数） ----------
    film_refs   = np.asarray(film_refs, dtype=float)
    mirror_refs = np.asarray(mirror_refs, dtype=float)
    if film_refs.size != 8 or mirror_refs.size != 8:
        raise ValueError("film_refs 与 mirror_refs 必须各包含 8 个数（对应 01..08）。")

    film_norm   = minmax_norm(film_refs)    # ∈[0,1]
    mirror_norm = minmax_norm(mirror_refs)  # ∈[0,1]

    y_off_film   = 2.0 - film_norm    # → [1,2]
    y_off_mirror = 1.0 - mirror_norm  # → [0,1]

    # ---------- 开始画图 ----------
    plt.figure(figsize=(10, 6))

    # 画 film：01..08
    for i in range(1, 9):
        fpath = film_dir / f"{i:02d}.npy"
        if not fpath.exists():
            print(f"[skip] 缺少 {fpath}")
            continue
        Pt = np.load(fpath)
        y = (Pt / Tmax) * amp_scale + y_off_film[i-1]

        if x is not None:
            x_use = x
            if len(x_use) != len(y):
                n = min(len(x_use), len(y))
                x_use = x_use[:n]; y = y[:n]
        else:
            x_use = np.arange(len(y))

        plt.plot(x_use, y, lw=1.5, alpha=line_alpha, color='C0', label="film" if i==1 else None)

    # 画 mirror：01..08
    for i in range(1, 9):
        mpath = mirror_dir / f"{i:02d}.npy"
        if not mpath.exists():
            print(f"[skip] 缺少 {mpath}")
            continue
        Pt = np.load(mpath)
        y = (Pt / Tmax) * amp_scale + y_off_mirror[i-1]

        if x is not None:
            x_use = x
            if len(x_use) != len(y):
                n = min(len(x_use), len(y))
                x_use = x_use[:n]; y = y[:n]
        else:
            x_use = np.arange(len(y))

        plt.plot(x_use, y, lw=1.5, alpha=line_alpha, color='C1', label="mirror" if i==1 else None)

    # ---------- y 轴刻度：基线位置 + 原始数值作为标签 ----------
    yticks  = list(y_off_film) + list(y_off_mirror)
    ylabels = [f"{v:.6g}" for v in film_refs] + [f"{v:.6g}" for v in mirror_refs]
    plt.yticks(yticks, ylabels)

    plt.ylim(0, 2.5)
    plt.xlabel(xlab)
    plt.ylabel("Offset (ridge levels)")
    plt.title("Ridgeline plot (min–max normalized baselines)")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()





def plot_all_pulses(base_path, sim=None, time_unit='fs', normalize=False):
    """
    base_path : 根目录路径（含 mirror/ 和 film/ 子目录）
    sim       : 可选，PulseGNLSESimulator（若没有 time_axis_T.npy 会尝试用 sim.T）
    time_unit : 'fs' | 'ps' | 's'
    normalize : 是否把曲线归一到峰值=1
    """
    base = Path(base_path)
    # ===== 时间轴 =====
    T = None
    time_file = base / "time_axis_T.npy"
    if time_file.exists():
        T = np.load(time_file)  # seconds
    elif sim is not None:
        try:
            T = sim.T.detach().cpu().numpy()
        except Exception:
            T = None

    # 单位换算
    xlabel = "Time [s]"
    if T is not None:
        if time_unit.lower() == 'fs':
            t = T * 1e15
            xlabel = "Time [fs]"
        elif time_unit.lower() == 'ps':
            t = T * 1e12
            xlabel = "Time [ps]"
        else:
            t = T
            xlabel = "Time [s]"
    else:
        t = None  # 用样本索引

    def _sorted_npy_list(folder: Path):
        if not folder.exists():
            return []
        files = list(folder.glob("*.npy"))
        # 尝试按前缀中的数字排序（如 01.npy、08.npy）；否则按名字
        def keyfn(p: Path):
            m = re.match(r"^\D*(\d+)\D*$", p.stem)  # 抓数字
            return int(m.group(1)) if m else p.stem
        return sorted(files, key=keyfn)

    def _plot_dir(kind: str):
        folder = base / kind
        files = _sorted_npy_list(folder)
        if not files:
            print(f"[INFO] No .npy files in {folder}")
            return
        for f in files:
            Pt = np.load(f)  # shape (Nt,)
            y = Pt / max(Pt.max(), 1e-30) if normalize else Pt
            plt.figure(figsize=(6, 3.2))
            if t is not None and len(t) == len(y):
                plt.plot(t, y)
                plt.xlabel(xlabel)
            else:
                plt.plot(np.arange(len(y)), y)
                plt.xlabel("Sample index")
            plt.ylabel("P(t) [J/s]" if not normalize else "Normalized P(t)")
            plt.title(f"{kind}: {f.stem}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

    # 依次画 mirror 与 film
    _plot_dir("mirror")
    _plot_dir("film")


# ===== 用法示例 =====
# base_dir = "/rds/general/ephemeral/user/fy624/ephemeral/Sep_11th/singlepass/res1"
# plot_all_pulses(base_dir, sim=sim, time_unit='fs', normalize=False)

# =============== 安全读取：优先 weights_only=True ============================
def safe_torch_load(path: str | Path, *, map_location="cpu"):
    """
    安全读取 .pt 文件：
      • 首选：torch.load(..., weights_only=True) —— 限制到张量/容器，避免任意代码执行
      • 回退：老版本 PyTorch 不支持该参数，则退回普通 load（可选择压制 FutureWarning）
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # 老版本 PyTorch（无 weights_only 参数）
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*weights_only=False.*", category=FutureWarning)
            return torch.load(path, map_location=map_location)

# =============== 基础：把 A(x,y,t) → P(t)，按 x‑tile 降内存 ==================
def compute_Pt_tiled(A, dx: float, dy: float, tile_x: int = 64) -> np.ndarray:
    """
    A : torch.Tensor 或 numpy.ndarray，形状 (Nx, Ny, Nt)，复数
    返回: P(t) = ∬ |A|^2 dx dy  (numpy, 形状 Nt)
    说明: 按 x 方向分块，避免一次性构造 |A|^2 的 3D 大张量。
    """
    if not torch.is_tensor(A):
        A = torch.as_tensor(A)
    if A.device.type != "cpu":
        A = A.to("cpu", non_blocking=True)
    if not torch.is_complex(A):
        A = A.to(torch.complex64)

    Nx, Ny, Nt = A.shape
    Pt = torch.zeros(Nt, dtype=torch.float64)
    tx = max(1, int(tile_x))

    for xs in range(0, Nx, tx):
        xe = min(xs + tx, Nx)
        Ab = A[xs:xe, :, :]  # (tx, Ny, Nt)
        Ib = Ab.real.to(torch.float32).pow(2) + Ab.imag.to(torch.float32).pow(2)
        Pt += Ib.sum(dim=(0, 1), dtype=torch.float64)
        del Ab, Ib
        if (xs // tx) % 8 == 0:
            gc.collect()

    Pt *= (dx * dy)
    return Pt.detach().cpu().numpy()

# =============== 源发现（mirror/film），带递归兜底与日志 =====================
_MIRROR_DIR_RE = re.compile(r"^\s*(\d{2})[_\s-]*mirror[_\s-]*(\d{2})\s*$", re.IGNORECASE)

def discover_mirror_sources(base: Path) -> List[Tuple[int, Path]]:
    """
    返回 [(idx, A_out_path), ...]
    优先找 base 的直接子目录 NN_mirror_NN/A_out.pt；若没有，再递归搜。
    idx 取目录名前缀两位数字（用于保存名 01.npy 等）。
    """
    found: List[Tuple[int, Path]] = []

    # 直接子目录
    for p in base.iterdir():
        if not p.is_dir():
            continue
        m = _MIRROR_DIR_RE.match(p.name)
        if m and (p / "A_out.pt").exists():
            idx = int(m.group(1))
            found.append((idx, p / "A_out.pt"))

    if found:
        return sorted(found, key=lambda x: x[0])

    # 递归兜底
    for a in base.rglob("A_out.pt"):
        m = _MIRROR_DIR_RE.match(a.parent.name)
        if m:
            idx = int(m.group(1))
            found.append((idx, a))

    return sorted(found, key=lambda x: x[0])

def discover_film_sources(base: Path) -> List[Tuple[int, Path]]:
    """
    返回 [(idx, film_pt_path), ...]
    优先找 base 根目录下 NN_film_*.pt；若没有，再递归搜。
    idx 取文件名前两位数字。
    """
    found: List[Tuple[int, Path]] = []
    # 根层
    for f in base.glob("[0-9][0-9]_film_*.pt"):
        idx = int(f.name[:2]); found.append((idx, f))
    if found:
        return sorted(found, key=lambda x: x[0])

    # 递归兜底
    for f in base.rglob("*_film_*.pt"):
        m = re.match(r"^(\d{2})[_\s-]*film[_\s-]*.*\.pt$", f.name.strip(), re.IGNORECASE)
        if m:
            idx = int(m.group(1))
            found.append((idx, f))

    return sorted(found, key=lambda x: x[0])

# =============== 主流程：读取 → 计算 P(t) → 保存 ============================
def export_Pt_series(sim, base_path: str | Path, *, tile_x: int = 64, save_time_axis: bool = True):
    """
    sim       : PulseGNLSESimulator 实例（用于 dx, dy, T）
    base_path : 根目录（包含 01_mirror_01/、01_film_01.pt 等）
    tile_x    : 计算 P(t) 时 x 方向分块大小
    """
    base = Path(base_path).resolve()
    print(f"[INFO] base path = {base}")

    # —— 保存共用时间轴 —— #
    if save_time_axis:
        t_out = base / "time_axis_T.npy"
        np.save(t_out, sim.T.detach().cpu().numpy())
        print(f"[SAVE] time axis -> {t_out}")

    # —— Mirror —— #
    mirror_out = base / "mirror"
    mirror_out.mkdir(parents=True, exist_ok=True)
    mirrors = discover_mirror_sources(base)
    print(f"[INFO] found mirror sources: {len(mirrors)}")
    if not mirrors:
        print("[WARN] 没找到任何 mirror 源（NN_mirror_NN/A_out.pt 或递归也未命中）。")

    for idx, a_path in mirrors:
        try:
            print(f"[mirror] #{idx:02d}: loading {a_path}")
            A = safe_torch_load(a_path, map_location="cpu")
            Pt = compute_Pt_tiled(A, sim.dx, sim.dy, tile_x=tile_x)
            dst = mirror_out / f"{idx:02d}.npy"
            np.save(dst, Pt)
            print(f"[mirror] saved -> {dst.name} (Nt={Pt.shape[0]})")
            del A, Pt
            gc.collect()
        except Exception as e:
            print(f"[ERROR] mirror #{idx:02d} failed: {e}")

    # —— Film —— #
    film_out = base / "film"
    film_out.mkdir(parents=True, exist_ok=True)
    films = discover_film_sources(base)
    print(f"[INFO] found film sources: {len(films)}")
    if not films:
        print("[WARN] 没找到任何 film 源（NN_film_*.pt 或递归也未命中）。")

    for idx, f_path in films:
        try:
            print(f"[film]   #{idx:02d}: loading {f_path}")
            A = safe_torch_load(f_path, map_location="cpu")
            Pt = compute_Pt_tiled(A, sim.dx, sim.dy, tile_x=tile_x)
            dst = film_out / f"{idx:02d}.npy"
            np.save(dst, Pt)
            print(f"[film]   saved -> {dst.name} (Nt={Pt.shape[0]})")
            del A, Pt
            gc.collect()
        except Exception as e:
            print(f"[ERROR] film #{idx:02d} failed: {e}")

    print("✅ Done.")

# ===================== 使用示例 =====================
# sim = PulseGNLSESimulator(...)  # 你已构造好
# base_dir = "/path/to/the/root"  # 指向截图那一层目录
# export_Pt_series(sim, base_dir, tile_x=64, save_time_axis=True)
