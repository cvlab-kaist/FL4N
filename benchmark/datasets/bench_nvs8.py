"""8-surrounding-camera NVS (Table 2) view selection."""
import os
import numpy as np

NVS8 = os.environ.get("BENCH_NVS8", "1") != "0"
K8 = 8
N_WINDOW = 9            # cameras in the NVS window
HOLD_IDX = 4            # 0-based index of the held-out target within the window (the 5th)


def window9(pool, stride=1, start=None):
    """9 cameras at `stride` from `start`; returns (8 cond ids, held-out target id = the 5th)."""
    pool = sorted(pool)
    if start is None:
        start = pool[0]
    try:
        s0 = pool.index(start)
    except ValueError:
        s0 = 0
    idx = [s0 + stride * i for i in range(N_WINDOW)]
    if idx[-1] >= len(pool):
        raise RuntimeError(f"NVS window: need {idx[-1] + 1} cams at stride {stride}, pool has {len(pool)}")
    win = [pool[i] for i in idx]
    return [c for j, c in enumerate(win) if j != HOLD_IDX], win[HOLD_IDX]


def times8(clip, k=K8):
    """uniform time-diagonal: round(linspace(0, clip-1, k))."""
    return [int(round(x)) for x in np.linspace(0, clip - 1, k)]


def nearest8(centers, tgt, pool, k=K8):
    """The k cameras in `pool` nearest to camera `tgt` by 3D-center distance (sorted asc index)."""
    d = sorted((float(np.linalg.norm(np.asarray(centers[c]) - np.asarray(centers[tgt]))), int(c))
               for c in pool if c != tgt)
    return sorted(c for _, c in d[:k])


def interior_cam(centers, pool):
    """Camera in `pool` nearest the pool centroid."""
    ct = np.asarray([centers[c] for c in pool])
    return int(pool[int(np.argmin(np.linalg.norm(ct - ct.mean(0), axis=1)))])
