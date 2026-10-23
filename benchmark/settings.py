"""Final evaluation settings for every dataset x protocol."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "common"))
from bench_paths import ENVS, VGGT_POSES  # noqa: E402

GAMMA = os.environ.get("CAM_SCALE", "1.35")
FL4N_CKPT = os.environ.get("FL4N_CKPT", os.path.join(ROOT, "checkpoints", "fl4n.ckpt"))
VGGT_CACHE = os.environ.get("VGGT_POSE_CACHE", str(VGGT_POSES / "{tag}"))

PROTOCOLS = {
    "ctvs": ["nvidia", "n3dv", "kubric4d", "camxtime", "charge"],
    "nvs":  ["nvidia", "n3dv", "kubric4d", "camxtime", "charge"],
    "nts":  ["adt", "dycheck", "tum"],
}
NSCENES = {"nvidia": 9, "n3dv": 6, "kubric4d": 100, "camxtime": 42, "charge": 8, "adt": 4, "dycheck": 5, "tum": 9}

MODELS = {
    "fl4n": ("fl4n", "bench_fl4n.py", "fl4n", 12000),
}

_COMMON = {"PYTHONNOUSERSITE": "1", "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "HF_HUB_OFFLINE": "1"}


def _fl4n_cameras(dataset, protocol):
    """Synthetic sets use dataset cameras; NVIDIA/N3DV/NTS use the per-dataset, per-protocol VGGT pose+intrinsics cache."""
    if dataset in ("kubric4d", "camxtime", "charge"):
        return {}
    tag = "nts" if protocol == "nts" else (f"n3dv_{protocol}" if dataset == "n3dv" else protocol)
    return {"FL4N_VGGT_POSE": "1", "FL4N_VGGT_INTR": "1", "FL4N_VGGT_POSE_DIR": VGGT_CACHE.format(tag=tag)}


def out_name(model, posed=False):
    """Result folder name of a cell; the posed FL4N variant gets its own folder."""
    return "fl4n_posed" if (model == "fl4n" and posed) else MODELS[model][2]


def final_env(model, dataset, protocol, posed=False):
    if dataset not in PROTOCOLS[protocol]:
        raise ValueError(f"{dataset} is not part of the final {protocol} table")
    env = dict(_COMMON)
    if protocol == "nvs":
        env["BENCH_TABLE2"] = "1"
    if dataset == "kubric4d":
        env["KUB4D_ALL"] = "1"
    envname, _, _, _ = MODELS[model]
    env["PATH"] = f"{ENVS}/{envname}/bin:" + os.environ.get("PATH", "")
    if model == "fl4n":
        env.update({"BENCH_LONG_EDGE": "512", "BENCH_NVS8": "1", "CAM_SCALE": GAMMA})
        env.update(_fl4n_cameras(dataset, protocol))
    else:
        raise KeyError(model)
    return env


def runner_cmd(model, dataset, posed=False):
    envname, script, _, _ = MODELS[model]
    cmd = [f"{ENVS}/{envname}/bin/python", "-u", os.path.join(HERE, "runners", script), "--dataset", dataset]
    if model == "fl4n":
        cmd += ["--ckpt", FL4N_CKPT, "--out-name", out_name(model, posed)] + (["--posed"] if posed else [])
    return cmd
