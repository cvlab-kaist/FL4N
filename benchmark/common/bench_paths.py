"""Every external location the benchmark touches, derived from FL4N_DATA, FL4N_BASELINES, BENCH_OUT_ROOT and CONDA_ENVS."""
import os
import sys
from pathlib import Path

DATA = Path(os.environ.get("FL4N_DATA", "data"))
BASELINES = Path(os.environ.get("FL4N_BASELINES", "baselines"))
OUT = Path(os.environ.get("BENCH_OUT_ROOT", "benchmark_out"))
ENVS = Path(os.environ.get("CONDA_ENVS", Path(sys.prefix).parent))
VGGT_POSES = DATA / "vggt_poses"


def dataset(name):
    """Root of one evaluation dataset: <FL4N_DATA>/<name>."""
    return DATA / name


def cache(name):
    """Writable cache directory derived from a dataset: <FL4N_DATA>/_cache/<name>."""
    return DATA / "_cache" / name


def weights(*parts):
    """A baseline checkpoint: <FL4N_BASELINES>/weights/<model>/<file>."""
    return BASELINES.joinpath("weights", *parts)
