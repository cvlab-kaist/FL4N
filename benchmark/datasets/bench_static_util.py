"""Static-baseline hook."""
import os


def maybe_static(d):
    if not os.environ.get("BENCH_STATIC"):
        return d
    cond = d["cond"]
    cammap = {}
    for c in cond:
        k = c.get("view", c.get("cam"))
        if k is not None:
            cammap[k] = int(c["frame"])
    ref0 = int(cond[0]["frame"])
    for _g, items in d["targets"].items():
        for it in items:
            cam = it.get("view", it.get("cam"))
            it["frame"] = cammap.get(cam, ref0)   # render at the row-camera's diagonal (or ref) time
    d.setdefault("meta", {})["static"] = True
    return d
