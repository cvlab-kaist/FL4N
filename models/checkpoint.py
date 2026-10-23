"""Load FL4N from checkpoints/fl4n.ckpt (all weights in one file) or from a Lightning training checkpoint (trainable modules only)."""
import argparse
import os

import torch

from models.fl4n import FL4N

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FL4N_CKPT = os.environ.get("FL4N_CKPT", os.path.join(ROOT, "checkpoints", "fl4n.ckpt"))
_RENAMES = {"static_reconstructor.": "static_enc.", "dynamic_reconstructor.": "dyn_enc.", "loss_fn.perceptual_loss_module.": "loss_fn.perceptual."}


def remap_keys(sd, strip_prefix=None):
    """Rename module prefixes per _RENAMES (with or without a leading 'model.') and optionally strip one prefix."""
    out = {}
    for k, v in sd.items():
        for old, new in _RENAMES.items():
            if k.startswith(old):
                k = new + k[len(old):]
            elif k.startswith("model." + old):
                k = "model." + new + k[len("model." + old):]
        if strip_prefix is not None:
            if not k.startswith(strip_prefix):
                continue
            k = k[len(strip_prefix):]
        out[k] = v
    return out


def load_checkpoint(path=FL4N_CKPT, device="cuda", base=FL4N_CKPT):
    """Full ckpt -> strict load; training ckpt -> frozen modules from `base` + trainable modules from the ckpt. Returns (model, step)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = remap_keys(ck["state_dict"], strip_prefix="model." if any(k.startswith("model.") for k in ck["state_dict"]) else None)
    full = any(k.startswith("static_enc.") for k in sd)
    model = FL4N(pretrained_path=None if full else base, dyn_warm_start=False)
    sd = {k: v.float() if v.is_floating_point() else v for k, v in sd.items()}
    miss, unexp = model.load_state_dict(sd, strict=False)
    miss = [k for k in miss if full or not (k.startswith("static_enc.") or k.startswith("renderer."))]
    assert not miss and not unexp, f"ckpt mismatch: missing={miss[:3]} unexpected={unexp[:3]}"
    return model.to(device).eval(), ck.get("global_step")


def main():
    ap = argparse.ArgumentParser(description="Merge a training checkpoint with the frozen modules into one full fl4n.ckpt.")
    ap.add_argument("ckpt", help="Lightning training checkpoint (trainable modules only)")
    ap.add_argument("out", help="output path, e.g. checkpoints/fl4n.ckpt")
    ap.add_argument("--base", default=FL4N_CKPT, help="source of the frozen static encoder + renderer (fl4n.ckpt or static-base .pt)")
    a = ap.parse_args()
    model, step = load_checkpoint(a.ckpt, "cpu", a.base)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "global_step": step}, a.out)
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1e9:.2f} GB, {len(model.state_dict())} tensors, step={step})")


if __name__ == "__main__":
    main()
