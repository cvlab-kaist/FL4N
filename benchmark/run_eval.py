"""Run the benchmark: models x protocols x datasets on a GPU pool, each cell with its final settings."""
import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from settings import MODELS, NSCENES, PROTOCOLS, final_env, out_name, runner_cmd


def free_mib(gpu):
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", str(gpu)])
    return int(out.decode().strip())


def cell_complete(out_dir, out_name, dataset):
    try:
        return len(json.load(open(os.path.join(out_dir, dataset, out_name, "summary.json"))).get("per_scene", {})) >= NSCENES[dataset]
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, choices=list(MODELS))
    ap.add_argument("--posed", action="store_true", help="fl4n only: feed conditioning camera poses (results go to fl4n_posed/)")
    ap.add_argument("--protocols", nargs="+", default=["ctvs", "nvs", "nts"], choices=list(PROTOCOLS))
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--scenes", nargs="+", default=None, help="evaluate only these scenes (not supported for c4g)")
    ap.add_argument("--gpus", nargs="+", type=int, required=True)
    ap.add_argument("--out", default="./benchmark_out")
    ap.add_argument("--logs", default="./benchmark_out/logs")
    ap.add_argument("--gate-minutes", type=int, default=120, help="max minutes to wait for free VRAM before skipping a cell")
    a = ap.parse_args()
    if a.scenes and "c4g" in a.models:
        ap.error("--scenes is not supported for c4g")
    os.makedirs(a.logs, exist_ok=True)

    jobs = [(m, p, d) for m in a.models for p in a.protocols for d in PROTOCOLS[p] if a.datasets is None or d in a.datasets]
    size = {"kubric4d": 0, "camxtime": 1, "n3dv": 2, "nvidia": 3, "tum": 4, "charge": 5, "dycheck": 6, "adt": 7}
    jobs.sort(key=lambda j: size.get(j[2], 9))
    q = queue.Queue()
    for j in jobs:
        q.put(j)
    print(f"{len(jobs)} cells | models={a.models} protocols={a.protocols} gpus={a.gpus}", flush=True)

    def run(gpu, job):
        model, proto, ds = job
        name, min_free = out_name(model, a.posed), MODELS[model][3]
        out = os.path.join(a.out, proto, ds)
        done = os.path.join(out, f"_done_{name}")
        if os.path.exists(done) or cell_complete(out, name, ds):
            print(f"[gpu{gpu}] skip {proto}/{ds}/{model} (complete)", flush=True)
            os.makedirs(out, exist_ok=True)
            open(done, "w").close()
            return
        for _ in range(a.gate_minutes):
            if free_mib(gpu) > min_free:
                break
            time.sleep(60)
        else:
            print(f"[gpu{gpu}] GATE TIMEOUT {proto}/{ds}/{model}: skipped", flush=True)
            return
        env = dict(os.environ)
        env.update(final_env(model, ds, proto, a.posed))
        env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "BENCH_OUT_ROOT": out})
        t0 = time.time()
        log = os.path.join(a.logs, f"{proto}_{ds}_{name}.log")
        with open(log, "a") as lf:
            rc = subprocess.run(runner_cmd(model, ds, a.posed) + (["--scenes", *a.scenes] if a.scenes else []), cwd=os.path.dirname(os.path.abspath(__file__)), env=env,
                                stdout=lf, stderr=subprocess.STDOUT).returncode
        ok = rc == 0 and (bool(a.scenes) or cell_complete(out, name, ds))
        if ok and not a.scenes:
            os.makedirs(out, exist_ok=True)
            open(done, "w").close()
        print(f"[gpu{gpu}] {'OK  ' if ok else 'FAIL'} {proto}/{ds}/{model} {(time.time() - t0) / 60:.1f} min (rc={rc}, log={log})", flush=True)

    def worker(gpu):
        while True:
            try:
                j = q.get_nowait()
            except queue.Empty:
                return
            run(gpu, j)
            q.task_done()

    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in a.gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
