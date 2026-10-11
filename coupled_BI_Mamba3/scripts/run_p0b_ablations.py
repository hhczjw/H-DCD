"""Run the archived P0B training pipeline with isolated ablation settings.

Default: print the plan. --execute starts training; --summarize aggregates
the predeclared validation-selected checkpoint without test-set selection.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
from argparse import Namespace

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "P0B/configs/ablation_plan.json"
METRICS = ("Acc2", "Acc2_has0", "Acc7", "F1", "MAE", "Corr")


def read_plan():
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    base = json.loads((ROOT / plan["historical_args"]).read_text(encoding="utf-8"))
    return plan, base


def effective_args(plan, base, case, seed, run_id):
    args = copy.deepcopy(base)
    args.update(case["overrides"])
    suffix = Path(run_id) / case["name"] / f"seed{seed}"
    for key, key_root in (("checkpoints_dir", "checkpoint_root"),
                          ("results_dir", "results_root"), ("logs_dir", "logs_root")):
        args[key] = str(ROOT / plan[key_root] / suffix)
    for key in ("featurePath", "feature_A", "feature_V"):
        args[key] = str(ROOT / args[key])
    return args


def preflight(base):
    for key in ("featurePath", "feature_A", "feature_V"):
        if not (ROOT / base[key]).is_file():
            raise FileNotFoundError(f"Missing {key}: {base[key]}")
    # Reject the experimental foreach=False optimizer, even if invoked later.
    import ast
    tree = ast.parse((ROOT / "trainer.py").read_text(encoding="utf-8"))
    adam_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute) and node.func.attr == "AdamW"]
    if len(adam_calls) != 1 or any(k.arg == "foreach" for k in adam_calls[0].keywords):
        raise RuntimeError("P0B requires the original AdamW call without a foreach override")


def command(case, seed, run_id):
    return [sys.executable, str(Path(__file__).resolve()), "--worker",
            "--cases", case["name"], "--seeds", str(seed), "--run-id", run_id]


def gpu_idle(gpu):
    out = subprocess.check_output([
        "nvidia-smi", f"--id={gpu}",
        "--query-gpu=uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"
    ], text=True).strip()
    uuid, used, utilization = [part.strip() for part in out.split(",")]
    jobs = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"
    ], text=True)
    idle = int(used) <= 384 and int(utilization) <= 5 and uuid not in jobs
    return idle, f"GPU{gpu}: used={used} MiB, utilization={utilization}%"


def wait_for_gpu(gpu, wait):
    # Two observations reduce accidental starts between short-lived GPU jobs.
    count, previous = 0, None
    while count < 2:
        idle, status = gpu_idle(gpu)
        if status != previous:
            print(status, flush=True)
            previous = status
        if not idle and not wait:
            raise RuntimeError("GPU is occupied; use --wait-gpu or retry when free")
        count = count + 1 if idle else 0
        if count < 2:
            time.sleep(5 if idle else 30)


def summarize(plan, cases, seeds, run_id):
    rows = []
    for case in cases:
        values = []
        for seed in seeds:
            directory = ROOT / plan["results_root"] / run_id / case["name"] / f"seed{seed}"
            path = directory / f"MOSI_p0babl_{case['name']}_seed{seed}_test.json"
            if not path.exists():
                print(f"Incomplete: {case['name']} seed={seed}")
                continue
            metrics = json.loads(path.read_text())[plan["report_checkpoint"]]
            values.append(metrics)
            rows.append(dict(case=case["name"], seed=seed, kind="single", **metrics))
        if len(values) == len(seeds) and values:
            mean = {k: statistics.mean(v[k] for v in values) for k in METRICS}
            std = {k: statistics.stdev(v[k] for v in values) if len(values) > 1 else 0.0
                   for k in METRICS}
            rows.append(dict(case=case["name"], seed="all", kind="mean", **mean))
            rows.append(dict(case=case["name"], seed="all", kind="sample_std", **std))
            print(case["name"], " ".join(f"{k}={mean[k]:.4f}+/-{std[k]:.4f}" for k in METRICS))
    if rows:
        directory = ROOT / plan["results_root"] / run_id
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["case", "seed", "kind", *METRICS],
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--summarize", action="store_true")
    modes.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--run-id", default="p0b_20261009")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--wait-gpu", action="store_true")
    cli = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", cli.run_id):
        parser.error("run-id must contain only letters, numbers, underscores or hyphens")
    plan, base = read_plan()
    names = cli.cases or [case["name"] for case in plan["cases"]]
    unknown = set(names) - {case["name"] for case in plan["cases"]}
    if unknown:
        parser.error(f"Unknown cases: {sorted(unknown)}")
    cases = [case for case in plan["cases"] if case["name"] in names]
    seeds = cli.seeds or plan["seeds"]
    preflight(base)
    if cli.summarize:
        summarize(plan, cases, seeds, cli.run_id)
        return
    if cli.worker:
        if len(cases) != 1 or len(seeds) != 1:
            parser.error("worker requires exactly one case and one seed")
        case, seed = cases[0], seeds[0]
        args = effective_args(plan, base, case, seed, cli.run_id)
        os.chdir(ROOT)
        sys.path.insert(0, str(ROOT))
        import train
        # Keep the historical training source unchanged. Only replace configuration input.
        train.load_config = lambda dataset: Namespace(**copy.deepcopy(args))
        sys.argv = ["train.py", "--dataset", "MOSI", "--seed", str(seed),
                    "--exp_tag", f"p0babl_{case['name']}"]
        train.main()
        return
    if not cli.execute:
        print(f"Plan only: {len(cases) * len(seeds)} runs; GPU{cli.gpu}; "
              f"report checkpoint={plan['report_checkpoint']}")
        for case in cases:
            print(case["name"], case["overrides"], "seeds=", seeds)
        return
    for case in cases:
        for seed in seeds:
            args = effective_args(plan, base, case, seed, cli.run_id)
            result = Path(args["results_dir"]) / f"MOSI_p0babl_{case['name']}_seed{seed}_test.json"
            if result.exists():
                print(f"Already completed: {result}", flush=True)
                continue
            wait_for_gpu(cli.gpu, cli.wait_gpu)
            outdir = Path(args["logs_dir"])
            outdir.mkdir(parents=True, exist_ok=True)
            if (outdir / "console.out").exists() or Path(args["checkpoints_dir"]).exists():
                raise RuntimeError("Previous incomplete run exists; use a new --run-id")
            record = dict(args=args, seed=seed, case=case, checkpoint=plan["report_checkpoint"],
                          source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in [ROOT / "train.py", ROOT / "trainer.py",
                                                   ROOT / "dataset/data_loader.py",
                                                   ROOT / "models/classifier.py",
                                                   ROOT / "mamba/mamba_ssm/modules/mamba_simple.py"]})
            (outdir / "effective_config.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(cli.gpu),
                       PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
            print(f"START {case['name']} seed={seed}", flush=True)
            with (outdir / "console.out").open("w") as stream:
                subprocess.run(command(case, seed, cli.run_id), cwd=ROOT, env=env,
                               stdout=stream, stderr=subprocess.STDOUT, check=True)
            if not result.exists():
                raise RuntimeError(f"Training did not produce expected result: {result}")
            print(f"DONE {case['name']} seed={seed}", flush=True)
    summarize(plan, cases, seeds, cli.run_id)


if __name__ == "__main__":
    main()
