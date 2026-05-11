"""
批量运行多数据集 (便于实验表生成):
    python run.py --datasets MOSI MOSEI SIMS --seeds 42 2026 5201314
"""
from __future__ import annotations

import _path_setup  # noqa: F401  注入内置 mamba_ssm 路径 (子进程也会通过 train.py 自行注入)

import argparse
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["MOSI", "MOSEI"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    args = p.parse_args()

    for ds in args.datasets:
        for s in args.seeds:
            cmd = [sys.executable, "train.py", "--dataset", ds, "--seed", str(s)]
            print(f"\n>>> Running: {' '.join(cmd)}\n")
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()