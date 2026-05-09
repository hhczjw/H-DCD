"""
读取 configs/config.json, 返回 argparse.Namespace 或 dict.
"""
from __future__ import annotations

import json
import os
from argparse import Namespace
from typing import Any, Dict

_CFG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


def load_config(dataset_name: str) -> Namespace:
    with open(_CFG_PATH, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = json.load(f)
    merged: Dict[str, Any] = {}
    merged.update(raw.get("common", {}))
    merged.update(raw.get("model", {}))
    ds = raw["datasets"][dataset_name]
    merged.update(ds)
    merged["dataset_name"] = dataset_name.lower()
    merged["dataset_root_dir"] = raw.get("dataset_root_dir", "")
    # 拼接完整特征路径
    if "featurePath" in merged and merged.get("dataset_root_dir"):
        merged["featurePath"] = os.path.join(merged["dataset_root_dir"], merged["featurePath"])
    return Namespace(**merged)