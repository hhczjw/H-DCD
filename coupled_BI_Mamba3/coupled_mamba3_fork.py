"""
[DEPRECATED] 顶层副本已迁移到 models/coupled_mamba3_fork.py.
本文件仅作 re-export, 保持旧 import 兼容, 建议新代码改为:
    from models.coupled_mamba3_fork import CoupledMamba3Fork, CrossMamba3Cell
"""
from models.coupled_mamba3_fork import *  # noqa: F401,F403
from models.coupled_mamba3_fork import CoupledMamba3Fork, CrossMamba3Cell  # noqa: F401