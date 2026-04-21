# layers/__init__.py

# 只导入实际存在的类
from .basic import MLP, BiGRU
from .special import GradientReversalLayer

# 定义一个列表，表示可以通过 from layers import * 导入哪些模块
__all__ = [
    'MLP', 
    'BiGRU',
    'GradientReversalLayer'
]