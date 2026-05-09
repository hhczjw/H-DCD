# Coupled-BI-Mamba3: 基于 Mamba-3 的跨模态情感识别

> 本项目对官方 Mamba-3 进行了**最小侵入的 fork**, 在保留全部工程特性
> (data-dep A、Trap、RoPE、RMSNormGated B/C、MIMO、D-skip 等) 的前提下,
> 引入跨模态 B/C 注入, 实现三模态 (Text / Audio / Vision) 的耦合双向状态空间融合.

## 目录结构

```text
coupled_BI_Mamba3/
├── README.md
├── requirements.txt
├── train.py              # 单数据集训练入口
├── run.py                # 多数据集 / 多种子批量运行
├── trainer.py            # 通用 Trainer
├── configs/
│   ├── __init__.py       # load_config()
│   └── config.json       # 5 个数据集 + 模型/通用超参
├── dataset/
│   ├── __init__.py
│   └── data_loader.py    # MMDataset / MMDataLoader (MOSI/MOSEI/SIMS/IEMOCAP/MELD)
├── models/
│   ├── __init__.py
│   ├── coupled_mamba3_fork.py   # CrossMamba3Cell + CoupledMamba3Fork
│   └── classifier.py            # MSAClassifier (端到端模型)
├── layers/
│   ├── __init__.py
│   └── feature_projection.py
├── losses/
│   ├── __init__.py
│   └── task_losses.py           # 回归/分类/多任务
├── utils/
│   ├── __init__.py
│   ├── seed.py
│   ├── logger.py
│   └── metrics.py               # eval_regression / eval_classification
├── checkpoints/                 # 保存最佳模型
├── logs/                        # 训练日志
└── results/                     # 测试结果 JSON
```

## 支持数据集

| 数据集   | 任务类型   | 主指标 | 子任务标签 |
|----------|------------|--------|------------|
| MOSI     | 回归       | MAE/Acc-2 | -      |
| MOSEI    | 回归       | MAE/Acc-2 | -      |
| SIMS     | 回归(多任务) | MAE/Acc-2 | T/A/V |
| IEMOCAP  | 4 类分类   | F1     | -          |
| MELD     | 7 类分类   | F1     | -          |

## 快速使用

```bash
# 1) 安装依赖 (Mamba-3 官方对齐)
pip install -r requirements.txt

# 2) 修改 configs/config.json 中的 dataset_root_dir 为本地路径

# 3) 训练 / 测试
python train.py --dataset MOSI --seed 42

# 4) 批量
python run.py --datasets MOSI MOSEI SIMS --seeds 42 2026 5201314
```

## 设计要点

- `CrossMamba3Cell`: 直接 fork [`Mamba3`](../mamba/mamba_ssm/modules/mamba3.py:26),
  在 `forward(x, b_external, c_external, ...)` 中接收外部模态的 B/C, 实现跨模态注入,
  完整保留 RoPE / RMSNormGated / MIMO / data-dep A / Trap / D-skip.
- `CoupledMamba3Fork`: 三模态等长输入, 每个模态各跑一个 Cell, B/C 来自其余两模态加权融合.
- **作者源码不动**: 仅通过 `import mamba_ssm` 复用, 任何能力升级随官方版本自动到位.