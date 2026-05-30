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

---

## 🏆 最佳实验结果 (CMU-MOSI, seed=42)

### 当前最优 (ism_sub_rawvision, Mamba-2 ISM)

| 指标 | 最佳值 | Checkpoint 类型 |
|:----|:-----:|:---------------|
| **Acc7** | **0.5044** | tertiary (best Acc7) |
| **Acc2** | **0.8918** | primary (best MAE) / secondary (best Acc2) |
| MAE | 0.6060 | primary (best MAE) |
| Corr | 0.8371 | primary (best MAE) |

<details>
<summary><b>📋 完整配置 (点击展开)</b></summary>

```json
{
  "bert_pretrained": "roberta-base",
  "feature_A": "features/mosi_audio_data2vec_full.pkl",
  "feature_V": "features/split_vision_openface3.pkl",
  "use_context": true,
  "use_bssm_gate": true,
  "use_gcmn_gate": true,
  "ism_full_frame": true,
  "multi_task": true,
  "sub_loss_lambda": 0.3,
  "lr": 0.0005,
  "dropout": 0.7,
  "batch_size": 16,
  "grad_clip": 0.1,
  "warmup_ratio": 0.15,
  "weight_decay": 0.0001,
  "ism_depth": 3,
  "ism_d_state": 32,
  "ism_mixer_type": "bimamba",
  "d_state": 64,
  "num_layers": 2,
  "d_model": 128
}
```
</details>

### 完整测试结果

```text
=== tertiary ckpt (best Acc7) ===
MAE=0.6531 | Corr=0.8142 | Acc2=0.8689 | Acc7=0.5044

=== primary ckpt (best MAE) ===
MAE=0.6060 | Corr=0.8371 | Acc2=0.8918 | Acc7=0.4927

=== secondary ckpt (best Acc2) ===
MAE=0.6060 | Corr=0.8371 | Acc2=0.8918 | Acc7=0.4927
```

### 运行命令

```bash
python train.py --dataset MOSI \
    --feature_A features/mosi_audio_data2vec_full.pkl \
    --feature_V features/split_vision_openface3.pkl \
    --use_context true --use_bssm_gate true --use_gcmn_gate true \
    --lr 5e-4 --dropout 0.7 --grad_clip 0.1 \
    --ism_depth 3 --d_state 64 --num_layers 2 \
    --warmup_ratio 0.15 --weight_decay 0.0001 \
    --batch_size 16 --bert_pretrained roberta-base \
    --ism_full_frame true --multi_task true --sub_loss_lambda 0.3 \
    --exp_tag best_config
```

### 历史最佳排名 (MOSI 测试集)

| 排名 | 实验 | Acc7 | Acc2 | MAE | 备注 |
|:---:|:----|:----:|:----:|:---:|:----|
| 🥇 | **ism_sub_rawvision** | **0.5044** | **0.8918** | **0.6060** | 当前最佳, Mamba-2 ISM |
| 🥈 | ism_fullframe | 0.4985 | 0.8765 | 0.6192 | 全帧 ISM + 50帧音频 |
| 🥉 | gs2_openface3_lr5e-4_do0.7 | 0.4971 | 0.8506 | 0.6863 | 网格搜索 OpenFace3 |
| 4 | roberta_keyacc7_ema | 0.4927 | 0.8674 | 0.6378 | EMA 衰减 |
| 5 | ema_best | 0.4898 | 0.8811 | 0.6326 | EMA 0.99 |
| 6 | roberta_best | 0.4898 | 0.8811 | 0.6344 | RoBERTa baseline |
| 7 | of3_full | 0.4840 | 0.8780 | 0.6432 | OF3 全帧 (NaN 崩溃) |
| 8 | bimamba3_lr2e4 | ⏳ 训练中 | 0.9028* | 0.6579* | Mamba-3 ISM, *验证集值 |
| 9 | subloss0.3 | 0.4694 | 0.8841 | 0.6510 | sub_loss 消融 |

### 参数演变记录

| 阶段 | Mamba 类型 | ism_d_state | lr | Acc7 | Acc2 |
|:----|:----------|:----------:|:--:|:----:|:----:|
| 初始基线 | Mamba-2 | 32 | 5e-4 | 0.4810 | 0.8750 |
| +RoBERTa | Mamba-2 | 32 | 5e-4 | 0.4898 | 0.8811 |
| +full_frame audio | Mamba-2 | 32 | 5e-4 | 0.4985 | 0.8765 |
| +sub_loss | Mamba-2 | 32 | 5e-4 | 0.5044 | 0.8918 |
| +bimamba3 | Mamba-3 | 64 | 2e-4 | ⏳ | 0.9028* |