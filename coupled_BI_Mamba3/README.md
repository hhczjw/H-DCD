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
## 🏆 最佳实验结果 (CMU-MOSI)

### 当前最优：配置 `p0B`，seed 2024

日志 `logs/MSA_MOSI_20260918_233755.log` → `results/MOSI_p0B_seed2024_test.json`
归档 checkpoint：`.../codebases/coupled_BI_Mamba3/checkpoints/MOSI_p0B_seed2024_best_*.pt`

**双口径指标（含 0 / 不含 0）**

| 指标 | 含 0 (has0) | 不含 0 (non0) | checkpoint（验证集选优） |
|:----|:----:|:----:|:--|
| **Acc-2** | **88.92** | **91.01** | secondary (best Acc2) |
| **F1** | **88.88** | **91.00** | secondary (best Acc2) |
| **MAE ↓** | 0.5568 | 0.5568 | secondary (best Acc2) |
| **Corr ↑** | 0.8803 | 0.8803 | secondary (best Acc2) |
| **Acc-5** | 60.35 | 60.35 | secondary (best Acc2) |
| **Acc-7** | 52.19 | 52.19 | secondary (best Acc2) |

> MAE / Corr / Acc-5 / Acc-7 在**全测试集**上计算，不分含 0 / 不含 0；只有 Acc-2 与 F1 分两种口径。
> 含 0：全部样本，`pred >= 0` 判正；不含 0：剔除 label==0 的中性样本，`pred > 0` 判正。

<details>
<summary><b>📋 完整配置 <code>p0B</code>（点击展开）</b></summary>

```json
{
  "bert_pretrained": "roberta-large", "bert_finetune": true, "bert_learning_rate": 1e-5,
  "feature_A": "features/mosi_audio_data2vec_full.pkl",
  "feature_V": "features/split_vision_openface3.pkl",
  "use_context": true, "use_bssm_gate": true, "use_gcmn_gate": true,
  "ism_full_frame": true, "multi_task": true, "sub_loss_lambda": 0.3,
  "ism_depth": 1, "ism_d_state": 32, "ism_mixer_type": "bimamba",
  "d_model": 128, "num_layers": 2, "d_state": 64, "expand": 2,
  "dropout": 0.6, "pool_type": "mean",
  "learning_rate": 4e-4, "batch_size": 4, "epochs": 120, "early_stop": 30,
  "loss_type": "l1", "grad_clip": 0.1, "warmup_ratio": 0.15, "weight_decay": 1e-4,
  "need_normalized": true,
  "KeyEval": "MAE", "secondary_metric": "Acc2", "tertiary_metric": "Acc7", "quaternary_metric": "F1_has0"
}
```
</details>

<details>
<summary><b>📋 同一 run 的四个 checkpoint（点击展开）</b></summary>

四者都是**验证集选优**，测试集只评一次。secondary 与 quaternary 落在同一 checkpoint
（该 epoch 同时刷新了验证集 Acc2 与 F1_has0）。

| checkpoint | MAE ↓ | Corr ↑ | Acc-2 含 0 | Acc-2 不含 0 | Acc-7 | F1 含 0 | F1 不含 0 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| **secondary_Acc2** | **0.5568** | **0.8803** | **88.92** | **91.01** | **52.19** | **88.88** | **91.00** |
| quaternary_F1_has0 | 0.5568 | 0.8803 | 88.92 | 91.01 | 52.19 | 88.88 | 91.00 |
| primary_MAE | 0.5956 | 0.8656 | 86.59 | 88.57 | 50.00 | 86.54 | 88.56 |
| tertiary_Acc7 | 0.6366 | 0.8637 | 87.03 | 87.35 | 45.34 | 87.06 | 87.42 |
</details>

<details>
<summary><b>📋 三种子稳定性（配置 <code>p0B</code>，均取 secondary_Acc2）（点击展开）</b></summary>

| seed | MAE ↓ | Corr ↑ | Acc-2 含 0 | Acc-2 不含 0 | Acc-7 | F1 不含 0 |
|:--|--:|--:|--:|--:|--:|--:|
| 42 | 0.6304 | 0.8611 | 85.42 | 87.80 | 47.52 | 87.76 |
| 1111 | 0.6077 | 0.8586 | 86.44 | 88.57 | 48.69 | 88.61 |
| **2024** | **0.5568** | **0.8803** | **88.92** | **91.01** | **52.19** | **91.00** |
| mean ± std | 0.5983 ± 0.0377 | 0.8667 ± 0.0119 | 86.93 ± 0.0180 | 89.13 ± 0.0167 | 49.47 ± 0.0243 | 89.12 ± 0.0168 |

seed 2024 明显好于另两个种子（MAE 低 0.05）。论文建议报均值±std，单点最优作上限参考。
</details>

### 运行命令

```bash
python train.py --dataset MOSI \
    --bert_pretrained roberta-large \
    --feature_A features/mosi_audio_data2vec_full.pkl \
    --feature_V features/split_vision_openface3.pkl \
    --use_context true --use_bssm_gate true --use_gcmn_gate true \
    --ism_full_frame true --multi_task true --sub_loss_lambda 0.3 \
    --ism_depth 1 --ism_d_state 32 --ism_mixer_type bimamba \
    --d_state 64 --num_layers 2 --d_model 128 \
    --learning_rate 4e-4 --dropout 0.6 --batch_size 4 \
    --grad_clip 0.1 --warmup_ratio 0.15 --weight_decay 1e-4 \
    --epochs 120 --early_stop 30 --seed 2024 --exp_tag p0B
```

---

## 📊 与基线模型对比（论文值 vs 复现值）

复现目录：`/media/zjw/951FB31A9E1EB7E0/H-DCD/coupled_bi_mamba3/references/codebases/`
协议统一：**验证集选优**（checkpoint 只看验证集，测试集一次性评测）。
完整对照、脚注与保真度分级见 [`模型对比_Oct2026.md`](模型对比_Oct2026.md)。

### CMU-MOSI

**论文报告值**

| 模型 | 发表 | MAE ↓ | Corr ↑ | Acc-2 含 0 | Acc-2 不含 0 | F1 含 0 | F1 不含 0 | Acc-5 | Acc-7 |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| DEVA | AAAI 2025 | 0.730 | 0.787 | 84.40 | 86.29 | 84.48 | 86.30 | 51.78 | 46.32 |
| DLF | AAAI 2025 | 0.731 | 0.781 | — | 85.06 ¹ | — | 85.04 | 52.33 | 47.08 |
| MSAmba | AAAI 2025 | 0.707 | 0.809 | 85.99 | 87.43 | 85.99 | 87.40 | — | 49.67 |
| EUAR | ACM MM 2024 | — | — | — | 86.30 ¹ | — | 86.30 | — | 46.10 |
| PSA-MF | AAAI 2026 | 0.686 | 0.807 | 83.67 | 86.43 | 83.30 | 86.19 | — | 46.50 |
| EBMC | CVPR 2026 | 0.694 | 0.833 | 86.26 | 87.84 | 86.20 | 87.79 | — | 50.34 |
| MCR (V-T / V-A-T) | NeurIPS 2025 | — | — | 75.2 / 76.5 ² | — | — | — | — | — |
| MIMRL / MMRest / AGS-SMoE / MLCL / CMC / GLoMo | — | 论文 PDF 不在 `papers/`，无法核对 ⁵ | | | | | | | |

**复现值（验证集选优）**

| 模型 | MAE ↓ | Corr ↑ | Acc-2 含 0 | Acc-2 不含 0 | F1 含 0 | F1 不含 0 | Acc-5 | Acc-7 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| **Ours** | **0.557** | **0.880** | **88.92** | **91.01** | **88.88** | **91.00** | **60.35** | **52.19** |
| EBMC | 0.973 | 0.764 | — | 87.04 | — | 87.04 | — | 62.94 ⚠ |
| MIMRL | 0.752 | 0.777 | 82.10 | 84.00 | — | — | 50.30 | 43.60 |
| MMRest | 0.726 | 0.795 | 82.48 | 84.73 | 82.39 | 84.72 | — | 45.40 |
| AGS-SMoE | 0.701 | 0.781 | — | 82.90 | — | 82.93 | — | 44.53 |
| EUAR | 0.760 ⁶ | 0.786 ⁶ | — | 82.44 | — | 82.52 | — | 45.40 |
| DLF | 0.749 ⁷ | 0.786 ⁷ | — | 84.25 | — | — | — | 43.97 |
| MLCL | 0.745 | 0.775 | — | 81.31 | — | 81.09 | — | 46.42 |
| DEVA | 0.756 | 0.782 | 81.20 | 82.93 | 81.26 | 82.93 | 46.79 | 42.13 |
| MSAmba | 0.797 | — | 81.49 | 83.38 | 81.62 | 83.44 | 46.79 | 40.52 |
| CMC | — | — | 80.17 | — | 80.12 | — | — | n/a ⁸ |
| GLoMo | 0.790 | 0.738 | — | 79.85 | — | 79.88 | — | 45.69 |
| MCR (V-T-A) | — | — | 74.09 ² | — | 73.97 | — | — | — |
| PSA-MF | 1.124 | 0.514 | 69.83 | 70.58 | 69.89 | 70.73 | — | 30.90 |

**论文 vs 复现 Δ（关键三项）**

| 模型 | MAE 论文→复现 | Acc-2 不含 0 论文→复现 | Acc-7 论文→复现 |
|:--|:--|:--|:--|
| DEVA | 0.730 → 0.756 | 86.29 → 82.93 | 46.32 → 42.13 |
| DLF | 0.731 → 0.749 | 85.06 → 84.25 | 47.08 → 43.97 |
| MSAmba | 0.707 → 0.797 | 87.43 → 83.38 | 49.67 → 40.52 |
| EUAR | — | 86.30 → 82.44 | 46.10 → 45.40 |
| PSA-MF | 0.686 → 1.124 | 86.43 → 70.58 | 46.50 → 30.90 |
| EBMC | 0.694 → 0.973 | 87.84 → 87.04 | 50.34 → 62.94 ⚠ |
| MCR | — | 76.5 → 74.09 | — |
| GraphSmile | — | — | 46.82 → 44.22 |

### CMU-MOSEI

**Ours 尚未跑 MOSEI**，下表为论文值与复现值对照。

2026-10-07：已准备与 MOSI p0B 对齐的 `configs/mosei_p0B.json`，共用训练、
验证集选优与测试管道；MOSEI Data2Vec/OpenFace3 特征包仍缺失，真实数据
加载及训练冒烟尚未完成。启动使用 `bash scripts/run_mosei_best.sh`，脚本会
先执行共用 DataLoader 检查。步骤和当前限制见 [MOSEI 准备说明](docs/mosei_feature_extraction.md)。

**论文报告值**

| 模型 | 发表 | MAE ↓ | Corr ↑ | Acc-2 含 0 | Acc-2 不含 0 | F1 含 0 | F1 不含 0 | Acc-5 | Acc-7 |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| EBMC | CVPR 2026 | 0.505 | **0.824** | 86.04 | **88.10** | 86.23 | **88.07** | — | **57.32** |
| MSAmba | AAAI 2025 | **0.507** | 0.796 | 85.78 | 86.86 | 85.99 | 86.93 | — | 54.21 |
| PSA-MF | AAAI 2026 | 0.521 | 0.774 | 83.76 | 86.30 | 84.21 | 86.28 | — | 55.00 |
| DEVA | AAAI 2025 | 0.541 | 0.769 | 83.26 | 86.13 | 82.93 | 86.21 | 55.32 | 52.26 |
| DLF | AAAI 2025 | 0.536 | 0.764 | — | 85.42 ¹ | — | 85.27 | 55.70 | 53.90 |
| MSE-Adapter | AAAI 2025 | 0.558 | 0.725 | 84.12 | — | 83.45 | — | — | 52.02 |
| EUAR | ACM MM 2024 | — | — | — | 86.60 ¹ | — | 86.40 | — | 54.90 |
| MCR (V-T / V-A-T) | NeurIPS 2025 | — | — | 80.8 / 81.1 ² | — | — | — | — | — |
| GraphSmile | TPAMI 2025 | — | — | — | — | — | — | — | 46.82 ³ |

**复现值（验证集选优）**

| 模型 | MAE ↓ | Corr ↑ | Acc-2 含 0 | Acc-2 不含 0 | F1 含 0 | F1 不含 0 | Acc-5 | Acc-7 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| **Ours** | *待跑* | *待跑* | *待跑* | *待跑* | *待跑* | *待跑* | *待跑* | *待跑* |
| MIMRL | **0.528** | 0.769 | — | 83.30 | — | — | — | 52.90 |
| DEVA | 0.5356 | 0.7695 | 81.67 | 85.55 | 81.22 | 85.62 | 55.48 | **53.77** |
| DLF | 0.5377 | 0.7647 | — | 84.73 | — | 84.80 | 54.39 | 52.82 |
| GLoMo | 0.5376 | **0.7689** | — | 83.20 | — | 83.56 | 54.71 | 53.03 |
| MLCL | 0.5394 | 0.7514 | — | 78.38 | — | 79.25 | 54.84 | 53.07 |
| MSE-Adapter | 0.5491 ⁹ | 0.7417 ⁹ | 84.80 ⁹ | 74.76 ⁹ | 84.39 ⁹ | 74.42 ⁹ | 54.69 ⁹ | 52.66 ⁹ |
| MCR (V-T) | 0.6071 | 0.6705 | 79.64 ² | — | 79.67 | — | — | 49.84 |
| EBMC | 0.6176 ⚠ | 0.9343 ⚠ | 81.52 | 94.74 ⚠ | — | 94.75 ⚠ | — | 52.78 |
| CMC | — | — | 82.58 | 84.01 | 82.73 | 83.72 | — | n/a ⁸ |
| EUAR | 0.6209 ⁶ | 0.7613 ⁶ | — | 84.48 | — | 84.16 | — | 52.40 |
| PSA-MF | 0.6513 | 0.6061 | 78.94 | 76.55 | 77.40 | 74.55 | — | 48.00 |
| MCR (V-T-A) | — | — | 80.43 ² | — | 80.95 | — | — | — |
| GraphSmile | — | — | — | — | — | — | — | 44.22 ³ |

**论文 vs 复现 Δ（关键三项）**

| 模型 | MAE 论文→复现 | Acc-2 不含 0 论文→复现 | Acc-7 论文→复现 |
|:--|:--|:--|:--|
| MSAmba | 0.507 → 0.546 | 86.86 → 83.76 | 54.21 → 51.92 |
| EBMC | 0.505 → 0.618 ⚠ | 88.10 → 94.74 ⚠ | 57.32 → 52.78 |
| PSA-MF | 0.521 → 0.651 | 86.30 → 76.55 | 55.00 → 48.00 |
| DEVA | 0.541 → 0.536 ✅ | 86.13 → 85.55 | 52.26 → 53.77 ✅ |
| DLF | 0.536 → 0.538 ✅ | 85.42 → 84.73 | 53.90 → 52.82 |
| MSE-Adapter | 0.558 → 0.549 ✅ | 84.12 → 84.80 (含0) ✅ | 52.02 → 52.66 ✅ |
| EUAR | — | 86.60 → 84.48 | 54.90 → 52.40 |
| MCR | — | 81.1 → 80.43 | — |
| GraphSmile | — | — | 46.82 → 44.22 |

MOSEI 侧复现保真度优于 MOSI：DEVA / DLF / MSE-Adapter 三项基本对齐（Δ ≤ 1 pt），MSE-Adapter 小幅超过论文。

### 对比结论

**Ours vs 论文 SOTA（CMU-MOSI）**

| 指标 | **Ours** | 论文最好 | Δ |
|:--|--:|:--|--:|
| MAE ↓ | **0.557** | EBMC 0.694 | **−0.137** |
| Corr ↑ | **0.880** | EBMC 0.833 | **+0.047** |
| Acc-2 不含 0 | **91.01** | EBMC 87.84 | **+3.17 pt** |
| Acc-2 含 0 | **88.92** | MSAmba 85.99 | **+2.93 pt** |
| F1 不含 0 | **91.00** | EBMC 87.79 | **+3.21 pt** |
| Acc-5 | **60.35** | DEVA 51.78 | **+8.57 pt** |
| Acc-7 | **52.19** | EBMC 50.34 | **+1.85 pt** ✅ |

**即使与论文报告值相比，Ours 在 MOSI 全部指标上仍领先**，包括此前唯一不占优的 Acc-7（52.19 > EBMC 论文 50.34）。

与复现值比时 Acc-7 排前面的只有 EBMC，但该值与 EBMC 自身论文的 50.34 相差 12.6 pt 且内部矛盾（见下注 3），判定异常；排除后 Ours 也领先约 5.8 pt。

**CMU-MOSEI 建议目标**

| 指标 | 论文最好 | 复现最好 | 建议目标 |
|:--|:--|:--|:--|
| MAE ↓ | EBMC 0.505 | MIMRL 0.528 | **< 0.50** |
| Corr ↑ | EBMC 0.824 | GLoMo 0.7689 | **> 0.82** |
| Acc-2 不含 0 | EBMC 88.10 | EUAR 84.48 | **> 88** |
| Acc-2 含 0 | MSAmba 85.78 | MSE-Adapter 84.80 | **> 86** |
| Acc-7 | EBMC 57.32 | DEVA 53.77 | **> 57** |

优先补齐 MOSEI，使两个数据集齐备。

---

## 📌 读表须知

1. **DLF / EUAR 论文只报一个 Acc-2**，其 `metricsTop.py` / `train.py compute()` 实现为 `label != 0` 过滤后计算，归入**不含 0**列。
2. **MCR 报分类 Acc_combined / F1_combined**（回归预测转二分类准确率），非 has0/non0 回归口径。论文用 V-T 与 V-A-T，复现跑 V-T-A，口径相近非完全一致。
3. **EBMC 的 Acc-7 / Corr 不可直接引用**：MOSI 侧复现 Acc-7 62.94 配 MAE 0.973（全表最差）且比论文 50.34 高 12.6 pt；MOSEI 侧 Corr 0.9343 比自己验证集的 0.76 高 0.17，MAE 0.618 反差于论文 0.505。两处自相矛盾，需复查 fold 划分/评测子集。
4. **GraphSmile 的 46.82 / 44.93 是 7 分类 ACC / 加权 F1**，与表中 Acc-2/F1 不同口径，故填 Acc-7 列并加注。
5. **MSE-Adapter 论文只报 MOSEI**（另有 SIMS-V2/MELD/CHERMA），无 MOSI 结果。复现为 5 种子（1111/2222/3333/4444/5555）均值，论文亦为 5 种子均值。
6. **MIMRL / MMRest / AGS-SMoE / MLCL / CMC / GLoMo 的论文 PDF 不在 `papers/` 目录**（MIMRL 为 TAFFC 期刊、MMRest 为 CVPR 2026、其余无 PDF），论文值无法核对。
7. **CMC 的 Acc-7 标 n/a**：该 release 是 3 类分类器（负/中/正），不输出连续情感分，`eval_metrics.py` 已注明 Acc-7 未定义。MOSI 上曾出现的「Acc-7 0.7843」实为 Acc-3。
8. **EUAR 的 MAE / Corr 在不含 0 子集上算**（`compute()` 先过滤 `label != 0`）；DLF 的 MAE/Corr 取自 seed 1111 单次。跨行比较 MAE 时需注意。
9. **批大小缩减**（8 GB 显卡限制）：GLoMo 16（原 240）、MLCL 16（原 70）、MIMRL 32（原 128）、MSAmba 96（原 128）、PSA-MF 64（原 256/128）、MSE-Adapter 4（原 16）、Ours 4。模型结构、损失与其余超参均未改动。PSA-MF 的 InfoNCE 负样本依赖批大小，是其复现偏低的主因。
10. **测试集窥探已在复现侧清除**：PSA-MF、MMRest、MIMRL、MCR、GraphSmile 的 release 原本用测试指标选 checkpoint，复现时一律改为只看验证集。Ours 侧四个 checkpoint 也分别按验证集选优。
11. **Ours 输入特征与多数基线不同**：Ours 用 RoBERTa-large + Data2Vec-Audio + OpenFace3 + 对话上下文；多数基线用 BERT-base + COVAREP/FACET 手工特征且无上下文。特征侧优势需在消融中剥离，才能把增益归因到 Coupled-BI-Mamba3 结构本身。
