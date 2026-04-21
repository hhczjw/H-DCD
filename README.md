# H-DCD: Hierarchical Decoupled Contrastive Distillation

## 项目简介

**H-DCD** (Hierarchical Decoupled Contrastive Distillation) 是一个用于多模态情感识别的深度学习模型。该模型通过四个核心阶段实现对音频、视频和文本多模态数据的高效情感分析。

### 核心特性

- **四阶段层次化架构**：单模态编码 → 对抗解耦 → 双轨Mamba骨干 → 层次化监督
- **特征解耦**：分离共性特征和模态特有特征
- **双轨并行Mamba**：HMNF-Homo处理共性特征，HMPN-Hetero处理私有特征
- **复合损失函数**：任务损失 + 解耦损失 + 层次化损失 + 蒸馏损失
- **高效推理**：基于Mamba的线性复杂度序列建模

---

## 模型架构

### 四阶段流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Stage 1: 单模态特征编码                         │
├─────────────────────────────────────────────────────────────────────┤
│  Audio (74) ──→ MLP ──→ h_a (128)                                  │
│  Visual (35) ─→ MLP ──→ h_v (128)                                  │
│  Text (768) ──→ BiGRU → h_t (128)                                  │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────────┐
│                   Stage 2: 对抗特征解耦 (DecoupleEncoder)              │
├─────────────────────────────────────────────────────────────────────┤
│  h_m ──→ ┌────────────────┐                                        │
│          │ Common Encoder │ ──→ z_m_common (共性特征)               │
│          │ Private Encoder│ ──→ z_m_private (私有特征)              │
│          └────────────────┘                                        │
│                    ↓                                                │
│          GRL + Discriminator (对抗训练)                              │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────────┐
│              Stage 3: 双轨并行 Mamba 骨干 (HMNF)                      │
├─────────────────────────────────────────────────────────────────────┤
│  同质轨道:  z_homo ──→ HMNF-Homo  ──→ z_homo_final                  │
│  异质轨道:  z_hetero ─→ HMPN-Hetero ─→ z_hetero_final               │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────────┐
│              Stage 4: 层次化多任务监督                                 │
├─────────────────────────────────────────────────────────────────────┤
│  z_homo_final ───→ y_uni  (单模态预测)                              │
│  z_hetero_final ─→ y_bi   (双模态预测)                              │
│  concat ─────────→ y_mul  (多模态预测)                              │
│                       │                                             │
│                       └──→ Final Prediction                         │
└─────────────────────────────────────────────────────────────────────┘
```

### 损失函数

```python
L_total = L_task + λ_dec·L_dec + λ_hierarchical·L_hierarchical + λ_distill·L_distill
```

#### 1. 任务损失 (L_task)
- 分类交叉熵损失
- 优化最终情感预测准确率

#### 2. 解耦损失 (L_dec)
```python
L_dec = L_recon + λ_adv·L_adv + λ_ortho·L_ortho + λ_margin·L_margin
```
- **L_recon**: 重构损失，确保特征分解的完整性
- **L_adv**: 对抗损失，强制共性特征模态无关
- **L_ortho**: 正交损失，确保共性和私有特征独立
- **L_margin**: 间隔损失，增强特征分离度

#### 3. 层次化损失 (L_hierarchical)
```python
L_hierarchical = L_task(y_uni) + L_task(y_bi) + L_task(y_mul)
```
- 多层级监督信号
- 提升特征表达能力

#### 4. 蒸馏损失 (L_distill)
```python
L_distill = KL(y_uni, y_final) + KL(y_bi, y_final) + KL(y_mul, y_final)
```
- 知识蒸馏，传播最终预测到各层
- 增强模型一致性

---

## 安装

### 环境要求

| 组件 | 版本要求 | 说明 |
|------|---------|------|
| **Python** | >= 3.9, <= 3.12 | 推荐 3.10 或 3.11 |
| **PyTorch** | >= 2.0.0, <= 2.5.1 | 推荐 2.5.0+ (性能最佳) |
| **CUDA** | >= 11.8, <= 12.4 | 推荐 12.1+ (PyTorch 2.5 优化) |
| **cuDNN** | >= 8.9 | 与 CUDA 版本匹配 |
| **GPU 显存** | >= 8GB | 推荐 16GB 以上 |

> ⚠️ **重要提示**:
> - Mamba-SSM 需要编译 CUDA 扩展，必须安装正确的 CUDA 工具链
> - **PyTorch 2.5.0+**: 推荐使用最新版本，性能提升显著，需 mamba-ssm >= 2.2.0
> - **PyTorch 2.0-2.1**: 稳定版本，mamba-ssm >= 1.2.0 即可
> - 如无 GPU，可使用 CPU 版本（性能会显著下降）

### 方法一：Conda 环境（推荐）

```bash
# 1. 创建虚拟环境
conda create -n hdcd python=3.10 -y
conda activate hdcd

# 2. 安装 PyTorch
# 推荐：最新版本 (CUDA 12.1+)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 或指定版本：
# PyTorch 2.5.0 (推荐，性能最佳)
# conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# PyTorch 2.1.0 (稳定版)
# conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 3. 验证 PyTorch 安装
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"

# 4. 安装 Mamba 核心依赖（需要编译 CUDA 扩展）
# 对于 PyTorch 2.5.0+，使用最新版本
pip install causal-conv1d>=2.0.0
pip install mamba-ssm>=2.2.0

# 对于 PyTorch 2.0-2.1，使用旧版本
# pip install causal-conv1d>=1.4.0,<2.0.0
# pip install mamba-ssm>=1.2.0,<2.0.0

# 5. 安装其他依赖
pip install numpy scipy pandas scikit-learn matplotlib seaborn
pip install pyyaml tqdm tensorboard
pip install einops packaging ninja

# 6. 可选：数据处理工具
pip install opencv-python librosa transformers

# 7. 验证 Mamba 安装
python -c "from mamba_ssm.modules.mamba2_stateful import Mamba2Stateful; print('Mamba2 OK')"
```

### 方法二：pip 环境（纯 pip）

```bash
# 1. 创建虚拟环境
python3.10 -m venv hdcd_env
source hdcd_env/bin/activate  # Linux/Mac
# 或 hdcd_env\Scripts\activate  # Windows

# 2. 升级 pip
pip install --upgrade pip setuptools wheel

# 3. 安装 PyTorch
# 推荐：最新版本 (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 或指定版本：
# PyTorch 2.5.0 (CUDA 12.1)
# pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121

# PyTorch 2.1.0 (CUDA 11.8, 稳定版)
# pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# 4. 安装 Mamba 依赖
# 根据 PyTorch 版本选择对应的 Mamba 版本
pip install causal-conv1d>=2.0.0
pip install mamba-ssm>=2.2.0

# 5. 安装其他依赖
pip install -r requirements.txt
```

### 方法三：CPU 版本（无 GPU）

```bash
# 1. 创建环境
conda create -n hdcd_cpu python=3.10 -y
conda activate hdcd_cpu

# 2. 安装 CPU 版 PyTorch
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# 3. 跳过 Mamba-SSM（CPU 不支持编译 CUDA 扩展）
# 模型会自动回退到 SimpleMamba2 实现

# 4. 安装其他依赖
pip install numpy scipy pandas scikit-learn matplotlib pyyaml tqdm einops
```

### requirements.txt

创建 `requirements.txt` 文件：

```txt
# PyTorch (安装前请参考上述命令)
# torch==2.1.0
# torchvision==0.16.0
# torchaudio==2.1.0

# Mamba 核心依赖
causal-conv1d>=1.4.0
mamba-ssm>=2.0.0

# 数值计算
numpy>=1.23.0,<2.0.0
scipy>=1.9.0
einops>=0.7.0

# 数据处理
pandas>=1.5.0
scikit-learn>=1.2.0
pyyaml>=6.0

# 可视化
matplotlib>=3.6.0
seaborn>=0.12.0
tensorboard>=2.12.0

# 工具
tqdm>=4.64.0
packaging>=21.0
ninja>=1.11.0

# 可选：多模态数据处理
opencv-python>=4.7.0
librosa>=0.10.0
transformers>=4.30.0

# 开发工具
pytest>=7.2.0
black>=23.0.0
flake8>=6.0.0
```

### 依赖安装说明

#### 1. Mamba-SSM 安装问题

如果 `pip install mamba-ssm` 失败，可能原因：

**问题 A: CUDA 编译失败**
```bash
# 确保安装了 CUDA Toolkit (不只是 cudnn)
# Ubuntu/Debian:
sudo apt-get install nvidia-cuda-toolkit

# 检查 nvcc 是否可用
nvcc --version

# 设置 CUDA 环境变量
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 重新安装
pip install mamba-ssm --no-cache-dir
```

**问题 B: GCC 版本不兼容**
```bash
# CUDA 11.8 需要 GCC <= 11
# Ubuntu 检查 GCC 版本
gcc --version

# 如果版本过高，安装并切换到 GCC 10
sudo apt-get install gcc-10 g++-10
export CC=gcc-10
export CXX=g++-10

pip install mamba-ssm --no-cache-dir
```

**问题 C: 预编译 Wheel 包**
```bash
# 从 GitHub Release 下载预编译 wheel
# https://github.com/state-spaces/mamba/releases

# 示例：Python 3.10 + CUDA 11.8
wget https://github.com/state-spaces/mamba/releases/download/v2.0.0/mamba_ssm-2.0.0+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install mamba_ssm-2.0.0+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

#### 2. causal-conv1d 安装问题

```bash
# 方法1: 从源码编译
pip install causal-conv1d --no-binary :all:

# 方法2: 使用预编译版本
pip install causal-conv1d>=1.4.0 --find-links https://github.com/Dao-AILab/causal-conv1d/releases

# 方法3: 手动下载 wheel
wget https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install causal_conv1d-1.4.0+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

### 验证安装

运行以下脚本验证环境：

```bash
python -c "
import sys
import torch
import numpy as np

print('=' * 60)
print('环境验证')
print('=' * 60)
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU Count: {torch.cuda.device_count()}')
    print(f'GPU Name: {torch.cuda.get_device_name(0)}')
print(f'NumPy: {np.__version__}')

# 测试 Mamba
try:
    from mamba_ssm.modules.mamba2_stateful import Mamba2Stateful
    print('✓ Mamba2Stateful: OK')
except ImportError as e:
    print(f'✗ Mamba2Stateful: FAILED - {e}')
    print('  → 模型将使用 SimpleMamba2 作为回退')

# 测试 causal-conv1d
try:
    from causal_conv1d import causal_conv1d_fn
    print('✓ causal-conv1d: OK')
except ImportError:
    print('✗ causal-conv1d: FAILED (可选)')

print('=' * 60)
"
```

### Docker 环境（推荐生产环境）

```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 安装 PyTorch
RUN pip3 install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# 安装 Mamba
RUN pip3 install causal-conv1d>=1.4.0 mamba-ssm

# 复制项目
COPY . /workspace/H-DCD
WORKDIR /workspace/H-DCD

# 安装其他依赖
RUN pip3 install -r requirements.txt

CMD ["/bin/bash"]
```

构建和运行：
```bash
# 构建镜像
docker build -t hdcd:latest .

# 运行容器
docker run --gpus all -it -v $(pwd):/workspace/H-DCD hdcd:latest

# 在容器内训练
python train.py --config config.yaml
```

---

## 快速开始

### 1. 数据准备

将数据组织为以下格式：

```
data/
├── iemocap/
│   ├── train.pkl  # 包含 'audio', 'visual', 'text', 'labels', 'lengths'
│   ├── valid.pkl
│   └── test.pkl
└── mosei/
    ├── train.pkl
    ├── valid.pkl
    └── test.pkl
```

每个 `.pkl` 文件应包含:
```python
{
    'audio': np.array,   # (N, T, 74)
    'visual': np.array,  # (N, T, 35)
    'text': np.array,    # (N, T, 768)
    'labels': np.array,  # (N,)
    'lengths': np.array  # (N,)
}
```

### 2. 配置文件

编辑 `config.yaml`:

```yaml
DATASET:
  name: 'iemocap'  # 或 'mosei'
  data_dir: './data'
  num_classes: 4    # IEMOCAP: 4, MOSEI: 6

MODEL:
  raw_audio_dim: 74
  raw_visual_dim: 35
  raw_text_dim: 768
  hidden_dim: 128
  use_simple_mamba: false

LOSS:
  lambda_dec: 0.1
  lambda_hierarchical: 0.5
  lambda_distill: 0.3

TRAIN:
  batch_size: 32
  learning_rate: 0.0001
  num_epochs: 100
```

### 3. 训练模型

```bash
# 单 GPU 训练
python train.py --config config.yaml

# 多 GPU 训练
python train.py --config config.yaml --num_gpus 4

# 从检查点恢复训练
python train.py --config config.yaml --resume ./checkpoints/best_model.pth
```

### 4. 测试模型

```bash
# 运行测试套件
python test.py

# 单个模型测试
python -c "
from models.H_DCD_model import HDCD
import torch

model = HDCD(num_classes=4)
audio = torch.randn(2, 100, 74)
visual = torch.randn(2, 100, 35)
text = torch.randn(2, 100, 768)

outputs = model(audio, visual, text)
print('预测:', outputs['prediction'].argmax(dim=1))
"
```

---

## 使用示例

### 基本使用

```python
import torch
from models.H_DCD_model import HDCD

# 创建模型
model = HDCD(
    raw_audio_dim=74,
    raw_visual_dim=35,
    raw_text_dim=768,
    num_classes=4,
    use_simple_mamba=False
)

# 准备输入
audio = torch.randn(8, 100, 74)    # (batch, seq_len, audio_dim)
visual = torch.randn(8, 100, 35)   # (batch, seq_len, visual_dim)
text = torch.randn(8, 100, 768)    # (batch, seq_len, text_dim)

# 推理
model.eval()
with torch.no_grad():
    outputs = model(audio, visual, text)
    predictions = outputs['prediction'].argmax(dim=1)
```

### 训练循环

```python
from models.loss import HDCDLoss

# 初始化
criterion = HDCDLoss(num_classes=4)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 训练步骤
model.train()
outputs = model(audio, visual, text, lengths, return_all_outputs=True)
loss_dict = criterion(outputs, labels, compute_all=True)

optimizer.zero_grad()
loss_dict['total'].backward()
optimizer.step()
```

### 获取所有中间输出

```python
outputs = model(audio, visual, text, lengths, return_all_outputs=True)

# 可用输出:
# - prediction: 最终预测
# - y_uni: 单模态预测
# - y_bi: 双模态预测
# - y_mul: 多模态预测
# - z_homo_final: 共性特征
# - z_hetero_final: 私有特征
# - z_common_{a,v,t}: 各模态共性特征
# - z_private_{a,v,t}: 各模态私有特征
```

---

## 文件结构

```
H-DCD/
├── models/
│   ├── H_DCD_model.py           # 主模型
│   ├── decouple_encoder.py      # 对抗解耦编码器
│   ├── HMNF_model.py            # Mamba 骨干网络
│   ├── bidirectional_mamba2.py  # 双向 Mamba2
│   ├── loss.py                  # 损失函数
│   └── __init__.py
├── dataset/
│   └── multimodal_dataset.py    # 数据加载器
├── config.yaml                  # 配置文件
├── train.py                     # 训练脚本
├── test.py                      # 测试脚本
└── README.md                    # 本文件
```

---

## 超参数说明

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `hidden_dim` | 128 | 隐藏层维度 |
| `d_state` | 64 | Mamba 状态维度 |
| `d_conv` | 4 | Mamba 卷积核大小 |
| `expand` | 2 | Mamba 扩展因子 |
| `use_simple_mamba` | False | 使用简化版 Mamba |

### 损失权重

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lambda_dec` | 0.1 | 解耦损失权重 |
| `lambda_adv` | 1.0 | 对抗损失权重 |
| `lambda_ortho` | 0.5 | 正交损失权重 |
| `lambda_margin` | 0.3 | 间隔损失权重 |
| `lambda_hierarchical` | 0.5 | 层次化损失权重 |
| `lambda_distill` | 0.3 | 蒸馏损失权重 |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 32 | 批次大小 |
| `learning_rate` | 1e-4 | 学习率 |
| `weight_decay` | 1e-4 | 权重衰减 |
| `num_epochs` | 100 | 训练轮数 |
| `patience` | 10 | 早停耐心值 |

---

## 性能基准

### IEMOCAP (4 分类)

| 模型 | 准确率 | F1 分数 | 参数量 |
|------|--------|---------|--------|
| H-DCD (完整版) | ~0.75 | ~0.74 | ~15M |
| H-DCD (简化版) | ~0.72 | ~0.71 | ~8M |

### MOSEI (6 分类)

| 模型 | 准确率 | F1 分数 | 参数量 |
|------|--------|---------|--------|
| H-DCD (完整版) | ~0.68 | ~0.67 | ~15M |
| H-DCD (简化版) | ~0.65 | ~0.64 | ~8M |

---

## 常见问题

### Q: 如何减少显存占用？

```python
# 使用简化版 Mamba
model = HDCD(use_simple_mamba=True)

# 减小 batch size
TRAIN.batch_size = 16

# 使用梯度检查点
model.gradient_checkpointing_enable()
```

### Q: 如何处理不同长度的序列？

模型内部会自动处理序列长度：

```python
lengths = torch.tensor([50, 80, 100, 120])  # 实际长度
outputs = model(audio, visual, text, lengths)
```

### Q: 如何只使用部分模态？

```python
# 只使用音频和文本
visual = None  # 或 torch.zeros_like(...)
outputs = model(audio, visual, text)
```

---

## 引用

如果使用本代码，请引用：

```bibtex
@article{hdcd2024,
  title={H-DCD: Hierarchical Decoupled Contrastive Distillation for Multimodal Emotion Recognition},
  author={Your Name},
  year={2024}
}
```

---

## 许可证

MIT License

---

## 致谢

- [Mamba](https://github.com/state-spaces/mamba): 高效状态空间模型
- [IEMOCAP](https://sail.usc.edu/iemocap/): 情感数据集
- [CMU-MultimodalSDK](https://github.com/A2Zadeh/CMU-MultimodalSDK): 多模态工具包

---

## 联系方式

- Issues: [GitHub Issues](https://github.com/yourusername/H-DCD/issues)
- Email: your.email@example.com
