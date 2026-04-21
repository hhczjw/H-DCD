#!/bin/bash
# H-DCD 快速安装脚本 (Linux/Mac)
# 自动安装所有依赖

set -e  # 遇到错误立即退出

echo "========================================"
echo "H-DCD 自动安装脚本"
echo "========================================"
echo ""

# 检测操作系统
OS="$(uname -s)"
echo "检测到操作系统: $OS"
echo ""

# 询问用户选择
echo "请选择安装方式:"
echo "1) Conda 环境 (推荐)"
echo "2) pip 虚拟环境"
echo "3) 仅安装依赖 (已有 Python 环境)"
read -p "请输入选项 (1-3): " choice
echo ""

# 询问 CUDA 版本
echo "请选择 CUDA 版本:"
echo "1) CUDA 11.8 (推荐，兼容性最好)"
echo "2) CUDA 12.1"
echo "3) CPU 版本 (无 GPU)"
read -p "请输入选项 (1-3): " cuda_choice
echo ""

# 设置 PyTorch 安装命令
if [ "$cuda_choice" == "1" ]; then
    CUDA_VERSION="cu118"
    PYTORCH_INDEX="https://download.pytorch.org/whl/cu118"
    CONDA_CUDA="pytorch-cuda=11.8"
elif [ "$cuda_choice" == "2" ]; then
    CUDA_VERSION="cu121"
    PYTORCH_INDEX="https://download.pytorch.org/whl/cu121"
    CONDA_CUDA="pytorch-cuda=12.1"
else
    CUDA_VERSION="cpu"
    PYTORCH_INDEX="https://download.pytorch.org/whl/cpu"
    CONDA_CUDA=""
fi

echo "开始安装..."
echo ""

# 安装逻辑
if [ "$choice" == "1" ]; then
    # Conda 安装
    echo "使用 Conda 创建环境..."
    
    # 检查 conda 是否安装
    if ! command -v conda &> /dev/null; then
        echo "错误: 未找到 conda 命令"
        echo "请先安装 Anaconda 或 Miniconda"
        exit 1
    fi
    
    # 创建环境
    conda create -n hdcd python=3.10 -y
    
    echo ""
    echo "激活环境并安装 PyTorch..."
    
    # 安装 PyTorch
    if [ "$cuda_choice" == "3" ]; then
        # CPU 版本
        conda run -n hdcd pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url $PYTORCH_INDEX
    else
        # GPU 版本
        conda run -n hdcd conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 $CONDA_CUDA -c pytorch -c nvidia -y
    fi
    
    echo ""
    echo "安装 Mamba 依赖..."
    if [ "$cuda_choice" != "3" ]; then
        conda run -n hdcd pip install causal-conv1d>=1.4.0
        conda run -n hdcd pip install mamba-ssm
    else
        echo "跳过 Mamba-SSM (CPU 模式)"
    fi
    
    echo ""
    echo "安装其他依赖..."
    conda run -n hdcd pip install -r requirements.txt
    
    echo ""
    echo "========================================"
    echo "安装完成！"
    echo "========================================"
    echo ""
    echo "激活环境: conda activate hdcd"
    echo "验证安装: bash verify_environment.sh"
    
elif [ "$choice" == "2" ]; then
    # pip 虚拟环境
    echo "创建 Python 虚拟环境..."
    
    python3.10 -m venv hdcd_env
    source hdcd_env/bin/activate
    
    echo ""
    echo "升级 pip..."
    pip install --upgrade pip setuptools wheel
    
    echo ""
    echo "安装 PyTorch..."
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url $PYTORCH_INDEX
    
    echo ""
    echo "安装 Mamba 依赖..."
    if [ "$cuda_choice" != "3" ]; then
        pip install causal-conv1d>=1.4.0
        pip install mamba-ssm
    else
        echo "跳过 Mamba-SSM (CPU 模式)"
    fi
    
    echo ""
    echo "安装其他依赖..."
    pip install -r requirements.txt
    
    echo ""
    echo "========================================"
    echo "安装完成！"
    echo "========================================"
    echo ""
    echo "激活环境: source hdcd_env/bin/activate"
    echo "验证安装: bash verify_environment.sh"
    
else
    # 仅安装依赖
    echo "在当前环境安装依赖..."
    
    echo ""
    echo "安装 PyTorch..."
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url $PYTORCH_INDEX
    
    echo ""
    echo "安装 Mamba 依赖..."
    if [ "$cuda_choice" != "3" ]; then
        pip install causal-conv1d>=1.4.0
        pip install mamba-ssm
    else
        echo "跳过 Mamba-SSM (CPU 模式)"
    fi
    
    echo ""
    echo "安装其他依赖..."
    pip install -r requirements.txt
    
    echo ""
    echo "========================================"
    echo "安装完成！"
    echo "========================================"
    echo ""
    echo "验证安装: bash verify_environment.sh"
fi

echo ""
echo "下一步："
echo "1. 运行 bash verify_environment.sh 验证安装"
echo "2. 参考 README.md 准备数据集"
echo "3. 运行 python train.py --config config.yaml 开始训练"
echo ""
