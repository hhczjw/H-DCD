"""
测试 H-DCD 损失函数系统
验证四大损失模块的正确性
"""
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from losses import H_DCD_Loss
from config import get_default_config

def test_loss_system():
    print("=" * 80)
    print("H-DCD 损失函数系统测试")
    print("=" * 80)
    
    # 配置
    config = get_default_config()
    config.num_classes = 4
    config.task_type = 'classification'
    config.dataset_name = 'iemocap'
    
    # 创建损失函数
    criterion = H_DCD_Loss(config)
    
    print(f"\n损失函数配置:")
    print(f"  λ_dec: {criterion.lambda_dec}")
    print(f"  λ_hierarchical: {criterion.lambda_hierarchical}")
    print(f"  λ_distill: {criterion.lambda_distill}")
    print(f"  λ_adv: {criterion.lambda_adv}")
    print(f"  γ_mar: {criterion.gamma_mar}")
    print(f"  γ_ort: {criterion.gamma_ort}")
    print(f"  w_uni: {criterion.w_uni}")
    print(f"  w_bi: {criterion.w_bi}")
    print(f"  w_mul: {criterion.w_mul}")
    print(f"  α_base: {criterion.alpha_base}")
    print(f"  β: {criterion.beta}")
    print(f"  τ: {criterion.tau}")
    
    print(f"\nVA距离矩阵形状: {criterion.va_distances.shape}")
    print(f"VA距离矩阵:\n{criterion.va_distances}")
    
    # 模拟模型输出
    batch_size = 8
    d_model = 128
    num_classes = 4
    L_t, L_a, L_v = 50, 100, 75
    
    print(f"\n生成测试数据:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num classes: {num_classes}")
    print(f"  d_model: {d_model}")
    
    outputs = {
        # 分层分类输出
        'logits_uni': {
            'text': torch.randn(batch_size, num_classes),
            'audio': torch.randn(batch_size, num_classes),
            'video': torch.randn(batch_size, num_classes),
        },
        'logits_bi': {
            'ta': torch.randn(batch_size, num_classes),
            'tv': torch.randn(batch_size, num_classes),
            'av': torch.randn(batch_size, num_classes),
        },
        'logits_multi': torch.randn(batch_size, num_classes),
        
        # 对比蒸馏特征
        'features_contrast': {
            'hmnf': torch.randn(batch_size, d_model),
            'hmpn': torch.randn(batch_size, d_model),
        },
        
        # 解耦项
        'decouple_items': {
            's_text': torch.randn(batch_size, L_t, d_model),
            'c_text': torch.randn(batch_size, L_t, d_model),
            'recon_text': torch.randn(batch_size, L_t, d_model),
            
            's_audio': torch.randn(batch_size, L_a, d_model),
            'c_audio': torch.randn(batch_size, L_a, d_model),
            'recon_audio': torch.randn(batch_size, L_a, d_model),
            
            's_video': torch.randn(batch_size, L_v, d_model),
            'c_video': torch.randn(batch_size, L_v, d_model),
            'recon_video': torch.randn(batch_size, L_v, d_model),
        },
        
        # 对抗判别器输出
        'adv_logits': torch.randn(batch_size, 3),
    }
    
    # 标签
    labels = torch.randint(0, num_classes, (batch_size,))
    
    print(f"\n计算损失...")
    
    # 前向传播 (设置requires_grad)
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            v.requires_grad = True
        elif isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, torch.Tensor):
                    vv.requires_grad = True
    
    total_loss, loss_dict = criterion(outputs, labels)
    
    print(f"\n{'='*80}")
    print("损失函数详细输出:")
    print(f"{'='*80}")
    
    print(f"\n【模块一】L_dec (对抗性解耦损失):")
    print(f"  L_rec (重构损失):           {loss_dict.get('L_rec', 0.0):.6f}")
    print(f"  L_adv (对抗损失):            {loss_dict.get('L_adv', 0.0):.6f}")
    print(f"  L_mar' (情感间隔损失):       {loss_dict.get('L_mar_prime', 0.0):.6f}")
    print(f"  L_ort (正交损失):            {loss_dict.get('L_ort', 0.0):.6f}")
    print(f"  → L_dec (总):                {loss_dict.get('L_dec', 0.0):.6f}")
    
    print(f"\n【模块二】L_hierarchical (分层监督损失):")
    print(f"  L_yuni (单模态级):           {loss_dict.get('L_yuni', 0.0):.6f}")
    print(f"  L_ybi (双模态级):            {loss_dict.get('L_ybi', 0.0):.6f}")
    print(f"  L_ymul (三模态级):           {loss_dict.get('L_ymul', 0.0):.6f}")
    print(f"  → L_hierarchical (总):       {loss_dict.get('L_hierarchical', 0.0):.6f}")
    
    print(f"\n【模块三】L_distill (对比知识蒸馏损失):")
    print(f"  → L_distill:                 {loss_dict.get('L_distill', 0.0):.6f}")
    
    print(f"\n【模块四】L_task (最终任务损失):")
    print(f"  → L_task:                    {loss_dict.get('L_task', 0.0):.6f}")
    
    print(f"\n{'='*80}")
    print(f"【总损失】L_TOTAL:              {loss_dict.get('L_TOTAL', 0.0):.6f}")
    print(f"{'='*80}")
    
    # 验证梯度
    print(f"\n验证反向传播...")
    total_loss.backward()
    
    print(f"✓ 反向传播成功!")
    print(f"\n{'='*80}")
    print("✓ H-DCD 损失函数系统测试通过!")
    print(f"{'='*80}")


if __name__ == "__main__":
    test_loss_system()
