"""
Training script for H-DCD
"""
import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import yaml
from tqdm import tqdm

from run import H_DCD_run
from losses import H_DCD_Loss as HDCDLoss

# Train on IEMOCAP (emotion classification)
H_DCD_run(
    dataset_name='mosi',
    seeds=[1111, 1112, 1113],
    model_save_dir="./checkpoints",
    res_save_dir="./results",
    log_dir="./logs",
    mode='train',
    gpu_ids=[0],
    num_workers=4,
    verbose_level=1
)

# Uncomment below to train on other datasets:

# Train on MOSI (sentiment regression)
# H_DCD_run(
#     dataset_name='mosi',
#     seeds=[1111],
#     model_save_dir="./checkpoints",
#     res_save_dir="./results",
#     log_dir="./logs",
#     mode='train',
#     gpu_ids=[0]
# )

# Train on MOSEI (sentiment regression)
# H_DCD_run(
#     dataset_name='mosei',
#     seeds=[1111],
#     model_save_dir="./checkpoints",
#     res_save_dir="./results",
#     log_dir="./logs",
#     mode='train',
#     gpu_ids=[0]
# )

# Train on MELD (emotion classification)
# H_DCD_run(
#     dataset_name='meld',
#     seeds=[1111],
#     model_save_dir="./checkpoints",
#     res_save_dir="./results",
#     log_dir="./logs",
#     mode='train',
#     gpu_ids=[0]
# )

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='H-DCD Training')
    
    # 数据集
    parser.add_argument('--dataset', type=str, default='mosi', choices=['mosi', 'mosei'])
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    
    # 其他
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='./checkpoints/hdcd')
    parser.add_argument('--log_dir', type=str, default='./logs/hdcd')
    
    return parser.parse_args()


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_logger(log_dir):
    """设置日志"""
    import logging
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path):
    """加载配置文件"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def build_hdcd_model(config):
    """构建 H-DCD 模型"""
    from models.h_dcd import HDCD
    return HDCD(**config)


def get_dataloader(dataset, split, batch_size, num_workers, data_dir):
    """获取数据加载器"""
    from dataset.data_loader import get_loader
    return get_loader(dataset, split, batch_size, num_workers, data_dir)


def train_epoch(model, dataloader, criterion, optimizer, device, args, logger):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        # 获取数据
        audio = batch['audio'].to(device)
        visual = batch['visual'].to(device)
        text = batch['text'].to(device)
        labels = batch['label'].to(device)
        lengths = batch.get('length', None)
        if lengths is not None:
            lengths = lengths.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(audio, visual, text, lengths, return_all_outputs=True)
        
        # 计算损失
        loss_dict = criterion(outputs, labels, compute_all=True)
        loss = loss_dict['total']
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        _, predicted = outputs['prediction'].max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
    
    return total_loss / len(dataloader), 100. * correct / total


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            audio = batch['audio'].to(device)
            visual = batch['visual'].to(device)
            text = batch['text'].to(device)
            labels = batch['label'].to(device)
            lengths = batch.get('length', None)
            if lengths is not None:
                lengths = lengths.to(device)
            
            outputs = model(audio, visual, text, lengths, return_all_outputs=False)
            loss_dict = criterion(outputs, labels, compute_all=False)
            loss = loss_dict['total']
            
            total_loss += loss.item()
            _, predicted = outputs['prediction'].max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    from sklearn.metrics import f1_score
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return total_loss / len(dataloader), 100. * correct / total, f1


def main():
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置种子
    set_seed(args.seed)
    
    # 设置日志
    logger = setup_logger(args.log_dir)
    logger.info(f"Arguments: {args}")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 构建模型
    logger.info("Building model...")
    model_config = config.get('MODEL', {})
    model_config.update({
        'num_classes': args.num_classes,
        'use_simple_mamba': args.use_simple_mamba
    })
    
    model = build_hdcd_model(model_config)
    model = model.to(args.device)
    
    params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {params / 1e6:.2f}M")
    
    # 构建数据加载器
    logger.info("Loading data...")
    try:
        train_loader = get_dataloader(args.dataset, 'train', args.batch_size, args.num_workers, args.data_dir)
        val_loader = get_dataloader(args.dataset, 'val', args.batch_size, args.num_workers, args.data_dir)
        test_loader = get_dataloader(args.dataset, 'test', args.batch_size, args.num_workers, args.data_dir)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        logger.info("Please set up your dataset properly.")
        return
    
    # 损失函数
    loss_config = config.get('LOSS', {})
    criterion = HDCDLoss(num_classes=args.num_classes, **loss_config)
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 训练循环
    best_val_acc = 0
    best_val_f1 = 0
    
    logger.info("Start training...")
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, args.device, args, logger)
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, args.device)
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'args': args,
            }, os.path.join(args.save_dir, 'best_model.pt'))
            logger.info(f"Saved best model")
        
        scheduler.step()
    
    # 测试
    logger.info("\nTesting best model...")
    checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, args.device)
    logger.info(f"Test: Loss={test_loss:.4f}, Acc={test_acc:.2f}%, F1={test_f1:.4f}")


if __name__ == '__main__':
    main()
