"""
Main training and testing script for H-DCD
Based on DMD framework
"""
import gc
import logging
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from config import get_config, get_default_config
from trainer import H_DCD_Trainer
from losses import H_DCD_Loss
import warnings
warnings.filterwarnings('ignore')

# Add dataset path
sys.path.insert(0, str(Path(__file__).parent / 'dataset'))
from data_loader import MMDataLoader

# Add models path  
sys.path.insert(0, str(Path(__file__).parent / 'models'))
from h_dcd import H_DCD

# Setup CUDA
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"

logger = logging.getLogger('H-DCD')


def _set_logger(log_dir, dataset_name, verbose_level=1):
    """Setup logger"""
    log_file_path = Path(log_dir) / f"h-dcd-{dataset_name}.log"
    logger = logging.getLogger('H-DCD')
    logger.setLevel(logging.DEBUG)

    # File handler
    fh = logging.FileHandler(log_file_path)
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # Stream handler
    stream_level = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}
    ch = logging.StreamHandler()
    ch.setLevel(stream_level[verbose_level])
    ch_formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger


def assign_gpu(gpu_ids):
    """Assign GPU device"""
    if len(gpu_ids) == 0:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_ids[0])
    return device


def setup_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def H_DCD_run(
    dataset_name,
    config=None,
    config_file="",
    seeds=[],
    model_save_dir="",
    res_save_dir="",
    log_dir="",
    gpu_ids=[0],
    num_workers=4,
    verbose_level=1,
    mode='train'
):
    """
    Main function to run H-DCD training or testing
    
    Args:
        dataset_name: 'mosi', 'mosei', 'iemocap', 'meld'
        config: config dict (optional)
        config_file: path to config file
        seeds: list of random seeds
        model_save_dir: directory to save models
        res_save_dir: directory to save results
        log_dir: directory for logs
        gpu_ids: list of GPU IDs
        num_workers: number of data loading workers
        verbose_level: 0=ERROR, 1=INFO, 2=DEBUG
        mode: 'train' or 'test'
    """
    # Initialization
    dataset_name = dataset_name.lower()
    
    # Config file
    if config_file != "":
        config_file = Path(config_file)
    else:
        config_file = Path(__file__).parent / "config" / "config.json"
    
    # Directories
    if model_save_dir == "":
        model_save_dir = Path(__file__).parent / "checkpoints"
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
    
    if res_save_dir == "":
        res_save_dir = Path(__file__).parent / "results"
    Path(res_save_dir).mkdir(parents=True, exist_ok=True)
    
    if log_dir == "":
        log_dir = Path(__file__).parent / "logs"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Seeds
    seeds = seeds if seeds != [] else [1111, 1112, 1113, 1114, 1115]
    
    # Logger
    logger = _set_logger(log_dir, dataset_name, verbose_level)
    
    # Get config
    if config_file.exists():
        args = get_config(dataset_name, config_file)
    else:
        logger.warning(f"Config file {config_file} not found, using default config")
        args = get_default_config()
        args.dataset_name = dataset_name
    
    # Override with custom config
    if config:
        args.update(config)
    
    # Additional args
    args['model_save_dir'] = str(model_save_dir)
    args['device'] = assign_gpu(gpu_ids)
    args['mode'] = mode
    
    logger.info("=" * 80)
    logger.info(f"H-DCD - {dataset_name.upper()}")
    logger.info(f"Mode: {mode}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Seeds: {seeds}")
    logger.info("=" * 80)
    
    # Run multiple seeds
    model_results = []
    for i, seed in enumerate(seeds):
        logger.info(f"\n{'='*80}")
        logger.info(f"Running seed {i+1}/{len(seeds)}: {seed}")
        logger.info(f"{'='*80}")
        
        setup_seed(seed)
        args['cur_seed'] = i + 1
        
        result = _run(args, num_workers, mode)
        model_results.append(result)
        
        # Clean up
        gc.collect()
        torch.cuda.empty_cache()
    
    # Save results
    if mode == 'train':
        _save_results(model_results, dataset_name, res_save_dir, logger)
    
    return model_results


def _run(args, num_workers=4, mode='train'):
    """Single run with one seed"""
    
    # Create data loader
    logger.info("Loading data...")
    dataloader = MMDataLoader(args, num_workers)
    
    # Create model
    logger.info("Creating model...")
    model = H_DCD(
        # 原有参数
        d_model=args.d_model,
        num_classes=args.num_classes,
        text_input_dim=args.text_input_dim,
        audio_input_dim=args.audio_input_dim,
        video_input_dim=args.video_input_dim,
        hmnf_d_state=args.hmnf_d_state,
        hmnf_d_conv=args.hmnf_d_conv,
        hmnf_expand=args.hmnf_expand,
        hmnf_num_layers=args.get('hmnf_num_layers', 1),
        hmpn_d_state=args.hmpn_d_state,
        hmpn_d_conv=args.hmpn_d_conv,
        hmpn_expand=args.hmpn_expand,
        hmpn_num_heads=args.hmpn_num_heads,
        dropout=args.get('dropout', 0.1),
        # === [创新1] SS-CD 因果去偏参数 (Mamba2化) ===
        use_causal_debias=args.get('use_causal_debias', True),
        debias_num_layers=args.get('debias_num_layers', 2),
        debias_confounder_size=args.get('debias_confounder_size', 50),
        debias_d_state=args.get('debias_d_state', 64),
        debias_headdim=args.get('debias_headdim', 32),
        debias_text=args.get('debias_text', True),
        debias_audio=args.get('debias_audio', True),
        debias_video=args.get('debias_video', True),
        confounder_npy_dir=args.get('confounder_npy_dir', None),
        dataset_name=args.get('dataset_name', 'mosi'),
        # === [创新2] SCI 反事实推断参数 (Mamba2化) ===
        use_counterfactual=args.get('use_counterfactual', True),
        counterfactual_type=args.get('counterfactual_type', 'shuffle'),
        counterfactual_num_layers=args.get('counterfactual_num_layers', 2),
        counterfactual_d_state=args.get('counterfactual_d_state', 64),
        counterfactual_headdim=args.get('counterfactual_headdim', 32),
        # === [创新3] 互信息约束参数 ===
        use_mutual_info=args.get('use_mutual_info', True),
        add_va_mi=args.get('add_va_mi', True),
        cpc_layers=args.get('cpc_layers', 1),
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # Create trainer
    trainer = H_DCD_Trainer(args, model, args.device)
    
    if mode == 'train':
        # Train
        logger.info("\nStarting training...")
        trainer.train(dataloader['train'], dataloader['valid'])
        
        # Test with best model
        logger.info("\nTesting with best model...")
        results = trainer.test(dataloader['test'])
    else:
        # Test only
        model_path = Path(args.model_save_dir) / 'best_model.pth'
        if model_path.exists():
            logger.info(f"Loading model from {model_path}")
            trainer.load_model(str(model_path))
        else:
            logger.error(f"Model file not found: {model_path}")
            return None
        
        logger.info("\nTesting...")
        results = trainer.test(dataloader['test'])
    
    return results


def _save_results(model_results, dataset_name, res_save_dir, logger):
    """Save results to CSV"""
    res_save_dir = Path(res_save_dir)
    res_save_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine metrics
    if len(model_results) == 0 or model_results[0] is None:
        logger.warning("No results to save")
        return
    
    criterions = list(model_results[0].keys())
    
    # Save result to csv
    csv_file = res_save_dir / f"{dataset_name}.csv"
    if csv_file.is_file():
        df = pd.read_csv(csv_file)
    else:
        df = pd.DataFrame(columns=["Model"] + criterions)
    
    # Compute mean and std
    res = ["h-dcd"]
    for c in criterions:
        values = [r[c] for r in model_results if r is not None]
        if len(values) > 0:
            mean = round(np.mean(values), 4)
            std = round(np.std(values), 4)
            res.append(f"{mean:.4f}±{std:.4f}")
        else:
            res.append("N/A")
    
    df.loc[len(df)] = res
    df.to_csv(csv_file, index=None)
    logger.info(f"Results saved to {csv_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='H-DCD Training/Testing')
    parser.add_argument('--dataset', type=str, default='iemocap',
                       choices=['mosi', 'mosei', 'iemocap', 'meld'],
                       help='Dataset name')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'test'],
                       help='Run mode')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0],
                       help='GPU IDs')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1111],
                       help='Random seeds')
    parser.add_argument('--config', type=str, default='',
                       help='Config file path')
    parser.add_argument('--model_dir', type=str, default='./checkpoints',
                       help='Model save directory')
    parser.add_argument('--res_dir', type=str, default='./results',
                       help='Results save directory')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Log directory')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--verbose', type=int, default=1, choices=[0, 1, 2],
                       help='Verbose level: 0=ERROR, 1=INFO, 2=DEBUG')
    
    args = parser.parse_args()
    
    H_DCD_run(
        dataset_name=args.dataset,
        config_file=args.config,
        seeds=args.seeds,
        model_save_dir=args.model_dir,
        res_save_dir=args.res_dir,
        log_dir=args.log_dir,
        gpu_ids=args.gpu,
        num_workers=args.num_workers,
        verbose_level=args.verbose,
        mode=args.mode
    )
