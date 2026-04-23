"""
Configuration file for H-DCD
"""
import json
import os
from pathlib import Path
from easydict import EasyDict as edict


def get_config(dataset_name, config_file=""):
    """
    Get the config of given dataset from config file.

    Parameters:
        dataset_name (str): Name of dataset (mosi, mosei, iemocap, meld)
        config_file (str): Path to config file, if empty, use default config file.

    Returns:
        config (edict): config of the given dataset
    """
    if config_file == "":
        config_file = Path(__file__).parent / "config" / "config.json"
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config_all = json.load(f)
    
    # Get dataset-specific config
    if dataset_name not in config_all['datasets']:
        raise ValueError(f"Dataset {dataset_name} not found in config file")
    
    dataset_config = config_all['datasets'][dataset_name]
    common_config = config_all['common']
    
    # Merge configs
    config = {}
    config['dataset_name'] = dataset_name
    config.update(common_config)
    config.update(dataset_config)
    
    # Update feature path to absolute path
    if 'dataset_root_dir' in config_all:
        config['featurePath'] = os.path.join(
            config_all['dataset_root_dir'], 
            config['featurePath']
        )
    
    config = edict(config)
    return config


def get_default_config():
    """
    Get default configuration for H-DCD
    """
    config = {
        # Model parameters
        'd_model': 128,
        'num_classes': 4,  # for emotion classification
        
        # Feature Projection
        'text_input_dim': 768,  # BERT
        'audio_input_dim': 74,  # ComParE
        'video_input_dim': 35,  # DenseFace
        
        # Decouple Encoder
        'decouple_num_layers': 2,
        'decouple_kernel_size': 3,
        
        # HMNF parameters
        'hmnf_d_state': 64,
        'hmnf_d_conv': 4,
        'hmnf_expand': 2,
        'hmnf_num_heads': 4,
        
        # HMPN parameters
        'hmpn_d_state': 64,
        'hmpn_d_conv': 4,
        'hmpn_expand': 2,
        'hmpn_num_heads': 4,
        
        # Training parameters
        'batch_size': 32,
        'learning_rate': 1e-4,
        'num_epochs': 100,
        'patience': 10,
        'grad_clip': 1.0,
        'weight_decay': 1e-4,
        
        # Loss weights
        'lambda_uni': 1.0,      # 单模态分类损失
        'lambda_bi': 1.0,       # 双模态分类损失
        'lambda_multi': 1.0,    # 多模态分类损失
        'lambda_recon': 0.1,    # 重构损失
        'lambda_adv': 0.1,      # 对抗损失
        'lambda_contrast': 0.5, # 对比蒸馏损失
        
        # Data parameters
        'need_data_aligned': True,
        'need_normalized': False,
        'use_bert': True,
        
        # Sequence lengths (if truncation needed)
        'seq_lens': [50, 100, 75],  # [text, audio, video]
        
        # Other
        'num_workers': 4,
        'device': 'cuda',
        'seed': 1111,
        
        # === AtCAF 创新点默认参数 ===
        # [创新1] SS-CD 因果去偏 (Mamba2化)
        'use_causal_debias': True,
        'debias_num_layers': 2,
        'debias_confounder_size': 50,
        'debias_d_state': 64,
        'debias_headdim': 32,
        'debias_text': True,
        'debias_audio': True,
        'debias_video': True,
        'confounder_npy_dir': None,
        # [创新2] SCI 反事实推断 (Mamba2化)
        'use_counterfactual': True,
        'counterfactual_type': 'shuffle',
        'counterfactual_num_layers': 2,
        'counterfactual_d_state': 64,
        'counterfactual_headdim': 32,
        'lambda_counterfactual': 0.5,
        # [创新3] 互信息约束
        'use_mutual_info': True,
        'add_va_mi': True,
        'cpc_layers': 1,
        'alpha_nce': 0.1,
        'beta_lld': 0.1,
        'mi_warmup_epochs': 5,
        'mi_memory_size': 10,
    }
    return edict(config)
