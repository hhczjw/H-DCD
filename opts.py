import argparse

def return_args():
    parser = argparse.ArgumentParser(description="Multimodal Sentiment Analysis Configuration")

    # =================================================================================
    # 1. 基础环境与模式配置
    # =================================================================================
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'valid', 'test'],
                        help='运行模式')
    parser.add_argument('--model_name', type=str, default='lf_dnn',
                        help='模型名称，例如: tfn, lmf, misa, lf_dnn 等')
    parser.add_argument('--seed', type=int, default=1111,
                        help='随机种子')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader 的线程数 (调试建议设为0)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='运行设备: cpu 或 cuda')

    # =================================================================================
    # 2. 数据集路径与名称配置
    # =================================================================================
    parser.add_argument('--dataset_name', type=str, default='mosi',
                        choices=['mosi', 'mosei', 'sims', 'sims2', 'dvlog', 'lmvd'],
                        help='数据集名称')
    
    parser.add_argument('--featurePath', type=str, default='datasets/mosi/mosi_data.pkl',
                        help='数据集特征文件的绝对路径或文件夹路径')

    # 外部特征路径 (用于替换默认特征，标准数据集使用)
    parser.add_argument('--feature_T', type=str, default='', help='自定义文本特征路径')
    parser.add_argument('--feature_A', type=str, default='', help='自定义音频特征路径')
    parser.add_argument('--feature_V', type=str, default='', help='自定义视觉特征路径')

    # =================================================================================
    # 3. 训练超参数
    # =================================================================================
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='权重衰减')
    parser.add_argument('--grad_clip', type=float, default=-1.0,
                        help='梯度裁剪阈值 (小于0表示不裁剪)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early Stopping 的耐心轮数')

    # =================================================================================
    # 4. 数据预处理配置 (标准数据集: MOSI/MOSEI/SIMS)
    # =================================================================================
    parser.add_argument('--use_bert', action='store_true',
                        help='是否使用 BERT 提取的文本特征')
    parser.add_argument('--need_data_aligned', action='store_true',
                        help='是否使用对齐后的数据 (Word-aligned)')
    parser.add_argument('--need_normalized', action='store_true',
                        help='是否需要对特征进行归一化')
    parser.add_argument('--need_truncated', action='store_true',
                        help='是否需要截断变长序列')

    # =================================================================================
    # 5. 自定义数据集专用配置 (DVlog / LMVD)
    #    对应 data_loader.py 中的 self.gender 和 self.aug
    # =================================================================================
    parser.add_argument('--gender', type=str, default='both', choices=['male', 'female', 'both'],
                        help='[DVlog专用] 性别过滤: male, female 或 both')
    
    parser.add_argument('--aug', action='store_true',
                        help='[DVlog/LMVD专用] 是否在训练时开启随机裁剪数据增强')

    # =================================================================================
    # 6. 解耦编码器 (DecoupleEncoder) 结构参数
    # =================================================================================
    parser.add_argument('--shared_dim', type=int, default=256,
                        help='DecoupleEncoder 共享编码器输出维度')
    parser.add_argument('--common_dim', type=int, default=128,
                        help='DecoupleEncoder 通用特征维度（跨模态共享的情感表示）')
    parser.add_argument('--private_dim', type=int, default=128,
                        help='DecoupleEncoder 特有特征维度（模态独有的补充信息）')
    parser.add_argument('--decouple_dropout', type=float, default=0.1,
                        help='DecoupleEncoder 各子模块默认 Dropout')
    parser.add_argument('--lambda_grl', type=float, default=1.0,
                        help='梯度反转层 (GRL) 系数，控制对抗训练强度')
    parser.add_argument('--temporal_kernel_size', type=int, default=3,
                        help='时序卷积核大小（奇数），用于捕获局部时序 n-gram 模式')
    parser.add_argument('--discriminator_hidden_dim', type=int, default=128,
                        help='模态判别器内部隐藏层维度')
    
    # 情感感知对比损失参数
    parser.add_argument('--margin_alpha_base', type=float, default=0.5,
                        help='情感间隔损失的基础间隔值')
    parser.add_argument('--margin_beta', type=float, default=0.2,
                        help='情感间隔损失的动态间隔系数（VA空间距离权重）')
    parser.add_argument('--margin_max_label', type=float, default=3.0,
                        help='情感标签的最大绝对值（用于 VA 空间归一化）')

    # =================================================================================
    # 7. 损失函数权重配置
    # =================================================================================
    parser.add_argument('--loss_weight_reconstruction', type=float, default=1.0,
                        help='重构损失权重')
    parser.add_argument('--loss_weight_adversarial', type=float, default=0.1,
                        help='对抗损失权重')
    parser.add_argument('--loss_weight_orthogonal', type=float, default=0.01,
                        help='正交损失权重（通用/特有特征独立性约束）')
    parser.add_argument('--loss_weight_margin', type=float, default=0.5,
                        help='情感间隔损失权重')

    # =================================================================================
    # 8. [创新1] SS-CD 状态空间因果去偏参数 (State-Space Causal Debiasing)
    # =================================================================================
    parser.add_argument('--use_causal_debias', action='store_true', default=True,
                        help='启用 SS-CD 因果去偏模块 (默认开启)')
    parser.add_argument('--no_causal_debias', dest='use_causal_debias', action='store_false',
                        help='禁用 SS-CD 因果去偏模块')
    parser.add_argument('--debias_num_layers', type=int, default=2,
                        help='SS-CD 双路径的 Mamba2 层数')
    parser.add_argument('--debias_confounder_size', type=int, default=50,
                        help='混杂因子字典大小（KMeans聚类数）')
    parser.add_argument('--debias_d_state', type=int, default=64,
                        help='SS-CD Mamba2 的状态维度 (d_state)')
    parser.add_argument('--debias_headdim', type=int, default=32,
                        help='SS-CD Mamba2 的头维度 (headdim)')
    parser.add_argument('--debias_text', action='store_true', default=True,
                        help='对文本模态去偏 (默认开启)')
    parser.add_argument('--no_debias_text', dest='debias_text', action='store_false',
                        help='禁用文本模态去偏')
    parser.add_argument('--debias_audio', action='store_true', default=True,
                        help='对音频模态去偏 (默认开启)')
    parser.add_argument('--no_debias_audio', dest='debias_audio', action='store_false',
                        help='禁用音频模态去偏')
    parser.add_argument('--debias_video', action='store_true', default=True,
                        help='对视频模态去偏 (默认开启)')
    parser.add_argument('--no_debias_video', dest='debias_video', action='store_false',
                        help='禁用视频模态去偏')
    parser.add_argument('--confounder_npy_dir', type=str, default=None,
                        help='KMeans聚类中心.npy文件目录路径（为空则随机初始化）')

    # =================================================================================
    # 9. [创新2] SCI 选择性反事实推断参数 (Selective Counterfactual Inference)
    # =================================================================================
    parser.add_argument('--use_counterfactual', action='store_true', default=True,
                        help='启用 SCI 反事实推断模块 (默认开启)')
    parser.add_argument('--no_counterfactual', dest='use_counterfactual', action='store_false',
                        help='禁用 SCI 反事实推断模块')
    parser.add_argument('--counterfactual_type', type=str, default='shuffle',
                        choices=['random', 'shuffle', 'reversed', 'uniform'],
                        help='反事实干预策略类型')
    parser.add_argument('--counterfactual_num_layers', type=int, default=2,
                        help='SCI Mamba2 层数')
    parser.add_argument('--counterfactual_d_state', type=int, default=64,
                        help='SCI Mamba2 的状态维度 (d_state)')
    parser.add_argument('--counterfactual_headdim', type=int, default=32,
                        help='SCI Mamba2 的头维度 (headdim)')
    parser.add_argument('--lambda_counterfactual', type=float, default=0.5,
                        help='反事实效应损失权重 (η)')

    # =================================================================================
    # 10. [创新3] 多模态互信息约束参数 (Mutual Information Constraint)
    # =================================================================================
    parser.add_argument('--use_mutual_info', action='store_true', default=True,
                        help='启用互信息约束 MMILB+CPC (默认开启)')
    parser.add_argument('--no_mutual_info', dest='use_mutual_info', action='store_false',
                        help='禁用互信息约束')
    parser.add_argument('--add_va_mi', action='store_true', default=True,
                        help='添加 visual-audio 互信息估计 (默认开启)')
    parser.add_argument('--no_va_mi', dest='add_va_mi', action='store_false',
                        help='禁用 visual-audio 互信息估计')
    parser.add_argument('--cpc_layers', type=int, default=1,
                        help='CPC预测网络层数')
    parser.add_argument('--alpha_nce', type=float, default=0.1,
                        help='CPC NCE损失权重 (α)')
    parser.add_argument('--beta_lld', type=float, default=0.1,
                        help='MMILB lld损失权重 (β)')
    parser.add_argument('--mi_warmup_epochs', type=int, default=5,
                        help='互信息预训练(阶段0)轮数: 仅训练MMILB最大化lld')
    parser.add_argument('--mi_memory_size', type=int, default=10,
                        help='MMILB memory缓存大小（保存最近N个batch的正负样本）')

    args = parser.parse_args()
    return args

def get_config():
    """
    获取参数并转换为字典格式，同时初始化必要的占位符参数。
    MMSA 框架通常使用字典 (args['key']) 而非对象 (args.key) 访问参数。
    
    Returns:
        config: dict, 包含所有配置参数和运行时占位符
    """
    args = return_args()
    config = vars(args)  # 将 Namespace 转换为字典

    # ========================================================================
    # 初始化运行时占位符（这些会在 DataLoader 初始化时被自动填充）
    # ========================================================================
    
    # 特征维度: [Text_Dim, Audio_Dim, Visual_Dim]
    # - 标准数据集 (MOSI/MOSEI/SIMS): 三个位置分别存储文本/音频/视觉维度
    # - DVlog/LMVD: 只使用 Audio 位 (index 1) 存储音频+视觉拼接后的维度
    config['feature_dims'] = [0, 0, 0]
    
    # 序列长度: [Text_Len, Audio_Len, Visual_Len]
    # 由 DataLoader 根据数据集实际序列长度自动填充
    config['seq_lens'] = [0, 0, 0]
    
    # ========================================================================
    # 数据集特定配置覆盖
    # ========================================================================
    
    # DVlog/LMVD 特殊处理：这些抑郁症检测数据集只有音频+视觉，无文本
    if config['dataset_name'] in ['dvlog', 'lmvd']:
        config['use_bert'] = False  # 强制关闭 BERT 特征提取
        config['need_data_aligned'] = False  # 这些数据集本身是 Frame-level 对齐的
        
    # ========================================================================
    # 参数一致性检查
    # ========================================================================
    
    # 确保卷积核大小为奇数（对称 padding）
    if config['temporal_kernel_size'] % 2 == 0:
        config['temporal_kernel_size'] += 1
        print(f"警告: temporal_kernel_size 已自动调整为奇数 {config['temporal_kernel_size']}")

    return config

# 用于测试 opts.py 是否正常
if __name__ == '__main__':
    cfg = get_config()
    print("配置参数加载成功:")
    for k, v in cfg.items():
        print(f"{k}: {v}")