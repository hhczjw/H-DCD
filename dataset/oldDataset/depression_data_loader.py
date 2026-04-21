# ============================================================================
# LMVD 数据集加载器
# 作用：从磁盘读取 LMVD 数据集的音频和视频特征，创建用于训练的数据集
# 
# 与 DVlog 的区别：
# 1. LMVD 的音视频特征存储在独立的目录中（'visual/' 和 'audio/'）
# 2. 音频特征维度不同：LMVD 用 128 维（VGGish），DVlog 用 25 维（LDDs）
# 3. 标签直接是数值（0 或 1），不需要字符串比较
# 4. 文件夹结构不同：DVlog 是按样本 ID 分类，LMVD 是按模态分类
# ============================================================================

from pathlib import Path                    # 用于跨平台路径操作
from typing import Union, Optional          # 类型提示

import torch                                # PyTorch 张量和神经网络
from torch.utils import data                # PyTorch 数据集基类
from torch.nn.utils.rnn import pad_sequence # 批处理时对变长序列进行零填充
import numpy as np                          # NumPy 数组操作
import random                               # 随机数生成（数据增强）

# ============================================================================
# 类 LMVD：LMVD 数据集类，继承 PyTorch 的 Dataset 基类
# ============================================================================
class DepressionDataSet(data.Dataset):
    def __init__(
        self, 
        data_set_name: str,
        root: Union[str, Path],              # 数据集根目录，包含 labels.csv、visual/ 和 audio/ 文件夹
        fold: str="train",                   # 数据分割：train / valid / test
        gender: str="both",                  # 性别过滤（LMVD 中实际未使用，保留用于接口一致性）
        transform=None,                      # 特征变换函数（可选）
        target_transform=None,               # 标签变换函数（可选）
        aug=False                            # 是否进行数据增强
    ):
        # ====================================================================
        # 初始化实例变量
        # ====================================================================
        self.data_set_name = data_set_name  # 数据集名称（用于区分 LMVD 和其他数据集）
        # 将根目录转换为 Path 对象（便于跨平台路径操作）
        self.root = root if isinstance(root, Path) else Path(root)
        
        # 保存数据集配置
        self.fold = fold              # 当前使用的分割（train/valid/test）
        self.gender = gender          # 性别过滤条件（LMVD 中可能未使用）
        self.transform = transform    # 特征预处理函数
        self.target_transform = target_transform  # 标签预处理函数
        self.aug = aug                # 是否启用数据增强

        # ====================================================================
        # 初始化存储数据的列表
        # ====================================================================
        
        self.features = []            # 存储所有样本的特征（numpy 数组）
        self.labels = []              # 存储所有样本的标签（0 或 1）


        # ====================================================================
        # 从 CSV 文件读取样本元信息
        # ====================================================================
        
        with open(self.root / "labels.csv", "r") as f:
            # 逐行读取 CSV 文件
            # LMVD CSV 格式假设为：id, label, fold
            # 注意：与 DVlog 不同，LMVD 的格式更简洁
            for line in f:
                # 去除行首尾空白，按逗号分割得到字段
                sample = line.strip().split(",")
                
                # ============================================================
                # 检查该样本是否满足过滤条件（分割）
                # ============================================================
                
                if self.is_sample(sample):
                    # 样本满足条件，开始处理
                    
                    # 获取样本 ID
                    s_id = sample[0]
                    
                    # ========================================================
                    # 跳过 CSV 文件的表头行
                    # ========================================================
                    
                    # LMVD 的 CSV 可能包含一个表头行，其中样本 ID 是 "index"
                    # 这一行需要跳过，不应该被加载为实际样本
                    if 'index' in s_id:
                        continue
                    
                    # ========================================================
                    # 直接从整数标签（与 DVlog 的字符串比较不同）
                    # ========================================================
                    
                    # LMVD 的标签是整数形式：0（正常）或 1（抑郁）
                    # DVlog 是字符串形式："depression" 或其他
                    if self.data_set_name == 'lmvd':
                        s_label = int(sample[1])
                    else:
                        s_label = int(sample[1] == "depression") 
                    
                    # 将标签添加到列表
                    self.labels.append(s_label)

                    # ========================================================
                    # 构建特征文件路径（目录结构不同）
                    # ========================================================
                    
                    # LMVD 的目录结构：
                    # root/
                    #   labels.csv
                    #   visual/
                    #     sample_id1_visual.npy
                    #     sample_id2_visual.npy
                    #     ...
                    #   audio/
                    #     sample_id1.npy
                    #     sample_id2.npy
                    #     ...
                    
                    # 与 DVlog 的区别：
                    # DVlog: root/id/id_visual.npy 和 root/id/id_acoustic.npy
                    # LMVD:  root/visual/id_visual.npy 和 root/audio/id.npy
                    if self.data_set_name == 'lmvd':
                        v_feature_path = self.root / 'visual' / f"{s_id}_visual.npy"   # 视频特征路径
                        a_feature_path = self.root / 'audio' / f"{s_id}.npy"          # 音频特征路径
                    else:
                        v_feature_path = self.root / s_id / f"{s_id}_visual.npy"    # 视频特征路径
                        a_feature_path = self.root / s_id / f"{s_id}_acoustic.npy"
                    # 从磁盘加载两个模态的特征--LMVD
                    # v_feature: [T_v, 136]  (136 维视频特征，T_v 个时间步)
                    # a_feature: [T_a, 128]  (128 维音频特征，T_a 个时间步 - 与 DVlog 不同！)
                    # dv -log 
                    # v_feature: [T_v, 136]  (136 维视频特征，T_v 个时间步)
                    # a_feature: [T_a, 25]   (25 维音频特征，T_a 个时间步)
                    v_feature = np.load(v_feature_path)
                    a_feature = np.load(a_feature_path)
                    
                    # ========================================================
                    # 处理两个模态长度不一致的情况（与 DVlog 完全相同）
                    # ========================================================
                    
                    # 获取两个模态的时间步数
                    T_v, T_a = v_feature.shape[0], a_feature.shape[0]
                    
                    # 情况 1：两个模态长度相同，直接拼接
                    if T_v == T_a:
                        # 按特征维度（第 1 轴）拼接：[T, 136] + [T, 128] = [T, 264]
                        # 注意：LMVD 的总特征维度是 264 = 136 + 128（vs DVlog 的 161 = 136 + 25）
                        feature = np.concatenate(
                            (v_feature, a_feature), axis=1
                        ).astype(np.float32)  # 转换为 float32 类型
                    
                    # 情况 2：两个模态长度不同，截断到较短者
                    else:
                        # 取两者中较短的长度
                        T = min(T_v, T_a)
                        
                        # 都截断到长度 T，然后拼接
                        # [T, 136] + [T, 128] = [T, 264]
                        feature = np.concatenate(
                            (v_feature[:T], a_feature[:T]), axis=1
                        ).astype(np.float32)
                    
                    # 将处理后的特征添加到列表
                    self.features.append(feature)

                    # ========================================================
                    # 数据增强（仅在训练集进行 - 与 DVlog 相同逻辑）
                    # ========================================================
                    
                    if self.aug and self.fold == 'train':
                        # 获取当前样本的总时间步数
                        t_length = feature.shape[0]
                        
                        # 随机裁剪 5 次（为训练集创建 5 个增强样本）
                        for i in range(5):
                            # 随机生成裁剪长度（0 到总长度之间的随机数）
                            f_length = int(random.random() * t_length)
                            
                            # 如果裁剪长度小于 400，跳过（太短的片段无用）
                            # 注意：LMVD 使用 400 作为最小长度阈值
                            # DVlog 使用 500 作为最小长度阈值（因为 LMVD 视频通常更短）
                            if self.data_set_name == 'lmvd':
                                if f_length < 400:
                                    continue
                            else:
                                if f_length < 500:
                                    continue
                            
                            # 随机生成裁剪的起始位置
                            # 确保不会超出边界：起始位置在 [1, t_length - f_length] 范围内
                            t_start = random.randint(1, t_length - f_length)
                            
                            # 添加增强后的标签和特征
                            self.labels.append(s_label)  # 标签保持不变
                            self.features.append(feature[t_start:t_start+f_length, :])  # 随机时间片段

        # ====================================================================
        # 打印数据集统计信息
        # ====================================================================
        
        # 统计样本总数和正负类比例
        print(
            f"ALL:{len(self.labels)}, "                                      # 总样本数
            f"Positive:{np.sum(self.labels)}, "                              # 正类（抑郁症）样本数
            f"Negative:{len(self.labels) - np.sum(self.labels)}"             # 负类（正常）样本数
        )

    # ========================================================================
    # 方法：is_sample() - 检查样本是否满足过滤条件
    # ========================================================================
    
    def is_sample(self, sample) -> bool:
        """
        判断样本是否应该被包含在当前数据集中。
        
        参数:
            sample: list，从 CSV 行分割得到的字段列表
                    格式: [id, label, fold]
        
        返回:
            bool，True 表示样本满足条件，False 表示不满足
        
        与 DVlog 的区别：
        - DVlog: 需要检查性别 (sample[3]) 和 fold (sample[4])
        - LMVD: 只需要检查 fold (sample[2])，且没有性别过滤
        """
        
        # 从样本字段中提取数据集分割信息
        # LMVD 的 CSV 格式：[id, label, fold]
        # 索引 2 位置是 fold 字段
        if self.data_set_name == 'lmvd':
            fold = sample[2]
            return fold == self.fold
        else:
            gender = sample[3]
            fold = sample[4]
        
         # 情况 1：性别不受限制（"both"），只需检查分割是否匹配
        if self.gender == "both":
            # 只有当样本的 fold 与初始化时指定的 fold 相同，才返回 True
            return fold == self.fold
        
        # 情况 2：性别受限制（"m" 或 "f"），需要同时检查性别和分割
        # 只有当性别匹配 AND 分割匹配时，才返回 True
        return (fold == self.fold) and (gender == self.gender)
        


    # ========================================================================
    # 方法：__getitem__() - 按索引获取单个样本
    # ========================================================================
    
    def __getitem__(self, i: int):
        """
        PyTorch Dataset 要求实现的方法。
        根据索引返回对应的样本。
        
        参数:
            i: int，样本在数据集中的索引
        
        返回:
            tuple，(特征, 标签)
                特征: numpy 数组，形状 [T, 264]（LMVD 特有：264 = 136 + 128）
                标签: int，0 或 1
        """
        
        # 获取第 i 个样本的特征和标签
        feature = self.features[i]   # 特征：[T, 264]（时间步数可变）
        label = self.labels[i]       # 标签：0（正常）或 1（抑郁）
        
        # ====================================================================
        # 应用特征变换（如果提供了的话）
        # ====================================================================
        
        if self.transform is not None:
            print("Transform 1")  # 调试打印
            feature = self.transform(feature)
        
        # ====================================================================
        # 应用标签变换（如果提供了的话）
        # ====================================================================
        
        if self.target_transform is not None:
            print("Transform 2")  # 调试打印
            label = self.target_transform(label)
        
        # 返回变换后的特征和标签
        return feature, label

    # ========================================================================
    # 方法：__len__() - 返回数据集大小
    # ========================================================================
    
    def __len__(self):
        """
        PyTorch Dataset 要求实现的方法。
        返回数据集中样本的总数。
        """
        return len(self.labels)


# ============================================================================
# 函数：_collate_fn() - 批处理合并函数
# ============================================================================

def _collate_fn(batch):
    """
    自定义的批处理函数，用于处理变长序列。
    
    作用：
    - 将多个样本组成一个 batch
    - 对变长序列进行零填充，使其长度相同
    - 生成 padding_mask，标记哪些位置是真实数据，哪些是填充
    
    参数:
        batch: list，包含多个 (feature, label) 元组
               feature: numpy 数组，形状 [T_i, 264]（T_i 可变）
               label: int，0 或 1
    
    返回:
        tuple，(padded_features, labels, padding_mask)
        padded_features: PyTorch 张量，形状 [B, T_max, 264]
                         所有序列零填充到最长序列的长度
        labels: PyTorch 张量，形状 [B]
                每个样本的标签
        padding_mask: PyTorch 张量，形状 [B, T_max]
                      1 表示真实数据位置，0 表示填充位置
    
    注意：此函数与 DVlog 中的 _collate_fn 完全相同，只是处理特征维度不同
    """
    
    # 将 batch 中的 (feature, label) 对分离成两个序列
    features, labels = zip(*batch)
    
    # ====================================================================
    # 使用 pad_sequence 对变长序列进行零填充
    # ====================================================================
    
    # pad_sequence 的工作方式：
    # - 输入：一个张量列表，每个张量形状为 [T_i, 264]
    # - 输出：一个张量，形状为 [B, T_max, 264]（T_max 是最长序列的长度）
    # - 短序列用 0 填充到 T_max
    padded_features = pad_sequence(
        [torch.from_numpy(f) for f in features],  # 转换 numpy 为 PyTorch 张量
        batch_first=True                           # 返回 [B, T, F] 而不是 [T, B, F]
    )
    
    # ====================================================================
    # 生成 padding_mask
    # ====================================================================
    
    # 计算掩蔽张量：
    # - 对所有特征求和（按最后一维）
    # - 如果某个时间步的所有特征都是 0（即被填充），则求和为 0
    # - 使用 != 0 将其转换为 0/1 掩蔽（1 表示真实，0 表示填充）
    padding_mask = (padded_features.sum(dim=-1) != 0).long()
    
    # ====================================================================
    # 转换标签为张量
    # ====================================================================
    
    labels = torch.tensor(labels)
    
    # ====================================================================
    # 返回处理后的 batch
    # ====================================================================
    
    return padded_features, labels, padding_mask


# ============================================================================
# 函数：get_lmvd_dataloader() - 创建 LMVD 数据加载器
# ============================================================================

def get_depression_dataloader(
    data_set_name: str,
    root: Union[str, Path],                    # 数据集根目录
    fold: str="train",                         # 数据分割：train / valid / test
    batch_size: int=8,                         # 批次大小
    gender: str="both",                        # 性别过滤（LMVD 中未使用，保留用于接口一致性）
    transform=None,                            # 特征变换函数（可选）
    target_transform=None,                     # 标签变换函数（可选）
    aug=True                                   # 是否进行数据增强
):
    """
    为 LMVD 数据集创建一个 PyTorch DataLoader。
    
    参数:
        root: 数据集根目录，应该包含 labels.csv、visual/ 和 audio/ 文件夹
        fold: 选择使用哪个数据分割（train/valid/test）
        batch_size: 每个 batch 中样本的数量
        gender: 性别过滤（LMVD 中此参数未使用）
        transform: 应用于特征的变换函数
        target_transform: 应用于标签的变换函数
        aug: 是否对训练集进行数据增强
    
    返回:
        DataLoader 对象，可以迭代生成 batch
    """
    
    # ========================================================================
    # Step 1：创建 Dataset 对象
    # ========================================================================
    
    # 实例化 LMVD 数据集
    # 这一步会读取 CSV 和所有特征文件，构建 self.features 和 self.labels 列表
    dataset = DepressionDataSet(
        data_set_name,
        root,                      # 数据集路径
        fold,                      # 数据分割
        gender,                    # 性别过滤（LMVD 中未使用）
        transform,                 # 特征变换
        target_transform,          # 标签变换
        aug                        # 数据增强
    )
    
    # ========================================================================
    # Step 2：创建 DataLoader 对象
    # ========================================================================
    
    dataloader = data.DataLoader(
        dataset,                           # 数据集对象
        batch_size=batch_size,             # 批次大小
        collate_fn=_collate_fn,            # 批处理函数
        shuffle=(fold == "train"),         # 只有训练集需要打乱顺序
    )
    
    # 返回 DataLoader 对象
    return dataloader


# ============================================================================
# 主程序：用于测试和调试
# ============================================================================

if __name__ == '__main__':
    # ========================================================================
    # 测试代码：创建数据加载器并打印信息
    # ========================================================================
    
    # 创建训练集数据加载器
    train_loader = get_depression_dataloader(
        "dvlog",
        "/media/zjw/951FB31A9E1EB7E0/dateSet/dvlog", "train"
    )
    print(f"train_loader: {len(train_loader.dataset)} samples")
    
    # 创建验证集数据加载器
    valid_loader = get_depression_dataloader(
        "dvlog",
        "/media/zjw/951FB31A9E1EB7E0/dateSet/dvlog", "valid"
    )
    print(f"valid_loader: {len(valid_loader.dataset)} samples")
    
    # 创建测试集数据加载器
    test_loader = get_depression_dataloader(
        "dvlog",
        "/media/zjw/951FB31A9E1EB7E0/dateSet/dvlog", "test"
    )
    print(f"test_loader: {len(test_loader.dataset)} samples")

    # ========================================================================
    # 获取并打印第一个 batch 的信息
    # ========================================================================
    
    # 从训练集获取第一个 batch
    b1 = next(iter(train_loader))[0]
    print(f"A train_loader batch: shape={b1.shape}, dtype={b1.dtype}")
    # 预期输出：shape=[B, T_max, 264]（与 DVlog 的 161 不同）
    
    # 从验证集获取第一个 batch
    b2 = next(iter(valid_loader))[0]
    print(f"A valid_loader batch: shape={b2.shape}, dtype={b2.dtype}")
    # 预期输出：shape=[B, T_max, 264]
    
    # 从测试集获取第一个 batch
    b3 = next(iter(test_loader))[0]
    print(f"A test_loader batch: shape={b3.shape}, dtype={b3.dtype}")
    # 预期输出：shape=[B, T_max, 264]
