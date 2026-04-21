import logging  # 导入日志模块，用于记录信息，例如加载了多少样本
import pickle  # 导入 pickle 模块，用于读取 .pkl 文件，这是一种序列化 Python 对象的文件格式
import numpy as np  # 导入 NumPy，用于高效地处理数值数组
import torch  # 导入 PyTorch 核心库
from torch.utils.data import DataLoader, Dataset  # 从 PyTorch 中导入 Dataset 和 DataLoader 这两个数据处理的核心类

# __all__ 定义了当其他文件使用 `from data_loader import *` 时，可以导入的名称。这里是 'MMDataLoader'。
__all__ = ['MMDataLoader']
# 获取名为 'MMSA' 的日志记录器实例，与 run.py 中使用的是同一个
logger = logging.getLogger('H-DCD')

# ------------------------------------------------------------------------------------
# 1. MMDataset 类：定义单个数据集（如训练集）
# ------------------------------------------------------------------------------------
# 这个类继承自 PyTorch 的 `Dataset`，是自定义数据集的标准写法。
# 它告诉 PyTorch 如何获取单个数据样本。

class MMDataset(Dataset):
    # --- 初始化方法 ---
    def __init__(self, args, mode='train'):
        # self.mode 存储当前数据集的模式，是 'train', 'valid', 还是 'test'
        self.mode = mode
        # self.args 存储从 run.py 传过来的完整配置对象，包含了所有路径、超参数等信息
        self.args = args
        # 创建一个字典，将数据集名称映射到对应的初始化函数。这是一种优雅的替代 if/elif/else 的方式。
        DATASET_MAP = {
            'mosi': self.__init_mosi,
            'mosei': self.__init_mosei,
            'sims': self.__init_sims,
            'sims2': self.__init_sims2,
            
        }
        # 根据配置中的 'dataset_name' (例如 'mosi')，调用对应的初始化函数 (例如 self.__init_mosi())
        DATASET_MAP[args['dataset_name']]()

    def __init_mosi(self):
        with open(self.args['featurePath'], 'rb') as f:
            data = pickle.load(f)
            
        logger.info(f"Dataset keys: {data.keys()}")
        logger.info(f"Mode {self.mode} keys: {data[self.mode].keys()}")
        
        # --- 加载文本特征 ---
        # 检查配置中是否需要使用 BERT 特征
        if 'use_bert' in self.args and self.args['use_bert']:
            # 如果是，则从 data 字典中加载 'text_bert' 特征
            if 'text_bert' in data[self.mode]:
                self.text = data[self.mode]['text_bert'].astype(np.float32)  # BERT feature
                logger.info(f"Loaded text_bert shape: {self.text.shape}")
            else:
                logger.warning("text_bert not found, falling back to text")
                self.text = data[self.mode]['text'].astype(np.float32)
                self.args['use_bert'] = False
        else:
            # 否则，加载默认的 GLoVE 特征
            self.text = data[self.mode]['text'].astype(np.float32)  # GLOVE feature
            logger.info(f"Loaded text (Glove) shape: {self.text.shape}")
            
        # Check for transpose (N, D, L) -> (N, L, D)
        # Assuming L=50 based on filename
        if len(self.text.shape) == 3:
            if self.text.shape[2] == 50 and self.text.shape[1] != 50:
                logger.info(f"Transposing text from {self.text.shape} to (N, L, D)")
                self.text = np.transpose(self.text, (0, 2, 1))
                
        # Update input dim in args
        self.args['text_input_dim'] = self.text.shape[-1]
        logger.info(f"Final text shape: {self.text.shape}, text_input_dim: {self.args['text_input_dim']}")

        # --- 加载视觉和音频特征 ---
        # 从 data 字典中加载视觉特征
        self.vision = data[self.mode]['vision'].astype(np.float32)
        # 从 data 字典中加载音频特征
        self.audio = data[self.mode]['audio'].astype(np.float32)
        
        # Check for transpose for audio and vision
        if len(self.audio.shape) == 3:
            if self.audio.shape[2] == 50 and self.audio.shape[1] != 50:
                logger.info(f"Transposing audio from {self.audio.shape} to (N, L, D)")
                self.audio = np.transpose(self.audio, (0, 2, 1))
        self.args['audio_input_dim'] = self.audio.shape[-1]
        logger.info(f"Final audio shape: {self.audio.shape}, audio_input_dim: {self.args['audio_input_dim']}")
        
        if len(self.vision.shape) == 3:
            if self.vision.shape[2] == 50 and self.vision.shape[1] != 50:
                logger.info(f"Transposing vision from {self.vision.shape} to (N, L, D)")
                self.vision = np.transpose(self.vision, (0, 2, 1))
        self.args['video_input_dim'] = self.vision.shape[-1]
        logger.info(f"Final vision shape: {self.vision.shape}, video_input_dim: {self.args['video_input_dim']}")

        # 加载原始文本句子，可能用于调试或展示
        self.raw_text = data[self.mode]['raw_text']
        # 加载每个样本的唯一ID
        self.ids = data[self.mode]['id']

         # --- 动态替换特征（如果提供了外部特征路径）---
        # 这是一个灵活的设计，允许用户在运行时传入自定义的特征文件来替换默认特征。
        if self.args.get('feature_T', '') != "": # 如果传入了文本特征路径
            with open(self.args['feature_T'], 'rb') as f:
                data_T = pickle.load(f)
            if 'use_bert' in self.args and self.args['use_bert']:
                self.text = data_T[self.mode]['text_bert'].astype(np.float32)
                self.args['feature_dims'][0] = 768 # 更新配置中的特征维度为 BERT 的 768
            else:
                self.text = data_T[self.mode]['text'].astype(np.float32)
                self.args['feature_dims'][0] = self.text.shape[2] # 动态更新特征维度
        
        if self.args.get('feature_A', '') != "": # 如果传入了音频特征路径
            with open(self.args['feature_A'], 'rb') as f:
                data_A = pickle.load(f)
            self.audio = data_A[self.mode]['audio'].astype(np.float32)
            self.args['feature_dims'][1] = self.audio.shape[2] # 动态更新特征维度
        
        if self.args.get('feature_V', '') != "": # 如果传入了视频特征路径
            with open(self.args['feature_V'], 'rb') as f:
                data_V = pickle.load(f)
            self.vision = data_V[self.mode]['vision'].astype(np.float32)
            self.args['feature_dims'][2] = self.vision.shape[2] # 动态更新特征维度

         # --- 加载标签 ---
        # 创建一个名为 'labels' 的字典来存储所有标签
        self.labels = {
            # 'M' 代表多模态任务，加载回归标签（情感分数）
            'M': np.array(data[self.mode]['regression_labels']).astype(np.float32)
        }
        if self.args.dataset_name == 'sims' or self.args.dataset_name == 'sims2':
            for m in "TAV":
                # 对于 SIMS 数据集，还需要加载单模态的标签
                self.labels[m] = np.array(data[self.mode][f'regression_labels_{m}']).astype(np.float32)

        # 使用日志记录当前模式（train/valid/test）加载了多少个样本
        logger.info(f"{self.mode} samples: {self.labels['M'].shape}")

        # --- 处理非对齐数据 ---
        # 如果配置中指定数据是非对齐的 (need_data_aligned: false)
        if not self.args['need_data_aligned']:
            # 加载每个样本的音频和视频序列的真实长度
            if self.args['feature_A'] != "":
                self.audio_lengths = list(data_A[self.mode]['audio_lengths'])
            else:
                self.audio_lengths = data[self.mode]['audio_lengths']
            if self.args.get('feature_V', '') != "":
                self.vision_lengths = list(data_V[self.mode]['vision_lengths'])
            else:
                self.vision_lengths = data[self.mode]['vision_lengths']
        
        # 将音频特征中的 -inf (负无穷大) 值替换为 0，防止计算错误
        self.audio[self.audio == -np.inf] = 0
        if self.args.get('need_truncated', False):
            self.__truncate()

        # --- 特征归一化 ---
        # 如果配置中需要对特征进行归一化
        if 'need_normalized' in self.args and self.args['need_normalized']:
            # 调用私有的归一化方法
            self.__normalize()
    
    # --- MOSEI 数据集初始化函数 ---
    def __init_mosei(self):
        # 直接调用 mosi 的初始化函数，因为它们的数据结构是相同的
        return self.__init_mosi()

    def __init_sims(self):
        # 直接调用 mosi 的初始化函数，因为它们的数据结构是相同的
        return self.__init_mosi()
    def __init_sims2(self):
        return self.__init_mosi()


    # --- 私有方法：截断序列 ---
    def __truncate(self):
        # 这是一个内部辅助函数，用于将序列截断到指定的长度。
        def do_truncate(modal_features, length):
            # 如果当前特征长度与目标长度一致，则无需操作
            if length == modal_features.shape[1]:
                return modal_features
            # 准备一个空列表来存放截断后的特征
            truncated_feature = []
            # 创建一个全零的 padding 向量，用于判断序列的有效部分从哪里开始
            padding = np.array([0 for i in range(modal_features.shape[2])])
            # 遍历批次中的每一个样本
            for instance in modal_features:
                # 遍历一个样本中的每一个时间步
                for index in range(modal_features.shape[1]):
                    # 检查当前时间步是否是 padding
                    if((instance[index] == padding).all()):
                        # 如果是 padding，并且截取窗口不会超出边界
                        if(index + length >= modal_features.shape[1]):
                            # 从当前位置截取指定长度的片段
                            truncated_feature.append(instance[index:index+length]) # 修正：这里应该是 length 而不是 20
                            break # 处理完当前样本，跳出内层循环
                    else: # 如果不是 padding，说明这是有效序列的开始
                        # 从有效序列的开始处截取指定长度的片段
                        truncated_feature.append(instance[index:index+length]) # 修正：这里应该是 length 而不是 20
                        break # 处理完当前样本，跳出内层循环
            # 将截断后的特征列表转换回 NumPy 数组
            truncated_feature = np.array(truncated_feature)
            return truncated_feature
        
        # 从配置中获取文本、音频、视频的目标序列长度
        text_length, audio_length, video_length = self.args['seq_lens']
        # 对每个模态的特征执行截断操作
        self.vision = do_truncate(self.vision, video_length)
        self.text = do_truncate(self.text, text_length)
        self.audio = do_truncate(self.audio, audio_length)

     # --- 私有方法：归一化特征 ---
    def __normalize(self):
        # 将视觉特征的维度从 (N, L, D) 转置为 (L, N, D)，其中 N 是样本数，L 是序列长度，D 是特征维度
        self.vision = np.transpose(self.vision, (1, 0, 2))
        # 对音频特征做同样的操作
        self.audio = np.transpose(self.audio, (1, 0, 2))
        # 沿着样本维度 (axis=0) 计算均值，实现 Z-score 归一化中的减均值步骤。keepdims=True 保持维度。
        self.vision = np.mean(self.vision, axis=0, keepdims=True)
        # 对音频特征做同样的操作
        self.audio = np.mean(self.audio, axis=0, keepdims=True)

        # 将计算中可能产生的 NaN (Not a Number) 值替换为 0
        self.vision[self.vision != self.vision] = 0
        self.audio[self.audio != self.audio] = 0

        # 将维度转置回原来的 (N, L, D) 格式
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))

    # --- __len__ 方法 (必须实现) ---
    # 这个方法返回数据集中样本的总数。DataLoader 会用它来确定迭代的次数。
    def __len__(self):
        return len(self.labels['M'])

    # --- get_seq_len 方法 ---
    # 获取序列长度。注意这里的实现可能有点问题，它返回的是特征维度，而不是序列长度。
    def get_seq_len(self):
        if 'use_bert' in self.args and self.args['use_bert']:
            # 对于 BERT，(N, L, D)，shape[2] 是维度
            return (self.text.shape[2], self.audio.shape[1], self.vision.shape[1])
        else:
            # 对于 GLoVE，(N, L, D)，shape[1] 是序列长度
            return (self.text.shape[1], self.audio.shape[1], self.vision.shape[1])

    # --- get_feature_dim 方法 ---
    # 获取每个模态的特征维度
    def get_feature_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    # --- __getitem__ 方法 (必须实现) ---
    # 这是 Dataset 类的核心。它定义了如何根据索引 `index` 获取一个数据样本。
    def __getitem__(self, index):
        # 创建一个字典 `sample` 来存放单个样本的所有信息
        sample = {
            'raw_text': self.raw_text[index],
            # 将 NumPy 数组转换为 PyTorch 张量 (Tensor)
            'text': torch.Tensor(self.text[index]), 
            'audio': torch.Tensor(self.audio[index]),
            'video': torch.Tensor(self.vision[index]),
            'index': index,
            'id': self.ids[index],
            # 对标签也进行同样的处理，并 reshape 成一维张量
            'label': torch.Tensor(self.labels['M'][index].reshape(-1)),
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()}
        } 
        # 如果数据是非对齐的，额外添加长度信息到样本中
        if not self.args['need_data_aligned']:
            sample['audio_lengths'] = self.audio_lengths[index]
            sample['vision_lengths'] = self.vision_lengths[index]
        # 返回包含一个完整样本的字典
        return sample

# ------------------------------------------------------------------------------------
# 2. MMDataLoader 函数：创建数据加载器
# ------------------------------------------------------------------------------------
# 这是一个工厂函数，它创建并返回 train, valid, test 三个模式的数据加载器。
def MMDataLoader(args, num_workers):

    # 创建一个字典，为 'train', 'valid', 'test' 每种模式都实例化一个 MMDataset
    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test')
    }

    # 如果配置中需要序列长度信息，则从训练集中获取并更新到 args 中
    if 'seq_lens' in args:
        args['seq_lens'] = datasets['train'].get_seq_len() 

    # 使用字典推导式，为每种模式创建一个 DataLoader
    dataLoader = {
        ds: DataLoader(datasets[ds], # 使用上面创建的 dataset
                       batch_size=args['batch_size'], # 从配置中获取 batch size
                       num_workers=num_workers, # 设置用于加载数据的工作线程数
                       shuffle=True) # 在每个 epoch 开始时打乱数据顺序，这对于训练至关重要
        for ds in datasets.keys()
    }
    
    # 返回包含三个 DataLoader 的字典
    return dataLoader
