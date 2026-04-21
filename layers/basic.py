
import torch.nn as nn
import torch.nn.functional as F
class BiGRU(nn.Module):
    """双向GRU用于文本编码"""
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.output_dim = hidden_dim * 2
    
    def forward(self, x):
        """
        Args:
            x: (B, L, D_in)
        Returns:
            output: (B, L, hidden_dim*2)
        """
        output, _ = self.gru(x)
        return output


class MLP(nn.Module):
    """多层感知机"""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, num_layers=3):
        super().__init__()
        layers = []
        
        # 第一层
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # 中间层
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # 最后一层
        if num_layers > 1:
            layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)