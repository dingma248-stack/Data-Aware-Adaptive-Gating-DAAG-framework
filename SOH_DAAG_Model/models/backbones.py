import torch
import torch.nn as nn
from layers.attention import SelfAttention, CausalConv1d

class ClassicLSTM(nn.Module):
    """
    许多基线论文中使用的标准 LSTM。
    """
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=2):
        super(ClassicLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        
    def forward(self, x):
        # x: [batch, seq, dim]
        out, _ = self.lstm(x)
        return out[:, -1, :] # 最后一个时间步

class AdvancedCNNLSTM(nn.Module):
    """
    1D-CNN + LSTM + 注意力机制
    1. CNN 提取局部形状特征（充电曲线斜率等）
    2. LSTM 捕捉长期依赖关系
    3. 注意力机制加权最重要的时间步
    """
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=2, seq_len=256):
        super(AdvancedCNNLSTM, self).__init__()
        
        # 多尺度 CNN
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2) # 256 -> 128
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2) # 128 -> 64
        )
        
        self.lstm = nn.LSTM(32, hidden_dim, num_layers, batch_first=True, bidirectional=False)
        self.attention = SelfAttention(hidden_dim)
        
    def forward(self, x):
        # x: [batch, seq, dim] -> [batch, dim, seq]
        x = x.permute(0, 2, 1)
        
        x = self.conv1(x)
        x = self.conv2(x)
        
        # [batch, dim, seq] -> [batch, seq, dim]
        x = x.permute(0, 2, 1)
        
        lstm_out, _ = self.lstm(x) # [batch, seq, hidden]
        
        # 注意力聚合
        feat, weights = self.attention(lstm_out)
        return feat


class AblationBackbone(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, use_cnn=True, use_lstm=True, use_attn=True):
        super(AblationBackbone, self).__init__()
        self.use_cnn = use_cnn
        self.use_lstm = use_lstm
        self.use_attn = use_attn

        # 1. CNN 模块
        if self.use_cnn:
            self.cnn = nn.Sequential(
                nn.Conv1d(input_dim, 16, kernel_size=3, padding=1),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2)
            )
            lstm_input_dim = 32  # CNN 输出通道数
        else:
            lstm_input_dim = input_dim  # 原始输入维度 (3)

        # 2. LSTM 模块
        if self.use_lstm:
            self.lstm = nn.LSTM(lstm_input_dim, hidden_dim, num_layers=2, batch_first=True)
            feat_dim = hidden_dim
        else:
            # 如果不用 LSTM，直接用全连接层把维度对齐，方便后续处理
            # 这是一个简单的替代，实际中很少只用 CNN+Attn 处理时序，但为了代码跑通可以这样
            self.fc_replace = nn.Linear(lstm_input_dim, hidden_dim)
            feat_dim = hidden_dim

        # 3. Attention 模块
        if self.use_attn:
            self.attention = SelfAttention(hidden_dim)

    def forward(self, x):
        # x: [batch, seq, dim]

        # --- Step 1: CNN ---
        if self.use_cnn:
            x = x.permute(0, 2, 1)  # [B, D, S]
            x = self.cnn(x)
            x = x.permute(0, 2, 1)  # [B, S, D]

        # --- Step 2: LSTM ---
        if self.use_lstm:
            x, _ = self.lstm(x)  # x: [B, S, Hidden]
        else:
            x = self.fc_replace(x)  # [B, S, Hidden]

        # --- Step 3: Attention vs Last Step ---

        global_feat = x[:, -1, :]
        if self.use_attn:
            # 注意力聚合所有时间步
            # feat, _ = self.attention(x)  # [B, Hidden]
            local_feat, _ = self.attention(x)
            feat = torch.cat([local_feat, global_feat], dim=1)
        else:
            # 如果没有 Attention，通常取最后一个时间步
            # feat = x[:, -1, :]  # [B, Hidden]
            feat = global_feat

        return feat

