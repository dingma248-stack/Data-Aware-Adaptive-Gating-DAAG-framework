import torch
import torch.nn as nn
from layers.attention import SelfAttention, CausalConv1d

class ClassicLSTM(nn.Module):

    def __init__(self, input_dim=3, hidden_dim=64, num_layers=2):
        super(ClassicLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        
    def forward(self, x):
        # x: [batch, seq, dim]
        out, _ = self.lstm(x)
        return out[:, -1, :] 

class AdvancedCNNLSTM(nn.Module):

    def __init__(self, input_dim=3, hidden_dim=64, num_layers=2, seq_len=256):
        super(AdvancedCNNLSTM, self).__init__()
        
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
        
        feat, weights = self.attention(lstm_out)
        return feat


class AblationBackbone(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, use_cnn=True, use_lstm=True, use_attn=True):
        super(AblationBackbone, self).__init__()
        self.use_cnn = use_cnn
        self.use_lstm = use_lstm
        self.use_attn = use_attn

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
            lstm_input_dim = 32 
        else:
            lstm_input_dim = input_dim 

        if self.use_lstm:
            self.lstm = nn.LSTM(lstm_input_dim, hidden_dim, num_layers=2, batch_first=True)
            feat_dim = hidden_dim
        else:
            self.fc_replace = nn.Linear(lstm_input_dim, hidden_dim)
            feat_dim = hidden_dim

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
            # feat, _ = self.attention(x)  # [B, Hidden]
            local_feat, _ = self.attention(x)
            feat = torch.cat([local_feat, global_feat], dim=1)
        else:
            # feat = x[:, -1, :]  # [B, Hidden]
            feat = global_feat

        return feat

