import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        # 编码器输出：[批量大小, 序列长度, 隐藏维度]
        energy = self.projection(encoder_outputs) # [batch, seq, 1]
        weights = F.softmax(energy.squeeze(-1), dim=1) # [batch, seq]
        # 加权求和
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1) # [batch, hidden]
        return outputs, weights

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)

    def forward(self, x):
        x = self.conv(x)
        return x[:, :, :-self.padding] if self.padding > 0 else x
