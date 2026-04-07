import torch
import torch.nn as nn
from models.backbones import ClassicLSTM, AdvancedCNNLSTM, AblationBackbone

class TransferNet(nn.Module):
    def __init__(self, ablation_mode='complete', input_dim=3, hidden_dim=64):
        super(TransferNet, self).__init__()

        use_cnn = True
        use_lstm = True
        use_attn = True

        if ablation_mode == 'no_cnn':
            use_cnn = False
        elif ablation_mode == 'no_attn':
            use_attn = False
        elif ablation_mode == 'lstm_only':
            use_cnn = False
            use_attn = False
        elif ablation_mode == 'complete':
            pass  

        self.backbone = AblationBackbone(input_dim, hidden_dim, use_cnn, use_lstm, use_attn)

        # base_dim = hidden_dim * 2
        if use_attn:
            feat_dim = hidden_dim * 2
        else:
            feat_dim = hidden_dim


        self.predictor = nn.Sequential(
            nn.Linear(feat_dim, 32),  
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        feat = self.backbone(x)
        out = self.predictor(feat)
        return feat, out
