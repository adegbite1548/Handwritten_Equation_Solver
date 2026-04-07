
import torch.nn as nn

class WSCM(nn.Module):
    def __init__(self, encoder_dim, vocab_size):
        super(WSCM, self).__init__()
        # Global Average Pooling reduces [Batch, Channels, H, W] to [Batch, Channels, 1, 1]
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Linear layer maps the channels to the vocabulary size to predict counts
        self.fc = nn.Linear(encoder_dim, vocab_size)

    def forward(self, x):
        # x is the output from DenseNetEncoder: [B, 1024, H, W]
        pooled = self.avg_pool(x)        # [B, 1024, 1, 1]
        pooled = pooled.view(pooled.size(0), -1) # [B, 1024]
        counts = self.fc(pooled)         # [B, vocab_size]
        return counts


