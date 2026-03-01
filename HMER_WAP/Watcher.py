import torch
import torch.nn as nn
import torchvision

class DenseNetEncoder(nn.Module):
    def __init__(self, growth_rate=24, block_config=(6, 12, 24, 16), num_init_features=64):
        super(DenseNetEncoder, self).__init__()
        
        # 1. Load a pre-built DenseNe to save time
        # We use DenseNet121.
        self.densenet = torchvision.models.densenet121(weights='DEFAULT')
        
        # 2. Extract the "Features" part (The CNN layers)
        # The idea is to strip away the classifier part of this FCN.
        self.features = self.densenet.features
        
        # Add a transition layer to squash the channels
        # DenseNet121 outputs 1024 channels. 
        self.out_channels = 1024 

    def forward(self, x):
        # Input: [Batch, 3, H, W] (RGB Images)
        # Output: [Batch, 1024, H', W'] (Feature Maps)
        features = self.features(x)
        
        # We return the raw spatial features so the Attention mechanism 
        # can look at different parts of the grid.
        return features


