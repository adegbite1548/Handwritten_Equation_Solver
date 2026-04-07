import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torchvision

# Initialize your exact encoder
densenet = torchvision.models.densenet121(weights='DEFAULT').features

# Create a fake "dummy" image with a batch size of 1, 3 channels, 128 height, 728 width
dummy_image = torch.randn(1, 3, 128, 728)

# Pass it through the network
output = densenet(dummy_image)

# Print the final shape!
print(f"Final Feature Map Shape: {output.shape}")