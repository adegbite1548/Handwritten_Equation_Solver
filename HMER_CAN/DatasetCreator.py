import torch
import json
from torch.utils.data import DataLoader, ConcatDataset, Subset
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pad_sequence
from MathDataset import MathDataset

def pad_collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images = torch.stack(images)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=2)
    return images, padded_targets

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

with open('vocab.json', 'r') as f:
    my_vocab_dictionary = json.load(f)

# --- 1. Load All Three Datasets ---

print("Loading datasets...")
real_dataset = MathDataset(
    csv_file="../../mathwriting_dataset_images/train/data/data.csv", 
    img_dir="../../mathwriting_dataset_images/train", 
    vocab_dict=my_vocab_dictionary, 
    transform=transform
)

symbol_dataset = MathDataset(
    csv_file="../../mathwriting_dataset_images/symbols/data/data.csv", 
    img_dir="../../mathwriting_dataset_images/symbols", 
    vocab_dict=my_vocab_dictionary, 
    transform=transform
)

full_synth_dataset = MathDataset(
    csv_file="../../mathwriting_dataset_images/synthetic/data/data.csv", 
    img_dir="../../mathwriting_dataset_images/synthetic", 
    vocab_dict=my_vocab_dictionary, 
    transform=transform
)

# --- 2. Create the Synthetic Subset ---
# Let's dynamically set the subset size to match the size of the real dataset 
# to keep the training perfectly balanced (1:1 ratio).
subset_size = len(real_dataset)

# Safety check: ensure we don't ask for more synthetic data than actually exists
if subset_size > len(full_synth_dataset):
    subset_size = len(full_synth_dataset)

# Use torch.randperm to get random indices. This ensures we get a diverse mix 
# of synthetic equations rather than just the first N rows of the CSV.
random_indices = torch.randperm(len(full_synth_dataset))[:subset_size].tolist()

synth_subset = Subset(full_synth_dataset, random_indices)
print(f"Sampled {len(synth_subset)} images from the Synthetic dataset.")

# --- 3. Concatenate Everything ---
combined_dataset = ConcatDataset([real_dataset, symbol_dataset, synth_subset])
print(f"Total training images: {len(combined_dataset)}")

# --- 4. Pass to the DataLoader ---
train_loader = DataLoader(
    combined_dataset, 
    batch_size=16, 
    shuffle=True, # This shuffles the 3 datasets together
    collate_fn=pad_collate_fn
    
)