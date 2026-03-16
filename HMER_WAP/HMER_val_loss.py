import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import json
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pad_sequence

# Import your custom modules
from MathDataset import MathDataset
from Watcher import DenseNetEncoder
from Parser import WAPDecoder

# --- 1. Configuration & Setup ---
CHECKPOINT_PATH = "checkpoints/hmer_checkpoint_epoch_12.pth"
VAL_CSV = "../../mathwriting_dataset_images/valid/data/data.csv" # Update to your val CSV
VAL_IMG_DIR = "../../mathwriting_dataset_images/valid"           # Update to your val images

VOCAB_SIZE = 231     
EMBED_DIM = 256       
DECODER_DIM = 512     
ENCODER_DIM = 1024
PAD_IDX = 2 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Evaluating on device: {device}")

# --- 2. Data Loading ---
def pad_collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images = torch.stack(images)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=PAD_IDX)
    return images, padded_targets

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

with open('vocab.json', 'r') as f:
    my_vocab_dictionary = json.load(f)

print("Loading validation dataset...")
val_dataset = MathDataset(
    csv_file=VAL_CSV, 
    img_dir=VAL_IMG_DIR, 
    vocab_dict=my_vocab_dictionary, 
    transform=transform
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=16, 
    shuffle=False, # No need to shuffle for validation
    collate_fn=pad_collate_fn
)
print(f"Total validation images: {len(val_dataset)}")

# --- 3. Model Initialization ---
encoder = DenseNetEncoder().to(device)
decoder = WAPDecoder(embed_dim=EMBED_DIM, decoder_dim=DECODER_DIM, vocab_size=VOCAB_SIZE).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# --- 4. Load Checkpoint ---
if os.path.exists(CHECKPOINT_PATH):
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    print("Checkpoint loaded successfully.")
else:
    raise FileNotFoundError(f"Could not find {CHECKPOINT_PATH}. Check the path.")

# Set models to evaluation mode (critical to disable Dropout/BatchNorm during inference)
encoder.eval()
decoder.eval()

# --- 5. Validation Step Function ---
def val_step(images, targets, encoder, decoder, criterion):
    batch_size = images.size(0)
    seq_len = targets.size(1)
    
    loss = 0
    valid_steps = 0 

    with torch.no_grad(): # Disable gradients completely
        encoder_features = encoder(images)
        b, c, h, w = encoder_features.size()
        encoder_features = encoder_features.view(b, c, -1).permute(0, 2, 1)
        
        decoder_hidden = torch.zeros(batch_size, decoder.decoder_dim).to(device)
        
        num_pixels = encoder_features.size(1)
        coverage = torch.zeros(batch_size, num_pixels).to(device)
        
        # Start with the <SOS> token
        decoder_input = targets[:, 0] 
        
        for t in range(1, seq_len):
            predictions, decoder_hidden, alpha, coverage = decoder(
                decoder_input, decoder_hidden, encoder_features, coverage
            )
            
            correct_token = targets[:, t]
            
            if (correct_token != PAD_IDX).any():
                step_loss = criterion(predictions, correct_token)
                loss += step_loss
                valid_steps += 1
            
            # Autoregressive decoding: strictly use own predictions
            decoder_input = predictions.argmax(1).detach()
            
    average_loss = loss / valid_steps if valid_steps > 0 else loss
    return average_loss.item()

# --- 6. Main Evaluation Loop ---
print("\nStarting Validation...")
val_loss_total = 0

with torch.no_grad():
    for batch_idx, (images, targets) in enumerate(val_loader):
        images = images.to(device)
        targets = targets.to(device)
        
        batch_loss = val_step(images, targets, encoder, decoder, criterion)
        val_loss_total += batch_loss
        
        if batch_idx % 10 == 0:
            print(f"Validating... Batch [{batch_idx}/{len(val_loader)}] | Loss: {batch_loss:.4f}")

avg_val_loss = val_loss_total / len(val_loader)
print("\n" + "="*50)
print(f"FINAL VALIDATION LOSS FOR {CHECKPOINT_PATH}: {avg_val_loss:.4f}")
print("="*50 + "\n")