import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import torch.nn.functional as F  # Needed for one_hot and relu
import json
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pad_sequence

# Import your custom modules
from MathDataset import MathDataset
from Watcher import DenseNetEncoder
# UPDATED: Import the CAN Decoder and WSCM
from Parser import WAPDecoderCAN
from Weakly_Supervised_Counter import WSCM

# --- 1. Configuration & Setup ---
CHECKPOINT_PATH = "checkpoints/hmer_can_checkpoint_epoch_9.pth" 
VAL_CSV = "../../mathwriting_dataset_images/valid/data/data.csv" 
VAL_IMG_DIR = "../../mathwriting_dataset_images/valid"           

VOCAB_SIZE = 231     
EMBED_DIM = 256       
DECODER_DIM = 512     
ENCODER_DIM = 1024
PAD_IDX = 2 
SPECIAL_TOKENS = [0, 1, 2, 3] # SOS, EOS, PAD, UNK

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
    shuffle=False, 
    collate_fn=pad_collate_fn
)
print(f"Total validation images: {len(val_dataset)}")

# --- 3. Model Initialization ---
encoder = DenseNetEncoder().to(device)
wscm = WSCM(encoder_dim=ENCODER_DIM, vocab_size=VOCAB_SIZE).to(device) 
decoder = WAPDecoderCAN(embed_dim=EMBED_DIM, decoder_dim=DECODER_DIM, vocab_size=VOCAB_SIZE).to(device)

ce_criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
count_criterion = nn.MSELoss() # Loss for the counting module

# --- 4. Load Checkpoint ---
if os.path.exists(CHECKPOINT_PATH):
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    wscm.load_state_dict(checkpoint['wscm_state_dict']) # NEW
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    print("Checkpoint loaded successfully.")
else:
    raise FileNotFoundError(f"Could not find {CHECKPOINT_PATH}. Check the path.")

# Set models to evaluation mode
encoder.eval()
wscm.eval() 
decoder.eval()

# --- 5. Validation Step Function ---
def val_step(images, targets, encoder, wscm, decoder, ce_criterion, count_criterion):
    batch_size = images.size(0)
    seq_len = targets.size(1)

    with torch.no_grad(): # Disable gradients completely
        # 1. Encode and Count
        encoder_features = encoder(images)
        initial_counts = wscm(encoder_features)
        
        # 2. Generate Ground Truth Counts
        gt_counts = torch.zeros(batch_size, VOCAB_SIZE).to(device)
        for i in range(batch_size):
            for token in targets[i]:
                val = token.item()
                if val not in SPECIAL_TOKENS:
                    gt_counts[i, val] += 1
                    
        # Calculate Count Loss
        loss_count = count_criterion(initial_counts, gt_counts)
        
        # 3. Prepare for Decoding
        b, c, h, w = encoder_features.size()
        encoder_features_seq = encoder_features.view(b, c, -1).permute(0, 2, 1)
        
        decoder_hidden = torch.zeros(batch_size, decoder.decoder_dim).to(device)
        num_pixels = encoder_features_seq.size(1)
        coverage = torch.zeros(batch_size, num_pixels).to(device)
        
        # Start with the <SOS> token
        decoder_input = targets[:, 0] 
        
        loss_seq = 0
        valid_steps = 0 
        
        #Clone counts for dynamic subtraction
        current_counts = initial_counts.clone()

        # 4. Decode Loop
        for t in range(1, seq_len):
            predictions, decoder_hidden, alpha, coverage = decoder(
                decoder_input, decoder_hidden, encoder_features_seq, coverage, current_counts
            )
            
            correct_token = targets[:, t]
            
            if (correct_token != PAD_IDX).any():
                step_loss = ce_criterion(predictions, correct_token)
                loss_seq += step_loss
                valid_steps += 1
            
            # Autoregressive decoding: strictly use own predictions
            chosen_tokens = predictions.argmax(1).detach()
            
            # DYNAMIC SUBTRACTION
            one_hot_chosen = F.one_hot(chosen_tokens, num_classes=VOCAB_SIZE).float()
            mask = (chosen_tokens > 3).float().unsqueeze(1)
            one_hot_chosen = one_hot_chosen * mask
            current_counts = F.relu(current_counts - one_hot_chosen)
            
            
            decoder_input = chosen_tokens
            
    avg_seq_loss = loss_seq / valid_steps if valid_steps > 0 else loss_seq
    total_loss = avg_seq_loss + loss_count
    
    # Return all three to track them
    return total_loss.item(), avg_seq_loss.item(), loss_count.item()

# --- 6. Main Evaluation Loop ---
print("\nStarting Validation...")
val_total_loss_sum = 0
val_seq_loss_sum = 0
val_count_loss_sum = 0

with torch.no_grad():
    for batch_idx, (images, targets) in enumerate(val_loader):
        images = images.to(device)
        targets = targets.to(device)
        
        batch_total, batch_seq, batch_count = val_step(
            images, targets, encoder, wscm, decoder, ce_criterion, count_criterion
        )
        
        val_total_loss_sum += batch_total
        val_seq_loss_sum += batch_seq
        val_count_loss_sum += batch_count
        
        if batch_idx % 10 == 0:
            print(f"Validating Batch [{batch_idx}/{len(val_loader)}] | Total: {batch_total:.4f} (Seq: {batch_seq:.4f}, Count: {batch_count:.4f})")

avg_val_total = val_total_loss_sum / len(val_loader)
avg_val_seq = val_seq_loss_sum / len(val_loader)
avg_val_count = val_count_loss_sum / len(val_loader)

print("\n" + "="*50)
print(f"FINAL VALIDATION METRICS FOR: {CHECKPOINT_PATH}")
print(f"Total Loss: {avg_val_total:.4f}")
print(f"Sequence Loss: {avg_val_seq:.4f}")
print(f"Count Loss:   {avg_val_count:.4f}")
print("="*50 + "\n")