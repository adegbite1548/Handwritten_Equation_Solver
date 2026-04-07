import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import torch.nn.functional as F  
import json
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pad_sequence

from MathDataset import MathDataset
from Watcher import DenseNetEncoder
from Parser import WAPDecoderCAN
from Weakly_Supervised_Counter import WSCM

# --- 1. Configuration & Setup ---
VAL_CSV = "../../mathwriting_dataset_images/valid/data/data.csv" 
VAL_IMG_DIR = "../../mathwriting_dataset_images/valid"           

VOCAB_SIZE = 231     
EMBED_DIM = 256       
DECODER_DIM = 512     
ENCODER_DIM = 1024
PAD_IDX = 2 
SPECIAL_TOKENS = [0, 1, 2, 3] # SOS, EOS, PAD, UNK
NUM_EPOCHS = 12 # Loop bounds

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

# --- 3. Model & Loss Initialization (Done ONCE) ---
encoder = DenseNetEncoder().to(device)
wscm = WSCM(encoder_dim=ENCODER_DIM, vocab_size=VOCAB_SIZE).to(device) 
decoder = WAPDecoderCAN(embed_dim=EMBED_DIM, decoder_dim=DECODER_DIM, vocab_size=VOCAB_SIZE).to(device)

ce_criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
count_criterion = nn.MSELoss() 

# --- 4. Validation Step Function ---
def val_step(images, targets, encoder, wscm, decoder, ce_criterion, count_criterion):
    batch_size = images.size(0)
    seq_len = targets.size(1)

    with torch.no_grad(): 
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
        
        # Clone counts for dynamic subtraction
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
            
            # Autoregressive decoding i.e strictly use own predictions
            chosen_tokens = predictions.argmax(1).detach()
            
            # DYNAMIC SUBTRACTION
            one_hot_chosen = F.one_hot(chosen_tokens, num_classes=VOCAB_SIZE).float()
            mask = (chosen_tokens > 3).float().unsqueeze(1)
            one_hot_chosen = one_hot_chosen * mask
            current_counts = F.relu(current_counts - one_hot_chosen)
            
            decoder_input = chosen_tokens
            
    avg_seq_loss = loss_seq / valid_steps if valid_steps > 0 else loss_seq
    total_loss = avg_seq_loss + loss_count
    
    return total_loss.item(), avg_seq_loss.item(), loss_count.item()


# --- 5. Main Evaluation Loop Over Epochs ---
print("\nStarting Multi-Epoch Validation...")

evaluated_epochs = []
epoch_total_losses = []
epoch_seq_losses = []
epoch_count_losses = []

for epoch in range(1, NUM_EPOCHS + 1):
    checkpoint_path = f"checkpoints_CAN/hmer_can_checkpoint_epoch_{epoch}.pth"
    
    if os.path.exists(checkpoint_path):
        print(f"\n--- Loading checkpoint: {checkpoint_path} ---")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        wscm.load_state_dict(checkpoint['wscm_state_dict']) 
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        
        # Set models to evaluation mode
        encoder.eval()
        wscm.eval() 
        decoder.eval()
        
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
                
                if batch_idx % 20 == 0:
                    print(f"Epoch {epoch} | Batch [{batch_idx}/{len(val_loader)}] | Total: {batch_total:.4f}")

        avg_val_total = val_total_loss_sum / len(val_loader)
        avg_val_seq = val_seq_loss_sum / len(val_loader)
        avg_val_count = val_count_loss_sum / len(val_loader)
        
        evaluated_epochs.append(epoch)
        epoch_total_losses.append(avg_val_total)
        epoch_seq_losses.append(avg_val_seq)
        epoch_count_losses.append(avg_val_count)
        
        print(f"FINAL METRICS EPOCH {epoch} | Total: {avg_val_total:.4f} | Seq: {avg_val_seq:.4f} | Count: {avg_val_count:.4f}")
        
    else:
         print(f"\nWarning: Checkpoint {checkpoint_path} not found. Skipping.")


# --- 6. Plotting the Results ---
if len(evaluated_epochs) > 0:
    print("\nGenerating Validation Loss Plot...")
    
    teacher_forcing_ratios = {
        1: 1.0, 2: 0.9, 3: 0.8, 4: 0.7, 5: 0.6, 
        6: 0.5, 7: 0.4, 8: 0.3, 9: 0.2, 10: 0.2, 
        11: 0.2, 12: 0.2
    }

    plt.figure(figsize=(12, 7))
    
    # Plot all three metrics
    plt.plot(evaluated_epochs, epoch_total_losses, marker='o', linestyle='-', color='b', label='Total Loss')
    plt.plot(evaluated_epochs, epoch_seq_losses, marker='s', linestyle='--', color='g', label='Sequence Loss')
    plt.plot(evaluated_epochs, epoch_count_losses, marker='^', linestyle='-.', color='r', label='Count Loss')
    
    # Annotate the Total Loss line with the Teacher Forcing Ratios
    for epoch, loss in zip(evaluated_epochs, epoch_total_losses):
        ratio = teacher_forcing_ratios.get(epoch, "N/A")
        plt.annotate(
            f'TF: {ratio}', 
            (epoch, loss), 
            textcoords="offset points", 
            xytext=(0, 15), 
            ha='center', 
            fontsize=9,
            color='darkblue',
            fontweight='bold'
        )
    
    # Adjust y-axis to accommodate annotations at the top
    y_min = min(min(epoch_total_losses), min(epoch_seq_losses), min(epoch_count_losses))
    y_max = max(max(epoch_total_losses), max(epoch_seq_losses), max(epoch_count_losses))
    plt.ylim(y_min - (y_max - y_min) * 0.1, y_max + (y_max - y_min) * 0.15)
    
    plt.title('CAN Validation Losses vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.xticks(evaluated_epochs)  
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save and show
    plot_filename = "can_validation_loss_plot.png"
    plt.savefig(plot_filename)
    print(f"Plot saved successfully as '{plot_filename}'")
    plt.show()
else:
    print("No checkpoints were successfully evaluated. Cannot generate plot.")