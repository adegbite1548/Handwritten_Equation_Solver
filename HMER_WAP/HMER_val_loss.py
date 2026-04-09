import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import json
import matplotlib.pyplot as plt  # Added for plotting
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pad_sequence

from MathDataset import MathDataset
from Watcher import DenseNetEncoder
from Parser import WAPDecoder

# --- 1. Configuration & Setup ---
VAL_CSV = "../../mathwriting_dataset_images/valid/data/data.csv" 
VAL_IMG_DIR = "../../mathwriting_dataset_images/valid"           

VOCAB_SIZE = 231     
EMBED_DIM = 256       
DECODER_DIM = 512     
ENCODER_DIM = 1024
PAD_IDX = 2 
NUM_EPOCHS = 12  # Setup loop variable

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

# --- 3. Model & Loss Initialization ---
encoder = DenseNetEncoder().to(device)
decoder = WAPDecoder(embed_dim=EMBED_DIM, decoder_dim=DECODER_DIM, vocab_size=VOCAB_SIZE).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# --- 4. Validation Step Function ---
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
            
            decoder_input = predictions.argmax(1).detach()
            
    average_loss = loss / valid_steps if valid_steps > 0 else loss
    return average_loss.item()


# --- 5. Main Evaluation Loop Over Epochs ---
print("\nStarting Multi-Epoch Validation...")
epoch_losses = []      # To store losses for plotting
evaluated_epochs = []  # To keep track of which epochs were successfully loaded

for epoch in range(1, NUM_EPOCHS + 1):
    checkpoint_path = f"checkpoints/hmer_checkpoint_epoch_{epoch}.pth"
    
    if os.path.exists(checkpoint_path):
        print(f"\n--- Loading checkpoint: {checkpoint_path} ---")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        
        # Set models to evaluation mode 
        encoder.eval()
        decoder.eval()
        
        val_loss_total = 0
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_loader):
                images = images.to(device)
                targets = targets.to(device)
                
                batch_loss = val_step(images, targets, encoder, decoder, criterion)
                val_loss_total += batch_loss
                
                if batch_idx % 20 == 0: 
                    print(f"Epoch {epoch} | Batch [{batch_idx}/{len(val_loader)}] | Loss: {batch_loss:.4f}")

        avg_val_loss = val_loss_total / len(val_loader)
        epoch_losses.append(avg_val_loss)
        evaluated_epochs.append(epoch)
        print(f"FINAL VALIDATION LOSS FOR EPOCH {epoch}: {avg_val_loss:.4f}")
        
    else:
        print(f"\n Checkpoint {checkpoint_path} not found. Skipping.")

# --- 6. Plotting the Results ---
if len(evaluated_epochs) > 0:
    print("\nGenerating Validation Loss Plot...")
    
    # Define the teacher forcing ratios for each epoch
    teacher_forcing_ratios = {
        1: 1.0, 2: 0.9, 3: 0.8, 4: 0.7, 5: 0.6, 
        6: 0.5, 7: 0.4, 8: 0.3, 9: 0.2, 10: 0.2, 
        11: 0.2, 12: 0.2
    }

    plt.figure(figsize=(10, 6))
    plt.plot(evaluated_epochs, epoch_losses, marker='o', linestyle='-', color='b', label='Validation Loss')
    
    # Annotate each point with its Teacher Forcing Ratio
    for epoch, loss in zip(evaluated_epochs, epoch_losses):
        ratio = teacher_forcing_ratios.get(epoch, "N/A")
        plt.annotate(
            f'TF: {ratio}',          
            (epoch, loss),           
            textcoords="offset points", 
            xytext=(0, 10),          
            ha='center',             # Horizontally center the text
            fontsize=9,
            color='darkred'          # Added color to make it stand out
        )
    
    # Expand the y-axis limit slightly so the top annotations don't get cut off
    y_min, y_max = min(epoch_losses), max(epoch_losses)
    plt.ylim(y_min - (y_max - y_min) * 0.1, y_max + (y_max - y_min) * 0.15)
    
    plt.title('Validation Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Validation Loss')
    plt.xticks(evaluated_epochs)  
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save the plot to a file
    plot_filename = "validation_loss_plot.png"
    plt.savefig(plot_filename)
    print(f"Plot saved successfully as '{plot_filename}'")
    
    # Display the plot
    plt.show()
else:
    print("No checkpoints were successfully evaluated. Cannot generate plot.")