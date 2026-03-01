import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.optim as optim
from Watcher import DenseNetEncoder
from Parser import WAPDecoder
from DatasetCreator import train_loader
import random



def train_step(images, targets, encoder, decoder, optimizer, criterion, teacher_forcing_ratio):
    batch_size = images.size(0)
    seq_len = targets.size(1)
    PAD_IDX = 2 
    
    optimizer.zero_grad()
    loss = 0
    
    encoder_features = encoder(images)
    b, c, h, w = encoder_features.size()
    encoder_features = encoder_features.view(b, c, -1).permute(0, 2, 1)
    
    decoder_hidden = torch.zeros(batch_size, decoder.decoder_dim).to(images.device)
    
    # Initialize Coverage (Memory) to zeros
    num_pixels = encoder_features.size(1)
    coverage = torch.zeros(batch_size, num_pixels).to(images.device)
    
    # Track how many valid (non-padded) steps we actually take
    valid_steps = 0 
    
    for t in range(1, seq_len):
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        
        if t == 1 or use_teacher_forcing:
            decoder_input = targets[:, t-1] 
        else:
            decoder_input = predictions.argmax(1).detach() 
            
        # Pass in coverage, get updated coverage out
        predictions, decoder_hidden, alpha, coverage = decoder(
            decoder_input, decoder_hidden, encoder_features, coverage
        )
        
        correct_token = targets[:, t]
        
        # Only calculate loss if there are actual tokens to predict in this step
        if (correct_token != PAD_IDX).any():
            step_loss = criterion(predictions, correct_token)
            loss += step_loss
            valid_steps += 1
        
    # Divide by the actual number of valid steps, not the raw sequence length!
    # This prevents the gradients from vanishing on shorter equations.
    average_loss = loss / valid_steps if valid_steps > 0 else loss
    
    average_loss.backward()
    torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), max_norm=5.0)
    optimizer.step()
    
    return average_loss.item()



# --- 1. Model Initialization ---
VOCAB_SIZE = 231     
EMBED_DIM = 256       
DECODER_DIM = 512     
ENCODER_DIM = 1024    

encoder = DenseNetEncoder()
decoder = WAPDecoder(embed_dim=EMBED_DIM, decoder_dim=DECODER_DIM, vocab_size=VOCAB_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = encoder.to(device)
decoder = decoder.to(device)

# --- 2. Optimizer and Loss ---
ENCODER_LR = 1e-5  # Protect the pre-trained DenseNet
DECODER_LR = 5e-4  # Fast learning for the Decoder

# We deleted the scheduler completely.
optimizer = torch.optim.Adam([
    {'params': encoder.parameters(), 'lr': ENCODER_LR},
    {'params': decoder.parameters(), 'lr': DECODER_LR}
])


PAD_IDX = 2 
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)



# --- 3. The Epoch Loop ---
EPOCHS = 10

print(f"Training on {device}...")
print(f"Total batches per epoch: {len(train_loader)}")

for epoch in range(EPOCHS):
    encoder.train()
    decoder.train()
    
    epoch_loss = 0
    
    # Decay teacher forcing. 
    # Starts at 1.0, drops by 0.1 each epoch, stops dropping at 0.2
    current_tf_ratio = max(0.2, 1.0 - (epoch * 0.1)) 
    print(f"Epoch {epoch+1} starting with Teacher Forcing Ratio: {current_tf_ratio:.1f}")
    
    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)
        
        # Run the training step
        batch_loss = train_step(images, targets, encoder, decoder, optimizer, criterion, current_tf_ratio)
        
        epoch_loss += batch_loss
        
        # Print progress every 50 batches so we know it hasn't frozen
        if batch_idx % 50 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Batch [{batch_idx}/{len(train_loader)}] | Batch Loss: {batch_loss:.4f}")
            
    # --- End of Epoch Processing ---
    # Calculate the average loss across all batches
    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"==== Epoch [{epoch+1}/{EPOCHS}] Completed | Average Epoch Loss: {avg_epoch_loss:.4f} ====")

    
    save_dir = "checkpoints"
    
    
    # exist_ok=True means it won't crash if the folder already exists from a previous run
    os.makedirs(save_dir, exist_ok=True)
    
    # Create the full file path (e.g., "checkpoints/hmer_checkpoint_epoch_1.pth")
    file_name = f'hmer_checkpoint_epoch_{epoch+1}.pth'
    checkpoint_path = os.path.join(save_dir, file_name)
    
    # --- Checkpoint Saving ---
    checkpoint = {
        'epoch': epoch + 1,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'vocab_size': VOCAB_SIZE,
        'loss': avg_epoch_loss
    }
    
    # 4. Save to the new path
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint for epoch {epoch+1} at: {checkpoint_path}\n")

print("Training Complete!")