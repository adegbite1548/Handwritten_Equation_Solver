import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.optim as optim
from Watcher import DenseNetEncoder
from Parser import WAPDecoderCAN
from DatasetCreator import train_loader
from Weakly_Supervised_Counter import WSCM
import random



import torch.nn.functional as F

def train_step(images, targets, encoder, wscm, decoder, optimizer, ce_criterion, count_criterion, teacher_forcing_ratio):
    batch_size = images.size(0)
    seq_len = targets.size(1)
    
    # Special tokens: <SOS>=0, <EOS>=1, <PAD>=2, <UNK>=3
    SPECIAL_TOKENS = [0, 1, 2, 3] 
    PAD_IDX = 2 
    
    optimizer.zero_grad()
    
    # 1. Encode and Count
    encoder_features = encoder(images)           
    initial_counts = wscm(encoder_features)      # [B, Vocab_Size]
    
    # 2. Generate Ground Truth Counts (for the counting loss)
    gt_counts = torch.zeros(batch_size, decoder.vocab_size).to(images.device)
    for i in range(batch_size):
        for token in targets[i]:
            val = token.item()
            if val not in SPECIAL_TOKENS:
                gt_counts[i, val] += 1
                
    # Calculate Counting Loss
    loss_count = count_criterion(initial_counts, gt_counts)
    
    # Prepare features for Attention
    b, c, h, w = encoder_features.size()
    encoder_features_seq = encoder_features.view(b, c, -1).permute(0, 2, 1)
    
    decoder_hidden = torch.zeros(batch_size, decoder.decoder_dim).to(images.device)
    num_pixels = encoder_features_seq.size(1)
    coverage = torch.zeros(batch_size, num_pixels).to(images.device)
    
    loss_seq = 0
    valid_steps = 0 
    
    # This is a dynamic checklistt, the idea is to clone the initial counts so we can modify them step-by-step
    current_counts = initial_counts.clone() 
    
    # 3. Decode Sequence
    for t in range(1, seq_len):
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        
        if t == 1 or use_teacher_forcing:
            decoder_input = targets[:, t-1] 
        else:
            decoder_input = predictions.argmax(1).detach() 
            
        # Pass the current ticking-down counts to the decoder
        predictions, decoder_hidden, alpha, coverage = decoder(
            decoder_input, decoder_hidden, encoder_features_seq, coverage, current_counts
        )
        
        # Determine which token to subtract based on the training mode
        if use_teacher_forcing:
            chosen_tokens = targets[:, t]
        else:
            chosen_tokens = predictions.argmax(1).detach()
            
        #  DYNAMIC SUBTRACTION LOGIC
        # 1. Create a one-hot vector of the chosen token
        one_hot_chosen = F.one_hot(chosen_tokens, num_classes=decoder.vocab_size).float()
        
        # 2. Mask out special tokens (we don't count PAD, SOS, EOS, UNK)
        # We only want to subtract physical math symbols
        mask = (chosen_tokens > 3).float().unsqueeze(1)
        one_hot_chosen = one_hot_chosen * mask
        
        # 3. Subtract from current counts and use ReLU to prevent negative counts
        current_counts = F.relu(current_counts - one_hot_chosen)
        ## j
        
        correct_token = targets[:, t]
        
        if (correct_token != PAD_IDX).any():
            step_loss = ce_criterion(predictions, correct_token)
            loss_seq += step_loss
            valid_steps += 1
            
    avg_seq_loss = loss_seq / valid_steps if valid_steps > 0 else loss_seq
    
    # Joint Loss Calculation
    lambda_weight = 1.0
    total_loss = avg_seq_loss + (lambda_weight * loss_count)
    
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(wscm.parameters()) + list(decoder.parameters()), max_norm=5.0)
    optimizer.step()
    
    return total_loss.item(), avg_seq_loss.item(), loss_count.item()



# 1. Model Initialization
VOCAB_SIZE = 231     
EMBED_DIM = 256       
DECODER_DIM = 512     
ENCODER_DIM = 1024    



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = DenseNetEncoder().to(device)
wscm = WSCM(encoder_dim=ENCODER_DIM, vocab_size=VOCAB_SIZE).to(device)
decoder = WAPDecoderCAN(embed_dim=EMBED_DIM, decoder_dim=DECODER_DIM, vocab_size=VOCAB_SIZE).to(device)


VOCAB_SIZE = 231     
EMBED_DIM = 256       
DECODER_DIM = 512     
ENCODER_DIM = 1024    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = DenseNetEncoder().to(device)
wscm = WSCM(encoder_dim=ENCODER_DIM, vocab_size=VOCAB_SIZE).to(device)
decoder = WAPDecoderCAN(embed_dim=EMBED_DIM, decoder_dim=DECODER_DIM, vocab_size=VOCAB_SIZE).to(device)

# 2. Optimizer and Loss 
ENCODER_LR = 1e-5  # Protect the pre-trained DenseNet
DECODER_LR = 5e-4  # Fast learning for the Decoder
COUNTER_LR = 5e-4

optimizer = torch.optim.Adam([
    {'params': encoder.parameters(), 'lr': ENCODER_LR},
    {'params': wscm.parameters(), 'lr': COUNTER_LR},
    {'params': decoder.parameters(), 'lr': DECODER_LR}
])

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

PAD_IDX = 2 
ce_criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
count_criterion = nn.MSELoss()

# 3. Checkpoint Loading 
checkpoint_path = "checkpoints_CAN/hmer_checkpoint_WAP_baseline.pth"


if os.path.exists(checkpoint_path):
    print(f"Found WAP baseline: {checkpoint_path}. Loading...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 1. Load the fully trained Encoder
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    
    # 2. Extract the old decoder weights
    old_decoder_state = checkpoint['decoder_state_dict']
    
    # 3. This creates a new dictionary without any 'gru' keys
    filtered_decoder_state = {k: v for k, v in old_decoder_state.items() if 'gru' not in k}
    
    # 4. Load the filtered dictionary. 
    decoder.load_state_dict(filtered_decoder_state, strict=False)
    
    print("Baseline weights loaded! WSCM and updated GRU are starting fresh.")
else:
    print("Checkpoint not found! Starting completely from scratch.")

# --- 4. The Epoch Loop ---
EPOCHS = 12 

print(f"Training on {device}...")
print(f"Total batches per epoch: {len(train_loader)}")

for epoch in range(EPOCHS):
    encoder.train()
    wscm.train() 
    decoder.train()
    
    # Track all three losses separately
    epoch_total_loss = 0
    epoch_seq_loss = 0
    epoch_count_loss = 0
    
    current_tf_ratio = max(0.2, 1.0 - (epoch * 0.1)) 
    print(f"\nEpoch {epoch+1} starting with Teacher Forcing Ratio: {current_tf_ratio:.1f}") 
    
    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)
        
        # Run the training step
        total_batch_loss, seq_loss, count_loss = train_step(
            images, targets, encoder, wscm, decoder, optimizer, ce_criterion, count_criterion, current_tf_ratio
        ) 
        
        epoch_total_loss += total_batch_loss
        epoch_seq_loss += seq_loss
        epoch_count_loss += count_loss
        
        
        if batch_idx % 50 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Batch [{batch_idx}/{len(train_loader)}] "
                  f"| Total: {total_batch_loss:.4f} (Seq: {seq_loss:.4f}, Count: {count_loss:.4f})")
            
    # End of Epoch Processing
    avg_total_loss = epoch_total_loss / len(train_loader)
    avg_seq_loss = epoch_seq_loss / len(train_loader)
    avg_count_loss = epoch_count_loss / len(train_loader)
    
    print(f"==== Epoch [{epoch+1}/{EPOCHS}] Completed ====")
    print(f"Avg Total Loss: {avg_total_loss:.4f} | Avg Seq Loss: {avg_seq_loss:.4f} | Avg Count Loss: {avg_count_loss:.4f}") 
    
    save_dir = "checkpoints_CAN"
    os.makedirs(save_dir, exist_ok=True)
    
    file_name = f'hmer_can_checkpoint_epoch_{epoch+1}.pth' 
    save_path = os.path.join(save_dir, file_name)
    
    # Checkpoint Saving
    checkpoint_dict = {
        'epoch': epoch + 1,
        'encoder_state_dict': encoder.state_dict(),
        'wscm_state_dict': wscm.state_dict(), 
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'vocab_size': VOCAB_SIZE,
        'loss_total': avg_total_loss,
        'loss_seq': avg_seq_loss,
        'loss_count': avg_count_loss
    }
    
    torch.save(checkpoint_dict, save_path)
    print(f"Saved CAN checkpoint for epoch {epoch+1} at: {save_path}") 

    scheduler.step()
    current_lr = optimizer.param_groups[2]['lr'] # index 2 is the Decoder
    print(f"Current Decoder Learning Rate: {current_lr:.6f}\n") # just to see if it works

print("\nTraining Complete!")