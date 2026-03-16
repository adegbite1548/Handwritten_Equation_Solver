import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import json
import random
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from Watcher import DenseNetEncoder
from Parser import WAPDecoderCAN
from Weakly_Supervised_Counter import WSCM
from MathDataset import MathDataset
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F 

def plot_attention_maps(image_tensor, predicted_tokens, saved_alphas, feature_h, feature_w,initial_counts=None, idx_to_token=None):
    # 1. Convert the PyTorch image back to a format Matplotlib can display
    # Un-normalize it (approximate for quick viewing)
    image = image_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
    image = np.clip((image * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406], 0, 1)
    
    num_tokens = len(predicted_tokens)
    
    # 2. Set up a grid of subplots (e.g., 2 rows, X columns depending on length)
    cols = 1 # Just 1 image per row so it spans the whole screen width
    rows = num_tokens

    # Make the figure very wide (e.g., 12 inches) and give each row 2 inches of height
    fig = plt.figure(figsize=(12, 2 * rows))

    if initial_counts is not None and idx_to_token is not None:
        inventory = []
        # Loop through the vocabulary to find counts >= 0.5 (rounds to 1 or more)
        for idx, count_tensor in enumerate(initial_counts):
            count = count_tensor.item()
            if idx > 3 and count >= 0.5: # Skip special tokens (0,1,2,3)
                symbol = idx_to_token.get(idx, '<UNK>')
                inventory.append(f"{symbol}: {round(count)}")
        
        inventory_str = "Predicted WSCM Inventory | " + ", ".join(inventory)
        if not inventory:
            inventory_str = "Predicted WSCM Inventory | [None Detected]"
            
        fig.suptitle(inventory_str, fontsize=16, fontweight='bold', color='darkblue')
    
    for i in range(num_tokens):
        ax = fig.add_subplot(rows, cols, i + 1)
        
        # 3. Get the attention weights for this specific character
        # alpha shape is currently [1, H*W]. We reshape it back to [H, W]
        alpha = saved_alphas[i].view(feature_h, feature_w).unsqueeze(0).unsqueeze(0)
        
        # 4. Resize the tiny attention map to match the full original image size
        alpha_resized = F.interpolate(alpha, size=(image.shape[0], image.shape[1]), mode='bilinear', align_corners=False)
        alpha_map = alpha_resized.squeeze().numpy()
        
        # 5. Draw the original image, then overlay the heatmap
        ax.imshow(image)
        ax.imshow(alpha_map, cmap='jet', alpha=0.5) # 'jet' gives that nice blue-to-red heatmap look
        
        ax.set_title(f"Predicting: {predicted_tokens[i]}")
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()

# --- 1. The Translation Function ---
def translate_tokens(token_ids, idx_to_token):
    """Converts a list of integer IDs back into a readable LaTeX string"""
    words = []
    for token in token_ids:
        if token == 1:  # <EOS> token (End of Sequence)
            break
        if token not in [0, 2]:  # Skip <SOS> (0) and <PAD> (2)
            words.append(idx_to_token.get(token, '<UNK>'))
    return " ".join(words)

def run_pulse_check(checkpoint_path, num_samples=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # --- 2. Load the Vocabulary and Reverse It ---
    with open('vocab.json', 'r') as f:
        vocab_dict = json.load(f)
    idx_to_token = {v: k for k, v in vocab_dict.items()}
    
    # --- 3. Initialize Models ---
    VOCAB_SIZE = 231     
    EMBED_DIM = 256       
    DECODER_DIM = 512     
    ENCODER_DIM = 1024
    
    encoder = DenseNetEncoder().to(device)
    wscm = WSCM(encoder_dim=ENCODER_DIM, vocab_size=VOCAB_SIZE).to(device) # NEW
    decoder = WAPDecoderCAN(embed_dim=EMBED_DIM, decoder_dim=DECODER_DIM, vocab_size=VOCAB_SIZE).to(device) # NEW
    
    # Load the weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    wscm.load_state_dict(checkpoint['wscm_state_dict']) # NEW
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    encoder.eval()
    wscm.eval() # NEW
    decoder.eval()

    # --- 4. Build the Pulse Check Loader ---
    print("Loading random validation images...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load validation set
    val_dataset = MathDataset(
        csv_file="../../mathwriting_dataset_images/valid/data/data.csv", 
        img_dir="../../mathwriting_dataset_images/valid", 
        vocab_dict=vocab_dict, 
        transform=transform
    )
    
    random_indices = random.sample(range(len(val_dataset)), num_samples)
    pulse_subset = Subset(val_dataset, random_indices)
    val_loader = DataLoader(pulse_subset, batch_size=1, shuffle=False)
    
    # --- 5. The Inference Loop ---
    print("\nStarting Translation Test...\n" + "="*50)
    
    exact_matches = 0
    
    with torch.no_grad():
        for i, (image, target) in enumerate(val_loader):
            image = image.to(device)
            target_ids = target.squeeze(0).tolist()
            
            # Extract spatial features and predict initial counts
            encoder_features = encoder(image)
            current_counts = wscm(encoder_features) # NEW: Get global count

            # We squeeze it to remove the batch dimension so it's just a 1D list
            initial_counts_for_plot = current_counts.clone().detach().cpu().squeeze(0)
            
            b, c, h, w = encoder_features.size()
            encoder_features = encoder_features.view(b, c, -1).permute(0, 2, 1)
            
            decoder_hidden = torch.zeros(1, decoder.decoder_dim).to(device)
            current_token = torch.tensor([0]).to(device) # <SOS>
            
            # Initialize Coverage for inference
            num_pixels = encoder_features.size(1)
            coverage = torch.zeros(1, num_pixels).to(device)
            
            predicted_ids = []
            saved_alphas = []

            for _ in range(150):
                # NEW: Pass current_counts to the decoder
                predictions, decoder_hidden, alpha, coverage = decoder(
                    current_token, decoder_hidden, encoder_features, coverage, current_counts
                )
                
                saved_alphas.append(alpha.detach().cpu())

                # Greedy decode: take the most likely token
                top_token = predictions.argmax(1).item()
                predicted_ids.append(top_token)
                
                if top_token == 1: # <EOS>
                    break
                    
                # --- NEW: DYNAMIC SUBTRACTION LOGIC ---
                pred_tensor = torch.tensor([top_token]).to(device)
                
                # 1. Create a one-hot vector of the predicted token
                one_hot_chosen = F.one_hot(pred_tensor, num_classes=decoder.vocab_size).float()
                
                # 2. Mask out special tokens (0, 1, 2, 3)
                mask = (pred_tensor > 3).float().unsqueeze(1)
                one_hot_chosen = one_hot_chosen * mask
                
                # 3. Subtract from current counts and apply ReLU
                current_counts = F.relu(current_counts - one_hot_chosen)
                # --------------------------------------
                
                current_token = torch.tensor([top_token]).to(device)
                
            # --- TRANSLATE TO STRINGS ---
            target_str = translate_tokens(target_ids, idx_to_token)
            pred_str = translate_tokens(predicted_ids, idx_to_token)
            
            if target_str == pred_str:
                exact_matches += 1
                
            # Print the first 10 examples so you can physically see the math
            if i < 10:
                print(f"Image {i+1}")
                print(f"Target:     {target_str}")
                print(f"Prediction: {pred_str}")
                print("-" * 50)

                token_list = pred_str.split() 
                plot_attention_maps(image, token_list, saved_alphas, h, w,initial_counts=initial_counts_for_plot, 
                    idx_to_token=idx_to_token)
                
    print(f"\nPulse Check Complete!")
    print(f"Exact Match Rate (Greedy): {(exact_matches/num_samples)*100:.1f}%")

if __name__ == '__main__':
    # Make sure to point this to your new CAN checkpoint!
    ckpt_path = os.path.join("checkpoints", "hmer_can_checkpoint_epoch_12.pth") 
    run_pulse_check(ckpt_path, num_samples=100) # Set to 10 for a quick test