import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn.functional as F
import json
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pad_sequence

from MathDataset import MathDataset
from Watcher import DenseNetEncoder
from Parser import WAPDecoderCAN
from Weakly_Supervised_Counter import WSCM

# --- 1. Configuration & Setup ---
CHECKPOINT_PATH = "checkpoints_CAN/hmer_can_checkpoint_epoch_9.pth" 
TEST_CSV = "../../mathwriting_dataset_images/test/data/data.csv" 
TEST_IMG_DIR = "../../mathwriting_dataset_images/test"           

VOCAB_SIZE = 231     
EMBED_DIM = 256       
DECODER_DIM = 512     
ENCODER_DIM = 1024

SOS_IDX = 0
EOS_IDX = 1
PAD_IDX = 2 
UNK_IDX = 3
SPECIAL_TOKENS = [SOS_IDX, EOS_IDX, PAD_IDX, UNK_IDX]

MAX_DECODE_LEN = 150 # Maximum length to prevent infinite loops during inference
TOLERANCES = [0, 1, 2, 3] # The 'n' values for Levenshtein distance tolerance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Testing on device: {device}")

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

print("Loading test dataset...")
test_dataset = MathDataset(
    csv_file=TEST_CSV, 
    img_dir=TEST_IMG_DIR, 
    vocab_dict=my_vocab_dictionary, 
    transform=transform
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=16, 
    shuffle=False, 
    collate_fn=pad_collate_fn
)
print(f"Total test images: {len(test_dataset)}")


# --- 3. Helper Functions for Evaluation ---
def levenshtein_distance(seq1, seq2):
    """Calculates the minimum edit distance between two lists of tokens."""
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1

    matrix = [[0] * size_y for _ in range(size_x)]
    
    for x in range(size_x):
        matrix[x][0] = x

    for y in range(size_y):
        matrix[0][y] = y

    for x in range(1, size_x):

        for y in range(1, size_y):


            if seq1[x-1] == seq2[y-1]:
                matrix[x][y] = min(matrix[x-1][y] + 1, matrix[x-1][y-1], matrix[x][y-1] + 1)
            else:
                matrix[x][y] = min(matrix[x-1][y] + 1, matrix[x-1][y-1] + 1, matrix[x][y-1] + 1)
                
    return matrix[size_x - 1][size_y - 1]

def clean_sequence(seq):
    """Removes SOS, PAD, and everything after EOS from a sequence."""
    clean_seq = []
    
    for token in seq:
        if token == SOS_IDX or token == PAD_IDX:
            continue
        if token == EOS_IDX:
            break
        clean_seq.append(token)
    return clean_seq


# --- 4. Model Initialization & Checkpoint Loading ---
encoder = DenseNetEncoder().to(device)
wscm = WSCM(encoder_dim=ENCODER_DIM, vocab_size=VOCAB_SIZE).to(device) 
decoder = WAPDecoderCAN(embed_dim=EMBED_DIM, decoder_dim=DECODER_DIM, vocab_size=VOCAB_SIZE).to(device)

if os.path.exists(CHECKPOINT_PATH):
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    wscm.load_state_dict(checkpoint['wscm_state_dict']) 
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
else:
    raise FileNotFoundError(f"Could not find {CHECKPOINT_PATH}.")

encoder.eval()
wscm.eval() 
decoder.eval()


# --- 5. Test Inference Loop ---
print("\nStarting Evaluation on Test Set...")

# Dictionaries to track how many sequences fall within our tolerance ranges
correct_within_n = {n: 0 for n in TOLERANCES}
total_samples = 0
total_levenshtein_distance = 0

with torch.no_grad():
    for batch_idx, (images, targets) in enumerate(test_loader):
        images = images.to(device)
        targets = targets.to(device)
        batch_size = images.size(0)
        
        # 1. Encode & Count
        encoder_features = encoder(images)
        initial_counts = wscm(encoder_features)
        current_counts = initial_counts.clone()
        
        # 2. Prepare Decoder
        b, c, h, w = encoder_features.size()
        encoder_features_seq = encoder_features.view(b, c, -1).permute(0, 2, 1)
        
        decoder_hidden = torch.zeros(batch_size, decoder.decoder_dim).to(device)
        num_pixels = encoder_features_seq.size(1)
        coverage = torch.zeros(batch_size, num_pixels).to(device)
        
        # Start with <SOS> for all items in batch
        decoder_input = torch.full((batch_size,), SOS_IDX, dtype=torch.long, device=device)
        
        # Store predictions
        batch_predictions = [[] for _ in range(batch_size)]
        
        # 3. Decode Step-by-Step
        for t in range(MAX_DECODE_LEN):
            predictions, decoder_hidden, alpha, coverage = decoder(
                decoder_input, decoder_hidden, encoder_features_seq, coverage, current_counts
            )
            
            chosen_tokens = predictions.argmax(1).detach()
            
            # Dynamic Subtraction for CAN
            one_hot_chosen = F.one_hot(chosen_tokens, num_classes=VOCAB_SIZE).float()
            mask = (chosen_tokens > 3).float().unsqueeze(1)
            one_hot_chosen = one_hot_chosen * mask
            current_counts = F.relu(current_counts - one_hot_chosen)
            
            # Save token and set up next input
            for i in range(batch_size):
                batch_predictions[i].append(chosen_tokens[i].item())
                
            decoder_input = chosen_tokens
            
            # If all sequences have generated <EOS>, we can stop the loop early
            if (decoder_input == EOS_IDX).all():
                break

        # 4. Evaluate against Ground Truth
        targets_list = targets.cpu().tolist()
        
        for i in range(batch_size):
            pred_seq = clean_sequence(batch_predictions[i])
            true_seq = clean_sequence(targets_list[i])
            
            distance = levenshtein_distance(pred_seq, true_seq)
            total_levenshtein_distance += distance
            total_samples += 1
            
            # Update tolerance trackers
            for n in TOLERANCES:
                if distance <= n:
                    correct_within_n[n] += 1

        if batch_idx % 10 == 0:
            print(f"Processed Batch [{batch_idx}/{len(test_loader)}]")


# --- 6. Results Summary ---
print("\n" + "="*50)
print(f"TEST SET EVALUATION RESULTS ({total_samples} samples)")
print("="*50)
print(f"Average Levenshtein Distance (Errors per sequence): {total_levenshtein_distance / total_samples:.2f}\n")

for n in TOLERANCES:
    accuracy = (correct_within_n[n] / total_samples) * 100
    if n == 0:
        print(f"Exact Match (ExpRate, n=0): {accuracy:.2f}%")
    else:
        print(f"Accuracy within tolerance n<={n}: {accuracy:.2f}%")
print("="*50 + "\n")