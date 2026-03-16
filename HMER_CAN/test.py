import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
from Watcher import DenseNetEncoder
from Parser import WAPDecoder



VOCAB_SIZE = 231     
EMBED_DIM = 256       
DECODER_DIM = 512     
ENCODER_DIM = 1024    

encoder = DenseNetEncoder()
decoder = WAPDecoder(embed_dim=EMBED_DIM, decoder_dim=DECODER_DIM, vocab_size=VOCAB_SIZE)

checkpoint_path = "checkpoints/hmer_checkpoint_epoch_1.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = encoder.to(device)
decoder = decoder.to(device)

if os.path.exists(checkpoint_path):
    print(f"Found checkpoint: {checkpoint_path}. Loading...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load the adult weights
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
        
    print("Successfully loaded Epoch 10. Resuming training...")
else:
    print("Checkpoint not found! Double check the file path.")


print(checkpoint['loss'])