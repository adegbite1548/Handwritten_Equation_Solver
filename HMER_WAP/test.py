import torch
import torch.nn as nn
from Watcher import DenseNetEncoder
from Parser import WAPDecoder

def test_wap_architecture():
    print("--- Starting WAP Architecture Sanity Check ---")
    
    # 1. Hyperparameters 
    BATCH_SIZE = 4
    VOCAB_SIZE = 231   # From the vocab.json
    EMBED_DIM = 256    # Size of vector for each token
    DECODER_DIM = 512  # Hidden state size of GRU
    MAX_LEN = 20       # Length of the caption (e.g., 20 tokens long)
    IMG_H = 128        # Height of input image
    IMG_W = 128        # Width of input image

    # 2. Instantiate Models
    encoder = DenseNetEncoder()
    decoder = WAPDecoder(embed_dim=EMBED_DIM, decoder_dim=DECODER_DIM, vocab_size=VOCAB_SIZE)

    # 3. Create Fake Data
    # Random noise image: [Batch, Channels, Height, Width]
    fake_images = torch.randn(BATCH_SIZE, 3, IMG_H, IMG_W)
    
    # Random token IDs: [Batch, Max_Len] (e.g., <SOS>, 45, 99, ...)
    fake_captions = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_LEN))

    print(f"Input Image Shape: {fake_images.shape}")
    print(f"Input Caption Shape: {fake_captions.shape}")

    # 4. Forward Pass (Watcher)
    features = encoder(fake_images)
    print(f"\n[Watcher Output] Raw Feature Map: {features.shape}")
    
    # Flatten features for Attention: [Batch, Channels, H, W] -> [Batch, H*W, Channels]
    batch_size, channels, h, w = features.size()
    features_flat = features.permute(0, 2, 3, 1).view(batch_size, -1, channels)
    print(f"[Watcher Output] Flattened for Attention: {features_flat.shape}")

    # 5. Forward Pass (Parser - One Step)
    # Initialize hidden state (zeros)
    decoder_hidden = torch.zeros(BATCH_SIZE, DECODER_DIM)
    
    # Take the first token from our fake caption (e.g., <SOS>)
    first_token = fake_captions[:, 0]
    
    print("\n--- Testing Parser (Decoder) Single Step ---")
    prediction, new_hidden, alpha = decoder(first_token, decoder_hidden, features_flat)
    
    print(f"Prediction Shape (Should be [Batch, Vocab_Size]): {prediction.shape}")
    print(f"New Hidden State Shape (Should be [Batch, Decoder_Dim]): {new_hidden.shape}")
    print(f"Attention Weights (Alpha) Shape (Should be [Batch, H*W]): {alpha.shape}")

    # 6. Verify Output
    assert prediction.shape == (BATCH_SIZE, VOCAB_SIZE), "Prediction shape mismatch!"
    assert new_hidden.shape == (BATCH_SIZE, DECODER_DIM), "Hidden state shape mismatch!"
    assert alpha.shape == (BATCH_SIZE, h*w), "Attention weights shape mismatch!"
    
    print("\nSUCCESS: All shapes match expected dimensions!")

if __name__ == "__main__":
    test_wap_architecture()