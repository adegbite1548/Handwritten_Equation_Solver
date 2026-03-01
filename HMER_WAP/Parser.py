import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        # These layers calculate the "Energy" (importance) of each pixel
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # W_h
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # W_s

        self.coverage_att = nn.Linear(1, attention_dim)

        self.full_att = nn.Linear(attention_dim, 1)               # v
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_features, decoder_hidden, coverage):
        # encoder_features: [Batch, Num_Pixels, Encoder_Dim]
        # decoder_hidden: [Batch, Decoder_Dim]
        
        # 1. Calculate Energy
        att1 = self.encoder_att(encoder_features) 
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1) # Add 'pixel' dimension

        att3 = self.coverage_att(coverage.unsqueeze(2))
        
        # Non-linear combo of Encoder and Decoder
        energy = self.full_att(self.relu(att1 + att2 + att3)).squeeze(2)
        
        alpha = self.softmax(energy) # [Batch, Num_Pixels]
        
        # NEW: Update the running coverage vector for the next time step
        new_coverage = coverage + alpha
        
        # 3. Calculate Context Vector (Weighted sum of image features)
        # We multiply the features by their "importance" (alpha)
        attention_weighted_encoding = (encoder_features * alpha.unsqueeze(2)).sum(dim=1)
        
        return attention_weighted_encoding, alpha, new_coverage

class WAPDecoder(nn.Module):
    def __init__(self, embed_dim, decoder_dim, vocab_size, encoder_dim=1024, dropout=0.5):
        super(WAPDecoder, self).__init__()
        
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        
        # 1. Embedding Layer: Turns Token IDs (4, 15, 99) into Vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 2. Attention Mechanism
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim=256)
        
        # 3. The RNN (We use GRU because it's faster/stable)
        # Input to GRU = Embedding of previous char + Context Vector from Attention
        self.gru = nn.GRUCell(embed_dim + encoder_dim, decoder_dim)
        
        # 4. Deep Output Layer (optional but recommended for better results)
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, decoder_hidden, encoder_features, coverage):
        embedded = self.embedding(x) 
        
        # Apply dropout to the embedding to prevent overfitting to specific tokens
        embedded = self.dropout(embedded) 
        
        context, alpha, new_coverage = self.attention(encoder_features, decoder_hidden, coverage)
        
        gru_input = torch.cat([embedded, context], dim=1)
        new_hidden = self.gru(gru_input, decoder_hidden)
        
        # Remove dropout from here
        predictions = self.fc(new_hidden) 
        
        return predictions, new_hidden, alpha, new_coverage