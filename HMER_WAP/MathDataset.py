import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import re


class MathDataset(Dataset):
    def __init__(self, csv_file, img_dir, vocab_dict, transform=None):
        # Read the CSV into a pandas DataFrame
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.vocab = vocab_dict
        self.transform = transform
        
        # Define the special token IDs 
        self.sos_id = self.vocab.get('<SOS>', 0)
        self.eos_id = self.vocab.get('<EOS>', 1)
        self.unk_id = self.vocab.get('<UNK>', 3) # Fallback for unknown symbols

    def __len__(self):
        # Returns the total number of equations in your CSV
        return len(self.data_frame)

    def __getitem__(self, idx):
        # --- 1. Load the Image ---
        img_name = self.data_frame.iloc[idx, 0] # Assuming Col 0 is 'image_name'
        img_path = os.path.join(self.img_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # --- 2. Load and Tokenize the Label ---
        latex_string = str(self.data_frame.iloc[idx, 1]) # Assuming Col 1 is 'label'
        
        
        # 2. Use RegEx to intelligently slice commands vs. single characters
        raw_tokens = re.findall(r"\\[a-zA-Z]+|\\[^a-zA-Z]|.", latex_string)
        
        # 3. Map each string symbol to its integer ID
        tokens = [t for t in raw_tokens if t.strip()]

        token_ids = [self.vocab.get(token, self.unk_id) for token in tokens]

        # --- 3. Append <SOS> and <EOS> ---
        final_sequence = [self.sos_id] + token_ids + [self.eos_id]
        
        return image, torch.tensor(final_sequence)
    




