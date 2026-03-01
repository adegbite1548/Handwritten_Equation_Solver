import json
import re
import pandas as pd

def tokenize_latex(latex_string):

    return str(latex_string).split()

path_to_dataset = "../mathwriting_dataset_images"

train_data_df = pd.read_csv(path_to_dataset + "/train/data/data.csv", header  = 0)
synthetic_data_df = pd.read_csv(path_to_dataset + "/synthetic/data/data.csv", header = 0)
symbols_data_df = pd.read_csv(path_to_dataset + "/symbols/data/data.csv", header = 0)
valid_data_df = pd.read_csv(path_to_dataset + "/valid/data/data.csv", header=0)
test_data_df = pd.read_csv(path_to_dataset + "/test/data/data.csv", header=0)



all_labels = train_data_df['label'].tolist() + \
            synthetic_data_df['label'].tolist() + \
            symbols_data_df['label'].tolist() + \
              valid_data_df['label'].tolist() + \
              test_data_df['label'].tolist()

vocab_set = set()
for latex_string in all_labels:
    tokens = tokenize_latex(latex_string)
    vocab_set.update(tokens)

sorted_tokens = sorted(list(vocab_set))
vocab = {
    "<SOS>": 0, "<EOS>" : 1, "<PAD>": 2, "<UNK>": 3
}

for i, token in enumerate(sorted_tokens):
    vocab[token] = i + 4

with open("vocab.json", "w") as f:
    json.dump(vocab, f, indent=4)

print(f"Vocab built! Total unique tokens: {len(vocab)}")