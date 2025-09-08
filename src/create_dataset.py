import json
import torch
from torch.utils.data import Dataset
from pathlib import Path

class ElectricGuitarDataset(Dataset):
    def __init__(self, token_folder, context_window=256):
        self.samples = []
        token_files = list(Path(token_folder).glob("*.json"))

        for token_file in token_files:
            with open(token_file, 'r') as f:
                tokens = json.load(f)
            tokens = tokens["ids"] # this is what contains the tokens

            #