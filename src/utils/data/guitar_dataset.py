import json
import torch
from torch.utils.data import Dataset
from pathlib import Path


class GuitarDataset(Dataset):
    def __init__(self, block_size=128, stride=None, file_list=None):
        self.block_size = block_size
        self.stride = stride or block_size // 2 # overlap between two input example sequences
        self.samples = []  # this actually stores the dataset's data
        self.total_tokens = 0

        # preparing path files for datasets
        token_files = [Path(t) for t in file_list]

        # reading json from token files
        for tf in token_files:
            with open(tf) as f:
                data = json.load(f)
            ids = data["ids"] if isinstance(data, dict) else data

            # build windows
            L = len(ids)
            self.total_tokens += L

            if L <= block_size + 1: # need to wait for first block_size+1 tokens to create fist sequence
                continue
            for s in range(0, L - (block_size + 1), self.stride):
                x = ids[s:s + block_size]
                y = ids[s + 1:s + block_size + 1]
                self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        x, y = self.samples[item]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)