import json, math, random
from torch.utils.data import Dataset
import torch
from pathlib import Path


class RandomGuitarSeqDataset(Dataset):
    def __init__(self, file_list, block_size=128, epoch_len=None, min_len=None):
        self.block_size = block_size
        self.seqs = []
        self.lengths = []
        keep_threshold = (min_len or (block_size + 1))

        # load all the files
        for fp in map(Path, file_list):
            with open(fp) as f:
                obj = json.load(f)
            ids = obj["ids"] if isinstance(obj, dict) else obj
            if len(ids) > keep_threshold:
                self.seqs.append(ids)
                self.lengths.append(len(ids)- (block_size + 1))

        assert len(self.seqs) > 0, "No sequences long enough for block_size"

        # cumulative distribution for length-proportional sampling
        total_usable = sum(self.lengths)
        self.cum_probs = []
        acc = 0.0
        for L in self.lengths:
            acc += L / total_usable
            self.cum_probs.append(acc)

        # define how many samples per epoch
        self._epoch_len = epoch_len if epoch_len is not None else total_usable

        # handy diagnostics
        self.total_tokens = sum(len(s) for s in self.seqs)
        self.num_files = len(self.seqs)

    def __len__(self):
        return self._epoch_len

    def _pick_file_index(self, u):
        # binary search in cum_probs
        lo, hi = 0, len(self.cum_probs) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if u <= self.cum_probs[mid]:
                hi = mid
            else:
                lo = mid + 1
        return lo

    def __getitem__(self, _):
        # choose a file proportional to its usable length
        u = random.random()
        fi = self._pick_file_index(u)
        ids = self.seqs[fi]

        # pick a random start in that file
        max_start = len(ids) - (self.block_size + 1)
        s = random.randint(0, max_start)

        x = ids[s : s + self.block_size]
        y = ids[s + 1 : s + self.block_size + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)