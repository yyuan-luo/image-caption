import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    """
    token lists of the sentences in the batch has different length, 
    pad to make them the same length
    """
    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = []
        seq_lens = []
        for item in batch:
            targets.append(item[1])     # [tokens[], tokens[], tokens[], ..., tokens[]]
            seq_lens.append(len(item[1]))
        targets = pad_sequence(targets, True, self.pad_idx)
        return imgs, targets, seq_lens
