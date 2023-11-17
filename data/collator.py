import torch
from torch.nn.utils.rnn import pad_sequence

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]   # [Tensor[], Tensor[], Tensor[], ..., Tensor[]]
        targets = pad_sequence(targets, False, self.pad_idx)    # padding in which dimension?
        targets = torch.transpose(targets, 0, 1)
        return imgs, targets