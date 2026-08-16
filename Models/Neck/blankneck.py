from torch import nn
import torch
import torch.nn.functional as F


class BlankNeck(nn.Module):
    def __init__(self, cfg):
        super(BlankNeck, self).__init__()
        


    def forward(self, inputs):
        if isinstance(inputs, (list, tuple)):
            return list(inputs)
        elif isinstance(inputs, torch.Tensor):
            return [inputs]