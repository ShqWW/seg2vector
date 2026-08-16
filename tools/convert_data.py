import numpy as np
import torch

def result_to_np(result_dict):
    for key in result_dict.keys():
        item = result_dict[key]
        if type(item) == list:
            for i in range(len(item)):
                if type(item[i]) == torch.Tensor:
                    item[i] = item[i].cpu().numpy()
        elif type(item) == torch.Tensor:
            item = item.cpu().numpy()
        result_dict[key] = item
    return result_dict