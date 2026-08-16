import torch

def to_gpu(sample_batch):
    for key in sample_batch.keys():
        if 'list' in key:
            if isinstance(sample_batch[key][0], torch.Tensor):
                sample_batch[key] = [data.cuda() for data in sample_batch[key]]
        else:
            if isinstance(sample_batch[key], torch.Tensor):
                sample_batch[key] = sample_batch[key].cuda() 
    return sample_batch