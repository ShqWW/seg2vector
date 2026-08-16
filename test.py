import torch
import torch.distributed as dist
from tqdm import tqdm
from tools.get_config import get_cfg
from tools.convert_data import result_to_np
from Dataset.build import build_testset
from Models.build import build_model
import argparse
import os
from utils.load_weight import load_segnet_weight, load_vectornet_weight, load_full_weight
from Eval.evaluator import Evaluator

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='inference-config')
    parser.add_argument('--gpu_no', default=0, type=int)
    parser.add_argument('--is_multigpu', default=0, type=int)
    parser.add_argument('--cfg', default='./Config/config.py', type=str)
    parser.add_argument('--load_path', default='./work_dir/ckpt', type=str)
    parser.add_argument('--load_no', nargs='+', type=str, help='the no of the pretrain weight')
    parser.add_argument('--result_path', default='./result_demo', type=str)
    parser.add_argument('--view_path', default='./view_demo', type=str)
    parser.add_argument('--is_view', default=1, type=int)
    parser.add_argument('--in_domain_test', default=1, type=int)
    args = parser.parse_args()
    return args

cfg = get_cfg(parse_args())
MULTIGPU = cfg.is_multigpu   # use multiple gpu or not
GPUNO = cfg.gpu_no  # single gpu no
ISMAINPROCESS = True
if __name__ == "__main__":
    if MULTIGPU:
        GPUNO = int(os.environ["LOCAL_RANK"])
        device_ids = range(torch.cuda.device_count())
        torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(GPUNO)

    net = build_model(cfg)

    load_segnet_weight(net, cfg.load_path, cfg.load_no)
    load_vectornet_weight(net, cfg.load_path, cfg.load_no)

    net = net.cuda()
    net.eval()

    tsset = build_testset(cfg)
    total_len = len(tsset)

    if MULTIGPU:
        world_size = dist.get_world_size()
        indices = list(range(GPUNO, total_len, world_size))
        print(f"Total dataset length: {total_len}")
        print(f"World size: {world_size}, Current Rank: {GPUNO}, Handling data count: {len(indices)}")
    else:
        indices = list(range(total_len))
        print(f"Total dataset length: {total_len}")

    evaluator = Evaluator(cfg)

    for idx in tqdm(indices):
        with torch.no_grad():
            sample_dict = tsset[idx]
            sample_dict['imgs'] = sample_dict['imgs'].cuda()
            outputs = net(sample_dict)
            outputs = result_to_np(outputs)
            evaluator.process(sample_dict, outputs)

    if MULTIGPU:
        dist.destroy_process_group()