import os
import subprocess
import numpy as np
import argparse
import json
from functools import partial
from pathos.multiprocessing import ProcessingPool

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='apls-config')
    parser.add_argument('--dataset', default='', type=str)
    parser.add_argument('--data_path', default='./', type=str)
    parser.add_argument('--result_path', default='./', type=str)
    parser.add_argument('--num_processes', default=32, type=int)
    parser.add_argument('--in_domain_test', default=1, type=int)

    args = parser.parse_args()
    return args

def get_path_list():
    if args.dataset == 'spacenet':
        datajson =os.path.join(args.data_path, 'RGB_1.0_meter', 'dataset.json')
        with open(datajson, 'r', encoding='utf-8') as file:
            meta_name_list = json.load(file)['test']
        pred_path_list = [os.path.join(args.result_path, 'graph', meta_name + '.p') for meta_name in meta_name_list]
        gt_path_list = [os.path.join(args.data_path, 'RGB_1.0_meter', meta_name + '__gt_graph.p') for meta_name in meta_name_list]

    if args.dataset == 'cityscale':
        with open(os.path.join(args.data_path, 'data_split.json'), 'r', encoding='utf-8') as file:
            meta_name_list = [str(i) for i in json.load(file)['test']]
        pred_path_list = [os.path.join(args.result_path, 'graph', meta_name + '.p') for meta_name in meta_name_list]
        gt_path_list = [os.path.join(args.data_path, '20cities', f'region_{meta_name}_graph_gt.pickle') for meta_name in meta_name_list]

    if args.dataset == 'globalscale':
        if args.in_domain_test:
            meta_name_list = [str(i) for i in range(624)]
            gt_path_list = [os.path.join(args.data_path, 'in-domain-test', f'region_{meta_name}_graph_gt.pickle') for meta_name in
                            meta_name_list]
        else:
            meta_name_list = [str(i) for i in range(130)]
            gt_path_list = [os.path.join(args.data_path, 'out_of_domain', f'region_{meta_name}_graph_gt.pickle') for meta_name in
                            meta_name_list]
        pred_path_list = [os.path.join(args.result_path, 'graph', meta_name + '.p') for meta_name in meta_name_list]
    return pred_path_list, gt_path_list, meta_name_list

def process_item(pred_path, gt_path, meta_name, result_dir):
    print(f"========================{meta_name}======================")
    out_path = os.path.join(result_dir, f'{meta_name}.txt')
    subprocess.run([
        'python', 'road_graph_metric/topo/main.py',
        '--graph_gt', gt_path,
        '--graph_prop', pred_path,
        '--output', out_path
    ])

def get_final_topo(out_path, result_dir):
    topo = []
    precision = []
    recall = []
    for file_name in os.listdir(result_dir):
        if '.txt' not in file_name:
            continue
        with open(os.path.join(result_dir, file_name)) as f:
            lines = f.readlines()
        p = float(lines[-1].split(' ')[0].split('=')[-1])
        r = float(lines[-1].split(' ')[-1].split('=')[-1])

        if p==0 and r==0:
            print(f'Error for file: {file_name}')
            if args.dataset != 'spacenet': #  # the spacenet has some cases with very low TOPOS, which chould be ingored (same as the samroad)
                precision.append([file_name, 0])
                recall.append([file_name, 0])
                topo.append([file_name, 0])
        else:
            precision.append([file_name, p])
            recall.append([file_name, r])
            if args.dataset != 'spacenet':
                topo.append([file_name, 2 * p * r / (p + r)])
    precision_mean = np.mean(np.array([p[1] for p in precision]))
    recall_mean = np.mean(np.array([r[1] for r in recall]))

    if args.dataset != 'spacenet':
        topo_mean = np.mean(np.array([topo[1] for topo in topo]))
    else:
        topo_mean = 2 * precision_mean * recall_mean / (precision_mean + recall_mean)

    print('TOPO', topo_mean, 'Precision', precision_mean, 'Recall', recall_mean)
    save_dir = os.path.join(out_path, 'score')
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'topo.json'), 'w') as jf:
        json.dump({'mean topo': [topo_mean, precision_mean, recall_mean], 'prec': precision, 'recall': recall, 'f1': topo}, jf)



args = parse_args()
if __name__ == '__main__':
    print('start...')
    pred_path_list, gt_path_list, meta_name_list = get_path_list()
    result_dir = os.path.join(args.result_path, 'results/topo')
    os.makedirs(result_dir, exist_ok=True)
    if args.num_processes == 1:
        for pred_path, gt_path, meta_name in zip(pred_path_list, gt_path_list, meta_name_list):
            process_item(pred_path, gt_path, meta_name, result_dir)
    else:
        func = partial(process_item, result_dir=result_dir)
        with ProcessingPool(processes=args.num_processes) as pool:
            list(pool.imap(func, pred_path_list, gt_path_list, meta_name_list))
    get_final_topo(args.result_path, result_dir)










