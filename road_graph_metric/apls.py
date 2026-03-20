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
    parser.add_argument('--tmp_path', default='./metric_tmp', type=str)
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
    gt_json = os.path.join(args.tmp_path, f'gt_{meta_name}.json')
    prop_json = os.path.join(args.tmp_path, f'prop_{meta_name}.json')

    subprocess.run(['python', 'road_graph_metric/apls/convert.py', gt_path, gt_json])
    subprocess.run(['python', 'road_graph_metric/apls/convert.py', pred_path, prop_json])
    if args.dataset == 'spacenet':
        subprocess.run(['go', 'run', 'road_graph_metric/apls/main.go', gt_json, prop_json, os.path.join(result_dir, f'{meta_name}.txt'), 'spacenet'])
    else:
        subprocess.run(['go', 'run', 'road_graph_metric/apls/main.go', gt_json, prop_json, os.path.join(result_dir, f'{meta_name}.txt')])

    os.remove(gt_json)
    os.remove(prop_json)

def get_final_apls(out_path, result_dir):
    name_list = os.listdir(result_dir)
    name_list.sort()
    apls = []
    output_apls = []
    for file_name in name_list:
        with open(os.path.join(result_dir, file_name)) as f:
            lines = f.readlines()
        if 'NaN' in lines[0]:
            print(f'Error for file: {file_name}')
            if args.dataset != 'spacenet': # the spacenet has some cases with very low APLS, which chould be ingored (same as the samroad)
                apls.append(0.0)
                output_apls.append([file_name, 0.0])
        else:
            apls.append(float(lines[0].split(' ')[-1]))
            output_apls.append([file_name, float(lines[0].split(' ')[-1])])

    print('APLS', np.sum(apls) / len(apls))
    out_path = os.path.join(out_path, 'score')
    os.makedirs(out_path, exist_ok=True)
    with open(os.path.join(out_path, 'apls.json'), 'w') as jf:
        json.dump({'apls': output_apls, 'final_APLS': np.mean(apls)}, jf)

args = parse_args()
if __name__ == '__main__':
    print('start...')
    pred_path_list, gt_path_list, meta_name_list = get_path_list()
    result_dir = os.path.join(args.result_path, 'results/apls')
    os.makedirs(args.tmp_path, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    if args.num_processes == 1:
        for pred_path, gt_path, meta_name in zip(pred_path_list, gt_path_list, meta_name_list):
            process_item(pred_path, gt_path, meta_name, result_dir)
    else:
        func = partial(process_item, result_dir=result_dir)
        with ProcessingPool(processes=args.num_processes) as pool:
            list(pool.imap(func, pred_path_list, gt_path_list, meta_name_list))
    get_final_apls(args.result_path, result_dir)

    # os.remove()









