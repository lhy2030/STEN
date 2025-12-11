import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import time
import importlib as imp
import optuna
import numpy as np
from utils.dataloading import import_ts_data_unsupervised
from metrics import ts_metrics, point_adjustment
from metrics.metrics import *
from metrics import point_adjustment
from metrics import ts_metrics_enhanced

dataset_root = f'./data/'

def objective(trial, args):
    seq_len = trial.suggest_int('seq_len', 10, 50)
    batch_size = trial.suggest_int('batch_size', 32, 128)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    hidden_dim = trial.suggest_int('hidden_dim', 128, 512)
    alpha = trial.suggest_uniform('alpha', 0.5, 2.0)
    beta = trial.suggest_uniform('beta', 0.5, 2.0)

    model_configs = {
        'seq_len': seq_len,
        'stride': 1,
        'alpha': alpha,
        'beta': beta,
        'lr': lr,
        'batch_size': batch_size,
        'epoch': 10,
        'hidden_dim': hidden_dim
    }

    module = imp.import_module('models')
    model_class = getattr(module, args.model)

    dataset_name = args.dataset
    data_pkg = import_ts_data_unsupervised(dataset_root, dataset_name, entities=args.entities, combine=args.entity_combined)
    train_lst, test_lst, label_lst, name_lst = data_pkg
    train_data = train_lst[0]
    test_data = test_lst[0]
    labels = label_lst[0]

    clf = model_class(**model_configs, random_state=42)
    clf.fit(train_data)
    scores = clf.decision_function(test_data)

    eval_metrics = ts_metrics(labels, scores)
    
    return eval_metrics[0]

def run_optuna(args):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, args), n_trials=100)  # 優化 100 次

    best_trial = study.best_trial
    print(f"Best trial: {best_trial.params}")

    best_params = best_trial.params
    model_configs = {
        'seq_len': best_params['seq_len'],
        'stride': 1,
        'alpha': best_params['alpha'],
        'beta': best_params['beta'],
        'lr': best_params['lr'],
        'batch_size': best_params['batch_size'],
        'epoch': 10,
        'hidden_dim': best_params['hidden_dim']
    }

    return model_configs

parser = argparse.ArgumentParser()
parser.add_argument("--runs", type=int, default=5,
                    help="how many times we repeat the experiments to "
                         "obtain the average performance")
parser.add_argument("--output_dir", type=str, default='@records/',
                    help="the output file path")
parser.add_argument("--dataset", type=str,
                    default='Epilepsy',
                    help='dataset name or a list of names split by comma')
parser.add_argument("--entities", type=str,
                    default='FULL',
                    help='FULL represents all the csv file in the folder, '
                         'or a list of entity names split by comma')
parser.add_argument("--entity_combined", type=int, default=1)
parser.add_argument("--model", type=str, default='STEN', help="training model")

parser.add_argument('--silent_header', action='store_true')
parser.add_argument("--flag", type=str, default='')
parser.add_argument("--note", type=str, default='')
parser.add_argument('--seq_len', type=int, default=10)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.00001)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--hidden_dim', type=int, default=256)

parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--beta', type=float, default=1)
parser.add_argument('--delta', type=float, default=0.6)

args = parser.parse_args()

module = imp.import_module('models')
model_class = getattr(module, args.model)

path = 'configs.yaml'
with open(path) as f:
    d = yaml.safe_load(f)
    try:
        model_configs = d[args.model]
    except KeyError:
        print(f'config file does not contain default parameter settings of {args.model}')
        model_configs = {}

model_configs['seq_len'] = args.seq_len
model_configs['stride'] = args.stride
model_configs['alpha'] = args.alpha
model_configs['beta'] = args.beta
model_configs['lr'] = args.lr
model_configs['batch_size'] = args.batch_size
model_configs['epoch'] = args.epoch
model_configs['hidden_dim'] = args.hidden_dim

print(f'Model Configs: {model_configs}')

cur_time = time.strftime("%m-%d %H.%M.%S", time.localtime())
os.makedirs(args.output_dir, exist_ok=True)
result_file = os.path.join(args.output_dir, f'{args.model}.{args.flag}.csv')

if not args.silent_header:
    with open(result_file, 'a') as f:
        print('\n---------------------------------------------------------', file=f)
        print(f'model: {args.model}, dataset: {args.dataset}, {args.runs} runs, {cur_time}', file=f)
        for k in model_configs.keys():
            print(f'Parameters,\t [{k}], \t\t  {model_configs[k]}', file=f)
        print(f'Note: {args.note}', file=f)
        print(f'---------------------------------------------------------', file=f)

model_configs = run_optuna(args)

dataset_name_lst = args.dataset.split(',')
for dataset in dataset_name_lst:
    entity_metric_lst = []
    entity_metric_std_lst = []
    data_pkg = import_ts_data_unsupervised(dataset_root, dataset, entities=args.entities, combine=args.entity_combined)
    train_lst, test_lst, label_lst, name_lst = data_pkg

    for train_data, test_data, labels, dataset_name in zip(train_lst, test_lst, label_lst, name_lst):
        entries = []
        t_lst = []
        for i in range(args.runs):
            start_time = time.time()
            print(f'\nRunning [{i + 1}/{args.runs}] of [{args.model}] on Dataset [{dataset_name}]')

            clf = model_class(**model_configs, random_state=42 + i)
            clf.fit(train_data)
            scores = clf.decision_function(test_data)

            t = time.time() - start_time
            eval_metrics = ts_metrics(labels, scores)
            adj_eval_metrics_raw = ts_metrics(labels, point_adjustment(labels, scores))

            anormly_ratio = args.delta
            thresh = np.percentile(scores, 100 - anormly_ratio)
            print("Threshold :", thresh)

            gt = labels.astype(int)
            pred = (scores > thresh).astype(int)

            txt = f'{dataset_name},'
            txt += ', '.join(['%.4f' % a for a in eval_metrics]) + \
                   ', pa, ' + \
                   ', '.join(['%.4f' % a for a in adj_eval_metrics_raw])
            txt += f', model, {args.model}, time, {t:.1f} s, runs, {i + 1}/{args.runs}'
            print(txt)

            adj_eval_metrics = ts_metrics_enhanced(labels, point_adjustment(labels, scores), pred)
            entries.append(adj_eval_metrics)
            t_lst.append(t)

        avg_entries = np.average(np.array(entries), axis=0)
        std_entries = np.std(np.array(entries), axis=0)

        with open(result_file, 'a') as f:
            print(f'data, auroc, std, aupr, std, best_f1, std, best_p, std, best_r, std, aff_p, std, '
                  f'aff_r, std, vus_r_auroc, std, vus_r_aupr, std, vus_roc, std, vus_pr, std, time, model',
                  file=f)
            txt = '%s, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, ' \
                  '%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, ' \
                  '%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.1f, %s, %s ' % \
                  (dataset_name, avg_entries[0], std_entries[0], avg_entries[1], std_entries[1],
                   avg_entries[2], std_entries[2], avg_entries[3], std_entries[3],
                   avg_entries[4], std_entries[4], avg_entries[5], std_entries[5],
                   avg_entries[6], std_entries[6], avg_entries[7], std_entries[7],
                   avg_entries[8], std_entries[8], avg_entries[9], std_entries[9],
                   avg_entries[10], std_entries[10], np.average(t_lst), args.model, str(model_configs))
            print(txt)
            print(txt, file=f)

            f.close()
