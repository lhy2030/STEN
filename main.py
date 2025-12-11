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

# 定義 Optuna 目標函數
def objective(trial):
    # 從 Optuna 中選擇超參數
    seq_len = trial.suggest_int('seq_len', 10, 50)
    batch_size = trial.suggest_int('batch_size', 32, 128)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    hidden_dim = trial.suggest_int('hidden_dim', 128, 512)
    alpha = trial.suggest_uniform('alpha', 0.5, 2.0)
    beta = trial.suggest_uniform('beta', 0.5, 2.0)

    # 更新模型的超參數配置
    model_configs = {
        'seq_len': seq_len,
        'batch_size': batch_size,
        'lr': lr,
        'hidden_dim': hidden_dim,
        'alpha': alpha,
        'beta': beta,
        'stride': 1,
        'epoch': 10  # 固定epoch數量
    }

    # 模型初始化
    module = imp.import_module('models')
    model_class = getattr(module, 'STEN')  # 預設使用 'STEN' 模型

    # 加載資料集
    dataset_name_lst = ['Epilepsy']  # 使用 'Epilepsy' 作為範例
    dataset_name = dataset_name_lst[0]  # 假設只有一個資料集

    data_pkg = import_ts_data_unsupervised(dataset_root, dataset_name, entities='FULL', combine=1)
    train_lst, test_lst, label_lst, name_lst = data_pkg

    # 選擇一組資料進行訓練
    train_data = train_lst[0]
    test_data = test_lst[0]
    labels = label_lst[0]
    
    clf = model_class(**model_configs, random_state=42)
    clf.fit(train_data)
    scores = clf.decision_function(test_data)

    eval_metrics = ts_metrics(labels, scores)
    adj_eval_metrics_raw = ts_metrics(labels, point_adjustment(labels, scores))

    # 返回目標函數的評估指標（如 AUROC）
    return eval_metrics[0]  # 假設返回 AUROC 作為優化目標

# 在 Optuna 中進行超參數優化
study = optuna.create_study(direction='maximize')  # 假設目標是最大化 AUROC
study.optimize(objective, n_trials=100)  # 優化 100 次

# 顯示最佳結果
best_trial = study.best_trial
print(f"Best trial: {best_trial.params}")

# 使用最佳超參數配置進行最終模型訓練
best_params = best_trial.params
model_configs = {
    'seq_len': best_params['seq_len'],
    'batch_size': best_params['batch_size'],
    'lr': best_params['lr'],
    'hidden_dim': best_params['hidden_dim'],
    'alpha': best_params['alpha'],
    'beta': best_params['beta'],
    'stride': 1,
    'epoch': 10
}

# 最終結果文件
cur_time = time.strftime("%m-%d %H.%M.%S", time.localtime())
os.makedirs(args.output_dir, exist_ok=True)
result_file = os.path.join(args.output_dir, f'{args.model}.{args.flag}.csv')

# 實驗的詳細結果
dataset_name_lst = args.dataset.split(',')
for dataset in dataset_name_lst:
    entity_metric_lst = []
    entity_metric_std_lst = []
    data_pkg = import_ts_data_unsupervised(dataset_root, dataset, entities=args.entities, combine=args.entity_combined)
    train_lst, test_lst, label_lst, name_lst = data_pkg

    for train_data, test_data, labels, dataset_name in zip(train_lst, test_lst, label_lst, name_lst):
        entries = []
        t_lst = []
        runs = args.runs

        for i in range(runs):
            start_time = time.time()
            print(f'\nRunning [{i + 1}/{args.runs}] of [{args.model}] on Dataset [{dataset_name}]')

            clf = model_class(**model_configs, random_state=42 + i)
            clf.fit(train_data)
            scores = clf.decision_function(test_data)

            t = time.time() - start_time
            eval_metrics = ts_metrics(labels, scores)
            adj_eval_metrics_raw = ts_metrics(labels, point_adjustment(labels, scores))

            thresh = np.percentile(scores, 100 - args.delta)
            gt = labels.astype(int)
            pred = (scores > thresh).astype(int)

            adj_eval_metrics = ts_metrics_enhanced(labels, point_adjustment(labels, scores), pred)

            entries.append(adj_eval_metrics)
            t_lst.append(t)

        avg_entries = np.average(np.array(entries), axis=0)
        std_entries = np.std(np.array(entries), axis=0)

        entity_metric_lst.append(avg_entries)
        entity_metric_std_lst.append(std_entries)

        f = open(result_file, 'a')
        print(f'data, auroc, std, aupr, std, best_f1, std, best_p, std, best_r, std, aff_p, std, '
              f'aff_r, std, vus_r_auroc, std, vus_r_aupr, std, vus_roc, std, vus_pr, std, time, model',
            file=f)
        txt = '%s, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, ' \
              '%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, ' \
              '%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.1f, %s, %s '% \
              (dataset_name, avg_entries[0], std_entries[0], avg_entries[1], std_entries[1],
               avg_entries[2], std_entries[2], avg_entries[3], std_entries[3],
               avg_entries[4], std_entries[4], avg_entries[5], std_entries[5],
               avg_entries[6], std_entries[6], avg_entries[7], std_entries[7],
               avg_entries[8], std_entries[8], avg_entries[9], std_entries[9],
               avg_entries[10], std_entries[10], np.average(t_lst), args.model, str(model_configs))
        print(txt)
        print(txt, file=f)

        f.close()
