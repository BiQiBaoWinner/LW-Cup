import pandas as pd
import numpy as np
import json
import os
import sys
import pickle
import datetime
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
from typing import List
from config import data_path, results_path, range_split

import optuna

np.NaN = np.nan  # 确保兼容性
from numpy import nan as npNaN

from tqdm import tqdm
import lightgbm
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, f1_score, recall_score, log_loss
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

from factor_pool.pipeline import FactorPipeline


class LGBM():
    def __init__(self, Task_label: str, factor_list: List[str], log_dir: str, save_dir: str, seed: int):
        self.task = [Task_label]
        self.log_dir = log_dir
        self.save_dir = save_dir
        self.factor_list = factor_list
        self.seed = seed 
        self.binary_auc = 0
    
    def DataPreparing(self, skip_nulldate = True):
        '''
        columns of df in data: [date, sym, features, label_5, label_10, label_20, label_40, label_60]
        '''
        
        train_dates = [str(i) for i in range(int(range_split['train'][0]), int(range_split['train'][1]) + 1)]
        valid_dates = [str(i) for i in range(int(range_split['valid'][0]), int(range_split['valid'][1]) + 1)]
        test_dates = [str(i) for i in range(int(range_split['test'][0]), int(range_split['test'][1]) + 1)]
        
        tot_tick_df = pd.read_parquet(f"{results_path}/merge_data/merge_data.parquet")
        
        train_pip = FactorPipeline(tot_tick_df, date_range=train_dates)
        train_f = train_pip.load_factor_exposure(n_jobs=16)
        
        valid_pip = FactorPipeline(tot_tick_df, date_range=valid_dates)
        valid_f = valid_pip.load_factor_exposure(n_jobs=16)
        
        test_pip = FactorPipeline(tot_tick_df, date_range=test_dates)
        test_f = test_pip.load_factor_exposure(n_jobs=16)
        
        if skip_nulldate:
            with open(f"{os.path.expanduser(results_path)}/merge_data/null_sym_date_cols.json", 'r') as f:
                null_sym_date_cols = json.load(f)
            for sym, date_cols in null_sym_date_cols.items():
                for date, cols in date_cols.items():
                    # 直接扔掉这些日期的数据
                    if date in train_dates:
                        train_f = train_f.drop(index=(sym, int(date)), errors='ignore')
                    if date in valid_dates:
                        valid_f = valid_f.drop(index=(sym, int(date)), errors='ignore')
                    if date in test_dates:
                        test_f = test_f.drop(index=(sym, int(date)), errors='ignore')
            
        tot_label_df = tot_tick_df.set_index(['sym', 'timestamp']).loc[:, self.task]
        train_label = tot_label_df.loc[train_f.index]
        valid_label = tot_label_df.loc[valid_f.index]
        test_label = tot_label_df.loc[test_f.index]
        
        data_pack = {
            "train": (train_f, train_label),
            "valid": (valid_f, valid_label),
            "test": (test_f, test_label)
        }
        
        self.train_x = train_f
        self.train_y = train_label
        self.valid_x = valid_f
        self.valid_y = valid_label
        self.test_x = test_f
        self.test_y = test_label
        
        return data_pack
    
    def fit_model(self, seed):
        
        train_feature = self.train_x.loc[:, self.factor_list ].copy()
        train_label = self.train_y.loc[:, self.task]
        valid_feature = self.valid_x.loc[:, self.factor_list ].copy()
        valid_label = self.valid_y.loc[:, self.task]
        test_feature = self.test_x.loc[:, self.factor_list ].copy()
        test_label = self.test_y.loc[:, self.task]
        
        def objective(trial):
            params = {
                "objective": "multiclass",
                "random_state": seed,
                "device": "gpu",
                "metric": "multi_logloss",
                "num_class": len(np.unique(train_label.values.ravel())),
                "reg_lambda": trial.suggest_float("L2 regular", 1e-1, 20),
                # "reg_alpha": trial.suggest_float("L1 regular", 1e-1, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 200, 600),
                "max_depth": trial.suggest_categorical("max_depth", [6, 10, 20, 30]),
                "num_leaves": trial.suggest_categorical("num_leaves", [11, 21, 31, 41, 61, 91, 121, 151, 181]),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_samples": trial.suggest_categorical("min_child_samples", [150, 200, 250, 300]),
                "min_split_gain": trial.suggest_categorical("min_split_gain", [0.05, 0.1, 0.5, 1, 2]),
                "min_child_weight": trial.suggest_float("min_child_weight", 1e-4, 1e-2, log=True),
            }
            model = lightgbm.LGBMClassifier(**params)
            model.fit(
                train_feature, train_label.values.ravel(),
                eval_set=[(valid_feature, valid_label.values.ravel())],
                callbacks=[early_stopping(stopping_rounds=30, verbose=-1)],
                eval_metric="multi_logloss")
            val_pred_prob = model.predict_proba(valid_feature)
            auc = roc_auc_score(valid_label, val_pred_prob, multi_class='ovr')
            return auc
        
        study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=10, n_jobs=-1) 
        
        best_params = study.best_params
        best_params.update({
            "objective": "multiclass",
            "random_state": seed,
            "device": "gpu",
            "metric": "multi_logloss",
            "num_class": len(np.unique(train_label.values.ravel())),
        })
        
        self.best_model = lightgbm.LGBMClassifier(**best_params)
        self.best_model.fit(
            train_feature, train_label.values.ravel(),
            eval_set=[(valid_feature, valid_label.values.ravel())],
            callbacks=[early_stopping(stopping_rounds=30, verbose=-1)],
            eval_metric="multi_logloss")

        test_pred_label = self.best_model.predict(test_feature)
        test_pred_prob = self.best_model.predict_proba(test_feature)
        print(f"Performance on Test Set:{self.evaluate(test_pred_label, test_label, test_pred_prob)}")
        print(f"Confusion Matrix:\n{pd.crosstab(test_label.values.ravel(), test_pred_label, rownames=['True'], colnames=['Predicted'])}")
        self.save_model(f"{self.save_dir}/lgbm_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pkl")
        
        return self.best_model
    
    def predict(self, f):
        pred_label = self.best_model.predict(f)
        return pred_label
    
    def evaluate(self, pred_label, true_label, pred_prob=None):
        # 获取所有可能的类别标签，确保 log_loss 和指标计算一致
        labels = np.unique(true_label.values.ravel())
        
        if pred_prob is not None:
            auc = roc_auc_score(true_label, pred_prob, multi_class='ovr', labels=labels)
            logloss = log_loss(true_label, pred_prob, labels=labels)
        else:
            # 如果没有概率
            auc = accuracy_score(true_label, pred_label) 
            logloss = np.nan
            
        acc = accuracy_score(true_label, pred_label)
        precision = precision_score(true_label, pred_label, average='weighted', zero_division=0)
        recall = recall_score(true_label, pred_label, average='weighted', zero_division=0)
        f1 = f1_score(true_label, pred_label, average='weighted', zero_division=0)
        
        eval_results = {
            "auc": auc,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "log_loss": logloss
        }
        
        return eval_results
    
    def save_model(self, path):
        # 按pkl格式保存模型
        with open(path, "wb") as f:
            pickle.dump(self.best_model, f)

    
if __name__=='__main__':
    
    Task_label = 'label_20'
    factor_list = ['tick_OBI']
    log_dir = f"{results_path}/lgbm_logs"
    save_dir = f"{results_path}/lgbm_models"
    seed = 42
    model = LGBM(Task_label, factor_list, log_dir, save_dir, seed)
    data_pack = model.DataPreparing(skip_nulldate=True)
    best_model = model.fit_model(seed)
    
