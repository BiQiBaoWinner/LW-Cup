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
    def __init__(self, Task_label: str, factor_registry: dict, log_dir: str, save_dir: str, seed: int, sft=False):
        self.task = [Task_label]
        self.log_dir = log_dir
        self.save_dir = save_dir
        self.factor_list = list(factor_registry.keys())
        self.factor_registry = factor_registry
        self.seed = seed 
        self.binary_auc = 0
        self.sft = sft
    
    def DataPreparing(self, skip_nulldate = True):
        '''
        columns of df in data: [date, sym, features, label_5, label_10, label_20, label_40, label_60]
        '''
        
        train_dates = [str(i) for i in range(int(range_split['train'][0]), int(range_split['train'][1]) + 1)]
        valid_dates = [str(i) for i in range(int(range_split['valid'][0]), int(range_split['valid'][1]) + 1)]
        test_dates = [str(i) for i in range(int(range_split['test'][0]), int(range_split['test'][1]) + 1)]
        
        tot_tick_df = pd.read_parquet(f"{results_path}/merge_data/merge_data.parquet")
        
        train_pip = FactorPipeline(tot_tick_df, date_range=train_dates)
        train_pip.Tick_Factor_Pool.registry = self.factor_registry
        train_f = train_pip.load_factor_exposure(n_jobs=16)
        
        valid_pip = FactorPipeline(tot_tick_df, date_range=valid_dates)
        valid_pip.Tick_Factor_Pool.registry = self.factor_registry
        valid_f = valid_pip.load_factor_exposure(n_jobs=16)
        
        test_pip = FactorPipeline(tot_tick_df, date_range=test_dates)
        test_pip.Tick_Factor_Pool.registry = self.factor_registry
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
    
    def tough_tuning_hyper(self, train_feature, train_label, valid_feature, valid_label, seed):
        def objective(trial):
            params = {
                "objective": "multiclass",
                "random_state": seed,
                "device": "gpu",
                "metric": "multi_logloss",
                "num_class": len(np.unique(train_label.values.ravel())),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                # 正则化
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 50, log=True),
                # 可选：开启 L1 正则化，有时对稀疏特征有效
                # "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 10, log=True),
                # 树结构：
                "num_leaves": trial.suggest_int("num_leaves", 15, 200),
                # 采样率
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                # 分裂约束
                "min_child_samples": trial.suggest_categorical("min_child_samples", [20, 50, 100, 150]),
                "min_split_gain": trial.suggest_categorical("min_split_gain", [0.0, 0.01, 0.05]),

                "max_depth": -1, # 不限制深度，由 num_leaves 控制
                "verbose": -1,
            }
            model = lightgbm.LGBMClassifier(**params, n_estimators=1000)
            model.fit(
                train_feature, train_label.values.ravel(),
                eval_set=[(valid_feature, valid_label.values.ravel())],
                callbacks=[early_stopping(stopping_rounds=50, verbose=-1)],
                eval_metric="multi_logloss")
            val_pred_prob = model.predict_proba(valid_feature, num_iteration=model.best_iteration_)
            auc = roc_auc_score(valid_label, val_pred_prob, multi_class='ovr')
            return auc
        
        study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
        study.optimize(objective, n_trials=20, n_jobs=1)
        
        return study
    
    def soft_tuning_hyper(self, train_feature, train_label, valid_feature, valid_label, seed):
        unique_classes = np.unique(train_label.values.ravel())
        if len(unique_classes) != 2:
            raise ValueError(f"Expected exactly 2 classes for binary classification, got {len(unique_classes)}.")

        def objective(trial):
            params = {
                "objective": "binary",
                "random_state": seed,
                "device": "gpu",
                "metric": "binary_logloss",
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 50, log=True),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 200),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_samples": trial.suggest_categorical("min_child_samples", [20, 50, 100, 150]),
                "min_split_gain": trial.suggest_categorical("min_split_gain", [0.0, 0.01, 0.05]),
                "max_depth": -1,
                "verbose": -1,
            }
            
            model = lightgbm.LGBMClassifier(**params, n_estimators=1000)
            model.fit(
                train_feature, train_label.values.ravel(),
                eval_set=[(valid_feature, valid_label.values.ravel())],
                callbacks=[
                    lightgbm.early_stopping(stopping_rounds=50, verbose=-1),
                    lightgbm.log_evaluation(period=0)
                ],
                eval_metric="binary_logloss"
            )
            
            val_pred_prob = model.predict_proba(valid_feature, num_iteration=model.best_iteration_)
            auc = roc_auc_score(valid_label, val_pred_prob[:, 1])
            return auc

        study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
        study.optimize(objective, n_trials=20, n_jobs=1)
        
        return study
    
    def fit_model(self, seed):
        train_feature = self.train_x.loc[:, self.factor_list].copy()
        train_label = self.train_y.loc[:, self.task]
        valid_feature = self.valid_x.loc[:, self.factor_list].copy()
        valid_label = self.valid_y.loc[:, self.task]
        test_feature = self.test_x.loc[:, self.factor_list].copy()
        test_label = self.test_y.loc[:, self.task]

        FINAL_N_ESTIMATORS = 2000
        EARLY_STOPPING_ROUNDS = 50

        if self.sft:
            train_mask = train_label.values.ravel() != 1
            valid_mask = valid_label.values.ravel() != 1
            
            train_feature = train_feature[train_mask]
            train_label = train_label[train_mask].replace({2: 1})
            valid_feature = valid_feature[valid_mask]
            valid_label = valid_label[valid_mask].replace({2: 1})

            if len(np.unique(train_label)) < 2:
                raise ValueError("After filtering for SFT, less than 2 classes remain.")

            study = self.soft_tuning_hyper(train_feature, train_label, valid_feature, valid_label, seed)
            best_params = study.best_params
            
            final_params = {
                "objective": "binary",
                "random_state": seed,
                "device": "gpu",
                "metric": "binary_logloss",
                "verbose": -1,
                **best_params
            }
            
            self.best_model = lightgbm.LGBMClassifier(**final_params, n_estimators=FINAL_N_ESTIMATORS)
            self.best_model.fit(
                train_feature, train_label.values.ravel(),
                eval_set=[(valid_feature, valid_label.values.ravel())],
                callbacks=[
                    lightgbm.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=-1),
                    lightgbm.log_evaluation(period=0)
                ],
                eval_metric="binary_logloss"
            )
            
            best_iter = self.best_model.best_iteration_
            test_pred_prob_raw = self.best_model.predict_proba(test_feature, num_iteration=best_iter)
            
            thres1, thres2 = 0.45, 0.65
            prob_class_1 = test_pred_prob_raw[:, 1]
            test_pred_label_adjusted = np.zeros_like(prob_class_1, dtype=int)
            test_pred_label_adjusted[prob_class_1 >= thres2] = 2
            test_pred_label_adjusted[(prob_class_1 > thres1) & (prob_class_1 < thres2)] = 1 

            test_pred_label_hard = np.argmax(test_pred_prob_raw, axis=1)
            test_pred_label_hard = np.where(test_pred_label_hard == 1, 2, 0)

            original_test_label = test_label.values.ravel()
            eval_mask = original_test_label != 1
            if np.sum(eval_mask) > 0:
                cm = pd.crosstab(original_test_label[eval_mask], test_pred_label_hard[eval_mask], rownames=['True'], colnames=['Predicted'])
                print(f"Performance on Test Set (Hard Pred, excluding label 1):\n{cm}")
                cm_adj = pd.crosstab(original_test_label[eval_mask], test_pred_label_adjusted[eval_mask], rownames=['True'], colnames=['Predicted'])
                print(f"Confusion Matrix Adjusted ({thres1}, {thres2}):\n{cm_adj}")

        else:
            study = self.tough_tuning_hyper(train_feature, train_label, valid_feature, valid_label, seed)
            best_params = study.best_params
            
            final_params = {
                "objective": "multiclass",
                "random_state": seed,
                "device": "gpu",
                "metric": "multi_logloss",
                "num_class": len(np.unique(train_label.values.ravel())),
                "verbose": -1,
                **best_params
            }
            
            self.best_model = lightgbm.LGBMClassifier(**final_params, n_estimators=FINAL_N_ESTIMATORS)
            self.best_model.fit(
                train_feature, train_label.values.ravel(),
                eval_set=[(valid_feature, valid_label.values.ravel())],
                callbacks=[
                    lightgbm.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=-1),
                    lightgbm.log_evaluation(period=0)
                ],
                eval_metric="multi_logloss"
            )
            
            best_iter = self.best_model.best_iteration_
            test_pred_prob = self.best_model.predict_proba(test_feature, num_iteration=best_iter)
            test_pred_label = np.argmax(test_pred_prob, axis=1)
            
            print(f"Performance on Test Set: {self.evaluate(test_pred_label, test_label, test_pred_prob)}")
            print(f"Confusion Matrix:\n{pd.crosstab(test_label.values.ravel(), test_pred_label, rownames=['True'], colnames=['Predicted'])}")

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        suffix = "_sft" if self.sft else ""
        save_path = f"{self.save_dir}/lgbm_{self.task[0].replace('_', '')}{suffix}_{timestamp}.pkl"
        self.save_model(save_path)
        
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
        
        cm = pd.crosstab(true_label.values.ravel(), pred_label, rownames=['True'], colnames=['Predicted'])
        precision_0 = cm.loc[0, 0] / cm.loc[0, :].sum() if cm.loc[0, :].sum() > 0 else 0
        recall_0 = cm.loc[0, 0] / cm.loc[:, 0].sum() if cm.loc[:, 0].sum() > 0 else 0
        precision_2 = cm.loc[2, 2] / cm.loc[2, :].sum() if cm.loc[2, :].sum() > 0 else 0
        recall_2 = cm.loc[2, 2] / cm.loc[:, 2].sum() if cm.loc[:, 2].sum() > 0 else 0
        
        eval_results = {
            "auc": auc,
            "accuracy": acc,
            "precision": precision,
            "precision_0": precision_0,
            "precision_2": precision_2,
            "recall": recall,
            "recall_0": recall_0,
            "recall_2": recall_2,
            "f1_score": f1,
        }
        
        return eval_results
    
    def save_model(self, path):
        # 按pkl格式保存模型
        with open(path, "wb") as f:
            pickle.dump(self.best_model, f)

    
if __name__=='__main__':
    import os 
    
    Task_label = 'label_20'
    factor_list = ['tick_OBI']
    log_dir = os.path.expanduser(f"{results_path}/lgbm_logs")
    save_dir = os.path.expanduser(f"{results_path}/lgbm_models")
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    seed = 42
    model = LGBM(Task_label, factor_list, log_dir, save_dir, seed)
    data_pack = model.DataPreparing(skip_nulldate=True)
    best_model = model.fit_model(seed)
    
