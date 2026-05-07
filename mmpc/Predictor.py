from __future__ import annotations

import os
import numpy as np
import pandas as pd
import torch
import pickle
import json
from joblib import Parallel, delayed

import sys
here = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(here)
for p in (here, parent):
    if p not in sys.path:
        sys.path.append(p)

from pipeline import FactorPipeline
from tick_factor_pool import *
from Myconfig import FACTORS, ob_cols_derive1, ob_cols_derive2, ob_cols_pro
    
import json

class Predictor:
    def __init__(self) -> None:
        
        self.window = 1
        self.num_classes = 3
        self.n_jobs = 16
        self.factor_names = {}
        
        with open(os.path.join(os.path.dirname(__file__), 'lgbm_label60.pkl'), 'rb') as f:
            self.model = pickle.load(f)
        with open(os.path.join(os.path.dirname(__file__), 'factor_label60.json'), 'r') as f:
            self.factor_names[60] = list(json.load(f).keys())
            
        with open(os.path.join(os.path.dirname(__file__), 'config.json'), 'r') as f:
            config = json.load(f)
        
        # for i in [5, 10, 20, 40, 60]:
        #     with open(os.path.join(os.path.dirname(__file__), f'lgbm_label{i}.pkl'), 'rb') as f:
        #         if not hasattr(self, 'models'):
        #             self.models = []
        #         self.models.append(pickle.load(f))
        #     with open(os.path.join(os.path.dirname(__file__), f'factor_label{i}.json'), 'r') as f:
        #         self.factor_names[i] = list(json.load(f).keys())

        # self.cols 应该是原始特征列（用于因子计算）
        self.cols = config['feature']
    
    def _build_features(self, df: pd.DataFrame, label_idx: int = 5) -> pd.DataFrame:

        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input is not a DataFrame.")
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        if 'date' not in df.columns or 'time' not in df.columns:
            raise ValueError("Input DataFrame must contain 'date' and 'time' columns.")

        df = df.copy()
        df['Ndate'] = pd.to_datetime(df['date'].astype(int), unit='D', origin=pd.Timestamp('2020-01-01'))
        df['timestamp'] = pd.to_datetime(
            df['Ndate'].dt.strftime('%Y-%m-%d') + ' ' + df['time'].astype(str),
            format='%Y-%m-%d %H:%M:%S'
        )

        pip = FactorPipeline(df, date_range=None)
        # 从字典FACTORS中获取对应标签的因子列表
        factor_name = self.factor_names[label_idx]
        # 将这些因子注册到 Tick_Factor_Pool 中
        pip.Tick_Factor_Pool.registry = {f: FACTORS.get(f, None) for f in factor_name}
        factor_panel = pip.load_factor_exposure(n_jobs = min(16, os.cpu_count() or 1))
        if factor_panel.empty:
            error_info = "因子面板为空！"
            raise ValueError(error_info)

        factor_panel = factor_panel.reset_index()
        factor_panel = factor_panel.sort_values("timestamp", kind="stable")
        features = factor_panel[factor_name]

        if len(features) < self.window:
            raise ValueError(f"Not enough rows for window={self.window}. Got {len(features)}")
        
        return features.iloc[-self.window:, :]
    
    def _process_single_batch(self, batch: pd.DataFrame) -> list[int]:
        """单个batch的处理逻辑"""
        factor = self._build_features(batch, label_idx=60)
        pred = self.model.predict(factor)[0]
        return [int(pred)]
    
    def _process_single_batch_multitask(self, batch: pd.DataFrame) -> list[int]:
        """单个batch的处理逻辑"""
        concat_res = pd.concat([pd.Series(m.predict(self._build_features(batch)), name=f'model_{i}') for i, m in enumerate(self.models)], axis=1)
        pred = [int(concat_res.iloc[0, i]) for i in range(5)]
        return pred
    
    def predict(self, batches: list[pd.DataFrame]) -> list[list[int]]:
        preds = Parallel(n_jobs=self.n_jobs, pre_dispatch='2*n_jobs')(
            delayed(self._process_single_batch)(batch) for batch in batches
        )
        return list(preds)


if __name__ == "__main__":

    
    predictor = Predictor()
    
    with open(os.path.expanduser("~/LWCUP/results/test_data/test_data.pkl"), 'rb') as f:
        test_data = pickle.load(f)

    batches = [ d for d in test_data ]
    # print(batches)
    y = predictor.predict(batches[0:500])
    print(y)