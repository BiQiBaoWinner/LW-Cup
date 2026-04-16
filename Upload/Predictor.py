from __future__ import annotations

import os
import numpy as np
import pandas as pd
import torch
import pickle

import sys
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

import pickle
from factor_pool.pipeline import FactorPipeline
from factor_pool.tick_factor_pool import *
from config import FACTORS
import json

class Predictor:
    def __init__(self) -> None:
        
        with open(os.path.join(os.path.dirname(__file__), 'best_model.pkl'), 'rb') as f:
            self.model = pickle.load(f)
            
        with open(os.path.join(os.path.dirname(__file__), 'config.json'), 'r') as f:
            config = json.load(f)
            
        self.cols = config['feature'] + ['date', 'sym', 'time']

    def _calc_factors(self, single_df):
        
        if single_df.empty or 'date' not in single_df.columns or 'time' not in single_df.columns:
            raise ValueError("输入的 DataFrame 为空或缺少必要的 'date' 或 'time' 列。")
        
        single_df['Ndate'] = pd.to_datetime(single_df['date'].astype(int), unit='D', origin=pd.Timestamp('2020-01-01'))
        single_df['timestamp'] = pd.to_datetime(single_df['Ndate'].dt.strftime('%Y-%m-%d') + ' ' + single_df['time'].astype(str), format='%Y-%m-%d %H:%M:%S')
        
        pip = FactorPipeline(single_df, date_range=None)
        pip.Tick_Factor_Pool.registry = FACTORS
        
        f = pip.load_factor_exposure(n_jobs=16)
        
        return f
    
    def predict(self, batches: list[pd.DataFrame]) -> list[list[int]]:
        """
        返回: [[y_label0, y_label1, ...], ...]，与 batches 等长；各头 argmax 类下标 int，三分类时为 [0,1,2]。
        """
        inputs = []
        for i, df in enumerate(batches):
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Batch {i} is not a DataFrame.")
            if df.empty:
                raise ValueError(f"Batch {i} is empty.")
            
            f = self._calc_factors(df[self.cols])
            inputs.append(f.iloc[-1, :])
        inputs = pd.DataFrame(inputs, columns=f.columns)
        
        pred_label = self.model.predict(inputs)
        pred_label = [ [int(l)] for l in pred_label]
        
        return pred_label


if __name__ == "__main__":

    
    predictor = Predictor()
    
    with open(os.path.expanduser("~/LWCUP/results/test_data/test_data.pkl"), 'rb') as f:
        test_data = pickle.load(f)

    batches = [ d for d in test_data ]
    
    y = predictor.predict(batches[0:5])
    print(y)