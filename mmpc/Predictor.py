from __future__ import annotations

import os
import numpy as np
import pandas as pd
import torch
import pickle
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import sys
here = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(here)
for p in (here, parent):
    if p not in sys.path:
        sys.path.append(p)

from pipeline import FactorPipeline
from tick_factor_pool import *
from config import FACTORS, ob_cols_derive1, ob_cols_derive2, ob_cols_pro
from DLOB import DLOB
    
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
        
        f = pip.load_factor_exposure(n_jobs=1)
        
        return f
    
    def predict(self, batches: list[pd.DataFrame], n_jobs: int = 1) -> list[list[int]]:
        """
        返回: [[y_label0, y_label1, ...], ...]，与 batches 等长；各头 argmax 类下标 int，三分类时为 [0,1,2]。
        """
        inputs = []

        def _validate_batch(i, df):
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Batch {i} is not a DataFrame.")
            if df.empty:
                raise ValueError(f"Batch {i} is empty.")
            
            if 'sym' not in df.columns:
                df['sym'] = 'unknown_sym'
            
            return df[self.cols]

        if n_jobs <= 1:
            for i, df in enumerate(batches):
                data = _validate_batch(i, df)
                f = self._calc_factors(data)
                if f.empty:
                    raise ValueError(f"Batch {i} factor result is empty.")
                inputs.append(f.iloc[-1, :])
        else:
            tasks = []
            for i, df in enumerate(batches):
                data = _validate_batch(i, df)
                tasks.append((i, data))

            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                future_to_idx = {
                    executor.submit(_calc_factors_worker, data): i
                    for i, data in tasks
                }
                results = {}
                for future in as_completed(future_to_idx):
                    i = future_to_idx[future]
                    f = future.result()
                    if f is None or f.empty:
                        raise ValueError(f"Batch {i} factor result is empty.")
                    results[i] = f.iloc[-1, :]

            inputs = [results[i] for i in range(len(batches))]

        inputs = pd.DataFrame(inputs, columns=inputs[0].index)
        
        pred_label = self.model.predict(inputs)
        pred_label = [ [int(l)] for l in pred_label]
        
        return pred_label


class DLOBPredictor:
    def __init__(self) -> None:
        with open(os.path.join(os.path.dirname(__file__), 'config.json'), 'r') as f:
            config = json.load(f)

        self.window = 100
        self.num_classes = 3
        self.extra_cols = ob_cols_derive1 + ob_cols_derive2 + ob_cols_pro

        model_path = os.path.join(os.path.dirname(__file__), 'dlob_model.pt')
        if not os.path.isabs(model_path):
            model_path = os.path.join(os.path.dirname(__file__), model_path)

        self.model = DLOB(
            task_label=config['label'],
            use_pool=False,
            extra_cols=self.extra_cols,
            window=self.window,
        )
        self.model.load_model(model_path, num_classes=self.num_classes)

    def _process_single_batch(self, df: pd.DataFrame) -> np.ndarray:
        return _process_batch_worker(df, self.extra_cols)

    def predict(self, batches: list[pd.DataFrame], n_jobs: int = 1) -> list[list[int]]:
        windows = []
        if n_jobs <= 1:
            for i, df in enumerate(batches):
                if not isinstance(df, pd.DataFrame):
                    raise ValueError(f"Batch {i} is not a DataFrame.")
                windows.append(self._process_single_batch(df))
        else:
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                future_to_idx = {
                    executor.submit(_process_batch_worker, df, self.extra_cols): i
                    for i, df in enumerate(batches)
                }
                results = {}
                for future in as_completed(future_to_idx):
                    i = future_to_idx[future]
                    results[i] = future.result()
                windows = [results[i] for i in range(len(batches))]

        X = np.stack(windows, axis=0)
        preds = self.model.predict(X)
        return [[int(p)] for p in preds]


def _process_batch_worker(df: pd.DataFrame, extra_cols: list[str]) -> np.ndarray:
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    df = df.copy()
    df['Ndate'] = pd.to_datetime(df['date'].astype(int), unit='D', origin=pd.Timestamp('2020-01-01'))
    df['timestamp'] = pd.to_datetime(
        df['Ndate'].dt.strftime('%Y-%m-%d') + ' ' + df['time'].astype(str),
        format='%Y-%m-%d %H:%M:%S'
    )

    pip = FactorPipeline(df, date_range=None)
    pip.Tick_Factor_Pool.registry = FACTORS
    factor_panel = pip.load_factor_exposure(n_jobs=1)

    raw_indexed = df.set_index(['sym', 'timestamp']).sort_index()
    raw_features = raw_indexed.loc[:, extra_cols] if extra_cols else pd.DataFrame(index=raw_indexed.index)
    combined = factor_panel.join(raw_features, how='inner')

    feature_cols = list(factor_panel.columns) + extra_cols
    features = combined[feature_cols]
    print(features)
    features = features.ffill().bfill()

    return features


def _calc_factors_worker(single_df):
    if single_df.empty or 'date' not in single_df.columns or 'time' not in single_df.columns:
        raise ValueError("输入的 DataFrame 为空或缺少必要的 'date' 或 'time' 列。")

    single_df = single_df.copy()
    single_df['Ndate'] = pd.to_datetime(single_df['date'].astype(int), unit='D', origin=pd.Timestamp('2020-01-01'))
    single_df['timestamp'] = pd.to_datetime(
        single_df['Ndate'].dt.strftime('%Y-%m-%d') + ' ' + single_df['time'].astype(str),
        format='%Y-%m-%d %H:%M:%S'
    )

    pip = FactorPipeline(single_df, date_range=None)
    pip.Tick_Factor_Pool.registry = FACTORS
    return pip.load_factor_exposure(n_jobs=1)


if __name__ == "__main__":

    
    predictor = DLOBPredictor()
    
    with open(os.path.expanduser("~/LWCUP/results/test_data/test_data.pkl"), 'rb') as f:
        test_data = pickle.load(f)

    batches = [ d.iloc[-100:, :] for d in test_data ]
    
    y = predictor.predict(batches[0:100], n_jobs=16)
    print(y)