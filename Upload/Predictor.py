from __future__ import annotations

import os
import numpy as np
import pandas as pd
import torch
import pickle

# try:
#     from model import DeepLOB
# except ImportError:
#     from .model import DeepLOB


class Predictor:
    def __init__(self) -> None:
        # pkl_path = os.path.join(os.path.dirname(__file__), 'best_model.pt')

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # try:
        #     ckpt = torch.load(pkl_path, map_location=self.device, weights_only=False)
        # except TypeError:
        #     ckpt = torch.load(pkl_path, map_location=self.device)

        # self.meta = dict(ckpt.get("meta") or {})
        # nums = self.meta["num_classes_per_head"]
        # self.model = DeepLOB(list(nums)).to(self.device)
        # self.model.load_state_dict(ckpt["model_state"], strict=True)
        # self.model.eval()
        
        with open(os.path.join(os.path.dirname(__file__), 'best_model.pkl'), 'rb') as f:
            self.model = pickle.load(f)

    def predict_raw(self, batches: list[pd.DataFrame]) -> list[list[int]]:
        """
        返回: [[y_label0, y_label1, ...], ...]，与 batches 等长；各头 argmax 类下标 int，三分类时为 [0,1,2]。
        """
        arrs = [df.to_numpy(dtype=np.float32, copy=False) for df in batches]  # each: (T, D)
        x_np = np.ascontiguousarray(np.stack(arrs, axis=0))  # (B, T, D)
        x = torch.from_numpy(x_np).unsqueeze(1).to(self.device, dtype=torch.float32)  # (B, 1, T, D)

        with torch.no_grad():
            heads = self.model(x)  # tuple of K tensors, each: (B, Ck)

        preds_per_head = [h.argmax(1).cpu().numpy() for h in heads]  # each: (B,)
        pred_matrix = np.stack(preds_per_head, axis=1)  # (B, K)
        return pred_matrix.astype(int).tolist()
    
    def predict(self, batches: list[pd.DataFrame]) -> list[list[int]]:
        """
        返回: [[y_label0, y_label1, ...], ...]，与 batches 等长；各头 argmax 类下标 int，三分类时为 [0,1,2]。
        """
        


if __name__ == "__main__":
    import pickle
    from factor_pool.pipeline import FactorPipeline
    
    cols = ["bid1", "ask1", "bsize1", "asize1"]
    
    predictor = Predictor()
    
    with open(os.path.expanduser("~/LWCUP/results/test_data/test_data.pkl"), 'rb') as f:
        test_data = pickle.load(f)
    
    test_data = [ d[cols] for d in test_data ]
    
    def _calc_factors(single_df):
        
        pip = FactorPipeline(single_df, date_range=None)
        f = pip.load_factor_exposure(n_jobs=16)
        
        return f
    
    batches = []
    y = predictor.predict(batches[0:5])
    print(y)