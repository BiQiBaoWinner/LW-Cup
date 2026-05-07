import os 
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from config import results_path

import pickle
import pandas as pd
import json

if __name__ == "__main__":
    
    results_path = os.path.expanduser(results_path)
    
    for task in [5, 10, 20, 40, 60]:

        with open(f'/home1/zhuzhoufan/LWCUP/mmpc/lgbm_label{task}.pkl', 'rb') as f:
            model = pickle.load(f)
        # 获取树模型的IV值
        iv_values = model.feature_importances_
        # 获取特征名称
        feature_names = model.feature_name_
        # 将特征名称和IV值组合成一个DataFrame
        iv_df = pd.DataFrame({'Feature': feature_names, 'IV': iv_values})
        # 按照IV值降序排序
        iv_df = iv_df.sort_values(by='IV', ascending=False).reset_index(drop=True)
        iv_df['cumulative_percent'] = iv_df['IV'].cumsum() / iv_df['IV'].sum()  # 计算累计IV占比

        f_iv_dict = dict(zip(iv_df[iv_df['cumulative_percent'] <= 0.9]['Feature'], iv_df[iv_df['cumulative_percent'] <= 0.9]['IV']))

        with open(os.path.join(results_path, f'factor_label{task}.json'), 'w') as f:
            # print(os.path.join(results_path, f'factor_label{task}.json'))
            json.dump(f_iv_dict, f)