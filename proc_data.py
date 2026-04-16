import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from config import data_path, results_path

from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle

if __name__ == "__main__":
    
    concat_list = []
    local_test_concat_list = []
    
    for data_parquet in tqdm(os.listdir(os.path.expanduser(data_path))[:5], desc="Processing parquet files", unit="file", leave=True):
        if data_parquet.endswith(".parquet"):
            df = pd.read_parquet(os.path.join(os.path.expanduser(data_path), data_parquet))
            local_test_concat_list.append(df)
            
        else:
            print(f"Skipping non-parquet file: {data_parquet}")
            
    with open(os.path.join(os.path.expanduser(results_path), "test_data/test_data.pkl"), "wb") as f:
        pickle.dump(local_test_concat_list, f)
    
    for data_parquet in tqdm(os.listdir(os.path.expanduser(data_path))[:5], desc="Processing parquet files", unit="file", leave=True):
        if data_parquet.endswith(".parquet"):
            df = pd.read_parquet(os.path.join(os.path.expanduser(data_path), data_parquet))
            concat_list.append(df)
            
        else:
            print(f"Skipping non-parquet file: {data_parquet}")
    
    # Concatenate all DataFrames
    final_df = pd.concat(concat_list, ignore_index=True)
    
    # date列是纯数字，例如0，1，2，转化为日期，便于和time合并
    final_df['Ndate'] = pd.to_datetime(final_df['date'].astype(int), unit='D', origin=pd.Timestamp('2020-01-01'))
    final_df['timestamp'] = pd.to_datetime(final_df['Ndate'].dt.strftime('%Y-%m-%d') + ' ' + final_df['time'].astype(str), format='%Y-%m-%d %H:%M:%S')
        
    final_df.to_parquet(os.path.join(os.path.expanduser(results_path), "merge_data/merge_data.parquet"), index=False)