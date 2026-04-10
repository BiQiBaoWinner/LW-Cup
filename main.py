import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)
    
from factor_pool.pipeline import FactorPipeline
from model.lgbm import LGBM
from config import data_path, results_path, range_split

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="LWCUP Model Training")
    parser.add_argument("--task", type=str, default="label_20", help="Task label")
    parser.add_argument("--factors", nargs="+", default=["tick_OBI"], help="Factor list")
    parser.add_argument("--log_dir", type=str, default=f"{results_path}/lgbm_logs/", help="Log directory")
    parser.add_argument("--save_dir", type=str, default=f"{results_path}/lgbm_models/", help="Model save directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    Task_label = args.task
    factor_list = args.factors
    log_dir = args.log_dir
    save_dir = args.save_dir
    seed = args.seed
    
    model = LGBM(Task_label, factor_list, log_dir, save_dir, seed)
    data_pack = model.DataPreparing(skip_nulldate=True)
    best_model = model.fit_model(seed)