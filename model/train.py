import os 
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
    
from model.lgbm import LGBM
from config import results_path, FACTORS
import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LGBM model for LWCUP")
    parser.add_argument("--task_label", type=str, default='label_10', help="The target label for the classification task (e.g., 'label_5', 'label_10', etc.)")
    parser.add_argument("--sft", action='store_true', help='Disable SFT (default is False)')
    args = parser.parse_args()

    Task_label = args.task_label
    
    results_path = os.path.expanduser(results_path)
    with open(os.path.join(results_path, f'factor_{Task_label.replace("_", "")}.json'), 'r') as f:
        factor_name = list(json.load(f).keys())
    if factor_name == None:
        factor_registry = FACTORS
    else:
        factor_registry = {f: FACTORS.get(f, None) for f in factor_name}
    log_dir = os.path.expanduser(f"{results_path}/lgbm_logs")
    save_dir = os.path.expanduser(f"{results_path}/lgbm_models")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    seed = 42
    
    model = LGBM(Task_label, factor_registry, log_dir, save_dir, seed, args.sft)
    data_pack = model.DataPreparing(skip_nulldate=True)
    best_model = model.fit_model(seed)