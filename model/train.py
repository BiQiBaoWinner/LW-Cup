import os 
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
    
from model.lgbm import LGBM
from config import results_path, FACTORS

if __name__ == "__main__":

    Task_label = 'label_20'
    factor_registry = FACTORS
    log_dir = os.path.expanduser(f"{results_path}/lgbm_logs")
    save_dir = os.path.expanduser(f"{results_path}/lgbm_models")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    seed = 42
    
    model = LGBM(Task_label, factor_registry, log_dir, save_dir, seed)
    data_pack = model.DataPreparing(skip_nulldate=True)
    best_model = model.fit_model(seed)