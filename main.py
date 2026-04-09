import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)
    
from factor_pool.pipeline import FactorPipeline
from config import data_path, results_path, range_split

