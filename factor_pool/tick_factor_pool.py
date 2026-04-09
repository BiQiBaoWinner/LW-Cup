import pandas as pd
import numpy as np
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from config import data_path, results_path
from factor_pool.utils import tick_long_to_wide_, safe_divide

# tick因子计算函数
def tick_Orderbook_Imbalance_single_day(tick_df):
    """订单簿不平衡度因子

    Args:
        tick_df: 包含订单簿数据的DataFrame，必须包含以下列：
            - 'bid1': 买一价
            - 'bsize1': 买一量
            - 'ask1': 卖一价
            - 'asize1': 卖一量
    """
    
    wide_bid1 = tick_long_to_wide_(tick_df, 'bid1')
    wide_bsize1 = tick_long_to_wide_(tick_df, 'bsize1')
    wide_ask1 = tick_long_to_wide_(tick_df, 'ask1')
    wide_asize1 = tick_long_to_wide_(tick_df, 'asize1')
    
    # 计算订单簿不平衡度
    imbalance = safe_divide(wide_bsize1 * wide_ask1 - wide_asize1 * wide_bid1, wide_bsize1 * wide_ask1 + wide_asize1 * wide_bid1)
    
    imbalance.index.name = 'timestamp'
    imbalance.columns.name = 'sym'
    
    return imbalance

class TickFactorPool:
    def __init__(self, tick_df):
        self.data = tick_df
        self.registry = {}
    
    # 因子登记簿
    def register_factor(self, factor_name, factor_func, need_cols):
        """注册因子计算函数

        Args:
            factor_name: 因子名称
            factor_func: 因子计算函数，接受一个DataFrame作为输入，返回一个DataFrame作为输出
            need_cols: 计算因子所需的列名列表
            
        示例：
            --- IGNORE ---
            self.registry['tick_OBI'] = (tick_Orderbook_Imbalance_single_day, ['bid1', 'ask1', 'bsize1', 'asize1'])
            --- IGNORE ---
        之后在 build_factor_pool 方法中会自动调用注册的因子计算函数来构建因子池
        
        """
        self.registry[factor_name] = {
            'factor_func': factor_func,
            'need_cols': need_cols,
        }
    
    # 基于登记簿计算因子池
    def build_factor_pool(self):
        """
        构建因子池，返回一个DataFrame，索引为 'sym' 和 'timestamp'，列为不同的因子名称
        """
        
        factor_frames = []
        for factor_name, factor_meta in self.registry.items():
            factor_func = factor_meta['factor_func']
            need_cols = factor_meta['need_cols']

            if not all(col in self.data.columns for col in need_cols):
                print(f"缺少计算 {factor_name} 因子所需的列: {need_cols}")
                continue

            factor_df = factor_func(self.data)
            if factor_df is None or factor_df.empty:
                continue

            factor_df.columns = pd.MultiIndex.from_product(
                [[factor_name], factor_df.columns],
                names=['factor', 'sym']
            )
            factor_frames.append(factor_df)

        if not factor_frames:
            return pd.DataFrame()

        factor_pool = pd.concat(factor_frames, axis=1)
        factor_pool.index.name = 'timestamp'
        factor_pool = factor_pool.stack(level='sym').swaplevel('timestamp', 'sym').sort_index()
        
        return factor_pool

if __name__ == "__main__":
    
    tot_tick_df = pd.read_parquet(f"{results_path}/merge_data/merge_data.parquet")
    
    date = '0'
    
    tot_tick_df = tot_tick_df[tot_tick_df['date'] == date]
    
    tick_factor_pool = TickFactorPool(tot_tick_df)
    
    tick_factor_pool.register_factor(
        factor_name = 'tick_OBI', 
        factor_func = tick_Orderbook_Imbalance_single_day, 
        need_cols = ['bid1', 'ask1', 'bsize1', 'asize1']
    )
    
    factor_pool = tick_factor_pool.build_factor_pool()
    
    print(factor_pool)
    
    