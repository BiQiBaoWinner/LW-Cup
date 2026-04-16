import pandas as pd
import numpy as np
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

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

def tick_Orderbook_Imbalance_single_day_v2(tick_df):
    """订单簿不平衡度因子，版本2：直接在长格式数据上计算，不进行宽化

    Args:
        tick_df: 包含订单簿数据的DataFrame，必须包含以下列：
            - 'bid1': 买一价
            - 'bsize1': 买一量
            - 'ask1': 卖一价
            - 'asize1': 卖一量
            - 'bid2': 买二价
            - 'bsize2': 买二量
            - 'ask2': 卖二价
            - 'asize2': 卖二量
    """
    
    wide_bid1 = tick_long_to_wide_(tick_df, 'bid1')
    wide_bid2 = tick_long_to_wide_(tick_df, 'bid2')
    wide_bsize1 = tick_long_to_wide_(tick_df, 'bsize1')
    wide_bsize2 = tick_long_to_wide_(tick_df, 'bsize2')
    wide_ask1 = tick_long_to_wide_(tick_df, 'ask1')
    wide_ask2 = tick_long_to_wide_(tick_df, 'ask2')
    wide_asize1 = tick_long_to_wide_(tick_df, 'asize1')
    wide_asize2 = tick_long_to_wide_(tick_df, 'asize2')
    
    # 计算订单簿不平衡度
    imbalance = safe_divide(wide_bsize1 * wide_ask1 + wide_bsize2 * wide_ask2 - wide_asize1 * wide_bid1 - wide_asize2 * wide_bid2, wide_bsize1 * wide_ask1 + wide_bsize2 * wide_ask2 + wide_asize1 * wide_bid1 + wide_asize2 * wide_bid2)
    
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
        factor_pool = factor_pool.stack(level='sym', future_stack=True).swaplevel('timestamp', 'sym').sort_index()
        
        return factor_pool

if __name__ == "__main__":
    from config import results_path
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
    
    