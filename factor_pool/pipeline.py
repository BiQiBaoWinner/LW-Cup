import os
import sys
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from factor_pool.daily_factor_pool import DailyFactorPool, daily_prevday_tick_OBI_mean
from factor_pool.tick_factor_pool import TickFactorPool, tick_Orderbook_Imbalance_single_day

class FactorPipeline:
    """
    因子计算流水线，支持面向两个因子库：
    1. DailyFactorPool (日度因子：基于前一日 tick 数据聚合)
    2. TickFactorPool (高频因子：保持 tick 原始粒度)
    """
    def __init__(self, tick_df, date_range):
        self.tick_df = tick_df
        self.date_range = date_range
        self.daily_builder = DailyFactorPool(tick_df)
        self.tick_builder = TickFactorPool(tick_df)
        
        self._setup_default_factors()

    def _setup_default_factors(self):
        """
        预定义可选的因子及其对应的计算函数和依赖列。
        """
        self.daily_builder.register_factor(
            factor_name='OBI_mean',
            factor_func=daily_prevday_tick_OBI_mean,
            need_cols=['bid1', 'ask1', 'bsize1', 'asize1']
        )
        
        # 注册 Tick 因子
        self.tick_builder.register_factor(
            factor_name='tick_OBI',
            factor_func=tick_Orderbook_Imbalance_single_day,
            need_cols=['bid1', 'ask1', 'bsize1', 'asize1']
        )

    def get_tick_panel(self, target_date):
        """
        获取单日的所有已注册的高频 Tick Panel 数据。
        返回 (sym, timestamp) 为 MultiIndex 的 DataFrame。
        """
        day_tick = self.tick_df[self.tick_df['date'].astype(str) == str(target_date)]
        if day_tick.empty:
            return pd.DataFrame()

        builder = TickFactorPool(day_tick)
        builder.registry = self.tick_builder.registry
        
        return builder.build_factor_pool()

    def get_daily_panel(self, target_dates):
        """
        获取日度所有已注册因子的 Panel 数据。
        """
        all_dates_data = []
        for date in target_dates:
            try:
                daily_pool = self.daily_builder.build_daily_factor_pool(target_date=date)
                if not daily_pool.empty:
                    all_dates_data.append(daily_pool)
            except Exception as e:
                print(f"跳过日度日期 {date}: {e}")

        if not all_dates_data:
            return pd.DataFrame()

        total = pd.concat(all_dates_data, axis=0)
        panel = total.stack(level='sym').swaplevel('timestamp', 'sym').sort_index()
        return panel
    
    def load_factor_exposure(self, n_jobs=1):
        """
        根据初始化时传入的 date_range 进行并行化计算高频 Tick 因子并合并。
        返回 (sym, timestamp) 为 MultiIndex 的 Panel 数据。
        """
        if not hasattr(self, 'date_range') or self.date_range is None:
            raise ValueError("Pipeline 初始化时未传入 date_range")

        print(f"开始并行计算高频 Tick 因子，日期范围: {self.date_range}, 进程数: {n_jobs}")
        
        all_panels = []
        
        if n_jobs <= 1:
            # 串行计算
            for date in self.date_range:
                panel = self.get_tick_panel(date)
                if not panel.empty:
                    all_panels.append(panel)
        else:
            # 并行计算
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                # 提交任务，每个任务计算单日的高频 Panel
                future_to_date = {executor.submit(self.get_tick_panel, date): date for date in self.date_range}
                
                for future in as_completed(future_to_date):
                    date = future_to_date[future]
                    try:
                        panel = future.result()
                        if not panel.empty:
                            all_panels.append(panel)
                    except Exception as e:
                        print(f"日期 {date} 计算出错: {e}")

        if not all_panels:
            return pd.DataFrame()

        # 合并所有日期的高频 Panel 数据
        # 因为 get_tick_panel 返回的已经是 (sym, timestamp) 索引，直接 concat 即可
        final_panel = pd.concat(all_panels, axis=0).sort_index()
        return final_panel

if __name__ == "__main__":
    from config import results_path
    
    # 示例用法
    tot_tick_df = pd.read_parquet(f"{results_path}/merge_data/merge_data.parquet")
    pipeline = FactorPipeline(tot_tick_df, date_range=['0', '1', '2'])
    
    # 获取特定的 Panel 数据
    daily_panel = pipeline.load_factor_exposure(n_jobs=4)
    print(daily_panel)
