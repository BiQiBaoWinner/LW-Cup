import os
import sys
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from factor_pool.daily_factor_pool import DailyFactorPool, daily_prevday_tick_OBI_mean
from factor_pool.tick_factor_pool import TickFactorPool, tick_Orderbook_Imbalance_single_day

def _calc_single_day_tick_panel(day_tick, registry):
    """
    在子进程中执行，仅接收当天的经过切分后的 tick 数据和因子注册表。
    """
    if day_tick is None or day_tick.empty:
        return pd.DataFrame()
    
    # 构造临时的 builder，由于 registry 已经传入，子进程不需要访问完整的 FactorPipeline 对象
    builder = TickFactorPool(day_tick)
    builder.registry = registry
    return builder.build_factor_pool()

class FactorPipeline:
    """
    因子计算流水线，支持面向两个因子库：
    1. DailyFactorPool (日度因子：基于前一日 tick 数据聚合)
    2. TickFactorPool (高频因子：保持 tick 原始粒度)
    """
    def __init__(self, tick_df, date_range):
        if isinstance(tick_df, pd.DataFrame):
            print("输入为 DataFrame，直接使用")
            self.tick_df = tick_df
        elif isinstance(tick_df, list):
            print(f"输入为 DataFrame 列表，长度: {len(tick_df)}，正在合并...")
            self.tick_df = pd.concat(tick_df, axis=0)
        else:
            raise ValueError("tick_df 应为 DataFrame 或 DataFrame 列表")
        
        if date_range is None:
            date_range = self.tick_df['date'].astype(str).unique().tolist()
        self.date_range = date_range

        self.Daily_Factor_Pool = DailyFactorPool(self.tick_df)
        self.Tick_Factor_Pool = TickFactorPool(self.tick_df)

        self._setup_default_factors()
        
        # 将 keys 转换为 list 以便能够被 pickle (解决 ProcessPoolExecutor 报错)
        self.d_factors = list(self.Daily_Factor_Pool.registry.keys())
        self.t_factors = list(self.Tick_Factor_Pool.registry.keys())

    def _setup_default_factors(self):
        """
        预定义可选的因子及其对应的计算函数和依赖列。
        """
        self.Daily_Factor_Pool.register_factor(
            factor_name='OBI_mean',
            factor_func=daily_prevday_tick_OBI_mean,
            need_cols=['bid1', 'ask1', 'bsize1', 'asize1']
        )
        
        # 注册 Tick 因子
        self.Tick_Factor_Pool.register_factor(
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
        builder.registry = self.Tick_Factor_Pool.registry
        
        return builder.build_factor_pool()

    def get_daily_panel(self, target_dates):
        """
        获取日度所有已注册因子的 Panel 数据。
        """
        all_dates_data = []
        for date in target_dates:
            try:
                daily_pool = self.Daily_Factor_Pool.build_daily_factor_pool(target_date=date)
                if not daily_pool.empty:
                    all_dates_data.append(daily_pool)
            except Exception as e:
                print(f"跳过日度日期 {date}: {e}")

        if not all_dates_data:
            return pd.DataFrame()

        total = pd.concat(all_dates_data, axis=0)
        panel = total.stack(level='sym').swaplevel('timestamp', 'sym').sort_index()
        return panel
    
    def load_factor_exposure(self, n_jobs=1, ret_type='df'):
        """
        根据初始化时传入的 date_range 进行并行化计算高频 Tick 因子并合并。
        返回 (sym, timestamp) 为 MultiIndex 的 Panel 数据。
        """
        if not hasattr(self, 'date_range') or self.date_range is None:
            raise ValueError("Pipeline 初始化时未传入 date_range")

        print(f"开始计算 Tick 因子, 进程数: {n_jobs}")
        
        all_panels = []
        
        if n_jobs <= 1:
            # 串行计算并添加进度条
            for date in tqdm(self.date_range, desc="串行计算因子"):
                panel = self.get_tick_panel(date)
                if not panel.empty:
                    all_panels.append(panel)
        else:
            # 并行计算方案A：在主进程预先切分数据，避免子进程重复拷贝巨大的 self.tick_df
            print("===预切分数据===")
            tasks = []
            for date in self.date_range:
                day_data = self.tick_df[self.tick_df['date'].astype(str) == str(date)]
                if not day_data.empty:
                    # 只保存必要的元组：(日期, 该日数据切片)
                    tasks.append((date, day_data))

            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                # 提交任务时，调用全局函数并只传入 registry 和该日切片数据
                # 这样 Pickle 序列化时只处理当前日期的数据，开销大幅减小
                future_to_date = {
                    executor.submit(_calc_single_day_tick_panel, day_data, self.Tick_Factor_Pool.registry): date 
                    for date, day_data in tasks
                }
                # 使用 tqdm 包装 as_completed
                for future in tqdm(as_completed(future_to_date), total=len(future_to_date), desc="并行计算因子"):
                    date = future_to_date[future]
                    try:
                        panel = future.result()
                        # print(panel)
                        if not panel.empty:
                            all_panels.append(panel)
                    except Exception as e:
                        print(f"日期 {date} 计算出错: {e}")

        if not all_panels:
            return pd.DataFrame()

        if ret_type == 'df':
            final_panel = pd.concat(all_panels, axis=0).sort_index()
            return final_panel
        elif ret_type == 'list':
            return all_panels

if __name__ == "__main__":
    from config import results_path
    
    # 示例用法
    tot_tick_df = pd.read_parquet(f"{results_path}/merge_data/merge_data.parquet")
    pipeline = FactorPipeline(tot_tick_df, date_range=['0', '1', '2'])
    
    # 获取特定的 Panel 数据
    daily_panel = pipeline.load_factor_exposure(n_jobs=4)
    print(daily_panel)
