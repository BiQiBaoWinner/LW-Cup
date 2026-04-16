import os
import sys
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
	sys.path.append(project_root)

from config import results_path
from factor_pool.tick_factor_pool import tick_Orderbook_Imbalance_single_day


def daily_prevday_tick_OBI_mean(tick_df):
	"""示例：在因子函数内部完成聚合，返回单行(sym列)日度因子。"""
	intraday = tick_Orderbook_Imbalance_single_day(tick_df)
	if intraday is None or intraday.empty:
		return pd.DataFrame()
	return intraday.mean(axis=0).to_frame().T


class DailyFactorPool:
	def __init__(self, tick_df):
		self.data = tick_df
		self.registry = {}

	def register_factor(self, factor_name, factor_func, need_cols):
		"""注册日度因子的底层 tick 因子函数。"""
		self.registry[factor_name] = {
			'factor_func': factor_func,
			'need_cols': need_cols,
		}

	def _get_prev_date(self, target_date):
		"""根据目标日期找到前一个可用交易日。"""
		date_series = self.data['date'].astype(str)
		unique_dates = sorted(date_series.unique(), key=lambda x: int(x))

		target_date = str(target_date)
		if target_date not in unique_dates:
			raise ValueError(f"目标日期 {target_date} 不在数据中")

		target_idx = unique_dates.index(target_date)
		if target_idx == 0:
			raise ValueError(f"目标日期 {target_date} 没有前一交易日")

		return unique_dates[target_idx - 1]

	def build_daily_factor_pool(self, target_date):
		"""
		使用 target_date 的前一交易日 tick 数据构建日度因子池。
		每个注册因子函数必须在函数内部完成聚合并返回单行DataFrame/Series。

		Returns:
			DataFrame: index 为单行 timestamp，columns 为多层索引 (factor, sym)
		"""
		prev_date = self._get_prev_date(target_date)
		prev_tick = self.data[self.data['date'].astype(str) == prev_date]

		factor_frames = []
		for factor_name, factor_meta in self.registry.items():
			factor_func = factor_meta['factor_func']
			need_cols = factor_meta['need_cols']

			if not all(col in prev_tick.columns for col in need_cols):
				print(f"缺少计算 {factor_name} 因子所需的列: {need_cols}")
				continue

			daily_factor = factor_func(prev_tick)
			if isinstance(daily_factor, pd.Series):
				daily_factor = daily_factor.to_frame().T

			if daily_factor is None or daily_factor.empty:
				continue

			if not isinstance(daily_factor, pd.DataFrame):
				raise TypeError(f"因子 {factor_name} 必须返回DataFrame或Series")

			if daily_factor.shape[0] != 1:
				raise ValueError(
					f"因子 {factor_name} 必须在函数内聚合为单行，当前返回 {daily_factor.shape[0]} 行"
				)

			daily_factor.columns = pd.MultiIndex.from_product(
				[[factor_name], daily_factor.columns],
				names=['factor', 'sym']
			)
			factor_frames.append(daily_factor)

		if not factor_frames:
			return pd.DataFrame()

		daily_factor_pool = pd.concat(factor_frames, axis=1)
		# 用目标日期作为日度标签；每个目标日期只返回一行时间戳
		daily_factor_pool.index = pd.to_datetime([int(target_date)], unit='D', origin=pd.Timestamp('2020-01-01'))
		daily_factor_pool.index.name = 'timestamp'
        
		return daily_factor_pool


if __name__ == "__main__":
	tot_tick_df = pd.read_parquet(f"{results_path}/merge_data/merge_data.parquet")

	# 示例：计算目标日期 1 的日度因子，使用 date=0 的 tick 数据
	target_date = '1'

	daily_factor_pool_builder = DailyFactorPool(tot_tick_df)
	daily_factor_pool_builder.register_factor(
		factor_name='tick_OBI',
		factor_func=daily_prevday_tick_OBI_mean,
		need_cols=['bid1', 'ask1', 'bsize1', 'asize1']
	)

	daily_factor_pool = daily_factor_pool_builder.build_daily_factor_pool(target_date=target_date)
	print(daily_factor_pool)
