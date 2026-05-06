import os
import sys
import pandas as pd
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from config import results_path
from factor_pool.tick_factor_pool import (
    tick_Orderbook_Imbalance_single_day,
    tick_LiquidityShortfall_single_day,
    tick_Amount_Orderbook_Imbalance_single_day,
    tick_Volume_Amount_Orderbook_Imbalance_single_day,
    tick_LogVolume_Amount_Orderbook_Imbalance_single_day,
    tick_VWOBI_ma5_single_day,
    tick_FOBI_ma5_single_day,
    tick_TOBI_ma5_single_day,
    tick_MPC_single_day,
    tick_MidTrend_single_day,
    tick_VolMaRatio_single_day,
    tick_SpreadSlope5_single_day,
    tick_VolTrend_single_day,
    tick_VolPriceTrend_single_day,
    tick_DepthStrength_single_day,
    tick_OrderBookSkew_single_day,
    tick_Concentration_single_day,
    tick_SpreadIntensity_single_day,
    tick_BookEntropy_single_day,
    tick_ImbalanceStrength_single_day,
    tick_BidConcentration_single_day,
    tick_AskConcentration_single_day,
    tick_SpreadSlope_single_day,
    tick_BaRatio_single_day,
    tick_BidSpread_single_day,
    tick_AskSpread_single_day,
    tick_OrderBookVolatility_single_day,
    tick_VolPriceFeatures_single_day,
    tick_BidEntropy_single_day,
    tick_AskEntropy_single_day,
    tick_BidConvex_single_day,
    tick_AskConvex_single_day,
    tick_RollImbStats_single_day,
    tick_MidAccel_single_day,
    tick_BuyPressure_single_day,
    tick_SellPressure_single_day,
    tick_PressureRatio_single_day,
    tick_VolImbInteraction_single_day,
    tick_BidAskSlopeDiff_single_day,
    tick_VolAdjImpact_single_day,
    tick_MidAccelVol_single_day,
    tick_BidTop3Concentration_single_day,
    tick_AskTop3Concentration_single_day,
    tick_BookThickness_single_day,
    tick_PriceVolatility_single_day,
    tick_VolVolatility_single_day,
    tick_MomentumReversal_single_day,
    tick_LiquidIndex_single_day,
    tick_DepthDeviation_single_day,
    tick_VolAcceleration_single_day,
    tick_PriceAcceleration_single_day,
    tick_SpreadDispersion_single_day,
    tick_VolPriceRes_single_day,
    tick_RelBookStrength_single_day,
    tick_TurnTrendStrength_single_day,
    tick_PriceTrendStrength_single_day,
    tick_VolRatio_single_day,
    tick_BookPriceDeviation_single_day,
    tick_BidVolatility_single_day,
    tick_AskVolatility_single_day,
    tick_CumsumVolStats_single_day,
)


# ---------------------------------------------------------------------------
# 聚合辅助函数
# ---------------------------------------------------------------------------

def _aggregate_intraday(intraday_df, agg_method):
    """将日内宽表（index=timestamp, columns=sym）聚合为单行 Series。

    Args:
        intraday_df: 日内因子 DataFrame，index 为 timestamp，columns 为 sym
        agg_method: 聚合方式，支持:
            - 字符串: 'mean', 'std', 'last', 'sum', 'median', 'max', 'min', 'skew', 'kurt'
            - 可调用对象: 接受 DataFrame 返回 Series

    Returns:
        Series: index 为 sym 的单行聚合结果
    """
    if callable(agg_method):
        result = agg_method(intraday_df)
        if isinstance(result, pd.DataFrame):
            result = result.squeeze()
        return result

    agg_map = {
        'mean':   lambda df: df.mean(axis=0),
        'std':    lambda df: df.std(axis=0),
        'last':   lambda df: df.iloc[-1] if len(df) > 0 else pd.Series(dtype=float),
        'sum':    lambda df: df.sum(axis=0),
        'median': lambda df: df.median(axis=0),
        'max':    lambda df: df.max(axis=0),
        'min':    lambda df: df.min(axis=0),
        'skew':   lambda df: df.skew(axis=0),
        'kurt':   lambda df: df.kurt(axis=0),
    }

    if agg_method not in agg_map:
        raise ValueError(f"未知聚合方式: {agg_method}，支持: {list(agg_map.keys())}")

    return agg_map[agg_method](intraday_df)


# ---------------------------------------------------------------------------
# DailyFactorPool
# ---------------------------------------------------------------------------

class DailyFactorPool:
    """日度因子池。

    使用前一交易日的 tick 数据计算日内因子，再聚合为日度因子。

    用法示例:
        >>> pool = DailyFactorPool(tot_tick_df)
        >>> # 手动注册额外因子
        >>> pool.register_factor('my_factor', some_tick_func, ['col1', 'col2'], agg_method='std')
        >>> daily_df = pool.build_daily_factor_pool(target_date='1')
    """

    def __init__(self, tick_df):
        """
        Args:
            tick_df: 包含多日 tick 数据的 DataFrame，
                     必须包含 'date', 'sym', 'time' 列以及各因子所需字段。
        """
        self.data = tick_df
        self.registry = {}
        self._register_default_factors()

    # ------------------------------------------------------------------
    # 注册因子
    # ------------------------------------------------------------------

    def register_factor(self, daily_factor_name, tick_factor_func, need_cols, agg_method='mean'):
        """注册一个日度因子。

        Args:
            daily_factor_name: 日度因子名称（出现在输出列中）
            tick_factor_func:   日内因子计算函数，接受 tick_df 返回宽表 DataFrame
                                （index=timestamp, columns=sym）
            need_cols:          计算所需的列名列表
            agg_method:         日内到日度的聚合方式，默认 'mean'
        """
        self.registry[daily_factor_name] = {
            'tick_factor_func': tick_factor_func,
            'need_cols': need_cols,
            'agg_method': agg_method,
        }

    # ------------------------------------------------------------------
    # 日期工具
    # ------------------------------------------------------------------

    def _get_prev_date(self, target_date):
        """根据目标日期找到前一个可用交易日。

        Args:
            target_date: 目标日期（字符串或整数）

        Returns:
            str: 前一交易日

        Raises:
            ValueError: 目标日期不在数据中或没有前一交易日
        """
        date_series = self.data['date'].astype(str)
        unique_dates = sorted(date_series.unique(), key=lambda x: int(x))

        target_date = str(target_date)
        if target_date not in unique_dates:
            raise ValueError(f"目标日期 {target_date} 不在数据中")

        target_idx = unique_dates.index(target_date)
        if target_idx == 0:
            raise ValueError(f"目标日期 {target_date} 没有前一交易日")

        return unique_dates[target_idx - 1]

    # ------------------------------------------------------------------
    # 构建日度因子池
    # ------------------------------------------------------------------

    def build_daily_factor_pool(self, target_date):
        """使用 target_date 的前一交易日 tick 数据构建日度因子池。

        Args:
            target_date: 目标日期（字符串或整数）

        Returns:
            DataFrame: index 为单行 timestamp（目标日期），
                       columns 为 MultiIndex (factor, sym)
        """
        prev_date = self._get_prev_date(target_date)
        prev_tick = self.data[self.data['date'].astype(str) == prev_date]

        factor_frames = []
        for daily_name, meta in self.registry.items():
            tick_func = meta['tick_factor_func']
            need_cols = meta['need_cols']
            agg_method = meta['agg_method']

            # 检查所需列
            if not all(col in prev_tick.columns for col in need_cols):
                print(f"缺少计算 {daily_name} 因子所需的列: {need_cols}")
                continue

            # 计算日内因子
            intraday = tick_func(prev_tick)
            if intraday is None:
                continue

            # 处理返回 tuple 的情况（如 tick_RollImbStats_single_day）
            if isinstance(intraday, tuple):
                print(f"警告: {daily_name} 的 tick 函数返回 tuple，"
                      f"请使用 lambda 提取单个组件后注册")
                continue

            if isinstance(intraday, pd.Series):
                intraday = intraday.to_frame()

            if intraday.empty:
                continue

            # 过滤有效 sym 列
            valid_syms = set(prev_tick['sym'].unique())
            valid_cols = [c for c in intraday.columns if c in valid_syms]
            if not valid_cols:
                print(f"警告: 因子 {daily_name} 没有有效的 sym 列，跳过")
                continue
            intraday = intraday[valid_cols]

            # 聚合为日度单行
            daily_series = _aggregate_intraday(intraday, agg_method)
            if daily_series.empty:
                continue

            daily_df = daily_series.to_frame().T
            daily_df.columns = pd.MultiIndex.from_product(
                [[daily_name], daily_df.columns],
                names=['factor', 'sym']
            )
            factor_frames.append(daily_df)

        if not factor_frames:
            return pd.DataFrame()

        daily_pool = pd.concat(factor_frames, axis=1)
        # 用目标日期作为日度标签
        daily_pool.index = pd.to_datetime(
            [int(target_date)], unit='D', origin=pd.Timestamp('2020-01-01')
        )
        daily_pool.index.name = 'timestamp'
        return daily_pool

    # ------------------------------------------------------------------
    # 默认因子注册
    # ------------------------------------------------------------------

    def _register_default_factors(self):
        """自动注册所有内置 tick 因子的日度版本（默认 mean 聚合）。"""

        # ---- 辅助：批量注册同一 tick 函数的多种聚合 ----
        def _reg_multi(base_name, tick_func, need_cols, agg_list):
            for agg in agg_list:
                self.register_factor(f'{base_name}_{agg}', tick_func, need_cols, agg_method=agg)

        # ================================================================
        # 第一批：原有及第一批整合因子
        # ================================================================
        _reg_multi('tick_OBI', tick_Orderbook_Imbalance_single_day,
                   ['bsize1', 'asize1'], ['mean', 'std'])
        _reg_multi('tick_LiquidityShortfall', tick_LiquidityShortfall_single_day,
                   ['volume_delta', 'midprice'], ['mean', 'std'])
        _reg_multi('tick_Amount_OBI', tick_Amount_Orderbook_Imbalance_single_day,
                   ['bid1', 'bsize1', 'ask1', 'asize1'], ['mean', 'std'])
        _reg_multi('tick_Vol_Amount_OBI', tick_Volume_Amount_Orderbook_Imbalance_single_day,
                   ['volume_delta', 'bid1', 'bsize1', 'ask1', 'asize1'], ['mean', 'std'])
        _reg_multi('tick_LogVol_Amount_OBI', tick_LogVolume_Amount_Orderbook_Imbalance_single_day,
                   ['volume_delta', 'bid1', 'bsize1', 'ask1', 'asize1'], ['mean', 'std'])

        b_cols = [f'bsize{i}' for i in range(1, 11)]
        a_cols = [f'asize{i}' for i in range(1, 11)]
        _reg_multi('tick_VWOBI_ma5', tick_VWOBI_ma5_single_day,
                   b_cols + a_cols, ['mean', 'std'])
        _reg_multi('tick_FOBI_ma5', tick_FOBI_ma5_single_day,
                   ['totalbsize', 'totalasize'], ['mean', 'std'])
        _reg_multi('tick_TOBI_ma5', tick_TOBI_ma5_single_day,
                   ['bsize1', 'asize1'], ['mean', 'std'])
        _reg_multi('tick_MPC_5', tick_MPC_single_day,
                   ['midprice1'], ['mean', 'std'])
        _reg_multi('tick_MidTrend_5', tick_MidTrend_single_day,
                   ['midprice1'], ['mean', 'std'])
        _reg_multi('tick_VolMaRatio_10', tick_VolMaRatio_single_day,
                   ['volume_delta'], ['mean', 'std'])
        _reg_multi('tick_SpreadSlope5', tick_SpreadSlope5_single_day,
                   ['spread5', 'spread1', 'midprice1'], ['mean', 'std'])
        _reg_multi('tick_VolTrend_5', tick_VolTrend_single_day,
                   ['volume_delta'], ['mean', 'std'])
        _reg_multi('tick_VolPriceTrend_5', tick_VolPriceTrend_single_day,
                   ['midprice1', 'volume_delta'], ['mean', 'std'])
        _reg_multi('tick_DepthStrength', tick_DepthStrength_single_day,
                   ['totalbsize', 'totalasize', 'midprice1'], ['mean', 'std'])
        _reg_multi('tick_OrderBookSkew', tick_OrderBookSkew_single_day,
                   b_cols + a_cols, ['mean', 'std'])
        _reg_multi('tick_ConcentrationDiff', tick_Concentration_single_day,
                   ['bsize1', 'bsize2', 'asize1', 'asize2', 'totalbsize', 'totalasize'],
                   ['mean', 'std'])
        _reg_multi('tick_SpreadIntensity', tick_SpreadIntensity_single_day,
                   ['spread1', 'midprice1'], ['mean', 'std'])
        _reg_multi('tick_BookEntropyDiff', tick_BookEntropy_single_day,
                   b_cols + a_cols, ['mean', 'std'])

        # ================================================================
        # 第二批：补全因子
        # ================================================================
        _reg_multi('tick_ImbalanceStrength', tick_ImbalanceStrength_single_day,
                   ['totalbsize', 'totalasize'], ['mean', 'std'])
        _reg_multi('tick_BidConcentration', tick_BidConcentration_single_day,
                   ['bsize1', 'bsize2', 'totalbsize'], ['mean', 'std'])
        _reg_multi('tick_AskConcentration', tick_AskConcentration_single_day,
                   ['asize1', 'asize2', 'totalasize'], ['mean', 'std'])
        _reg_multi('tick_SpreadSlope', tick_SpreadSlope_single_day,
                   ['spread5', 'spread1'], ['mean', 'std'])
        _reg_multi('tick_BaRatio', tick_BaRatio_single_day,
                   ['totalbsize', 'totalasize'], ['mean', 'std'])
        _reg_multi('tick_BidSpread', tick_BidSpread_single_day,
                   b_cols, ['mean', 'std'])
        _reg_multi('tick_AskSpread', tick_AskSpread_single_day,
                   a_cols, ['mean', 'std'])
        _reg_multi('tick_OrderBookVolatility', tick_OrderBookVolatility_single_day,
                   ['bid1', 'ask1'], ['mean', 'std'])

        # tick_VolPriceFeatures_single_day 返回多组件，分别注册
        vp_need = ['close', 'volume_delta']
        for comp in ['vol_price_sign', 'mom_vol', 'impact_cost', 'vol_price_corr', 'liquid_slope']:
            self.register_factor(
                f'tick_{comp}_mean',
                lambda df, c=comp: tick_VolPriceFeatures_single_day(df)[c],
                vp_need, agg_method='mean'
            )
            self.register_factor(
                f'tick_{comp}_std',
                lambda df, c=comp: tick_VolPriceFeatures_single_day(df)[c],
                vp_need, agg_method='std'
            )

        _reg_multi('tick_BidEntropy', tick_BidEntropy_single_day,
                   b_cols, ['mean', 'std'])
        _reg_multi('tick_AskEntropy', tick_AskEntropy_single_day,
                   a_cols, ['mean', 'std'])
        _reg_multi('tick_BidConvex', tick_BidConvex_single_day,
                   ['bsize1', 'totalbsize'] + [f'bsize{i}' for i in range(6, 11)],
                   ['mean', 'std'])
        _reg_multi('tick_AskConvex', tick_AskConvex_single_day,
                   ['asize1', 'totalasize'] + [f'asize{i}' for i in range(6, 11)],
                   ['mean', 'std'])

        # tick_RollImbStats_single_day 返回 (mean, std) 元组
        ri_need = ['totalbsize', 'totalasize']
        self.register_factor('tick_RollImbMean5_mean',
                             lambda df: tick_RollImbStats_single_day(df)[0],
                             ri_need, agg_method='mean')
        self.register_factor('tick_RollImbStd5_mean',
                             lambda df: tick_RollImbStats_single_day(df)[1],
                             ri_need, agg_method='mean')

        _reg_multi('tick_MidAccel', tick_MidAccel_single_day,
                   ['midprice1'], ['mean', 'std'])
        _reg_multi('tick_BuyPressure', tick_BuyPressure_single_day,
                   ['bsize1', 'spread1'], ['mean', 'std'])
        _reg_multi('tick_SellPressure', tick_SellPressure_single_day,
                   ['asize1', 'spread1'], ['mean', 'std'])
        _reg_multi('tick_PressureRatio', tick_PressureRatio_single_day,
                   ['bsize1', 'asize1', 'spread1'], ['mean', 'std'])
        _reg_multi('tick_VolImbInteraction', tick_VolImbInteraction_single_day,
                   ['volume_delta', 'totalbsize', 'totalasize'], ['mean', 'std'])
        _reg_multi('tick_BidAskSlopeDiff', tick_BidAskSlopeDiff_single_day,
                   ['bsize1', 'bsize5', 'asize1', 'asize5'], ['mean', 'std'])
        _reg_multi('tick_VolAdjImpact', tick_VolAdjImpact_single_day,
                   ['close', 'volume_delta'], ['mean', 'std'])
        _reg_multi('tick_MidAccelVol', tick_MidAccelVol_single_day,
                   ['midprice1', 'volume_delta'], ['mean', 'std'])
        _reg_multi('tick_BidTop3Ratio', tick_BidTop3Concentration_single_day,
                   ['bsize1', 'bsize2', 'bsize3', 'totalbsize'], ['mean', 'std'])
        _reg_multi('tick_AskTop3Ratio', tick_AskTop3Concentration_single_day,
                   ['asize1', 'asize2', 'asize3', 'totalasize'], ['mean', 'std'])
        _reg_multi('tick_BookThickness', tick_BookThickness_single_day,
                   ['totalbsize', 'totalasize', 'midprice1'], ['mean', 'std'])
        _reg_multi('tick_PriceVol5', tick_PriceVolatility_single_day,
                   ['close'], ['mean', 'std'])
        _reg_multi('tick_VolVol5', tick_VolVolatility_single_day,
                   ['volume_delta'], ['mean', 'std'])

        # 简单移动平均因子
        self.register_factor('tick_ShortMom3_mean',
                             lambda df: tick_long_to_wide_(df, 'close').rolling(3, min_periods=1).mean(),
                             ['close'], agg_method='mean')
        self.register_factor('tick_MidMom10_mean',
                             lambda df: tick_long_to_wide_(df, 'close').rolling(10, min_periods=1).mean(),
                             ['close'], agg_method='mean')

        _reg_multi('tick_MomReversal', tick_MomentumReversal_single_day,
                   ['close', 'volume_delta'], ['mean', 'std'])
        _reg_multi('tick_LiquidIndex', tick_LiquidIndex_single_day,
                   ['totalbsize', 'totalasize', 'midprice1', 'close'], ['mean', 'std'])
        _reg_multi('tick_BidDepthDev', tick_DepthDeviation_single_day,
                   b_cols, ['mean', 'std'])

        # tick_AskDepthDev: 手动 lambda
        ask_dev_func = lambda df: safe_divide(
            tick_long_to_wide_(df, 'asize1') - pd.concat(
                [tick_long_to_wide_(df, f'asize{i}') for i in range(1, 11)], axis=1
            ).mean(axis=1),
            pd.concat(
                [tick_long_to_wide_(df, f'asize{i}') for i in range(1, 11)], axis=1
            ).mean(axis=1)
        )
        _reg_multi('tick_AskDepthDev', ask_dev_func, a_cols, ['mean', 'std'])

        _reg_multi('tick_VolAccel', tick_VolAcceleration_single_day,
                   ['volume_delta'], ['mean', 'std'])
        _reg_multi('tick_PxAccel', tick_PriceAcceleration_single_day,
                   ['close'], ['mean', 'std'])
        _reg_multi('tick_SpreadDisp', tick_SpreadDispersion_single_day,
                   [f'spread{i}' for i in range(1, 11)], ['mean', 'std'])
        _reg_multi('tick_VolPriceRes', tick_VolPriceRes_single_day,
                   ['close', 'volume_delta'], ['mean', 'std'])

        # 压力衰减
        self.register_factor('tick_BidPressureDecay_mean',
                             lambda df: safe_divide(
                                 tick_long_to_wide_(df, 'bsize1') - tick_long_to_wide_(df, 'bsize10'),
                                 tick_long_to_wide_(df, 'bsize1')
                             ),
                             ['bsize1', 'bsize10'], agg_method='mean')
        self.register_factor('tick_AskPressureDecay_mean',
                             lambda df: safe_divide(
                                 tick_long_to_wide_(df, 'asize1') - tick_long_to_wide_(df, 'asize10'),
                                 tick_long_to_wide_(df, 'asize1')
                             ),
                             ['asize1', 'asize10'], agg_method='mean')

        _reg_multi('tick_RelBookStrength', tick_RelBookStrength_single_day,
                   ['totalbsize', 'totalasize', 'midprice1'], ['mean', 'std'])
        _reg_multi('tick_TurnTrendStr', tick_TurnTrendStrength_single_day,
                   ['volume_delta'], ['mean', 'std'])
        _reg_multi('tick_PriceTrendStr', tick_PriceTrendStrength_single_day,
                   ['close'], ['mean', 'std'])
        _reg_multi('tick_VolRatio5', tick_VolRatio_single_day,
                   ['volume_delta'], ['mean', 'std'])
        _reg_multi('tick_BookPxDev', tick_BookPriceDeviation_single_day,
                   ['close', 'midprice1'], ['mean', 'std'])
        _reg_multi('tick_BidVol5', tick_BidVolatility_single_day,
                   ['bsize1'], ['mean', 'std'])
        _reg_multi('tick_AskVol5', tick_AskVolatility_single_day,
                   ['asize1'], ['mean', 'std'])

        # tick_CumsumVolStats_single_day 返回 (mean, std) 元组
        cv_need = ['volume_delta']
        self.register_factor('tick_CumsumVolMean_mean',
                             lambda df: tick_CumsumVolStats_single_day(df)[0],
                             cv_need, agg_method='mean')
        self.register_factor('tick_CumsumVolStd_mean',
                             lambda df: tick_CumsumVolStats_single_day(df)[1],
                             cv_need, agg_method='mean')


# ---------------------------------------------------------------------------
# 主程序入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from factor_pool.utils import tick_long_to_wide_, safe_divide

    tot_tick_df = pd.read_parquet(f"{results_path}/merge_data/merge_data.parquet")

    # 示例：计算目标日期 1 的日度因子，使用 date=0 的 tick 数据
    target_date = '1'

    daily_pool_builder = DailyFactorPool(tot_tick_df)
    daily_factor_pool = daily_pool_builder.build_daily_factor_pool(target_date=target_date)

    print(f"日度因子池形状: {daily_factor_pool.shape}")
    print(daily_factor_pool.head())

