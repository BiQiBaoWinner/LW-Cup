import pandas as pd
import numpy as np
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from factor_pool.utils import tick_long_to_wide_, safe_divide

# tick因子计算函数

# 纯价因子

# 纯量因子
def tick_Orderbook_Imbalance_single_day(tick_df):
    """订单簿不平衡度因子

    Args:
        tick_df: 包含订单簿数据的DataFrame，必须包含以下列：
            - 'bsize1': 买一量
            - 'asize1': 卖一量
    """
    
    wide_bsize1 = tick_long_to_wide_(tick_df, 'bsize1')
    wide_asize1 = tick_long_to_wide_(tick_df, 'asize1')
    
    # 计算订单簿不平衡度
    imbalance = safe_divide(wide_bsize1 - wide_asize1, wide_asize1 + wide_bsize1)
    
    imbalance.index.name = 'timestamp'
    imbalance.columns.name = 'sym'
    
    return imbalance

# 量价交叉因子
def tick_LiquidityShortfall_single_day(tick_df):
    """流动性缺口因子

    Args:
        tick_df: 包含订单簿数据的DataFrame，必须包含以下列：
            - 'volume_delta': 成交流量变化
            - 'spread': 买卖价差
    """
    
    wide_volume = tick_long_to_wide_(tick_df, 'volume_delta')
    wide_midprice = tick_long_to_wide_(tick_df, 'midprice')
    
    mpc = safe_divide(wide_midprice, wide_midprice.shift(5)) - 1
    shortfall = safe_divide(mpc.abs(), wide_volume)
    
    shortfall.index.name = 'timestamp'
    shortfall.columns.name = 'sym'
    
    return shortfall

def tick_Amount_Orderbook_Imbalance_single_day(tick_df):
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
    
    tot_bid = wide_bid1 * wide_bsize1
    tot_ask = wide_ask1 * wide_asize1
    
    # 计算订单簿不平衡度
    imbalance = safe_divide(tot_bid - tot_ask, tot_ask + tot_bid)
    
    imbalance.index.name = 'timestamp'
    imbalance.columns.name = 'sym'
    
    return imbalance

def tick_Volume_Amount_Orderbook_Imbalance_single_day(tick_df):
    """成交量不平衡度因子

    Args:
        tick_df: 包含订单簿数据的DataFrame，必须包含以下列：
            - 'volume_delta': 成交流量变化
            - 'bid1': 买一价
            - 'bsize1': 买一量
            - 'ask1': 卖一价
            - 'asize1': 卖一量
    """
    
    wide_volume = tick_long_to_wide_(tick_df, 'volume_delta')
    wide_bid1 = tick_long_to_wide_(tick_df, 'bid1')
    wide_bsize1 = tick_long_to_wide_(tick_df, 'bsize1')
    wide_ask1 = tick_long_to_wide_(tick_df, 'ask1')
    wide_asize1 = tick_long_to_wide_(tick_df, 'asize1')
    
    tot_bid = wide_bid1 * wide_bsize1
    tot_ask = wide_ask1 * wide_asize1
    
    imbalance = safe_divide(wide_volume * (tot_bid - tot_ask),  (tot_ask + tot_bid))
    
    imbalance.index.name = 'timestamp'
    imbalance.columns.name = 'sym'
    
    return imbalance

def tick_LogVolume_Amount_Orderbook_Imbalance_single_day(tick_df):
    """成交量不平衡度因子，取对数版本

    Args:
        tick_df: 包含订单簿数据的DataFrame，必须包含以下列：
            - 'volume_delta': 成交流量变化
            - 'bid1': 买一价
            - 'bsize1': 买一量
            - 'ask1': 卖一价
            - 'asize1': 卖一量
    """
    
    wide_volume = tick_long_to_wide_(tick_df, 'volume_delta')
    wide_bid1 = tick_long_to_wide_(tick_df, 'bid1')
    wide_bsize1 = tick_long_to_wide_(tick_df, 'bsize1')
    wide_ask1 = tick_long_to_wide_(tick_df, 'ask1')
    wide_asize1 = tick_long_to_wide_(tick_df, 'asize1')
    
    tot_bid = wide_bid1 * wide_bsize1
    tot_ask = wide_ask1 * wide_asize1
    
    imbalance = safe_divide(np.log1p(wide_volume) * (tot_bid - tot_ask),  (tot_ask + tot_bid))
    
    imbalance.index.name = 'timestamp'
    imbalance.columns.name = 'sym'
    
    return imbalance

# --- 整合自 因子库/run.py 的因子函数 ---

def tick_VWOBI_ma5_single_day(tick_df):
    """量比加权订单簿不平衡度（前10档）"""
    weights = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    
    # 提取买卖各10档的量
    bsizes = [tick_long_to_wide_(tick_df, f'bsize{i}') for i in range(1, 11)]
    asizes = [tick_long_to_wide_(tick_df, f'asize{i}') for i in range(1, 11)]
    
    bw = sum(b * w for b, w in zip(bsizes, weights))
    aw = sum(a * w for a, w in zip(asizes, weights))
    
    roll_bw = bw.rolling(window=5, min_periods=1).sum()
    roll_aw = aw.rolling(window=5, min_periods=1).sum()
    
    return safe_divide(roll_bw - roll_aw, roll_bw + roll_aw)

def tick_FOBI_ma5_single_day(tick_df):
    """全档位订单簿不平衡度"""
    wide_totalbsize = tick_long_to_wide_(tick_df, 'totalbsize')
    wide_totalasize = tick_long_to_wide_(tick_df, 'totalasize')
    
    roll_bs = wide_totalbsize.rolling(window=5, min_periods=1).sum()
    roll_as = wide_totalasize.rolling(window=5, min_periods=1).sum()
    
    return safe_divide(roll_bs - roll_as, roll_bs + roll_as)

def tick_TOBI_ma5_single_day(tick_df):
    """一档订单簿不平衡度"""
    wide_bsize1 = tick_long_to_wide_(tick_df, 'bsize1')
    wide_asize1 = tick_long_to_wide_(tick_df, 'asize1')
    
    roll_bs1 = wide_bsize1.rolling(window=5, min_periods=1).sum()
    roll_as1 = wide_asize1.rolling(window=5, min_periods=1).sum()
    
    result = safe_divide(roll_bs1 - roll_as1, roll_bs1 + roll_as1)
    result.index.name = 'timestamp'
    result.columns.name = 'sym'
    return result

def tick_MPC_single_day(tick_df, k=5):
    """中间价变化率 (Midpoint Change)"""
    wide_midprice = tick_long_to_wide_(tick_df, 'midprice1')
    result = safe_divide(wide_midprice - wide_midprice.shift(k), wide_midprice.shift(k))
    result.index.name = 'timestamp'
    result.columns.name = 'sym'
    return result

def tick_MidTrend_single_day(tick_df, window=5):
    """中间价均线趋势"""
    wide_midprice = tick_long_to_wide_(tick_df, 'midprice1')
    return wide_midprice.rolling(window).mean() - wide_midprice.rolling(window*2).mean()

def tick_VolMaRatio_single_day(tick_df, window=10):
    """成交量与其均值的比例"""
    wide_vol = tick_long_to_wide_(tick_df, 'volume_delta')
    return safe_divide(wide_vol, wide_vol.rolling(window).mean())

def tick_SpreadSlope5_single_day(tick_df):
    """5档价差斜率"""
    s5 = tick_long_to_wide_(tick_df, 'spread5')
    s1 = tick_long_to_wide_(tick_df, 'spread1')
    m1 = tick_long_to_wide_(tick_df, 'midprice1')
    return safe_divide(s5 - s1, m1)

def tick_VolTrend_single_day(tick_df, window=5):
    """成交量变化趋势"""
    wide_vol = tick_long_to_wide_(tick_df, 'volume_delta')
    ma = wide_vol.rolling(window).mean()
    return ma - ma.shift(1)

def tick_VolPriceTrend_single_day(tick_df, window=5):
    """量价驱动趋势"""
    wide_mid = tick_long_to_wide_(tick_df, 'midprice1')
    wide_vol = tick_long_to_wide_(tick_df, 'volume_delta')
    px_chg = wide_mid.pct_change(window)
    vol_ma = wide_vol.rolling(window).mean()
    return px_chg * vol_ma

def tick_DepthStrength_single_day(tick_df):
    """深度强度"""
    bs = tick_long_to_wide_(tick_df, 'totalbsize')
    as_ = tick_long_to_wide_(tick_df, 'totalasize')
    m1 = tick_long_to_wide_(tick_df, 'midprice1')
    return safe_divide(bs + as_, m1)

def tick_OrderBookSkew_single_day(tick_df):
    """订单簿偏度差"""
    bsizes = [tick_long_to_wide_(tick_df, f'bsize{i}') for i in range(1, 11)]
    asizes = [tick_long_to_wide_(tick_df, f'asize{i}') for i in range(1, 11)]
    
    b_mean = sum(bsizes) / 10
    b_std = np.sqrt(sum((b - b_mean)**2 for b in bsizes) / 9)
    
    a_mean = sum(asizes) / 10
    a_std = np.sqrt(sum((a - a_mean)**2 for a in asizes) / 9)
    
    return safe_divide(b_mean, b_std) - safe_divide(a_mean, a_std)

def tick_Concentration_single_day(tick_df):
    """挂单集中度差"""
    bs1 = tick_long_to_wide_(tick_df, 'bsize1')
    bs2 = tick_long_to_wide_(tick_df, 'bsize2')
    as1 = tick_long_to_wide_(tick_df, 'asize1')
    as2 = tick_long_to_wide_(tick_df, 'asize2')
    tbs = tick_long_to_wide_(tick_df, 'totalbsize')
    tas = tick_long_to_wide_(tick_df, 'totalasize')
    
    b_conc = safe_divide(bs1 + bs2, tbs)
    a_conc = safe_divide(as1 + as2, tas)
    return b_conc - a_conc

def tick_SpreadIntensity_single_day(tick_df):
    """价差强度"""
    s1 = tick_long_to_wide_(tick_df, 'spread1')
    m1 = tick_long_to_wide_(tick_df, 'midprice1')
    return safe_divide(s1, m1)

def tick_BookEntropy_single_day(tick_df):
    """订单簿信息熵差"""
    bsizes = [tick_long_to_wide_(tick_df, f'bsize{i}') for i in range(1, 11)]
    sum_b = sum(bsizes)
    b_entropy = 0
    for b in bsizes:
        p = safe_divide(b, sum_b)
        b_entropy -= p * np.log2(p + 1e-12)
        
    asizes = [tick_long_to_wide_(tick_df, f'asize{i}') for i in range(1, 11)]
    sum_a = sum(asizes)
    a_entropy = 0
    for a in asizes:
        p = safe_divide(a, sum_a)
        a_entropy -= p * np.log2(p + 1e-12)
        
    return b_entropy - a_entropy

def tick_ConcentrationTop3_single_day(tick_df):
    """前三档挂单集中度差"""
    bsum3 = sum(tick_long_to_wide_(tick_df, f'bsize{i}') for i in range(1, 4))
    asum3 = sum(tick_long_to_wide_(tick_df, f'asize{i}') for i in range(1, 4))
    tbs = tick_long_to_wide_(tick_df, 'totalbsize')
    tas = tick_long_to_wide_(tick_df, 'totalasize')
    return safe_divide(bsum3, tbs) - safe_divide(asum3, tas)

def tick_BookThickness_single_day(tick_df):
    """订单簿厚度"""
    tbs = tick_long_to_wide_(tick_df, 'totalbsize')
    tas = tick_long_to_wide_(tick_df, 'totalasize')
    m1 = tick_long_to_wide_(tick_df, 'midprice1')
    return safe_divide(tbs + tas, m1)

def tick_PressureRatio_single_day(tick_df):
    """买卖压力比"""
    bs1 = tick_long_to_wide_(tick_df, 'bsize1')
    as1 = tick_long_to_wide_(tick_df, 'asize1')
    s1 = tick_long_to_wide_(tick_df, 'spread1')
    buy_p = safe_divide(bs1, s1)
    sell_p = safe_divide(as1, s1)
    return safe_divide(buy_p, sell_p)

def tick_VolImbInteraction_single_day(tick_df):
    """成交量与不平衡度交互"""
    wide_vol = tick_long_to_wide_(tick_df, 'volume_delta')
    tbs = tick_long_to_wide_(tick_df, 'totalbsize')
    tas = tick_long_to_wide_(tick_df, 'totalasize')
    imb = safe_divide(tbs - tas, tbs + tas)
    return imb * np.log1p(wide_vol)

def tick_BookConvex_single_day(tick_df):
    """订单簿凸度差"""
    bs1 = tick_long_to_wide_(tick_df, 'bsize1')
    as1 = tick_long_to_wide_(tick_df, 'asize1')
    tbs = tick_long_to_wide_(tick_df, 'totalbsize')
    tas = tick_long_to_wide_(tick_df, 'totalasize')
    
    b_tail_mean = sum(tick_long_to_wide_(tick_df, f'bsize{i}') for i in range(6, 11)) / 5
    a_tail_mean = sum(tick_long_to_wide_(tick_df, f'asize{i}') for i in range(6, 11)) / 5
    
    b_convex = safe_divide(bs1 - b_tail_mean, tbs)
    a_convex = safe_divide(as1 - a_tail_mean, tas)
    return b_convex - a_convex

def tick_RollImbStats_single_day(tick_df, window=5):
    """滚动不平衡度统计量"""
    tbs = tick_long_to_wide_(tick_df, 'totalbsize')
    tas = tick_long_to_wide_(tick_df, 'totalasize')
    imb = safe_divide(tbs - tas, tbs + tas)
    return imb.rolling(window).mean(), imb.rolling(window).std()

def tick_MidAccel_single_day(tick_df, k=1):
    """中间价加速度"""
    m1 = tick_long_to_wide_(tick_df, 'midprice1')
    chg = safe_divide(m1 - m1.shift(k), m1.shift(k))
    return chg - chg.shift(1)

def tick_VolPriceFeatures_single_day(tick_df, window=10):
    """量价综合特征"""
    close = tick_long_to_wide_(tick_df, 'close')
    vol = tick_long_to_wide_(tick_df, 'volume_delta')
    
    res = {}
    res['vol_price_sign'] = np.sign(close) * np.sign(vol)
    res['mom_vol'] = close * vol
    res['impact_cost'] = close.abs() * vol
    res['vol_price_corr'] = close.rolling(window).corr(vol)
    
    p_std = close.rolling(window).std()
    v_std = vol.rolling(window).std()
    res['liquid_slope'] = safe_divide(p_std, v_std)
    return res

def tick_OrderBookVolatility_single_day(tick_df, window=10):
    """订单簿波动率"""
    bid1 = tick_long_to_wide_(tick_df, 'bid1')
    ask1 = tick_long_to_wide_(tick_df, 'ask1')
    ba = bid1 + ask1
    roll = ba.rolling(window)
    return safe_divide(roll.std(), roll.mean())

def tick_Gradient_single_day(tick_df):
    """买卖报价梯度差"""
    bs1 = tick_long_to_wide_(tick_df, 'bsize1')
    bs5 = tick_long_to_wide_(tick_df, 'bsize5')
    as1 = tick_long_to_wide_(tick_df, 'asize1')
    as5 = tick_long_to_wide_(tick_df, 'asize5')
    
    b_grad = safe_divide(bs1 - bs5, bs1 + bs5)
    a_grad = safe_divide(as1 - as5, as1 + as5)
    return b_grad - a_grad

def tick_DepthDeviation_single_day(tick_df):
    """深度偏离度差"""
    bsizes = [tick_long_to_wide_(tick_df, f'bsize{i}') for i in range(1, 11)]
    asizes = [tick_long_to_wide_(tick_df, f'asize{i}') for i in range(1, 11)]
    
    b_mean = sum(bsizes) / 10
    a_mean = sum(asizes) / 10
    
    b_dev = safe_divide(bsizes[0] - b_mean, b_mean)
    a_dev = safe_divide(asizes[0] - a_mean, a_mean)
    return b_dev - a_dev

def tick_PriceVolAccel_single_day(tick_df):
    """价量加速度"""
    close = tick_long_to_wide_(tick_df, 'close')
    vol = tick_long_to_wide_(tick_df, 'volume_delta')
    
    px_ma3 = close.rolling(3).mean()
    px_ma6 = close.rolling(6).mean()
    vol_ma3 = vol.rolling(3).mean()
    vol_ma6 = vol.rolling(6).mean()
    
    return px_ma3 - px_ma6, vol_ma3 - vol_ma6

def tick_SpreadDispersion_single_day(tick_df):
    """价差离散度"""
    spreads = [tick_long_to_wide_(tick_df, f'spread{i}') for i in range(1, 11)]
    s_mean = sum(spreads) / 10
    s_std = np.sqrt(sum((s - s_mean)**2 for s in spreads) / 9)
    return safe_divide(s_std, s_mean)

def tick_MomentumReversal_single_day(tick_df):
    """价格动量反转特征"""
    close = tick_long_to_wide_(tick_df, 'close')
    vol = tick_long_to_wide_(tick_df, 'volume_delta')
    return -close * vol.abs()

def tick_ImbalanceStrength_single_day(tick_df):
    """盘口不平衡强度"""
    tbs = tick_long_to_wide_(tick_df, 'totalbsize')
    tas = tick_long_to_wide_(tick_df, 'totalasize')
    return safe_divide(tbs - tas, tbs + tas)

def tick_BidConcentration_single_day(tick_df):
    """买盘集中度"""
    b1 = tick_long_to_wide_(tick_df, 'bsize1')
    b2 = tick_long_to_wide_(tick_df, 'bsize2')
    tbs = tick_long_to_wide_(tick_df, 'totalbsize')
    return safe_divide(b1 + b2, tbs)

def tick_AskConcentration_single_day(tick_df):
    """卖盘集中度"""
    a1 = tick_long_to_wide_(tick_df, 'asize1')
    a2 = tick_long_to_wide_(tick_df, 'asize2')
    tas = tick_long_to_wide_(tick_df, 'totalasize')
    return safe_divide(a1 + a2, tas)

def tick_SpreadSlope_single_day(tick_df):
    """价差斜率"""
    s5 = tick_long_to_wide_(tick_df, 'spread5')
    s1 = tick_long_to_wide_(tick_df, 'spread1')
    return safe_divide(s5 - s1, 4)

def tick_BaRatio_single_day(tick_df):
    """买卖盘总量比"""
    tbs = tick_long_to_wide_(tick_df, 'totalbsize')
    tas = tick_long_to_wide_(tick_df, 'totalasize')
    return safe_divide(tbs, tas)

def tick_BidSpread_single_day(tick_df):
    """买盘离散度"""
    bframe = pd.concat([tick_long_to_wide_(tick_df, f'bsize{i}') for i in range(1, 11)], axis=1)
    result = safe_divide(bframe.max(axis=1) - bframe.min(axis=1), bframe.mean(axis=1))
    # 确保结果有正确的列名
    result = result.to_frame()
    # 使用第一个股票的列名作为基准
    if not bframe.empty:
        result.columns = [bframe.columns[0]]
    result.index.name = 'timestamp'
    result.columns.name = 'sym'
    return result

def tick_AskSpread_single_day(tick_df):
    """卖盘离散度"""
    aframe = pd.concat([tick_long_to_wide_(tick_df, f'asize{i}') for i in range(1, 11)], axis=1)
    result = safe_divide(aframe.max(axis=1) - aframe.min(axis=1), aframe.mean(axis=1))
    # 确保结果有正确的列名
    result = result.to_frame()
    # 使用第一个股票的列名作为基准
    if not aframe.empty:
        result.columns = [aframe.columns[0]]
    result.index.name = 'timestamp'
    result.columns.name = 'sym'
    return result

def _book_entropy_single_side(side_frames):
    side_frame = pd.concat(side_frames, axis=1)
    total = side_frame.sum(axis=1)
    p = side_frame.div(total.replace(0, np.nan), axis=0).fillna(0)
    p = p.mask(p <= 0, 1e-12)
    return (-p * np.log2(p)).sum(axis=1)

def tick_BidEntropy_single_day(tick_df):
    """买盘信息熵"""
    result = _book_entropy_single_side([tick_long_to_wide_(tick_df, f'bsize{i}') for i in range(1, 11)])
    # 确保结果有正确的列名
    result = result.to_frame()
    # 使用第一个股票的列名作为基准
    bframe = tick_long_to_wide_(tick_df, 'bsize1')
    if not bframe.empty:
        result.columns = [bframe.columns[0]]
    result.index.name = 'timestamp'
    result.columns.name = 'sym'
    return result

def tick_AskEntropy_single_day(tick_df):
    """卖盘信息熵"""
    result = _book_entropy_single_side([tick_long_to_wide_(tick_df, f'asize{i}') for i in range(1, 11)])
    result = result.to_frame()
    aframe = tick_long_to_wide_(tick_df, 'asize1')
    if not aframe.empty:
        result.columns = [aframe.columns[0]]
    result.index.name = 'timestamp'
    result.columns.name = 'sym'
    return result

def tick_BidConvex_single_day(tick_df):
    """买盘凸度"""
    b1 = tick_long_to_wide_(tick_df, 'bsize1')
    tbs = tick_long_to_wide_(tick_df, 'totalbsize')
    b_tail_mean = sum(tick_long_to_wide_(tick_df, f'bsize{i}') for i in range(6, 11)) / 5
    return safe_divide(b1 - b_tail_mean, tbs)

def tick_AskConvex_single_day(tick_df):
    """卖盘凸度"""
    a1 = tick_long_to_wide_(tick_df, 'asize1')
    tas = tick_long_to_wide_(tick_df, 'totalasize')
    a_tail_mean = sum(tick_long_to_wide_(tick_df, f'asize{i}') for i in range(6, 11)) / 5
    return safe_divide(a1 - a_tail_mean, tas)

def tick_BuyPressure_single_day(tick_df):
    """买压强度"""
    b1 = tick_long_to_wide_(tick_df, 'bsize1')
    s1 = tick_long_to_wide_(tick_df, 'spread1')
    return safe_divide(b1, s1)

def tick_SellPressure_single_day(tick_df):
    """卖压强度"""
    a1 = tick_long_to_wide_(tick_df, 'asize1')
    s1 = tick_long_to_wide_(tick_df, 'spread1')
    return safe_divide(a1, s1)

def tick_BidAskSlopeDiff_single_day(tick_df):
    """买卖梯度差"""
    b1 = tick_long_to_wide_(tick_df, 'bsize1')
    b5 = tick_long_to_wide_(tick_df, 'bsize5')
    a1 = tick_long_to_wide_(tick_df, 'asize1')
    a5 = tick_long_to_wide_(tick_df, 'asize5')
    bid_grad = safe_divide(b1 - b5, b1 + b5)
    ask_grad = safe_divide(a1 - a5, a1 + a5)
    return bid_grad - ask_grad

def tick_VolAdjImpact_single_day(tick_df):
    """成交量调整后的冲击成本"""
    close = tick_long_to_wide_(tick_df, 'close')
    vol = tick_long_to_wide_(tick_df, 'volume_delta')
    log_vol = np.log1p(vol).rolling(5, min_periods=1).mean()
    return safe_divide(close.abs(), log_vol + 1e-8)

def tick_RelBookStrength_single_day(tick_df):
    """相对盘口强度"""
    tbs = tick_long_to_wide_(tick_df, 'totalbsize')
    tas = tick_long_to_wide_(tick_df, 'totalasize')
    m1 = tick_long_to_wide_(tick_df, 'midprice1')
    return safe_divide(tbs, tas) * m1 - m1

def tick_TurnTrendStrength_single_day(tick_df, window=5):
    """换手趋势强度"""
    vol = tick_long_to_wide_(tick_df, 'volume_delta')
    return safe_divide(vol.rolling(window, min_periods=1).sum(), vol.abs().rolling(window, min_periods=1).sum())

def tick_PriceTrendStrength_single_day(tick_df, window=5):
    """价格趋势强度"""
    close = tick_long_to_wide_(tick_df, 'close')
    return safe_divide(close.rolling(window, min_periods=1).sum(), close.abs().rolling(window, min_periods=1).sum())

def tick_VolRatio_single_day(tick_df, window=5):
    """成交量占比"""
    vol = tick_long_to_wide_(tick_df, 'volume_delta')
    return safe_divide(vol, vol.rolling(window, min_periods=1).sum())

def tick_BookPriceDeviation_single_day(tick_df):
    """盘口价格偏离"""
    close = tick_long_to_wide_(tick_df, 'close')
    m1 = tick_long_to_wide_(tick_df, 'midprice1')
    return safe_divide(m1 - close, m1)

def tick_BidVolatility_single_day(tick_df, window=5):
    """买盘波动率"""
    b1 = tick_long_to_wide_(tick_df, 'bsize1')
    return b1.rolling(window, min_periods=1).std()

def tick_AskVolatility_single_day(tick_df, window=5):
    """卖盘波动率"""
    a1 = tick_long_to_wide_(tick_df, 'asize1')
    return a1.rolling(window, min_periods=1).std()

def tick_CumsumVolStats_single_day(tick_df):
    """累计成交量统计"""
    vol = tick_long_to_wide_(tick_df, 'volume_delta')
    cumsum = vol.cumsum()
    return cumsum.expanding().mean(), cumsum.expanding().std()

def tick_BidTop3Concentration_single_day(tick_df):
    """买盘前三档集中度"""
    top3 = sum(tick_long_to_wide_(tick_df, f'bsize{i}') for i in range(1, 4))
    tbs = tick_long_to_wide_(tick_df, 'totalbsize')
    return safe_divide(top3, tbs)

def tick_AskTop3Concentration_single_day(tick_df):
    """卖盘前三档集中度"""
    top3 = sum(tick_long_to_wide_(tick_df, f'asize{i}') for i in range(1, 4))
    tas = tick_long_to_wide_(tick_df, 'totalasize')
    return safe_divide(top3, tas)

def tick_LiquidIndex_single_day(tick_df):
    """流动性指数"""
    tbs = tick_long_to_wide_(tick_df, 'totalbsize')
    tas = tick_long_to_wide_(tick_df, 'totalasize')
    m1 = tick_long_to_wide_(tick_df, 'midprice1')
    close = tick_long_to_wide_(tick_df, 'close')
    return safe_divide((tbs + tas) * m1, close.abs() + 1e-8)

def tick_VolAcceleration_single_day(tick_df):
    """成交量加速度"""
    vol = tick_long_to_wide_(tick_df, 'volume_delta')
    return vol.rolling(3, min_periods=1).mean() - vol.rolling(6, min_periods=1).mean()

def tick_PriceAcceleration_single_day(tick_df):
    """价格加速度"""
    close = tick_long_to_wide_(tick_df, 'close')
    return close.rolling(3, min_periods=1).mean() - close.rolling(6, min_periods=1).mean()

def tick_MidAccelVol_single_day(tick_df):
    """中间价加速度与成交量交互"""
    return tick_MidAccel_single_day(tick_df) * np.log1p(tick_long_to_wide_(tick_df, 'volume_delta'))

def tick_PriceVolatility_single_day(tick_df, window=5):
    """价格波动率"""
    close = tick_long_to_wide_(tick_df, 'close')
    return close.rolling(window, min_periods=1).std()

def tick_VolVolatility_single_day(tick_df, window=5):
    """成交量波动率"""
    vol = tick_long_to_wide_(tick_df, 'volume_delta')
    return vol.rolling(window, min_periods=1).std()

def tick_VolPriceRes_single_day(tick_df, window=5):
    """量价共振"""
    close = tick_long_to_wide_(tick_df, 'close')
    vol = tick_long_to_wide_(tick_df, 'volume_delta')
    px_norm = close / (close.shift(1).rolling(window, min_periods=1).mean() + 1e-8)
    vol_norm = vol / (vol.shift(1).rolling(window, min_periods=1).mean() + 1e-8)
    return px_norm * vol_norm

def tick_VolPriceSign_single_day(tick_df):
    """量价信号 — 从 tick_VolPriceFeatures_single_day 提取"""
    return tick_VolPriceFeatures_single_day(tick_df)['vol_price_sign']

def tick_MomVol_single_day(tick_df):
    """动量成交量 — 从 tick_VolPriceFeatures_single_day 提取"""
    return tick_VolPriceFeatures_single_day(tick_df)['mom_vol']

def tick_ImpactCost_single_day(tick_df):
    """冲击成本 — 从 tick_VolPriceFeatures_single_day 提取"""
    return tick_VolPriceFeatures_single_day(tick_df)['impact_cost']

def tick_VolPriceCorr_single_day(tick_df):
    """量价相关性 — 从 tick_VolPriceFeatures_single_day 提取"""
    return tick_VolPriceFeatures_single_day(tick_df)['vol_price_corr']

def tick_LiquidSlope_single_day(tick_df):
    """流动性斜率 — 从 tick_VolPriceFeatures_single_day 提取"""
    return tick_VolPriceFeatures_single_day(tick_df)['liquid_slope']

def tick_RollImbMean5_single_day(tick_df):
    """滚动不平衡均值 — 从 tick_RollImbStats_single_day 提取"""
    return tick_RollImbStats_single_day(tick_df)[0]

def tick_RollImbStd5_single_day(tick_df):
    """滚动不平衡标准差 — 从 tick_RollImbStats_single_day 提取"""
    return tick_RollImbStats_single_day(tick_df)[1]

def tick_CumsumVolMean_single_day(tick_df):
    """累积成交量均值 — 从 tick_CumsumVolStats_single_day 提取"""
    return tick_CumsumVolStats_single_day(tick_df)[0]

def tick_CumsumVolStd_single_day(tick_df):
    """累积成交量标准差 — 从 tick_CumsumVolStats_single_day 提取"""
    return tick_CumsumVolStats_single_day(tick_df)[1]

def tick_ShortMom3_single_day(tick_df):
    """短期动量(3) — tick_long_to_wide_ + rolling mean"""
    return tick_long_to_wide_(tick_df, 'close').rolling(3, min_periods=1).mean()

def tick_MidMom10_single_day(tick_df):
    """中期动量(10) — tick_long_to_wide_ + rolling mean"""
    return tick_long_to_wide_(tick_df, 'close').rolling(10, min_periods=1).mean()

def tick_AskDepthDev_single_day(tick_df):
    """卖盘深度偏离 — asize1 相对所有档位均值的偏离"""
    wide_asize1 = tick_long_to_wide_(tick_df, 'asize1')
    all_asizes = pd.concat([tick_long_to_wide_(tick_df, f'asize{i}') for i in range(1, 11)], axis=1)
    mean_asize = all_asizes.mean(axis=1)
    return safe_divide(wide_asize1 - mean_asize, mean_asize)

def tick_BidPressureDecay_single_day(tick_df):
    """买盘压力衰减 — bsize1 到 bsize10 的衰减比例"""
    return safe_divide(
        tick_long_to_wide_(tick_df, 'bsize1') - tick_long_to_wide_(tick_df, 'bsize10'),
        tick_long_to_wide_(tick_df, 'bsize1')
    )

def tick_AskPressureDecay_single_day(tick_df):
    """卖盘压力衰减 — asize1 到 asize10 的衰减比例"""
    return safe_divide(
        tick_long_to_wide_(tick_df, 'asize1') - tick_long_to_wide_(tick_df, 'asize10'),
        tick_long_to_wide_(tick_df, 'asize1')
    )

class TickFactorPool:
    def __init__(self, tick_df):
        self.data = tick_df
        self.registry = {}
        self._register_default_factors()

    def _register_default_factors(self):
        """自动注册所有内置因子"""
        # --- 原有及第一批整合因子 ---
        self.register_factor('tick_OBI', tick_Orderbook_Imbalance_single_day, ['bsize1', 'asize1'])
        self.register_factor('tick_LS', tick_LiquidityShortfall_single_day, ['volume_delta', 'midprice'])
        self.register_factor('tick_AOBI', tick_Amount_Orderbook_Imbalance_single_day, ['bid1', 'bsize1', 'ask1', 'asize1'])
        self.register_factor('tick_vAOBI', tick_Volume_Amount_Orderbook_Imbalance_single_day, ['volume_delta', 'bid1', 'bsize1', 'ask1', 'asize1'])
        self.register_factor('tick_logvAOBI', tick_LogVolume_Amount_Orderbook_Imbalance_single_day, ['volume_delta', 'bid1', 'bsize1', 'ask1', 'asize1'])
        self.register_factor('tick_VWOBI_ma5', tick_VWOBI_ma5_single_day, [f'bsize{i}' for i in range(1,11)] + [f'asize{i}' for i in range(1,11)])
        self.register_factor('tick_FOBI_ma5', tick_FOBI_ma5_single_day, ['totalbsize', 'totalasize'])
        self.register_factor('tick_TOBI_ma5', tick_TOBI_ma5_single_day, ['bsize1', 'asize1'])
        self.register_factor('tick_MPC_5', tick_MPC_single_day, ['midprice1'])
        self.register_factor('tick_MidTrend_5', tick_MidTrend_single_day, ['midprice1'])
        self.register_factor('tick_VolMaRatio_10', tick_VolMaRatio_single_day, ['volume_delta'])
        self.register_factor('tick_SpreadSlope5', tick_SpreadSlope5_single_day, ['spread5', 'spread1', 'midprice1'])
        self.register_factor('tick_VolTrend_5', tick_VolTrend_single_day, ['volume_delta'])
        self.register_factor('tick_VolPriceTrend_5', tick_VolPriceTrend_single_day, ['midprice1', 'volume_delta'])
        self.register_factor('tick_DepthStrength', tick_DepthStrength_single_day, ['totalbsize', 'totalasize', 'midprice1'])
        self.register_factor('tick_OrderBookSkew', tick_OrderBookSkew_single_day, [f'bsize{i}' for i in range(1,11)] + [f'asize{i}' for i in range(1,11)])
        self.register_factor('tick_ConcentrationDiff', tick_Concentration_single_day, ['bsize1', 'bsize2', 'asize1', 'asize2', 'totalbsize', 'totalasize'])
        self.register_factor('tick_SpreadIntensity', tick_SpreadIntensity_single_day, ['spread1', 'midprice1'])
        self.register_factor('tick_BookEntropyDiff', tick_BookEntropy_single_day, [f'bsize{i}' for i in range(1,11)] + [f'asize{i}' for i in range(1,11)])

        # --- 第二批补全的所有登记因子 ---
        self.register_factor('tick_ImbalanceStrength', tick_ImbalanceStrength_single_day, ['totalbsize', 'totalasize'])
        self.register_factor('tick_BidConcentration', tick_BidConcentration_single_day, ['bsize1', 'bsize2', 'totalbsize'])
        self.register_factor('tick_AskConcentration', tick_AskConcentration_single_day, ['asize1', 'asize2', 'totalasize'])
        self.register_factor('tick_SpreadSlope', tick_SpreadSlope_single_day, ['spread5', 'spread1'])
        self.register_factor('tick_BaRatio', tick_BaRatio_single_day, ['totalbsize', 'totalasize'])
        self.register_factor('tick_BidSpread', tick_BidSpread_single_day, [f'bsize{i}' for i in range(1, 11)])
        self.register_factor('tick_AskSpread', tick_AskSpread_single_day, [f'asize{i}' for i in range(1, 11)])
        self.register_factor('tick_OrderBookVolatility', tick_OrderBookVolatility_single_day, ['bid1', 'ask1'])
        self.register_factor('tick_VolPriceSign', tick_VolPriceSign_single_day, ['close', 'volume_delta'])
        self.register_factor('tick_MomVol', tick_MomVol_single_day, ['close', 'volume_delta'])
        self.register_factor('tick_ImpactCost', tick_ImpactCost_single_day, ['close', 'volume_delta'])
        self.register_factor('tick_VolPriceCorr', tick_VolPriceCorr_single_day, ['close', 'volume_delta'])
        self.register_factor('tick_LiquidSlope', tick_LiquidSlope_single_day, ['close', 'volume_delta'])
        self.register_factor('tick_BidEntropy', tick_BidEntropy_single_day, [f'bsize{i}' for i in range(1, 11)])
        self.register_factor('tick_AskEntropy', tick_AskEntropy_single_day, [f'asize{i}' for i in range(1, 11)])
        self.register_factor('tick_BidConvex', tick_BidConvex_single_day, ['bsize1', 'totalbsize'] + [f'bsize{i}' for i in range(6,11)])
        self.register_factor('tick_AskConvex', tick_AskConvex_single_day, ['asize1', 'totalasize'] + [f'asize{i}' for i in range(6,11)])
        self.register_factor('tick_RollImbMean5', tick_RollImbMean5_single_day, ['totalbsize', 'totalasize'])
        self.register_factor('tick_RollImbStd5', tick_RollImbStd5_single_day, ['totalbsize', 'totalasize'])
        self.register_factor('tick_MidAccel', tick_MidAccel_single_day, ['midprice1'])
        self.register_factor('tick_BuyPressure', tick_BuyPressure_single_day, ['bsize1', 'spread1'])
        self.register_factor('tick_SellPressure', tick_SellPressure_single_day, ['asize1', 'spread1'])
        self.register_factor('tick_PressureRatio', tick_PressureRatio_single_day, ['bsize1', 'asize1', 'spread1'])
        self.register_factor('tick_VolImbInteraction', tick_VolImbInteraction_single_day, ['volume_delta', 'totalbsize', 'totalasize'])
        self.register_factor('tick_BidAskSlopeDiff', tick_BidAskSlopeDiff_single_day, ['bsize1', 'bsize5', 'asize1', 'asize5'])
        self.register_factor('tick_VolAdjImpact', tick_VolAdjImpact_single_day, ['close', 'volume_delta'])
        self.register_factor('tick_MidAccelVol', tick_MidAccelVol_single_day, ['midprice1', 'volume_delta'])
        self.register_factor('tick_BidTop3Ratio', tick_BidTop3Concentration_single_day, ['bsize1', 'bsize2', 'bsize3', 'totalbsize'])
        self.register_factor('tick_AskTop3Ratio', tick_AskTop3Concentration_single_day, ['asize1', 'asize2', 'asize3', 'totalasize'])
        self.register_factor('tick_BookThickness', tick_BookThickness_single_day, ['totalbsize', 'totalasize', 'midprice1'])
        self.register_factor('tick_PriceVol5', tick_PriceVolatility_single_day, ['close'])
        self.register_factor('tick_VolVol5', tick_VolVolatility_single_day, ['volume_delta'])
        self.register_factor('tick_ShortMom3', tick_ShortMom3_single_day, ['close'])
        self.register_factor('tick_MidMom10', tick_MidMom10_single_day, ['close'])
        self.register_factor('tick_MomReversal', tick_MomentumReversal_single_day, ['close', 'volume_delta'])
        self.register_factor('tick_LiquidIndex', tick_LiquidIndex_single_day, ['totalbsize', 'totalasize', 'midprice1', 'close'])
        self.register_factor('tick_BidDepthDev', tick_DepthDeviation_single_day, [f'bsize{i}' for i in range(1,11)])
        self.register_factor('tick_AskDepthDev', tick_AskDepthDev_single_day, [f'asize{i}' for i in range(1,11)])
        self.register_factor('tick_VolAccel', tick_VolAcceleration_single_day, ['volume_delta'])
        self.register_factor('tick_PxAccel', tick_PriceAcceleration_single_day, ['close'])
        self.register_factor('tick_SpreadDisp', tick_SpreadDispersion_single_day, [f'spread{i}' for i in range(1,11)])
        self.register_factor('tick_VolPriceRes', tick_VolPriceRes_single_day, ['close', 'volume_delta'])
        self.register_factor('tick_BidPressureDecay', tick_BidPressureDecay_single_day, ['bsize1', 'bsize10'])
        self.register_factor('tick_AskPressureDecay', tick_AskPressureDecay_single_day, ['asize1', 'asize10'])
        self.register_factor('tick_RelBookStrength', tick_RelBookStrength_single_day, ['totalbsize', 'totalasize', 'midprice1'])
        self.register_factor('tick_TurnTrendStr', tick_TurnTrendStrength_single_day, ['volume_delta'])
        self.register_factor('tick_PriceTrendStr', tick_PriceTrendStrength_single_day, ['close'])
        self.register_factor('tick_VolRatio5', tick_VolRatio_single_day, ['volume_delta'])
        self.register_factor('tick_BookPxDev', tick_BookPriceDeviation_single_day, ['close', 'midprice1'])
        self.register_factor('tick_BidVol5', tick_BidVolatility_single_day, ['bsize1'])
        self.register_factor('tick_AskVol5', tick_AskVolatility_single_day, ['asize1'])
        self.register_factor('tick_CumsumVolMean', tick_CumsumVolMean_single_day, ['volume_delta'])
        self.register_factor('tick_CumsumVolStd', tick_CumsumVolStd_single_day, ['volume_delta'])
    
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
            
            if isinstance(factor_df, pd.Series):
                print(f"警告: 因子 {factor_name} 返回 Series，转换为 DataFrame")
                factor_df = factor_df.to_frame()
            
            # 过滤非股票代码的列名，防止污染 sym 索引
            valid_syms = set(self.data['sym'].unique())
            valid_cols = [c for c in factor_df.columns if c in valid_syms]
            if not valid_cols:
                print(f"警告: 因子 {factor_name} 没有有效的 sym 列，跳过")
                continue
            factor_df = factor_df[valid_cols]
            
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
    
    factor_pool = tick_factor_pool.build_factor_pool()
    
    print(factor_pool)

