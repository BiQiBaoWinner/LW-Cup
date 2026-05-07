import os
import sys
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)
from tick_factor_pool import *

data_path = '~/LWCUP/data'
results_path = '~/LWCUP/results'

tot_date_range = [str(i) for i in range(120)]
train_beg = '0'
train_end = '99'
valid_beg = '100'
valid_end = '109'
test_beg = '110'
test_end = '119'
range_split = {
    'train': (train_beg, train_end),
    'valid': (valid_beg, valid_end),
    'test': (test_beg, test_end),
}

tot_cols = ['date', 'sym', 'time', 'open', 'high', 'low', 'close', 'volume_delta', 'amount_delta', 'bid1', 'bsize1', 'bid2', 'bsize2', 'bid3', 'bsize3', 'bid4', 'bsize4', 'bid5', 'bsize5', 'bid6', 'bsize6', 'bid7', 'bsize7', 'bid8', 'bsize8', 'bid9', 'bsize9', 'bid10', 'bsize10', 'ask1', 'asize1', 'ask2', 'asize2', 'ask3', 'asize3', 'ask4', 'asize4', 'ask5', 'asize5', 'ask6', 'asize6', 'ask7', 'asize7', 'ask8', 'asize8', 'ask9', 'asize9', 'ask10', 'asize10', 'avgbid', 'avgask', 'totalbsize', 'totalasize', 'lb_intst', 'la_intst', 'mb_intst', 'ma_intst', 'cb_intst', 'ca_intst', 'lb_ind', 'la_ind', 'mb_ind', 'ma_ind', 'cb_ind', 'ca_ind', 'lb_acc', 'la_acc', 'mb_acc', 'ma_acc', 'cb_acc', 'ca_acc', 'midprice1', 'midprice2', 'midprice3', 'midprice4', 'midprice5', 'midprice6', 'midprice7', 'midprice8', 'midprice9', 'midprice10', 'spread1', 'spread2', 'spread3', 'spread4', 'spread5', 'spread6', 'spread7', 'spread8', 'spread9', 'spread10', 'bid_diff1', 'bid_diff2', 'bid_diff3', 'bid_diff4', 'bid_diff5', 'bid_diff6', 'bid_diff7', 'bid_diff8', 'bid_diff9', 'bid_diff10', 'ask_diff1', 'ask_diff2', 'ask_diff3', 'ask_diff4', 'ask_diff5', 'ask_diff6', 'ask_diff7', 'ask_diff8', 'ask_diff9', 'ask_diff10', 'bid_mean', 'ask_mean', 'bsize_mean', 'asize_mean', 'cumspread', 'imbalance', 'bid_rate1', 'bid_rate2', 'bid_rate3', 'bid_rate4', 'bid_rate5', 'bid_rate6', 'bid_rate7', 'bid_rate8', 'bid_rate9', 'bid_rate10', 'ask_rate1', 'ask_rate2', 'ask_rate3', 'ask_rate4', 'ask_rate5', 'ask_rate6', 'ask_rate7', 'ask_rate8', 'ask_rate9', 'ask_rate10', 'bsize_rate1', 'bsize_rate2', 'bsize_rate3', 'bsize_rate4', 'bsize_rate5', 'bsize_rate6', 'bsize_rate7', 'bsize_rate8', 'bsize_rate9', 'bsize_rate10', 'asize_rate1', 'asize_rate2', 'asize_rate3', 'asize_rate4', 'asize_rate5', 'asize_rate6', 'asize_rate7', 'asize_rate8', 'asize_rate9', 'asize_rate10', 'midprice', 'label_5', 'label_10', 'label_20', 'label_40', 'label_60']
add_ols = ['Ndate', 'timestamp'] # 人工加上的标准化时间戳

labels = ['label_5', 'label_10', 'label_20', 'label_40', 'label_60']

pv_cols = ['open', 'high', 'low', 'close', 'volume_delta', 'amount_delta']

ob_cols_base = ['bid1', 'bsize1', 'bid2', 'bsize2', 'bid3', 'bsize3', 'bid4', 'bsize4', 'bid5', 'bsize5', 'bid6', 'bsize6', 'bid7', 'bsize7', 'bid8', 'bsize8', 'bid9', 'bsize9', 'bid10', 'bsize10', 'ask1', 'asize1', 'ask2', 'asize2', 'ask3', 'asize3', 'ask4', 'asize4', 'ask5', 'asize5', 'ask6', 'asize6', 'ask7', 'asize7', 'ask8', 'asize8', 'ask9', 'asize9', 'ask10', 'asize10', 'midprice1', 'midprice2', 'midprice3', 'midprice4', 'midprice5', 'midprice6', 'midprice7', 'midprice8', 'midprice9', 'midprice10', 'spread1', 'spread2', 'spread3', 'spread4', 'spread5', 'spread6', 'spread7', 'spread8', 'spread9', 'spread10']

ob_cols_derive1 = ['bid_diff1', 'bid_diff2', 'bid_diff3', 'bid_diff4', 'bid_diff5', 'bid_diff6', 'bid_diff7', 'bid_diff8', 'bid_diff9', 'bid_diff10', 'ask_diff1', 'ask_diff2', 'ask_diff3', 'ask_diff4', 'ask_diff5', 'ask_diff6', 'ask_diff7', 'ask_diff8', 'ask_diff9', 'ask_diff10']

ob_cols_derive2 = ['bid_mean', 'ask_mean', 'bsize_mean', 'asize_mean', 'cumspread', 'imbalance', 'bid_rate1', 'bid_rate2', 'bid_rate3', 'bid_rate4', 'bid_rate5', 'bid_rate6', 'bid_rate7', 'bid_rate8', 'bid_rate9', 'bid_rate10', 'ask_rate1', 'ask_rate2', 'ask_rate3', 'ask_rate4', 'ask_rate5', 'ask_rate6', 'ask_rate7', 'ask_rate8', 'ask_rate9', 'ask_rate10', 'bsize_rate1', 'bsize_rate2', 'bsize_rate3', 'bsize_rate4', 'bsize_rate5', 'bsize_rate6', 'bsize_rate7', 'bsize_rate8', 'bsize_rate9', 'bsize_rate10', 'asize_rate1', 'asize_rate2', 'asize_rate3', 'asize_rate4', 'asize_rate5', 'asize_rate6', 'asize_rate7', 'asize_rate8', 'asize_rate9', 'asize_rate10']

# 'avgbid', 'avgask', 'totalbsize', 'totalasize' 是千档
ob_cols_pro = ['avgbid', 'avgask', 'totalbsize', 'totalasize', 'lb_intst', 'la_intst', 'mb_intst', 'ma_intst', 'cb_intst', 'ca_intst', 'lb_ind', 'la_ind', 'mb_ind', 'ma_ind', 'cb_ind', 'ca_ind', 'lb_acc', 'la_acc', 'mb_acc', 'ma_acc', 'cb_acc', 'ca_acc']

FACTORS = {
    # 纯价类
    
    # 纯量类
    "tick_OBI": {
        'factor_func': tick_Orderbook_Imbalance_single_day,
        'need_cols': ['bsize1', 'asize1']
    },  
    
    # 量价交叉因子
    "tick_LS": {
        'factor_func': tick_LiquidityShortfall_single_day,
        'need_cols': ['volume_delta', 'midprice']
    },
    
    "tick_AOBI": {
        'factor_func': tick_Amount_Orderbook_Imbalance_single_day,
        'need_cols': ['bid1', 'bsize1', 'ask1', 'asize1']
    },

    "tick_vAOBI": {
        'factor_func': tick_Volume_Amount_Orderbook_Imbalance_single_day,
        'need_cols': ['bid1', 'bsize1', 'ask1', 'asize1', 'volume_delta']
    },

    "tick_logvAOBI": {
        'factor_func': tick_LogVolume_Amount_Orderbook_Imbalance_single_day,
        'need_cols': ['bid1', 'bsize1', 'ask1', 'asize1', 'volume_delta']
    },
}

FACTORS.update({
    "tick_VWOBI_ma5": {
        'factor_func': tick_VWOBI_ma5_single_day,
        'need_cols': [f'bsize{i}' for i in range(1, 11)] + [f'asize{i}' for i in range(1, 11)]
    },
    "tick_FOBI_ma5": {
        'factor_func': tick_FOBI_ma5_single_day,
        'need_cols': ['totalbsize', 'totalasize']
    },
    "tick_TOBI_ma5": {
        'factor_func': tick_TOBI_ma5_single_day,
        'need_cols': ['bsize1', 'asize1']
    },
    "tick_MPC_5": {
        'factor_func': tick_MPC_single_day,
        'need_cols': ['midprice1']
    },
    "tick_MidTrend_5": {
        'factor_func': tick_MidTrend_single_day,
        'need_cols': ['midprice1']
    },
    "tick_VolMaRatio_10": {
        'factor_func': tick_VolMaRatio_single_day,
        'need_cols': ['volume_delta']
    },
    "tick_SpreadSlope5": {
        'factor_func': tick_SpreadSlope5_single_day,
        'need_cols': ['spread5', 'spread1', 'midprice1']
    },
    "tick_VolTrend_5": {
        'factor_func': tick_VolTrend_single_day,
        'need_cols': ['volume_delta']
    },
    "tick_VolPriceTrend_5": {
        'factor_func': tick_VolPriceTrend_single_day,
        'need_cols': ['midprice1', 'volume_delta']
    },
    "tick_DepthStrength": {
        'factor_func': tick_DepthStrength_single_day,
        'need_cols': ['totalbsize', 'totalasize', 'midprice1']
    },
    "tick_OrderBookSkew": {
        'factor_func': tick_OrderBookSkew_single_day,
        'need_cols': [f'bsize{i}' for i in range(1, 11)] + [f'asize{i}' for i in range(1, 11)]
    },
    "tick_ConcentrationDiff": {
        'factor_func': tick_Concentration_single_day,
        'need_cols': ['bsize1', 'bsize2', 'asize1', 'asize2', 'totalbsize', 'totalasize']
    },
    "tick_SpreadIntensity": {
        'factor_func': tick_SpreadIntensity_single_day,
        'need_cols': ['spread1', 'midprice1']
    },
    "tick_BookEntropyDiff": {
        'factor_func': tick_BookEntropy_single_day,
        'need_cols': [f'bsize{i}' for i in range(1, 11)] + [f'asize{i}' for i in range(1, 11)]
    },
    "tick_ImbalanceStrength": {
        'factor_func': tick_ImbalanceStrength_single_day,
        'need_cols': ['totalbsize', 'totalasize']
    },
    "tick_BidConcentration": {
        'factor_func': tick_BidConcentration_single_day,
        'need_cols': ['bsize1', 'bsize2', 'totalbsize']
    },
    "tick_AskConcentration": {
        'factor_func': tick_AskConcentration_single_day,
        'need_cols': ['asize1', 'asize2', 'totalasize']
    },
    "tick_SpreadSlope": {
        'factor_func': tick_SpreadSlope_single_day,
        'need_cols': ['spread5', 'spread1']
    },
    "tick_BaRatio": {
        'factor_func': tick_BaRatio_single_day,
        'need_cols': ['totalbsize', 'totalasize']
    },
    "tick_BidSpread": {
        'factor_func': tick_BidSpread_single_day,
        'need_cols': [f'bsize{i}' for i in range(1, 11)]
    },
    "tick_AskSpread": {
        'factor_func': tick_AskSpread_single_day,
        'need_cols': [f'asize{i}' for i in range(1, 11)]
    },
    "tick_OrderBookVolatility": {
        'factor_func': tick_OrderBookVolatility_single_day,
        'need_cols': ['bid1', 'ask1']
    },
    "tick_VolPriceSign": {
        'factor_func': tick_VolPriceSign_single_day,
        'need_cols': ['close', 'volume_delta']
    },
    "tick_MomVol": {
        'factor_func': tick_MomVol_single_day,
        'need_cols': ['close', 'volume_delta']
    },
    "tick_ImpactCost": {
        'factor_func': tick_ImpactCost_single_day,
        'need_cols': ['close', 'volume_delta']
    },
    "tick_VolPriceCorr": {
        'factor_func': tick_VolPriceCorr_single_day,
        'need_cols': ['close', 'volume_delta']
    },
    "tick_LiquidSlope": {
        'factor_func': tick_LiquidSlope_single_day,
        'need_cols': ['close', 'volume_delta']
    },
    "tick_BidEntropy": {
        'factor_func': tick_BidEntropy_single_day,
        'need_cols': [f'bsize{i}' for i in range(1, 11)]
    },
    "tick_AskEntropy": {
        'factor_func': tick_AskEntropy_single_day,
        'need_cols': [f'asize{i}' for i in range(1, 11)]
    },
    "tick_BidConvex": {
        'factor_func': tick_BidConvex_single_day,
        'need_cols': ['bsize1', 'totalbsize'] + [f'bsize{i}' for i in range(6, 11)]
    },
    "tick_AskConvex": {
        'factor_func': tick_AskConvex_single_day,
        'need_cols': ['asize1', 'totalasize'] + [f'asize{i}' for i in range(6, 11)]
    },
    "tick_RollImbMean5": {
        'factor_func': tick_RollImbMean5_single_day,
        'need_cols': ['totalbsize', 'totalasize']
    },
    "tick_RollImbStd5": {
        'factor_func': tick_RollImbStd5_single_day,
        'need_cols': ['totalbsize', 'totalasize']
    },
    "tick_MidAccel": {
        'factor_func': tick_MidAccel_single_day,
        'need_cols': ['midprice1']
    },
    "tick_BuyPressure": {
        'factor_func': tick_BuyPressure_single_day,
        'need_cols': ['bsize1', 'spread1']
    },
    "tick_SellPressure": {
        'factor_func': tick_SellPressure_single_day,
        'need_cols': ['asize1', 'spread1']
    },
    "tick_PressureRatio": {
        'factor_func': tick_PressureRatio_single_day,
        'need_cols': ['bsize1', 'asize1', 'spread1']
    },
    "tick_VolImbInteraction": {
        'factor_func': tick_VolImbInteraction_single_day,
        'need_cols': ['volume_delta', 'totalbsize', 'totalasize']
    },
    "tick_MidTrend": {
        'factor_func': tick_MidTrend_single_day,
        'need_cols': ['midprice1']
    },
    "tick_VolMaRatio": {
        'factor_func': tick_VolMaRatio_single_day,
        'need_cols': ['volume_delta']
    },
    "tick_BidAskSlopeDiff": {
        'factor_func': tick_BidAskSlopeDiff_single_day,
        'need_cols': ['bsize1', 'bsize5', 'asize1', 'asize5']
    },
    "tick_VolPriceTrend": {
        'factor_func': tick_VolPriceTrend_single_day,
        'need_cols': ['midprice1', 'volume_delta']
    },
    "tick_VolAdjImpact": {
        'factor_func': tick_VolAdjImpact_single_day,
        'need_cols': ['close', 'volume_delta']
    },
    "tick_MidAccelVol": {
        'factor_func': tick_MidAccelVol_single_day,
        'need_cols': ['midprice1', 'volume_delta']
    },
    "tick_BidTop3Ratio": {
        'factor_func': tick_BidTop3Concentration_single_day,
        'need_cols': ['bsize1', 'bsize2', 'bsize3', 'totalbsize']
    },
    "tick_AskTop3Ratio": {
        'factor_func': tick_AskTop3Concentration_single_day,
        'need_cols': ['asize1', 'asize2', 'asize3', 'totalasize']
    },
    "tick_BookThickness": {
        'factor_func': tick_BookThickness_single_day,
        'need_cols': ['totalbsize', 'totalasize', 'midprice1']
    },
    "tick_PriceVol5": {
        'factor_func': tick_PriceVolatility_single_day,
        'need_cols': ['close']
    },
    "tick_VolVol5": {
        'factor_func': tick_VolVolatility_single_day,
        'need_cols': ['volume_delta']
    },
    "tick_ShortMom3": {
        'factor_func': tick_ShortMom3_single_day,
        'need_cols': ['close']
    },
    "tick_MidMom10": {
        'factor_func': tick_MidMom10_single_day,
        'need_cols': ['close']
    },
    "tick_MomReversal": {
        'factor_func': tick_MomentumReversal_single_day,
        'need_cols': ['close', 'volume_delta']
    },
    "tick_LiquidIndex": {
        'factor_func': tick_LiquidIndex_single_day,
        'need_cols': ['totalbsize', 'totalasize', 'midprice1', 'close']
    },
    "tick_BidDepthDev": {
        'factor_func': tick_DepthDeviation_single_day,
        'need_cols': [f'bsize{i}' for i in range(1, 11)]
    },
    "tick_AskDepthDev": {
        'factor_func': tick_AskDepthDev_single_day,
        'need_cols': [f'asize{i}' for i in range(1, 11)]
    },
    "tick_VolAccel": {
        'factor_func': tick_VolAcceleration_single_day,
        'need_cols': ['volume_delta']
    },
    "tick_PxAccel": {
        'factor_func': tick_PriceAcceleration_single_day,
        'need_cols': ['close']
    },
    "tick_SpreadDisp": {
        'factor_func': tick_SpreadDispersion_single_day,
        'need_cols': [f'spread{i}' for i in range(1, 11)]
    },
    "tick_VolPriceRes": {
        'factor_func': tick_VolPriceRes_single_day,
        'need_cols': ['close', 'volume_delta']
    },
    "tick_BidPressureDecay": {
        'factor_func': tick_BidPressureDecay_single_day,
        'need_cols': ['bsize1', 'bsize10']
    },
    "tick_AskPressureDecay": {
        'factor_func': tick_AskPressureDecay_single_day,
        'need_cols': ['asize1', 'asize10']
    },
    "tick_RelBookStrength": {
        'factor_func': tick_RelBookStrength_single_day,
        'need_cols': ['totalbsize', 'totalasize', 'midprice1']
    },
    "tick_TurnTrendStr": {
        'factor_func': tick_TurnTrendStrength_single_day,
        'need_cols': ['volume_delta']
    },
    "tick_PriceTrendStr": {
        'factor_func': tick_PriceTrendStrength_single_day,
        'need_cols': ['close']
    },
    "tick_VolRatio5": {
        'factor_func': tick_VolRatio_single_day,
        'need_cols': ['volume_delta']
    },
    "tick_BookPxDev": {
        'factor_func': tick_BookPriceDeviation_single_day,
        'need_cols': ['close', 'midprice1']
    },
    "tick_BidVol5": {
        'factor_func': tick_BidVolatility_single_day,
        'need_cols': ['bsize1']
    },
    "tick_AskVol5": {
        'factor_func': tick_AskVolatility_single_day,
        'need_cols': ['asize1']
    },
    "tick_CumsumVolMean": {
        'factor_func': tick_CumsumVolMean_single_day,
        'need_cols': ['volume_delta']
    },
    "tick_CumsumVolStd": {
        'factor_func': tick_CumsumVolStd_single_day,
        'need_cols': ['volume_delta']
    },
})

if __name__ == "__main__":
    
    import sys
    import os
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.append(project_root)
    print(sys.path)