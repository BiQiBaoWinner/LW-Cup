from calendar import c
from numpy.core.defchararray import lower, upper
import pandas as pd
import pickle
import os
day_feature =['mpc_max_1_decay', 'mpc_skew_1_decay', 'logvol_skew_decay', 'logvol_90tail_decay', 'logvol_10tail_decay', 'volroc_skew_decay', 'volroc_kurt_decay', 'vol_entropy_20d_decay', 'vol_maxmean_decay', 'vol_maxstd_decay', 'vol_maxmean_roll_std_decay', 'vol_maxstd_roll_std_decay', 'daily_rtn_decay', 'daily_volatility_decay', 'daily_amplitude_decay', 'daily_volume_mean_decay', 'daily_volume_skew_decay', 'daily_spread_mean_decay', 'daily_imbalance_mean_decay', 'daily_trend_strength_decay', 'daily_price_efficiency_decay', 'realized_vol_decay', 'bipower_var_decay', 'jump_var_decay', 'jump_ratio_decay', 'ret_q05_decay', 'ret_q25_decay', 'ret_q50_decay', 'ret_q75_decay', 'ret_q95_decay', 'ret_range_decay', 'jump_count_decay', 'avg_jump_size_decay', 'pos_jump_size_decay', 'neg_jump_size_decay', 'jump_volatility_decay', 'first_half_return_decay', 'second_half_return_decay', 'intraday_mom_diff_decay', 'large_trade_ratio_decay', 'large_trade_num_ratio_decay', 'active_buy_ratio_decay', 'active_sell_ratio_decay', 'active_imbalance_decay', 'spread_mean_decay', 'spread_median_decay', 'spread_std_decay', 'spread_skew_decay', 'spread_kurt_decay', 'spread_max_decay', 'spread_min_decay', 'amihud_illiquidity_decay', 'volume_autocorr_decay', 'opening_gap_decay', 'volume_weighted_spread_decay', 'volatility_ratio_decay', 'imb_mean_decay', 'imb_std_decay', 'imb_skew_decay', 'imb_kurt_decay', 'high_low_time_ratio_decay']
tick_feature =  ['bid_rate1','bid_rate2','bid_rate3','bid_rate4','bid_rate5','bid_rate6','bid_rate7','bid_rate8','bid_rate9','bid_rate10','ask_rate1','ask_rate2','ask_rate3','ask_rate4','ask_rate5','ask_rate6','ask_rate7','ask_rate8','ask_rate9','ask_rate10','bsize_rate1','bsize_rate2','bsize_rate3','bsize_rate4','bsize_rate5','bsize_rate6','bsize_rate7','bsize_rate8','bsize_rate9','bsize_rate10','asize_rate1','asize_rate2','asize_rate3','asize_rate4','asize_rate5','asize_rate6','asize_rate7','asize_rate8','asize_rate9','asize_rate10','bid_diff1','bid_diff2','bid_diff3','bid_diff4','bid_diff5','bid_diff6','bid_diff7','bid_diff8','bid_diff9','bid_diff10','ask_diff1','ask_diff2','ask_diff3','ask_diff4','ask_diff5','ask_diff6','ask_diff7','ask_diff8','ask_diff9','ask_diff10','lb_intst','la_intst','mb_intst','ma_intst','cb_intst','ca_intst','lb_ind','la_ind','mb_ind','ma_ind','cb_ind','ca_ind','lb_acc','la_acc','mb_acc','ma_acc','cb_acc','ca_acc','imbalance','midprice1','midprice2','midprice3','midprice4','midprice5','midprice6','midprice7','midprice8','midprice9','midprice10','spread1','spread2','spread3','spread4','spread5','spread6','spread7','spread8','spread9','spread10','midprice', 'cumspread','avgbid', 'avgask', 'totalbsize', 'totalasize','bid_mean', 'ask_mean', 'bsize_mean', 'asize_mean','open', 'high', 'low', 'close', 'bid1','bsize1','bid2','bsize2','bid3','bsize3','bid4','bsize4','bid5','bsize5','bid6','bsize6','bid7','bsize7','bid8','bsize8','bid9','bsize9','bid10','bsize10','ask1','asize1','ask2','asize2','ask3','asize3','ask4','asize4','ask5','asize5','ask6','asize6','ask7','asize7','ask8','asize8','ask9','asize9','ask10','asize10','volume_delta','amount_delta','VWOBI_ma5', 'FOBI_ma5', 'TOBI_ma5', 'mpc_1', 'vol_price_sign', 'mom_vol', 'impact_cost', 'vol_price_corr', 'liquid_slope', 'spread_intensity', 'depth_slope', 'bid_entropy', 'ask_entropy', 'bid_gradient', 'ask_gradient', 'bid_convex', 'ask_convex', 'roll_imb_mean5', 'roll_imb_std5', 'mid_accel', 'buy_pressure', 'sell_pressure', 'pressure_ratio', 'vol_imb_int', 'mid_trend', 'vol_ma_ratio', 'spread_slope_5', 'bid_ask_slope_diff', 'vol_price_trend', 'depth_strength', 'vol_adj_impact', 'order_book_skew', 'mid_accel_vol', 'vol_trend', 'bid_top3_ratio', 'ask_top3_ratio', 'book_thickness', 'price_vol_5', 'vol_vol_5', 'short_mom_3', 'mid_mom_10', 'mom_reversal', 'liquid_index', 'bid_depth_dev', 'ask_depth_dev', 'vol_accel', 'px_accel', 'spread_disp', 'vol_price_res', 'bid_pressure_decay', 'ask_pressure_decay', 'rel_book_strength', 'turn_trend_str', 'price_trend_str', 'vol_ratio_5', 'book_px_dev', 'bid_vol_5', 'ask_vol_5', 'cumsumvol_mean', 'cumsumvol_std', 'imbalance_strength', 'vol_entropy', 'daily_volume_sum', 'daily_volume_kurt']
raw_factor_list = day_feature + tick_feature
ret_faature = ['label_5', 'label_10', 'label_20', 'label_40', 'label_60']


def standardize_series(s,window=60):

    s = s.copy()
    if s.isna().all():
        return s
    mean = s.rolling(window=window, min_periods=10).mean().shift(1)
    std = s.rolling(window=window, min_periods=10).std().shift(1)
    
    return (s - mean) / (std + 1e-8)

def rolling_winsorize(series, window=60):

    q05 = series.rolling(window=window, min_periods=10).quantile(0.05).shift(1)
    q95 = series.rolling(window=window, min_periods=10).quantile(0.95).shift(1)
        
    series = series.clip(lower=q05,upper=q95)

    return series

def process_factors_separately(df):
    
    df = df.copy()

    for col in raw_factor_list:
        if col in df.columns:
            print(col)
            df[col] = df.groupby(["sym", "date"])[col].transform(rolling_winsorize)
            df[col] = df.groupby(["sym", "date"])[col].transform(standardize_series)

    df = df.groupby('sym', group_keys=False).apply(
        lambda x: x.dropna(subset=raw_factor_list)
    )
    
    return df


if __name__=='__main__':
    work_path = os.path.dirname(__file__)
    data_path = os.path.join(work_path,"data.pkl")
    save_path = os.path.join(work_path,"data_normal.pkl")
    df = pd.read_pickle(data_path)
    data = process_factors_separately(df)
    stock_class = data["sym"].unique() 
    idx_df = []
    for i in stock_class:
        data_select = data[data["sym"]==i].reset_index(drop=True)
        idx_df.append(data_select)
    with open(save_path, "wb") as f:
        pickle.dump(idx_df, f)

    