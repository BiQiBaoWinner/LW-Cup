import pandas as pd
import numpy as np
import pickle
import os
from tqdm import tqdm

ret_faature = ['label_5', 'label_10', 'label_20', 'label_40', 'label_60']
day_feature =['mpc_max_1_decay', 'mpc_skew_1_decay', 'logvol_skew_decay', 'logvol_90tail_decay', 'logvol_10tail_decay', 'volroc_skew_decay', 'volroc_kurt_decay', 'vol_entropy_20d_decay', 'vol_maxmean_decay', 'vol_maxstd_decay', 'vol_maxmean_roll_std_decay', 'vol_maxstd_roll_std_decay', 'daily_rtn_decay', 'daily_volatility_decay', 'daily_amplitude_decay', 'daily_volume_mean_decay', 'daily_volume_skew_decay', 'daily_spread_mean_decay', 'daily_imbalance_mean_decay', 'daily_trend_strength_decay', 'daily_price_efficiency_decay', 'realized_vol_decay', 'bipower_var_decay', 'jump_var_decay', 'jump_ratio_decay', 'ret_q05_decay', 'ret_q25_decay', 'ret_q50_decay', 'ret_q75_decay', 'ret_q95_decay', 'ret_range_decay', 'jump_count_decay', 'avg_jump_size_decay', 'pos_jump_size_decay', 'neg_jump_size_decay', 'jump_volatility_decay', 'first_half_return_decay', 'second_half_return_decay', 'intraday_mom_diff_decay', 'large_trade_ratio_decay', 'large_trade_num_ratio_decay', 'active_buy_ratio_decay', 'active_sell_ratio_decay', 'active_imbalance_decay', 'spread_mean_decay', 'spread_median_decay', 'spread_std_decay', 'spread_skew_decay', 'spread_kurt_decay', 'spread_max_decay', 'spread_min_decay', 'amihud_illiquidity_decay', 'volume_autocorr_decay', 'opening_gap_decay', 'volume_weighted_spread_decay', 'volatility_ratio_decay', 'imb_mean_decay', 'imb_std_decay', 'imb_skew_decay', 'imb_kurt_decay', 'high_low_time_ratio_decay']
tick_feature =  ['bid_rate1','bid_rate2','bid_rate3','bid_rate4','bid_rate5','bid_rate6','bid_rate7','bid_rate8','bid_rate9','bid_rate10','ask_rate1','ask_rate2','ask_rate3','ask_rate4','ask_rate5','ask_rate6','ask_rate7','ask_rate8','ask_rate9','ask_rate10','bsize_rate1','bsize_rate2','bsize_rate3','bsize_rate4','bsize_rate5','bsize_rate6','bsize_rate7','bsize_rate8','bsize_rate9','bsize_rate10','asize_rate1','asize_rate2','asize_rate3','asize_rate4','asize_rate5','asize_rate6','asize_rate7','asize_rate8','asize_rate9','asize_rate10','bid_diff1','bid_diff2','bid_diff3','bid_diff4','bid_diff5','bid_diff6','bid_diff7','bid_diff8','bid_diff9','bid_diff10','ask_diff1','ask_diff2','ask_diff3','ask_diff4','ask_diff5','ask_diff6','ask_diff7','ask_diff8','ask_diff9','ask_diff10','lb_intst','la_intst','mb_intst','ma_intst','cb_intst','ca_intst','lb_ind','la_ind','mb_ind','ma_ind','cb_ind','ca_ind','lb_acc','la_acc','mb_acc','ma_acc','cb_acc','ca_acc','imbalance','midprice1','midprice2','midprice3','midprice4','midprice5','midprice6','midprice7','midprice8','midprice9','midprice10','spread1','spread2','spread3','spread4','spread5','spread6','spread7','spread8','spread9','spread10','midprice', 'cumspread','avgbid', 'avgask', 'totalbsize', 'totalasize','bid_mean', 'ask_mean', 'bsize_mean', 'asize_mean','open', 'high', 'low', 'close', 'bid1','bsize1','bid2','bsize2','bid3','bsize3','bid4','bsize4','bid5','bsize5','bid6','bsize6','bid7','bsize7','bid8','bsize8','bid9','bsize9','bid10','bsize10','ask1','asize1','ask2','asize2','ask3','asize3','ask4','asize4','ask5','asize5','ask6','asize6','ask7','asize7','ask8','asize8','ask9','asize9','ask10','asize10','volume_delta','amount_delta','VWOBI_ma5', 'FOBI_ma5', 'TOBI_ma5', 'mpc_1', 'vol_price_sign', 'mom_vol', 'impact_cost', 'vol_price_corr', 'liquid_slope', 'spread_intensity', 'depth_slope', 'bid_entropy', 'ask_entropy', 'bid_gradient', 'ask_gradient', 'bid_convex', 'ask_convex', 'roll_imb_mean5', 'roll_imb_std5', 'mid_accel', 'buy_pressure', 'sell_pressure', 'pressure_ratio', 'vol_imb_int', 'mid_trend', 'vol_ma_ratio', 'spread_slope_5', 'bid_ask_slope_diff', 'vol_price_trend', 'depth_strength', 'vol_adj_impact', 'order_book_skew', 'mid_accel_vol', 'vol_trend', 'bid_top3_ratio', 'ask_top3_ratio', 'book_thickness', 'price_vol_5', 'vol_vol_5', 'short_mom_3', 'mid_mom_10', 'mom_reversal', 'liquid_index', 'bid_depth_dev', 'ask_depth_dev', 'vol_accel', 'px_accel', 'spread_disp', 'vol_price_res', 'bid_pressure_decay', 'ask_pressure_decay', 'rel_book_strength', 'turn_trend_str', 'price_trend_str', 'vol_ratio_5', 'book_px_dev', 'bid_vol_5', 'ask_vol_5', 'cumsumvol_mean', 'cumsumvol_std', 'imbalance_strength', 'vol_entropy', 'daily_volume_sum', 'daily_volume_kurt']
raw_factor_list = day_feature + tick_feature

# raw_factor_list = ['n_close', 'n_open', 'n_high', 'n_low', 'amount_delta', 'n_midprice', 'n_bid1', 'n_bsize1', 'n_bid2', 'n_bsize2', 'n_bid3', 'n_bsize3', 'n_bid4', 'n_bsize4', 'n_bid5', 'n_bsize5', 'n_ask1', 'n_asize1', 'n_ask2', 'n_asize2', 'n_ask3', 'n_asize3', 'n_ask4', 'n_asize4', 'n_ask5', 'n_asize5', 'SMA', 'EMA', 'WMA', 'DEMA', 'TEMA', 'TRIMA', 'KAMA', 'RSI', 'MOM', 'ROC', 'ROCP', 'ROCR', 'ROCR100', 'CMO', 'TRIX', 'TSF', 'MACD', 'MACD_signal', 'MACD_hist', 'HT_TRENDLINE', 'HT_SINE', 'HT_SINE_lead', 'HT_DCPERIOD', 'HT_DCPHASE', 'HT_PHASOR_inphase', 'HT_PHASOR_quadrature', 'HT_TRENDMODE', 'OBV', 'spread', 'bid1_depth', 'ask1_depth', 'bid5_depth', 'ask5_depth', 'ob_imbalance', 'weighted_mid', 'bid_ask_ratio', 'bid_slope', 'ask_slope', 'bid5_mean', 'ask5_mean', 'pressure', 'bid5_range', 'ask5_range', 'bid5_var', 'ask5_var', 'bid5_q25', 'ask5_q25', 'bid5_q50', 'ask5_q50', 'bid5_q75', 'ask5_q75', 'bid1_bid5_ratio', 'ask1_ask5_ratio', 'bid5_size_std', 'ask5_size_std', 'bid5_size_range', 'ask5_size_range', 'bid5_size_q25', 'ask5_size_q25', 'bid5_size_q50', 'ask5_size_q50', 'bid5_size_q75', 'ask5_size_q75', 'bid5_weighted', 'ask5_weighted', 'bid5_size_max', 'bid5_size_min', 'ask5_size_max', 'ask5_size_min', 'roll20_ATR', 'roll20_NATR', 'roll20_TRANGE', 'roll20_ADX', 'roll20_ADXR', 'roll20_PLUS_DI', 'roll20_MINUS_DI', 'roll20_PLUS_DM', 'roll20_MINUS_DM', 'roll20_WILLR', 'roll20_CCI', 'roll20_ULTOSC', 'roll20_MFI', 'roll20_BBANDS_upper', 'roll20_BBANDS_middle', 'roll20_BBANDS_lower', 'roll20_STOCH_slowk', 'roll20_STOCH_slowd', 'roll20_SAR', 'roll20_AROON_DOWN', 'roll20_AROON_UP', 'roll20_AROONOSC', 'roll20_BETA', 'roll20_CORREL', 'roll20_DX', 'roll20_MIN', 'roll20_MAX', 'roll20_MEDPRICE', 'roll20_MIDPRICE', 'roll20_MIDPOINT', 'roll20_TRIMA', 'roll20_AD', 'roll20_ADOSC', 'roll20_OBV', 'roll100_ATR', 'roll100_NATR', 'roll100_TRANGE', 'roll100_ADX', 'roll100_ADXR', 'roll100_PLUS_DI', 'roll100_MINUS_DI', 'roll100_PLUS_DM', 'roll100_MINUS_DM', 'roll100_WILLR', 'roll100_CCI', 'roll100_ULTOSC', 'roll100_MFI', 'roll100_BBANDS_upper', 'roll100_BBANDS_middle', 'roll100_BBANDS_lower', 'roll100_STOCH_slowk', 'roll100_STOCH_slowd', 'roll100_SAR', 'roll100_AROON_DOWN', 'roll100_AROON_UP', 'roll100_AROONOSC', 'roll100_BETA', 'roll100_CORREL', 'roll100_DX', 'roll100_MIN', 'roll100_MAX', 'roll100_MEDPRICE', 'roll100_MIDPRICE', 'roll100_MIDPOINT', 'roll100_TRIMA', 'roll100_AD', 'roll100_ADOSC', 'roll100_OBV']


if __name__=='__main__':
    work_path = os.path.dirname(__file__)
    data_path = os.path.join(work_path,"data_normal.pkl")
    with open(data_path, "rb") as f:
        factor_data = pickle.load(f)

    for target_label in ret_faature:
        # import argparse
        # parser = argparse.ArgumentParser()
        # parser.add_argument('--target_label', type=str, default='label_5')
        # args = parser.parse_args()
        target_label = target_label
        factor_data = [df.dropna(axis=0) for df in tqdm(factor_data)]
        stock_idx = [ i for i in range(len(factor_data))] # number of stocks
        ic_df = pd.DataFrame(columns=stock_idx, index=raw_factor_list)
        print(ic_df.shape)
        for idx in tqdm(stock_idx, desc='stock idx'):
            stock_df = factor_data[idx][raw_factor_list + [target_label]]###0--idx
            stock_df_corr = stock_df.corr()
            print(stock_df_corr.shape)
            ic_df[idx] = stock_df_corr.loc[raw_factor_list,target_label]
        
        cross_factor_ic = ic_df.mean(axis=1)
        cross_factor_ir = ic_df.mean(axis=1)/ic_df.std(axis=1)
        df_corr = pd.concat(factor_data, axis=0).corr(numeric_only=True)
        
        ic_threshold = cross_factor_ic.abs().quantile(0.05)
        ir_threshold = cross_factor_ir.abs().quantile(0.05)
        
        valid_factor_ic = cross_factor_ic[cross_factor_ic.abs() > ic_threshold]
        valid_factor_ir = cross_factor_ir[cross_factor_ir.abs() > ir_threshold]
        
        valid_factor_list = np.intersect1d(valid_factor_ic.index, valid_factor_ir.index).tolist()
        print(f'时序IC&IR筛选后，剩余因子数量：{len(valid_factor_list)}')
        
        cross_factor_ic = cross_factor_ic[valid_factor_list]
        cross_factor_corr = df_corr.loc[valid_factor_list, valid_factor_list]
        
        selected_factor_list = []
        for i in range(100):
            selected_factor = cross_factor_ic.abs().idxmax()  # 找到IC最大的因子
            selected_factor_list.append(selected_factor)  # 存到我要的list里面去
            # 取出已经被选出来的因子和其他因子对应的相关系数
            corr_tem = cross_factor_corr[selected_factor]
            remain_factor = corr_tem.loc[corr_tem.abs() <= 0.7].index.values  # 看看能够保留下来的是哪些因子

            cross_factor_corr = cross_factor_corr.loc[remain_factor, remain_factor]
            cross_factor_ic = cross_factor_ic[remain_factor]

            if cross_factor_ic.shape[0] < 1:
                break
        save_path = os.path.join(work_path,f"selected_factor_{target_label}.pkl")
        print(save_path)
        with open(save_path, 'wb') as f:
            pickle.dump(selected_factor_list, f)
        print(f'{target_label}: 筛选后剩{len(selected_factor_list)}个，是{selected_factor_list}')