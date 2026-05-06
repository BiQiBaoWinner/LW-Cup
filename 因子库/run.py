import os 
import pandas as pd
import pyarrow
import numpy as np
from scipy.stats import kurtosis,bootstrap

##tick
def imbalance_tick(date_all,window=5):
    date =  date_all.copy()     
    weight = np.array([10,9,8,7,6,5,4,3,2,1])   
    date = date.sort_values(["sym","date","time"])
    date['sum_bw'] = date[[f'bsize{i}' for i in range(1,11)]].mul(weight, axis=1).sum(axis=1)
    date['sum_aw'] = date[[f'asize{i}' for i in range(1,11)]].mul(weight, axis=1).sum(axis=1)
    group = date.groupby(["sym","date"],group_keys=False)
    
    roll_totalbsize = group["totalbsize"].rolling(window=window,min_periods=1).sum().reset_index(level=[0,1],drop=True)
    roll_totalasize = group["totalasize"].rolling(window=window,min_periods=1).sum().reset_index(level=[0,1],drop=True)
    
    roll_bsize1 = group["bsize1"].rolling(window=window,min_periods=1).sum().reset_index(level=[0,1],drop=True)
    roll_asize1 = group["asize1"].rolling(window=window,min_periods=1).sum().reset_index(level=[0,1],drop=True)
        
    roll_bw = group["sum_bw"].rolling(window=window,min_periods=1).sum().reset_index(level=[0,1],drop=True)
    roll_aw = group["sum_aw"].rolling(window=window,min_periods=1).sum().reset_index(level=[0,1],drop=True)
    
    date['VWOBI_ma5'] = (roll_bw - roll_aw) / (roll_bw + roll_aw + 1e-8)                
    date["FOBI_ma5"] = (roll_totalbsize - roll_totalasize)/(roll_totalbsize + roll_totalasize + 1e-8)
    date["TOBI_ma5"] = (roll_bsize1 - roll_asize1)/(roll_bsize1+ roll_asize1 + 1e-8)
    
    del date["sum_bw"],date["sum_aw"]
    
    return date

def calc_mpc(data_all,k):
    data = data_all.copy()
    # data.loc[:,"mid_price"] = (data["bid1"] + data["ask1"]) / 2
    group = data.groupby(["sym","date"],group_keys=False)
    
    def cal_mid_price(serise,k):
        return (serise - serise.shift(k))/serise.shift(k)

    data.loc[:,"mid_price"] = group["midprice1"].apply(cal_mid_price,k)
    data[f"mpc_{k}"] = data["mid_price"].replace([np.inf,-np.inf],np.nan)
    
    del data["mid_price"]
    
    return data

def calc_midprice_trend(data_all, window=5):
    
    data = data_all.copy()
    group = data.groupby(["sym","date"], group_keys=False)
    data["mid_trend"] = group["midprice1"].transform(lambda x: x.rolling(window).mean() - x.rolling(window*2).mean())
    
    return data

def calc_volume_ma_ratio(data_all, window=10):
    
    data = data_all.copy()
    group = data.groupby(["sym","date"], group_keys=False)
    data["vol_ma_ratio"] = data["volume_delta"] / (group["volume_delta"].transform(lambda x: x.rolling(window).mean()) + 1e-8)
    
    return data

def calc_spread_slope_5(data_all):
    
    data = data_all.copy()
    data["spread_slope_5"] = (data["spread5"] - data["spread1"]) / data["midprice1"]
    
    return data

def calc_vol_trend(data_all, window=5):
    
    data = data_all.copy()
    group = data.groupby(["sym","date"], group_keys=False)
    data["vol_trend"] = group["volume_delta"].transform(lambda x: x.rolling(window).mean() - x.rolling(window).mean().shift(1))
    
    return data

def calc_bid_ask_slope_diff(data_all):
    
    data = data_all.copy()
    data["bid_ask_slope_diff"] = data["bid_gradient"] - data["ask_gradient"]
    
    return data

def calc_vol_price_trend(data_all, window=5):
    
    data = data_all.copy()
    group = data.groupby(["sym","date"], group_keys=False)
    px = group["midprice1"].transform(lambda x: x.pct_change(window))
    vol = group["volume_delta"].transform(lambda x: x.rolling(window).mean())
    data["vol_price_trend"] = px * vol
    
    return data

def calc_vol_adj_impact(data_all):
    
    data = data_all.copy()
    data["log_vol"] = np.log1p(data["volume_delta"])
    group = data.groupby(["sym", "date"], group_keys=False)
    data["log_vol"] = group["log_vol"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    data["vol_adj_impact"] = np.abs(data["close"]) / (data["log_vol"] + 1e-8)
    
    del data["log_vol"]
    
    return data

def calc_depth_strength(data_all):
    
    data = data_all.copy()
    data["depth_strength"] = (data["totalbsize"] + data["totalasize"]) / (data["midprice1"] + 1e-8)
    
    return data

def calc_order_book_skew(data_all):
    
    data = data_all.copy()
    b_skew = data[[f"bsize{i}" for i in range(1,11)]].mean(axis=1) / (data[[f"bsize{i}" for i in range(1,11)]].std(axis=1)+1e-8)
    a_skew = data[[f"asize{i}" for i in range(1,11)]].mean(axis=1) / (data[[f"asize{i}" for i in range(1,11)]].std(axis=1)+1e-8)
    data["order_book_skew"] = b_skew - a_skew
    
    return data

def calc_mid_accel_vol(data_all):
    
    data = data_all.copy()
    data["mid_accel_vol"] = data["mid_accel"] * np.log1p(data["volume_delta"])
    
    return data

def calc_skew(x):     
    
    ud = np.nanmean(x)
    sd = np.nanstd(x,ddof=1)
    z = (x - ud) / sd
    n = np.sum(~np.isnan(x))
    skew = np.nansum(z**3)/(n - 1)
    
    return skew

def calc_kurt(x):

    return kurtosis(x, nan_policy='omit', fisher=True)

def imbalance_strength(data_all):
    
    data = data_all.copy()
    data["imbalance_strength"] = (data["totalbsize"] - data["totalasize"]) / (data["totalbsize"] + data["totalasize"] + 1e-8)
    return data
    
def bid_ask_concentration(data_all):
    
    data = data_all.copy()
    data["bid_concentration"] = (data["bsize1"] + data["bsize2"]) / data["totalbsize"]
    data["ask_concentration"] = (data["asize1"] + data["asize2"]) / data["totalasize"]
    
    return data
    
def calc_slope(data_all):
    
    data = data_all.copy()    
    data["spread_slope"] = (data["spread5"] - data["spread1"]) / 4
    
    return data

def calc_gradient(data_all):
    
    data = data_all.copy()
    data["bid_gradient"] = (data["bsize1"] - data["bsize5"]) / (data["bsize1"] + data["bsize5"] + 1e-8)
    data["ask_gradient"] = (data["asize1"] - data["asize5"]) / (data["asize1"] + data["asize5"] + 1e-8)
    
    return data

def calc_ba_ratio(data_all):
    
    data = data_all.copy()
    data["ba_ratio"] = data["totalbsize"] / (data["totalasize"] + 1e-8)
    
    return data

def Order_Book_Dispersion(data_all):

    data = data_all.copy()
    data["bid_spread"] = (data["bsize_max"] - data["bsize_min"]) / (data["bsize_mean"] + 1e-8)
    data["ask_spread"] = (data["asize_max"] - data["asize_min"]) / (data["asize_mean"] + 1e-8)
    
    return data

def calc_order_book_volatility(data_all):
    
    data = data_all.copy()
    data["bid_ask"] = data["bid1"] + data["ask1"]
    
    def roll_stats(x):
        roll = x.rolling(window=10,min_periods=1)    
        return roll.std()/(roll.mean() + 1e-8)
    
    data["order_book_volatility"] = data.groupby(["sym", "date"])["bid_ask"].transform(roll_stats)
    
    del data["bid_ask"]

    return data

def calc_volume_price_features(data_all, window=10):
    data = data_all.copy()
    data = data.sort_values(["sym", "date", "time"]).reset_index(drop=True)

    g = data.groupby(["sym", "date"], group_keys=False)

    # 1. 量价方向同步性
    data["vol_price_sign"] = np.sign(data["close"]) * np.sign(data["volume_delta"])

    # 2. 量能加权动量（趋势强度）
    data["mom_vol"] = data["close"] * data["volume_delta"]

    # 3. 价格冲击成本（流动性）
    data["impact_cost"] = np.abs(data["close"]) * data["volume_delta"]

    # 4. 量价相关系数（配合度）
    def corr_fun(x):
        return x["close"].rolling(window=window, min_periods=1).corr(x["volume_delta"])

    data["vol_price_corr"] = g.apply(corr_fun)

    # 5. 流动性斜率（波动 / 量波动）
    data["price_std"] = g["close"].rolling(window, min_periods=1).std().droplevel([0,1])
    data["vol_std"] = g["volume_delta"].rolling(window, min_periods=1).std().droplevel([0,1])
    data["liquid_slope"] = data["price_std"] / (data["vol_std"] + 1e-8)

    del data["price_std"],data["vol_std"]
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    return data

def calc_spread_intensity(data_all):
    
    data = data_all.copy()
    data["spread_intensity"] = data["spread1"] / (data["midprice1"] + 1e-8)
    
    return data

def calc_depth_slope(data_all):
    
    data = data_all.copy()
    data["depth_slope"] = (data["totalbsize"] - data["totalasize"]) / (data["bid1"] + data["ask1"] + 1e-8)
    
    return data

def calc_bid_entropy(data_all):
    
    data = data_all.copy()
    bsize = data[[f"bsize{i}" for i in range(1,11)]].values
    sum_b = bsize.sum(axis=1, keepdims=True)
    p = np.divide(bsize, sum_b, out=np.zeros_like(bsize), where=sum_b!=0)
    p[p==0] = 1e-12
    data["bid_entropy"] = (-p * np.log2(p)).sum(axis=1)
    
    return data

def calc_ask_entropy(data_all):
    
    data = data_all.copy()
    asize = data[[f"asize{i}" for i in range(1,11)]].values
    sum_b = asize.sum(axis=1, keepdims=True)
    p = np.divide(asize, sum_b, out=np.zeros_like(asize), where=sum_b!=0)
    p[p==0] = 1e-12
    data["ask_entropy"] = (-p * np.log2(p)).sum(axis=1)
    
    return data

def calc_book_convex(data_all):
    
    data = data_all.copy()
    data["bid_convex"] = (data["bsize1"] - data[["bsize6","bsize7","bsize8","bsize9","bsize10"]].mean(axis=1)) / (data["totalbsize"] + 1e-8)
    data["ask_convex"] = (data["asize1"] - data[["asize6","asize7","asize8","asize9","asize10"]].mean(axis=1)) / (data["totalasize"] + 1e-8)
    
    return data

def calc_roll_imbalance(data_all, window=5):
    
    data = data_all.copy()
    data["imbalance_cal"] = (data["totalbsize"] - data["totalasize"]) / (data["totalbsize"] + data["totalasize"] + 1e-8)
    group = data.groupby(["sym","date"], group_keys=False)
    data["roll_imb_mean5"] = group["imbalance"].transform(lambda x: x.rolling(window, min_periods=1).mean())
    data["roll_imb_std5"]  = group["imbalance"].transform(lambda x: x.rolling(window, min_periods=1).std())
    del data["imbalance_cal"]
    
    return data

def calc_mid_accel(data_all, k=1):
    
    data = data_all.copy()
    group = data.groupby(["sym","date"], group_keys=False)
    data["mid_chg"] = group["midprice1"].transform(lambda x: (x-x.shift(k))/(x.shift(k)+1e-8))
    group = data.groupby(["sym","date"], group_keys=False)
    data["mid_accel"] = group["mid_chg"].transform(lambda x:x-x.shift(1))
    
    del data["mid_chg"]
     
    return data

def calc_pressure_ratio(data_all):
    
    data = data_all.copy()
    data["buy_pressure"] = data["bsize1"] / (data["spread1"] + 1e-8)
    data["sell_pressure"] = data["asize1"] / (data["spread1"] + 1e-8)
    data["pressure_ratio"] = data["buy_pressure"] / (data["sell_pressure"] + 1e-8)
    
    return data

def calc_vol_imb_interaction(data_all):
    
    data = data_all.copy()
    imb = (data["totalbsize"] - data["totalasize"]) / (data["totalbsize"] + data["totalasize"] + 1e-8)
    data["vol_imb_int"] = imb * np.log1p(data["volume_delta"])
    
    return data

def calc_bid_top3_concentration(data_all):
    
    data = data_all.copy()
    
    data.loc[:,"top3"] = data[['bsize1','bsize2','bsize3']].sum(axis=1)
    data['bid_top3_ratio'] = data["top3"] / (data['totalbsize'] + 1e-8)
    del data["top3"]
    
    return data


def calc_ask_top3_concentration(data_all):
    
    data = data_all.copy()
    
    data.loc[:,"top3"] = data[['asize1','asize2','asize3']].sum(axis=1)
    data['ask_top3_ratio'] = data["top3"] / (data['totalasize'] + 1e-8)
    del data["top3"]
    
    return data

def calc_book_thickness_ratio(data_all):
    
    data = data_all.copy()
    
    data['book_thickness'] = (data['totalbsize'] + data['totalasize']) / (data['midprice1'] + 1e-8)
    
    return data

def calc_price_volatility(data_all, window=5):
    
    data = data_all.copy()
    
    group = data.groupby(['sym','date'], group_keys=False)
    data['price_vol_5'] = group['close'].rolling(window, min_periods=1).std().reset_index(level=[0,1], drop=True)

    return data

def calc_vol_volatility(data_all, window=5):
    
    data = data_all.copy()
    
    group = data.groupby(['sym','date'], group_keys=False)
    data['vol_vol_5'] = group['volume_delta'].rolling(window, min_periods=1).std().reset_index(level=[0,1], drop=True)
    
    return data

def calc_short_momentum(data_all, window=3):
    
    data = data_all.copy()
    
    group = data.groupby(['sym','date'], group_keys=False)
    data['short_mom_3'] = group['close'].rolling(window, min_periods=1).mean().reset_index(level=[0,1], drop=True)
    
    return data

def calc_mid_momentum(data_all, window=10):
    
    data = data_all.copy()
    
    group = data.groupby(['sym','date'], group_keys=False)
    data['mid_mom_10'] = group['close'].rolling(window, min_periods=1).mean().reset_index(level=[0,1], drop=True)
    
    return data

def calc_momentum_reversal(data_all):
    
    data = data_all.copy()
    
    data['mom_reversal'] = -data['close'] * np.abs(data['volume_delta'])
    
    return data

def calc_liquidity_index(data_all):
    
    data = data_all.copy()
    data['liquid_index'] = (data['totalbsize'] + data['totalasize']) * data['midprice1'] / (np.abs(data['close']) + 1e-8)
    
    return data

def calc_bid_depth_deviation(data_all):
    
    data = data_all.copy()
    
    mean_bid = data[[f'bsize{i}' for i in range(1,11)]].mean(axis=1)
    data['bid_depth_dev'] = (data['bsize1'] - mean_bid) / (mean_bid + 1e-8)
    
    return data

def calc_ask_depth_deviation(data_all):
    
    data = data_all.copy()
    
    mean_ask = data[[f'asize{i}' for i in range(1,11)]].mean(axis=1)
    data['ask_depth_dev'] = (data['asize1'] - mean_ask) / (mean_ask + 1e-8)
    
    return data

def calc_volume_acceleration(data_all):
    
    data = data_all.copy()
    
    group = data.groupby(['sym','date'], group_keys=False)
    vol_ma3 = group['volume_delta'].rolling(3, min_periods=1).mean().reset_index(level=[0,1], drop=True)
    vol_ma6 = group['volume_delta'].rolling(6, min_periods=1).mean().reset_index(level=[0,1], drop=True)
    data['vol_accel'] = vol_ma3 - vol_ma6
    
    return data

def calc_price_acceleration(data_all):
    
    data = data_all.copy()
    
    group = data.groupby(['sym','date'], group_keys=False)
    px_ma3 = group['close'].rolling(3, min_periods=1).mean().reset_index(level=[0,1], drop=True)
    px_ma6 = group['close'].rolling(6, min_periods=1).mean().reset_index(level=[0,1], drop=True)
    data['px_accel'] = px_ma3 - px_ma6
    
    return data

def calc_spread_dispersion(data_all):
    
    data = data_all.copy()
    
    spreads = data[[f'spread{i}' for i in range(1,11)]].values
    data['spread_disp'] = np.std(spreads, axis=1) / (np.mean(spreads, axis=1) + 1e-8)
    
    return data

def calc_vol_price_resonance(data_all, window=5):
    
    data = data_all.copy()
    
    group = data.groupby(['sym','date'], group_keys=False)
    px_norm = group['close'].apply(lambda x: x / (x.shift(1).rolling(window, min_periods=1).mean() + 1e-8))
    vol_norm = group['volume_delta'].apply(lambda x: x / (x.shift(1).rolling(window, min_periods=1).mean() + 1e-8))

    data['vol_price_res'] = px_norm * vol_norm
    
    return data

def calc_bid_pressure_decay(data_all):
    
    data = data_all.copy()
    
    data['bid_pressure_decay'] = (data['bsize1'] - data['bsize10']) / (data['bsize1'] + 1e-8)
    
    return data

def calc_ask_pressure_decay(data_all):
    
    data = data_all.copy()
    
    data['ask_pressure_decay'] = (data['asize1'] - data['asize10']) / (data['asize1'] + 1e-8)
    
    return data

def calc_relative_book_strength(data_all):

    data = data_all.copy()
    
    data['rel_book_strength'] = (data['totalbsize'] / data['totalasize'] - 1) * data['midprice1']

    return data

def calc_turn_trend_strength(data_all, window=5):
    
    data = data_all.copy()
    
    group = data.groupby(['sym','date'], group_keys=False)
    data['turn_trend_str'] = group['volume_delta'].apply(lambda x: x.rolling(window,min_periods=1).sum() / (x.abs().rolling(window,min_periods=1).sum() + 1e-8))#.reset_index(level=[0,1], drop=True)
    
    return data

def calc_price_trend_strength(data_all, window=5):
    
    data = data_all.copy()
    
    group = data.groupby(['sym','date'], group_keys=False)
    data['price_trend_str'] = group['close'].apply(lambda x: x.rolling(window,min_periods=1).sum() / (x.abs().rolling(window,min_periods=1).sum() + 1e-8))#.reset_index(level=[0,1], drop=True)
    
    return data

def calc_vol_ratio(data_all, window=5):
    
    data = data_all.copy()
    
    group = data.groupby(['sym','date'], group_keys=False)
    vol_sum = group['volume_delta'].rolling(window, min_periods=1).sum().reset_index(level=[0,1], drop=True)
    data['vol_ratio_5'] = data['volume_delta'] / (vol_sum + 1e-8)
    
    return data

def calc_book_price_deviation(data_all):
    
    data = data_all.copy()
    
    data['book_px_dev'] = (data['midprice1'] - data['close']) / (data['midprice1'] + 1e-8)
    
    return data

def calc_bid_volatility(data_all, window=5):
    
    data = data_all.copy()
    
    group = data.groupby(['sym','date'], group_keys=False)
    data['bid_vol_5'] = group['bsize1'].rolling(window, min_periods=1).std().reset_index(level=[0,1], drop=True)
    
    return data

def calc_ask_volatility(data_all, window=5):
    
    data = data_all.copy()
    
    group = data.groupby(['sym','date'], group_keys=False)
    data['ask_vol_5'] = group['asize1'].rolling(window, min_periods=1).std().reset_index(level=[0,1], drop=True)
    
    return data


def calc_cumsumvol(data_all):
    
    data = data_all.copy()
    group = data.groupby(["sym","date"])
    data["cumsumvol"] = group["volume_delta"].cumsum()
    
    group = data.groupby(["sym","date"])
    data["cumsumvol_mean"] = group["cumsumvol"].expanding().mean().values
    data["cumsumvol_std"] =  group["cumsumvol"].expanding().std().values

    del data["cumsumvol"]

    return data



##日频
def downsample_tick_to_daily(data_all,k):

    data = data_all.copy()
    g = data.groupby(["sym","date"])
    
    daily = pd.DataFrame()
    daily['sym'] = g['sym'].first()
    daily['date'] = g['date'].first()
    
    daily[f"mpc_max_{k}"] = g[f"mpc_{k}"].apply("max")
    daily[f"mpc_skew_{k}"] = g[f"mpc_{k}"].apply(calc_skew)
    daily = daily.reset_index(drop=True)
    daily['next_date'] = daily.groupby('sym')['date'].shift(-1)
    daily = daily.drop('date', axis=1).rename(columns={'next_date': 'date'})
    return daily

def calc_logvol_skew(data_all):
    
    data = data_all.copy()    
    
    data.loc[:,"logvol"] = np.log(data["volume_delta"] + 1e-8)
    
    g = data.groupby(["sym","date"])
    
    daily = pd.DataFrame()
    daily['sym'] = g['sym'].first()
    daily['date'] = g['date'].first()
    daily["logvol_skew"] = g["logvol"].apply(calc_skew)
    daily = daily.reset_index(drop=True)
    daily['next_date'] = daily.groupby('sym')['date'].shift(-1)
    daily = daily.drop('date', axis=1).rename(columns={'next_date': 'date'})
    print(daily)
    return daily

def calc_logvol_tail(data_all):
    
    def ratio_90tail(g):
        
        vol = g["volume_delta"]
        logvol = g["logvol"]

        q90 = np.nanpercentile(logvol, 90)
        mask = logvol >= q90
        top_vol = vol[mask].sum()
        total_vol = vol.sum()
        return top_vol / (total_vol + 1e-8)
            
    def ratio_10tail(g):
        
        vol = g["volume_delta"]
        logvol = g["logvol"]

        q90 = np.nanpercentile(logvol, 10)
        mask = logvol <= q90
        top_vol = vol[mask].sum()
        total_vol = vol.sum()
        return top_vol / (total_vol + 1e-8)
    
    data = data_all.copy()
    data.loc[:,"logvol"] = np.log(data["volume_delta"] + 1e-8)
    
    g = data.groupby(["sym","date"],group_keys=False)  ##可能报错，要添加参数

    daily = pd.DataFrame()
    daily['sym'] = g['sym'].first()
    daily['date'] = g['date'].first()
    daily["logvol_90tail"] = g.apply(ratio_90tail)
    daily["logvol_10tail"] = g.apply(ratio_10tail)
    daily = daily.reset_index(drop=True)
    daily['next_date'] = daily.groupby('sym')['date'].shift(-1)
    daily = daily.drop('date', axis=1).rename(columns={'next_date': 'date'})
    
    return daily

def calc_Volume_Change(data_all):
    
    data = data_all.copy()
    
    data['dummy_dt'] = pd.to_datetime('2000-01-01 ' + data['time'].astype(str))
    data = data.set_index('dummy_dt')
    g = data.groupby(["sym","date"])
    
    daily = pd.DataFrame()
    daily['sym'] = g['sym'].first()
    daily['date'] = g['date'].first()

    df_15min = g["volume_delta"].resample('15min').sum().reset_index()
    df_15min = df_15min.sort_values(by=['sym', 'date', 'dummy_dt'])
    df_15min = df_15min.reset_index(drop=True)
    
    df_15min['volroc'] = df_15min.groupby(["sym", "date"])['volume_delta'].transform('pct_change')
    df_15min["volroc"] = df_15min["volroc"].replace([np.inf,-np.inf],np.nan)
    # print(df_15min.head(20))
    group = df_15min.groupby(["sym","date"])
    daily["volroc_skew"] = group["volroc"].apply(calc_skew)
    daily["volroc_kurt"]= group["volroc"].apply(calc_kurt)
    daily = daily.reset_index(drop=True)
    daily['next_date'] = daily.groupby('sym')['date'].shift(-1)
    daily = daily.drop('date', axis=1).rename(columns={'next_date': 'date'})

    return daily

def vol_entropy(data_all, n_bins=10,window=20,normalize=True):

    data = data_all.copy()
    g = data.groupby(["sym","date"])

    def calculate_entropy(group_vol):
        
        vol = group_vol.values
        vol_min, vol_max = vol.min(), vol.max()

        if vol_min == vol_max:
            return 1.0 if normalize else np.log2(n_bins)

        bins = np.linspace(vol_min, vol_max, n_bins + 1)
        counts, _ = np.histogram(vol, bins=bins) ##统计tick数量
        
        p = counts[counts > 0] / counts.sum()
        
        entropy = -np.sum(p * np.log2(p))

        if normalize:
            max_ent = np.log2(n_bins)
            entropy = entropy / max_ent
            
        return entropy
    
    daily = g['volume_delta'].apply(calculate_entropy).reset_index(name="vol_entropy")
    
    daily["vol_entropy_20d"] = daily.groupby("sym")["vol_entropy"].rolling(window=window,min_periods=1).std().values##1.这会导致整天数据缺失 2.每只股票100天左右，20天的周期
    daily['next_date'] = daily.groupby('sym')['date'].shift(-1)
    daily = daily.drop('date', axis=1).rename(columns={'next_date': 'date'})

    return daily

def vol_max(data_all,window=15,bootstrap_n=1000,extreme_pct=0.05):
    
    data = data_all.copy()
    
    def get_daily_extreme(vol_series):
        threshold = np.quantile(vol_series, 1 - extreme_pct)
        return vol_series[vol_series >= threshold]

    daily_extreme = data.groupby(['sym', 'date'],)['volume_delta'].apply(get_daily_extreme).reset_index(name="extreme_vol")
    
    def bootstrap_factors(extreme_vol):
        extreme_vol = extreme_vol.values
        if len(extreme_vol) == 0:
            return np.nan, np.nan
        res = bootstrap((extreme_vol,), np.mean, n_resamples=bootstrap_n, random_state=42)

        return res.bootstrap_distribution.mean(), res.bootstrap_distribution.std()

    result = daily_extreme.groupby(["sym","date"])["extreme_vol"].apply(bootstrap_factors)
    result = pd.DataFrame(result.tolist(), index=result.index, columns=["vol_maxmean", "vol_maxstd"]).reset_index()

    result["vol_maxmean_roll_std"] = result.groupby("sym")["vol_maxmean"].transform(
    lambda x: x.rolling(window=window, min_periods=1).std()) 
    
    result["vol_maxstd_roll_std"] = result.groupby("sym")["vol_maxstd"].transform(
    lambda x: x.rolling(window=window, min_periods=1).std())
    
    result['next_date'] = result.groupby('sym')['date'].shift(-1)
    result = result.drop('date', axis=1).rename(columns={'next_date': 'date'})
    
    return result

def aggregate_daily_from_tick(tick_data):

    data = tick_data.copy()
    g = data.groupby(['sym', 'date'])

    daily = pd.DataFrame()
    daily['sym'] = g['sym'].first()
    daily['date'] = g['date'].first()
    
    # 价格类
    daily['daily_rtn'] = g['close'].last()
    daily['daily_volatility'] = g['close'].std()
    daily['daily_amplitude'] = (g['close'].max() - g['close'].min()) / (g['close'].first() + 1e-8)
    
    # 成交量类
    daily['daily_volume_mean'] = g['volume_delta'].mean()
    daily['daily_volume_sum'] = g['volume_delta'].sum()
    daily['daily_volume_skew'] = g['volume_delta'].skew()
    daily['daily_volume_kurt'] = g['volume_delta'].apply(lambda x: x.kurtosis())
    
    # 盘口类
    daily['daily_spread_mean'] = g['spread1'].mean()
    daily['daily_imbalance_mean'] = g['imbalance_strength'].mean()
    
    # 趋势/效率类
    daily['daily_trend_strength'] = g['close'].apply(lambda x: x.sum() / (x.abs().sum() + 1e-8))
    daily['daily_price_efficiency'] = g['close'].apply(lambda x: (x.iloc[-1]-x.iloc[0])/(x.diff().abs().sum()+1e-8))

    daily = daily.reset_index(drop=True)
    
    daily['next_date'] = daily.groupby('sym')['date'].shift(-1)
    daily = daily.drop('date', axis=1).rename(columns={'next_date': 'date'})
    
    return daily

def calc_realized_volatility(data_all):

    data = data_all.copy()
    data['tick_return'] = data.groupby(['sym', 'date'])['close'].transform(
        lambda x: x.diff()
    ).replace([np.inf, -np.inf], np.nan)
    
    def rv_bpv(group):
        returns = group['tick_return'].dropna().values
        n = len(returns)
        if n < 2:
            return pd.Series([np.nan]*4, index=['realized_vol', 'bipower_var', 'jump_var', 'jump_ratio'])
        rv = np.sqrt(np.sum(returns**2))
        bpv = (np.pi / 2) * np.sum(np.abs(returns[1:]) * np.abs(returns[:-1]))
        jump_var = max(rv**2 - bpv, 0)
        jump_ratio = jump_var / (rv**2 + 1e-8) if rv != 0 else np.nan
        return pd.Series([rv, bpv, jump_var, jump_ratio], 
                        index=['realized_vol', 'bipower_var', 'jump_var', 'jump_ratio'])
    
    daily = data.groupby(['sym', 'date']).apply(rv_bpv).reset_index()
    daily['next_date'] = daily.groupby('sym')['date'].shift(-1)
    daily = daily.drop('date', axis=1).rename(columns={'next_date': 'date'})
    return daily

def calc_intraday_return_quantiles(data_all):

    data = data_all.copy()
    data['tick_return'] = data.groupby(['sym', 'date'])['close'].transform(
        lambda x: x.diff()
    ).replace([np.inf, -np.inf], np.nan)
    
    def return_quantiles(group):
        returns = group['tick_return'].dropna()
        if len(returns) == 0:
            return pd.Series([np.nan]*6, index=['ret_q05', 'ret_q25', 'ret_q50', 'ret_q75', 'ret_q95', 'ret_range'])
        q05, q25, q50, q75, q95 = returns.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        return pd.Series([q05, q25, q50, q75, q95, q95 - q05], 
                        index=['ret_q05', 'ret_q25', 'ret_q50', 'ret_q75', 'ret_q95', 'ret_range'])
    
    daily = data.groupby(['sym', 'date']).apply(return_quantiles).reset_index()
    daily['next_date'] = daily.groupby('sym')['date'].shift(-1)
    daily = daily.drop('date', axis=1).rename(columns={'next_date': 'date'})
    return daily

def calc_jump_factors(data_all):

    data = data_all.copy()
    data['tick_return'] = data.groupby(['sym', 'date'])['close'].transform(
        lambda x: x.diff()
    ).replace([np.inf, -np.inf], np.nan)
    
    def jump_detection(group):
        returns = group['tick_return'].dropna().values
        if len(returns) < 2:
            return pd.Series([np.nan]*5, index=['jump_count', 'avg_jump_size', 'pos_jump_size', 'neg_jump_size', 'jump_volatility'])
        mean_ret, std_ret = np.mean(returns), np.std(returns, ddof=1)
        if std_ret == 0:
            return pd.Series([0]*5, index=['jump_count', 'avg_jump_size', 'pos_jump_size', 'neg_jump_size', 'jump_volatility'])
        jump_mask = np.abs(returns - mean_ret) > 3 * std_ret
        jump_returns = returns[jump_mask]
        jump_count = len(jump_returns)
        if jump_count == 0:
            return pd.Series([0]*5, index=['jump_count', 'avg_jump_size', 'pos_jump_size', 'neg_jump_size', 'jump_volatility'])
        pos_jumps = jump_returns[jump_returns > 0]
        neg_jumps = jump_returns[jump_returns < 0]
        return pd.Series([
            jump_count, np.mean(np.abs(jump_returns)),
            np.mean(pos_jumps) if len(pos_jumps)>0 else 0,
            np.mean(np.abs(neg_jumps)) if len(neg_jumps)>0 else 0,
            np.std(jump_returns, ddof=1) if jump_count>1 else 0
        ], index=['jump_count', 'avg_jump_size', 'pos_jump_size', 'neg_jump_size', 'jump_volatility'])
    
    daily = data.groupby(['sym', 'date']).apply(jump_detection).reset_index()
    daily['next_date'] = daily.groupby('sym')['date'].shift(-1)
    daily = daily.drop('date', axis=1).rename(columns={'next_date': 'date'})
    return daily

def calc_intraday_momentum(data_all):

    data = data_all.copy()
    data['time_sec'] = pd.to_timedelta(data['time'].astype(str)).dt.total_seconds()
    
    def intraday_mom(group):
        if len(group) < 2:
            return pd.Series([np.nan]*3, index=['first_half_return', 'second_half_return', 'intraday_mom_diff'])
        min_time, max_time = group['time_sec'].min(), group['time_sec'].max()
        mid_time = (min_time + max_time) / 2
        # 直接用close计算时段收益率（因为close本身是累计值）
        first_half_end = group[group['time_sec'] < mid_time]['close'].iloc[-1] if len(group[group['time_sec'] < mid_time])>0 else 0
        second_half_end = group['close'].iloc[-1]
        first_half_return = first_half_end
        second_half_return = second_half_end - first_half_end
        return pd.Series([first_half_return, second_half_return, first_half_return - second_half_return], 
                        index=['first_half_return', 'second_half_return', 'intraday_mom_diff'])
    
    daily = data.groupby(['sym', 'date']).apply(intraday_mom).reset_index()
    daily['next_date'] = daily.groupby('sym')['date'].shift(-1)
    daily = daily.drop('date', axis=1).rename(columns={'next_date': 'date'})
    return daily


def calc_large_trade_ratio(data_all):

    data = data_all.copy()
    
    def large_trade(group):
        vol = group['volume_delta']
        if len(vol) == 0:
            return pd.Series([np.nan]*2, index=['large_trade_ratio', 'large_trade_num_ratio'])
        threshold = vol.quantile(0.9)
        large_vol = vol[vol >= threshold].sum()
        large_num = (vol >= threshold).sum()
        total_vol, total_num = vol.sum(), len(vol)
        return pd.Series([large_vol/(total_vol+1e-8), large_num/(total_num+1e-8)], 
                        index=['large_trade_ratio', 'large_trade_num_ratio'])
    
    daily = data.groupby(['sym', 'date']).apply(large_trade).reset_index()
    daily['next_date'] = daily.groupby('sym')['date'].shift(-1)
    daily = daily.drop('date', axis=1).rename(columns={'next_date': 'date'})
    return daily

def calc_active_buy_sell_ratio(data_all):

    data = data_all.copy()
    data['midprice_chg'] = data.groupby(['sym', 'date'])['midprice1'].transform(lambda x: x - x.shift(1))
    
    def active_trade(group):
        group = group.dropna(subset=['midprice_chg'])
        if len(group) == 0:
            return pd.Series([np.nan]*3, index=['active_buy_ratio', 'active_sell_ratio', 'active_imbalance'])
        buy_vol = group[group['midprice_chg'] > 0]['volume_delta'].sum()
        sell_vol = group[group['midprice_chg'] < 0]['volume_delta'].sum()
        total_vol = group['volume_delta'].sum()
        return pd.Series([buy_vol/(total_vol+1e-8), sell_vol/(total_vol+1e-8), (buy_vol-sell_vol)/(total_vol+1e-8)], 
                        index=['active_buy_ratio', 'active_sell_ratio', 'active_imbalance'])
    
    daily = data.groupby(['sym', 'date']).apply(active_trade).reset_index()
    daily['next_date'] = daily.groupby('sym')['date'].shift(-1)
    daily = daily.drop('date', axis=1).rename(columns={'next_date': 'date'})
    return daily

def calc_spread_stats(data_all):

    data = data_all.copy()
    
    def spread_stats(group):
        spread = group['spread1']
        if len(spread) == 0:
            return pd.Series([np.nan]*7, index=['spread_mean', 'spread_median', 'spread_std', 'spread_skew', 'spread_kurt', 'spread_max', 'spread_min'])
        return pd.Series([spread.mean(), spread.median(), spread.std(), spread.skew(), spread.kurtosis(), spread.max(), spread.min()], 
                        index=['spread_mean', 'spread_median', 'spread_std', 'spread_skew', 'spread_kurt', 'spread_max', 'spread_min'])
    
    daily = data.groupby(['sym', 'date']).apply(spread_stats).reset_index()
    daily['next_date'] = daily.groupby('sym')['date'].shift(-1)
    daily = daily.drop('date', axis=1).rename(columns={'next_date': 'date'})
    return daily

def calc_amihud_illiquidity(data_all):

    data = data_all.copy()
    
    def amihud(group):
        daily_rtn = group['close'].iloc[-1]  # 正确：close本身就是日收益率
        daily_turnover = group['volume_delta'].sum()  # 正确：累计换手率变化
        return pd.Series([np.abs(daily_rtn)/(daily_turnover+1e-8)], index=['amihud_illiquidity'])
    
    daily = data.groupby(['sym', 'date']).apply(amihud).reset_index()
    daily['next_date'] = daily.groupby('sym')['date'].shift(-1)
    daily = daily.drop('date', axis=1).rename(columns={'next_date': 'date'})
    return daily

def calc_volume_autocorr(data_all):

    data = data_all.copy()
    
    def vol_autocorr(group):
        vol = group['volume_delta'].values
        if len(vol) < 2:
            return pd.Series([np.nan], index=['volume_autocorr'])
        return pd.Series([np.corrcoef(vol[:-1], vol[1:])[0,1]], index=['volume_autocorr'])
    
    daily = data.groupby(['sym', 'date']).apply(vol_autocorr).reset_index()
    daily['next_date'] = daily.groupby('sym')['date'].shift(-1)
    daily = daily.drop('date', axis=1).rename(columns={'next_date': 'date'})
    return daily

def calc_opening_gap(data_all):

    data = data_all.copy()
    
    def opening_gap(group):
        return pd.Series([group['close'].iloc[0]], index=['opening_gap'])
    
    daily = data.groupby(['sym', 'date']).apply(opening_gap).reset_index()
    daily['next_date'] = daily.groupby('sym')['date'].shift(-1)
    daily = daily.drop('date', axis=1).rename(columns={'next_date': 'date'})
    return daily

def calc_volume_weighted_spread(data_all):

    data = data_all.copy()
    
    def vw_spread(group):
        spread, vol = group['spread1'], group['volume_delta'].abs()
        total_vol = vol.sum()
        return pd.Series([(spread * vol).sum()/(total_vol+1e-8)], index=['volume_weighted_spread'])
    
    daily = data.groupby(['sym', 'date']).apply(vw_spread).reset_index()
    daily['next_date'] = daily.groupby('sym')['date'].shift(-1)
    daily = daily.drop('date', axis=1).rename(columns={'next_date': 'date'})
    return daily


def calc_volatility_ratio(data_all):

    data = data_all.copy()
    data['tick_return'] = data.groupby(['sym', 'date'])['close'].transform(
        lambda x: x.diff()
    )
    
    def vol_ratio(group):
        returns = group['tick_return'].dropna()
        if len(returns) == 0:
            return pd.Series([np.nan], index=['volatility_ratio'])
        up_vol = np.sqrt(np.mean(returns[returns>0]**2)) if len(returns[returns>0])>0 else 0
        down_vol = np.sqrt(np.mean(returns[returns<0]**2)) if len(returns[returns<0])>0 else 0
        return pd.Series([up_vol/(down_vol+1e-8)], index=['volatility_ratio'])
    
    daily = data.groupby(['sym', 'date']).apply(vol_ratio).reset_index()
    daily['next_date'] = daily.groupby('sym')['date'].shift(-1)
    daily = daily.drop('date', axis=1).rename(columns={'next_date': 'date'})
    return daily

def calc_order_book_imbalance_stats(data_all):

    data = data_all.copy()
    # 先计算逐tick不平衡度（复用你原有函数）
    data['imbalance'] = (data['totalbsize'] - data['totalasize']) / (data['totalbsize'] + data['totalasize'] + 1e-8)
    
    def imb_stats(group):
        imb = group['imbalance']
        if len(imb) == 0:
            return pd.Series([np.nan]*4, index=['imb_mean', 'imb_std', 'imb_skew', 'imb_kurt'])
        return pd.Series([imb.mean(), imb.std(), imb.skew(), imb.kurtosis()], 
                        index=['imb_mean', 'imb_std', 'imb_skew', 'imb_kurt'])
    
    daily = data.groupby(['sym', 'date']).apply(imb_stats).reset_index()
    daily['next_date'] = daily.groupby('sym')['date'].shift(-1)
    daily = daily.drop('date', axis=1).rename(columns={'next_date': 'date'})
    return daily

def calc_high_low_time_ratio(data_all):

    data = data_all.copy()
    data['time_sec'] = pd.to_timedelta(data['time'].astype(str)).dt.total_seconds()
    
    def high_low_time(group):
        if len(group) < 2:
            return pd.Series([np.nan], index=['high_low_time_ratio'])
        total_time = group['time_sec'].max() - group['time_sec'].min()
        high_idx = group['close'].idxmax()
        low_idx = group['close'].idxmin()
        high_time = group.loc[high_idx, 'time_sec'] - group['time_sec'].min()
        low_time = group.loc[low_idx, 'time_sec'] - group['time_sec'].min()
        return pd.Series([(high_time - low_time)/(total_time+1e-8)], index=['high_low_time_ratio'])
    
    daily = data.groupby(['sym', 'date']).apply(high_low_time).reset_index()
    daily['next_date'] = daily.groupby('sym')['date'].shift(-1)
    daily = daily.drop('date', axis=1).rename(columns={'next_date': 'date'})
    return daily

def calc_SATD_factor(data_all):

    data = data_all.copy()
    
    data['minute'] = data['time'].apply(lambda x: x.strftime("%H:%M"))
    minute_rtn = data.groupby(['sym', 'date', 'minute'])['close'].sum().reset_index(name='minute_rtn')

    minute_rtn['type'] = 'sideways'
    minute_rtn.loc[minute_rtn['minute_rtn'] > 0.0001, 'type'] = 'up'
    minute_rtn.loc[minute_rtn['minute_rtn'] < -0.0001, 'type'] = 'down'
    
    data = data.merge(minute_rtn[['sym', 'date', 'minute', 'type']], 
                      on=['sym', 'date', 'minute'], how='left')
    
    type_stats = data.groupby(['sym', 'date', 'type']).agg(
        type_amount=('amount_delta', 'sum'),
        type_count=('volume_delta', 'count')
    ).reset_index()
    type_stats['ATD_p'] = type_stats['type_amount'] / (type_stats['type_count'] + 1e-8)
    

    daily_stats = data.groupby(['sym', 'date']).agg(
        total_amount=('amount_delta', 'sum'),
        total_count=('volume_delta', 'count')
    ).reset_index()
    daily_stats['ATD_T'] = daily_stats['total_amount'] / (daily_stats['total_count'] + 1e-8)

    daily = type_stats.merge(daily_stats[['sym', 'date', 'ATD_T']], on=['sym', 'date'], how='left')
    daily['SATD'] = daily['ATD_p'] / (daily['ATD_T'] + 1e-8)
    
    daily = daily.pivot_table(
        index=['sym', 'date'],
        columns='type',
        values='SATD'
    ).reset_index()
    daily.rename(columns={
        'up': 'SATD_up',
        'down': 'SATD_down',
        'sideways': 'SATD_sideways'
    }, inplace=True)
    daily[["SATD_up","SATD_down","SATD_sideways"]] = daily[["SATD_up","SATD_down","SATD_sideways"]].fillna(0)
    daily['next_date'] = daily.groupby('sym')['date'].shift(-1)
    daily = daily.drop('date', axis=1).rename(columns={'next_date': 'date'})

    return daily

def merge_prev_day_daily_to_tick(tick_data, daily_data):

    tick = tick_data.copy()
    daily = daily_data.copy()
    
    tick = tick.merge(
        daily,
        on=['sym', 'date'],
        how='left'
    )
    
    return tick

def add_decay_to_prev_day_features(data, lam=5.0):

    data['time_sec'] = pd.to_timedelta(data['time'].astype(str)).dt.total_seconds()
    data['time_progress'] = data['time_sec'] / 23400.0##部分交易日之间间隔不止一天
    
    data['decay'] = np.exp(-lam * data['time_progress'])

    decay_cols = [
        "mpc_max_1",
        "mpc_skew_1",
        "logvol_skew",
        "logvol_90tail",
        "logvol_10tail",
        "volroc_skew",
        "volroc_kurt",
        "vol_entropy_20d",
        "vol_maxmean",
        "vol_maxstd",
        "vol_maxmean_roll_std",
        "vol_maxstd_roll_std",
        'daily_rtn',
        'daily_volatility',
        'daily_amplitude',
        'daily_volume_mean',
        'daily_volume_skew',
        'daily_spread_mean',
        'daily_imbalance_mean',
        'daily_trend_strength',
        'daily_price_efficiency',
        "realized_vol",
        "bipower_var",
        "jump_var",
        "jump_ratio",
        "ret_q05",
        "ret_q25",
        "ret_q50",
        "ret_q75",
        "ret_q95",
        "ret_range",
        "jump_count",
        "avg_jump_size",
        "pos_jump_size",
        "neg_jump_size",
        "jump_volatility",
        "first_half_return",
        "second_half_return",
        "intraday_mom_diff",
        "large_trade_ratio",
        "large_trade_num_ratio",
        "active_buy_ratio",
        "active_sell_ratio",
        "active_imbalance",
        "spread_mean",
        "spread_median",
        "spread_std",
        "spread_skew",
        "spread_kurt",
        "spread_max",
        "spread_min",
        "amihud_illiquidity",
        "volume_autocorr",
        "opening_gap",
        "volume_weighted_spread",
        "volatility_ratio",
        "imb_mean",
        "imb_std",
        "imb_skew",
        "imb_kurt",
        "high_low_time_ratio",

    ]
    
    for c in decay_cols:
        data[f'{c}_decay'] = data[c] * data['decay']
        del data[c]
        
    return data


class tick_feature():
    
    def __init__(self):
        
        self.factor_pool = {
            'TOBI_ma5': imbalance_tick,
            "FOBI_ma5": imbalance_tick,
           "VWOBI_ma5": imbalance_tick,
           "mpc_k":calc_mpc,
           "imbalance_strength":imbalance_strength,
           "bid_concentration":bid_ask_concentration,
           "ask_concentration":bid_ask_concentration,
           "spread_slope":calc_slope,
           "bid_gradient":calc_gradient,
           "ask_gradient":calc_gradient,
           "ba_ratio":calc_ba_ratio,
           "bid_spread":Order_Book_Dispersion,
           "ask_spread":Order_Book_Dispersion,
            "order_book_volatility":calc_order_book_volatility,
            "vol_price_sign": calc_volume_price_features,
            "mom_vol": calc_volume_price_features,
            "impact_cost": calc_volume_price_features,
            "vol_price_corr": calc_volume_price_features,
            "liquid_slope": calc_volume_price_features,
            "spread_intensity": calc_spread_intensity,
            "depth_slope": calc_depth_slope,
            "bid_entropy": calc_bid_entropy,
            "ask_entropy": calc_ask_entropy,
            "bid_convex": calc_book_convex,
            "ask_convex": calc_book_convex,
            "roll_imb_mean5": calc_roll_imbalance,
            "roll_imb_std5": calc_roll_imbalance,
            "mid_accel": calc_mid_accel,
            "buy_pressure": calc_pressure_ratio,
            "sell_pressure": calc_pressure_ratio,
            "pressure_ratio": calc_pressure_ratio,
            "vol_imb_int": calc_vol_imb_interaction,
            "mid_trend": calc_midprice_trend,
            "vol_ma_ratio": calc_volume_ma_ratio,
            "spread_slope_5": calc_spread_slope_5,
            "vol_trend": calc_vol_trend,
            "bid_ask_slope_diff": calc_bid_ask_slope_diff,
            "vol_price_trend": calc_vol_price_trend,
            "depth_strength": calc_depth_strength,
            "vol_adj_impact": calc_vol_adj_impact,
            "order_book_skew": calc_order_book_skew,
            "mid_accel_vol": calc_mid_accel_vol,
            "bid_top3_ratio":calc_bid_top3_concentration,
            "ask_top3_ratio":calc_ask_top3_concentration,
            "book_thickness":calc_book_thickness_ratio,
            "price_vol_5":calc_price_volatility,
            "vol_vol_5":calc_vol_volatility,
            "short_mom_3":calc_short_momentum,
            "mid_mom_10":calc_mid_momentum,
            "mom_reversal":calc_momentum_reversal,
            "liquid_index":calc_liquidity_index,
            'bid_depth_dev': calc_bid_depth_deviation,
            'ask_depth_dev': calc_ask_depth_deviation,
            'vol_accel': calc_volume_acceleration,
            'px_accel': calc_price_acceleration,
            'spread_disp': calc_spread_dispersion,
            'vol_price_res': calc_vol_price_resonance,
            'bid_pressure_decay': calc_bid_pressure_decay,
            'ask_pressure_decay': calc_ask_pressure_decay,
            'rel_book_strength': calc_relative_book_strength,
            'turn_trend_str': calc_turn_trend_strength,
            'price_trend_str': calc_price_trend_strength,
            'vol_ratio_5': calc_vol_ratio,
            'book_px_dev': calc_book_price_deviation,
            'bid_vol_5': calc_bid_volatility,
            'ask_vol_5': calc_ask_volatility,
            'cumsumvol_mean':calc_cumsumvol,
            'cumsumvol_std':calc_cumsumvol,
        #    "cumsumvol":calc_cumsumvol
        }
        
        self.factor_register = {
            'TOBI_ma5': ["bsize1","asize1"],
            "FOBI_ma5": ["bsize1","bsize2","bsize3","bsize4","bsize5","bsize6","bsize7","bsize8","bsize9","bsize10",
                          "asize1","asize2","asize3","asize4","asize5","asize6","asize7","asize8","asize10","asize10"],
            "VWOBI_ma5": ["bsize1","bsize2","bsize3","bsize4","bsize5","bsize6","bsize7","bsize8","bsize9","bsize10",
                          "asize1","asize2","asize3","asize4","asize5","asize6","asize7","asize8","asize10","asize10"],
            "mpc_k":["bid1","ask1","k"],
            "imbalance_strength":["totalbsize","totalasize"],
            "bid_concentration":["totalbsize","bsize1","asize1"],
            "ask_concentration":["totalbsize","bsize1","asize1"],
            "bid_gradient":["bsize1","bsize5","aszie1","asize5"],
            "ask_gradient":["bsize1","bsize5","aszie1","asize5"],
            "spread_slope":["spread5","spread1"],
            "ba_ratio":["totalbsize","totalasize"],
            "bid_spread":["bsize_max","bsize_min","bsize_mean"],
            "ask_spread":["asize_max","asize_min","asize_mean"],
            "order_book_volatility":["ask1","bid1"],
            "vol_price_sign": ["close", "volume_delta"],
            "mom_vol": ["close", "volume_delta"],
            "impact_cost": ["close", "volume_delta"],
            "vol_price_corr": ["close", "volume_delta", "window"],
            "liquid_slope": ["close", "volume_delta", "window"],
            "spread_intensity": ["spread1", "midprice1"],
            "depth_slope": ["totalbsize", "totalasize", "bid1", "ask1"],
            "bid_entropy": ["bsize1","bsize2","bsize3","bsize4","bsize5","bsize6","bsize7","bsize8","bsize9","bsize10"],
            "ask_entropy": ["asize1","asize2","asize3","asize4","asize5","asize6","asize7","asize8","asize9","asize10"],
            "bid_convex": ["bsize1","bsize6","bsize7","bsize8","bsize9","bsize10","totalbsize"],
            "ask_convex": ["asize1","asize6","asize7","asize8","asize9","asize10","totalasize"],
            "roll_imb_mean5": ["totalbsize","totalasize","window"],
            "roll_imb_std5": ["totalbsize","totalasize","window"],
            "mid_accel": ["midprice1","k"],
            "buy_pressure": ["bsize1","spread1"],
            "sell_pressure": ["asize1","spread1"],
            "pressure_ratio": ["bsize1","asize1","spread1"],
            "vol_imb_int": ["totalbsize","totalasize","volume_delta"],
            "mid_trend": ["midprice1","window"],
            "vol_ma_ratio": ["volume_delta","window"],
            "spread_slope_5": ["spread5","spread1","midprice1"],
            "vol_trend": ["volume_delta","window"],
            "bid_ask_slope_diff": ["bid_gradient","ask_gradient"],
            "vol_price_trend": ["midprice1","volume_delta","window"],
            "depth_strength": ["totalbsize","totalasize","midprice1"],
            "vol_adj_impact": ["close","volume_delta"],
            "order_book_skew": ["bsize1","bsize2","bsize3","bsize4","bsize5","bsize6","bsize7","bsize8","bsize9","bsize10",
                          "asize1","asize2","asize3","asize4","asize5","asize6","asize7","asize8","asize10","asize10"],
            "mid_accel_vol": ["mid_accel","volume_delta"],
            "bid_top3_ratio":["bsize1","bsize2","bsize3","totalbsize"],
            "ask_top3_ratio":["asize1","asize2","asize3","totalasize"],
            "book_thickness":["totalbsize","totalasize","midprice1"],
            "price_vol_5":["close"],
            "vol_vol_5":["volume_delta"],
            "short_mom_3":["close"],
            "mid_mom_10":["close"],
            "mom_reversal":["close","volume_delta"],
            "liquid_index":["totalbsize","totalasize","midprice1","close"],
            'bid_depth_dev': [["bsize1","bsize2","bsize3","bsize4","bsize5","bsize6","bsize7","bsize8","bsize9","bsize10"]],
            'ask_depth_dev': ["asize1","asize2","asize3","asize4","asize5","asize6","asize7","asize8","asize9","asize10"],
            'vol_accel': ["volume_delta"],
            'px_accel': ["close"],
            'spread_disp': ["spread1","spread2","spread3","spread4","spread5","spread6","spread7","spread8","spread9","spread10"],
            'vol_price_res': ["close","volume_delta"],
            'bid_pressure_decay': ["bsize1","bsize10"],
            'ask_pressure_decay': ["asize1","asize10"],
            'rel_book_strength': ["totalbsize","totalasize","midprice1"],
            'turn_trend_str': ["volume_delta"],
            'price_trend_str': ["close"],
            'vol_ratio_5': ["volume_delta"],
            'book_px_dev': ["close","midprice1"],
            'bid_vol_5': ["bsize1"],
            'ask_vol_5': ["asize1"],
            'cumsumvol_mean':["volume_delta"],
            'cumsumvol_std':["volume_delta"],

        }
    
    def compute(self, df, window=5, k=1):
    
        df = self.factor_pool['VWOBI_ma5'](df, window=window)
        df = self.factor_pool['mpc_k'](df, k=k)
        df = self.factor_pool["vol_price_sign"](df)
        df = self.factor_pool["spread_intensity"](df)
        df = self.factor_pool["depth_slope"](df)
        df = self.factor_pool["bid_entropy"](df)
        df = self.factor_pool["ask_entropy"](df)
        df = self.factor_pool["bid_gradient"](df)
        df = self.factor_pool["bid_convex"](df)
        df = self.factor_pool["ask_convex"](df)
        df = self.factor_pool["roll_imb_mean5"](df)
        df = self.factor_pool["roll_imb_std5"](df)
        df = self.factor_pool["mid_accel"](df)
        df = self.factor_pool["buy_pressure"](df)
        df = self.factor_pool["sell_pressure"](df)
        df = self.factor_pool["pressure_ratio"](df)
        df = self.factor_pool["vol_imb_int"](df)
        df = self.factor_pool["mid_trend"](df)
        df = self.factor_pool["vol_ma_ratio"](df)
        df = self.factor_pool["spread_slope_5"](df)
        df = self.factor_pool["bid_ask_slope_diff"](df)
        df = self.factor_pool["vol_price_trend"](df)
        df = self.factor_pool["depth_strength"](df)
        df = self.factor_pool["vol_adj_impact"](df)
        df = self.factor_pool["order_book_skew"](df)
        df = self.factor_pool["mid_accel_vol"](df)
        df = self.factor_pool["vol_trend"](df)
        df = self.factor_pool["bid_top3_ratio"](df)
        df = self.factor_pool["ask_top3_ratio"](df)
        df = self.factor_pool["book_thickness"](df)
        df = self.factor_pool["price_vol_5"](df)
        df = self.factor_pool["vol_vol_5"](df)
        df = self.factor_pool["short_mom_3"](df)
        df = self.factor_pool["mid_mom_10"](df)
        df = self.factor_pool["mom_reversal"](df)
        df = self.factor_pool["liquid_index"](df)
        df = self.factor_pool['bid_depth_dev'](df)
        df = self.factor_pool['ask_depth_dev'](df)
        df = self.factor_pool['vol_accel'](df)
        df = self.factor_pool['px_accel'](df)
        df = self.factor_pool['spread_disp'](df)
        df = self.factor_pool['vol_price_res'](df, window=window)
        df = self.factor_pool['bid_pressure_decay'](df)
        df = self.factor_pool['ask_pressure_decay'](df)
        df = self.factor_pool['rel_book_strength'](df)
        df = self.factor_pool['turn_trend_str'](df, window=window)
        df = self.factor_pool['price_trend_str'](df, window=window)
        df = self.factor_pool['vol_ratio_5'](df, window=window)
        df = self.factor_pool['book_px_dev'](df)
        df = self.factor_pool['bid_vol_5'](df, window=window)
        df = self.factor_pool['ask_vol_5'](df, window=window)
        df = self.factor_pool['cumsumvol_mean'](df)
        df = self.factor_pool['imbalance_strength'](df)
        return df
    
class daily_feature():
    
    def __init__(self):
        
        self.factor_pool = {
           "mpc_max_k":downsample_tick_to_daily,
           "mpc_skew_k":downsample_tick_to_daily,
           "logvol_skew":calc_logvol_skew,
           "logvol_90tail":calc_logvol_tail,
           "logvol_10tail":calc_logvol_tail,
           "volroc_skew":calc_Volume_Change,
           "volroc_kurt":calc_Volume_Change,
            "vol_entropy_20d":vol_entropy,
            "vol_maxmean":vol_max,
            "vol_maxstd":vol_max,
            "vol_maxmean_roll_std":vol_max,
            "vol_maxstd_roll_std":vol_max,
            "daily_rtn":aggregate_daily_from_tick,
            "daily_volatility":aggregate_daily_from_tick,
            "daily_amplitude":aggregate_daily_from_tick,
            "daily_volume_mean":aggregate_daily_from_tick,
            "daily_volume_sum":aggregate_daily_from_tick,
            "daily_volume_kurt":aggregate_daily_from_tick,
            "daily_volume_skew":aggregate_daily_from_tick,
            "daily_spread_mean":aggregate_daily_from_tick,
            "daily_imbalance_mean":aggregate_daily_from_tick,
            "daily_trend_strength":aggregate_daily_from_tick,
            "daily_price_efficiency":aggregate_daily_from_tick,
            "realized_vol": calc_realized_volatility,
            "intraday_return_quantiles": calc_intraday_return_quantiles,
            "jump_factors": calc_jump_factors,
            "intraday_momentum": calc_intraday_momentum,
            "large_trade_ratio": calc_large_trade_ratio,
            "active_buy_sell_ratio": calc_active_buy_sell_ratio,
            "spread_stats": calc_spread_stats,
            "amihud_illiquidity": calc_amihud_illiquidity,
            "volume_autocorr": calc_volume_autocorr,
            "opening_gap": calc_opening_gap,
            "volume_weighted_spread": calc_volume_weighted_spread,
            "volatility_ratio": calc_volatility_ratio,
            "order_book_imbalance_stats": calc_order_book_imbalance_stats,
            "high_low_time_ratio": calc_high_low_time_ratio,
            "SATD_up":calc_SATD_factor

        }

        
        self.factor_register = {
            "mpc_max_k":["mpc","k"],
            "mpc_skew_k":["mpc","k"],
            "logvol_skew":["volume_delta"],
            "logvol_90tail":["volume_delta"],
            "logvol_10tail":["volume_delta"],
            "volroc_skew":["volume_delta"],
            "volroc_kurt":["volume_delta"],
            "vol_entropy_20d":["volume_delta"],
            "vol_maxmean":["volume_delta","bootstrap_n","extreme_pct"],
            "vol_maxstd":["volume_delta","bootstrap_n","extreme_pct"],
            "vol_maxmean_roll_std":["volume_delta","window","bootstrap_n","extreme_pct"],
            "vol_maxstd_roll_std":["volume_delta","window","bootstrap_n","extreme_pct"],
            "daily_rtn":["close"],
            "daily_volatility":["close"],
            "daily_amplitude":["close"],
            "daily_volume_mean":["volume_delta"],
            "daily_volume_sum":["volume_delta"],
            "daily_volume_skew":["volume_delta"],
            "daily_volume_kurt":["volume_delta"],
            "daily_spread_mean":["spread1"],
            "daily_imbalance_mean":["imbalance_strength"],
            "daily_trend_strength":["close"],
            "daily_price_efficiency":["close"],
            "realized_vol": ["close"],
            "bipower_var": ["close"],
            "jump_var": ["close"],
            "jump_ratio": ["close"],
            "ret_q05": ["close"],
            "ret_q25": ["close"],
            "ret_q50": ["close"],
            "ret_q75": ["close"],
            "ret_q95": ["close"],
            "ret_range": ["close"],
            "jump_count": ["close"],
            "avg_jump_size": ["close"],
            "pos_jump_size": ["close"],
            "neg_jump_size": ["close"],
            "jump_volatility": ["close"],
            "first_half_return": ["close", "time"],
            "second_half_return": ["close", "time"],
            "intraday_mom_diff": ["close", "time"],
            "large_trade_ratio": ["volume_delta"],
            "large_trade_num_ratio": ["volume_delta"],
            "active_buy_ratio": ["midprice1", "volume_delta"],
            "active_sell_ratio": ["midprice1", "volume_delta"],
            "active_imbalance": ["midprice1", "volume_delta"],
            "spread_mean": ["spread1"],
            "spread_median": ["spread1"],
            "spread_std": ["spread1"],
            "spread_skew": ["spread1"],
            "spread_kurt": ["spread1"],
            "spread_max": ["spread1"],
            "spread_min": ["spread1"],
            "amihud_illiquidity": ["close", "volume_delta"],
            "volume_autocorr": ["volume_delta"],
            "opening_gap": ["close"],
            "volume_weighted_spread": ["spread1", "volume_delta"],
            "volatility_ratio": ["close"],
            "imb_mean": ["totalbsize", "totalasize"],
            "imb_std": ["totalbsize", "totalasize"],
            "imb_skew": ["totalbsize", "totalasize"],
            "imb_kurt": ["totalbsize", "totalasize"],
            "high_low_time_ratio": ["close", "time"],

        }
        
    def compute(self, df, k=1):
        
        data = df.copy()
        g = data.groupby(["sym","date"],group_keys=False)
        daily = pd.DataFrame()
        daily['sym'] = g['sym'].first()
        daily['date'] = g['date'].first()
        daily = daily.reset_index(drop=True)
        df1 = self.factor_pool['mpc_max_k'](df,k)
        daily = daily.merge(df1,on=["sym","date"],how="left")
        df1 = self.factor_pool['logvol_skew'](df)
        daily = daily.merge(df1,on=["sym","date"],how="left")
        df1 = self.factor_pool['logvol_90tail'](df)
        daily = daily.merge(df1,on=["sym","date"],how="left")
        df1 = self.factor_pool['volroc_skew'](df)
        daily = daily.merge(df1,on=["sym","date"],how="left")
        df1 = self.factor_pool['vol_entropy_20d'](df)
        daily = daily.merge(df1,on=["sym","date"],how="left")
        df1 = self.factor_pool['vol_maxmean'](df)  
        daily = daily.merge(df1,on=["sym","date"],how="left")
        df1 = self.factor_pool["daily_rtn"](df)
        daily = daily.merge(df1,on=["sym","date"],how="left")
        df1 = self.factor_pool['realized_vol'](df)
        daily = daily.merge(df1, on=["sym","date"], how="left")
        df1 = self.factor_pool['intraday_return_quantiles'](df)
        daily = daily.merge(df1, on=["sym","date"], how="left")
        df1 = self.factor_pool['jump_factors'](df)
        daily = daily.merge(df1, on=["sym","date"], how="left")
        df1 = self.factor_pool['intraday_momentum'](df)
        daily = daily.merge(df1, on=["sym","date"], how="left")
        df1 = self.factor_pool['large_trade_ratio'](df)
        daily = daily.merge(df1, on=["sym","date"], how="left")
        df1 = self.factor_pool['active_buy_sell_ratio'](df)
        daily = daily.merge(df1, on=["sym","date"], how="left")
        df1 = self.factor_pool['spread_stats'](df)
        daily = daily.merge(df1, on=["sym","date"], how="left")
        df1 = self.factor_pool['amihud_illiquidity'](df)
        daily = daily.merge(df1, on=["sym","date"], how="left")
        df1 = self.factor_pool['volume_autocorr'](df)
        daily = daily.merge(df1, on=["sym","date"], how="left")
        df1 = self.factor_pool['opening_gap'](df)
        daily = daily.merge(df1, on=["sym","date"], how="left")
        df1 = self.factor_pool['volume_weighted_spread'](df)
        daily = daily.merge(df1, on=["sym","date"], how="left")
        df1 = self.factor_pool['volatility_ratio'](df)
        daily = daily.merge(df1, on=["sym","date"], how="left")
        df1 = self.factor_pool['order_book_imbalance_stats'](df)
        daily = daily.merge(df1, on=["sym","date"], how="left")
        df1 = self.factor_pool['high_low_time_ratio'](df)
        daily = daily.merge(df1, on=["sym","date"], how="left")

        data_all = merge_prev_day_daily_to_tick(data,daily)
        data_all = add_decay_to_prev_day_features(data_all)
        return data_all
    
if __name__ == '__main__':
    work_path = os.path.dirname(__file__)
    data_path = os.path.join(work_path,"2026train_set")
    data_list = []
    for path in os.listdir(data_path):
        path = os.path.join(data_path,path)
        data = pd.read_parquet(path)
        data_list.append(data)
        # print(data.columns.tolist())
        # raise
    data_all = pd.concat(data_list,ignore_index=True)
    data_all = data_all.sort_values(by=["sym","date","time"]).reset_index(drop=True)
    tick = tick_feature()
    day = daily_feature()
    data = tick.compute(data_all)
    data = day.compute(data)
    save_path = os.path.join(work_path,"data.pkl")
    data.to_pickle(save_path)
    print("==============")
    print(data.columns.tolist())
    