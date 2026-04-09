import pandas as pd
import numpy as np

# 转宽表函数
def tick_long_to_wide_(long_df, col):
    """将长表格转换为宽表格
    
    Args:
        long_df: 包含长格式数据的DataFrame，必须包含以下列：
            - 'date': 日期
            - 'sym': 股票代码
            - 'time': 时间
        col: 需要转换的列名，例如 'bid1'、'ask1' 等
        
    Returns:
        包含宽格式数据的DataFrame，索引为 'timestamp'，列为不同股票代码的 col 值
    """
    
    
    if long_df.empty:
        return pd.DataFrame()  # 返回一个空的DataFrame
    
    if not {'date', 'sym'}.issubset(long_df.columns):
        raise ValueError("输入的DataFrame必须包含 'date' 和 'sym' 列")
    
    # 确保 'time' 列存在，如果不存在则创建一个默认的 'time' 列
    if 'time' not in long_df.columns:
        return pd.DataFrame()  # 返回一个空的DataFrame
    
    if 'timestamp' not in long_df.columns:
        # date列是纯数字，例如0，1，2，转化为日期，便于和time合并
        long_df['Ndate'] = pd.to_datetime(long_df['date'].astype(int), unit='D', origin=pd.Timestamp('2020-01-01'))
        long_df['timestamp'] = pd.to_datetime(long_df['Ndate'].dt.strftime('%Y-%m-%d') + ' ' + long_df['time'].astype(str), format='%Y-%m-%d %H:%M:%S')
        
    wide_df = long_df.pivot(index='timestamp', columns=['sym'], values=col)
    
    return wide_df


def safe_divide(numerator, denominator):
    """安全除法函数，避免除以零

    Args:
        numerator: 分子，可以是数值或数组
        denominator: 分母，可以是数值或数组

    Returns:
        除法结果，如果分母为零则返回0
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.true_divide(numerator, denominator)
        result[~np.isfinite(result)] = 0  # 将无穷大和NaN替换为0
    return result