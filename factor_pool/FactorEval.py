import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
    
from factor_pool.pipeline import FactorPipeline
from factor_pool.tick_factor_pool import *
from config import FACTORS

class FactorEval:
    def __init__(self, tick_df, target: str = 'label_20'):
        self.tick_df = tick_df
        self.target = target
        
        if isinstance(self.tick_df, pd.DataFrame):
            if 'timestamp' not in self.tick_df.columns:
                self.tick_df['Ndate'] = pd.to_datetime(self.tick_df['date'].astype(int), unit='D', origin=pd.Timestamp('2020-01-01'))
                self.tick_df['timestamp'] = pd.to_datetime(self.tick_df['Ndate'].dt.strftime('%Y-%m-%d') + ' ' + self.tick_df['time'].astype(str), format='%Y-%m-%d %H:%M:%S')
        elif isinstance(self.tick_df, list):
            tmp_list = []
            for df in self.tick_df:
                if 'timestamp' not in df.columns:
                    df['Ndate'] = pd.to_datetime(df['date'].astype(int), unit='D', origin=pd.Timestamp('2020-01-01'))
                    df['timestamp'] = pd.to_datetime(df['Ndate'].dt.strftime('%Y-%m-%d') + ' ' + df['time'].astype(str), format='%Y-%m-%d %H:%M:%S')
                tmp_list.append(df)
            self.tick_df = tmp_list

    def calc_single_factor(self, factor_name):
        if factor_name not in FACTORS:
            raise ValueError(f"Factor '{factor_name}' is not registered in the FACTORS registry.")
        
        pip = FactorPipeline(self.tick_df, date_range=None)
        pip.Tick_Factor_Pool.registry = {
            factor_name: FACTORS[factor_name]
        }
        
        f = pip.load_factor_exposure(n_jobs=16)
        
        return f
    
    def calc_all_factors(self):
        
        if not FACTORS:
            raise ValueError("No factors registered in the FACTORS registry.")
        
        pip = FactorPipeline(self.tick_df, date_range=None)
        print("Now evaluating factors in Pool Registry:", list(FACTORS.keys()))
        pip.Tick_Factor_Pool.registry = FACTORS
        
        f = pip.load_factor_exposure(n_jobs=16)
        
        return f
    
    def eval_single_factor(self, factor_name):
        f = self.calc_single_factor(factor_name)
        ret = self.ReturnColumn()
        ret = ret.reindex(f.index)
        
        wide_f = f.reset_index().pivot(index='timestamp', columns='sym', values=factor_name)
        wide_ret = ret.reset_index().pivot(index='timestamp', columns='sym', values='ret_lead')
        
        # 按资产时间序列计算IC，再按资产维度取平均
        ic_df = pd.Series(index=wide_f.columns, dtype=float)
        for sym in wide_f.columns:
            
            f_ret = pd.DataFrame({
                'factor': wide_f[sym],
                'ret': wide_ret[sym]
            })
            f_ret = f_ret.dropna()
            if len(f_ret) > 0:
                ic_df[sym] = f_ret['factor'].corr(f_ret['ret'])
            else:
                ic_df[sym] = float('nan')
        
        return ic_df.mean(), ic_df.mean() / ic_df.std()  # 返回平均IC和IR
    
    def eval_all_factors(self):
        f = self.calc_all_factors()
        ret = self.ReturnColumn()
        ret = ret.reindex(f.index)
        
        ic_results = {}
        for factor_name in f.columns:
            
            wide_f = f.reset_index().pivot(index='timestamp', columns='sym', values=factor_name)
            wide_ret = ret.reset_index().pivot(index='timestamp', columns='sym', values='ret_lead')
            
            ic_df = pd.Series(index=wide_f.columns, dtype=float)
            for sym in wide_f.columns:
                
                f_ret = pd.DataFrame({
                    'factor': wide_f[sym],
                    'ret': wide_ret[sym]
                })
                f_ret = f_ret.dropna()
                
                if len(f_ret) > 0:
                    ic_df[sym] = f_ret['factor'].corr(f_ret['ret'])
                else:
                    ic_df[sym] = float('nan')
            
            ic_results[factor_name] = (ic_df.mean(), ic_df.mean() / ic_df.std())
        
        f_cor = f.dropna(axis=1, how='all').corr()
        
        Ret_JSON = {
            "IC&IR": {
                factor_name: {
                    'IC': f'{ic_results[factor_name][0]:.4f}',
                    'IR': f'{ic_results[factor_name][1]:.4f}'
                }
                for factor_name in ic_results
            },
            "Factor Correlation Matrix": f_cor.to_dict()
        }
        
        return Ret_JSON  # 返回每个因子的平均IC和IR
    
    def ReturnColumn(self):
        
        if isinstance(self.tick_df, pd.DataFrame):
            if 'midprice' not in self.tick_df.columns:
                raise ValueError("Input DataFrame must contain 'midprice' column.")
            if 'sym' and 'date' and 'timestamp' not in self.tick_df.columns:
                raise ValueError("Input DataFrame must contain 'sym', 'date', and 'timestamp' columns.")
            ret_df = self.tick_df[['sym', 'date', 'timestamp', 'midprice']].copy()
            
        elif isinstance(self.tick_df, list):
            if not all(isinstance(df, pd.DataFrame) for df in self.tick_df):
                raise ValueError("All elements in the input list must be DataFrames.")
            if not all('midprice' in df.columns for df in self.tick_df):
                raise ValueError("All DataFrames in the input list must contain 'midprice' column.")
            if 'sym' and 'date' and 'timestamp' not in self.tick_df[0].columns:
                raise ValueError("Input DataFrame must contain 'sym', 'date', and 'timestamp' columns.")
            
            ret_df = pd.concat(self.tick_df, ignore_index=True)[['sym', 'date', 'timestamp', 'midprice']].copy()
        
        ret_df.loc[:, 'ret_lead'] = ret_df.groupby(['sym', 'date'])['midprice'].shift(-20) / ret_df['midprice'] - 1
        
        ret_df = ret_df.replace([np.inf, -np.inf], np.nan)
        
        return ret_df[['sym', 'timestamp', 'ret_lead']].set_index(['sym', 'timestamp'])
    
if __name__ == "__main__":
    
    from config import results_path
    import pickle
    import json
    
    # with open(os.path.expanduser(f"{results_path}/test_data/test_data.pkl"), 'rb') as f:
    #     tick_df = pickle.load(f)
    tick_df = pd.read_parquet(f"{results_path}/merge_data/merge_data.parquet")
    
    with open(f"{os.path.expanduser(results_path)}/merge_data/null_sym_date_cols.json", 'r') as f:
        null_sym_date_cols = json.load(f)
    
    
    null_date0 = list(null_sym_date_cols['0'].keys()) if '0' in null_sym_date_cols else []
    null_date1 = list(null_sym_date_cols['1'].keys()) if '1' in null_sym_date_cols else []
        
    tick_df = tick_df[((tick_df['sym'] == '0') | (tick_df['sym'] == '1')) & ~((tick_df['date'].isin(null_date0)) | (tick_df['date'].isin(null_date1)))]
    
    factor_eval = FactorEval(tick_df)
    ret = factor_eval.eval_all_factors()
    
    print(ret)