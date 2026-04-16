import os
import sys
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)
from factor_pool.tick_factor_pool import *

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
    'tick_OBI': {
        'factor_func': tick_Orderbook_Imbalance_single_day,
        'need_cols': ['bid1', 'ask1', 'bsize1', 'asize1']
    },
    'tick_OBI_v2':{
        'factor_func': tick_Orderbook_Imbalance_single_day_v2,
        'need_cols': ['bid1', 'ask1', 'bsize1', 'asize1', 'bid2', 'ask2', 'bsize2', 'asize2']
    }
}

if __name__ == "__main__":
    
    import sys
    import os
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.append(project_root)
    print(sys.path)