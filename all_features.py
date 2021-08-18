
import math
import numpy as np
import pandas as pd
import pickle as pkl
import ta

from other_utils.get_top2000list import get_universe, get_industry_group, get_all_stock

from alpha_post_processing.alpha_evaluation import iterator_function_ranking

return_file = "/export/scratch/for_kgrag/pv_info/returns"

close_file = "/export/scratch/for_kgrag/pv_info/close"

high_file = "/export/scratch/for_kgrag/pv_info/high"

low_file = "/export/scratch/for_kgrag/pv_info/low"

volume_file = "/export/scratch/for_kgrag/pv_info/volume"

# industry_file = "/export/scratch/for_kgrag/pv_info/subindustry"

return_data = pd.read_csv(return_file, sep = "|")

close_data = pd.read_csv(close_file, sep = "|")

high_data = pd.read_csv(high_file, sep = "|")

low_data = pd.read_csv(low_file, sep = "|")

volume_data = pd.read_csv(volume_file, sep = "|")

# industry_data = pd.read_csv(industry_file, sep = "|")

# print("here")

##### Momentum Indicators

list_top_2000_stocks_complete = get_all_stock()
list_top_2000_stocks_complete = np.insert(list_top_2000_stocks_complete,0, 'Date')

return_data = return_data[list_top_2000_stocks_complete]
close_data = close_data[list_top_2000_stocks_complete]
high_data = high_data[list_top_2000_stocks_complete]
low_data = low_data[list_top_2000_stocks_complete]
volume_data = volume_data[list_top_2000_stocks_complete]

# industry_data = industry_data[list_top_2000_stocks_complete]
industry_data = {}

# print(return_data.shape)

def awesome_oscillator_indicator(gamma_val = 0.8):
    print("awesome oscillator indicator")    
    new_data = {}
    new_data = pd.DataFrame(new_data)
    for col in high_data.columns:
        if col == 'Date':
            continue
        else :
            new_data[col] = ta.momentum.ao(high_data[col], low_data[col])
    print(gamma_val)    
    # return iterator_function(return_data, industry_data, new_data, gamma_val, -1)
    return new_data


# Kaufman’s Adaptive Moving Average indicator
def kama_indicator_return(gamma_val = 0.8):
    print("kama indicator return")
    new_data = {}
    new_data = pd.DataFrame(new_data)
    for col in return_data.columns:
        if col == 'Date':
            continue
        else :
            new_data[col] = ta.momentum.kama(return_data[col], 12, 2, 5, True)
    
    print(gamma_val)
    # return iterator_function(return_data, industry_data, new_data, gamma_val, -1)
    return new_data

# Kaufman’s Adaptive Moving Average indicator
def kama_indicator_close(gamma_val = 0.8):
    print("kama indicator close")
    new_data = {}
    new_data = pd.DataFrame(new_data)
    for col in close_data.columns:
        if col == 'Date':
            continue
        else :
            new_data[col] = ta.momentum.kama(close_data[col], 12, 2, 5, True)
    
    print(gamma_val)
    # return iterator_function(return_data, industry_data, new_data, gamma_val, -1)
   
    return new_data 

    
# Rate of Change indicator
def roc_indicator_return(gamma_val = 0.8):
    print("roc indicator return")
    new_data = {}
    new_data = pd.DataFrame(new_data)
    for col in return_data.columns:
        if col == 'Date':
            continue
        else :
            new_data[col] = ta.momentum.roc(return_data[col], fillna = True)
    
    print(gamma_val)    
    # return iterator_function(return_data, industry_data, new_data, gamma_val, 1)
    return new_data

def roc_indicator_close(gamma_val = 0.8):
    print("roc indicator close")
    new_data = {}
    new_data = pd.DataFrame(new_data)
    for col in close_data.columns:
        if col == 'Date':
            continue
        else :
            new_data[col] = ta.momentum.roc(close_data[col], fillna = True)
    
    print(gamma_val)    
    # return iterator_function(return_data, industry_data, new_data, gamma_val, 1)
    return new_data


# relative strength index signal
def rsi_indicator_return(gamma_val = 0.8):
    print("rsi signal return")
    new_data = {}
    new_data = pd.DataFrame(new_data)
    for col in return_data.columns:
        if col == 'Date':
            continue
        else :
            new_data[col] = ta.momentum.rsi(return_data[col], fillna = True)
    
    print(gamma_val)
    # return iterator_function(return_data, industry_data, new_data, gamma_val, -1)
    return new_data

# relative strength index signal
def rsi_indicator_close(gamma_val = 0.8):
    print("rsi signal close")
    new_data = {}
    new_data = pd.DataFrame(new_data)
    for col in close_data.columns:
        if col == 'Date':
            continue
        else :
            new_data[col] = ta.momentum.rsi(close_data[col], fillna = True)
    
    print(gamma_val)
    # return iterator_function(return_data, industry_data, new_data, gamma_val, -1)
   
    return new_data 

def stochastic_oscillator_indicator(gamma_val = 0.8):
    print("stochastic oscillator indicator")    
    new_data = {}
    new_data = pd.DataFrame(new_data)
    for col in high_data.columns:
        if col == 'Date':
            continue
        else :
            new_data[col] = ta.momentum.stoch(close_data[col], high_data[col], low_data[col])
    print(gamma_val)    
    # return iterator_function(return_data, industry_data, new_data, gamma_val, -1)
    return new_data

def stochastic_oscillator_signal_indicator(gamma_val = 0.8):
    print("stochastic oscillator signal indicator")    
    new_data = {}
    new_data = pd.DataFrame(new_data)
    for col in high_data.columns:
        if col == 'Date':
            continue
        else :
            new_data[col] = ta.momentum.stoch_signal(close_data[col], high_data[col], low_data[col])
    print(gamma_val)    
    # return iterator_function(return_data, industry_data, new_data, gamma_val, -1)
    return new_data

    
# True strength index     
def tsi_indicator_return(gamma_val = 0.8):
    print("tsi indicator return")
    new_data = {}
    new_data = pd.DataFrame(new_data)
    for col in return_data.columns:
        if col == 'Date':
            continue
        else :
            new_data[col] = ta.momentum.tsi(return_data[col], 25, 13, fillna = True)
            
    print(gamma_val)    
    # return iterator_function(return_data, industry_data, new_data, gamma_val, -1)
    return new_data

def tsi_indicator_close(gamma_val = 0.8):
    print("tsi indicator close")
    new_data = {}
    new_data = pd.DataFrame(new_data)
    for col in close_data.columns:
        if col == 'Date':
            continue
        else :
            new_data[col] = ta.momentum.tsi(close_data[col], 25, 13, fillna = True)
    print(gamma_val)    
    # return iterator_function(return_data, industry_data, new_data, gamma_val, -1)
    return new_data

def ultimate_oscillator_signal_indicator(gamma_val = 0.8):
    print("ultimate oscillator signal indicator")    
    new_data = {}
    new_data = pd.DataFrame(new_data)
    for col in high_data.columns:
        if col == 'Date':
            continue
        else :
            new_data[col] = ta.momentum.uo(high_data[col], low_data[col], close_data[col])
    print(gamma_val)    
    # return iterator_function(return_data, industry_data, new_data, gamma_val, -1)
    return new_data


def william_indicator(gamma_val = 0.8):
    print("william signal indicator")    
    new_data = {}
    new_data = pd.DataFrame(new_data)
    for col in high_data.columns:
        if col == 'Date':
            continue
        else :
            new_data[col] = ta.momentum.wr(high_data[col], low_data[col], close_data[col])
    print(gamma_val)    
    # return iterator_function(return_data, industry_data, new_data, gamma_val, -1)
    return new_data

### volume indicator 

def accumulation_distribution_index(gamma_val = 0.8):
    print("accumulation distribution index")    
    new_data = {}
    new_data = pd.DataFrame(new_data)
    for col in high_data.columns:
        if col == 'Date':
            continue
        else :
            new_data[col] = ta.volume.acc_dist_index(high_data[col], low_data[col], close_data[col], volume_data[col])
    print(gamma_val)    
    # return iterator_function(return_data, industry_data, new_data, gamma_val, -1)
    return new_data


def chaikin_money_flow(gamma_val = 0.8):
    print("chaikin money flow")    
    new_data = {}
    new_data = pd.DataFrame(new_data)
    for col in high_data.columns:
        if col == 'Date':
            continue
        else :
            new_data[col] = ta.volume.chaikin_money_flow(high_data[col], low_data[col], close_data[col], volume_data[col])
    print(gamma_val)    
    # return iterator_function(return_data, industry_data, new_data, gamma_val, -1)
    return new_data


def ease_of_movement(gamma_val = 0.8):
    print("ease of movement")    
    new_data = {}
    new_data = pd.DataFrame(new_data)
    for col in high_data.columns:
        if col == 'Date':
            continue
        else :
            new_data[col] = ta.volume.ease_of_movement(high_data[col], low_data[col], volume_data[col])
    print(gamma_val)    
    # return iterator_function(return_data, industry_data, new_data, gamma_val, -1)
    return new_data

def sma_ease_of_movement(gamma_val = 0.8):
    print("sma ease of movement")    
    new_data = {}
    new_data = pd.DataFrame(new_data)
    for col in high_data.columns:
        if col == 'Date':
            continue
        else :
            new_data[col] = ta.volume.sma_ease_of_movement(high_data[col], low_data[col], volume_data[col])
    print(gamma_val)    
    # return iterator_function(return_data, industry_data, new_data, gamma_val, -1)
    return new_data


def force_index_close(gamma_val = 0.8):
    print("force index close")    
    new_data = {}
    new_data = pd.DataFrame(new_data)
    for col in close_data.columns:
        if col == 'Date':
            continue
        else :
            new_data[col] = ta.volume.force_index(close_data[col], volume_data[col])
    print(gamma_val)    
    # return iterator_function(return_data, industry_data, new_data, gamma_val, -1)
    return new_data

def force_index_return(gamma_val = 0.8):
    print("force index return")    
    new_data = {}
    new_data = pd.DataFrame(new_data)
    for col in return_data.columns:
        if col == 'Date':
            continue
        else :
            new_data[col] = ta.volume.force_index(return_data[col], volume_data[col])
    print(gamma_val)    
    # return iterator_function(return_data, industry_data, new_data, gamma_val, -1)
    return new_data

def money_flow_index(gamma_val = 0.8):
    print("money flow index")    
    new_data = {}

    new_data = pd.DataFrame(new_data)
    for col in close_data.columns:
        if col == 'Date':
            continue
        else :
            new_data[col] = ta.volume.money_flow_index(high_data[col], low_data[col], close_data[col], volume_data[col])
    print(gamma_val)    
    # return iterator_function(return_data, industry_data, new_data, gamma_val, -1)
    return new_data

def on_balance_volume_return(gamma_val = 0.8):
    print("on balance volume return")    
    new_data = {}
    new_data = pd.DataFrame(new_data)
    for col in return_data.columns:
        if col == 'Date':
            continue
        else :
            new_data[col] = ta.volume.on_balance_volume(return_data[col], volume_data[col])
    print(gamma_val)    
    # return iterator_function(return_data, industry_data, new_data, gamma_val, -1)
    return new_data


def on_balance_volume_close(gamma_val = 0.8):
    print("on balance volume close")    
    new_data = {}
    new_data = pd.DataFrame(new_data)
    for col in close_data.columns:
        if col == 'Date':
            continue
        else :
            new_data[col] = ta.volume.on_balance_volume(close_data[col], volume_data[col])
    print(gamma_val)    
    # return iterator_function(return_data, industry_data, new_data, gamma_val, -1)
    return new_data



def volume_price_trend_return(gamma_val = 0.8):
    print("volume price trend return")    
    new_data = {}
    new_data = pd.DataFrame(new_data)
    for col in return_data.columns:
        if col == 'Date':
            continue
        else :
            new_data[col] = ta.volume.volume_price_trend(return_data[col], volume_data[col])
    print(gamma_val)    
    # return iterator_function(return_data, industry_data, new_data, gamma_val, -1)
    return new_data


def volume_price_trend_close(gamma_val = 0.8):
    print("volume price trend close")    
    new_data = {}
    new_data = pd.DataFrame(new_data)
    for col in close_data.colummns:
        if col == 'Date':
            continue
        else :
            new_data[col] = ta.volume.volume_price_trend(close_data[col], volume_data[col])
    print(gamma_val)    
    # return iterator_function(return_data, industry_data, new_data, gamma_val, -1)
    return new_data

def volume_weighted_avg_price(gamma_val = 0.8):
    print("volume weighted avg price")    
    new_data = {}
    new_data = pd.DataFrame(new_data)
    for col in close_data.columns:
        if col == 'Date':
            continue
        else :
            new_data[col] = ta.volume.volume_weighted_average_price(high_data[col], low_data[col], close_data[col], volume_data[col])
    print(gamma_val)    
    # return iterator_function(return_data, industry_data, new_data, gamma_val, -1)
    return new_data


#### Trend indicators

#Average Directional Movement Index

def avg_direction_movement_index(gamma_val = 0.8):
    print("avg direction movement index")
    new_data = {}
    # count = 0
    new_data = pd.DataFrame(new_data)
    for col in close_data.columns:
        if col == 'Date':
            continue
        else :
            # print(count)
            # count = count + 1
            new_data[col] = ta.trend.adx(high_data[col], low_data[col], close_data[col], n = 14, fillna = True)

    print(gamma_val)    
    # return iterator_function(return_data, industry_data, new_data, gamma_val, -1)
    return new_data

def aroon_indicator_return(gamma_val = 0.8):
    print("aroon indicator return")
    new_data = {}
    new_data = pd.DataFrame(new_data)
    for col in return_data.columns:
        if col == 'Date':
            continue
        else :
            arr_ind = ta.trend.AroonIndicator(return_data[col])
            new_data[col] = arr_ind.aroon_indicator()

    print(gamma_val)    
    # return iterator_function(return_data, industry_data, new_data, gamma_val, -1)
    return new_data

def aroon_indicator_close(gamma_val = 0.8):
    print("aroon indicator close")
    new_data = {}
    new_data = pd.DataFrame(new_data)
    for col in close_data.columns:
        if col == 'Date':
            continue
        else :
            arr_ind = ta.trend.AroonIndicator(close_data[col])
            new_data[col] = arr_ind.aroon_indicator()            
    print(gamma_val)    
    # return iterator_function(return_data, industry_data, new_data, gamma_val, -1)
    return new_data

def commodity_channel_index(gamma_val = 0.8):
    print("commodity channel indicator")
    new_data = {}
    new_data = pd.DataFrame(new_data)
    for col in close_data.columns:
        if col == 'Date':
            continue
        else :
            new_data[col] = ta.trend.cci(high_data[col],low_data[col], close_data[col])
            
    print(gamma_val)    
    # return iterator_function(return_data, industry_data, new_data, gamma_val, -1)
    return new_data

def dpo_return(gamma_val = 0.8):
    print("dpo return")
    new_data = {}
    new_data = pd.DataFrame(new_data)
    for col in close_data.columns:
        if col == 'Date':
            continue
        else :
            new_data[col] = ta.trend.dpo(return_data[col])
            
    print(gamma_val)    
    # return iterator_function(return_data, industry_data, new_data, gamma_val, -1)
    return new_data

def dpo_close(gamma_val = 0.8):
    print("dpo close")
    new_data = {}
    new_data = pd.DataFrame(new_data)
    for col in close_data.columns:
        if col == 'Date':
            continue
        else :
            new_data[col] = ta.trend.dpo(close_data[col])
            
    print(gamma_val)    
    # return iterator_function(return_data, industry_data, new_data, gamma_val, -1)
    return new_data

def ichimoku_indicator(gamma_val = 0.8):
    print("ichimoku indicator")
    new_data = {}
    new_data = pd.DataFrame(new_data)
    for col in close_data.columns:
        if col == 'Date':
            continue
        else :
            new_data[col] = ta.trend.ichimoku_a(high_data[col],low_data[col])
            
    print(gamma_val)    
    # return iterator_function(return_data, industry_data, new_data, gamma_val, -1)
    return new_data

def kst_indicator_return(gamma_val = 0.8):
    print("kst indicator return")
    new_data = {}
    new_data = pd.DataFrame(new_data)
    for col in close_data.columns:
        if col == 'Date':
            continue
        else :
            arr_ind = ta.trend.KSTIndicator(return_data[col])
            new_data[col] = arr_ind.kst_diff()
            
    print(gamma_val)    
    # return iterator_function(return_data, industry_data, new_data, gamma_val, -1)
    return new_data

def kst_indicator_close(gamma_val = 0.8):
    print("kst indicator close")
    new_data = {}
    new_data = pd.DataFrame(new_data)
    for col in close_data.columns:
        if col == 'Date':
            continue
        else :
            arr_ind = ta.trend.KSTIndicator(close_data[col])
            new_data[col] = arr_ind.kst_diff()
            
    print(gamma_val)    
    # return iterator_function(return_data, industry_data, new_data, gamma_val, -1)
    return new_data

def mass_index_indicator(gamma_val = 0.8):
    print("mass index indicator")
    new_data = {}
    new_data = pd.DataFrame(new_data)
    for col in close_data.columns:
        if col == 'Date':
            continue
        else :
            new_data[col] = ta.trend.mass_index(high_data[col], low_data[col])
            
    print(gamma_val)    
    # return iterator_function(return_data, industry_data, new_data, gamma_val, -1)
    return new_data

def trix_indicator_return(gamma_val = 0.8):
    print("trix indicator return")

    new_data = {}
    new_data = pd.DataFrame(new_data)
    for col in close_data.columns:
        if col == 'Date':
            continue
        else :
            new_data[col] = ta.trend.trix(return_data[col])
            
    print(gamma_val)    
    # return iterator_function(return_data, industry_data, new_data, gamma_val, -1)
    return new_data

def trix_indicator_close(gamma_val = 0.8):
    print("trix indicator close")
    new_data = {}
    new_data = pd.DataFrame(new_data)
    for col in close_data.columns:
        if col == 'Date':
            continue
        else :
            new_data[col] = ta.trend.trix(close_data[col])
            
    print(gamma_val)    
    # return iterator_function(return_data, industry_data, new_data, gamma_val, -1)
    return new_data

def vortex_indicator(gamma_val = 0.8):
    print("vortex indicator close")
    new_data = {}
    new_data = pd.DataFrame(new_data)
    for col in close_data.columns:
        if col == 'Date':
            continue
        else :
            arr_ind = ta.trend.VortexIndicator(high_data[col] , low_data[col], close_data[col])
            new_data[col] = arr_ind.vortex_indicator_diff()
            
    print(gamma_val)    
    # return iterator_function(return_data, industry_data, new_data, gamma_val, -1)
    return new_data

# simple moving average indicator
def sma_indicator_return(gamma_val = 0.8):
    print("Last day avg return")
    new_data = {}
    new_data = pd.DataFrame(new_data)
    for col in return_data.columns:
        if col == 'Date':
            continue
        else :
            new_data[col] = ta.trend.sma_indicator(return_data[col], 5, fillna = True)

    print(gamma_val)    
    return iterator_function_zscore(return_data, industry_data, new_data, gamma_val, -1)
#     return new_data

def sma_indicator_close(gamma_val = 0.8):
    print("Last day avg close")
    new_data = {}
    new_data = pd.DataFrame(new_data)
    for col in close_data.columns:
        if col == 'Date':
            continue
        else :
            new_data[col] = ta.trend.sma_indicator(close_data[col], 5, fillna = True)
    print(gamma_val)    
    # return iterator_function(return_data, industry_data, new_data, gamma_val, 1)
#     iterator_function_zscore(return_data, industry_data, new_data, gamma_val, -1)
    return new_data

# exponential moving avg
def ema_indicator_return(gamma_val = 0.8):
    print("Last day exp avg return")
    new_data = {}
    new_data = pd.DataFrame(new_data)
    for col in return_data.columns:
        if col == 'Date':
            continue
        else :
            new_data[col] = ta.trend.ema_indicator(return_data[col], 6, fillna = True)

#     avg_prices = test_data.rolling(window = window_size , min_periods=1).mean()
    
    print(gamma_val)    
    # return iterator_function(return_data, industry_data, new_data, gamma_val, -1)
#     iterator_function_zscore(return_data, industry_data, new_data, gamma_val, -1)
    return new_data

def ema_indicator_close(gamma_val = 0.8):
    print("Last day exp avg close")
    new_data = {}
    new_data = pd.DataFrame(new_data)
    for col in close_data.columns:
        if col == 'Date':
            continue
        else :
            new_data[col] = ta.trend.ema_indicator(close_data[col], 6, fillna = True)

#     avg_prices = test_data.rolling(window = window_size , min_periods=1).mean()
    
    print(gamma_val)    
    # return iterator_function(return_data, industry_data, new_data, gamma_val, -1)
    return new_data

# difference between simple moving average of two windows
def simple_moving_avg_difference_return(gamma_val = 0.8):
    print("difference simple moving avg return")
    new_data_one = {}
    new_data_one = pd.DataFrame(new_data_one)
    for col in return_data.columns:
        if col == 'Date':
            continue
        else :
            new_data_one[col] = ta.trend.sma_indicator(return_data[col], 1, fillna = True)
    new_data_two = {}
    new_data_two = pd.DataFrame(new_data_two)
    for col in return_data.columns:
        if col == 'Date':
            continue
        else :
            new_data_two[col] = ta.trend.sma_indicator(return_data[col], 5, fillna = True)
    new_data = {}
    new_data = pd.DataFrame(new_data)
    for col in return_data.columns:
        if col == 'Date':
            continue
        else :
            new_data[col] = new_data_one[col] - new_data_two[col]
    
    print(gamma_val)
    # return iterator_function(return_data, industry_data, new_data, gamma_val, -1)
#     iterator_function_zscore(return_data, industry_data, new_data, gamma_val, -1)
    return new_data

def simple_moving_avg_difference_close(gamma_val = 0.8):
    print("difference simple moving avg close")
    new_data_one = {}
    new_data_one = pd.DataFrame(new_data_one)
    for col in close_data.columns:
        if col == 'Date':
            continue
        else :
            new_data_one[col] = ta.trend.sma_indicator(close_data[col], 1, fillna = True)
    new_data_two = {}
    new_data_two = pd.DataFrame(new_data_two)
    for col in close_data.columns:
        if col == 'Date':
            continue
        else :
            new_data_two[col] = ta.trend.sma_indicator(close_data[col], 5, fillna = True)
    new_data = {}
    new_data = pd.DataFrame(new_data)
    for col in close_data.columns:
        if col == 'Date':
            continue
        else :
            new_data[col] = new_data_one[col] - new_data_two[col]
    
    print(gamma_val)
    print(new_data.shape)
    # return iterator_function(return_data, industry_data, new_data, gamma_val, -1) 
    return new_data

# macd diff indicator 
def macd_diff_indicator_return(gamma_val = 0.8):
    print("macd diff return")
    macd_basic = {}
    macd_basic = pd.DataFrame(macd_basic)
    macd_signal = {}
    macd_signal = pd.DataFrame(macd_signal)
    new_data = {}
    new_data = pd.DataFrame(new_data)
    for col in return_data.columns:
        if col == 'Date':
            continue
        else :
            macd_indicator = ta.trend.MACD(return_data[col],26,12,9, fillna = True)
            macd_basic[col] = macd_indicator.macd()
            macd_signal[col] = macd_indicator.macd_signal()
            new_data[col] = macd_indicator.macd_diff()

    print(gamma_val)
#     print("diff")
    # return iterator_function(return_data, industry_data, new_data, gamma_val, -1)
    return new_data

# macd diff indicator 
def macd_diff_indicator_close(gamma_val = 0.8):
    print("macd diff close")
    macd_basic = {}
    macd_basic = pd.DataFrame(macd_basic)
    macd_signal = {}
    macd_signal = pd.DataFrame(macd_signal)
    new_data = {}
    new_data = pd.DataFrame(new_data)
    for col in close_data.columns:
        if col == 'Date':
            continue
        else :
            macd_indicator = ta.trend.MACD(close_data[col],26,12,9, fillna = True)
            macd_basic[col] = macd_indicator.macd()
            macd_signal[col] = macd_indicator.macd_signal()
            new_data[col] = macd_indicator.macd_diff()

    print(gamma_val)
    # return iterator_function(return_data, industry_data, new_data, gamma_val, -1)
   
    return new_data 

#### volatility indicators

def average_true_range_indicator(gamma_val = 0.8):
    print("average true range")
    new_data = {}
    new_data = pd.DataFrame(new_data)
    for col in close_data.columns:
        if col == 'Date':
            continue
        else :
            new_data[col] = ta.volatility.average_true_range(high_data[col], low_data[col], close_data[col])

    print(gamma_val)    
    # return iterator_function(return_data, industry_data, new_data, gamma_val, -1)
    return new_data


def bollinger_indicator_return(gamma_val = 0.8):
    print("bollinger band indicator return")
    new_data = {}
    new_data = pd.DataFrame(new_data)
    new_data_two = {}
    new_data_two = pd.DataFrame(new_data)
    for col in close_data.columns:
        if col == 'Date':
            continue
        else :
            new_data_two[col] = ta.volatility.bollinger_mavg(return_data[col])
            new_data[col] = ta.volatility.bollinger_wband(return_data[col])
            new_data[col] = new_data_two[col]/new_data[col]

    print(gamma_val)    
    # return iterator_function(return_data, industry_data, new_data, gamma_val, -1)
    return new_data


def bollinger_indicator_close(gamma_val = 0.8):
    print("bollinger band indicator close")
    new_data = {}
    new_data = pd.DataFrame(new_data)
    new_data_two = {}
    new_data_two = pd.DataFrame(new_data)
    for col in close_data.columns:
        if col == 'Date':
            continue
        else :
            new_data_two[col] = ta.volatility.bollinger_mavg(close_data[col])
            new_data[col] = ta.volatility.bollinger_wband(close_data[col])
            new_data[col] = new_data_two[col]/new_data[col]

    print(gamma_val)    
    # return iterator_function(return_data, industry_data, new_data, gamma_val, -1)
    return new_data



def donchian_channel(gamma_val = 0.8):
    print("donchian channel")
    new_data = {}
    new_data = pd.DataFrame(new_data)
    new_data_two = {}
    new_data_two = pd.DataFrame(new_data)
    for col in close_data.columns:
        if col == 'Date':
            continue
        else :
            new_data_two[col] = ta.volatility.donchian_channel_mband(high_data[col], low_data[col], close_data[col])
            new_data[col] = ta.volatility.donchian_channel_wband(high_data[col], low_data[col], close_data[col])
            new_data[col] = new_data_two[col]/new_data[col]

    print(gamma_val)    
    # return iterator_function(return_data, industry_data, new_data, gamma_val, -1)
    return new_data


def keltner_channel(gamma_val = 0.8):
    print("keltner channel")
    new_data = {}
    new_data = pd.DataFrame(new_data)
    new_data_two = {}
    new_data_two = pd.DataFrame(new_data)
    for col in close_data.columns:
        if col == 'Date':
            continue
        else :
            new_data_two[col] = ta.volatility.keltner_channel_mband(high_data[col], low_data[col], close_data[col])
            new_data[col] = ta.volatility.keltner_channel_wband(high_data[col], low_data[col], close_data[col])
            new_data[col] = new_data_two[col]/new_data[col]

    print(gamma_val)    
    # return iterator_function(return_data, industry_data, new_data, gamma_val, -1)
   
    return new_data 
# iterator_function_zscore(return_data, industry_data, new_data, gamma_val, -1)
# last_day_avg(5, 0)
# last_day_exp_avg(5, 0.3)
# macd_diff_indicator()
# difference_simple_moving_avg(1,5,0.06)
# rsi_signal(14,a)
# kama_indicator(1)
# roc_indicator(5,1)
# roc_indicator(12,0.0004)
# tsi_indicator(0.6)
# sma_indicator_return(5, 0.8)

