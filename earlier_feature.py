import math
import numpy as np
import pandas as pd
import pickle as pkl
import ta

from get_top2000list import get_universe, get_industry_group, get_all_stock
# from alpha_evaluation import smooth_alpha, money_neutralisation, book_size_unification, industrial_neutralisation
# from alpha_evaluation import calculate_sharpe_ratio, calculate_turnover, get_mean_val, cross_section_ranking, calculate_profit
from alpha_evaluation import iterator_function, iterator_function_zscore

return_file = "/export/scratch/for_kgrag/pv_info/returns"

close_file = "/export/scratch/for_kgrag/pv_info/close"

industry_file = "/export/scratch/for_kgrag/pv_info/subindustry"

return_data = pd.read_csv(return_file, sep = "|")

close_data = pd.read_csv(close_file, sep = "|")

industry_data = pd.read_csv(industry_file, sep = "|")


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

#     avg_prices = test_data.rolling(window = window_size , min_periods=1).mean()
    
    print(gamma_val)    
    return iterator_function(return_data, industry_data, new_data, gamma_val, -1)

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
    return iterator_function(return_data, industry_data, new_data, gamma_val, 1)
#     iterator_function_zscore(return_data, industry_data, new_data, gamma_val, -1)

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
    return iterator_function(return_data, industry_data, new_data, gamma_val, -1)
#     iterator_function_zscore(return_data, industry_data, new_data, gamma_val, -1)


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
    return iterator_function(return_data, industry_data, new_data, gamma_val, -1)

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
    final_ans = {}
    final_ans = pd.DataFrame(final_ans)
    for col in return_data.columns:
        if col == 'Date':
            continue
        else :
            final_ans[col] = new_data_one[col] - new_data_two[col]
    
#     avg_prices = test_data.rolling(window = window_size , min_periods=1).mean()
#     for a in [0.02,0.04,0.06,0.08,0.1]:
#         print(a)
    print(gamma_val)
    return iterator_function(return_data, industry_data, final_ans, gamma_val, -1)
#     iterator_function_zscore(return_data, industry_data, final_ans, gamma_val, -1)

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
    final_ans = {}
    final_ans = pd.DataFrame(final_ans)
    for col in close_data.columns:
        if col == 'Date':
            continue
        else :
            final_ans[col] = new_data_one[col] - new_data_two[col]
    
#     avg_prices = test_data.rolling(window = window_size , min_periods=1).mean()
#     for a in [0.02,0.04,0.06,0.08,0.1]:
#         print(a)
    print(gamma_val)
    return iterator_function(return_data, industry_data, final_ans, gamma_val, -1) 

# macd diff indicator 
def macd_diff_indicator_return(gamma_val = 0.8):
    print("macd diff return")
    macd_basic = {}
    macd_basic = pd.DataFrame(macd_basic)
    macd_signal = {}
    macd_signal = pd.DataFrame(macd_signal)
    macd_diff = {}
    macd_diff = pd.DataFrame(macd_diff)
    for col in return_data.columns:
        if col == 'Date':
            continue
        else :
            macd_indicator = ta.trend.MACD(return_data[col],26,12,9, fillna = True)
            macd_basic[col] = macd_indicator.macd()
            macd_signal[col] = macd_indicator.macd_signal()
            macd_diff[col] = macd_indicator.macd_diff()

    print(gamma_val)
#     print("diff")
    return iterator_function(return_data, industry_data, macd_diff, gamma_val, -1)

# macd diff indicator 
def macd_diff_indicator_close(gamma_val = 0.8):
    print("macd diff close")
    macd_basic = {}
    macd_basic = pd.DataFrame(macd_basic)
    macd_signal = {}
    macd_signal = pd.DataFrame(macd_signal)
    macd_diff = {}
    macd_diff = pd.DataFrame(macd_diff)
    for col in close_data.columns:
        if col == 'Date':
            continue
        else :
            macd_indicator = ta.trend.MACD(close_data[col],26,12,9, fillna = True)
            macd_basic[col] = macd_indicator.macd()
            macd_signal[col] = macd_indicator.macd_signal()
            macd_diff[col] = macd_indicator.macd_diff()

    print(gamma_val)
    return iterator_function(return_data, industry_data, macd_diff, gamma_val, -1)
    
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
    return iterator_function(return_data, industry_data, new_data, gamma_val, -1)

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
    return iterator_function(return_data, industry_data, new_data, gamma_val, -1)

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
    return iterator_function(return_data, industry_data, new_data, gamma_val, -1)

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
    return iterator_function(return_data, industry_data, new_data, gamma_val, -1)
    
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
    return iterator_function(return_data, industry_data, new_data, gamma_val, 1)

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
    return iterator_function(return_data, industry_data, new_data, gamma_val, 1)

# true strength index     
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
    return iterator_function(return_data, industry_data, new_data, gamma_val, -1)

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
    return iterator_function(return_data, industry_data, new_data, gamma_val, -1)

def awesome_oscillator_indicator(gamma_val):
    

list_of_features = [sma_indicator_return, sma_indicator_close]
for feature in list_of_features:
    feature()
    
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