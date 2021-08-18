import numpy as np
import pickle as pkl
import pandas as pd
import math
from other_utils.get_top2000list import get_universe, get_industry_group, get_all_stock
import pickle
import warnings
warnings.filterwarnings('error')

daily_stock_map = {}
daily_industry_map = {}

with open('daily_universe.pickle', 'rb') as handle:
    daily_stock_map = pickle.load(handle)

with open('industry_group.pickle', 'rb') as handle:
    daily_industry_map = pickle.load(handle)

    
# responsible for smoothing alpha values
def smooth_alpha(map_last_day, map_today, gamma_value):
  for key in map_today:
    if key in map_last_day.keys():
      map_today[key] = gamma_value*map_today[key] + (1 - gamma_value)*map_last_day[key]
    else :
      map_today[key] = gamma_value*map_today[key]
  return map_today


# def cross_section_ranking(map_today):
#   count = 0
#   for key, value in sorted(map_today.items(), key=lambda item: item[1]):
#     map_today[key] = count
#     count = count + 1
#   return map_today  


def cross_section_ranking(map_today):
  count_total = 0
  last_val = 0
  last_key = ""
  for key, value in sorted(map_today.items(), key=lambda item: item[1]):
    if count_total == 0:
      map_today[key] = count_total
      last_val = value
      last_key = key
      count_total = count_total + 1
      continue
    else :
      if last_val == value: 
        map_today[key] = map_today[last_key]
        count_total = count_total + 1
        last_val = value
        last_key = key 
      else :
        map_today[key] = count_total
        count_total = count_total + 1
        last_val = value
        last_key = key
  return map_today  


def cross_section_zscore(map_today):
    stock_value_list = []
    for key, value in map_today.items():
        stock_value_list.append(value)
    mean_stock_value = np.mean(stock_value_list)
    std_dev_stock = np.std(stock_value_list)
    for key, value in map_today.items():
        map_today[key] = (value - mean_stock_value)/std_dev_stock
        if map_today[key] > 5:
            map_today[key] = 5
        elif map_today[key] < -5:
            map_today[key] = -5
    return map_today


def industrial_neutralisation(last_day_dict, complete_stock_map, industry_group_map):
  for key, value in industry_group_map.items():
    stock_values_list = []
    for stock in value:
      if str(complete_stock_map[stock]) == 'nan':
        continue
      stock_values_list.append(last_day_dict[stock])
    if len(stock_values_list) == 0:
        continue
    industry_mean = np.mean(stock_values_list)
    for stock in value:
      if str(complete_stock_map[stock]) == 'nan':
        continue
      last_day_dict[stock] = last_day_dict[stock] - industry_mean
  return last_day_dict


#def cross_section_ranking(alpha_val):
#  temp = np.argsort(alpha_val)
#  sorted_values = np.empty_like(temp)
#  sorted_values[temp] = np.arange(len(alpha_val))
#  return sorted_values


def money_neutralisation(sorted_map):
  all_values = list(sorted_map.values())
  mean_val = np.mean(all_values)
  for key in sorted_map:
    sorted_map[key] = sorted_map[key] - mean_val
  return sorted_map



def book_size_unification(sorted_map):
  divisor = np.sum(np.abs(list(sorted_map.values())))
  for key in sorted_map:
    sorted_map[key] = (2*sorted_map[key])/divisor
  return sorted_map

#  divisor = np.sum(np.abs(sorted_arr))
#  final_val = 2*sorted_arr/divisor
#  return final_val

def calculate_profit(index_val, map_today, complete_data):
  profit_val = 0
#   print(complete_data.iloc[index_val, 0], "profit day")
  for key in map_today:
    if str(complete_data[key].iloc[index_val]) == 'nan':
      continue
    else :
      profit_val = profit_val + complete_data[key].iloc[index_val]*map_today[key]
  return profit_val


def calculate_sharpe_ratio(profit_arr):
  mean_profit = np.mean(profit_arr)
  profit_arr = profit_arr - mean_profit
  profit_arr = np.array(profit_arr)
  profit_arr = profit_arr*profit_arr
  denominator = np.mean(profit_arr)
  denominator = np.sqrt(denominator)
  ans = mean_profit/denominator
  return ans

def calculate_turnover(map_last_day, map_today):
  turnover = 0
  sum_alpha_last = 0
  for key in map_today :
    sum_alpha_last = sum_alpha_last + np.abs(map_today[key])
    if key in map_last_day.keys():
      turnover = turnover + np.abs(map_today[key] - map_last_day[key])
    else :
      turnover = turnover + np.abs(map_today[key])

  for key in map_last_day:
    if key in map_today.keys():
      continue
    else :
      turnover = turnover + np.abs(map_last_day[key])

  turnover = turnover/sum_alpha_last
#  print(sum_alpha_last)
  return turnover

def get_maximum_drawdown(profit_value_array):
    profit_value_array = np.array(profit_value_array)
    max_drawdown = 0
    max_value = [profit_value_array[0]]
    for i in range(1, len(profit_value_array)):
        profit_value_array[i] = profit_value_array[i-1] + profit_value_array[i]
        temp_max = np.max([profit_value_array[i], max_value[i-1]])
        max_value.append(temp_max)
        max_drawdown = np.max([temp_max - profit_value_array[i], max_drawdown])
        
    # print(profit_value_array)
    # print(max_value)
    return max_drawdown
    


def get_mean_val(location):
  turnover_arr = pkl.load(open(location, "rb"))
  turnover_mean = np.mean(turnover_arr)
  return turnover_mean



def iterator_function_test(complete_data, complete_industry,avg_prices, gamma_val, factor):
    profit_value_array  = []
    all_top_2000 = []
    turnover_value_array = []

    last_day_dict = {}
    current_day_dict = {}
    last_day_alpha_values = {}
    current_day_alpha_values = {}
#     print(complete_data)
    for index_value in range(1763,2266):
      
#       print(complete_data.iloc[index_value-1,0], "alpha day")
#       top2000_stocks = get_universe(complete_data.iloc[index_value-1,0])
      top2000_stocks = daily_stock_map[complete_data.iloc[index_value-1,0]]

    #  print(len(top2000_stocks))
      all_top_2000.append(len(top2000_stocks))
      final_stocks = []
      stock_values = []
#       print(avg_prices.iloc[index_value-313,0])
#       sorted_data = avg_prices.iloc[index_value-313]
#       print(avg_prices.iloc[index_value-1763, 0], "alpha value day") 
      sorted_data = avg_prices.iloc[index_value-1763]
      industry_group_map = {}
      complete_2000_stocks_map = {}
      for stock in top2000_stocks:
        complete_2000_stocks_map[stock] = sorted_data[stock]
        if str(sorted_data[stock]) == 'nan':
          continue
        else :
          final_stocks.append(stock)
          stock_values.append(sorted_data[stock])

      industry_group_map = daily_industry_map[complete_data.iloc[index_value-1,0]]

    #  final_stocks = final_stocks[1:6]
    #  stock_values = stock_values[1:6]
    #  print(final_stocks)
    #  print(stock_values)

      stock_values = np.array(stock_values)*factor
        
      mean_value_stock = np.mean(stock_values)
      std_dev_stock = np.std(stock_values)
      stock_values = (stock_values - mean_value_stock)/std_dev_stock
      stock_values = np.clip(stock_values, a_min = -5, a_max = 5)
        
    #  print(stock_values)
    #  sorted_values = cross_section_ranking(stock_values)
      profit_1265 = 0
      turnover_val = 0

      if index_value == (1763):

        for j in range(0, len(final_stocks)):
          last_day_dict[final_stocks[j]] = stock_values[j]
          last_day_alpha_values[final_stocks[j]] = stock_values[j]
    #    print(last_day_dict)
    #    print(last_day_alpha_values)
        last_day_dict = cross_section_ranking(last_day_dict)
    #    print(last_day_dict)
        last_day_dict = industrial_neutralisation(last_day_dict, complete_2000_stocks_map, industry_group_map)

        last_day_dict = money_neutralisation(last_day_dict)
    #    print(last_day_dict)
        last_day_dict = book_size_unification(last_day_dict)
    #    print(last_day_dict)
        profit_1265 = calculate_profit(index_value, last_day_dict, complete_data)

      else : 
        current_day_dict = {}
        current_day_alpha_values = {}
        for j in range(0, len(final_stocks)):
          current_day_dict[final_stocks[j]] = stock_values[j]

    #    print(current_day_dict)
        current_day_dict = smooth_alpha(last_day_alpha_values, current_day_dict, gamma_val)
        last_day_alpha_values = current_day_dict
    #    print(last_day_alpha_values)
    #    print(current_day_dict)
        current_day_dict = cross_section_ranking(current_day_dict)
    #    print(current_day_dict)
        current_day_dict = industrial_neutralisation(current_day_dict, complete_2000_stocks_map, industry_group_map)

        current_day_dict = money_neutralisation(current_day_dict)
    #    print(current_day_dict)
        current_day_dict = book_size_unification(current_day_dict)
    #    print(current_day_dict)
        profit_1265 = calculate_profit(index_value, current_day_dict, complete_data)
        turnover_val = calculate_turnover(last_day_dict, current_day_dict)
#         print(turnover_val)
        turnover_value_array.append(turnover_val)
        last_day_dict = current_day_dict

#       print(profit_1265)
      profit_value_array.append(profit_1265)

    avg_turnover = np.mean(turnover_value_array)
    print("Avg number of Universe", np.mean(all_top_2000))
    print("Annualised Return", np.mean(profit_value_array)*250)
    print("Avg Turnover", avg_turnover)
    print("Maximum Drawdown", get_maximum_drawdown(profit_value_array))

    #print(profit_value_array)

    sharpe_ratio = calculate_sharpe_ratio(profit_value_array)
    
    print("Sharpe_ratio", sharpe_ratio)


    profit_value_array = np.array(profit_value_array)
    pkl.dump(profit_value_array, open("/tmp/kritin/profit_ridge_2_test.pkl", "wb"))
    return sharpe_ratio, avg_turnover
#     print(pkl.load(open("/tmp/kritin/turnover_zero.pkl", "rb")))

#     turnover_value_array = np.array(turnover_value_array)
#     pkl.dump(turnover_value_array, open("/tmp/kritin/sma_diff/turnover_industry_diff" + str(gamma_val) + ".pkl", "wb"))
    #print(pkl.load(open("/tmp/kritin/turnover_zero.pkl", "rb")))
    
def iterator_function_train(complete_data, complete_industry,avg_prices, gamma_val, factor):
    profit_value_array  = []
    all_top_2000 = []
    turnover_value_array = []

    last_day_dict = {}
    current_day_dict = {}
    last_day_alpha_values = {}
    current_day_alpha_values = {}
#     print(complete_data)
    for index_value in range(1261,1762):
      
#       print(complete_data.iloc[1755:1765,0], "alpha day")
#       input()
#       top2000_stocks = get_universe(complete_data.iloc[index_value-1,0])
      top2000_stocks = daily_stock_map[complete_data.iloc[index_value-1,0]]

    #  print(len(top2000_stocks))
      all_top_2000.append(len(top2000_stocks))
      final_stocks = []
      stock_values = []
#       print(avg_prices.iloc[index_value-1261,0])
#       sorted_data = avg_prices.iloc[index_value-313]
#       print(avg_prices.iloc[index_value-1261, 0], "alpha value day") 
      sorted_data = avg_prices.iloc[index_value-1261]
      industry_group_map = {}
      complete_2000_stocks_map = {}

      for stock in top2000_stocks:
        complete_2000_stocks_map[stock] = sorted_data[stock]
        if str(sorted_data[stock]) == 'nan':
          continue
        else :
          final_stocks.append(stock)
          stock_values.append(sorted_data[stock])
      industry_group_map = daily_industry_map[complete_data.iloc[index_value-1,0]]
    #  final_stocks = final_stocks[1:6]
    #  stock_values = stock_values[1:6]
    #  print(final_stocks)
    #  print(stock_values)

      stock_values = np.array(stock_values)*factor
        
      mean_value_stock = np.mean(stock_values)
      std_dev_stock = np.std(stock_values)
      stock_values = (stock_values - mean_value_stock)/std_dev_stock
      stock_values = np.clip(stock_values, a_min = -5, a_max = 5)
        
    #  print(stock_values)
    #  sorted_values = cross_section_ranking(stock_values)
      profit_1265 = 0
      turnover_val = 0

      if index_value == (1261):

        for j in range(0, len(final_stocks)):
          last_day_dict[final_stocks[j]] = stock_values[j]
          last_day_alpha_values[final_stocks[j]] = stock_values[j]
    #    print(last_day_dict)
    #    print(last_day_alpha_values)
        last_day_dict = cross_section_ranking(last_day_dict)
    #    print(last_day_dict)
        last_day_dict = industrial_neutralisation(last_day_dict,complete_2000_stocks_map, industry_group_map)

        last_day_dict = money_neutralisation(last_day_dict)
    #    print(last_day_dict)
        last_day_dict = book_size_unification(last_day_dict)
    #    print(last_day_dict)
        profit_1265 = calculate_profit(index_value, last_day_dict, complete_data)

      else : 
        current_day_dict = {}
        current_day_alpha_values = {}
        for j in range(0, len(final_stocks)):
          current_day_dict[final_stocks[j]] = stock_values[j]

    #    print(current_day_dict)
        current_day_dict = smooth_alpha(last_day_alpha_values, current_day_dict, gamma_val)
        last_day_alpha_values = current_day_dict
    #    print(last_day_alpha_values)
    #    print(current_day_dict)
        current_day_dict = cross_section_ranking(current_day_dict)
    #    print(current_day_dict)
        current_day_dict = industrial_neutralisation(current_day_dict, complete_2000_stocks_map,industry_group_map)

        current_day_dict = money_neutralisation(current_day_dict)
    #    print(current_day_dict)
        current_day_dict = book_size_unification(current_day_dict)
    #    print(current_day_dict)
        profit_1265 = calculate_profit(index_value, current_day_dict, complete_data)
        turnover_val = calculate_turnover(last_day_dict, current_day_dict)
#         print(turnover_val)
        turnover_value_array.append(turnover_val)
        last_day_dict = current_day_dict

#       print(profit_1265)
      profit_value_array.append(profit_1265)

    avg_turnover = np.mean(turnover_value_array)
    print("Avg number of Universe", np.mean(all_top_2000))
    print("Annualised Return", np.mean(profit_value_array)*250)
    print("Avg Turnover", avg_turnover)
    print("Maximum Drawdown", get_maximum_drawdown(profit_value_array))

    #print(profit_value_array)

    sharpe_ratio = calculate_sharpe_ratio(profit_value_array)
    
    print("Sharpe_ratio", sharpe_ratio)


    profit_value_array = np.array(profit_value_array)
    pkl.dump(profit_value_array, open("/tmp/kritin/profit_ridge_2_train.pkl", "wb"))
    return sharpe_ratio, avg_turnover
#     print(pkl.load(open("/tmp/kritin/turnover_zero.pkl", "rb")))

#     turnover_value_array = np.array(turnover_value_array)
#     pkl.dump(turnover_value_array, open("/tmp/kritin/sma_diff/turnover_industry_diff" + str(gamma_val) + ".pkl", "wb"))
    #print(pkl.load(open("/tmp/kritin/turnover_zero.pkl", "rb")))
    

date_industry_map = {}

def iterator_function_zscore(complete_data, complete_industry,avg_prices, gamma_val, factor):
    profit_value_array  = []
    all_top_2000 = []
    turnover_value_array = []

    last_day_dict = {}
    current_day_dict = {}
    last_day_alpha_values = {}
    current_day_alpha_values = {}
#     1260
    for index_value in range(1260,2266): 
#       top2000_stocks = get_universe(complete_data.iloc[index_value-1,0])
      top2000_stocks = daily_stock_map[complete_data.iloc[index_value-1,0]]

      all_top_2000.append(len(top2000_stocks))
      final_stocks = []
      stock_values = []
      sorted_data = avg_prices.iloc[index_value-1]

      industry_group_map = {}

#       for stock in top2000_stocks:
#         if str(sorted_data[stock]) == 'nan':
#           continue
#         else :
#           final_stocks.append(stock)
#           stock_values.append(sorted_data[stock])
#       industry_group_map = get_industry_group(complete_industry, final_stocks, index_value)        
      complete_2000_stocks_map = {}
      for stock in top2000_stocks:
        complete_2000_stocks_map[stock] = sorted_data[stock]
        if str(sorted_data[stock]) == 'nan':
          continue
        else :
          final_stocks.append(stock)
          stock_values.append(sorted_data[stock])

      industry_group_map = daily_industry_map[complete_data.iloc[index_value-1,0]]
    
      stock_values = np.array(stock_values)*factor
      profit_1265 = 0
      turnover_val = 0
      if index_value == (1260):
        for j in range(0, len(final_stocks)):
          last_day_dict[final_stocks[j]] = stock_values[j]
          last_day_alpha_values[final_stocks[j]] = stock_values[j]
        last_day_dict = cross_section_ranking(last_day_dict)
        last_day_dict = industrial_neutralisation(last_day_dict, complete_2000_stocks_map,industry_group_map)
        last_day_dict = money_neutralisation(last_day_dict)
        last_day_dict = book_size_unification(last_day_dict)
        profit_1265 = calculate_profit(index_value, last_day_dict, complete_data)
      else : 
        current_day_dict = {}
        current_day_alpha_values = {}
        for j in range(0, len(final_stocks)):
          current_day_dict[final_stocks[j]] = stock_values[j]

        current_day_dict = smooth_alpha(last_day_alpha_values, current_day_dict, gamma_val)
        last_day_alpha_values = current_day_dict
        current_day_dict = cross_section_ranking(current_day_dict)
        current_day_dict = industrial_neutralisation(current_day_dict,complete_2000_stocks_map, industry_group_map)
        current_day_dict = money_neutralisation(current_day_dict)
        current_day_dict = book_size_unification(current_day_dict)
        profit_1265 = calculate_profit(index_value, current_day_dict, complete_data)
        turnover_val = calculate_turnover(last_day_dict, current_day_dict)
        turnover_value_array.append(turnover_val)
        last_day_dict = current_day_dict
      profit_value_array.append(profit_1265)
    print(np.mean(all_top_2000))
    print(np.mean(profit_value_array)*250)
    print(np.mean(turnover_value_array))
    sharpe_ratio = calculate_sharpe_ratio(profit_value_array)
    print(sharpe_ratio)

#def calculate_profits(index_val, sorted_arr, stock_name):
#  profit_val = 0
#  print(complete_data.iloc[index_val, 0])
#  for i in range(0, len(stock_name)) :
#    stock = stock_name[i]
#    if str(complete_data[stock].iloc[index_val]) == 'nan':
#      continue
#    else :
#      profit_val = profit_val + complete_data[stock].iloc[index_val]*sorted_arr[i]
#  return profit_val


#print(get_mean_val("/tmp/kritin/profit.pkl"))

#print(get_mean_val("/tmp/kritin/turnover.pkl"))




#print(calculate_turnover({"a" : 1, "b" : 2}, {"a" : 2,  "v"  : 3}))

