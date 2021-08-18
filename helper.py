import numpy as np
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


def smooth_alpha(map_last_day, map_today, gamma_value):
  for key in map_today:
    if key in map_last_day.keys():
      map_today[key] = gamma_value*map_today[key] + (1 - gamma_value)*map_last_day[key]
  return map_today


def industrial_neutralisation(last_day_dict, industry_group_map):
  for key, value in industry_group_map.items():
    stock_values_list = []
    for stock in value:
      stock_values_list.append(last_day_dict[stock])
    industry_mean = np.mean(stock_values_list)
    for stock in value:
      last_day_dict[stock] = last_day_dict[stock] - industry_mean
  return last_day_dict

# print(industrial_neutralisation({"a" : 1.6, "b" : - 1.8, "c" : 3.6, "d" : -2.4, "e" : 6}, {1:["a","b"], 2 : ["c","d"], -1 : ["e"]}))



# 0.05584981673994589
# 0.5531880281538966
# 0.04303413452640864

def cross_section_rank(map_today):
  count = 0
  for key, value in sorted(map_today.items(), key=lambda item: item[1]):
    map_today[key] = count
    count = count + 1
  return map_today  

# def cross_section_zscore(map_today):
#     stock_value_list = []
#     for key, value in map_today.items():
#         stock_value_list.append(value)
#     mean_stock_value = np.mean(stock_value_list)
#     std_dev_stock = np.std(stock_value_list)
#     # print(mean_stock_value)
#     # print(std_dev_stock)
#     for key, value in map_today.items():
#         map_today[key] = (value - mean_stock_value)/std_dev_stock
#     return map_today

def cross_section_zscore(stock_values):
    mean_value_stock = np.mean(stock_values)
    print(mean_value_stock)
    print(stock_values)
    std_dev_stock = np.std(stock_values)
    print(std_dev_stock)
    stock_values = (stock_values - mean_value_stock)/std_dev_stock
    print(stock_values)
    stock_values = np.clip(stock_values, a_min = -5, a_max = 5)
    return stock_values
      

# print(cross_section_zscore([1,2,3,4,5,6,7]))
# print(cross_section_zscore({"a" : -500, "b" : 4.4, "c" : 4.0, "d" : 6.0, "e" : 500}))

#print(cross_section_rank({"a" : -1.4, "b" : -9000, "c" : 45, "d" : 9.365, "e" : 12.258}))

#print(smooth_alpha({"a" : -4.5, "b" : 4.5, "c" : 6}, {"a" : 4.5, "c" : -2.5}, 0.3))


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
        # count_total = count_total + 1
        last_val = value
        last_key = key 
      else :
        map_today[key] = count_total
        count_total = count_total + 1
        last_val = value
        last_key = key
  return map_today  

# print(cross_section_ranking({"a" : 100.0, "b" : 50.0, "c" : 74.0, "d" : 100.0, "e" : 50.0}))


def func1():
  return 1

def func2():
  return func1()

# lis = [func1, func2]

# for a in lis:
#   shar = a()
#   print(shar)
#   # print(b)



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
    

print(get_maximum_drawdown([1.2,1.4,-1.1,1.6,-1.3,0.6]))