import math
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import time
import csv
from datetime import date, datetime, time, timedelta
from matplotlib import pyplot as plt
from pylab import rcParams
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tqdm import tqdm_notebook
import os

root_dir = '/export/scratch/for_kgrag/target/'


def get_universe(date_val):
    date_val = str(date_val)
    year = date_val[0:4]
    month = date_val[4:6]
    file_name = root_dir + year '/' + month + '/' + 'mltgts_returns_universe.' + date_val 
    df = pd.read_csv(file_name, sep = "|")

    first_col, other_col  = df.iloc[:, 0], df.iloc[:,12]
    final_ans = []
    for i in range(len(first_col)) :
        if str(other_col[i]) == 'nan':
            continue
        else :
            helper_list = other_col[i].split(',')
            if 'top2000' in helper_list:
                final_ans.append(first_col[i])
    return final_ans


def money_neutralisation(sorted_arr):
  sorted_arr = sorted_arr - np.mean(sorted_arr)
  return sorted_arr


def book_size_unification(sorted_arr):
  divisor = np.sum(np.abs(sorted_arr))
  final_val = 2*sorted_arr/divisor
  return final_val


def smooth_alpha(map_last_day, map_today, gamma_value):
  for key in map_today:
    if key in map_last_day.keys():
      map_today[key] = gamma_value*map_today[key] + (1 - gamma_value)*map_last_day[key]
  return map_today
  
      