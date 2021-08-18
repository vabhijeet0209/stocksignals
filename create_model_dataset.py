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
import pickle as pkl
from other_utils.get_top2000list import get_all_stock

target_dir = '/export/scratch/for_kgrag/target/'
list_year = os.listdir(target_dir)

list_top_2000_stocks_complete = get_all_stock()
list_top_2000_stocks_complete = np.insert(list_top_2000_stocks_complete,0, 'Date')

df = pd.DataFrame(columns = list_top_2000_stocks_complete)
print(df)
print(df.shape)

print(list_year)

for year in list_year:
    path_dir = target_dir + year + '/'
    all_months = os.listdir(path_dir)
    for month in all_months:
        path_final = path_dir + month + '/' 
        all_files = os.listdir(path_final)
        for file_name in all_files:
            if 'mltgts_returns_universe.' not in file_name:
                continue
            print(file_name)

