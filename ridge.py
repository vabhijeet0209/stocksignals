import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import preprocessing, svm 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
import os
from other_utils.get_top2000list import get_all_stock
from alpha_post_processing.alpha_evaluation import iterator_function_test
from sklearn.linear_model import Ridge 
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler



train_dir = '/home/kgrag/first_env/all_stocks/train/'
test_dir = '/home/kgrag/first_env/all_stocks/test/'

# list_features_lr = ['sma_indicator_return', 'rsi_indicator_return']
# , 'tsi_indicator_return', 'ema_indicator_return' ]

# list_features_lr = ['dpo_return',
# 'simple_moving_avg_difference_return',
# 'volume_price_trend_close',
# 'force_index_return',
# 'macd_diff_indicator_return',
# 'simple_moving_avg_difference_close',
# 'sma_indicator_return',
# 'stochastic_oscillator_indicator',
# 'bollinger_indicator_return',
# 'kama_indicator_return',
# 'ultimate_oscillator_signal_indicator',
# 'commodity_channel_index',
# 'roc_indicator_close',
# 'force_index_close',
# 'stochastic_oscillator_signal_indicator',
# 'rsi_indicator_close',
# 'money_flow_index',
# 'vortex_indicator',
# 'chaikin_money_flow',
# 'macd_diff_indicator_close'
# ]


# list_feature_check_nan = ['dpo_return',
# 'simple_moving_avg_difference_return',
# 'volume_price_trend_close',
# 'force_index_return',
# 'macd_diff_indicator_return',
# 'simple_moving_avg_difference_close',
# 'sma_indicator_return',
# 'stochastic_oscillator_indicator',
# 'bollinger_indicator_return',
# 'kama_indicator_return',
# 'ultimate_oscillator_signal_indicator',
# 'commodity_channel_index',
# 'roc_indicator_close',
# 'force_index_close',
# 'stochastic_oscillator_signal_indicator',
# 'rsi_indicator_close',
# 'money_flow_index',
# 'vortex_indicator',
# 'chaikin_money_flow',
# 'macd_diff_indicator_close',
# 'Target']

# list_features_lr = [ 'dpo_return',
# 'rsi_indicator_return',
# 'simple_moving_avg_difference_return',
# 'roc_indicator_return',
# 'ease_of_movement',
# 'volume_price_trend_close',
# 'force_index_return',
# 'macd_diff_indicator_return',
# 'simple_moving_avg_difference_close',
# 'ema_indicator_return',
# 'dpo_close',
# 'sma_indicator_return',
# 'tsi_indicator_return',
# 'stochastic_oscillator_indicator',
# 'william_indicator',
# 'bollinger_indicator_return',
# 'kst_indicator_return',
# 'kama_indicator_return',
# 'ultimate_oscillator_signal_indicator',
# 'commodity_channel_index',
# 'roc_indicator_close',
# 'force_index_close',
# 'trix_indicator_return',
# 'stochastic_oscillator_signal_indicator',
# 'aroon_indicator_return',
# 'rsi_indicator_close',
# 'money_flow_index',
# 'vortex_indicator',
# 'sma_ease_of_movement',
# 'chaikin_money_flow',
# 'aroon_indicator_close',
# 'macd_diff_indicator_close']

# list_feature_check_nan = ['dpo_return',
# 'rsi_indicator_return',
# 'simple_moving_avg_difference_return',
# 'roc_indicator_return',
# 'ease_of_movement',
# 'volume_price_trend_close',
# 'force_index_return',
# 'macd_diff_indicator_return',
# 'simple_moving_avg_difference_close',
# 'ema_indicator_return',
# 'dpo_close',
# 'sma_indicator_return',
# 'tsi_indicator_return',
# 'stochastic_oscillator_indicator',
# 'william_indicator',
# 'bollinger_indicator_return',
# 'kst_indicator_return',
# 'kama_indicator_return',
# 'ultimate_oscillator_signal_indicator',
# 'commodity_channel_index',
# 'roc_indicator_close',
# 'force_index_close',
# 'trix_indicator_return',
# 'stochastic_oscillator_signal_indicator',
# 'aroon_indicator_return',
# 'rsi_indicator_close',
# 'money_flow_index',
# 'vortex_indicator',
# 'sma_ease_of_movement',
# 'chaikin_money_flow',
# 'aroon_indicator_close',
# 'macd_diff_indicator_close', 'Target']



list_features_lr = ['sma_indicator_return', 'rsi_indicator_return', 'simple_moving_avg_difference_close', 'ease_of_movement', 'ultimate_oscillator_signal_indicator', 'william_indicator', 'tsi_indicator_return', 'ema_indicator_return', 'macd_diff_indicator_return', 'bollinger_indicator_return']

list_feature_check_nan = ['sma_indicator_return', 'rsi_indicator_return', 'simple_moving_avg_difference_close', 'ease_of_movement', 'ultimate_oscillator_signal_indicator', 'william_indicator', 'tsi_indicator_return', 'ema_indicator_return', 'macd_diff_indicator_return', 'bollinger_indicator_return', 'Target']

list_top_2000_stocks_complete = get_all_stock()
list_top_2000_stocks_complete = np.insert(list_top_2000_stocks_complete,0, 'Date')


all_created_files = os.listdir(test_dir)
print(len(all_created_files))



helper_test_data = pd.read_csv(test_dir + 'EQ0010043000001000' + '.csv', sep = ",")

helper_train_data = pd.read_csv(train_dir + 'EQ0010043000001000' + '.csv', sep = ",")

# print(helper_train_data.iloc[942:950,0])

index_list = np.arange(948, 1449) - 948
values = np.array(helper_train_data['Date'].iloc[948:1449])

sharpe_calculate_pandas_train = pd.DataFrame(index = index_list, columns = list_top_2000_stocks_complete) 
sharpe_calculate_pandas_train['Date'].iloc[:] = values



sharpe_calculate_pandas_test = pd.DataFrame(index = helper_test_data.index, columns = list_top_2000_stocks_complete)
sharpe_calculate_pandas_test['Date'] = helper_test_data['Date']



for each_file in all_created_files:
# for each_file in ['EQ0000000000132369.csv']:
    column_val = each_file[:-4]
    print(column_val)
    complete_training_data = pd.read_csv(train_dir + column_val + '.csv', sep = ",")
    complete_training_data = complete_training_data.replace([np.inf, -np.inf], np.nan)
#     print(complete_training_data[list_feature_check_nan])
    temp_training_data = complete_training_data.iloc[948:1449]

    complete_training_data = complete_training_data.dropna(subset = list_feature_check_nan)
#     print(complete_training_data.shape)
    complete_test_data = pd.read_csv(test_dir + column_val + '.csv', sep = ",")
    complete_test_data = complete_test_data.replace([np.inf, -np.inf], np.nan)
    
    if complete_training_data.shape[0] == 0:
        continue
    
    scaler_train = StandardScaler()
    scaler_train_second = StandardScaler()
    scaler_test = StandardScaler()

    X_train = complete_training_data[list_features_lr]
    temp_training_data = temp_training_data[list_features_lr]
    

    X_test = complete_test_data[list_features_lr]
    
    Y_train = np.array(complete_training_data['Target']).reshape(-1,1)
    
#     print(X_test)

    X_train = scaler_train.fit_transform(X_train)
    temp_training_data = scaler_train_second.fit_transform(temp_training_data)
    X_test = scaler_test.fit_transform(X_test)

    Y_test = np.array(complete_test_data['Target']).reshape(-1,1)

    print(X_train.shape)

    regr = RidgeCV(alphas = (0.1,1.0,10.0,50.0,100.0,200.0,300.0,400.0,500.0)) 
    regr.fit(X_train, Y_train) 
    print(regr.score(X_train, Y_train))

#     print(regr.intercept_)
#     print(regr.coef_)
#     print(Y_test[0:10])
#     print(complete_test_data[['Date' , 'Target']])
    for index_val in range(0, len(X_test)):
#         if X_test.iloc[index_val].isnull().values.any():
#             continue
        if np.isnan(X_test[index_val]).any():
            continue
        else :    
#             print(index_val)
#             print(X_test.iloc[index_val])
            Y_pred = regr.predict(X_test[index_val].reshape(1, -1))
#             print(Y_pred[0][0])
            sharpe_calculate_pandas_test.at[index_val, column_val] = Y_pred[0][0]
#             input()
#             break
#     print(temp_training_data.iloc[600])
    for index_val in range(0, len(temp_training_data)):
#         if temp_training_data.iloc[index_val].isnull().values.any():
#             continue
        if np.isnan(temp_training_data[index_val]).any():
            continue
        else :    
#             print(index_val)
#             print(temp_training_data.iloc[index_val])
            Y_pred = regr.predict(temp_training_data[index_val].reshape(1, -1))
#             print(Y_pred[0][0])
            sharpe_calculate_pandas_train.at[index_val, column_val] = Y_pred[0][0]
#             break
#     break

print(sharpe_calculate_pandas_test)
print(sharpe_calculate_pandas_train)

# input()


save_dir_train = '/home/kgrag/first_env/Technical_Analysis_Indicators/models/linear_regression/train_profit/' 
save_dir_test = '/home/kgrag/first_env/Technical_Analysis_Indicators/models/linear_regression/test_profit/'

sharpe_calculate_pandas_train.to_csv(save_dir_train + 'ridge_lasso_cv_norm.csv', index=False)
sharpe_calculate_pandas_test.to_csv(save_dir_test + 'ridge_lasso_cv_norm.csv', index=False)


# /home/kgrag/first_env/Technical_Analysis_Indicators/models/linear_regression
# if str(sharpe_calculate_pandas_test.iloc[0,2]) == 'nan':
#     print("here")

# iterator_function(return_data, industry_data, sharpe_calculate_pandas_test, 0.5, 1)
