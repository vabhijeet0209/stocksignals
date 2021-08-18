import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import preprocessing, svm 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
import os
from other_utils.get_top2000list import get_all_stock
from alpha_post_processing.alpha_evaluation import iterator_function

train_dir = '/home/kgrag/first_env/all_stocks/train/'
test_dir = '/home/kgrag/first_env/all_stocks/test/'


list_feature_check_nan = ['Target', 'sma_indicator_return', 'rsi_indicator_return', 'simple_moving_avg_difference_close', 'tsi_indicator_return', 'ema_indicator_return', 'macd_diff_indicator_return', 'ultimate_oscillator_signal_indicator', 'william_indicator']


list_top_2000_stocks_complete = get_all_stock()
list_top_2000_stocks_complete = np.insert(list_top_2000_stocks_complete,0, 'Date')


all_created_files = os.listdir(test_dir)
print(len(all_created_files))


return_file = "/export/scratch/for_kgrag/pv_info/returns"
return_data = pd.read_csv(return_file, sep = "|")
return_data = return_data[list_top_2000_stocks_complete]


industry_file = "/export/scratch/for_kgrag/pv_info/subindustry"
industry_data = pd.read_csv(industry_file, sep = "|")
industry_data = industry_data[list_top_2000_stocks_complete]


helper_test_data = pd.read_csv(test_dir + 'EQ0010043000001000' + '.csv', sep = ",")

helper_train_data = pd.read_csv(train_dir + 'EQ0010043000001000' + '.csv', sep = ",")

# print(helper_train_data.iloc[942:950,0])

index_list = np.arange(948, 1449) - 948
values = np.array(helper_train_data['Date'].iloc[948:1449])
sharpe_calculate_pandas_train = pd.DataFrame(index = index_list, columns = list_top_2000_stocks_complete) 
sharpe_calculate_pandas_train['Date'].iloc[:] = values


# print(sharpe_calculate_pandas_train)



sharpe_calculate_pandas_test = pd.DataFrame(index = helper_test_data.index, columns = list_top_2000_stocks_complete)
sharpe_calculate_pandas_test['Date'] = helper_test_data['Date']

# print(sharpe_calculate_pandas_test)
# input()


list_features_lr = ['sma_indicator_return', 'rsi_indicator_return']
# , 'rsi_indicator_return', 'simple_moving_avg_difference_close', 'tsi_indicator_return', 'ema_indicator_return', 'macd_diff_indicator_return', 'ultimate_oscillator_signal_indicator', 'william_indicator',
# complete_training_data.to_csv(train_dir + 'combined_train.csv', index=False)




for each_file in all_created_files:
    column_val = each_file[:-4]
    print(column_val)
    complete_training_data = pd.read_csv(train_dir + column_val + '.csv', sep = ",")
    temp_training_data = complete_training_data.iloc[948:1449]
    complete_training_data = complete_training_data.dropna(subset = list_feature_check_nan)
    
    complete_test_data = pd.read_csv(test_dir + column_val + '.csv', sep = ",")


    X_train = complete_training_data[list_features_lr]
    temp_training_data = temp_training_data[list_features_lr]
    
    Y_train = np.array(complete_training_data['Target']).reshape(-1,1)
    
    if X_train.shape[0] == 0:
        continue

    X_test = complete_test_data[list_features_lr]

    Y_test = np.array(complete_test_data['Target']).reshape(-1,1)

    print(X_train.shape)
    print(Y_train.shape)

    print(X_test.shape)
    print(Y_test.shape)

    regr = LinearRegression()
    regr.fit(X_train, Y_train) 
    print(regr.score(X_train, Y_train))
#     print(X_test)
#     print(regr.score(X_test, Y_test)) 
#     print(regr.intercept_)
#     print(regr.coef_)
#     print(Y_test[0:10])
#     print(complete_test_data[['Date' , 'Target']])
    for index_val in range(0, len(X_test)):
        if X_test.iloc[index_val].isnull().values.any():
            continue
        else :    
#             print(index_val)
#             print(X_test.iloc[index_val])
            Y_pred = regr.predict(np.array(X_test.iloc[index_val]).reshape(1, -1))
#             print(Y_pred[0][0])
            sharpe_calculate_pandas_test.at[index_val, column_val] = Y_pred[0][0]
#             break

    for index_val in range(0, len(X_test)):
        if X_test.iloc[index_val].isnull().values.any():
            continue
        else :    
#             print(index_val)
#             print(X_test.iloc[index_val])
            Y_pred = regr.predict(np.array(X_test.iloc[index_val]).reshape(1, -1))
#             print(Y_pred[0][0])
            sharpe_calculate_pandas_test.at[index_val, column_val] = Y_pred[0][0]
#             break
print(sharpe_calculate_pandas_test)
print(sharpe_calculate_pandas_train)

save_dir_train = '/home/kgrag/first_env/Technical_Analysis_Indicators/models/linear_regression/train_profit/' 
save_dir_test = '/home/kgrag/first_env/Technical_Analysis_Indicators/models/linear_regression/test_profit/'

sharpe_calculate_pandas_train.to_csv(save_dir_train + 'lr_2.csv', index=False)
sharpe_calculate_pandas_test.to_csv(save_dir_test + 'lr_2.csv', index=False)


# /home/kgrag/first_env/Technical_Analysis_Indicators/models/linear_regression
# if str(sharpe_calculate_pandas_test.iloc[0,2]) == 'nan':
#     print("here")

# print(return_data[1750:1770])
iterator_function(return_data, industry_data, sharpe_calculate_pandas_test, 0.5, 1)

#     print(Y_pred[0:10])
#     break







# print(train_data)
# print(test_data[0:9])

# # df_binary.fillna(method ='ffill', inplace = True)



# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25) 




# # plt.scatter(X_test, Y_test, color ='b') 
# # plt.plot(X_test, Y_pred, color ='k') 
  
# # plt.savefig("test.png")   