import numpy as np

intercept = [4.09480834e-05] 

list_features_lr = ['sma_indicator_return', 'rsi_indicator_return', 'simple_moving_avg_difference_return', 'ease_of_movement', 'ultimate_oscillator_signal_indicator', 'tsi_indicator_return', 'ema_indicator_return', 'macd_diff_indicator_return', 'bollinger_indicator_return']

coeff_values = [-2.10301152e-03,  1.16347918e-04, -7.16557356e-04, -7.44356262e-06,
  -4.13177882e-05,  2.97386284e-04,  9.60913477e-04, -7.96352708e-05,
  -8.47829389e-05]

all_features = [ 'dpo_return',
'rsi_indicator_return',
'simple_moving_avg_difference_return',
'roc_indicator_return',
'ease_of_movement',
'volume_price_trend_close',
'force_index_return',
'macd_diff_indicator_return',
'simple_moving_avg_difference_close',
'ema_indicator_return',
'dpo_close',
'sma_indicator_return',
'tsi_indicator_return',
'stochastic_oscillator_indicator',
'william_indicator',
'bollinger_indicator_return',
'kst_indicator_return',
'kama_indicator_return',
'ultimate_oscillator_signal_indicator',
'commodity_channel_index',
'roc_indicator_close',
'force_index_close',
'trix_indicator_return',
'stochastic_oscillator_signal_indicator',
'aroon_indicator_return',
'rsi_indicator_close',
'money_flow_index',
'vortex_indicator',
'sma_ease_of_movement',
'chaikin_money_flow',
'aroon_indicator_close',
'macd_diff_indicator_close']

feature_value_map = {}

for i in range(0, len(list_features_lr)):
    feature_value_map[list_features_lr[i]] = coeff_values[i]

for i in all_features:
    if i in feature_value_map.keys():
        continue
    else :
        feature_value_map[i] = 0.0
        

print(feature_value_map)
