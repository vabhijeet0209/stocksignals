import numpy as np
list_val = np.array(
	# [0.09137328, 0.04399781, 0.15282733, 0.09792739, 0.11446475, 0.15667481,
 # 0.15586988, 0.14665796, 0.04818902, 0.10374369, 0.09033751, 0.22140439,
 # 0.05218327, 0.04984558, 0.06311522, 0.0997932 , 0.07366907, 0.07744225,
 # 0.04516738, 0.07482007, 0.11113256, 0.0990167 , 0.09735929, 0.02381713,
 # 0.05264607, 0.02392466, 0.05656774, 0.0278523 , 0.10459449, 0.05242732,
 # 0.04278104, 0.06484746,] 
 #    [261, 278, 257, 262, 258, 254, 267, 273, 284, 276, 271, 247, 284, 278,
 # 284, 269, 259, 281, 281, 281, 276, 283, 260, 287, 277, 287, 275, 289,
 # 267, 270, 276, 276,]
[737, 795, 712, 745, 723, 728, 752, 768, 804, 778, 774, 698, 804, 789,
 811, 766, 742, 810, 795, 809, 796, 796, 756, 814, 777, 821, 788, 819,
 744, 775, 787, 791,]
)


a = [False,  True, False,  True,  True, False, False, False, False,  True,  True, False,
  True, False,  True, False,  True, False, False, False, False, False,  True, False,
  True, False, False, False,  True, False,  True, False,]



list_features =['dpo_return', 
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

dict_val = {}
for i in range(0, len(list_features)):
	dict_val[list_features[i]] = a[i]



for k, v in dict_val.items():
	if v == False:
	    print("'" + k + "'," )



