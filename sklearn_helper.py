from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

custom_data = pd.DataFrame(index = [0,1,2], columns = [0,1,2])
print(custom_data)

custom_data.iloc[0,0] = 1
custom_data.iloc[0,1] = 2
# custom_data.iloc[0,2] = 3

# custom_data.iloc[1,2] = 4
custom_data.iloc[1,1] = np.inf
custom_data.iloc[2,0] = 1

print(custom_data)

# data = [[2, 1], [3, 2], [2, np.nan], [0, 1]]

scaler = StandardScaler()

# print(scaler.fit(custom_data))

# print(scaler.mean_)
# print(np.sqrt(scaler.var_))


new_data = scaler.fit_transform(custom_data)
print(new_data[0])
print(np.isnan(new_data[0]).any())
