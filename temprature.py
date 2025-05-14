#How to choose an activation function is based on the known input and expected output

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from sklearn.preprocessing import MinMaxScaler
 
# data2 = pd.read_csv('jena_climate_2009_2016.csv')

# print(data)

# data2['T (degC)'].to_csv('jena-temp-c.csv' , index=False)
 
data = pd.read_csv('jena-temp-c.csv')

#print(data)
 
#plt.plot(data)

#plt.show()
 
scaler = MinMaxScaler()

data_array = np.array(data["T (degC)"])

data_array = data_array.reshape(-1, 1)

 
data_reshaped = scaler.fit_transform(data_array)
df_scaled = pd.DataFrame(data_reshaped)
df_scaled.columns=["Scaled temp"]
df_scaled.to_csv('scaled_data.csv', index=False)
 
#data_reshaped = data_reshaped.flatten()

data_reshaped = data_reshaped.reshape(data_reshaped.shape[0], )
print("Last value:", data_reshaped[-1])
print("Last but one value:", data_reshaped[-2])
 
# plt.plot(data_reshaped)

# plt.show()
 
nr_features = 4

X = []
y = []

# len(data_reshaped) - nr_features

# print(len(data_reshaped) - nr_features)
 
for i in range(len(data_reshaped) - nr_features ):

#for i in range(10):

    X_helper = data_reshaped[i:i + nr_features]
    X.append(X_helper)
    
    y_helper = data_reshaped[i + nr_features]
    y.append(y_helper)

print("Second row: ",X[1])
print("First y: ", y[0])
print(y[0] == X[1][-1])

X_array = np.array(X)

df = pd.DataFrame(X_array)
df['Label'] = y
df.to_csv("sequences.csv", index = False)



 