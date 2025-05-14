import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

data = pd.read_csv('jena-temp-c.csv')

data = data['T (degC)']

data = np.array(data)

data = data.reshape(-1,1)

scaler = MinMaxScaler()

data_scaled = scaler.fit_transform(data)

print(data_scaled[0])

data_one = scaler.inverse_transform(data_scaled[0].reshape(1,1))

print(data_one)