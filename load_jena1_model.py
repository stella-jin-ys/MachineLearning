from keras.models import load_model
import pandas as pd
import numpy as np

model = load_model('jena_1.keras')

data = pd.read_csv('sequences.csv')

# Approach 1
#row_1 = data.iloc[0]
#row_1 = row_1[0:-1]

#row_array = np.array(row_1)
#row_array = row_array.reshape(1,4)

#print(row_array)
#prediction = model.predict(row_array)
#print(prediction)

# Approach 2
row_1 = data.iloc[[0]]
print(type(row_1))
print(row_1)

row_1 = row_1.drop('Label', axis=1)
print(row_1)
prediction = model.predict(row_1)
print(prediction)