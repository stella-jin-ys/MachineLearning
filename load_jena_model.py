from keras.models import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
#row_1 = data.iloc[[0]]
#print(type(row_1))
#print(row_1)
#row_1 = row_1.drop('Label', axis=1)
#print(row_1)

rows = data.iloc[-50000:-1]
#print(type(rows))
#print(rows)

features = rows.drop('Label', axis=1)
label = rows['Label']
#print(rows)

loss = model.evaluate(features,label)
print("Loss: ",loss)

prediction = model.predict(features)
#print(prediction)

plt.plot(prediction)
plt.plot(np.array(label))
plt.show()


