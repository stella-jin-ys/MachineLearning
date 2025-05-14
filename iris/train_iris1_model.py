
from keras.models import Sequential
from keras.layers import Input, Dense
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
#from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('iris.csv')

feature_columns = ["sepal.length","sepal.width","petal.length","petal.width"]

# Features (properties)
X = data[feature_columns]

# Labels (classes)
y = data["variety"]

# Encode labels into numerical values
#label_encoder = LabelEncoder()
#y = label_encoder.fit_transform(y)
#print(y)

y_array = np.array(y)
y_reshaped = y_array.reshape((-1,1))

encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y_reshaped)
#print(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=25)
# print(len(X_train))
# print(len(X_test))


model = Sequential()

model.add(Input(shape=(4,))) 

model.add(Dense(12, activation="relu"))

model.add(Dense(4, activation="relu"))

model.add(Dense(3, activation="relu"))  # there are 3 classes/ labels in the data, how to count how many classes are when data is complicat4ed

model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

model.fit(X_train, y_train , epochs=8) 

loss, accuracy = model.evaluate(X_test, y_test)
print("Accuracy:", accuracy)

pred_array = np.array(X_test.iloc[0])
pred_array = pred_array.reshape(1,4)

prediction = model.predict(pred_array)

print(prediction)

