from keras.datasets import mnist
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils import to_categorical
from keras.models import Sequential
#from keras.layers import Input, Dense
from keras.models import load_model

(X_train, y_train),(X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 28*28)
X_test = X_test.reshape(10000, 28*28)

X_train = X_train.astype("float32") / 255
#print(X_train[0])
X_test = X_test.astype("float32") / 255

# print(X_test[0].shape)

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test,10)

model = load_model('mnist_1.keras')

# We can continue to train the model
#model.fit(X_train, y_train, epochs =5)

prediction = model.predict(X_test[[5000]])
print(f"Prediction:{prediction}")

print()

print(y_test[5000])
