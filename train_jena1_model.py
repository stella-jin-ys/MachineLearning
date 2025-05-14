import pandas as pd
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense

data = pd.read_csv('sequences.csv')

X = data[["0", "1", "2", "3"]]

y = data["Label"]

split = int(len(X) * 0.8)

X_train = X[0:split]
y_train = y[0:split]

X_test = X[split:]
y_test = y[split:]

model = Sequential()
model.add(Input(shape=(4,1)))
model.add(LSTM(8))
model.add(Dense(1))

model.compile(loss="mean_absolute_error", optimizer="adam" )

#print(model.summary())

model.fit(X_train, y_train, epochs=5)

model.save('jena_1.keras') # This is trained model

loss = model.evaluate(X_test, y_test)

print(loss)