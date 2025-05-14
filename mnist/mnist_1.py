#!/usr/bin/env python3
from keras.datasets import mnist
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Input, Dense

(X_train, y_train),(X_test, y_test) = mnist.load_data()

#print(len(X_train))
#print(len(X_test))

#plt.imshow(X_train[10001],cmap='gray')
#plt.show()

#print(y_train[10001])

# Shape is tuple, tuple is like a list but cant be changed. (60000rows, 28columns, 28depths)
#print(X_train.shape)
#print(X_train[10001])

#pd.set_option("display.max_columns", None)
#pd.set_option("display.width", None)

#image_to_print= X_train[110]

#tabulate = pd.DataFrame(image_to_print)
#tabulate.columns = [i for i in range(28)]

#print(tabulate.to_string(index=True))

#hex_tabulate = tabulate.applymap(lambda x:f"02X" if x!=0 else " ")

#print(hex_tabulate.to_string(index=False))

X_train = X_train.reshape(60000,28 * 28)
X_test = X_test.reshape(10000,28 * 28)

X_train = X_train.astype("float32") / 255
#print(X_train[0])
X_test = X_test.astype("float32") / 255

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test,10)

model = Sequential()
model.add(Input(shape=(784,)))
model.add(Dense(128, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

counter = range(2)

for i in counter:
    model.fit(X_train, y_train, epochs=1, validation_data=(X_test, y_test))
    model.save(f'mnist_1_v{i}.keras')

# =============================================================================
# print("Evaluation: ")
# loss, accuracy = model.evaluate(X_test, y_test)
# 
# print("Accuracy", accuracy)
# print("Loss: ", loss)
# =============================================================================

# my_callback = keras.callbacks.EarlyStopping(
#     patience = 15,
#     mode = "max",
#     monitor = "val_accuracy",
#     start_from_epoch = 25,
#     restore_best_weights = True,
#     )

# Define a custom learning rate schedule function
# =============================================================================
# def lr_schedule(epoch, lr):
#     if epoch < 10:
#         return lr
#     else:
#         return lr * math.exp(-0.1) # decay the learning rate
#     
# # Pass the callback to model.fit()
# lr_scheduler = LearningRateScheduler(lr.schedule)
# 
# model.fit(
#     X_train,
#     y_train,
#     epochs=30,
#     callbacks=[lr_scheduler]
#     )
# =============================================================================


































