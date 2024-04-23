import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report

# Kaggle_link = https://www.kaggle.com/code/ibliss/cifar/

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
print(X_train.shape)
plt.imshow(X_train[0])
ytrain_c = to_categorical(Y_train)
ytest_c = to_categorical(Y_test)
X_test = X_test / 255
X_train = X_train / 255

model = Sequential()
model.add(
    Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        strides=(1, 1),
        input_shape=(32, 32, 3),
        activation="relu",
    )
)
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(
    Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        strides=(1, 1),
        input_shape=(32, 32, 3),
        activation="relu",
    )
)
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dense(10, activation="softplus"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
early_stopping = EarlyStopping(monitor="val_loss", patience=1)
model.fit(
    X_train,
    ytrain_c,
    epochs=15,
    validation_data=(X_test, ytest_c),
    callbacks=early_stopping,
)

model.save("Cifar_model.keras")

metrics = pd.DataFrame(model.history.history)
print(metrics.columns)
metrics[["loss", "val_loss"]].plot()
plt.show()
metrics[["accuracy", "val_accuracy"]].plot()
plt.show()
my_number = X_test[0]

plt.imshow(my_number)
plt.show()


print("The Prediction of the Animal that is displayed")
test = model.predict(my_number.reshape(1, 32, 32, 3))
print(np.argmax(test, axis=1))

predictions = np.argmax(model.predict(X_test), axis=1)
print(classification_report(Y_test.flatten(), predictions))
