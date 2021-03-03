import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

THIS_PATH = os.path.dirname(os.path.abspath(__file__))

# 正規化
def zscore(x, axis=None):
    mean = x.mean(axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    return (x - mean) / std

# 読み込み
with open(THIS_PATH + "/dataset.bin", "rb") as f:
    (curves, labels) = pickle.load(f)

curves = zscore(curves, axis=0)

# モデルを構築
model = models.Sequential([
    layers.Dense(4, input_shape=(3,)),
    layers.Dense(1, activation="sigmoid")
])

model.summary()

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["acc"]
)

# 学習
history = model.fit(
    curves,
    labels,
    epochs=500,
    verbose=2,
    validation_split=0.2,

)

loss = history.history["loss"]
acc = history.history["acc"]
val_loss = history.history["val_loss"]
val_acc = history.history["val_acc"]
epochs = range(len(loss))

plt.plot(epochs, loss, "b", color="b")
plt.plot(epochs, val_loss, "b", color="g")
plt.show()

plt.plot(epochs, acc, "b", color="b")
plt.plot(epochs, val_acc, "b", color="g")
plt.show()

model.save(THIS_PATH + "/model.h5")
