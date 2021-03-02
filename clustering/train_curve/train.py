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

# 読み込み
with open(THIS_PATH + "/dataset.bin", "rb") as f:
    (curves, labels) = pickle.load(f)

train_cnt = int(len(curves) * 0.75)
train_curves = curves[:train_cnt]
train_labels = labels[:train_cnt]
test_curves = curves[train_cnt:]
test_labels = labels[train_cnt:]

# モデルを構築
model = models.Sequential([
    layers.Dense(4, input_shape=(3,)),
    # layers.Dense(2),
    layers.Dense(1, activation="sigmoid")
])

model.summary()

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 学習
history = model.fit(train_curves, train_labels, epochs=1000)

loss = history.history["loss"]
acc = history.history["accuracy"]
epochs = range(len(loss))

plt.plot(epochs, acc, "o")
plt.show()

model.evaluate(test_curves, test_labels, verbose=2)
model.save(THIS_PATH + "/model.h5")
