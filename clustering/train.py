import os
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

from const import *

# 読み込み
with open("dataset.bin", "rb") as f:
    (images, labels) = pickle.load(f)

print(len(images[0]))
print(len(labels))

train_cnt = int(len(images) * 0.7)
train_images = images[:train_cnt]
train_labels = labels[:train_cnt]
test_images = images[train_cnt:]
test_labels = labels[train_cnt:]

# モデルを構築
model = models.Sequential([
    layers.Dense(1024, activation="relu", input_shape=(TRIM_SIZE[0],)),
    layers.Dense(256, activation="relu", input_shape=(TRIM_SIZE[0],)),
    layers.Dropout(0.2),
    layers.Dense(1, activation="sigmoid")
])

model.summary()

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 学習
model.fit(train_images, train_labels, epochs=100)
model.evaluate(test_images, test_labels, verbose=2)
model.save("model.h5")
