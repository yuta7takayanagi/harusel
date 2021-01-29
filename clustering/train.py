import os
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

from const import *

# 読み込み
with open("dataset.bin", "rb") as f:
    (images, labels) = pickle.load(f)

train_cnt = int(len(images) * 0.8)
train_images = images[:train_cnt]
train_labels = labels[:train_cnt]
test_images = images[train_cnt:]
test_labels = labels[train_cnt:]

# モデルを構築
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(RES_SIZE[1], RES_SIZE[0], 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.summary()

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 学習
model.fit(train_images, train_labels, epochs=20)
model.evaluate(test_images, test_labels, verbose=2)
model.save("model.h5")
