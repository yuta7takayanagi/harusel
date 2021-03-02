import os
import sys
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

THIS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(THIS_PATH + "/../")

from const import *

# 読み込み
with open(THIS_PATH + "/dataset.bin", "rb") as f:
    (lines, labels) = pickle.load(f)

train_cnt = int(len(lines) * 0.75)
train_lines = lines[:train_cnt]
train_labels = labels[:train_cnt]
test_lines = lines[train_cnt:]
test_labels = labels[train_cnt:]

# モデルを構築
model = models.Sequential([
    layers.Dense(1024, activation="relu", input_shape=(TRIM_SIZE[0],)),
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
model.fit(train_lines, train_labels, epochs=100)
model.evaluate(test_lines, test_labels, verbose=2)
model.save(THIS_PATH + "/model.h5")
