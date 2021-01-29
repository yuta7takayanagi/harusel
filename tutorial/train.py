import pickle

import tensorflow as tf
from tensorflow.keras import layers, models

WIDTH = 150
HEIGHT = 150
CHANNEL = 3

# 画像読み込み
with open("images.bin", "rb") as f:
    (train_images, train_labels), (test_images, test_labels) = pickle.load(f)

# モデルの構築
model = models.Sequential([
    layers.Conv2D(16, 3, activation='relu', input_shape=(HEIGHT, WIDTH, CHANNEL)),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(train_images, train_labels, epochs=10)

model.evaluate(test_images, test_labels, verbose=2)
