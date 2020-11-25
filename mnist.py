import tensorflow as tf
from tensorflow.keras import layers, models

WIDTH = 28
HEIGHT = 28
CHANNEL = 1

# 画像読み込み
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((60000, HEIGHT, WIDTH, CHANNEL))
x_test = x_test.reshape((10000, HEIGHT, WIDTH, CHANNEL))

x_train, x_test = x_train / 255.0, x_test / 255.0

# モデルの構築
model = models.Sequential([
  layers.Conv2D(32, 3, activation='relu', input_shape=(HEIGHT, WIDTH, CHANNEL)),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, activation='relu'),
  layers.Flatten(),
  layers.Dense(64, activation='relu'),
  layers.Dense(10, activation='softmax')
])

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test, verbose=2)
