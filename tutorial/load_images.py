import numpy as np
import cv2
import glob
import random
import pickle

import tensorflow as tf
from tensorflow.keras import layers, models

WIDTH = 150
HEIGHT = 150
CHANNEL = 3

# 画像読み込み
datasets = []

# 猫
for path in glob.glob("./images/cat/*.jpg"):
    img_org = cv2.imread(path)
    img_res = cv2.resize(img_org, (WIDTH, HEIGHT))
    datasets.append((img_res, 0))

# 犬
for path in glob.glob("./images/dog/*.jpg"):
    img_org = cv2.imread(path)
    img_res = cv2.resize(img_org, (WIDTH, HEIGHT))
    datasets.append((img_res, 1))

# シャッフル
random.shuffle(datasets)

all_cnt = len(datasets)
train_cnt = int(all_cnt * 0.75)
test_cnt = all_cnt - train_cnt

train_images = []
train_labels = []
test_images = []
test_labels = []

for i in range(train_cnt):
    train_images.append(datasets[i][0])
    train_labels.append(datasets[i][1])

for i in range(train_cnt, all_cnt):
    test_images.append(datasets[i][0])
    test_labels.append(datasets[i][1])

train_images = np.array(train_images)
train_images = np.resize(train_images, (train_cnt, HEIGHT, WIDTH, CHANNEL))
train_images = train_images / 255.0

train_labels = np.array(train_labels)

test_images = np.array(test_images)
test_images = np.resize(test_images, (test_cnt, HEIGHT, WIDTH, CHANNEL))
test_images = test_images / 255.0

test_labels = np.array(test_labels)

with open("images.bin", "wb") as f:
    pickle.dump(((train_images, train_labels), (test_images, test_labels)), f)
