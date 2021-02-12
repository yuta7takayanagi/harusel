import os
import pickle
import glob
import numpy as np
import cv2

from const import *
from image import *

THIS_PATH = os.path.dirname(os.path.abspath(__file__))

# データセットを作成
def create_dataset(path_in, label, process_func):
    images = []
    labels = []

    for path in glob.glob(path_in + "/*"):
        img = cv2.imread(path)
        line = process_func(img)
        images.append(line)
        labels.append(label)

    return (images, labels)

if __name__ == "__main__":
    images = []
    labels = []

    datasets = create_dataset(THIS_PATH + "/images/normal", 0, line_image)
    images += datasets[0]
    labels += datasets[1]

    datasets = create_dataset(THIS_PATH + "/images/0.6", 1, line_image)
    images += datasets[0]
    labels += datasets[1]

    # シャッフル
    p = np.random.permutation(len(images))
    images = np.array(images)[p]
    labels = np.array(labels)[p]

    with open(THIS_PATH + "/dataset.bin", "wb") as f:
        pickle.dump((images, labels), f)
