import os
import sys
import pickle
import glob
import numpy as np
import cv2

THIS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(THIS_PATH + "/../")

from const import *
from image import *

# データセットを作成
def create_dataset(path_in, label, process_func):
    lines = []
    labels = []

    for path in glob.glob(path_in + "/*"):
        img_org = cv2.imread(path)
        img_temp = trim_image(img_org, TRIM_POS, TRIM_SIZE)
        line = process_func(img_temp)
        lines.append(line)
        labels.append(label)

    return (lines, labels)

if __name__ == "__main__":
    lines = []
    labels = []

    datasets = create_dataset(THIS_PATH + "/images/normal", 0, line_image)
    lines += datasets[0]
    labels += datasets[1]

    datasets = create_dataset(THIS_PATH + "/images/0.6", 1, line_image)
    lines += datasets[0]
    labels += datasets[1]

    # シャッフル
    p = np.random.permutation(len(lines))
    lines = np.array(lines)[p]
    labels = np.array(labels)[p]

    with open(THIS_PATH + "/dataset.bin", "wb") as f:
        pickle.dump((lines, labels), f)
