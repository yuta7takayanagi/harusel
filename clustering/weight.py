import os
import glob
import cv2
import numpy as np

from const import *
from image import *

THIS_PATH = os.path.dirname(os.path.abspath(__file__))

# 重み付けをした上で一次元化
def weight_line(img):
    n = len(WEIGHTS)
    w = np.array(WEIGHTS) / sum(WEIGHTS)
    t = sum(img[:, :, i] for i in range(3)) / 765
    x = np.zeros(TRIM_SIZE[0])

    for i in range(n):
        l = TRIM_SIZE[1] * i // n
        r = TRIM_SIZE[1] * (i + 1) // n
        x = x + np.median(t[l:r], axis=0) * w[i]

    return x

if __name__ == "__main__":
    lines = np.empty(0)
    cnt = 0

    for path in glob.glob(THIS_PATH + "/images/*/*"):
        img_org = cv2.imread(path)
        img_temp = trim_image(img_org, TRIM_POS, TRIM_SIZE)
        lines = np.append(lines, weight_line(img_temp))
        cnt += 1

    lines = np.reshape(lines, (cnt, TRIM_SIZE[0])).T
    np.savetxt(THIS_PATH + "/lines.csv", lines, delimiter=",")
