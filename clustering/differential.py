import os
import glob
import cv2
import numpy as np

from const import *
from image import *

THIS_PATH = os.path.dirname(os.path.abspath(__file__))

# 一次微分
def differential_1st(img):
    img_dst = np.diff(img, 1, axis=1)
    return img_dst

# 二次微分
def differential_2nd(img):
    img_dst = np.diff(img, 2, axis=1)
    return img_dst

# 一次元化
def differential_line(img):
    x = sum(img[:, :, i] for i in range(3)) / 765
    x = np.median(x, axis=0)
    # x = np.sort(x, axis=0)
    # x = np.mean(x[TRIM_SIZE[1] // 2 - 100 : TRIM_SIZE[1] // 2 + 100], axis=0)
    return x

if __name__ == "__main__":
    lines_1st = np.empty(0)
    lines_2nd = np.empty(0)
    cnt = 0

    for path in glob.glob(THIS_PATH + "/images/*/*"):
        img_org = cv2.imread(path)
        img_temp = trim_image(img_org, TRIM_POS, TRIM_SIZE)

        img_1st = differential_1st(img_temp)
        img_2nd = differential_2nd(img_temp)

        lines_1st = np.append(lines_1st, differential_line(img_1st))
        lines_2nd = np.append(lines_2nd, differential_line(img_2nd))

        cnt += 1

    lines_1st = np.reshape(lines_1st, (cnt, TRIM_SIZE[0] - 1)).T
    lines_2nd = np.reshape(lines_2nd, (cnt, TRIM_SIZE[0] - 2)).T

    np.savetxt(THIS_PATH + "/lines_1st.csv", lines_1st, delimiter=",")
    np.savetxt(THIS_PATH + "/lines_2nd.csv", lines_2nd, delimiter=",")
