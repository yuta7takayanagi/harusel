import os
import glob
import cv2
import numpy as np

from const import *
from image import *

THIS_PATH = os.path.dirname(os.path.abspath(__file__))

# 二次近似により係数を算出
def fit_image(img):
    x = line_image(img)

    for i in range(x.size):
        p = np.polyfit(np.arange(x.size - i), x[i:], 2)
        r2 = np.corrcoef(x[i:], np.poly1d(p)(np.arange(x.size - i)))[0][1]

        if r2 >= 0.995:
            return p

    print("Error")
    return None

if __name__ == "__main__":
    curves = np.empty(0)
    cnt = 0

    for path in glob.glob(THIS_PATH + "/images/*/*"):
        img_org = cv2.imread(path)
        img_temp = trim_image(img_org, TRIM_POS, TRIM_SIZE)
        curves = np.append(curves, fit_image(img_temp))
        cnt += 1

    curves = np.reshape(curves, (cnt, 3)).T
    np.savetxt(THIS_PATH + "/curves.csv", curves, delimiter=",")
