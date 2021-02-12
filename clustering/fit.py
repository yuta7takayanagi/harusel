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
            print(p)
            return p[0]

    print("Error")

if __name__ == "__main__":
    img = cv2.imread(THIS_PATH + "/images/normal/017.jpg")
    fit_image(img)

    # process_images(THIS_PATH + "/images/normal", fit_image)
    # print()
    # process_images(THIS_PATH + "/images/0.6", fit_image)
