import os
import glob
import cv2
import numpy as np

from const import *
from image import *

THIS_PATH = os.path.dirname(os.path.abspath(__file__))

# 一階微分
def derivate_1st(img):
    filter = np.array([[-1, 1]])
    img_temp = cv2.filter2D(img, -1, filter)
    return img_temp

# 二階微分
def derivate_2nd(img):
    filter = np.array([[-1, 1]])
    img_temp = cv2.filter2D(img, -1, filter)
    img_temp = cv2.filter2D(img_temp, -1, filter)
    return img_temp

# 行を抜き出し
def get_line(img):
    return img[RES_SIZE[1] // 2, :]

def line_1st(img):
    return get_line(derivate_1st(img))

def line_2nd(img):
    return get_line(derivate_2nd(img))

def sum_2nd(path_in):
    for path in glob.glob(path_in + "/*"):
        img_org = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_dst = line_2nd(img_org)
        print(np.sum(img_dst))

if __name__ == "__main__":
    sum_2nd(THIS_PATH + "/dataset/normal")
    print()
    sum_2nd(THIS_PATH + "/dataset/0.6")

    convert_images(THIS_PATH + "/dataset/normal", THIS_PATH + "/2nd/normal", line_2nd, True)
    convert_images(THIS_PATH + "/dataset/0.6", THIS_PATH + "/2nd/0.6", line_2nd, True)
