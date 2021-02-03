import os
import glob
import cv2
import numpy as np

from const import *

THIS_PATH = os.path.dirname(os.path.abspath(__file__))

# 差分
def derivate_first(img):
    filter = np.array([
        [0, 0, 0],
        [-8, 0, 8],
        [0, 0, 0]
    ])

    img_dst = cv2.filter2D(img, -1, filter)
    return img_dst

# 二次微分
def derivate_second(img):
    img_1st = derivate_first(img)
    # img_2nd = derivate_first(img_1st)
    return img_1st

# 画像加工して保存
def derivate_images(path_in, path_out):
    for path in glob.glob(path_in + "/*.png"):
        img_org = cv2.imread(path)
        img_dst = derivate_second(img_org)
        cv2.imwrite(path_out + "/" + os.path.basename(path), img_dst)

if __name__ == "__main__":
    derivate_images(THIS_PATH + "/dataset/normal", THIS_PATH + "/1st/normal")
    derivate_images(THIS_PATH + "/dataset/0.6", THIS_PATH + "/1st/0.6")
