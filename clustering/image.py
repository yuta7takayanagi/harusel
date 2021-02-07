import os
import glob
import cv2
import numpy as np

from const import *

THIS_PATH = os.path.dirname(os.path.abspath(__file__))

# 画像切り取り
def trim_image(img, pos, size):
    return img[pos[1] : pos[1] + size[1], pos[0] : pos[0] + size[0]]

# 画像加工
def process_image(img):
    img_temp = img.copy()
    img_temp = trim_image(img_temp, TRIM_POS, TRIM_SIZE)
    img_temp = cv2.medianBlur(img_temp, 99)
    img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)
    img_temp = cv2.resize(img_temp, RES_SIZE)
    return img_temp

# 画像加工して保存
def convert_images(path_in, path_out, convert_func, gray=False):
    for path in glob.glob(path_in + "/*"):
        img_org = cv2.imread(path, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
        img_dst = convert_func(img_org)
        cv2.imwrite(path_out + "/" + os.path.splitext(os.path.basename(path))[0] + ".png", img_dst)

if __name__ == "__main__":
    convert_images(THIS_PATH + "/images/normal", THIS_PATH + "/dataset/normal", process_image)
    convert_images(THIS_PATH + "/images/0.6", THIS_PATH + "/dataset/0.6", process_image)
