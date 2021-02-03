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
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_trim = trim_image(img_gray, TRIM_POS, TRIM_SIZE)
    img_med = cv2.medianBlur(img_trim, 75)
    img_res = cv2.resize(img_med, RES_SIZE)
    return img_res

# 画像加工して保存
def convert_images(path_in, path_out):
    for path in glob.glob(path_in + "/*.jpg"):
        img_org = cv2.imread(path)
        img_dst = process_image(img_org)
        cv2.imwrite(path_out + "/" + os.path.splitext(os.path.basename(path))[0] + ".png", img_dst)

if __name__ == "__main__":
    convert_images(THIS_PATH + "/images/normal", THIS_PATH + "/dataset/normal")
    convert_images(THIS_PATH + "/images/0.6", THIS_PATH + "/dataset/0.6")
