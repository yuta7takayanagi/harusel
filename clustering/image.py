import os
import glob
import cv2
import numpy as np

from const import *

THIS_PATH = os.path.dirname(os.path.abspath(__file__))

# 画像切り取り
def trim_image(img, pos, size):
    return img[pos[1] : pos[1] + size[1], pos[0] : pos[0] + size[0]]

# 一次元化
def line_image(img):
    img_temp = trim_image(img, TRIM_POS, TRIM_SIZE)
    x = sum(img_temp[:, :, i] for i in range(3)) / 765
    x = np.sort(x, axis=0)
    x = np.mean(x[TRIM_SIZE[1] // 2 - 100 : TRIM_SIZE[1] // 2 + 100], axis=0)
    return x

# 画像加工して保存
def convert_images(path_in, path_out, convert_func):
    for path in glob.glob(path_in + "/*"):
        img_org = cv2.imread(path)
        img_dst = convert_func(img_org)
        cv2.imwrite(path_out + "/" + os.path.splitext(os.path.basename(path))[0] + ".png", img_dst)

# フォルダ内の画像を処理
def process_images(path_in, process_func):
    for path in glob.glob(path_in + "/*"):
        img_org = cv2.imread(path)
        print(process_func(img_org))

if __name__ == "__main__":
    img = cv2.imread(THIS_PATH + "/images/normal/017.jpg")
    x = line_image(img)

    for t in x:
        print(t)
