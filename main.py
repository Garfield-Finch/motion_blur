import os
import cv2
from PIL import Image
import numpy as np


DATASET = '../dataset/MPI-Sintel-complete/'


def gen_nm(num):
    ans = 'frame_{0:04d}.png'.format(num)
    return ans


def get_image_info(image):
    print('type(image)', type(image))   # type() 函数如果只有第一个参数则返回对象的类型   在这里函数显示图片类型为 numpy类型的数组
    print('image.shape', image.shape)
    # 图像矩阵的shape属性表示图像的大小，shape会返回tuple元组，
    # 第一个元素表示矩阵行数，第二个元组表示矩阵列数，第三个元素是3，表示像素值由光的三原色组成
    # print('image.size', image.size)   # 返回图像的大小，size的具体值为shape三个元素的乘积
    print('image.dtype', image.dtype)  # 数组元素的类型通过dtype属性获得
    pixel_data = np.array(image)
    print(pixel_data.shape)
    # print(pixel_data)  # 打印图片矩阵     N维数组对象即矩阵对象


def avg_img(imgset_pth, img_n, mix_n):
    img = np.array(cv2.imread(os.path.join(imgset_pth, gen_nm(img_n)))).astype(np.float)
    total = 1
    for i in range(max(img_n - mix_n, 1), min(img_n + mix_n, 50)):
        total += 1
        img += np.array(cv2.imread(os.path.join(imgset_pth, gen_nm(i)))).astype(np.float)
    img = img / total
    img = img.astype(np.uint8)
    print('img num: {}; mix img number: {}'.format(img_n, mix_n))

    return img


def main():

    subset_pth = 'training/albedo/'
    # imgset = 'cave_4/'
    imgset = 'alley_1/'

    imgset_pth = os.path.join(DATASET, subset_pth, imgset)
    print(imgset_pth)
    img = avg_img(imgset_pth, 11, 2)

    cv2.namedWindow('test_image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('test_image', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
