import os
import cv2
from PIL import Image
import numpy as np
import time


MPI_PTH = '../dataset/MPI-Sintel-complete/'
OW_PTH = '../dataset/Overwatch/'
VIDEO_PTH = '../video/'


class IterImage:
    def __init__(self, pth):
        self.pth = pth
        self.filelist = os.listdir(pth)
        self.position = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.position >= len(self.filelist):
            raise StopIteration

        img_nm = self.filelist[self.position]
        ret_img = cv2.imread(self.pth + img_nm)
        self.position += 1
        return ret_img


def gen_nm(num):
    ans = 'frame_{0:04d}.png'.format(num)
    return ans


def get_image_info(image):
    print('================= get image info =================')
    print('== type(image): ', type(image))   # type() 函数如果只有第一个参数则返回对象的类型   在这里函数显示图片类型为 numpy类型的数组
    print('== image.shape: ', image.shape)
    # 图像矩阵的shape属性表示图像的大小，shape会返回tuple元组，
    # 第一个元素表示矩阵行数，第二个元组表示矩阵列数，第三个元素是3，表示像素值由光的三原色组成
    # print('image.size', image.size)   # 返回图像的大小，size的具体值为shape三个元素的乘积
    print('== image.dtype: ', image.dtype)  # 数组元素的类型通过dtype属性获得
    # pixel_data = np.array(image)
    # print(pixel_data.shape)
    # print(pixel_data)  # 打印图片矩阵     N维数组对象即矩阵对象
    print('--------------------------------------------------')


def gen_video(in_path, size, version):
    """
    generate video with images
    usage: =======================================
    # generate video from images
    img = cv2.imread(os.path.join(imgset_pth, gen_nm(1)))
    gen_video(imgset_pth, img.shape[:2], 1)
    ----------------------------------------------
    :param in_path:
    :param size: image resolution
    :param version: to generate video name
    :return:
    """
    '''
    fps:
    帧率：1秒钟有n张图片写进去[控制一张图片停留5秒钟，那就是帧率为1，重复播放这张图片5次] 
    如果文件夹下有50张 534*300的图片，这里设置1秒钟播放5张，那么这个视频的时长就是10秒
    '''
    fps = 12
    # size = (591,705) #图片的分辨率片
    out_pth = VIDEO_PTH + "video{0:02d}.avi".format(version)  # 导出路径
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')  # 不同视频编码对应不同视频格式（例：'I','4','2','0' 对应avi格式）

    video = cv2.VideoWriter(out_pth, fourcc, fps, size)

    filelist = os.listdir(in_path)  # 获取该目录下的所有文件名
    for item in filelist:
        if item.endswith('.png'):  # 判断图片后缀是否是.png
            item = in_path + item
            img = cv2.imread(item)  # 使用opencv读取图像，直接返回numpy.ndarray 对象，通道顺序为BGR ，注意是BGR，通道值默认范围0-255。
            video.write(img)  # 把图片写进视频

    video.release()  # 释放


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


def gen_opt_flow_video(video_pth):
    cap = cv2.VideoCapture(video_pth)

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    while 1:
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)

        cv2.imshow('frame', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cv2.destroyAllWindows()
    cap.release()


def gen_opt_flow_img(img_pth):
    cap = IterImage(img_pth)

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # Take first frame and find corners in it
    old_frame = next(cap)

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    while 1:
        try:
            frame = next(cap)
        except StopIteration:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)

        cv2.imshow('frame', img)

        # cv2.waitKey(0)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def blur_with_opt_flow():
    pass


def main():

    subset_pth = 'training/'
    # img_category = 'albedo/'
    img_category = 'final/'
    # imgset = 'cave_4/'
    imgset = 'alley_1/'
    imgset_pth = os.path.join(MPI_PTH, subset_pth, img_category, imgset)

    # # calculate average image
    # # ==================================================
    # print(imgset_pth)
    # img = avg_img(imgset_pth, 11, 2)
    # # visualize average image
    # cv2.namedWindow('test_image', cv2.WINDOW_AUTOSIZE)
    # cv2.imshow('test_image', img)
    # cv2.waitKey(0)
    # # --------------------------------------------------

    # # visualize optical flow by video
    # # ==================================================
    # video_pth = os.path.join(OW_PTH, 'Overwatch_v02.mp4')
    # print(video_pth)
    # gen_opt_flow_video(video_pth)
    # # --------------------------------------------------

    # # calculate optical flow by images
    # # ==================================================
    # gen_opt_flow_img(imgset_pth)
    # # --------------------------------------------------

    # # generate blur with optical flow
    # # ==================================================
    blur_with_opt_flow()
    # # --------------------------------------------------


if __name__ == '__main__':
    main()
