"""
Make a video from pictures in a folder

Written by Shaopeng Wang
"""
import os
from cv2 import cv2

if __name__ == '__main__':
    images_path = r"/home/wang/Project/MOT/data/MyData/test/tmp/img"    # path for images
    save_path = r"/home/wang/Project/MOT/data/MyData/test/test.mp4"     # path to store video
    frames = os.listdir(images_path)
    frames.sort(key=lambda x: (int(x.split('.')[0])))

    fps = 8     # video frame
    h = 1080    # height of video
    w = 1920    # width of video
    img_size = (w, h)

    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    videowriter = cv2.VideoWriter(save_path, fourcc, fps, img_size)

    for frame in frames:
        img = cv2.imread(os.path.join(images_path, frame))
        img = cv2.resize(img, img_size)
        videowriter.write(img)
        print("frame {} has been written".format(frame))

    videowriter.release()
