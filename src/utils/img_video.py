"""
Convert between pictures and videos

Written by Shaopeng Wang
"""

import os
from cv2 import cv2

def video2img(video_path, save_path, save_name):
    """
    Extract each frame in the video as images

    inputs：
        video_path : Video storage path, i.e. 'home/data/video.mp4'
        save_path : Folder for storing image sequences, i.e. 'home/data/runs/'
        save_name : name of image sequences, i.e. 'my' for my_1.jpg, my_2.jpg, ...
    """
    cap = cv2.VideoCapture(video_path)

    if cap.isOpened():
        print("Video is opened")
    else:
        print("Video not opened")
        exit(1)

    idx = 1
    while True:
        ret, img = cap.read()
        if not ret:
            print("{} images have been extract from {} to {} as {}_[num].jpg".format(idx-1, video_path, save_path, save_name))
            break

        save_path = os.path.join(save_path, "{}_{}.jpg".format(save_name, idx))
        idx += 1
        cv2.imwrite(save_path, img)
        print("{} is saved as {}".format(ret, save_path))


def img2video(images_path, save_path):
    """
    Make a video from multiple images

    inputs：
        images_path : Image sequences storage path, i.e. 'home/data/'
        save_path : File for video , i.e. 'home/data/runs/video.mp4'
    """
    frames = sorted(os.listdir(images_path))

    fps = 30    # video frame (can be changed)
    img = cv2.imread(os.path.join(images_path, frames[0]))
    img_size = (img.shape[0], img.shape[1])

    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    videowriter = cv2.VideoWriter(save_path, fourcc, fps, img_size)

    for frame in frames:
        img = cv2.imread(os.path.join(images_path, frame))
        videowriter.write(img)
        print("frame {} has been written".format(frame))

    videowriter.release()


if __name__ == '__main__':
    print("test")
