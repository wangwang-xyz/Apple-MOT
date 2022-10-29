"""
Visualization of the label.txt in the image

Written by Shaopeng Wang
"""
import os
import sys
import yaml
from cv2 import cv2
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parent  # YOLOv5 root directory
ROOT = ROOT / 'yolov5'
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

sys.path.append('..')
from yolov5.utils.plots import Annotator, colors
sys.path.remove('..')

data = r'../data/apple_disease/data.yaml'
img_name = r'../data/apple_disease/test/images/'
label_name = r'../data/apple_disease/test/labels/'
save_dir = r'../data/MyData/test/img/'


if __name__ == '__main__':              # visualize the label
    with open(data, errors='ignore') as f:
        names = yaml.safe_load(f)['names']  # class names

    imgs = os.listdir(img_name)
    labels = os.listdir(label_name)

    idx = 1
    for path_img in imgs:
        path_label = path_img[0:-3]+'txt'

        img0 = cv2.imread(img_name+path_img)
        img = img0.copy()
        h, w = img0.shape[:2]
        annotator = Annotator(img0, line_width=3)

        f = open(label_name+path_label, 'r+', encoding='utf-8')
        if os.path.exists(label_name+path_label) == True:
            new_lines = []
            while True:
                line = f.readline()
                if line:
                    msg = line.split(" ")
                    c = int(msg[0])
                    # print(x_center,",",y_center,",",width,",",height)
                    x1 = int((float(msg[1]) - float(msg[3]) / 2) * w)  # x_center - width/2
                    y1 = int((float(msg[2]) - float(msg[4]) / 2) * h)  # y_center - height/2
                    x2 = int((float(msg[1]) + float(msg[3]) / 2) * w)  # x_center + width/2
                    y2 = int((float(msg[2]) + float(msg[4]) / 2) * h)  # y_center + height/2
                    print(x1, ",", y1, ",", x2, ",", y2)
                    xyxy = [x1, y1, x2, y2]
                    annotator.box_label(xyxy, names[c], color=colors(c, True))
                else:
                    break

        img0 = annotator.result()
        cv2.imshow("show", img0)
        c = cv2.waitKey(0)
        print(c)
        if c == 121:
            save_name = os.path.join(save_dir + "git_{}.png".format(idx))
            idx += 1
            cv2.imwrite(save_name, img)



