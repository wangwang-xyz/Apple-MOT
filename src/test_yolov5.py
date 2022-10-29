"""
Testing of the Detector class

Written by Shaopeng Wang
"""
from cv2 import cv2
from detect import Detector

if __name__ == '__main__':      # test Detector class

    detector = Detector(nosave=True)

    cv2.namedWindow("camera")
    cv2.resizeWindow("camera", 800, 450)

    cap = cv2.VideoCapture(0)
    if cap.isOpened()==False:
        exit(1)

    i = 0
    while True:
        i += 1
        ret, img = cap.read()
        cv2.imshow("camera", img)
        pred = detector.detect(img, "{}".format(i))

        k = cv2.waitKey(1)
        if k == 27:
            cv2.destroyAllWindows()
            cap.release()
            break
