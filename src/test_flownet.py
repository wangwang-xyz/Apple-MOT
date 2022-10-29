"""
Testing of the OpticalFlow class

Written by Shaopeng Wang
"""
from cv2 import cv2
from opticalflow import OpticalFlow
from flownet2.utils.flow_utils import flow2img
from detect import Detector

if __name__ == '__main__':              # test the OpticalFlow class

    flownet = OpticalFlow(net='FlowNet2C')
    detector = Detector(nosave=True)#, classes=47)    # class num of apple is 47

    cv2.namedWindow("camera", 0)
    cv2.resizeWindow("camera", 800, 450)
    cv2.namedWindow("flow", 0)
    cv2.resizeWindow("flow", 800, 450)

    cap = cv2.VideoCapture(0)
    if cap.isOpened()==False:
        exit(1)

    ret, img = cap.read()
    cv2.imshow("camera", img)

    pim1 = img
    while True:
        ret, img = cap.read()
        cv2.imshow("camera", img)
        pred = detector.detect(img, '0')
        pim2 = img

        data = flownet.getflow(pim1, pim2)
        flow = flow2img(data)
        cv2.imshow("flow", flow)
        pim1 = pim2

        k = cv2.waitKey(1)
        if k == 27:
            cv2.destroyAllWindows()
            cap.release()
            break
