"""
Test the farneback algorithm and the FlowNet2 algorithm
Output the corresponding optical flow map respectively

Written by Shaopeng Wang
"""
import time
from cv2 import cv2
from opticalflow import OpticalFlow
from flownet2.utils.flow_utils import flow2img

def main():         # Compare the result of Farneback and FlowNet2
    filenames = ["52.1", "52.3", "52.2", "53.1", "53.2", "60", "61"]
    # filename = r"49"
    for filename in filenames:
        video_path = r'/media/wang/文件/Work/硕士论文/农业巡检/Data/产量估计/{}.mp4'.format(filename)
        save_path_ff = r'../run/{}_flow_ff.mp4'.format(filename)        # use Farneback algorithm
        save_path_fn = r'../run/{}_flow_fn2.mp4'.format(filename)       # use FlowNet2 model
        cap = cv2.VideoCapture(video_path)

        if cap.isOpened():
            print("Video is opened")
        else:
            print("Video not opened")
            exit(1)
        h = 450
        w = 800
        fps = 30
        fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
        ret, img = cap.read()
        img = cv2.resize(img, (w, h))
        img_size = (w, h)
        videowriter_FF = cv2.VideoWriter(save_path_ff, fourcc, fps, img_size)
        videowriter_FN = cv2.VideoWriter(save_path_fn, fourcc, fps, img_size)

        cv2.namedWindow("video", 0)
        cv2.resizeWindow("video", w, h)
        cv2.namedWindow("ff", 0)
        cv2.resizeWindow("ff", w, h)
        cv2.namedWindow("fn", 0)
        cv2.resizeWindow("fn", w, h)
        cv2.waitKey(5)

        cv2.imshow("video", img)

        flownet = OpticalFlow(net='FlowNet2CS')

        while True:
            prvs = img
            ret, img = cap.read()
            if not ret:
                print("Optical flow have been extract from {} as {} and {}".format(video_path, save_path_ff, save_path_fn))
                break
            img = cv2.resize(img, (w, h))
            cv2.imshow("video", img)

            start = time.time()
            prvs_gray = cv2.cvtColor(prvs, cv2.COLOR_BGR2GRAY)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            flow_ff = cv2.calcOpticalFlowFarneback(prvs_gray, img_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            end = time.time()
            print("{}ms used to calculate FF flow with opencv".format((end-start)*1000))
            ff_img = flow2img(flow_ff)
            cv2.imshow("ff", ff_img)
            videowriter_FF.write(ff_img)

            flow_fn = flownet.getflow(prvs, img)
            fn_img = flow2img(flow_fn)
            cv2.imshow("fn", fn_img)
            videowriter_FN.write(fn_img)

            cv2.waitKey(1)

        videowriter_FF.release()
        videowriter_FN.release()
        cv2.destroyAllWindows()



if __name__ == '__main__':
    main()