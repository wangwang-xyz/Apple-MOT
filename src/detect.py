"""
Object detection model based on YOLOv5

Writen by Shaopeng Wang
"""
import os
import sys
from pathlib import Path
import numpy as np
from cv2 import cv2
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parent  # YOLOv5 root directory
ROOT = ROOT / 'yolov5'
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

sys.path.append('..')
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import (LOGGER, check_img_size, colorstr, cv2, increment_path, non_max_suppression,
                                  scale_coords, strip_optimizer, xyxy2xywh)
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.torch_utils import select_device, time_sync
sys.path.remove('..')

del LOGGER.parent.handlers[1]

class Detector():                       # Detector with YOLOv5 model
    """
    Object detecting class based on YOLOv5 model

    Encapsulates the YOLOv5 model, provides a target detection interface,
    takes the image as input, and outputs the target type, BBOX, confidence
    and other information contained in the image.
    """
    def __init__(self,
            weights=ROOT / 'best640.pt',  # model.pt path(s)
            source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
            data=ROOT / 'data/data.yaml',  # dataset.yaml path
            imgsz=(640, 640),  # inference size (height, width)
            conf_thres=0.5,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=True,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project=ROOT / 'runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
    ):
        """
        Initialize the parameters of the YOLOv5 and load the model to the GPU
        """
        self.weights = weights
        self.source = str(source)
        self.data =data
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.device = device
        self.view_img = view_img
        self.save_txt = save_txt
        self.save_conf = save_conf
        self.save_crop = save_crop
        self.nosave = nosave
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.visualize = visualize
        self.update = update
        self.project = project
        self.name = name
        self.exist_ok = exist_ok
        self.line_thickness = line_thickness
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf
        self.half = half
        self.dnn = dnn

        self.save_img = not self.nosave and not self.source.endswith('.txt')  # save inference images

        # ------------------ Save Path
        self.save_dir = increment_path(Path(self.project) / self.name, exist_ok=exist_ok)  # increment run
        (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # ------------------ Load Detection Model
        self.device = select_device(self.device)
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data, fp16=self.half)
        self.model.eval()
        self.model.warmup(imgsz=(1, 3, *self.imgsz))
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size

    @torch.no_grad()
    def detect(self, img, frame):                      # Detect objects from a single image
        """
        Detect objects in images

        inputs:
            img: image to be detected
            frame: Sequence number of the video frame
        outputs:
            [[x1, y1, x2, y2, conf, cls], [...], ...]
            [x1, y1, x2, y2] is the corner coordinate of bbox
            conf is the class predicted probabilities
            cls is the class number of the object
        """
        dt, seen = [0.0, 0.0, 0.0], 0
        s = "frame {} ".format(frame)
        # ------------------ Preprocess
        t1 = time_sync()
        im = self.preprocess(img, self.imgsz)
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # ------------------ Inference
        pred = self.model(im, augment=self.augment, visualize=self.visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # ------------------ NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres,
                                   self.classes, self.agnostic_nms, max_det=self.max_det)
        dt[2] += time_sync() - t3

        if self.view_img:
            cv2.namedWindow("yolo-detect")
            cv2.resizeWindow("camera", 800, 450)

        # ------------------ Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0 = str(frame), img.copy()
            save_path = str(self.save_dir) + "/{}.jpg".format(p)  # im.jpg
            txt_path = str(self.save_dir) + "/labels/{}.txt".format(p)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if self.save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))     # annotator
            if len(det):
                # ------------------ Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # ------------------ Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # ------------------ Write results
                for *xyxy, conf, cls in reversed(det):
                    if self.save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if self.save_img or self.save_crop or self.view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if self.save_crop:
                            save_one_box(xyxy, imc, file=self.save_dir / 'crops' / self.names[c] / f'{p}.jpg', BGR=True)

            # ------------------ Stream results
            im0 = annotator.result()
            if self.view_img:
                cv2.imshow("yolo-detect", im0)
                cv2.waitKey(1)  # 1 millisecond

            # ------------------ Save results (image with detections)
            if self.save_img:
                cv2.imwrite(save_path, im0)

        # ------------------ Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

        # ------------------ Print results
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *self.imgsz)}' % t)
        if self.save_txt or self.save_img:
            s = f"\n{len(list(self.save_dir.glob('labels/*.txt')))} labels saved to {self.save_dir / 'labels'}" if self.save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
        if self.update:
            strip_optimizer(self.weights)  # update model (to fix SourceChangeWarning)
        return pred[0].data.cpu().numpy()


    def preprocess(self, img, img_size=640, stride=32, auto=True):
        """
        Image Preprocess
        resizing and padding the image to 640*640
        and then change the axis from H*W*C to C*H*W

        inputs:
            img: original image
        outputs:
            img: padded image which size is 640*640
        """
        # Padded resize
        img = letterbox(img, img_size, stride=stride, auto=auto)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = img[np.newaxis, :, :, :]
        img = np.ascontiguousarray(img)
        return img


if __name__=='__main__':
    print('test detect model')
    detector = Detector()
    img = cv2.imread(ROOT/"data/images/bus.jpg")
    pred = detector.detect(img, "bus")
    print(pred)
