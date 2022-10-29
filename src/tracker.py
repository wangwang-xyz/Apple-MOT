"""
Apple Tracking Technology Master Program
Enter the path where the video is located, and the inference result
Output the video about apple tracking and counting result

Written by Shaopeng Wang
"""
import numpy as np
from cv2 import cv2
from detect import Detector
from opticalflow import OpticalFlow
from matcher import Hungarian, KM

class Target():                 # Class used to describe target
    """
    Instance-level description class for targets
    Used to describe the category, current location, predicted location,
    historical location and other information of the tracked target
    """
    def __init__(self, name, xyxy, cls):
        """
        inputs:
            name: target's ID
            xyxy: target's bbox information
            cls: target's class number
        """
        self.name = name                        # Target name (ID)
        self.bbox = np.array(xyxy)              # Bounding box
        self.cls = cls                          # class num
        self.location = np.array([(xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2])        # center location of bbox
        self.pre_loc = self.location                # predict center location
        self.pre_bbox = self.bbox                   # predict bbox
        self.history = np.zeros((10, 2))            # history queue of center location
        self.his_len = 1                            # length of history queue
        self.history[0, :] = self.location
        self.his_head = 0                           # tail of history queue
        self.miss = 0                               # unmatched times

    def update(self, xyxy=[], cls=None):            # update target's info
        """
        Target state update function
        Update the target's position in the image
            case 1: If it is re-detected, then updated with the newly detected
                    location information
            case 2: if not detected, update with predicted location information

        inputs:
            xyxy: The bbox position of the target, or null if not detected [x1, y1, x2, y2]
            cls: The new class number of the target, considering that some targets
                may be misclassified in the previous detection, so a new category update
                is required
        """
        if len(xyxy):           # matched
            self.bbox = xyxy
            self.cls = cls
            self.location = np.array([(xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2])
            self.his_head = (self.his_head + 1) % 10
            if self.his_len<10:
                self.his_len = self.his_len + 1
            self.history[self.his_head, :] = self.location
            self.miss = 0
        else:                   # unmatched
            self.bbox = self.pre_bbox
            self.location = np.array([(self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2])
            self.his_head = (self.his_head + 1) % 10
            if self.his_len<10:
                self.his_len = self.his_len + 1
            self.history[self.his_head, :] = self.location
            self.miss = self.miss + 1


class Targets():                    # Class used to describe tracked targets
    """
    Category-level description class for targets
    Used to describe all target instances under the current category, as well
    as quantity information
    """
    def __init__(self, preds, w, h):
        """
        inputs:
            preds: targets' detect information list [[xyxy, conf, cls], ...]
            w: width of image
            h: height of image
        """
        self.w = w                  # width of image
        self.h = h                  # height of image
        self.count_num = 0          # counting num
        self.target_list = {}       # targets list
        self.disappeared_threshold = 50  # When the target disappears beyond the number of frames
                                    # specified by this threshold, the target is deleted
        for *xyxy, conf, cls in preds:
            self.count_num += 1
            self.target_list[str(self.count_num)] = Target(str(self.count_num), xyxy, cls)

    def update(self, cur_list, pred_list, pred_targets):            # update tracked targets
        """
        Category-level object description information update function

        inputs:
            cur_list: List of continuously tracked targets that were matched    (List: class Target)
            pred_list: List of newly detected targets that were matched     (List: class Target)
            pred_targets: All newly detected targets    (class Targets())
        """
        miss_list = []      # List of targets that were not matched
        new_list = []
        for name, t in enumerate(self.target_list):
            if t in cur_list:
                pred_t = pred_targets.target_list[pred_list[cur_list.index(t)]]
                self.target_list[t].update(pred_t.bbox, pred_t.cls)
            else:
                miss_list.append(t)
        """
        Eliminate targets that have not been detected for a long time and targets that are 
        beyond the field of view
        """
        for t in miss_list:
            self.target_list[t].update()
            if self.target_list[t].miss > self.disappeared_threshold or self.check_loc(self.target_list[t].location):
                self.target_list.pop(t)
        """
        Add new targets that are not matched
        """
        for name, t in enumerate(pred_targets.target_list):
            if t not in pred_list:
                new_list.append(t)
        for t in new_list:
            self.count_num += 1
            self.target_list[str(self.count_num)] = Target(str(self.count_num),
                                                           pred_targets.target_list[t].bbox,
                                                           pred_targets.target_list[t].cls)

    def check_loc(self, loc):                   # check if target's location in the image
        """
        Determine if the target is beyond the image boundaries

        inputs:
            loc: the coordinates of the center point of the target
        outputs:
            True: The center of the target is outside the image boundaries
            False: The center of the target does not exceed the image boundaries
        """
        if loc[0]<0 or loc[0]>self.w or loc[1]<0 or loc[1]>self.h:
            return True
        return False


class Tracker():                                # Class to realize tracker
    """
    Tracker class

    Integrate target detection, optical flow estimation, and target matching modules to achieve
    multi-target tracking
    """
    def __init__(self, img, w, h):
        """
        Initialization function

        inputs:
            img: first frame of video
            w: image's width
            h: image's height
        """
        self.cur_frame = img                    # current frame
        self.w = w                              # width of image
        self.h = h                              # height of image
        self.max_dis = 30                       # upper limit of bbox deviation
        self.iou_thr = 0.                       # IOU greater than this value is considered a potential matching pair
        self.next_frame = np.zeros_like(self.cur_frame)         # next frame
        self.detector = Detector(nosave=True)                   # object detector (YOLOv5)
        self.flownet = OpticalFlow(net='FlowNet2CS')            # Optical Flow model (FlowNet2CS)
        self.match_method = 'KM'                # Targets matcher
        if self.match_method == 'KM':
            self.matcher = KM()                 # KM matcher
        else:
            self.matcher = Hungarian()                              # Hungarian matcher
        preds = self.detector.detect(self.cur_frame, '0')
        self.targets = Targets(preds, self.w, self.h)           # tracked targets

    def predict(self, next_frame):              # predict location and bbox of tracked targets with Optical Flow
        """
        Targets' position estimation based on optical flow method

        inputs:
            frame: next frame of video
        """
        self.next_frame = next_frame
        self.flow = self.flownet.getflow(self.cur_frame, self.next_frame)

        """
        Use bbox to obtain the average movement information from the optical flow graph for 
        position estimation
        """
        for name, t in enumerate(self.targets.target_list):
            x1, y1, x2, y2 = self.targets.target_list[t].bbox.astype(int)
            x1 = np.max(np.array([x1, 0])).astype(int)
            x2 = np.max(np.array([x2, 0])).astype(int)
            y1 = np.max(np.array([y1, 0])).astype(int)
            y2 = np.max(np.array([y2, 0])).astype(int)
            x1 = np.min(np.array([x1, self.w])).astype(int)
            x2 = np.min(np.array([x2, self.w])).astype(int)
            y1 = np.min(np.array([y1, self.h])).astype(int)
            y2 = np.min(np.array([y2, self.h])).astype(int)

            move_x = np.mean(self.flow[y1:y2, x1:x2, 0])
            move_y = np.mean(self.flow[y1:y2, x1:x2, 1])
            self.targets.target_list[t].pre_loc = self.targets.target_list[t].location + np.array([move_x, move_y])
            self.targets.target_list[t].pre_bbox = self.targets.target_list[t].bbox + np.array([move_x, move_y, move_x, move_y])

    # objects matching
    def match(self):                                    # match the predict targets with targets detected in next frame
        """
        Match the estimated position of the tracked target with the actual position of the
        newly detected target

        outputs:
            cur: The image after marking the current position and historical trajectory of
                the target in the current frame
            pred:The image after marking the estimated target position and the actual detection
                position in the next frame
        """
        new_preds = self.detector.detect(self.next_frame, '0')      # detect targets in next frame
        new_targets = Targets(new_preds, self.w, self.h)            # targets in next frame
        """
        Generate an adjacency list for potential matching pairs
        """
        # Match with Kuhn-Munkers Algorithm
        if self.match_method == 'KM':
            cur_num = len(self.targets.target_list)
            new_num = len(new_targets.target_list)
            cur_name = []
            new_name = []
            for name, t in enumerate(self.targets.target_list):
                cur_name.append(t)
            for name, t in enumerate(new_targets.target_list):
                new_name.append(t)

            cost = np.zeros((cur_num, new_num))
            i = 0
            for name1, t1 in enumerate(self.targets.target_list):
                j = 0
                for name2, t2 in enumerate(new_targets.target_list):
                    iou = self.getiou(self.targets.target_list[t1].pre_bbox, new_targets.target_list[t2].bbox)
                    if iou == 0:
                        dis = abs(np.linalg.norm(
                            self.targets.target_list[t1].pre_loc-new_targets.target_list[t2].location))
                        if(dis < self.max_dis):
                            cost[i, j] = 1 - dis/self.max_dis
                    else:
                        if(iou > self.iou_thr):
                            cost[i, j] = 1 + iou
                    j += 1
                i += 1
            cost = (cost*100).astype(int)
            # print(cost.shape)
            match = self.matcher.run(cost)
            cur_list = [cur_name[i[0]] for i in match]
            new_list = [new_name[i[1]] for i in match]

        #  Match with Hungarian Algorithm
        else:
            adjust_map = {}                                 # Biparty Graph
            for name1, t1 in enumerate(self.targets.target_list):
                adjust_map[t1] = []
                for name2, t2 in enumerate(new_targets.target_list):
                    iou = self.getiou(self.targets.target_list[t1].pre_bbox, new_targets.target_list[t2].bbox)
                    if iou > 0:                             # iou > 0
                        adjust_map[t1].append(t2)
                    elif iou==0:                            # iou = 0 & distance < 10
                        if abs(np.linalg.norm(self.targets.target_list[t1].pre_loc-new_targets.target_list[t2].location))<10:
                            adjust_map[t1].append(t2)
                if len(adjust_map[t1]) == 0:
                    adjust_map.pop(t1)
            match = self.matcher.run(adjust_map)

            cur_list = []
            new_list = []
            for i, t in enumerate(match):
                cur_list.append(t)
                new_list.append(match[t])

        # Visualize the Matching Process
        cur = self.cur_frame.copy()
        pred = self.next_frame.copy()
        for name, t in enumerate(self.targets.target_list):             # visualize the result
            cur = self.draw_bbox(cur, self.targets.target_list[t].bbox, self.targets.target_list[t].name)       # tracking result
            cur = self.draw_trajectory(cur, t)                                  # trajectory
            pred = self.draw_bbox(pred, self.targets.target_list[t].pre_bbox, self.targets.target_list[t].name, color=(0, 255, 0))      # predicted location
        cur = self.draw_num(cur)

        self.targets.update(cur_list, new_list, new_targets)            # update the tracked targets

        for name, t in enumerate(new_targets.target_list):
            pred = self.draw_bbox(pred, new_targets.target_list[t].bbox)            # new detect result in next frame
        pred = self.draw_num(pred)

        self.cur_frame = self.next_frame            # time lapse
        return cur, pred


    def draw_num(self, img):                                    # visual the counting number
        """
        Visualize the total number of tracked objects in the upper left corner of the image
        Currently only supports explicitly a certain class of targets

        input:
            img: image to label
        output:
            img: labeled image
        """
        num = "apple num:" + str(self.targets.count_num)
        # Set font format and size
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Get label length and width
        num_size = cv2.getTextSize(num, font, 1, 2)
        # Set the label starting point
        num_origin = np.array([0, 0])
        cv2.putText(img, num, (0, 20), font, 1, (255, 0, 0), 2)
        return img

    def draw_bbox(self, image, xyxy, label=None, color=(0, 0, 255)):        # visualize the bounding box and target's name
        """
        Annotate the bbox of the target in the image

        imputs:
            image: image to label
            xyxy: bbox's information [x1, y1, x2, y2]
            label: Whether to label the target ID
            color: The color used for the label
        """
        # Get the label of the bbox to be labeled
        # bbox = [xl, yl, xr, yr]
        bbox1 = xyxy.astype(int)
        label1 = label
        if label1:
            # Set font format and size
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Get label length and width
            label_size1 = cv2.getTextSize(label1, font, 1, 2)

            # Set the label starting point
            text_origin1 = np.array([bbox1[0], bbox1[1] - label_size1[0][1]])

            cv2.rectangle(image, tuple(text_origin1), tuple(text_origin1 + label_size1[0]),
                          color=color, thickness=-1)  # thickness=-1 Indicates that the rectangle is filled with color
            # 1 is font scaling, 2 is self-weight
            cv2.putText(image, label1, (bbox1[0], bbox1[1] - 5), font, 1, (255, 255, 255), 2)

        cv2.rectangle(image, (bbox1[0], bbox1[1]), (bbox1[2], bbox1[3]),
                      color=color, thickness=2)
        return image

    def draw_trajectory(self, image, t):                    # visualize the trajectory of tracked target
        """
        The historical trajectory of the tracked object at the label in the image

        inputs:
            image:  image to label
            t: target's ID
        return:
            image: lageled image
        """
        traj = self.targets.target_list[t].history
        traj_head = self.targets.target_list[t].his_head
        traj_len = self.targets.target_list[t].his_len
        traj_head = (traj_head-traj_len+1+10) % 10
        for i in range(0, traj_len-1):
            idx0 = (traj_head+i) % 10
            idx1 = (traj_head+i+1) % 10
            pt1 = traj[idx0, :].astype(int)
            pt2 = traj[idx1, :].astype(int)
            cv2.line(image, pt1, pt2, (255, 0, 0), thickness=2, lineType=4)
        return image

    def getiou(self, boxA, boxB):                           # calculate the IOU
        """
        Calculate the IOU of two bboxes

        inputs:
            boxA: the first bbox [xa1, ya1, xa2, ya2]
            boxB: the second bbox [xb1, yb1, xb2, yb2]
        output:
            iou: IOU of boxA and boxB in the range of [0, 1]
        """
        boxA = [int(x) for x in boxA]
        boxB = [int(x) for x in boxB]
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

if __name__ == '__main__':
    filename = r"48"
    video_path = r'/media/wang/文件/Work/硕士论文/农业巡检/Data/产量估计/{}.mp4'.format(filename)
    save_path_ff = r'/home/wang/Project/MOT/run/{}_detect.mp4'.format(filename)
    save_path_fn = r'/home/wang/Project/MOT/run/{}_pred+detect.mp4'.format(filename)

    cap = cv2.VideoCapture(video_path)

    if cap.isOpened():
        print("Video is opened")
    else:
        print("Video not opened")
        exit(1)

    # 16:9   --->   800*450
    w = 800
    h = 450

    # Output parameters
    fps = 30
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    img_size = (w, h)
    detect = cv2.VideoWriter(save_path_ff, fourcc, fps, img_size)
    pred_detect = cv2.VideoWriter(save_path_fn, fourcc, fps, img_size)

    cv2.namedWindow("detect", 0)
    cv2.resizeWindow("detect", w, h)
    cv2.namedWindow("pred_detect", 0)
    cv2.resizeWindow("pred_detect", w, h)

    ret, img = cap.read()
    img = cv2.resize(img, (w, h))
    tracker = Tracker(img, w, h)
    i = 0
    while True:
        i+=1
        print("------------------------------{}".format(i))
        ret, img = cap.read()
        if not ret:
            print("Done!")
            break
        img = cv2.resize(img, (w, h))
        tracker.predict(img)
        cur, pred = tracker.match()

        detect.write(cur)
        pred_detect.write(pred)
        cv2.imshow("detect", cur)
        cv2.imshow("pred_detect", pred)

    print("Total {} apples was found".format(tracker.targets.count_num))

    detect.release()
    pred_detect.release()
    cap.release()
    cv2.destroyAllWindows()
