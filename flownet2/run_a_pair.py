import time

import torch
import torchvision.transforms as transforms
import numpy as np
import argparse

from models import FlowNet2, FlowNet2S  # the path is depended on where you create this module
from utils.frame_utils import read_gen  # the path is depended on where you create this module
from utils.flow_utils import visulize_flow_file

if __name__ == '__main__':
    # obtain the necessary args for construct the flownet framework
    parser = argparse.ArgumentParser()

    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)
    
    args = parser.parse_args()

    # initial a Net
    net = FlowNet2S(args).cuda()
    # load the state_dict
    dict = torch.load("/home/wang/Project/MOT/flownet2/checkpoints/FlowNet2-S_checkpoint.pth.tar")
    net.load_state_dict(dict["state_dict"])
    net.eval()

    # load the image pair, you can find this operation in dataset.py
    pim1 = read_gen("/home/wang/Project/MOT/data/OpticalFlow/1.png")
    pim2 = read_gen("/home/wang/Project/MOT/data/OpticalFlow/2.png")
    images = [pim1, pim2]
    images = np.array(images).transpose(3, 0, 1, 2)

    h, w = images.shape[2:]
    pad_h = pad_w = 0
    if h%64 != 0:
        pad_h = int((64-h%64)/2)
    if w%64 != 0:
        pad_w = int((64-w%64)/2)

    transform = transforms.Compose([
        transforms.Pad([pad_w, pad_h], 0)
    ])
    im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0)
    im = transform(im).cuda()
    # im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda()

    # process the image pair to obtian the flow
    start = time.time()
    result = net(im).squeeze()
    end = time.time()
    print("{}ms".format((end-start)*1000))


    # save flow, I reference the code in scripts/run-flownet.py in flownet2-caffe project
    def writeFlow(name, flow):
        f = open(name, 'wb')
        f.write('PIEH'.encode('utf-8'))
        np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
        flow = flow.astype(np.float32)
        flow.tofile(f)
        f.flush()
        f.close()


    data = result.data.cpu().numpy().transpose(1, 2, 0)
    writeFlow("/home/wang/Project/MOT/data/OpticalFlow/1.flo", data)
    visulize_flow_file("/home/wang/Project/MOT/data/OpticalFlow/1.flo", "/home/wang/Project/MOT/data/OpticalFlow/vis/")