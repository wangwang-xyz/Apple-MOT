"""
Target position estimation module based on optical flow method
Implementation based on FlowNet2 network model

Written by Shaopeng Wang
"""
import time
import numpy as np
import torch
import torchvision.transforms as transforms
import argparse

import sys
sys.path.append("..")
from flownet2.models import FlowNet2, FlowNet2S, FlowNet2C, FlowNet2CS  # the path is depended on where you create this module
sys.path.remove("..")


class OpticalFlow():                # OpticalFlow with FlowNet2 model
    """
    Encapsulation of FlowNet2 network model
    """
    def __init__(self, net='FlowNet2C'):
        """
        FlowNet2 network model parameter initialization and loading
        """
        parser = argparse.ArgumentParser()

        parser.add_argument('--fp16', action='store_true',
                            help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
        parser.add_argument("--rgb_max", type=float, default=255.)

        args = parser.parse_args()
        self.net_name = net
        if net=='FlowNet2':
            # initial a Net
            self.net = FlowNet2(args).cuda()
            # load the state_dict
            dict = torch.load("../flownet2/checkpoints/FlowNet2_checkpoint.pth.tar")
        if net=='FlowNet2C':
            self.net = FlowNet2C(args).cuda()
            dict = torch.load("../flownet2/checkpoints/FlowNet2-C_checkpoint.pth.tar")
        if net=='FlowNet2S':
            self.net = FlowNet2S(args).cuda()
            dict = torch.load("../flownet2/checkpoints/FlowNet2-S_checkpoint.pth.tar")
        if net=='FlowNet2CS':
            self.net = FlowNet2CS(args).cuda()
            dict = torch.load("../flownet2/checkpoints/FlowNet2-CS_checkpoint.pth.tar")

        self.net.load_state_dict(dict["state_dict"])
        self.net.eval()

    def getflow(self, prev, next):
        """
        Optical Flow Inference Interface

        inputs:
            prev: last frame of the video
            next: current frame of the video
        output:
            data: optical flow
        """
        s = "{}: ".format(self.net_name)

        start = time.time()                 # Preprocess
        images = [prev, next]
        images = np.array(images).transpose(3, 0, 1, 2)

        h, w = images.shape[2:]                 # padding
        pad_h = pad_w = 0
        if h % 64 != 0:
            pad_h = int((64 - h % 64) / 2)
        if w % 64 != 0:
            pad_w = int((64 - w % 64) / 2)

        transform = transforms.Compose([
            transforms.Pad([pad_w, pad_h], 0)
        ])
        im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0)
        im = transform(im).cuda()
        end = time.time()
        s = s + "preproces for {}ms ".format((end-start)*1000)
        # im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda()

        # process the image pair to obtian the flow
        start = time.time()
        result = self.net(im).squeeze()             # get OpticalFlow
        end = time.time()
        s = s + "inference for {}ms ".format((end - start) * 1000)
        print(s)

        data = result.data.cpu().numpy().transpose(1, 2, 0)         # Remove padding
        if pad_h!=0:
            data = data[pad_h:-pad_h, :, :]
        if pad_w!=0:
            data = data[:, pad_w:-pad_w, :]
        return data
