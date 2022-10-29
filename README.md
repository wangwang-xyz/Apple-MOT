# Apple-MOT

---

Code of paper: [Research on UAV Online Visual Tracking Algorithm based on YOLOv5 and FlowNet2 for Apple Yield Inspection](https://ieeexplore.ieee.org/document/9903925/)

---

![algorithm architecture](./img/Algorithm%20Architecture.png)

## Abstract

* A real-time apple tracking and yield estimation algorithm
* YOLOv5 and FlowNet2 are integrated into Tracking-by-Detecting framework
* The accuracy of apple detection is 85.5%
* Apple tracking and counting accuracy is 92.51%


## Quick Start
### Python requirements

* python == 3.10.4
* numpy == 1.22.3
* pytorch == 1.11.0
* torchvision == 0.12.0
* cuda == 11.3.1
* cudnn == 8.2.1
* opencv == 4.5.4
* tensorboard == 2.8.0

### Installation

Download the code

```shell
git clone https://github.com/wangwang-xyz/Apple-MOT.git
```

Download the data in [here](https://pan.baidu.com/s/1BtX1u6M-M_xdVlu5DOT8fQ?pwd=5gwr).
Extract code: 5gwr 

Add your GPU into FlowNet2 config files:

* flownet2/networks/channelnorm_package/setup.py
* flownet2/networks/correlation_package/setup.py
* flownet2/networks/resample2d_package/setup.py

```python
nvcc_args = [
    '-gencode', 'arch=compute_50,code=sm_50',
    '-gencode', 'arch=compute_52,code=sm_52',
    '-gencode', 'arch=compute_60,code=sm_60',
    '-gencode', 'arch=compute_61,code=sm_61',
    '-gencode', 'arch=compute_70,code=sm_70',
    '-gencode', 'arch=compute_86,code=sm_86',
    '-gencode', 'arch=compute_70,code=compute_70'
    # '-gencode', 'arch=compute_XX,code=sm_XX',
    # you can check in Nvidia website
]
```

Then install flownet2

```shell
cd flownet2
bash install.sh
```

At last, install yolov5

```shell
cd ..
cd yolov5
pip install -r requirements.txt  # install
```

The original pre-trained parameters for YOLOv5 
and FlowNet2 can be downloaded from their github websites, respectively.

## Inference

The core code is written in src/

Change the video root in track.py before run it
```shell
cd ..
cd src
python tracker.py
```
