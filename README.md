# FINAL PROJECT : Mamba YOLO for object detection specific task head

## Getting started

### 1. Installation

Mamba YOLO is developed based on `torch==2.3.0` `pytorch-cuda==12.1` and `CUDA Version==12.6`. Make sure you use this.

#### 2.Clone Project 

```bash
git clone this github
```

#### 3.Create and activate a conda environment.
```bash
conda create -n mbyolo -y python=3.11
conda activate mbyolo
```

#### 4. Install torch

```bash
pip3 install torch===2.3.0 torchvision torchaudio
```

use this if code was error when you build selective scan 
```bash

```

#### 5. Install Dependencies
```bash
pip install seaborn thop timm einops
cd selective_scan && pip install -v . --no-build-isolation && cd ..
pip install -v -e .
```

#### 6. Prepare YOLO Format Dataset
Make sure your dataset structure as follows:
```
├── dataset
│   ├── test
│   │   ├── images
│   │   └── labels
│   ├── train
│   │   ├── images
│   │   └── labels
│   ├── valid
│   │   ├── images
│   │   ├── labels
|   |── data.yaml
```

#### 7. Training Mamba-YOLO-T
```bash
Look file:  mbyolo_train.py for train mamba yolo
```

## Acknowledgement

This repo is modified from open source real-time object detection codebase [Mamba YOLO](https://github.com/HZAI-ZJNU/Mamba-YOLO).

