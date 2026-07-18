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
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

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

#### 8. Convert best.pt / best.pth to ONNX
```bash
# Otomatis cari best.pt/best.pth dari output_dir training
python mbyolo_export_onnx.py \
    --output-dir output_dir/freeze-backbone/9_lr0.01_b32_optSGD_e200 \
    --imgsz 640 \
    --device cpu

# Atau panggil langsung file weights
python mbyolo_export_onnx.py \
    --weights output_dir/freeze-backbone/9_lr0.01_b32_optSGD_e200/weights/best.pt \
    --output output_dir/freeze-backbone/9_lr0.01_b32_optSGD_e200/weights/best.onnx
```

## Acknowledgement

This repo is modified from open source real-time object detection codebase [Mamba YOLO](https://github.com/HZAI-ZJNU/Mamba-YOLO).
