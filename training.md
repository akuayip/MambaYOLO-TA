
# Training Commands

### 1.1 Download Pretrained Weights (Optional)
```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

### 1.2 Clean Previous Outputs
```bash
rm -rf output_dir dataset/*/labels.cache

rm -rf dataset/*/labels.cache
```

### 1.3  Train from Scratch
```bash
python mbyolo_train.py \
    --epochs 200 \
    --batch-size 16 \
    --lr0 0.01 \
    --optimizer AdamW \
    --amp
```

### 1.4 Fine-tuning with Frozen Backbone
```bash
python mbyolo_train.py \
    --pretrained yolov8n.pt \
    --freeze-backbone \
    --epochs 200 \
    --batch-size 16 \
    --lr0 0.01 \
    --optimizer SGD \
    --amp \
    --project output_dir/freeze-backbone\
    --name backbone_lr0.01_b16_optSGD_e200
```