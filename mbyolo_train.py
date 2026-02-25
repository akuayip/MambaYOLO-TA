#!/usr/bin/env python3
"""
Mamba-YOLO Training Script
Fokus: training saja (termasuk transfer learning + freeze backbone)
Untuk validasi dan model statistics, gunakan mbyolo_val.py
"""

from ultralytics import YOLO
import argparse
import os
import torch

ROOT = os.path.abspath(".") + "/"


def parse_opt():
    parser = argparse.ArgumentParser(description="Mamba-YOLO Training Script")

    # ── Dataset & Model ──────────────────────────────────────────
    parser.add_argument("--data",   type=str, default="dataset/data.yaml",
                        help="Path ke dataset config YAML")
    parser.add_argument("--config", type=str,
                        default="ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml",
                        help="Path ke model architecture YAML")

    # ── Transfer Learning ─────────────────────────────────────────
    parser.add_argument("--pretrained",      type=str, default=None,
                        help="Path ke pretrained weights, e.g. yolov8m.pt")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Freeze backbone (layer 0-8), hanya neck+head yang dilatih")

    # ── Hyperparameters ───────────────────────────────────────────
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch-size",   type=int,   default=8,    dest="batch_size")
    parser.add_argument("--imgsz",        type=int,   default=640)
    parser.add_argument("--lr0",          type=float, default=0.01,
                        help="Learning rate awal")
    parser.add_argument("--lrf",          type=float, default=0.01,
                        help="Faktor LR akhir (lr_final = lr0 * lrf)")
    parser.add_argument("--momentum",     type=float, default=0.937)
    parser.add_argument("--weight-decay", type=float, default=0.0005, dest="weight_decay")
    parser.add_argument("--optimizer",    type=str,   default="SGD",
                        help="SGD | Adam | AdamW | RMSProp")
    parser.add_argument("--amp",    action="store_true",
                        help="Automatic Mixed Precision (FP16)")
    parser.add_argument("--resume", action="store_true",
                        help="Lanjutkan dari checkpoint terakhir")

    # ── System ────────────────────────────────────────────────────
    parser.add_argument("--device",  default="0",
                        help="GPU: 0 | 0,1,2,3 | cpu")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="output_dir/detection")
    parser.add_argument("--name",    default="mambayolo",
                        help="Nama eksperimen (subfolder di dalam project)")

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────

def transfer_weights(model, pretrained_path):
    """
    Transfer weights yang kompatibel dari pretrained .pt ke Mamba-YOLO.

    Hanya layer dengan nama DAN shape yang sama yang di-copy.
    Layer yang tidak cocok (misal head YOLOv8 vs Mamba-YOLO) dilewati
    dan tetap pakai inisialisasi acak — tidak ada yang rusak.
    """
    print(f"\n{'='*60}")
    print(f"  Transfer Learning dari : {pretrained_path}")
    print(f"{'='*60}")

    try:
        ckpt      = torch.load(pretrained_path, map_location="cpu")
        src       = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        src_state = src.float().state_dict() if hasattr(src, "state_dict") else src

        tgt_state      = model.model.state_dict()
        new_state      = {}
        transferred, skipped = 0, 0

        for name, param in tgt_state.items():
            if name in src_state and param.shape == src_state[name].shape:
                new_state[name] = src_state[name]
                transferred += 1
            else:
                new_state[name] = param     # tetap pakai bobot asal
                skipped += 1

        model.model.load_state_dict(new_state, strict=False)
        print(f"  Transferred : {transferred} layer")
        print(f"  Skipped     : {skipped} layer  (shape/nama tidak cocok)")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"  [WARNING] Transfer gagal: {e}")
        print(f"  Lanjut dengan inisialisasi acak\n")

    return model


def freeze_backbone_layers(model):
    """
    Freeze backbone Mamba-YOLO (layer 0–8), neck+head tetap trainable.

    Struktur backbone (dari YAML):
        0  SimpleStem        4  VisionClueMerge    8  SPPF
        1  VSSBlock          5  VSSBlock
        2  VisionClueMerge   6  VisionClueMerge
        3  VSSBlock          7  VSSBlock

    Dengan freeze backbone, hanya mengupdate neck (layer 9–20)
    dan head (layer 21 / Detect). 
    """
    BACKBONE_END = 8

    frozen = trainable = 0
    for name, param in model.model.named_parameters():
        parts = name.split(".")
        if len(parts) >= 2 and parts[0] == "model" and parts[1].isdigit():
            if int(parts[1]) <= BACKBONE_END:
                param.requires_grad = False
                frozen += 1
                continue
        param.requires_grad = True
        trainable += 1

    total = frozen + trainable
    print(f"\n{'='*60}")
    print(f"  Freeze Backbone (layer 0–{BACKBONE_END})")
    print(f"  Frozen    : {frozen:,} / {total:,} parameter")
    print(f"  Trainable : {trainable:,} / {total:,} parameter")
    print(f"{'='*60}\n")
    return model


# ─────────────────────────────────────────────────────────────────

def main():
    opt = parse_opt()

    print(f"\n{'='*60}")
    print(f"  Mamba-YOLO Training")
    print(f"  Config      : {opt.config}")
    print(f"  Pretrained  : {opt.pretrained or 'None (from scratch)'}")
    print(f"  Epochs      : {opt.epochs}  |  Batch: {opt.batch_size}  |  imgsz: {opt.imgsz}")
    print(f"  LR          : {opt.lr0} → {opt.lr0 * opt.lrf:.6f}  |  Optim: {opt.optimizer}")
    print(f"  AMP         : {opt.amp}  |  Freeze Backbone: {opt.freeze_backbone}")
    print(f"  Device      : {opt.device}  |  Workers: {opt.workers}")
    print(f"  Output      : {opt.project}/{opt.name}")
    print(f"{'='*60}\n")

    # 1. Load arsitektur dari YAML
    model = YOLO(ROOT + opt.config)

    # 2. Transfer weights dari pretrained (opsional)
    if opt.pretrained:
        path = opt.pretrained if os.path.isabs(opt.pretrained) else ROOT + opt.pretrained
        model = transfer_weights(model, path)

    # 3. Freeze backbone (opsional, aktifkan untuk fase pretraining)
    if opt.freeze_backbone:
        model = freeze_backbone_layers(model)

    # 4. Training
    model.train(
        data         = ROOT + opt.data,
        epochs       = opt.epochs,
        batch        = opt.batch_size,
        imgsz        = opt.imgsz,
        lr0          = opt.lr0,
        lrf          = opt.lrf,
        momentum     = opt.momentum,
        weight_decay = opt.weight_decay,
        optimizer    = opt.optimizer,
        amp          = opt.amp,
        resume       = opt.resume,
        device       = opt.device,
        workers      = opt.workers,
        project      = ROOT + opt.project,
        name         = opt.name,
    )

    best = os.path.join(ROOT + opt.project, opt.name, "weights", "best.pt")
    print(f"\n  Training selesai. Best model: {best}\n")


if __name__ == "__main__":
    main()