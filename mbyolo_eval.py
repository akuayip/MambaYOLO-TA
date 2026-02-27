#!/usr/bin/env python3
"""
Mamba-YOLO Evaluation Script
Menghitung: Params, GFLOPs, Latency, Precision, Recall, mAP50, mAP50-95
"""

from ultralytics import YOLO
import argparse
import os

ROOT = os.path.abspath(".") + "/"


def parse_opt():
    parser = argparse.ArgumentParser(description="Mamba-YOLO Evaluation")

    parser.add_argument("--weights", type=str, required=True,
                        help="Path ke trained .pt weights")
    parser.add_argument("--data",    type=str, default="dataset/data.yaml",
                        help="Path ke dataset config YAML")
    parser.add_argument("--imgsz",   type=int, default=640)
    parser.add_argument("--batch",   type=int, default=32)
    parser.add_argument("--device",  type=str, default="0",
                        help="GPU: 0 | cpu")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--half",    action="store_true",
                        help="FP16 inference")
    parser.add_argument("--project", type=str, default="output_dir/eval")
    parser.add_argument("--name",    type=str, default="mambayolo_eval")

    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()

    # ── Load model ────────────────────────────────────────────────
    model = YOLO(opt.weights)

    # ── Validasi: Precision, Recall, mAP, Latency ────────────────
    # Dilakukan dulu agar model sudah di-load ke GPU yang benar
    print("\n========== VALIDATION ==========================")
    results = model.val(
        data    = ROOT + opt.data,
        split   = "test",
        imgsz   = opt.imgsz,
        batch   = opt.batch,
        device  = opt.device,
        workers = opt.workers,
        half    = opt.half,
        project = ROOT + opt.project,
        name    = opt.name,
    )

    # ── Params & GFLOPs ──────────────────────────────────────────
    # Dipanggil setelah val() agar model sudah berada di device yang benar
    # model.info() mengembalikan (n_layers, n_params, n_gradients, gflops)
    print("\n========== MODEL INFO ==========================")
    _, n_params, _, gflops = model.info(verbose=True, imgsz=opt.imgsz)

    # ── Summary ───────────────────────────────────────────────────
    rd  = results.results_dict
    spd = results.speed

    print("\n========== RESULTS SUMMARY =====================")
    print(f"  {'Precision':<18} {rd.get('metrics/precision(B)', 0):.4f}")
    print(f"  {'Recall':<18} {rd.get('metrics/recall(B)',    0):.4f}")
    print(f"  {'mAP@0.5':<18} {rd.get('metrics/mAP50(B)',    0):.4f}")
    print(f"  {'mAP@0.5:0.95':<18} {rd.get('metrics/mAP50-95(B)',0):.4f}")
    print(f"  {'Fitness':<18} {rd.get('fitness',             0):.4f}")
    print(f"  ---")
    print(f"  {'Parameters':<18} {n_params/1e6:.2f}M")
    print(f"  {'GFLOPs':<18} {gflops:.2f}")
    print(f"  ---")
    print(f"  Latency (ms/img):")
    print(f"    {'Preprocess':<16} {spd.get('preprocess',  0):.2f} ms")
    print(f"    {'Inference':<16} {spd.get('inference',   0):.2f} ms")
    print(f"    {'Postprocess':<16} {spd.get('postprocess', 0):.2f} ms")
    total = sum(spd.values())
    print(f"    {'Total':<16} {total:.2f} ms  ({1000/total:.1f} FPS)")
    print("================================================\n")