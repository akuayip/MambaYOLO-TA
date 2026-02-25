from ultralytics import YOLO
import argparse
import os

ROOT = os.path.abspath('.') + "/"


def parse_opt():
    parser = argparse.ArgumentParser(description="Mamba-YOLO Evaluation: FLOPs, Params, Latency, Precision, Recall, mAP50, mAP50-95")
    parser.add_argument('--weights',  type=str,   default=None,                                                        help='path to trained .pt weights (required for val metrics)')
    parser.add_argument('--config',   type=str,   default=ROOT + 'ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml',help='model config yaml (used if --weights not provided)')
    parser.add_argument('--data',     type=str,   default=ROOT + 'ultralytics/cfg/datasets/coco.yaml',                 help='dataset yaml path')
    parser.add_argument('--imgsz',    type=int,   default=640,                                                          help='input image size')
    parser.add_argument('--batch',    type=int,   default=32,                                                           help='batch size for validation')
    parser.add_argument('--device',   type=str,   default='0',                                                          help='cuda device, e.g. 0 or cpu')
    parser.add_argument('--workers',  type=int,   default=8,                                                            help='dataloader workers')
    parser.add_argument('--half',     action='store_true',                                                              help='use FP16 half-precision inference')
    parser.add_argument('--project',  type=str,   default=ROOT + 'output_dir/eval',                                    help='directory to save results')
    parser.add_argument('--name',     type=str,   default='mambayolo_eval',                                             help='save to project/name')
    return parser.parse_args()


if __name__ == '__main__':
    opt = parse_opt()

    # ── Load model ──────────────────────────────────────────
    if opt.weights and os.path.isfile(opt.weights):
        print(f"[INFO] Loading weights: {opt.weights}")
        model = YOLO(opt.weights)
    else:
        print(f"[INFO] No weights provided — loading config only: {opt.config}")
        model = YOLO(opt.config)

    # ── FLOPs & Params ──────────────────────────────────────
    print("\n========== MODEL INFO (Params & GFLOPs) ==========")
    model.info(verbose=True, imgsz=opt.imgsz)

    # ── Precision, Recall, mAP50, mAP50-95, Latency ────────
    print("\n========== VALIDATION METRICS ===================")
    results = model.val(
        data=opt.data,
        imgsz=opt.imgsz,
        batch=opt.batch,
        device=opt.device,
        workers=opt.workers,
        half=opt.half,
        project=opt.project,
        name=opt.name,
    )

    # ── Print Summary ───────────────────────────────────────
    print("\n========== RESULTS SUMMARY ======================")
    rd = results.results_dict
    print(f"  Precision  (B) : {rd.get('metrics/precision(B)', 'N/A'):.4f}")
    print(f"  Recall     (B) : {rd.get('metrics/recall(B)',    'N/A'):.4f}")
    print(f"  mAP50      (B) : {rd.get('metrics/mAP50(B)',     'N/A'):.4f}")
    print(f"  mAP50-95   (B) : {rd.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
    print(f"  Fitness        : {rd.get('fitness',              'N/A'):.4f}")

    spd = results.speed
    print(f"\n  Latency (ms/img):")
    print(f"    Preprocess   : {spd.get('preprocess',  0):.2f} ms")
    print(f"    Inference    : {spd.get('inference',   0):.2f} ms")
    print(f"    Postprocess  : {spd.get('postprocess', 0):.2f} ms")
    print(f"    Total        : {sum(spd.values()):.2f} ms")
    print("=================================================\n")
