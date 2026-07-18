#!/usr/bin/env python3
"""
Mamba-YOLO ONNX Export Script
Convert best.pt / best.pth dari output_dir ke format ONNX.
"""

from pathlib import Path
import argparse
import shutil


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = "ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml"


def resolve_path(path):
    """Resolve path relatif dari root project."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def find_best_model(output_dir):
    """Cari best.pt / best.pth di output_dir dan subfolder weights."""
    output_dir = resolve_path(output_dir)
    candidates = (
        output_dir / "weights" / "best.pt",
        output_dir / "best.pt",
        output_dir / "weights" / "best.pth",
        output_dir / "best.pth",
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"best.pt / best.pth tidak ditemukan. Lokasi yang dicek:\n{searched}")


def load_pth_model(weights, config):
    """
    Load checkpoint .pth ke arsitektur Mamba-YOLO.

    Gunakan --config jika .pth dibuat dari varian model selain Mamba-YOLO-T.
    """
    import torch
    from ultralytics import YOLO

    model = YOLO(str(resolve_path(config)))
    ckpt = torch.load(weights, map_location="cpu")

    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
    else:
        state_dict = ckpt

    if hasattr(state_dict, "state_dict"):
        state_dict = state_dict.float().state_dict()

    if not isinstance(state_dict, dict):
        raise TypeError("Format .pth tidak dikenali. Berikan checkpoint berisi state_dict atau model PyTorch.")

    cleaned = {}
    for key, value in state_dict.items():
        key = key[7:] if key.startswith("module.") else key
        cleaned[key] = value

    missing, unexpected = model.model.load_state_dict(cleaned, strict=False)
    print(f"  Loaded .pth checkpoint: {weights}")
    print(f"  Missing keys   : {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")
    return model


def parse_opt():
    parser = argparse.ArgumentParser(description="Convert Mamba-YOLO best.pt / best.pth ke ONNX")

    parser.add_argument("--weights", type=str, default=None,
                        help="Path ke best.pt / best.pth. Jika kosong, dicari dari --output-dir")
    parser.add_argument("--output-dir", type=str, default="output_dir",
                        help="Direktori hasil training yang berisi best.pt/best.pth atau weights/best.pt")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG,
                        help="YAML arsitektur model, diperlukan untuk checkpoint .pth state_dict")
    parser.add_argument("--output", type=str, default=None,
                        help="Path ONNX tujuan. Default: satu folder dengan weights, nama sama .onnx")

    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu",
                        help="GPU: 0 | 0,1 | cpu")
    parser.add_argument("--opset", type=int, default=None,
                        help="ONNX opset. Default mengikuti versi PyTorch")
    parser.add_argument("--dynamic", action="store_true",
                        help="Export dengan dynamic axes untuk batch/height/width")
    parser.add_argument("--simplify", action="store_true",
                        help="Slim/simplify ONNX dengan onnxslim jika tersedia")
    parser.add_argument("--half", action="store_true",
                        help="Export FP16. Gunakan device CUDA, bukan cpu")

    return parser.parse_args()


def main():
    opt = parse_opt()
    weights = resolve_path(opt.weights) if opt.weights else find_best_model(opt.output_dir)

    if not weights.exists():
        raise FileNotFoundError(f"Weights tidak ditemukan: {weights}")

    print("\n========== Mamba-YOLO ONNX Export =============")
    print(f"  Weights : {weights}")
    print(f"  imgsz   : {opt.imgsz} | batch: {opt.batch} | device: {opt.device}")
    print(f"  dynamic : {opt.dynamic} | simplify: {opt.simplify} | half: {opt.half}")
    print("================================================\n")

    if weights.suffix.lower() == ".pth":
        model = load_pth_model(weights, opt.config)
    else:
        from ultralytics import YOLO

        model = YOLO(str(weights))

    exported = Path(model.export(
        format="onnx",
        imgsz=opt.imgsz,
        batch=opt.batch,
        device=opt.device,
        opset=opt.opset,
        dynamic=opt.dynamic,
        simplify=opt.simplify,
        half=opt.half,
    ))

    output = resolve_path(opt.output) if opt.output else weights.with_suffix(".onnx")
    if exported.resolve() != output.resolve():
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(exported), str(output))

    print("\n========== EXPORT SELESAI =====================")
    print(f"  ONNX: {output}")
    print("================================================\n")


if __name__ == "__main__":
    main()
