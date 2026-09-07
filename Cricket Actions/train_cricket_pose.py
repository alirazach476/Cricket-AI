"""
Cricket Actions - YOLOv8-Pose Training Script
Optimized for local training on NVIDIA Quadro T1000 (4GB VRAM)

USAGE:
1. Put this script INSIDE your dataset project folder (the one containing
   data.yaml, README.dataset, README.roboflow, train/, valid/, test/)
2. Install deps (one time):
   pip install ultralytics torch --index-url https://download.pytorch.org/whl/cu118
3. Run:
   python train_cricket_pose.py
"""

import os
import sys
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO

# ------------------------------------------------------------------
# CONFIG - edit these if needed
# ------------------------------------------------------------------
PROJECT_DIR   = Path(__file__).resolve().parent   # folder containing data.yaml
DATA_YAML     = PROJECT_DIR / "data.yaml"
MODEL_WEIGHTS = "yolov8n-pose.pt"   # nano-pose = best fit for 4GB VRAM
EPOCHS        = 50
IMG_SIZE      = 640
BATCH_SIZE    = 8        # safe for 4GB VRAM; drop to 4 if you hit OOM
WORKERS       = 2        # keep low on laptops (avoid CPU/RAM bottleneck)
RUN_NAME      = "cricket_actions_pose_yolov8n"
PATIENCE      = 15       # early stopping patience


def fix_roboflow_data_yaml(yaml_path: Path):
    """
    Roboflow-exported data.yaml often has train/valid/test paths written as
    relative paths (e.g. '../train/images') that break once the folder is
    moved or run from a different working directory. This rewrites them to
    ABSOLUTE paths based on where data.yaml actually lives, so training
    never fails with 'dataset not found' errors.
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    base = yaml_path.parent

    for split in ["train", "val", "valid", "test"]:
        if split in data and data[split]:
            raw = str(data[split])
            # strip any leading ../ or ./ and re-resolve against base
            clean = raw.replace("\\", "/").lstrip("./")
            candidate = (base / clean).resolve()

            # Roboflow sometimes uses 'valid' key, ultralytics expects 'val'
            if not candidate.exists():
                # try common Roboflow folder names directly
                guess_folder = "valid" if split in ("val", "valid") else split
                candidate = (base / guess_folder / "images").resolve()

            data[split] = str(candidate)

    # ultralytics expects the key 'val', not 'valid'
    if "valid" in data and "val" not in data:
        data["val"] = data.pop("valid")

    fixed_path = base / "data_fixed.yaml"
    with open(fixed_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print(f"[INFO] Fixed data.yaml written to: {fixed_path}")
    for k in ["train", "val", "test"]:
        if k in data:
            exists = Path(data[k]).exists()
            print(f"   {k}: {data[k]}  {'OK' if exists else '!! MISSING !!'}")

    return fixed_path


def get_device():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[INFO] CUDA GPU detected: {name} ({vram:.1f} GB VRAM)")
        return 0
    print("[WARN] No CUDA GPU detected, falling back to CPU (will be slow).")
    return "cpu"


def main():
    if not DATA_YAML.exists():
        sys.exit(f"[ERROR] data.yaml not found at {DATA_YAML}. "
                  f"Place this script in the same folder as data.yaml.")

    fixed_yaml = fix_roboflow_data_yaml(DATA_YAML)
    device = get_device()

    print(f"[INFO] Loading base model: {MODEL_WEIGHTS}")
    model = YOLO(MODEL_WEIGHTS)

    print(f"[INFO] Starting training: {EPOCHS} epochs, imgsz={IMG_SIZE}, batch={BATCH_SIZE}")
    model.train(
        data=str(fixed_yaml),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        device=device,
        patience=PATIENCE,
        project=str(PROJECT_DIR / "runs_pose"),
        name=RUN_NAME,
        exist_ok=True,
        amp=True,          # mixed precision -> big VRAM/speed win on T1000
        cache=False,        # keep False on laptops with limited RAM
        val=True,
        plots=True,
        save=True,
        save_period=10,     # checkpoint every 10 epochs (resume safety)
    )

    print("[INFO] Training complete. Running validation on best weights...")
    best_weights = PROJECT_DIR / "runs_pose" / RUN_NAME / "weights" / "best.pt"
    if best_weights.exists():
        val_model = YOLO(str(best_weights))
        val_model.val(data=str(fixed_yaml), imgsz=IMG_SIZE, device=device)
        print(f"[INFO] Best weights saved at: {best_weights}")
    else:
        print("[WARN] best.pt not found, check runs_pose folder manually.")


if __name__ == "__main__":
    main()
