"""
Cricket Ball Detection - YOLOv8 Training Script
Optimized for local training on NVIDIA Quadro T1000 (4GB VRAM)

Dataset structure expected (Roboflow export):
project_folder/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml

Install requirements first:
    pip install ultralytics torch --index-url https://download.pytorch.org/whl/cu118
"""

import os
import gc
import torch
from ultralytics import YOLO

# ------------------- CONFIG -------------------
DATASET_YAML = r"data.yaml"           # path to your data.yaml (adjust if needed)
MODEL = "yolov8n.pt"                  # nano model -> best fit for 4GB VRAM
EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 16                      # T1000 has 4GB VRAM, keep batch small
PATIENCE = 5                         # early stopping patience (epochs w/o improvement)
PROJECT_DIR = "runs/train"
RUN_NAME = "cricket_ball_yolov8n"
WORKERS = 2                           # lower workers for laptop CPU
DEVICE = 0                            # GPU index, use 'cpu' if no GPU detected
# ------------------------------------------------


def free_gpu_memory():
    """Reset / clear GPU VRAM memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
    print("GPU VRAM memory cleared.")


def check_gpu():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU detected: {name} | Total VRAM: {total_mem:.2f} GB")
        return 0
    else:
        print("No GPU detected. Falling back to CPU (training will be slow).")
        return "cpu"


def main():
    # Clear any leftover VRAM before starting
    free_gpu_memory()

    device = check_gpu()

    if not os.path.exists(DATASET_YAML):
        raise FileNotFoundError(
            f"data.yaml not found at '{DATASET_YAML}'. "
            f"Update DATASET_YAML in the script to point to your dataset folder."
        )

    # Load pretrained YOLOv8 model
    model = YOLO(MODEL)

    # Train
    results = model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=device,
        workers=WORKERS,
        patience=PATIENCE,          # early stopping
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,
        optimizer="auto",
        cache=False,                # avoid RAM overload on laptop
        amp=True,                   # mixed precision -> saves VRAM
        verbose=True,
        plots=True,
        save=True,
        save_period=5,              # checkpoint every 5 epochs
    )

    print("\nTraining complete.")
    best_model_path = os.path.join(PROJECT_DIR, RUN_NAME, "weights", "best.pt")
    print(f"Best model saved at: {best_model_path}")

    # Validate on the test set (optional but recommended)
    try:
        metrics = model.val(data=DATASET_YAML, split="test", device=device)
        print("Test set evaluation metrics:")
        print(metrics)
    except Exception as e:
        print(f"Validation on test split skipped/failed: {e}")

    # Free model + GPU memory after training
    del model
    free_gpu_memory()


if __name__ == "__main__":
    main()
