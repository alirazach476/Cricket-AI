"""
train_drs_yolov8.py
--------------------
Filters your Roboflow-exported YOLOv8 dataset (train/valid/test, each with
images/ + labels/) to keep ONLY: bat, pad, stumps, pitch
(drops: balls, batsman - since you already trained those separately)

Then trains YOLOv8 for 100 epochs on your local NVIDIA Quadro T1000.

SETUP (run once):
    pip install ultralytics pyyaml

USAGE:
    1. Edit the CONFIG section below (set DATASET_DIR to your dataset folder)
    2. Run: python train_drs_yolov8.py
"""

import shutil
from pathlib import Path

import yaml

# ============================================================
# CONFIG - EDIT THESE BEFORE RUNNING
# ============================================================

# Folder that contains data.yaml, train/, valid/, test/  (the one in your screenshot)
DATASET_DIR = r"C:\Users\DELL\Downloads\drs\Drs detection"

# New folder that will be created with only the classes you want
FILTERED_DIR = r"C:\Users\DELL\Downloads\drs\Drs detection_filtered"

# Classes to KEEP (must match names in your data.yaml, case-insensitive)
KEEP_CLASSES = ["bat", "pad", "stumps", "pitch"]

# If True: images that end up with ZERO of the kept classes are skipped entirely.
# If False: they are still copied over as background images (empty label file).
SKIP_EMPTY_IMAGES = True

EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 8          # Quadro T1000 = 4GB VRAM, keep this low. Try 4 if you get CUDA OOM.
MODEL = "yolov8n.pt"    # nano = lightest/fastest, best fit for T1000. Use yolov8s.pt if you want more accuracy and have time.
DEVICE = 0              # Quadro T1000 GPU
WORKERS = 2             # dataloader workers, keep low on laptops

RUN_PROJECT = "drs_detection_runs"
RUN_NAME = "bat_pad_stumps_pitch"

# ============================================================


def load_data_yaml(dataset_dir):
    yaml_path = Path(dataset_dir) / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found at {yaml_path}")
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return data


def get_original_names(data):
    names = data["names"]
    if isinstance(names, dict):
        # some Roboflow exports store names as {0: 'a', 1: 'b', ...}
        names = [names[i] for i in sorted(names, key=int)]
    return list(names)


def build_class_map(original_names, keep_classes):
    """
    Returns:
        old_to_new: dict {old_class_id: new_class_id}  (only for kept classes)
        new_names:  list of kept class names, ordered by new id
    """
    keep_lower = [c.lower().strip() for c in keep_classes]
    old_to_new = {}
    new_names = []
    for old_id, name in enumerate(original_names):
        if name.lower().strip() in keep_lower:
            old_to_new[old_id] = len(new_names)
            new_names.append(name)
    return old_to_new, new_names


def filter_split(dataset_dir, filtered_dir, split, old_to_new):
    img_src = Path(dataset_dir) / split / "images"
    lbl_src = Path(dataset_dir) / split / "labels"
    img_dst = Path(filtered_dir) / split / "images"
    lbl_dst = Path(filtered_dir) / split / "labels"

    if not lbl_src.exists() or not img_src.exists():
        print(f"  [skip] {split}: images/ or labels/ folder not found")
        return 0, 0

    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    kept, skipped = 0, 0

    for lbl_file in sorted(lbl_src.glob("*.txt")):
        new_lines = []
        with open(lbl_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                old_id = int(parts[0])
                if old_id in old_to_new:
                    new_id = old_to_new[old_id]
                    new_lines.append(" ".join([str(new_id)] + parts[1:]))

        # find matching image (any common extension)
        img_file = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            candidate = img_src / (lbl_file.stem + ext)
            if candidate.exists():
                img_file = candidate
                break

        if not new_lines and SKIP_EMPTY_IMAGES:
            skipped += 1
            continue

        if img_file:
            shutil.copy2(img_file, img_dst / img_file.name)
        with open(lbl_dst / lbl_file.name, "w") as f:
            f.write("\n".join(new_lines) + ("\n" if new_lines else ""))
        kept += 1

    return kept, skipped


def main():
    print("Reading original data.yaml...")
    data = load_data_yaml(DATASET_DIR)
    original_names = get_original_names(data)
    print(f"Original classes ({len(original_names)}): {original_names}")

    old_to_new, new_names = build_class_map(original_names, KEEP_CLASSES)
    if not new_names:
        raise ValueError(
            "None of KEEP_CLASSES matched names in data.yaml. "
            f"data.yaml has: {original_names} -- check spelling/case."
        )
    if len(new_names) != len(KEEP_CLASSES):
        missing = set(c.lower() for c in KEEP_CLASSES) - set(n.lower() for n in new_names)
        print(f"  [warning] Could not find these in data.yaml: {missing}")

    print(f"Keeping classes -> new ids: {dict(zip(new_names, range(len(new_names))))}")

    for split in ["train", "valid", "test"]:
        print(f"Filtering {split}...")
        kept, skipped = filter_split(DATASET_DIR, FILTERED_DIR, split, old_to_new)
        print(f"  {split}: kept {kept} images, skipped {skipped}")

    new_yaml = {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(new_names),
        "names": new_names,
    }
    new_yaml_path = Path(FILTERED_DIR) / "data.yaml"
    with open(new_yaml_path, "w") as f:
        yaml.dump(new_yaml, f, default_flow_style=False)

    print(f"\nFiltered dataset ready at: {FILTERED_DIR}")
    print(f"New data.yaml: {new_yaml_path}\n")

    # ============================================================
    # TRAINING
    # ============================================================
    from ultralytics import YOLO

    print("Starting YOLOv8 training...")
    model = YOLO(MODEL)
    model.train(
        data=str(new_yaml_path),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        project=RUN_PROJECT,
        name=RUN_NAME,
        patience=30,
        cache=False,
    )

    print(f"\nDone. Results/weights in: {RUN_PROJECT}/{RUN_NAME}/weights/best.pt")


if __name__ == "__main__":
    main()
