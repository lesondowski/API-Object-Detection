# Semi-supervised YOLO (teacher-student + pseudo-labeling)
# Yêu cầu: ultralytics, opencv-python, tqdm, albumentations (nếu muốn)
# pip install ultralytics opencv-python tqdm albumentations --quiet

import os
import shutil
import time
from pathlib import Path
from tqdm import tqdm
import random
import json
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import ruamel.yaml as ry

# ----------------------------
# 1. CONFIG
# ----------------------------
# Paths (chỉnh theo project của bạn)
LABELED_DATA = "dataset_labeled"    # chứa images/labels/ và data.yaml cho labeled
UNLABELED_IMAGES = "dataset_unlab/images"  # chỉ ảnh (no labels)
WORK_DIR = "semi_workdir"
os.makedirs(WORK_DIR, exist_ok=True)

# YOLO model & device
PRETRAIN_WEIGHTS = "yolov8n.pt"     # hoặc path tới checkpoint
DEVICE = 0                          # 0 => cuda:0, "cpu" for CPU

# semi-supervised params
PSEUDO_CONF_THRESH = 0.45   # confidence cutoff để giữ box
PSEUDO_IOU_FILTER = 0.5     # iou threshold để NMS nếu cần
MIN_BOXES_PER_IMAGE = 1     # số box tối thiểu để coi ảnh có pseudo-label
MAX_PSEUDO_PER_ITER = None  # None => không giới hạn
EMA_ALPHA = 0.999           # teacher EMA factor
NUM_ITERATIONS = 3          # số vòng pseudo-labeling -> fine-tune
EPOCHS_SUPERVISED = 10      # epochs huấn luyện ban đầu
EPOCHS_FINE_TUNE = 5        # epochs mỗi lần fine-tune sau khi thêm pseudo

# data yaml name inside labeled dataset (Roboflow/YOLOv8 style)
LABELED_DATA_YAML = os.path.join(LABELED_DATA, "data.yaml")

# output / temp folders
PSEUDO_LABEL_DIR = os.path.join(WORK_DIR, "pseudo_labels")      # store labels .txt (yolo format)
PSEUDO_IMG_DIR = os.path.join(WORK_DIR, "pseudo_images")        # copy images considered with pseudo labels
MERGED_DATA_DIR = os.path.join(WORK_DIR, "merged_dataset")      # final merged dataset structure
os.makedirs(PSEUDO_LABEL_DIR, exist_ok=True)
os.makedirs(PSEUDO_IMG_DIR, exist_ok=True)
os.makedirs(MERGED_DATA_DIR, exist_ok=True)


# ----------------------------
# 2. Helper utilities
# ----------------------------
def ensure_yaml_copy(yaml_src, dest_dir):
    """Copy data.yaml to dest_dir (and adjust paths if needed)"""
    import ruamel.yaml as ry
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with open(yaml_src, 'r') as f:
        content = f.read()
    # naive copy, assumes relative paths; for simplicity just copy file
    with open(dest_dir / "data.yaml", "w") as fw:
        fw.write(content)

def write_yolo_label(txt_path, boxes):
    """boxes: list of [cls, x_center, y_center, w, h] (normalized)"""
    with open(txt_path, "w") as f:
        for b in boxes:
            cls,x,y,w,h = b
            f.write(f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

def nms_boxes(boxes, scores, iou_thresh=0.5):
    """boxes format: (x1,y1,x2,y2) in pixels"""
    if len(boxes)==0:
        return []
    boxes = torch.tensor(boxes, dtype=torch.float32)
    scores = torch.tensor(scores, dtype=torch.float32)
    keep_idx = torch.ops.torchvision.nms(boxes, scores, iou_thresh).cpu().numpy().tolist()
    return keep_idx

def xyxy_to_yolo_norm(box, img_w, img_h):
    """convert [x1,y1,x2,y2] pixel to yolo normalized [x_c, y_c, w, h]"""
    x1,y1,x2,y2 = box
    w = x2 - x1
    h = y2 - y1
    x_c = x1 + w/2
    y_c = y1 + h/2
    return [x_c/img_w, y_c/img_h, w/img_w, h/img_h]

# ----------------------------
# 3. Initialize teacher & student
# ----------------------------
# Student initial model (will be trained)
student = YOLO(PRETRAIN_WEIGHTS)   # loads ultralytics model (PyTorch)
# Teacher: clone student weights to start
teacher = YOLO(PRETRAIN_WEIGHTS)

# helper to update teacher weights by EMA from student
def update_teacher_by_ema(teacher_model, student_model, alpha=EMA_ALPHA):
    try:
        t_params = list(teacher_model.model.parameters())
        s_params = list(student_model.model.parameters())
        for tp, sp in zip(t_params, s_params):
            tp.data.mul_(alpha).add_(sp.data * (1.0 - alpha))
    except Exception as e:
        # fallback: copy state dict (non-EMA)
        teacher_model.model.load_state_dict(student_model.model.state_dict())
        print("EMA update failed, fallback to hard copy:", e)

# ----------------------------
# 4. Train supervised student initial
# ----------------------------
def train_supervised(student_model, data_yaml, epochs=EPOCHS_SUPERVISED, device=DEVICE, save_dir=None):
    """Train student from scratch / fine-tune on labeled set"""
    if save_dir is None:
        save_dir = os.path.join(WORK_DIR, f"supervised_train_{int(time.time())}")
    print(f"[Train supervised] data={data_yaml} epochs={epochs} device={device} save_dir={save_dir}")
    student_model.train(data=data_yaml, epochs=epochs, device=device, imgsz=640, workers=4, batch=16, patience=50, save=True, project=save_dir)
    # after train, return path to best.pt
    # ultralytics stores best in runs/..., but .train returns results with .best model? We will search
    # For simplicity, assume best weights stored under save_dir/runs/detect/train/weights/best.pt
    # The exact path depends on ultralytics version; user can adapt accordingly.
    candidate = os.path.join(save_dir, "train", "weights", "best.pt")
    if os.path.exists(candidate):
        return candidate
    # fallback: search runs
    for p in Path(".").rglob("best.pt"):
        return str(p)
    return None

# ----------------------------
# 5. Generate pseudo labels from teacher
# ----------------------------
def generate_pseudo_labels(teacher_model, unlabeled_images_dir, output_label_dir, output_img_dir,
                           conf_thresh=PSEUDO_CONF_THRESH, iou_filter=PSEUDO_IOU_FILTER, min_boxes=MIN_BOXES_PER_IMAGE, max_images=None, device=DEVICE):
    """
    Run teacher inference over unlabeled images, produce YOLO txt files in output_label_dir,
    and copy corresponding images to output_img_dir.
    """
    image_files = sorted([str(p) for p in Path(unlabeled_images_dir).rglob("*") if p.suffix.lower() in [".jpg",".jpeg",".png"]])
    if max_images:
        image_files = image_files[:max_images]
    kept = 0
    for img_path in tqdm(image_files, desc="Pseudo-label inference"):
        results = teacher_model.predict(source=img_path, conf=conf_thresh, device=device, imgsz=640, verbose=False)  # ultralytics returns list
        # results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls
        r = results[0]
        if getattr(r, "boxes", None) is None:
            continue
        boxes_xyxy = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, "xyxy") else np.array([])
        scores = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, "conf") else np.array([])
        classes = r.boxes.cls.cpu().numpy().astype(int) if hasattr(r.boxes, "cls") else np.array([])

        if len(scores)==0:
            continue

        # optional NMS (torchvision nms expects torch tensors)
        keep_idx = list(range(len(scores)))
        if len(scores) > 1:
            keep_idx = nms_boxes(boxes_xyxy.tolist(), scores.tolist(), iou_thresh=iou_filter)

        # build selected boxes list
        img = cv2.imread(img_path)
        h,w = img.shape[:2]
        final_boxes = []
        for idx in keep_idx:
            if scores[idx] < conf_thresh:
                continue
            xyxy = boxes_xyxy[idx]
            cls = int(classes[idx])
            # convert to yolo normalized
            yolo_box = xyxy_to_yolo_norm(xyxy, w, h)
            final_boxes.append([cls, *yolo_box])

        if len(final_boxes) < min_boxes:
            continue

        # save label file
        img_name = Path(img_path).name
        txt_name = Path(img_name).with_suffix(".txt").name
        out_txt = os.path.join(output_label_dir, txt_name)
        out_img = os.path.join(output_img_dir, img_name)
        write_yolo_label(out_txt, final_boxes)
        shutil.copy(img_path, out_img)
        kept += 1
    print(f"[Pseudo] kept images: {kept}")
    return kept

# ----------------------------
# 6. Merge labeled dataset + pseudo dataset to create merged 'data.yaml'
# ----------------------------
def create_merged_dataset(labeled_root, pseudo_img_dir, pseudo_label_dir, merged_root):
    """
    Prepare structure:
    merged_root/
      images/train/...
      images/val/...
      labels/train/...
      labels/val/...
      data.yaml
    We'll copy original labeled train/val, and add pseudo images to train.
    """
    merged_root = Path(merged_root)
    if merged_root.exists():
        shutil.rmtree(merged_root)
    (merged_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (merged_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (merged_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (merged_root / "labels" / "val").mkdir(parents=True, exist_ok=True)

    # copy labeled data (assumes standard structure dataset/images/train, dataset/images/val, dataset/labels/train, etc)
    def copy_tree(src, dst):
        if not os.path.exists(src):
            return
        for f in Path(src).glob("*"):
            if f.is_file():
                shutil.copy(str(f), str(Path(dst) / f.name))

    copy_tree(os.path.join(labeled_root, "images", "train"), os.path.join(merged_root, "images", "train"))
    copy_tree(os.path.join(labeled_root, "images", "val"), os.path.join(merged_root, "images", "val"))
    copy_tree(os.path.join(labeled_root, "labels", "train"), os.path.join(merged_root, "labels", "train"))
    copy_tree(os.path.join(labeled_root, "labels", "val"), os.path.join(merged_root, "labels", "val"))

    # add pseudo images + labels to train
    copy_tree(pseudo_img_dir, os.path.join(merged_root, "images", "train"))
    copy_tree(pseudo_label_dir, os.path.join(merged_root, "labels", "train"))

    # create data.yaml by reading labeled data.yaml and adjusting paths
    with open(os.path.join(labeled_root, "data.yaml"), "r") as f:
        data_yaml = f.read()
    # naive replacement: assume train/val paths are relative 'images/train'
    # we'll write a merged data.yaml that points to merged_root
    import ruamel.yaml as ry
    yaml = ry.YAML()
    d = yaml.load(data_yaml)
    d['train'] = str(Path(merged_root) / "images" / "train")
    d['val'] = str(Path(merged_root) / "images" / "val")
    # classes/names stay same
    with open(Path(merged_root) / "data.yaml", "w") as fw:
        yaml.dump(d, fw)
    return str(Path(merged_root) / "data.yaml")

# ----------------------------
# 7. Main semi-supervised loop
# ----------------------------
def semi_supervised_training_loop(student_model, teacher_model,
                                  labeled_yaml,
                                  unlabeled_images_dir,
                                  iters=NUM_ITERATIONS,
                                  device=DEVICE):
    # 1. train initial supervised student
    print("== Initial supervised training ==")
    initial_weights = train_supervised(student_model, labeled_yaml, epochs=EPOCHS_SUPERVISED, device=device, save_dir=os.path.join(WORK_DIR,"initial_train"))
    if initial_weights:
        # load weights into student and teacher
        student_model = YOLO(initial_weights)
        teacher_model = YOLO(initial_weights)
    else:
        print("Warning: initial_weights not found; continuing with pretrained student object")

    for it in range(iters):
        print(f"\n=== Semi iteration {it+1}/{iters} ===")
        # 2. Update teacher by EMA of student
        update_teacher_by_ema(teacher_model, student_model, alpha=EMA_ALPHA)
        # 3. Generate pseudo labels from teacher
        # clear previous pseudo folder to avoid duplicates
        shutil.rmtree(PSEUDO_LABEL_DIR, ignore_errors=True)
        shutil.rmtree(PSEUDO_IMG_DIR, ignore_errors=True)
        os.makedirs(PSEUDO_LABEL_DIR, exist_ok=True)
        os.makedirs(PSEUDO_IMG_DIR, exist_ok=True)

        kept = generate_pseudo_labels(teacher_model, unlabeled_images_dir, PSEUDO_LABEL_DIR, PSEUDO_IMG_DIR, conf_thresh=PSEUDO_CONF_THRESH, iou_filter=PSEUDO_IOU_FILTER, min_boxes=MIN_BOXES_PER_IMAGE, device=device)
        if kept == 0:
            print("No pseudo-labels generated; finishing loop.")
            break

        # 4. Merge labeled + pseudo into merged dataset and create data.yaml
        merged_yaml = create_merged_dataset(LABELED_DATA, PSEUDO_IMG_DIR, PSEUDO_LABEL_DIR, MERGED_DATA_DIR)

        # 5. Fine-tune student on merged dataset
        print(f"[Fine-tune] training on merged dataset: {merged_yaml}")
        new_weights_path = train_supervised(student_model, merged_yaml, epochs=EPOCHS_FINE_TUNE, device=device, save_dir=os.path.join(WORK_DIR, f"finetune_iter_{it}"))
        # 6. load new weights into student for next iter
        if new_weights_path:
            student_model = YOLO(new_weights_path)

    # final model is student_model
    return student_model, teacher_model

# ----------------------------
# 8. Run the pipeline (example)
# ----------------------------
if __name__ == "__main__":
    # run the semi-supervised training loop
    final_student, final_teacher = semi_supervised_training_loop(student, teacher, LABELED_DATA_YAML, UNLABELED_IMAGES, iters=NUM_ITERATIONS, device=DEVICE)
    # save final student weights
    final_student.model.save(os.path.join(WORK_DIR, "final_student_weights.pt"))
    print("Semi-supervised training finished. final weights saved.")
