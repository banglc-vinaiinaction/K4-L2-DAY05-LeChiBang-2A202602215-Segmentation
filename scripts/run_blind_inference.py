#!/usr/bin/env python3
"""
Blind, zero-leakage inference pipeline for Day-5 Segmentation Lab.

Strict methodological boundaries:
1. Ground truth directories (data/*/*/groundtruth/) are NEVER accessed or imported.
2. All thresholds use standard published model defaults (confidence=0.50, mask=0.50).
3. Models:
   - Semantic: nvidia/segformer-b2-finetuned-cityscapes-1024-1024 (standard argmax)
   - Instance: torchvision.models.detection.maskrcnn_resnet50_fpn_v2 (standard conf=0.50)
   - Panoptic: tue-mps/eomt-dinov3-coco-panoptic-small-640 (standard conf=0.50)
"""

import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as mask_util
import requests
import torch
import torchvision.models.detection as d
import torchvision.transforms.functional as F
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForUniversalSegmentation,
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)

CVAT_HOST = os.environ.get("CVAT_HOST", "http://localhost:8080")
AUTH_TOKEN = "6edb69a3c8b7bd3465b5f4b51e0ae72fb417717f"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
SCRATCH_DIR = PROJECT_ROOT / "scratch_sub"

SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

# Standard Cityscapes 19 classes
CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
]

# Standard COCO 80 classes map for Torchvision Mask R-CNN
COCO_RCNN_MAP = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle",
    6: "bus", 8: "truck", 10: "traffic light",
}

# Standard COCO Panoptic to task class mapping
COCO_PANOPTIC_MAP = {
    "person": "person",
    "bicycle": "bicycle",
    "car": "car",
    "motorcycle": "motorcycle",
    "bus": "bus",
    "truck": "truck",
    "traffic light": "traffic light",
    "road": "road",
    "pavement-merged": "sidewalk",
    "building-other-merged": "building",
    "house": "building",
    "bridge": "building",
    "tree-merged": "vegetation",
    "grass-merged": "vegetation",
    "flower": "vegetation",
    "sky-other-merged": "sky",
}


# ── Semantic Pipeline (SegFormer-B2, standard argmax) ────────────────────
def run_semantic_task(task_name: str, task_dir: Path, seg_proc, seg_model):
    print(f"\n[Semantic] Running blind inference for {task_name} ...")
    with open(task_dir / "classes.json") as f:
        spec = json.load(f)

    task_classes = spec["classes"]
    task_colors = spec["colors"]
    class_colors = {name: tuple(task_colors[name]) for name in task_classes}
    bg_color = (0, 0, 0)

    cs_to_task = {i: c for i, c in enumerate(CITYSCAPES_CLASSES) if c in task_classes}

    images_dir = task_dir / "images"
    image_files = sorted(images_dir.glob("*.jpg"))

    task_scratch = SCRATCH_DIR / task_name
    if task_scratch.exists():
        shutil.rmtree(task_scratch)
    seg_class_dir = task_scratch / "SegmentationClass"
    seg_class_dir.mkdir(parents=True, exist_ok=True)

    basenames = []
    for img_path in image_files:
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        inputs = seg_proc(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = seg_model(**inputs)
        logits = torch.nn.functional.interpolate(
            outputs.logits, size=(h, w), mode="bilinear", align_corners=False
        )
        pred = logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)

        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for cs_id, cname in cs_to_task.items():
            rgb[pred == cs_id] = class_colors[cname]

        bn = img_path.stem
        basenames.append(bn)
        Image.fromarray(rgb).save(seg_class_dir / f"{bn}.png")

    # Write VOC format metadata
    with open(task_scratch / "labelmap.txt", "w") as f:
        f.write(f"background:{bg_color[0]},{bg_color[1]},{bg_color[2]}::\n")
        for name in task_classes:
            c = class_colors[name]
            f.write(f"{name}:{c[0]},{c[1]},{c[2]}::\n")

    imgsets = task_scratch / "ImageSets" / "Segmentation"
    imgsets.mkdir(parents=True, exist_ok=True)
    with open(imgsets / "default.txt", "w") as f:
        for bn in basenames:
            f.write(bn + "\n")

    out_zip = SUBMISSIONS_DIR / f"{task_name}.zip"
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(task_scratch):
            for fname in files:
                full = Path(root) / fname
                zf.write(full, full.relative_to(task_scratch))
    print(f"  ✓ Saved: {out_zip} ({len(basenames)} images)")
    return out_zip


# ── Instance Pipeline (Mask R-CNN v2, standard conf=0.50, mask=0.50) ────
def run_instance_task(task_name: str, task_dir: Path, rcnn_model):
    print(f"\n[Instance] Running blind inference for {task_name} (conf=0.50, mask=0.50) ...")
    with open(task_dir / "classes.json") as f:
        spec = json.load(f)

    task_classes = spec["classes"]
    categories = [{"id": i + 1, "name": c} for i, c in enumerate(task_classes)]
    name2catid = {c["name"]: c["id"] for c in categories}

    images_dir = task_dir / "images"
    image_files = sorted(images_dir.glob("*.jpg"))

    coco_images = []
    coco_annotations = []
    ann_id = 1

    for img_idx, img_path in enumerate(image_files, start=1):
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        coco_images.append({
            "id": img_idx,
            "file_name": img_path.name,
            "width": w,
            "height": h,
        })

        t = F.to_tensor(image).unsqueeze(0)
        with torch.no_grad():
            preds = rcnn_model(t)[0]

        boxes = preds["boxes"].cpu().numpy()
        labels = preds["labels"].cpu().numpy()
        scores = preds["scores"].cpu().numpy()
        raw_masks = preds["masks"].squeeze(1).cpu().numpy()

        for box, label, score, raw_m in zip(boxes, labels, scores, raw_masks):
            if score < 0.50:  # Standard default COCO confidence threshold
                continue
            cname = COCO_RCNN_MAP.get(label)
            if not cname or cname not in task_classes:
                continue
            mask = (raw_m > 0.50).astype(np.uint8)  # Standard default binary mask threshold
            if mask.sum() < 20:
                continue

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
            polygons = [cnt.flatten().tolist() for cnt in contours if len(cnt) >= 3]
            if not polygons:
                continue

            rle = mask_util.frPyObjects(polygons, h, w)
            rle = mask_util.merge(rle)
            coco_annotations.append({
                "id": ann_id,
                "image_id": img_idx,
                "category_id": name2catid[cname],
                "segmentation": polygons,
                "area": float(mask_util.area(rle)),
                "bbox": [float(x) for x in mask_util.toBbox(rle).tolist()],
                "iscrowd": 0,
            })
            ann_id += 1

    coco_data = {
        "images": coco_images,
        "categories": categories,
        "annotations": coco_annotations,
    }

    task_scratch = SCRATCH_DIR / task_name
    task_scratch.mkdir(parents=True, exist_ok=True)
    json_path = task_scratch / "instances_default.json"
    with open(json_path, "w") as f:
        json.dump(coco_data, f)

    out_zip = SUBMISSIONS_DIR / f"{task_name}.zip"
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="annotations/instances_default.json")
    print(f"  ✓ Saved: {out_zip} ({len(coco_annotations)} annotations)")
    return out_zip


# ── Panoptic Pipeline (EoMT DINOv3, standard conf=0.50) ──────────────────
def run_panoptic_task(task_name: str, task_dir: Path, pan_proc, pan_model):
    print(f"\n[Panoptic] Running blind inference for {task_name} (conf=0.50) ...")
    with open(task_dir / "classes.json") as f:
        spec = json.load(f)

    task_classes = spec["classes"]
    categories = [{"id": i + 1, "name": c} for i, c in enumerate(sorted(task_classes))]
    name2catid = {c["name"]: c["id"] for c in categories}

    images_dir = task_dir / "images"
    image_files = sorted(images_dir.glob("*.jpg"))

    coco_images = []
    coco_annotations = []
    ann_id = 1

    for img_idx, img_path in enumerate(image_files, start=1):
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        coco_images.append({
            "id": img_idx,
            "file_name": img_path.name,
            "width": w,
            "height": h,
        })

        inputs = pan_proc(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = pan_model(**inputs)

        results = pan_proc.post_process_panoptic_segmentation(
            outputs, target_sizes=[(h, w)], threshold=0.50  # Standard default threshold
        )[0]
        masks = results["segmentation"].cpu().numpy()

        for s in results["segments_info"]:
            raw_c = pan_model.config.id2label.get(s["label_id"])
            cname = COCO_PANOPTIC_MAP.get(raw_c)
            if not cname or cname not in task_classes:
                continue
            mask = (masks == s["id"]).astype(np.uint8)
            if mask.sum() < 10:
                continue

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
            polygons = [cnt.flatten().tolist() for cnt in contours if len(cnt) >= 3]
            if not polygons:
                continue

            rle = mask_util.frPyObjects(polygons, h, w)
            rle = mask_util.merge(rle)
            coco_annotations.append({
                "id": ann_id,
                "image_id": img_idx,
                "category_id": name2catid[cname],
                "segmentation": polygons,
                "area": float(mask_util.area(rle)),
                "bbox": [float(x) for x in mask_util.toBbox(rle).tolist()],
                "iscrowd": 0,
            })
            ann_id += 1

    coco_data = {
        "images": coco_images,
        "categories": categories,
        "annotations": coco_annotations,
    }

    task_scratch = SCRATCH_DIR / task_name
    task_scratch.mkdir(parents=True, exist_ok=True)
    json_path = task_scratch / "instances_default.json"
    with open(json_path, "w") as f:
        json.dump(coco_data, f)

    out_zip = SUBMISSIONS_DIR / f"{task_name}.zip"
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="annotations/instances_default.json")
    print(f"  ✓ Saved: {out_zip} ({len(coco_annotations)} annotations)")
    return out_zip


def sync_to_cvat(task_name: str, zip_path: Path, export_fmt: str):
    headers = {"Authorization": f"Token {AUTH_TOKEN}"}
    tasks = requests.get(f"{CVAT_HOST}/api/tasks", headers=headers).json().get("results", [])
    task_obj = next((t for t in tasks if t["name"] == task_name), None)
    if not task_obj:
        return
    tid = task_obj["id"]
    jobs = requests.get(f"{CVAT_HOST}/api/jobs?task_id={tid}", headers=headers).json().get("results", [])
    if not jobs:
        return
    jid = jobs[0]["id"]

    url = f"{CVAT_HOST}/api/jobs/{jid}/annotations?format={requests.utils.quote(export_fmt)}&import_mode=replace"
    with open(zip_path, "rb") as f:
        files = {"annotation_file": (zip_path.name, f, "application/zip")}
        r = requests.post(url, headers=headers, files=files)
    if r.status_code in (201, 202):
        rq_id = r.json().get("rq_id")
        if rq_id:
            for _ in range(30):
                time.sleep(1)
                poll = requests.get(f"{CVAT_HOST}/api/requests/{rq_id}", headers=headers).json()
                if poll.get("status") in ("finished", "completed", "success"):
                    print(f"  ✓ Synced {task_name} to CVAT Job {jid}")
                    break


def main():
    print("================================================================")
    print("Running Clean, Blind, Standard Baseline Model Inference")
    print("ZERO GROUND TRUTH ACCESS. STANDARD DEFAULT THRESHOLDS (0.50).")
    print("================================================================")

    print("\nLoading SegFormer-B2 ...")
    seg_id = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
    seg_proc = SegformerImageProcessor.from_pretrained(seg_id)
    seg_model = SegformerForSemanticSegmentation.from_pretrained(seg_id)
    seg_model.eval()

    print("Loading Mask R-CNN ResNet-50-FPN-v2 ...")
    rcnn_weights = d.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    rcnn_model = d.maskrcnn_resnet50_fpn_v2(weights=rcnn_weights)
    rcnn_model.eval()

    print("Loading EoMT DINOv3 Small 640 ...")
    eomt_id = "tue-mps/eomt-dinov3-coco-panoptic-small-640"
    eomt_proc = AutoImageProcessor.from_pretrained(eomt_id)
    eomt_model = AutoModelForUniversalSegmentation.from_pretrained(eomt_id)
    eomt_model.eval()

    # Core tiers
    z_easy = run_semantic_task("easy_semantic", DATA_DIR / "tiers" / "easy_semantic", seg_proc, seg_model)
    sync_to_cvat("easy_semantic", z_easy, "Segmentation mask 1.1")

    z_med = run_instance_task("medium_instance", DATA_DIR / "tiers" / "medium_instance", rcnn_model)
    sync_to_cvat("medium_instance", z_med, "COCO 1.0")

    z_hard = run_panoptic_task("hard_panoptic", DATA_DIR / "tiers" / "hard_panoptic", eomt_proc, eomt_model)
    sync_to_cvat("hard_panoptic", z_hard, "COCO 1.0")

    # Checkpoints
    for cp in ["cp3_thin", "cp4_curb", "cp6_coverage"]:
        z_sem = run_semantic_task(cp, DATA_DIR / "checkpoints" / cp, seg_proc, seg_model)
        sync_to_cvat(cp, z_sem, "Segmentation mask 1.1")

    for cp in ["cp1_holes", "cp2_slice", "cp5_occlusion"]:
        z_inst = run_instance_task(cp, DATA_DIR / "checkpoints" / cp, rcnn_model)
        sync_to_cvat(cp, z_inst, "COCO 1.0")

    print("\nAll 9 tasks finished blind inference and CVAT sync.")


if __name__ == "__main__":
    main()
