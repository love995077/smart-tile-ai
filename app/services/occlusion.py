"""Zero-click foreground occlusion.

A COCO object detector (D-FINE-small, Apache-2.0) finds people, chairs, laptops,
bottles, plants, monitors, clocks and so on. Each detection box prompts MobileSAM,
whose image embedding is already cached for click masking, giving a full-resolution
instance mask. The union of those masks is subtracted from the clicked surface, so
users never have to right-click on foreground objects.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from app.config import DETECTOR_MODEL_ID, DEVICE, DTYPE
from app.services.gpu import LazyModel, LRUCache, image_key, inference
from app.services.segmentation import predict_box_masks

SCORE_THRESHOLD = 0.45
MAX_OBJECTS = 48
MAX_SURFACE_SHARE = 0.5      # one object mask covering more of the surface than this is a mis-segmentation
EDGE_DILATE_PX = 2           # also take the 1-2px fringe SAM leaves around objects
MIN_CLASS_SAMPLES = 40       # colour samples needed to model the object / the surrounding surface

# The detector uses VOC-style names for some COCO classes.
PRETTY = {"tvmonitor": "monitor", "sofa": "sofa", "diningtable": "table", "pottedplant": "plant",
          "motorbike": "motorcycle", "aeroplane": "airplane", "cell phone": "phone"}


@dataclass
class ForegroundObject:
    label: str
    score: float
    box: tuple          # detector box, xyxy
    region: tuple       # (x0, y0, x1, y1) the mask crop covers
    mask: np.ndarray    # bool crop of the instance mask


def _load():
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    processor = AutoImageProcessor.from_pretrained(DETECTOR_MODEL_ID)
    model = AutoModelForObjectDetection.from_pretrained(DETECTOR_MODEL_ID, dtype=DTYPE).to(DEVICE).eval()
    return processor, model


_detector = LazyModel("Object detector (D-FINE-small)", _load)
_cache = LRUCache(size=4)


def detect_objects(image_np):
    """List of (label, score, xyxy box) above SCORE_THRESHOLD, highest score first."""
    processor, model = _detector.get()
    h, w = image_np.shape[:2]
    with inference():
        inputs = processor(images=image_np, return_tensors="pt")
        outputs = model(pixel_values=inputs["pixel_values"].to(DEVICE, dtype=DTYPE))
        outputs.logits = outputs.logits.float()
        outputs.pred_boxes = outputs.pred_boxes.float()
        res = processor.post_process_object_detection(outputs, target_sizes=[(h, w)],
                                                      threshold=SCORE_THRESHOLD)[0]
        scores = res["scores"].cpu().numpy()
        labels = res["labels"].cpu().numpy()
        boxes = res["boxes"].cpu().numpy()
    order = np.argsort(-scores)[:MAX_OBJECTS]
    dets = []
    for i in order:
        x0, y0, x1, y1 = np.clip(boxes[i], 0, [w - 1, h - 1, w - 1, h - 1])
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        name = model.config.id2label[int(labels[i])]
        dets.append((PRETTY.get(name, name), float(scores[i]), (float(x0), float(y0), float(x1), float(y1))))
    return dets


def foreground_objects(image_np):
    """Detected objects with instance masks, cached per image (clicks re-use them)."""
    key = image_key(image_np)
    cached = _cache.get(key)
    if cached is not None:
        return cached

    h, w = image_np.shape[:2]
    dets = detect_objects(image_np)
    masks = predict_box_masks(image_np, [d[2] for d in dets])
    objects = []
    for (label, score, box), m in zip(dets, masks):
        x0, y0, x1, y1 = box
        mx, my = 0.04 * (x1 - x0) + 2, 0.04 * (y1 - y0) + 2
        rx0, ry0 = int(max(0, x0 - mx)), int(max(0, y0 - my))
        rx1, ry1 = int(min(w, x1 + mx + 1)), int(min(h, y1 + my + 1))
        crop = m[ry0:ry1, rx0:rx1]
        if crop.sum() < 0.04 * (x1 - x0) * (y1 - y0):
            continue                     # SAM found no coherent object inside the box
        objects.append(ForegroundObject(label, score, box, (rx0, ry0, rx1, ry1), crop))
    _cache.put(key, objects)
    return objects


def _centroids(samples, k):
    samples = np.float32(samples)
    if len(samples) > 3000:
        samples = samples[np.random.default_rng(0).choice(len(samples), 3000, replace=False)]
    k = int(min(k, len(samples)))
    cv2.setRNGSeed(0)   # deterministic palettes: the same clicks give the same mask
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, _, centers = cv2.kmeans(samples, k, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
    return centers


def _nearest(feats, centers):
    return np.sqrt(((feats[:, None, :] - centers[None, :, :]) ** 2).sum(-1)).min(axis=1)


def _object_pixels(lab, obj_mask, surface, x0, y0, x1, y1):
    """Contested pixels (in both the object mask and the surface) that really belong to the object.

    A box-prompted mask covers the object's whole outline, including floor seen between
    table legs or through a chair frame. Each contested pixel goes to whichever colour
    model it is closer to: the object (its pixels outside the surface) or the surface
    (surface pixels just around the object). Floor seen through stays floor; chrome bases,
    wheels and shoes that the surface mask leaked onto are removed.
    """
    h, w = surface.shape
    m = obj_mask
    surf = surface[y0:y1, x0:x1] > 0
    contested = m & surf
    obj_only = m & ~surf
    pad = max(8, int(0.15 * max(x1 - x0, y1 - y0)))
    X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
    full_obj = np.zeros((Y1 - Y0, X1 - X0), bool)
    full_obj[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = m
    ring = cv2.dilate(full_obj.astype(np.uint8), np.ones((2 * pad + 1, 2 * pad + 1), np.uint8)) > 0
    floor_ring = ring & ~full_obj & (surface[Y0:Y1, X0:X1] > 0)
    if obj_only.sum() < MIN_CLASS_SAMPLES or floor_ring.sum() < MIN_CLASS_SAMPLES:
        return contested                       # nothing to compare against: trust the detector
    crop = lab[y0:y1, x0:x1]
    obj_c = _centroids(crop[obj_only], 4)
    floor_c = _centroids(lab[Y0:Y1, X0:X1][floor_ring], 4)
    feats = crop[contested]
    is_obj = _nearest(feats, obj_c) < _nearest(feats, floor_c)
    keep = np.zeros_like(contested)
    keep[contested] = is_obj
    # Drop isolated specks; colour noise should not punch holes in the floor.
    keep = cv2.morphologyEx(keep.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    return keep


def subtract_foreground(surface, objects, positive_clicks, image_np=None):
    """Removes detected objects from a binary surface mask.

    Returns (mask uint8, applied objects, excluded uint8 mask). An object is skipped when
    the user clicked inside it (an explicit request to tile there wins over the detector)
    or when its mask would remove most of the surface (a bad box prompt). With the image,
    only the pixels that look like the object are removed (see _object_pixels).
    """
    out = surface.astype(np.uint8).copy()
    excluded = np.zeros_like(out)
    area = max(int(out.sum()), 1)
    kernel = np.ones((2 * EDGE_DILATE_PX + 1, 2 * EDGE_DILATE_PX + 1), np.uint8)
    lab = None
    if image_np is not None:
        lab = cv2.cvtColor(cv2.GaussianBlur(image_np, (3, 3), 0), cv2.COLOR_RGB2LAB).astype(np.float32)
    applied = []
    for obj in objects:
        x0, y0, x1, y1 = obj.region
        m = obj.mask
        overlap = int((out[y0:y1, x0:x1] & m).sum())
        if overlap == 0 or overlap > MAX_SURFACE_SHARE * area:
            continue
        if any(x0 <= x < x1 and y0 <= y < y1 and m[int(y) - y0, int(x) - x0] for x, y in positive_clicks):
            continue
        if lab is not None:
            m = _object_pixels(lab, m, out, x0, y0, x1, y1)
            if not m.any():
                continue
        grown = cv2.dilate(m.astype(np.uint8), kernel) > 0
        hit = grown & (out[y0:y1, x0:x1] > 0)
        excluded[y0:y1, x0:x1][hit] = 1
        out[y0:y1, x0:x1][hit] = 0
        applied.append(obj)
    return out, applied, excluded


def summarize(objects):
    """{'person': 3, 'chair': 5} style counts for the UI."""
    counts = {}
    for obj in objects:
        counts[obj.label] = counts.get(obj.label, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
