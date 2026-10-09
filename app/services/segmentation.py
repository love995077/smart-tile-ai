"""MobileSAM promptable segmentation (replaces the SegFormer-b5 semantic model).

The TinyViT image encoder is ~5M params, so it fits comfortably next to CLIP
and Depth Anything on a 4GB GPU. Image embeddings are cached per image, so
each extra click only re-runs the tiny mask decoder.
"""
import os

import numpy as np
import torch

from app.config import DEVICE, MOBILE_SAM_HF_FILE, MOBILE_SAM_HF_REPO, MOBILE_SAM_LOCAL_PATH
from app.services.gpu import LazyModel, LRUCache, image_key, inference


def _load_predictor():
    from mobile_sam import SamPredictor, sam_model_registry

    if os.path.exists(MOBILE_SAM_LOCAL_PATH):
        checkpoint = MOBILE_SAM_LOCAL_PATH
    else:
        from huggingface_hub import hf_hub_download
        checkpoint = hf_hub_download(MOBILE_SAM_HF_REPO, MOBILE_SAM_HF_FILE)

    sam = sam_model_registry["vit_t"](checkpoint=None)
    # map_location keeps CUDA-saved weights loadable on CPU-only machines.
    sam.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    sam.to(DEVICE).eval()
    return SamPredictor(sam)


_predictor = LazyModel("MobileSAM (vit_t)", _load_predictor)
_embeddings = LRUCache(size=4)


def _set_image(predictor, image_np):
    key = image_key(image_np)
    cached = _embeddings.get(key)
    if cached is not None:
        predictor.features, predictor.original_size, predictor.input_size = cached
        predictor.is_image_set = True
        return
    predictor.set_image(image_np)
    _embeddings.put(key, (predictor.features, predictor.original_size, predictor.input_size))


def _contains(logits, points):
    h, w = logits.shape
    return all(logits[int(np.clip(y, 0, h - 1)), int(np.clip(x, 0, w - 1))] > 0 for x, y in points)


def predict_logits(image_np, points, labels, require_points=None):
    """Runs MobileSAM for click prompts.

    image_np: HxWx3 RGB uint8. points: Nx2 (x, y) pixel coords. labels: N ints (1 = include, 0 = exclude).
    require_points: optional (x, y) points the returned mask should contain. SAM's most
    confident candidate does not always include the click itself (e.g. a click on the
    edge of a window frame), so the best-scoring candidate that does is preferred.
    Returns full-resolution mask logits (float32 HxW, > 0 means inside).
    """
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)

    predictor = _predictor.get()
    with inference():
        _set_image(predictor, image_np)

        # One positive click is ambiguous (a tile, a rug, the whole floor), so let SAM
        # propose three masks and keep the most confident one. With more clicks the
        # prompt is specific enough for a single mask.
        single = int((labels == 1).sum()) <= 1
        masks, scores, low_res = predictor.predict(
            point_coords=points, point_labels=labels, multimask_output=single, return_logits=True
        )
        order = np.argsort(-scores)
        best = int(order[0])
        if require_points:
            best = next((int(i) for i in order if _contains(masks[i], require_points)), best)

        # Second pass fed with the first low-res logits; this cleans up blotchy edges.
        refined, _, _ = predictor.predict(
            point_coords=points,
            point_labels=labels,
            mask_input=low_res[best:best + 1],
            multimask_output=False,
            return_logits=True,
        )
    if require_points and not _contains(refined[0], require_points) and _contains(masks[best], require_points):
        return masks[best].astype(np.float32)
    return refined[0].astype(np.float32)


def predict_box_masks(image_np, boxes, chunk=8):
    """Binary masks (N x H x W bool) for xyxy pixel boxes.

    A box is an unambiguous prompt, so one decoder pass per box suffices, and boxes are
    decoded in batches against the cached image embedding.
    """
    h, w = image_np.shape[:2]
    if len(boxes) == 0:
        return np.zeros((0, h, w), dtype=bool)
    predictor = _predictor.get()
    out = []
    with inference():
        _set_image(predictor, image_np)
        b = torch.as_tensor(np.asarray(boxes, dtype=np.float32), device=predictor.device)
        b = predictor.transform.apply_boxes_torch(b, predictor.original_size)
        for i in range(0, len(b), chunk):
            masks, _, _ = predictor.predict_torch(point_coords=None, point_labels=None, boxes=b[i:i + chunk],
                                                  multimask_output=False)
            out.append(masks[:, 0].cpu().numpy())
    return np.concatenate(out).astype(bool)


def detect_surface(image_np, surface_type="floor"):
    """Heuristic floor / wall mask for visual search, prompted with fixed points.

    MobileSAM has no class labels, so instead of ADE20K classes we click where the
    surface almost always is: the bottom band for floors, the upper-middle band for walls.
    Returns a uint8 mask (0/255).
    """
    h, w = image_np.shape[:2]
    if surface_type == "wall":
        points = [(0.5 * w, 0.30 * h), (0.25 * w, 0.35 * h), (0.75 * w, 0.35 * h), (0.5 * w, 0.95 * h)]
        labels = [1, 1, 1, 0]
    else:
        points = [(0.5 * w, 0.92 * h), (0.2 * w, 0.95 * h), (0.8 * w, 0.95 * h), (0.5 * w, 0.15 * h)]
        labels = [1, 1, 1, 0]
    logits = predict_logits(image_np, points, labels)
    return ((logits > 0) * 255).astype(np.uint8)
