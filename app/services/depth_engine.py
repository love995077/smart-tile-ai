"""Monocular depth with Depth Anything V2 Small (~25M params)."""
import numpy as np
import torch
import torch.nn.functional as F

from app.config import DEPTH_MODEL_ID, DEVICE, DTYPE
from app.services.gpu import LazyModel, LRUCache, image_key, inference


def _load():
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID, dtype=DTYPE)
    model.to(DEVICE).eval()
    return processor, model


_model = LazyModel("Depth Anything V2 Small", _load)
_cache = LRUCache(size=4)


def estimate_depth(image_np):
    """Returns relative disparity (float32 HxW, larger = closer), divided by its max.

    Depth Anything predicts affine-invariant inverse depth: disparity = s / Z + t with
    unknown s and t. The render engine resolves t from vanishing geometry, so the raw
    shift is kept (only the scale is normalised).
    """
    key = image_key(image_np)
    cached = _cache.get(key)
    if cached is not None:
        return cached

    processor, model = _model.get()
    h, w = image_np.shape[:2]
    with inference():
        inputs = processor(images=image_np, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(DEVICE, dtype=DTYPE)
        pred = model(pixel_values=pixel_values).predicted_depth
        pred = F.interpolate(pred[:, None].float(), size=(h, w), mode="bicubic", align_corners=False)
        disparity = pred[0, 0].clamp_min(0).cpu().numpy()
        del pred, pixel_values

    disparity = (disparity / max(float(disparity.max()), 1e-6)).astype(np.float32)
    _cache.put(key, disparity)
    return disparity
