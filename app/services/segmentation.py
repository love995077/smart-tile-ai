import torch
import numpy as np
import cv2
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image

MODEL_NAME = "nvidia/segformer-b5-finetuned-ade-640-640"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)
model.to(device)

def detect_surface(image_np, surface_type="floor"):
    orig_h, orig_w = image_np.shape[:2]

    if surface_type == "wall":
        VALID_CLASSES = [0]      # 0 = Wall
    else:
        VALID_CLASSES = [3, 54]  # 3 = Floor, 54 = Stairs

    # Downscale for memory safety
    max_dim = 1024
    scale_factor = 1.0
    if max(orig_h, orig_w) > max_dim:
        scale_factor = max_dim / max(orig_h, orig_w)
        new_w, new_h = int(orig_w * scale_factor), int(orig_h * scale_factor)
        image_for_ai = cv2.resize(image_np, (new_w, new_h))
    else:
        image_for_ai = image_np

    image_pil = Image.fromarray(image_for_ai)
    
    inputs = processor(images=image_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predictions = logits.argmax(dim=1)[0].cpu().numpy()

    surface_mask_small = np.isin(predictions, VALID_CLASSES)

    if not np.any(surface_mask_small):
        return np.zeros((orig_h, orig_w), dtype=np.uint8)

    surface_mask_small_img = (surface_mask_small * 255).astype(np.uint8)

    # --- FIX 1: Strict Boundaries ---
    # We use INTER_NEAREST to keep edges sharp, and we REMOVED the GaussianBlur 
    # so the mask doesn't bleed over the lamp and furniture.
    final_mask = cv2.resize(surface_mask_small_img, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    return final_mask