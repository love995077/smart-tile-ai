import os
import threading

import numpy as np
from PIL import Image

from app.config import CLIP_MODEL_ID, DEVICE, DTYPE
from app.services.gpu import LazyModel, inference


def _load_clip():
    from transformers import CLIPModel, CLIPProcessor
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID, dtype=DTYPE).to(DEVICE).eval()
    return processor, model


# Loaded on the first search, not at import time.
_clip = LazyModel("Visual Search AI (CLIP)", _load_clip)

CATALOG_FOLDER = "catalog_tiles"
catalog_embeddings = {}
_catalog_loaded = False
_catalog_lock = threading.Lock()


def _embed(image_pil):
    processor, model = _clip.get()
    with inference():
        inputs = processor(images=image_pil.convert("RGB"), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(DEVICE, dtype=DTYPE)
        outputs = model.get_image_features(pixel_values=pixel_values)
        features = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
        features = features.float()
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.cpu().numpy()

# --- THE UPGRADED COLOR BRAIN (SHADOW-PROOF HSV) ---
def get_color_fingerprint(image_pil):
    """Converts to HSV to compare true color (Hue) and ignores Lighting/Shadows (Value)."""
    # Convert to HSV (Hue, Saturation, Value)
    hsv_img = image_pil.convert("HSV").resize((50, 50))
    hsv_np = np.array(hsv_img)
    
    # We only take Hue (Color pigment) and Saturation (Richness). 
    # WE IGNORE Value (Brightness/Shadows) completely!
    color_data = hsv_np[:, :, :2].reshape(-1, 2)
    
    # Sort into mathematical buckets
    hist, _ = np.histogramdd(
        color_data, 
        bins=(8, 8), 
        range=((0, 256), (0, 256))
    )
    
    # Normalize the fingerprint
    return hist.flatten() / hist.sum()

def load_catalog():
    global _catalog_loaded
    with _catalog_lock:
        if _catalog_loaded:
            return
        _catalog_loaded = True
        if not os.path.exists(CATALOG_FOLDER):
            os.makedirs(CATALOG_FOLDER)
            return
        _index_catalog()


def _index_catalog():
    print("Indexing catalog for Patterns AND True Colors...")
    
    for root, dirs, files in os.walk(CATALOG_FOLDER):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                filepath = os.path.join(root, filename)
                relative_name = os.path.relpath(filepath, CATALOG_FOLDER).replace("\\", "/")
                
                try:
                    image = Image.open(filepath).convert("RGB")
                    
                    # 1. Extract the AI Pattern (CLIP)
                    features = _embed(image)
                    
                    # 2. Extract the Shadow-Proof Color Fingerprint
                    color_fingerprint = get_color_fingerprint(image)
                    
                    # 3. Save BOTH to memory
                    catalog_embeddings[relative_name] = {
                        "pattern": features,
                        "color": color_fingerprint
                    }
                except Exception as e:
                    print(f"Error indexing {relative_name}: {e}")

def find_matching_tiles(query_image_pil, top_n=1, min_score=30.0): 
    load_catalog()
    if not catalog_embeddings:
        return []

    # 1. Process Query Image
    query_features = _embed(query_image_pil)

    query_color = get_color_fingerprint(query_image_pil)

    results = []
    
    # 2. Compare using our new Dual-Engine Math
    for relative_name, data in catalog_embeddings.items():
        cat_pattern = data["pattern"]
        cat_color = data["color"]

        # Engine 1: AI Pattern Score
        pattern_score = np.dot(query_features, cat_pattern.T)[0][0] * 100
        
        # Engine 2: Shadow-Proof Color Score
        color_distance = np.linalg.norm(query_color - cat_color)
        color_score = max(0.0, 100.0 - (color_distance / 1.414) * 100)
        
        # Blend them!
        final_score = round(float(pattern_score * 0.5 + color_score * 0.5), 2)
        
        if final_score >= min_score:
            results.append({
                "filename": relative_name,
                "match_score": final_score
            })

    # 3. Sort by highest score first
    sorted_results = sorted(results, key=lambda x: x["match_score"], reverse=True)
    return sorted_results[:top_n]