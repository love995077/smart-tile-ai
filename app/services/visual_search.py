import torch
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np

# Automatically use GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading Visual Search AI (CLIP)...")
MODEL_ID = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(MODEL_ID)
model = CLIPModel.from_pretrained(MODEL_ID).to(device)

CATALOG_FOLDER = "catalog_tiles"
catalog_embeddings = {}

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
    global catalog_embeddings
    if not os.path.exists(CATALOG_FOLDER):
        os.makedirs(CATALOG_FOLDER)
        return

    print("Indexing catalog for Patterns AND True Colors...")
    
    for root, dirs, files in os.walk(CATALOG_FOLDER):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                filepath = os.path.join(root, filename)
                relative_name = os.path.relpath(filepath, CATALOG_FOLDER).replace("\\", "/")
                
                try:
                    image = Image.open(filepath).convert("RGB")
                    
                    # 1. Extract the AI Pattern (CLIP)
                    inputs = processor(images=image, return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = model.get_image_features(**inputs)
                        features = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
                        features = features / features.norm(p=2, dim=-1, keepdim=True)
                    
                    # 2. Extract the Shadow-Proof Color Fingerprint
                    color_fingerprint = get_color_fingerprint(image)
                    
                    # 3. Save BOTH to memory
                    catalog_embeddings[relative_name] = {
                        "pattern": features.cpu().numpy(),
                        "color": color_fingerprint
                    }
                except Exception as e:
                    print(f"Error indexing {relative_name}: {e}")

def find_matching_tiles(query_image_pil, top_n=1, min_score=30.0): 
    if not catalog_embeddings:
        return []

    # 1. Process Query Image
    inputs = processor(images=query_image_pil.convert("RGB"), return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
        query_features = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
        query_features = query_features / query_features.norm(p=2, dim=-1, keepdim=True)
        query_features = query_features.cpu().numpy()
        
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