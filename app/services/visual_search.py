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

def load_catalog():
    global catalog_embeddings
    if not os.path.exists(CATALOG_FOLDER):
        os.makedirs(CATALOG_FOLDER)
        return

    print("Indexing tile catalog...")
    for filename in os.listdir(CATALOG_FOLDER):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            filepath = os.path.join(CATALOG_FOLDER, filename)
            try:
                # 1. Load and process the image
                image = Image.open(filepath).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(device)
                
                # 2. Extract features and bypass the new Hugging Face wrapper object
                with torch.no_grad():
                    outputs = model.get_image_features(**inputs)
                    # Pull out the raw tensor if it is wrapped in BaseModelOutputWithPooling
                    features = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
                    features = features / features.norm(p=2, dim=-1, keepdim=True)
                
                # 3. Save to memory
                catalog_embeddings[filename] = features.cpu().numpy()
            except Exception as e:
                print(f"Error indexing {filename}: {e}")

def find_matching_tiles(query_image_pil, top_n=1, min_score=60.0): 
    if not catalog_embeddings:
        return []

    # 1. Process Query Image exactly like the catalog images
    inputs = processor(images=query_image_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
        query_features = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
        query_features = query_features / query_features.norm(p=2, dim=-1, keepdim=True)
        query_features = query_features.cpu().numpy()

    results = []
    
    # 2. Compare using Dot Product
    for filename, feat in catalog_embeddings.items():
        score = np.dot(query_features, feat.T)[0][0]
        match_score = round(float(score) * 100, 2)
        
        print(f"Comparing with {filename} ... Score: {match_score}%")
        
        # Only keep it if it passes our minimum score limit
        if match_score >= min_score:
            results.append({
                "filename": filename,
                "match_score": match_score
            })

    # 3. Sort by highest score first
    sorted_results = sorted(results, key=lambda x: x["match_score"], reverse=True)
    
    # 4. Return only the top requested amount
    return sorted_results[:top_n]