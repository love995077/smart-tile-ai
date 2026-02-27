from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
import numpy as np
from PIL import Image
import cv2
import uuid
import os
import mimetypes

# The AI foreground extractor
from rembg import remove 

from app.services.segmentation import detect_surface
from app.services.perspective import apply_tiles
from app.services.visual_search import load_catalog, find_matching_tiles

router = APIRouter()

# --- INITIALIZATION ---
load_catalog()

# --- HELPER 1: DYNAMIC ARCHITECTURAL GROUT GENERATOR ---
def format_as_tile_with_grout(img_np, grout_color=(170, 170, 170)):
    """Draws a crisp, realistic cement-grey border around the tile."""
    crop_px = 3
    h_orig, w_orig = img_np.shape[:2]
    if h_orig > crop_px*2 and w_orig > crop_px*2:
        img_np = img_np[crop_px:h_orig-crop_px, crop_px:w_orig-crop_px]
        
    # THE FIX: Dynamic Grout Thickness (1.5% of tile height) 
    # This guarantees the grout line scales perfectly whether the tile is huge or tiny!
    grout_thickness = max(2, int(img_np.shape[0] * 0.015))
        
    tiled_with_grout = cv2.copyMakeBorder(
        img_np,
        top=0, bottom=grout_thickness,
        left=0, right=grout_thickness,
        borderType=cv2.BORDER_CONSTANT,
        value=grout_color
    )
    return tiled_with_grout

# --- HELPER 2: ARCHITECTURAL "SANDPAPER" ---
def smooth_architectural_mask(mask, kernel_size=15):
    if mask is None or not np.any(mask): return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    smoothed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel)
    blurred = cv2.GaussianBlur(smoothed, (21, 21), 0)
    _, thresholded = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    return thresholded

def clean_mask_alpha(mask, blur_size=5):
    if mask is None or not np.any(mask): return mask
    blurred = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    return blurred

def alpha_blend(background, foreground, mask):
    alpha = mask.astype(np.float32) / 255.0
    if len(alpha.shape) == 2:
        alpha = alpha[..., None]
    blended = (foreground.astype(np.float32) * alpha) + (background.astype(np.float32) * (1.0 - alpha))
    return np.clip(blended, 0, 255).astype(np.uint8)

# --- HELPER 3: SEPARATE LIGHTING ENGINES ---
def apply_floor_lighting(original_img, tiled_img, mask):
    light_map = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    light_map = cv2.medianBlur(light_map, 15)
    light_map = cv2.GaussianBlur(light_map, (31, 31), 0)
    
    light_map_f = light_map.astype(np.float32) / 255.0
    mean_brightness = np.mean(light_map_f[mask > 0]) if np.any(mask) else np.mean(light_map_f)
    
    normalized_light = ((light_map_f - mean_brightness) * 0.8) + 1.0
    normalized_light = np.clip(normalized_light, 0.45, 1.1)
    
    light_map_3c = cv2.cvtColor(normalized_light, cv2.COLOR_GRAY2RGB)
    tiled_float = tiled_img.astype(np.float32) / 255.0
    blended = (tiled_float * light_map_3c) * 255.0
    return np.clip(blended, 0, 255).astype(np.uint8)

def apply_wall_lighting(original_img, tiled_img, mask):
    light_map = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    light_map = cv2.GaussianBlur(light_map, (45, 45), 0)
    
    light_map_f = light_map.astype(np.float32) / 255.0
    mean_brightness = np.mean(light_map_f[mask > 0]) if np.any(mask) else np.mean(light_map_f)
    mean_brightness += 1e-5
    
    normalized_light = light_map_f / mean_brightness
    normalized_light = np.clip(normalized_light, 0.65, 1.05)
    
    light_map_3c = cv2.cvtColor(normalized_light, cv2.COLOR_GRAY2RGB)
    tiled_float = tiled_img.astype(np.float32) / 255.0
    blended = (tiled_float * light_map_3c) * 255.0
    return np.clip(blended, 0, 255).astype(np.uint8)


# --- ENDPOINT 1: VISUALIZATION ---
@router.post("/visualize/")
async def visualize(
    room: UploadFile = File(...), 
    floor_tile: Optional[UploadFile] = File(None),
    wall_tile: Optional[UploadFile] = File(None),
    floor_scale: float = Form(1.0),
    wall_scale: float = Form(1.0)
):
    if not floor_tile and not wall_tile:
        raise HTTPException(status_code=400, detail="Upload a tile image.")

    os.makedirs("outputs", exist_ok=True)
    room_img = Image.open(room.file).convert("RGB")
    room_np = np.array(room_img)
    final_result = room_np.copy()

    base_floor_raw = detect_surface(room_np, surface_type="floor")
    base_wall_raw = detect_surface(room_np, surface_type="wall")

    smooth_wall = smooth_architectural_mask(base_wall_raw, kernel_size=15)
    smooth_floor = smooth_architectural_mask(base_floor_raw, kernel_size=15)
    
    if np.any(smooth_floor):
        smooth_wall = cv2.bitwise_and(smooth_wall, cv2.bitwise_not(smooth_floor))

    try:
        fg_mask_pil = remove(room_img, only_mask=True)
        fg_mask = np.array(fg_mask_pil)
        if len(fg_mask.shape) == 3: fg_mask = cv2.cvtColor(fg_mask, cv2.COLOR_RGB2GRAY)
        _, fg_mask = cv2.threshold(fg_mask, 10, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_fg = np.zeros_like(fg_mask)
        img_area = room_np.shape[0] * room_np.shape[1]
        for c in contours:
            if cv2.contourArea(c) < (img_area * 0.40): 
                cv2.drawContours(clean_fg, [c], -1, 255, thickness=cv2.FILLED)
        fg_mask = clean_fg
    except Exception:
        fg_mask = np.zeros(room_np.shape[:2], dtype=np.uint8)

    hsv = cv2.cvtColor(room_np, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    _, white_mask = cv2.threshold(v, 240, 255, cv2.THRESH_BINARY)
    _, color_mask = cv2.threshold(s, 100, 255, cv2.THRESH_BINARY)
    
    global_protection = cv2.bitwise_or(fg_mask, white_mask)
    global_protection = cv2.bitwise_or(global_protection, color_mask)
    
    contours, _ = cv2.findContours(global_protection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_protection = np.zeros_like(global_protection)
    for c in contours:
        if cv2.contourArea(c) > 300:
            cv2.drawContours(clean_protection, [c], -1, 255, thickness=cv2.FILLED)

    paste_back_mask = cv2.erode(clean_protection, np.ones((5, 5), np.uint8), iterations=1)
    soft_paste_back = clean_mask_alpha(paste_back_mask, blur_size=5)

    # 5. PROCESS FLOOR (THE PIPELINE FIX)
    if floor_tile:
        f_tile_img = Image.open(floor_tile.file).convert("RGB")
        # We apply grout directly to the raw, high-definition tile upload!
        f_tile_np = format_as_tile_with_grout(np.array(f_tile_img))

        soft_floor = clean_mask_alpha(smooth_floor, blur_size=7)
        # We pass your scale slider value directly into perspective.py!
        tiled_raw = apply_tiles(final_result, smooth_floor, f_tile_np, surface_type="floor", scale=floor_scale)
        tiled_lit = apply_floor_lighting(room_np, tiled_raw, smooth_floor)
        final_result = alpha_blend(final_result, tiled_lit, soft_floor)

    # 6. PROCESS WALL (THE PIPELINE FIX)
    if wall_tile:
        w_tile_img = Image.open(wall_tile.file).convert("RGB")
        w_tile_np = format_as_tile_with_grout(np.array(w_tile_img))

        soft_wall = clean_mask_alpha(smooth_wall, blur_size=7)
        tiled_raw = apply_tiles(final_result, smooth_wall, w_tile_np, surface_type="wall", scale=wall_scale)
        tiled_lit = apply_wall_lighting(room_np, tiled_raw, smooth_wall)
        final_result = alpha_blend(final_result, tiled_lit, soft_wall)

    final_result = alpha_blend(final_result, room_np, soft_paste_back)

    output_path = f"outputs/{uuid.uuid4()}.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(final_result, cv2.COLOR_RGB2BGR))
    return FileResponse(output_path)


# --- ENDPOINT 2: VISUAL SEARCH (Color Histograms) ---
@router.post("/search-tiles/", response_class=FileResponse)
async def search_tiles(query_image: UploadFile = File(...)):
    try:
        query_img = Image.open(query_image.file).convert("RGB")
        room_np = np.array(query_img)
        
        best_filename = None
        highest_score = -1.0
        catalog_dir = "catalog_tiles"

        def calculate_histogram_score(img1_np, img2_path):
            try:
                img2 = cv2.imread(img2_path)
                if img2 is None: return -1.0
                img1_bgr = cv2.cvtColor(img1_np, cv2.COLOR_RGB2BGR)
                hsv1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2HSV)
                hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
                hist1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
                hist2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])
                cv2.normalize(hist1, hist1, alpha=1, norm_type=cv2.NORM_L1)
                cv2.normalize(hist2, hist2, alpha=1, norm_type=cv2.NORM_L1)
                distance = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
                score = (1.0 - distance) * 100.0
                return score
            except Exception as e:
                return -1.0

        def search_and_update_best(crop_np):
            nonlocal best_filename, highest_score
            if not os.path.exists(catalog_dir): return
            for filename in os.listdir(catalog_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    filepath = os.path.join(catalog_dir, filename)
                    score = calculate_histogram_score(crop_np, filepath)
                    if score > highest_score:
                        highest_score = score
                        best_filename = filename

        def extract_texture_swatch(img_np, mask, box_size=150):
            y_idx, x_idx = np.where(mask > 0)
            if len(y_idx) == 0: return None
            center_y, center_x = int(np.median(y_idx)), int(np.median(x_idx))
            h, w = img_np.shape[:2]
            y1, y2 = max(0, center_y - box_size), min(h, center_y + box_size)
            x1, x2 = max(0, center_x - box_size), min(w, center_x + box_size)
            return img_np[y1:y2, x1:x2]

        search_and_update_best(room_np)
        try:
            floor_mask = detect_surface(room_np, surface_type="floor")
            floor_swatch = extract_texture_swatch(room_np, floor_mask)
            if floor_swatch is not None: search_and_update_best(floor_swatch)
        except Exception:
            pass
        try:
            wall_mask = detect_surface(room_np, surface_type="wall")
            wall_swatch = extract_texture_swatch(room_np, wall_mask)
            if wall_swatch is not None: search_and_update_best(wall_swatch)
        except Exception:
            pass

        if not best_filename:
            raise HTTPException(status_code=404, detail="No matching products found in catalog.")

        filepath = os.path.join(catalog_dir, best_filename)
        content_type, _ = mimetypes.guess_type(filepath)
        return FileResponse(filepath, media_type=content_type or "image/jpeg")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")