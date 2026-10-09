import base64
import json
import os
import uuid

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageOps
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.config import MAX_IMAGE_SIDE
from app.DB.database import get_db
from app.models.product import Product, ProductImage
from app.services.render_engine import render_surface
from app.services.sam_masking import build_alpha_matte, make_overlay
from app.services.segmentation import detect_surface
from app.services.visual_search import CATALOG_FOLDER, find_matching_tiles

router = APIRouter()

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")

DB_OFFLINE = {"status": "error", "message": "Database is offline. Please start MySQL."}


# --- HELPERS ---
# Endpoints are plain `def`, so FastAPI runs them in its threadpool and model
# inference never blocks the event loop.

def read_rgb(upload, max_side=None):
    """Returns (RGB array, downscale factor applied)."""
    try:
        img = ImageOps.exif_transpose(Image.open(upload.file)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail=f"'{upload.filename}' is not a readable image.")
    factor = 1.0
    if max_side and max(img.size) > max_side:
        factor = max_side / max(img.size)
        img = img.resize((round(img.width * factor), round(img.height * factor)), Image.LANCZOS)
    return np.array(img), factor


def read_mask(upload, shape):
    """Alpha matte from a PNG: its alpha channel if it has one, else its grey levels."""
    try:
        img = Image.open(upload.file)
    except Exception:
        raise HTTPException(status_code=400, detail="mask_image is not a readable image.")
    if img.mode in ("RGBA", "LA"):
        arr = np.array(img.getchannel("A"))
    else:
        arr = np.array(img.convert("L"))
    if arr.shape != shape:
        arr = cv2.resize(arr, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return arr.astype(np.float32) / 255.0


def parse_clicks(raw, name, factor=1.0):
    """'[[x, y], ...]' or '[{"x":..,"y":..}, ...]' in uploaded-image pixel coords."""
    try:
        data = json.loads(raw or "[]")
        points = [(float(p["x"]), float(p["y"])) if isinstance(p, dict) else (float(p[0]), float(p[1])) for p in data]
    except Exception:
        raise HTTPException(status_code=400, detail=f"{name} must be a JSON list of [x, y] points.")
    return [(x * factor, y * factor) for x, y in points]


def png_data_url(arr):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if arr.ndim == 3 else arr)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


# --- ENDPOINT 1: CLICK-TO-MASK (MobileSAM + ALPHA MATTING) ---
@router.post("/api/get-mask")
def get_mask(
    room_image: UploadFile = File(...),
    positive_clicks: str = Form("[]"),
    negative_clicks: str = Form("[]"),
    auto_exclude: bool = Form(True),
):
    """Returns the soft alpha matte (greyscale PNG, to send back to /api/apply-tile)
    and a tinted overlay of the selection for display.

    auto_exclude: detected foreground objects (people, chairs, laptops, ...) are cut
    out of the surface without any clicks; `auto_excluded` lists them by type."""
    room_np, factor = read_rgb(room_image, MAX_IMAGE_SIDE)
    positive = parse_clicks(positive_clicks, "positive_clicks", factor)
    negative = parse_clicks(negative_clicks, "negative_clicks", factor)
    if not positive:
        raise HTTPException(status_code=400, detail="Add at least one positive (left) click.")

    alpha, info = build_alpha_matte(room_np, positive, negative, auto_exclude=auto_exclude)
    return {
        "width": int(room_np.shape[1]),
        "height": int(room_np.shape[0]),
        "coverage": round(float(alpha.mean()), 4),
        "auto_excluded": info["objects"],
        "mask": png_data_url((alpha * 255).astype(np.uint8)),
        "overlay": png_data_url(make_overlay(room_np, alpha, info["excluded"])),
    }


# --- ENDPOINT 2: PHOTOREALISTIC TILE RENDER ---
@router.post("/api/apply-tile")
def apply_tile(
    room_image: UploadFile = File(...),
    tile_image: UploadFile = File(...),
    mask_image: UploadFile = File(...),
    tile_width: float = Form(600.0),
    tile_height: float = Form(600.0),
    scale: float = Form(1.0),
    is_glossy: bool = Form(False),
    surface_type: str = Form("auto"),
    grout_mm: float = Form(2.0),
):
    """tile_width / tile_height are physical tile dimensions in millimetres.
    surface_type is "auto", "floor" or "wall"."""
    if not (10 <= tile_width <= 5000 and 10 <= tile_height <= 5000):
        raise HTTPException(status_code=400, detail="Tile dimensions must be between 10 and 5000 mm.")

    room_np, _ = read_rgb(room_image, MAX_IMAGE_SIDE)
    tile_np, _ = read_rgb(tile_image, 2048)
    alpha = read_mask(mask_image, room_np.shape[:2])

    try:
        result, info = render_surface(
            room_np, tile_np, alpha,
            tile_w_mm=tile_width, tile_h_mm=tile_height, scale=scale,
            is_glossy=is_glossy, surface_type=surface_type.lower(),
            grout_mm=max(0.0, min(grout_mm, 20.0)),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    os.makedirs("outputs", exist_ok=True)
    output_path = f"outputs/{uuid.uuid4()}.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 94])
    return FileResponse(output_path, media_type="image/jpeg", headers={"X-Render-Info": json.dumps(info)})


# --- CATALOG LISTING (for the frontend tile picker) ---
@router.get("/api/catalog")
def list_catalog():
    tiles = []
    if os.path.isdir(CATALOG_FOLDER):
        for root, _, files in os.walk(CATALOG_FOLDER):
            for filename in sorted(files):
                if filename.lower().endswith(IMAGE_EXTS):
                    rel = os.path.relpath(os.path.join(root, filename), CATALOG_FOLDER).replace("\\", "/")
                    name = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ").strip().title()
                    tiles.append({"name": name, "path": rel, "url": f"/catalog/{rel}"})
    return tiles


# --- ENDPOINT 3: MULTI-SURFACE VISUAL SEARCH (MobileSAM SWATCHES + CLIP) ---
@router.post("/search-tiles/")
def search_tiles(
    request: Request, 
    query_image: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # Fail fast (before running MobileSAM + CLIP) when MySQL is not reachable.
    try:
        db.execute(text("SELECT 1"))
    except OperationalError as e:
        print(f"search-tiles: database unreachable: {e.orig}")
        return JSONResponse(status_code=503, content=DB_OFFLINE)

    try:
        query_img = Image.open(query_image.file).convert("RGB")
        room_np = np.array(query_img)

        def extract_swatch(img_np, mask, box_size=150):
            y_idx, x_idx = np.where(mask > 0)
            if len(y_idx) == 0: return None
            center_y, center_x = int(np.median(y_idx)), int(np.median(x_idx))
            h, w = img_np.shape[:2]
            y1, y2 = max(0, center_y - box_size), min(h, center_y + box_size)
            x1, x2 = max(0, center_x - box_size), min(w, center_x + box_size)
            return Image.fromarray(img_np[y1:y2, x1:x2])

        base_url = str(request.base_url).rstrip("/")

        def format_results(matches):
            formatted = []
            for match in matches:
                filename = match['filename']
                image_url = f"{base_url}/catalog/{filename}"
                
                product = db.query(Product).filter(Product.list_image == filename).first()
                if not product:
                    gallery_img = db.query(ProductImage).filter(ProductImage.product_image == filename).first()
                    if gallery_img:
                        product = gallery_img.product 

                if product:
                    formatted.append({
                        "name": product.name,
                        "price_per_sqft": float(product.price_per_sqft) if product.price_per_sqft else "Call for price",
                        "category": product.category_slug,
                        "product_link": f"/product/{product.slug}", 
                        "image_url": image_url,
                        "match_score": match['match_score']
                    })
                else:
                    formatted.append({
                        "name": "Unknown Tile (Not in DB)",
                        "image_url": image_url,
                        "match_score": match['match_score']
                    })
            return formatted

        final_results = {
            "wall_matches": [],
            "floor_matches": [],
            "overall_matches": []
        }

        # 1. SMART WALL DETECTION (MobileSAM)
        try:
            wall_mask = detect_surface(room_np, surface_type="wall")
            wall_swatch = extract_swatch(room_np, wall_mask)
            if wall_swatch:
                matches = find_matching_tiles(wall_swatch, top_n=3, min_score=10.0)
                final_results["wall_matches"] = format_results(matches)
        except OperationalError:
            raise
        except Exception as e:
            print(f"Wall detection skipped: {e}")

        # 2. SMART FLOOR DETECTION (MobileSAM)
        try:
            floor_mask = detect_surface(room_np, surface_type="floor")
            floor_swatch = extract_swatch(room_np, floor_mask)
            if floor_swatch:
                matches = find_matching_tiles(floor_swatch, top_n=3, min_score=10.0)
                final_results["floor_matches"] = format_results(matches)
        except OperationalError:
            raise
        except Exception as e:
            print(f"Floor detection skipped: {e}")

        # 3. FALLBACK (If MobileSAM can't clearly separate walls and floors)
        if not final_results["wall_matches"] and not final_results["floor_matches"]:
            overall_matches = find_matching_tiles(query_img, top_n=3, min_score=0.0)
            final_results["overall_matches"] = format_results(overall_matches)

        return final_results

    except OperationalError as e:
        # MySQL went away mid-request.
        print(f"search-tiles: database error: {e.orig}")
        return JSONResponse(status_code=503, content=DB_OFFLINE)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")