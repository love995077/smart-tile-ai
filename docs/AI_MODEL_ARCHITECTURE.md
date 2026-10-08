# 🏗️ Smart Tile AI: Model Architecture

## 📋 Table of Contents
* [1. Surface Selection (MobileSAM)](#1-surface-selection-mobilesam)
* [2. Depth Estimation (Depth Anything V2)](#2-depth-estimation-depth-anything-v2)
* [3. Visual Search & Matching (CLIP)](#3-visual-search--matching-clip)
* [Render pipeline](#render-pipeline)
* [Running locally](#running-locally)

All three models run locally (no external generative AI APIs) and fit together in **4GB of VRAM**.
Each one loads lazily on first use, inference is serialised behind one lock, and
`torch.cuda.empty_cache()` runs after every inference (`app/services/gpu.py`).

---

### **1. Surface Selection (MobileSAM)**
* **Weights**: `mobile_sam.pt` (~40MB), from `app/models/mobile_sam.pt` if present, else Hugging Face `dhkim2810/MobileSAM`
* **Library**: `mobile_sam` (installed from GitHub, see requirements.txt)
* **Code**: `app/services/segmentation.py`, `app/services/sam_masking.py`
* **Purpose**: Users left-click (include) and right-click (exclude) points; MobileSAM returns the surface mask.
  The mask is refined into a soft alpha matte with a guided filter (`cv2.ximgproc`, opencv-contrib)
  inside a thin trimap band, so edges follow real image edges. Image embeddings are cached per photo,
  so each extra click only re-runs the mask decoder.

### **2. Depth Estimation (Depth Anything V2)**
* **Model ID**: `depth-anything/Depth-Anything-V2-Small-hf` (Transformers port of `Depth-Anything-V2-Small`)
* **Library**: `transformers`
* **Code**: `app/services/depth_engine.py`
* **Purpose**: Relative disparity map used to find the surface's orientation and per-pixel normals.

### **3. Visual Search & Matching (CLIP)**
* **Model ID**: `openai/clip-vit-base-patch32`
* **Library**: `transformers`
* **Code**: `app/services/visual_search.py`
* **Purpose**: Embeds the catalog (indexed on the first search) and scores matches 50% CLIP + 50% Hue/Saturation histogram.

---

## Render pipeline

`app/services/render_engine.py`, called by `POST /api/apply-tile`:

1. **Plane orientation:** over a plane, disparity is affine in pixel position. A robust fit gives the horizon line's direction.
2. **Horizon position:** `cv2.HoughLinesP` on Canny edges around the surface votes for vanishing points.
   If there are no lines, it falls back to an eye-level prior (floors) or the raw depth.
3. **Metric mesh-grid:** each pixel is cast onto the 3D plane (focal length about 0.8 × long side).
   The tile texture is sized in mm, with grout and a bevel, and sampled with mip-mapping. The grid is aligned with the room's dominant lines.
4. **Intrinsic lighting:** the L channel (LAB) is bilateral-filtered into a shading map and multiplied onto the tiles in linear light.
5. **Glossy mode:** blurred planar reflection of objects standing on the floor (per-column mirror at each contact line), 10-16% opacity, plus specular sheen.
6. **Edge compositing:** colour decontamination stops the old floor's colour from haloing around thin objects.

## Running locally

```powershell
.\.venv\Scripts\activate
cd smart_tile_ai
pip install -r requirements.txt
# For the GPU, replace the CPU torch wheels:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
uvicorn app.main:app --reload --reload-dir app     # open http://localhost:8000
```

On the first run the weights download from Hugging Face (about 0.7GB in total).
