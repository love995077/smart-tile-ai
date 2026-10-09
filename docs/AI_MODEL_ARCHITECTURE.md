# 🏗️ Smart Tile AI: Model Architecture

## 📋 Table of Contents
* [1. Surface Selection (MobileSAM)](#1-surface-selection-mobilesam)
* [2. Foreground Occlusion (D-FINE + MobileSAM)](#2-foreground-occlusion-d-fine--mobilesam)
* [3. Depth Estimation (Depth Anything V2)](#3-depth-estimation-depth-anything-v2)
* [4. Visual Search & Matching (CLIP)](#4-visual-search--matching-clip)
* [Render pipeline](#render-pipeline)
* [Running locally](#running-locally)

All models run locally (no external generative AI APIs) and fit together in **4GB of VRAM**.
Each one loads lazily on first use, inference is serialised behind one lock, and
`torch.cuda.empty_cache()` runs after every inference (`app/services/gpu.py`).

---

### **1. Surface Selection (MobileSAM)**
* **Weights**: `mobile_sam.pt` (~40MB), from `app/models/mobile_sam.pt` if present, else Hugging Face `dhkim2810/MobileSAM`
* **Library**: `mobile_sam` (installed from GitHub, see requirements.txt)
* **Code**: `app/services/segmentation.py`, `app/services/sam_masking.py`
* **Purpose**: Users left-click (include) and right-click (exclude) points; MobileSAM returns the surface mask.
  Disjoint pieces are re-prompted and OR-ed together. Edges: a guided filter follows real image edges, while
  boundaries crossing flat image stay crisp; colour-line matting and object-gap filling close the rim SAM
  leaves around objects. Image embeddings are cached per photo, so each extra click only re-runs the decoder.

### **2. Foreground Occlusion (D-FINE + MobileSAM)**
* **Model ID**: `ustc-community/dfine-small-coco` (Apache-2.0, ~10M params)
* **Library**: `transformers`
* **Code**: `app/services/occlusion.py`
* **Purpose**: Detects people, chairs, laptops, bottles, plants, monitors, clocks and every other COCO class.
  Each box prompts MobileSAM for a full-resolution instance mask, which is subtracted from the surface with no
  clicks. Only pixels that look like the object are removed (colour models), so floor seen between table legs or
  through a chair frame stays selectable. A click inside an object overrides the detector.

### **3. Depth Estimation (Depth Anything V2)**
* **Model ID**: `depth-anything/Depth-Anything-V2-Small-hf` (Transformers port of `Depth-Anything-V2-Small`)
* **Library**: `transformers`
* **Code**: `app/services/depth_engine.py`
* **Purpose**: Relative disparity used for the RANSAC plane fit and surface normals.

### **4. Visual Search & Matching (CLIP)**
* **Model ID**: `openai/clip-vit-base-patch32`
* **Library**: `transformers`
* **Code**: `app/services/visual_search.py`
* **Purpose**: Embeds the catalog (indexed on the first search) and scores matches 50% CLIP + 50% Hue/Saturation histogram.

---

## Render pipeline

`app/services/render_engine.py`, called by `POST /api/apply-tile`:

1. **Rigid plane:** a RANSAC fit of disparity = a·x + b·y + c (exactly a plane for affine-invariant inverse depth).
   Camera roll is estimated from the room's vertical lines; walls are forced exactly vertical, floors exactly horizontal.
2. **Horizon position:** Hough-line vanishing points (walls: only level lines crossing near eye level; perspective capped
   by what depth supports), then an eye-level prior (floors) or raw depth.
3. **Metric mesh-grid:** every pixel is cast onto the single plane (focal length 0.8 × long side), so tile courses are
   straight lines by construction. The tile photo (uniform borders auto-cropped) is mip-mapped; grout lines are drawn
   analytically with exact box-filter anti-aliasing (no moiré); a small bevel darkens tile edges.
4. **Lighting:** luminance is resampled onto a top-down metric grid of the plane and split there by physical size:
   light level from a linear-light local mean, crisp shadow/sun-patch edges from a morphological filter, residue under
   ~12% dropped. Old-tile patterns smaller than ~45 cm (checkerboards, diamonds, planks) don't ghost; room-scale shadows
   stay. Floors get soft contact shadows at object bases. Highlights are compressed so sunlight never bleaches tiles.
5. **Glossy mode:** blurred planar reflection of objects standing on the floor (per-column mirror at each contact line),
   10-16% opacity, plus specular sheen.
6. **Edge compositing:** colour decontamination where the old surface is uniform, plain alpha blending where it is
   patterned.

## Running locally

```powershell
.\.venv\Scripts\activate
cd smart_tile_ai
pip install -r requirements.txt
# For the GPU, replace the CPU torch wheels:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
uvicorn app.main:app --reload --reload-dir app     # open http://localhost:8000
```

On the first run the weights download from Hugging Face (about 0.75GB in total).
