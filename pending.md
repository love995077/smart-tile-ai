# Pending: Phase-2 roadmap

Features planned for upcoming sprints. Each section lists what exists today that the
feature builds on, the design, and what counts as done.

---

## 1. Smart Tile Calculator & Cost Estimator ("sales closer")

Turn a render into a quote: real surface area, tile count, box count and price, shown
next to the render and printed on the spec sheet.

### What already exists

`render_engine.estimate_geometry` recovers a metric plane for the selected surface:

- `PlaneGeometry.normal`, `.distance`, `.e1`, `.e2`, `.f`, `.cx`, `.cy` define the plane
  in camera space. `distance` is in metres, fixed by `FLOOR_CAMERA_HEIGHT_M` (1.5 m) for
  floors and `WALL_DISTANCE_M` (3 m) for walls.
- `plane_coordinates(geo, x0, y0, x1, y1)` already returns per-pixel metric `(u, v)` in
  metres for the mask's bounding box.

### Area computation

1. For each mask pixel, the area it covers on the plane is the Jacobian determinant of
   the pixel-to-plane map: `|du/dx * dv/dy - du/dy * dv/dx|` (m² per pixel). Compute it
   with `np.gradient` on the `u`, `v` arrays `render_surface` already builds.
2. Surface area = `sum(alpha * jacobian)` over the bounding box. Using the soft alpha
   (not the binary mask) gives sub-pixel-correct edges.
3. Report m² and ft² (`1 m² = 10.7639 ft²`).
4. Guard rails:
   - Pixels near the horizon have a huge Jacobian, so clip it at the 99.5th percentile
     to stop a few far pixels from dominating.
   - Excluded objects (sofas, beds) hide floor that still needs tiling. Offer an
     "include hidden floor under furniture" option that adds the area of
     `info["excluded"]` pixels that lie on the plane (bottom-contact objects only).

### Scale calibration (accuracy)

The absolute scale depends on the assumed camera height / wall distance. A ±20% error
in height becomes a ±44% error in area, so the estimator needs a calibration path:

- **Reference object.** The user taps both ends of a known length: a door width (≈ 0.9 m),
  an A4 sheet, or one existing tile of known size. Solve for the scale factor `k` that
  maps the measured plane distance to the real one, then multiply every area by `k²`.
- **Manual override.** Fields for the room dimensions (L × W). When filled, they win
  over the photo estimate.
- Show the estimate as a range (±15%) until it is calibrated.

### Quantity and price

```
tile_area     = (tile_w_mm / 1000) * (tile_h_mm / 1000)
tiles_needed  = ceil(area * (1 + wastage) / tile_area)
boxes         = ceil(tiles_needed / tiles_per_box)
cost          = boxes * box_price        (or area_with_wastage * price_per_sqft)
```

- Wastage defaults: 10% for the standard grid, 12% for brick, 15% for 45° diamond and
  herringbone (more cuts), all configurable.
- `tiles_per_box`, `price_per_sqft` and `box_price` come from the `products` table
  (`app/models/product.py`). Add columns `tiles_per_box` and `coverage_sqft_per_box`
  if missing. When there is no DB match, fall back to manual fields in the UI.
- Grout and adhesive: grout kg ≈ `area * (L+W)/(L*W) * joint_width * depth * 1.6`
  (L, W in mm). Adhesive ≈ 4–5 kg/m². Both are optional line items.

### API

- `/api/apply-tile` adds `area_m2`, `area_ft2` and `area_confidence` to `X-Render-Info`.
- New `POST /api/estimate` takes `area_m2`, tile size, pattern, `wastage`, `product_id`
  and an optional calibration, and returns a quote breakdown as JSON. It is a separate
  endpoint so that changing the wastage or price never triggers a re-render.

### Frontend

- An "Estimate" card under the render controls: area, wastage slider, tiles, boxes, total.
- The spec-sheet export gets a quote block.
- Currency and units are configurable (₹ / sq.ft default for the Indian market).

### Done when

- On a synthetic floor with a known area (rendered from a known camera), the area error
  is within 5% once calibrated and 20% uncalibrated.
- Quantities round up correctly, and wastage per pattern is applied.

---

## 2. Dual-Surface Mode (Floor + Wall together)

Today "Tile another surface" bakes the render into the base photo. After that, the
first surface can't be changed without starting over. Dual-surface mode keeps each
surface as its own live layer.

### Frontend state model

```js
state.layers = {
  floor: { pos, neg, order, maskBlob, overlayImg, tile, size, pattern, orientation,
           grout: { mm, color }, glossy, resultBlob },
  wall:  { ...same },
};
state.activeLayer = 'floor';
```

- A layer switcher (Floor / Wall chips) above the canvas. Clicks, undo and every
  sidebar control read from and write to the active layer only.
- The overlay shows the active layer's mask in teal and the inactive one in a muted
  outline, so the user can see both.
- Each layer is rendered independently against the ORIGINAL photo (never a previous
  render). This keeps depth, lighting and occlusion estimates clean.

### Backend

- New `POST /api/apply-layers` takes the original room, plus 1–N `(mask, tile, params)`
  layer specs (multipart with indexed field names, or one JSON manifest plus files).
- Each layer is rendered with `render_surface` on the original photo. Depth, geometry
  and lighting are cached per image hash, so the second layer costs only the tiling.
- Composite in a fixed order (walls first, then floors), each with its own soft alpha.
  Where the masks overlap, the layer with the higher alpha wins, and the floor wins
  ties. That keeps the floor–wall junction (the line Phase 2 straightened) crisp.
- Glossy floors reflect the NEW walls: in `floor_reflection`, render the floor layer
  after the wall layers, using the wall-tiled image as the reflection source.

### Caching and performance

- Cache each layer's tiled RGB + alpha by
  `hash(room, mask, tile, params)` in an `LRUCache`. Changing only the wall tile then
  re-renders the wall and re-composites a cached floor (≈ 0.3 s instead of 2× 2 s).
- Memory: two 1600×1200 float32 RGB layers come to about 46 MB, well inside the
  container budget.

### UX details

- "Download Spec Sheet" lists both surfaces (two swatch blocks).
- Before/after compare works on the combined result. A "solo" toggle previews one layer.
- Migration: the existing "Tile another surface" button becomes "Add wall layer" /
  "Add floor layer".

### Done when

- Floor and wall can be edited in any order without re-uploading, and each keeps its own
  tile, pattern, grout and finish.
- Changing one layer never changes the other's pixels, except in the composite overlap band.
- Reflections on a glossy floor show the newly tiled wall.
