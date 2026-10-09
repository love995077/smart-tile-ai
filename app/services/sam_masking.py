"""Click-driven surface masks with soft, image-aware (alpha-matted) edges."""
import cv2
import numpy as np

from app.services.occlusion import foreground_objects, subtract_foreground, summarize
from app.services.segmentation import predict_logits

OVERLAY_COLOR = np.array([45, 212, 191], dtype=np.float32)   # teal-400: selected surface
EXCLUDED_COLOR = np.array([244, 63, 94], dtype=np.float32)   # rose-500: auto-excluded objects


def _guided_filter(guide, src, radius, eps):
    """He et al. guided filter. Uses cv2.ximgproc (opencv-contrib) when available."""
    if hasattr(cv2, "ximgproc"):
        return cv2.ximgproc.guidedFilter(guide, src, radius, eps, dDepth=-1)

    # Grayscale-guide fallback built from box filters.
    if guide.ndim == 3:
        guide = cv2.cvtColor(guide, cv2.COLOR_RGB2GRAY)
    ksize = (2 * radius + 1, 2 * radius + 1)
    box = lambda x: cv2.boxFilter(x, -1, ksize, borderType=cv2.BORDER_REFLECT)
    mean_i, mean_p = box(guide), box(src)
    cov_ip = box(guide * src) - mean_i * mean_p
    var_i = box(guide * guide) - mean_i * mean_i
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return box(a) * guide + box(b)


def _clean_binary(binary, positive_points):
    """Drops specks and pinholes that SAM leaves behind, keeping every clicked region."""
    h, w = binary.shape
    min_area = max(64, int(h * w * 0.0005))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clicked = {labels[int(np.clip(y, 0, h - 1)), int(np.clip(x, 0, w - 1))] for x, y in positive_points}
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    for lbl in clicked:
        keep[lbl] = lbl != 0
    cleaned = keep[labels].astype(np.uint8)

    # Fill small holes (texture noise), leave big ones (furniture, rugs) alone.
    inv = 1 - cleaned
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    small_hole = np.zeros(n, dtype=bool)
    small_hole[1:] = stats[1:, cv2.CC_STAT_AREA] < min_area
    cleaned[small_hole[labels]] = 1
    return cleaned


def _inside(mask, point):
    h, w = mask.shape
    x, y = point
    return bool(mask[int(np.clip(y, 0, h - 1)), int(np.clip(x, 0, w - 1))])


def _cluster_points(points, radius):
    """Single-linkage grouping: clicks closer than `radius` (transitively) share a group."""
    groups = []
    for p in points:
        near = [g for g in groups if any(np.hypot(p[0] - q[0], p[1] - q[1]) < radius for q in g)]
        merged = [p]
        for g in near:
            merged.extend(g)
            groups.remove(g)
        groups.append(merged)
    return groups


def segment_union(image_np, positive_clicks, negative_clicks):
    """Binary uint8 mask of the clicked surface, robust to surfaces split into pieces.

    MobileSAM is a small model. Given include points on disjoint pieces of a wall
    (separated by window frames, a lamp, shelving), one joint prompt often returns
    only the piece it is most confident about. So:
      1. A joint prompt with every click gives the base mask (most context, cleanest
         edges when SAM gets it right).
      2. Include clicks the base misses are grouped by proximity; each group gets its
         own prompt, then any click still missed gets a single-point prompt. Only the
         connected pieces containing that prompt's own clicks are kept (a partial
         prompt can also grab window frames elsewhere), and they are OR-ed into the
         union with cv2.bitwise_or.
      3. Each exclude click is segmented as an object of its own (with the include
         clicks marked as "not this") and subtracted from the union.
    Image embeddings are cached, so each extra prompt only re-runs the mask decoder.
    """
    h, w = image_np.shape[:2]
    positives = list(positive_clicks)
    negatives = list(negative_clicks)

    def segment(points, labels, require=None):
        return (predict_logits(image_np, points, labels, require_points=require) > 0).astype(np.uint8)

    def pieces_containing(mask, points):
        _, labels = cv2.connectedComponents(mask, connectivity=8)
        keep = {labels[int(np.clip(y, 0, h - 1)), int(np.clip(x, 0, w - 1))] for x, y in points} - {0}
        return np.isin(labels, list(keep)).astype(np.uint8)

    union = segment(positives + negatives, [1] * len(positives) + [0] * len(negatives))

    missed = [p for p in positives if not _inside(union, p)]
    for group in _cluster_points(missed, radius=0.15 * max(h, w)):
        extra = segment(group + negatives, [1] * len(group) + [0] * len(negatives), require=group)
        union = cv2.bitwise_or(union, pieces_containing(extra, group))

    for p in positives:
        if not _inside(union, p):
            extra = segment([p] + negatives, [1] + [0] * len(negatives), require=[p])
            union = cv2.bitwise_or(union, pieces_containing(extra, [p]))

    area = int(union.sum())
    for n in negatives:
        if not _inside(union, n):
            continue
        obj = segment([n] + list(positive_clicks), [1] + [0] * len(positive_clicks))
        # Guard against an ambiguous exclude prompt that grabs most of the surface itself.
        if obj.sum() < 0.6 * area:
            union[obj > 0] = 0
    return union


def _palette(samples, k=5):
    samples = np.float32(samples)
    if len(samples) > 3000:
        samples = samples[np.random.default_rng(0).choice(len(samples), 3000, replace=False)]
    k = int(min(k, len(samples)))
    cv2.setRNGSeed(0)   # deterministic palettes: the same clicks give the same mask
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    return cv2.kmeans(samples, k, None, criteria, 2, cv2.KMEANS_PP_CENTERS)[2]


def _nearest(feats, pal):
    return np.sqrt(((feats[:, None, :] - pal[None, :, :]) ** 2).sum(-1)).min(axis=1)


def _fill_object_gaps(image_np, binary, objects, reach):
    """Fills slivers of old surface left between the mask and a detected object.

    SAM's boundary can stop a few pixels short of an object, leaving e.g. slivers of a
    black checker square at a pot's base. Candidates are pixels outside the surface mask,
    within `reach` of it and of a detected object, but outside that object's mask. Each
    is filled only if its colour clearly matches the surface (sampled nearby) rather than
    the object's own colours (from its instance mask). Ambiguous pixels, such as a white
    toilet base against white tiles, are left alone, so tiles never creep onto objects.
    """
    if not objects:
        return binary
    h, w = binary.shape
    lab = cv2.cvtColor(cv2.GaussianBlur(image_np, (3, 3), 0), cv2.COLOR_RGB2LAB).astype(np.float32)
    near_mask = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 3) <= reach
    gap = np.zeros((h, w), np.uint8)
    for o in objects:
        x0, y0, x1, y1 = o.region
        pad = reach + 2
        X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
        om = np.zeros((Y1 - Y0, X1 - X0), np.uint8)
        om[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = o.mask
        if om.sum() < 30:
            continue
        near_o = cv2.distanceTransform(1 - om, cv2.DIST_L2, 3) <= pad
        outside_o = cv2.dilate(om, np.ones((3, 3), np.uint8)) == 0
        cand = near_o & outside_o & (binary[Y0:Y1, X0:X1] == 0) & near_mask[Y0:Y1, X0:X1]
        surf = (binary[Y0:Y1, X0:X1] > 0)
        if not cand.any() or surf.sum() < 30:
            continue
        crop = lab[Y0:Y1, X0:X1]
        feats = crop[cand]
        d_s = _nearest(feats, _palette(crop[surf]))
        d_o = _nearest(feats, _palette(crop[om > 0]))
        take = np.zeros_like(cand)
        take[cand] = (d_s < 0.5 * d_o) & (d_s < 14.0)
        gap[Y0:Y1, X0:X1] |= take.astype(np.uint8)
    merged = cv2.bitwise_or(binary, gap)
    n, labels = cv2.connectedComponents(merged, connectivity=4)
    keep = np.zeros(n, bool)
    keep[np.unique(labels[binary > 0])] = True
    keep[0] = False
    return keep[labels].astype(np.uint8)


def _absorb_slivers(image_np, binary, reach):
    """Absorbs thin leftovers of the old surface right at the mask boundary (any object).

    Within `reach` (a few pixels) outside the mask, a pixel joins the surface when its
    colour clearly matches the surface just inside and not what lies just beyond it
    (reach..3*reach out, which at this short range is the neighbouring object itself).
    Removes e.g. black checker slivers beside white stool legs while leaving the legs.
    """
    lab = cv2.cvtColor(cv2.GaussianBlur(image_np, (3, 3), 0), cv2.COLOR_RGB2LAB).astype(np.float32)
    d_out = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 3)
    d_in = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    cand = (binary == 0) & (d_out <= reach)
    inner = (binary > 0) & (d_in <= 2 * reach)
    beyond = (binary == 0) & (d_out > reach) & (d_out <= 3 * reach)
    if cand.sum() == 0 or inner.sum() < 50 or beyond.sum() < 50:
        return binary
    # Palettes are local: classify per tile of the boundary so distant colours don't mix.
    h, w = binary.shape
    out = binary.copy()
    step = max(64, 12 * reach)
    for ty in range(0, h, step):
        for tx in range(0, w, step):
            sl = (slice(max(0, ty - reach), min(h, ty + step + reach)), slice(max(0, tx - reach), min(w, tx + step + reach)))
            c, i, b = cand[sl], inner[sl], beyond[sl]
            if c.sum() == 0 or i.sum() < 20 or b.sum() < 20:
                continue
            crop = lab[sl]
            feats = crop[c]
            d_s = _nearest(feats, _palette(crop[i]))
            d_o = _nearest(feats, _palette(crop[b]))
            take = np.zeros_like(c)
            take[c] = (d_s < 0.4 * d_o) & (d_s < 12.0)
            out[sl][take] = 1
    n, labels = cv2.connectedComponents(out, connectivity=4)
    keep = np.zeros(n, bool)
    keep[np.unique(labels[binary > 0])] = True
    keep[0] = False
    return keep[labels].astype(np.uint8)


def _close_halo(image_np, binary, alpha, reach):
    """Colour-line matting just outside the mask, to close the halo SAM leaves around objects.

    SAM's boundary often stops a few pixels short of an object, leaving a rim of the old
    wall visible around heads and frames. Near the boundary, B is the local old-surface
    colour (inside) and F the local object colour (outside). An outside pixel whose colour
    is clearly surface-like gets alpha from its position on the B-F colour line, which also
    gives mixed edge pixels their true fraction. Only applied where B and F differ enough
    to tell apart, and only within `reach` pixels of the mask.
    """
    img = image_np.astype(np.float32)
    k5 = np.ones((5, 5), np.uint8)
    sigma = max(3.0, reach / 2.0)

    def local_mean(sel):
        m = sel.astype(np.float32)
        num = cv2.GaussianBlur(img * m[..., None], (0, 0), sigma)
        den = cv2.GaussianBlur(m, (0, 0), sigma)
        return num / np.maximum(den, 1e-4)[..., None], den

    B, b_support = local_mean(cv2.erode(binary, k5))
    F, f_support = local_mean(cv2.erode(1 - binary, k5))
    d = B - F
    sep = np.sqrt((d * d).sum(-1))
    proj = np.clip(((img - F) * d).sum(-1) / np.maximum(sep * sep, 1e-3), 0, 1)
    to_b = np.sqrt(((img - B) ** 2).sum(-1))
    outside = (binary == 0) & (cv2.distanceTransform(1 - binary, cv2.DIST_L2, 3) <= reach)
    surface_like = (to_b < np.minimum(25.0, 0.35 * sep)) & (sep > 30) & (b_support > 0.05) & (f_support > 0.05)
    take = outside & surface_like
    out = alpha.copy()
    out[take] = np.maximum(out[take], proj[take])
    return out


def build_alpha_matte(image_np, positive_clicks, negative_clicks, auto_exclude=True):
    """Returns (alpha, info): a float32 alpha matte in [0, 1] for the clicked surface.

    With auto_exclude, detected foreground objects (people, chairs, laptops, ...) are
    subtracted before matting; info carries their counts and the excluded-pixel mask.

    The binary SAM mask is only trusted away from its boundary. Inside a thin
    band around the edge (a trimap's "unknown" region), the alpha comes from a
    guided filter steered by the photo, so the edge follows real image edges
    (carpet fringes, soft-focus furniture) instead of SAM's 256px-grid stair steps.
    """
    if not positive_clicks:
        raise ValueError("At least one positive click is required.")

    union = segment_union(image_np, positive_clicks, negative_clicks)
    info = {"excluded": np.zeros(union.shape, np.uint8), "objects": {}}
    if auto_exclude:
        union, applied, excluded = subtract_foreground(union, foreground_objects(image_np), positive_clicks, image_np)
        info = {"excluded": excluded, "objects": summarize(applied)}
    binary = _clean_binary(union, positive_clicks)
    if not binary.any():
        return np.zeros(binary.shape, dtype=np.float32), info
    reach = max(6, int(round(max(binary.shape) * 0.008)))
    grown = binary
    if auto_exclude:
        grown = _fill_object_gaps(image_np, binary, foreground_objects(image_np), reach)
    grown = _absorb_slivers(image_np, grown, max(3, int(round(max(binary.shape) * 0.003))))
    if auto_exclude:
        grown[info["excluded"] > 0] = 0
    for x, y in negative_clicks:             # nor into what the user excluded
        hb, wb = binary.shape
        xi, yi = int(np.clip(x, 0, wb - 1)), int(np.clip(y, 0, hb - 1))
        if grown[yi, xi] and not binary[yi, xi]:
            grown = binary.copy()
            break
    binary = grown

    h, w = binary.shape
    radius = max(3, int(round(max(h, w) * 0.005)))
    guide = image_np.astype(np.float32) / 255.0
    soft = _guided_filter(guide, binary.astype(np.float32), radius, 1e-3)

    # Re-centre and steepen slightly: the guided filter alone gives a mushy ramp.
    soft = np.clip((soft - 0.5) * 1.6 + 0.5, 0.0, 1.0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    band = cv2.dilate(binary, kernel) - cv2.erode(binary, kernel)

    # The guided filter only helps where the photo has a real edge to follow. Where the
    # mask boundary crosses flat image (floor meeting floor in shadow, plain wall), it
    # smears the edge into a wide blurry ramp that reads as a glitch, so there the
    # boundary stays crisp with 1-2px anti-aliasing.
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    grad = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    edge_strength = cv2.dilate(cv2.GaussianBlur(grad, (0, 0), 1.0), kernel)
    follow = np.clip((edge_strength - 0.15) / 0.45, 0.0, 1.0)
    # Smooth SAM's slightly wavy outline (~2px) before the 1-2px anti-aliasing.
    smooth_outline = (cv2.GaussianBlur(binary.astype(np.float32), (0, 0), 2.0) > 0.5).astype(np.float32)
    crisp = cv2.GaussianBlur(smooth_outline, (0, 0), 0.8)
    alpha = np.where(band > 0, follow * soft + (1.0 - follow) * crisp, binary.astype(np.float32))
    # Keep the matte within ~2px of the outline on both sides. Next to a high-contrast old
    # pattern (black/white checker) the guided filter follows the pattern's edges and can
    # spread partial alpha onto neighbouring white objects (translucent tiles over a toilet
    # base) or punch translucent holes into the floor.
    bound = cv2.GaussianBlur(binary.astype(np.float32), (0, 0), 1.5)
    alpha = np.where(binary > 0, np.maximum(alpha, bound), np.minimum(alpha, bound))
    alpha = _close_halo(image_np, binary, alpha, reach=max(6, int(round(max(h, w) * 0.009))))
    return cv2.GaussianBlur(alpha, (3, 3), 0).astype(np.float32), info


def make_overlay(image_np, alpha, excluded=None):
    """Room photo with the selected surface tinted teal and auto-excluded objects outlined in rose."""
    a = (alpha * 0.45)[..., None]
    out = image_np.astype(np.float32) * (1 - a) + OVERLAY_COLOR * a

    k3 = np.ones((3, 3), np.uint8)
    if excluded is not None and excluded.any():
        e = (excluded > 0)[..., None] * 0.28
        out = out * (1 - e) + EXCLUDED_COLOR * e
        out[cv2.morphologyEx(excluded.astype(np.uint8), cv2.MORPH_GRADIENT, k3) > 0] = EXCLUDED_COLOR

    edge = cv2.morphologyEx((alpha > 0.5).astype(np.uint8), cv2.MORPH_GRADIENT, k3)
    out[edge > 0] = OVERLAY_COLOR
    return np.clip(out, 0, 255).astype(np.uint8)
