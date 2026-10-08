"""Click-driven surface masks with soft, image-aware (alpha-matted) edges."""
import cv2
import numpy as np

from app.services.segmentation import predict_logits

OVERLAY_COLOR = np.array([45, 212, 191], dtype=np.float32)  # teal-400


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


def build_alpha_matte(image_np, positive_clicks, negative_clicks):
    """Returns a float32 alpha matte in [0, 1] for the clicked surface.

    The binary SAM mask is only trusted away from its boundary. Inside a thin
    band around the edge (a trimap's "unknown" region), the alpha comes from a
    guided filter steered by the photo, so the edge follows real image edges
    (carpet fringes, soft-focus furniture) instead of SAM's 256px-grid stair steps.
    """
    if not positive_clicks:
        raise ValueError("At least one positive click is required.")

    union = segment_union(image_np, positive_clicks, negative_clicks)
    binary = _clean_binary(union, positive_clicks)
    if not binary.any():
        return np.zeros(binary.shape, dtype=np.float32)

    h, w = binary.shape
    radius = max(3, int(round(max(h, w) * 0.005)))
    guide = image_np.astype(np.float32) / 255.0
    soft = _guided_filter(guide, binary.astype(np.float32), radius, 1e-3)

    # Re-centre and steepen slightly: the guided filter alone gives a mushy ramp.
    soft = np.clip((soft - 0.5) * 1.6 + 0.5, 0.0, 1.0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    band = cv2.dilate(binary, kernel) - cv2.erode(binary, kernel)
    alpha = np.where(band > 0, soft, binary.astype(np.float32))
    return cv2.GaussianBlur(alpha, (3, 3), 0).astype(np.float32)


def make_overlay(image_np, alpha):
    """Room photo with the selected surface tinted and outlined."""
    a = (alpha * 0.45)[..., None]
    out = image_np.astype(np.float32) * (1 - a) + OVERLAY_COLOR * a

    edge = cv2.morphologyEx((alpha > 0.5).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    out[edge > 0] = OVERLAY_COLOR
    return np.clip(out, 0, 255).astype(np.uint8)
