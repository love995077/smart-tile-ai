"""Photorealistic tile rendering.

Pipeline (all arrays RGB):
  1. Depth Anything V2 gives relative disparity. Over a planar surface, disparity
     is an affine function of pixel position, so a robust plane fit gives the
     surface orientation (the direction of the plane's horizon line).
  2. Hough lines + Canny on and around the surface vote for vanishing points.
     Those anchor where the horizon sits (the one thing monocular depth cannot
     tell us, because its shift is unknown) and the grid rotation.
  3. With the horizon and an assumed focal length, every pixel is cast onto the
     3D plane, giving metric (u, v) coordinates. That per-pixel mesh-grid is
     used to sample a tile texture sized in millimetres (with grout), mip-mapped
     so distant tiles do not shimmer.
  4. Lighting: LAB luminance, bilateral-filtered, splits shading (shadows and
     highlights) from the old surface's paint/texture. That shading is
     multiplied onto the new tiles in linear light.
  5. Glossy mode adds a blurred, faded planar reflection of everything standing
     on the floor, plus specular sheen.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.services.depth_engine import estimate_depth

FOCAL_FACTOR = 0.8            # f = 0.8 * long side, about a 64 degree FOV (typical phone main camera)
FLOOR_CAMERA_HEIGHT_M = 1.5   # phone held at chest height
WALL_DISTANCE_M = 3.0         # typical camera-to-wall distance for interior shots
MAX_PERSPECTIVE = 0.95        # cap on far/near compression, keeps the horizon off the surface
MAX_FLOOR_ROTATION_DEG = 15.0 # larger detected angles are clutter (chairs, desks), not room architecture
MAX_WALL_PERSPECTIVE = 0.6    # walls: far edge at most 2.5x the near edge's distance
WALL_VP_MAX_ANGLE_DEG = 30.0  # only near-horizontal segments can vote for a wall's vanishing point
WALL_VP_EYE_BAND = 0.2        # ...and their crossing must lie within +-20% of image height of eye level
HIGHLIGHT_HEADROOM = (0.5, 0.7)   # max extra brightening above average light: (matte, glossy)


@dataclass
class PlaneGeometry:
    surface: str
    normal: np.ndarray
    distance: float
    e1: np.ndarray
    e2: np.ndarray
    f: float
    cx: float
    cy: float
    info: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- geometry

def surface_normals(disparity, f):
    """Per-pixel unit normals from the depth map (camera coords, y down, pointing away from camera).

    Back-projects with the raw disparity read as 1/Z. The unknown shift makes the
    normals approximate, but that is enough to tell floors from walls and to reject
    off-plane pixels (furniture legs, rugs) before the plane fit.
    """
    h, w = disparity.shape
    small = cv2.resize(disparity, (max(8, w // 4), max(8, h // 4)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 1.5)
    sh, sw = small.shape
    sf = f * sw / w
    z = 1.0 / np.maximum(small, 1e-3)
    ys, xs = np.mgrid[0:sh, 0:sw].astype(np.float32)
    pts = np.stack([(xs - sw / 2) * z / sf, (ys - sh / 2) * z / sf, z], axis=-1)
    n = np.cross(np.gradient(pts, axis=1), np.gradient(pts, axis=0))
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    return cv2.resize(n, (w, h), interpolation=cv2.INTER_LINEAR)


def _fit_disparity_plane(disparity, mask_bin, normals, upright=False):
    """Robust fit of disparity = a*x + b*y + c over the surface pixels.

    upright=True fixes b = 0. For a vertical wall seen by a level camera, inverse
    depth does not change along an image column, so this forces the wall to be
    vertical: its vertical grout lines stay vertical instead of picking up the
    tilt of noisy monocular depth.
    """
    ys, xs = np.nonzero(mask_bin)
    if len(xs) > 40000:
        pick = np.random.default_rng(0).choice(len(xs), 40000, replace=False)
        ys, xs = ys[pick], xs[pick]

    nm = normals[ys, xs]
    med = np.median(nm, axis=0)
    med /= np.linalg.norm(med) + 1e-9
    on_plane = nm @ med > np.cos(np.radians(35))
    if on_plane.sum() > 500:
        ys, xs = ys[on_plane], xs[on_plane]

    A = np.stack([xs, ys, np.ones_like(xs)], axis=1).astype(np.float64)
    cols = [0, 2] if upright else [0, 1, 2]
    d = disparity[ys, xs].astype(np.float64)
    keep = np.ones(len(d), dtype=bool)
    coef = np.zeros(3)
    for _ in range(4):
        sol, *_ = np.linalg.lstsq(A[keep][:, cols], d[keep], rcond=None)
        coef = np.zeros(3)
        coef[cols] = sol
        res = np.abs(A @ coef - d)
        mad = np.median(res[keep]) + 1e-6
        keep = res < 3.0 * 1.4826 * mad
        if keep.sum() < 50:
            break
    return coef, A @ coef, med


def _classify_surface(median_normal, coef, mask_bin):
    """Floor vs wall. Monocular normals are noisy (a photo shot tilted upwards makes
    walls look like they face up), so position in the frame decides too: a floor
    sits in the lower half and runs down to the bottom edge of the photo."""
    h = mask_bin.shape[0]
    ys, _ = np.nonzero(mask_bin)
    centroid = ys.mean() / h
    bottom = np.percentile(ys, 98) / h
    a, b, _ = coef
    looks_down = median_normal[1] > 0.45 or (b > 0 and b > 1.5 * abs(a))
    return "floor" if looks_down and centroid > 0.55 and bottom > 0.85 else "wall"


def _detect_segments(gray, band):
    """Dominant straight structural lines (baseboards, plank/tile edges, wall corners) near the surface."""
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    med = float(np.median(blurred))
    lo = int(max(10, 0.66 * med))
    hi = int(min(255, max(lo + 30, 1.33 * med)))
    edges = cv2.Canny(blurred, lo, hi)
    edges[band == 0] = 0

    min_len = 0.05 * min(h, w)
    segs = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=int(min_len * 0.6),
                           minLineLength=min_len, maxLineGap=0.01 * max(h, w))
    if segs is None:
        return np.zeros((0, 4))
    segs = segs.reshape(-1, 4).astype(np.float64)  # (N,1,4) in OpenCV 4, (N,4) in 5
    lengths = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    return segs[np.argsort(-lengths)[:80]]


def _homogeneous_lines(segs):
    p1 = np.column_stack([segs[:, 0], segs[:, 1], np.ones(len(segs))])
    p2 = np.column_stack([segs[:, 2], segs[:, 3], np.ones(len(segs))])
    lines = np.cross(p1, p2)
    lengths = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    angles = np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0])
    return lines, lengths, angles


def _horizon_shift_from_vanishing_points(segs, coef, d_far, d_near, wall_eye_row=None, image_h=None,
                                         max_rho=0.97):
    """Picks the disparity shift t so the plane's horizon passes through the dominant vanishing point.

    Lines that are parallel in 3D and lie on the surface meet at a vanishing point on its
    horizon, where disparity (extrapolated from the plane fit) equals t. Every pair of
    segments votes; the votes are histogrammed by the perspective strength they imply,
    and the strongest consistent cluster wins.

    For walls (wall_eye_row set), only horizontal 3D lines (baseboards, window heads,
    tile courses) carry the wall's vanishing point, and for a level camera they converge
    at eye level. So only near-horizontal segments vote, and only crossings near the
    eye-level row count. Window mullions, furniture legs and sofa edges, which made a
    nearly frontal wall look like it was seen at a grazing angle, are ignored.
    """
    if wall_eye_row is not None:
        ang = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0])) % 180
        segs = segs[np.minimum(ang, 180 - ang) <= WALL_VP_MAX_ANGLE_DEG]
    if len(segs) < 3:
        return None
    a, b, c = coef
    span = d_near - d_far
    lines, lengths, angles = _homogeneous_lines(segs)

    i, j = np.triu_indices(len(segs), 1)
    v = np.cross(lines[i], lines[j])
    dang = np.abs(angles[i] - angles[j]) % np.pi
    dang = np.minimum(dang, np.pi - dang)
    ok = (np.abs(v[:, 2]) > 1e-9) & (dang > np.radians(1.5))
    vz = np.where(ok, v[:, 2], 1.0)
    t = a * v[:, 0] / vz + b * v[:, 1] / vz + c
    rho = span / np.maximum(d_near - t, 1e-9)
    ok &= (t < d_far - 0.02 * span) & (rho > 0.05) & (rho < max_rho)
    if wall_eye_row is not None:
        vy = v[:, 1] / vz
        ok &= np.abs(vy - wall_eye_row) < WALL_VP_EYE_BAND * image_h
    if ok.sum() < 3:
        return None

    weights = (lengths[i] * lengths[j])[ok]
    rho = rho[ok]
    hist, edges = np.histogram(rho, bins=46, range=(0.05, max_rho), weights=weights)
    k = int(np.argmax(np.convolve(hist, [1, 2, 1], mode="same")))
    centre = 0.5 * (edges[k] + edges[k + 1])
    near = np.abs(rho - centre) < 0.04
    if near.sum() < 3 or weights[near].sum() < 0.25 * weights.sum():
        return None

    order = np.argsort(rho[near])
    cum = np.cumsum(weights[near][order])
    rho_hat = rho[near][order][np.searchsorted(cum, cum[-1] / 2)]
    return d_near - span / rho_hat


def _grid_rotation(segs, horizon, normal, e1, e2, f, cx, cy):
    """Rotation (radians) that aligns the tile grid with the room's dominant lines.

    Each segment's vanishing point is where it crosses the horizon; back-projected,
    that is the segment's 3D direction on the plane. The length-weighted mode of
    those directions (mod 90 degrees, since a grid is 4-fold symmetric) is the grid angle.
    """
    if len(segs) < 2:
        return 0.0
    lines, lengths, _ = _homogeneous_lines(segs)
    v = np.cross(lines, horizon[None, :])
    dirs = np.column_stack([(v[:, 0] - cx * v[:, 2]) / f, (v[:, 1] - cy * v[:, 2]) / f, v[:, 2]])
    dirs -= (dirs @ normal)[:, None] * normal[None, :]
    norm = np.linalg.norm(dirs, axis=1)
    ok = norm > 1e-9
    if ok.sum() < 2:
        return 0.0
    dirs = dirs[ok] / norm[ok, None]
    theta = np.degrees(np.arctan2(dirs @ e2, dirs @ e1)) % 90.0
    # Only near-square directions can vote; anything else is furniture or clutter.
    weights = np.where(np.minimum(theta, 90 - theta) <= MAX_FLOOR_ROTATION_DEG, lengths[ok], 0.0)
    if weights.sum() <= 0:
        return 0.0

    hist, _ = np.histogram(theta, bins=45, range=(0, 90), weights=weights)
    smooth = hist + 0.5 * (np.roll(hist, 1) + np.roll(hist, -1))
    k = int(np.argmax(smooth))
    peak = (k + 0.5) * 2.0
    diff = np.abs((theta - peak + 45) % 90 - 45)
    if weights[diff < 5].sum() < 0.35 * weights.sum():
        return 0.0
    rot = peak if peak <= 45 else peak - 90
    if abs(rot) > MAX_FLOOR_ROTATION_DEG:
        return 0.0
    return float(np.radians(rot))


def estimate_geometry(room_np, disparity, mask_bin, surface_type="auto"):
    h, w = mask_bin.shape
    f, cx, cy = FOCAL_FACTOR * max(w, h), w / 2.0, h / 2.0
    normals = surface_normals(disparity, f)

    k = max(3, int(0.01 * max(h, w)))
    eroded = cv2.erode(mask_bin, np.ones((k, k), np.uint8))
    fit_mask = eroded if eroded.sum() > 500 else mask_bin
    coef, dhat, median_normal = _fit_disparity_plane(disparity, fit_mask, normals)
    a, b, c = coef
    d_far, d_near = np.percentile(dhat, 1), np.percentile(dhat, 99)
    span = d_near - d_far

    surface = surface_type if surface_type in ("floor", "wall") else _classify_surface(median_normal, coef, mask_bin)
    if surface == "wall":
        coef, dhat, _ = _fit_disparity_plane(disparity, fit_mask, normals, upright=True)
        a, b, c = coef
        d_far, d_near = np.percentile(dhat, 1), np.percentile(dhat, 99)
        span = d_near - d_far

    gray = cv2.cvtColor(room_np, cv2.COLOR_RGB2GRAY)
    kb = max(5, int(0.02 * max(h, w)))
    band = cv2.dilate(mask_bin, np.ones((kb, kb), np.uint8))
    segs = _detect_segments(gray, band)

    t, source = None, "fronto-parallel"
    if span > 1e-3:
        # Walls: how much perspective the depth map itself supports. On a nearly frontal
        # wall the disparity gradient is tiny, so any vanishing point gets huge leverage;
        # stray converging lines just off the wall (sofa backs, cushions) must not turn it
        # into a grazing-angle wall.
        wall_cap = float(np.clip(0.15 + 4.0 * span / max(d_near, 1e-6), 0.15, MAX_WALL_PERSPECTIVE))
        if surface == "wall":
            t = _horizon_shift_from_vanishing_points(segs, coef, d_far, d_near, wall_eye_row=cy, image_h=h,
                                                     max_rho=wall_cap)
        else:
            t = _horizon_shift_from_vanishing_points(segs, coef, d_far, d_near)
        source = "vanishing lines"
        if t is None and surface == "floor":
            # Level camera: the floor's horizon passes through the principal point.
            t_eye = a * cx + b * cy + c
            if t_eye < d_far - 0.05 * span:
                t, source = t_eye, "eye-level prior"
        if t is None:
            # Last resort: trust the predicted disparity as if it had no shift.
            lo, hi = (0.3, 0.85) if surface == "floor" else (0.0, wall_cap)
            rho = float(np.clip(span / max(d_near, 1e-6), lo, hi))
            if rho > 0.02:
                t, source = d_near - span / rho, "depth"
        if t is not None:
            cap = MAX_PERSPECTIVE if surface == "floor" else wall_cap
            t = min(t, d_near - span / cap)

    if t is not None:
        horizon = np.array([a, b, c - t])
    elif surface == "floor":
        ys, _ = np.nonzero(mask_bin)
        horizon = np.array([0.0, 1.0, -min(cy, ys.min() - 0.05 * h)])
        source = "eye-level prior"
    else:
        horizon = np.array([0.0, 0.0, 1.0])  # line at infinity: plane faces the camera

    # Orient so horizon . p > 0 on the surface (points in front of the camera).
    ys, xs = np.nonzero(mask_bin)
    if horizon @ np.array([np.median(xs), np.median(ys), 1.0]) < 0:
        horizon = -horizon
    normal = np.array([f * horizon[0], f * horizon[1], cx * horizon[0] + cy * horizon[1] + horizon[2]])
    normal /= np.linalg.norm(normal)

    refs = [np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])]
    if surface != "floor":
        refs.reverse()
    for ref in refs:
        e2 = ref - (ref @ normal) * normal
        if np.linalg.norm(e2) > 0.2:
            break
    e2 /= np.linalg.norm(e2)
    e1 = np.cross(normal, e2)
    if e1[0] < 0:
        e1 = -e1

    # Wall tiles are always laid level, so only floors take their rotation from the room's lines.
    # Walls: rotation is always 0, whichever method set the perspective (vanishing lines,
    # depth or fronto-parallel). Tiles on a wall are laid level.
    rot = 0.0
    if surface == "floor" and t is not None:
        rot = _grid_rotation(segs, horizon, normal, e1, e2, f, cx, cy)
    if rot:
        e1, e2 = np.cos(rot) * e1 + np.sin(rot) * e2, -np.sin(rot) * e1 + np.cos(rot) * e2

    if surface == "floor":
        distance = FLOOR_CAMERA_HEIGHT_M
    else:
        rays = np.column_stack([(xs - cx) / f, (ys - cy) / f, np.ones(len(xs))])
        z_unit = 1.0 / np.maximum(rays @ normal, 1e-6)
        distance = WALL_DISTANCE_M / float(np.median(z_unit))

    info = {
        "surface": surface,
        "horizon_source": source,
        "line_segments": int(len(segs)),
        "grid_rotation_deg": round(float(np.degrees(rot)), 1),
        "normal": [round(float(x), 3) for x in normal],
    }
    return PlaneGeometry(surface, normal, distance, e1, e2, f, cx, cy, info)


def plane_coordinates(geo, x0, y0, x1, y1):
    """Metric (u, v) plane coordinates in metres for every pixel of a bounding box."""
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    rx, ry = (xs - geo.cx) / geo.f, (ys - geo.cy) / geo.f
    nr = rx * geo.normal[0] + ry * geo.normal[1] + geo.normal[2]
    scale = geo.distance / np.maximum(nr, 1e-6)
    X, Y, Z = rx * scale, ry * scale, scale
    u = X * geo.e1[0] + Y * geo.e1[1] + Z * geo.e1[2]
    v = X * geo.e2[0] + Y * geo.e2[1] + Z * geo.e2[2]
    return u, v, nr / np.sqrt(rx * rx + ry * ry + 1.0)


# --------------------------------------------------------------------------- tile texture

def build_tile_texture(tile_np, tile_w_mm, tile_h_mm, grout_mm=2.0):
    """Tile image resized to its physical aspect (about 1px per mm) with grout and a soft edge bevel."""
    ppm = min(2.0, 1024.0 / max(tile_w_mm, tile_h_mm))
    tw, th = max(8, int(round(tile_w_mm * ppm))), max(8, int(round(tile_h_mm * ppm)))
    interp = cv2.INTER_AREA if tile_np.shape[1] > tw else cv2.INTER_CUBIC
    tex = cv2.resize(tile_np, (tw, th), interpolation=interp).astype(np.float32)

    g = max(1, int(round(grout_mm * ppm)))
    bevel = max(1, int(round(1.0 * ppm)))
    tex[g:g + bevel, :] *= 1.06          # top edge catches light
    tex[:, g:g + bevel] *= 1.03
    tex[th - bevel:, :] *= 0.90          # bottom / right edges fall into shadow
    tex[:, tw - bevel:] *= 0.93

    # Grout picks up the tile's tint, recessed so slightly darker.
    grout = (0.35 * tex.reshape(-1, 3).mean(0) + 0.65 * np.array([190.0, 186.0, 180.0])) * 0.72
    tex[:g, :] = grout
    tex[:, :g] = grout
    return np.clip(tex, 0, 255)


def sample_tiled(texture, s, t, lod_bias=-0.6):
    """Samples an infinitely repeated texture at texel coords (s, t) with trilinear mip-mapping.

    The texel footprint per screen pixel comes from the coordinate derivatives, so tiles
    near the horizon read as their averaged colour instead of shimmering grout noise.
    """
    th, tw = texture.shape[:2]
    jx = np.hypot(np.gradient(s, axis=1), np.gradient(t, axis=1))
    jy = np.hypot(np.gradient(s, axis=0), np.gradient(t, axis=0))
    footprint = np.maximum(np.maximum(jx, jy), 1e-6)

    levels = [texture]
    while min(levels[-1].shape[:2]) > 4:
        lh, lw = levels[-1].shape[:2]
        levels.append(cv2.resize(texture, (max(1, lw // 2), max(1, lh // 2)), interpolation=cv2.INTER_AREA))

    lod = np.clip(np.log2(footprint) + lod_bias, 0, len(levels) - 1)
    lo = np.floor(lod).astype(np.int32)
    frac = (lod - lo).astype(np.float32)

    out = np.zeros(s.shape + (3,), dtype=np.float32)
    s_unit, t_unit = np.mod(s / tw, 1.0), np.mod(t / th, 1.0)
    for k, lvl in enumerate(levels):
        weight = np.where(lo == k, 1 - frac, 0) + np.where(lo + 1 == k, frac, 0)
        if not weight.any():
            continue
        lh, lw = lvl.shape[:2]
        sample = cv2.remap(lvl, (s_unit * lw).astype(np.float32), (t_unit * lh).astype(np.float32),
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        out += sample * weight[..., None].astype(np.float32)
    return out


# --------------------------------------------------------------------------- lighting

def srgb_to_linear(x):
    x = x / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055) * 255.0


def _masked_blur(img, weight, sigma):
    num = cv2.GaussianBlur(img * weight, (0, 0), sigma)
    den = cv2.GaussianBlur(weight, (0, 0), sigma)
    return num / np.maximum(den, 1e-4)


def intrinsic_shading(room_np, alpha):
    """Shading ratio map (1.0 = average light on the surface) for the selected surface.

    Intrinsic split: luminance = albedo x shading. In LAB, an edge-preserving bilateral
    filter on L keeps crisp shadow and highlight boundaries but flattens fine texture
    (wood grain, old tile patterns). Busy textures such as checkerboards survive a
    bilateral filter, so the crisp detail is faded out wherever edge density says
    "texture" rather than "shadow", leaving only the smooth light falloff there.
    """
    h, w = alpha.shape
    L = cv2.cvtColor(room_np, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32) / 255.0
    lum = np.power((L * 100.0 + 16.0) / 116.0, 3.0)          # linear relative luminance Y
    log_lum = np.log(lum + 1e-3)

    s = min(1.0, 512.0 / max(h, w))
    sw, sh = max(8, int(w * s)), max(8, int(h * s))
    small = cv2.resize(log_lum, (sw, sh), interpolation=cv2.INTER_AREA)
    m_small = cv2.resize(alpha, (sw, sh), interpolation=cv2.INTER_AREA)

    crisp = small
    for _ in range(3):
        crisp = cv2.bilateralFilter(crisp, d=7, sigmaColor=0.35, sigmaSpace=4)

    sigma = 0.04 * max(sw, sh)
    base = _masked_blur(small, m_small + 1e-3, sigma)
    # Old-surface albedo edges (plank-to-plank tone shifts) look like weak shadows; only
    # strong, large-area luminance changes are kept at full strength.
    detail = crisp - _masked_blur(crisp, m_small + 1e-3, sigma)
    detail = np.clip(np.sign(detail) * np.maximum(np.abs(detail) - 0.08, 0) * 0.85, -1.2, 1.0)

    small_gray = cv2.resize(cv2.cvtColor(room_np, cv2.COLOR_RGB2GRAY), (sw, sh), interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(cv2.GaussianBlur(small_gray, (3, 3), 0), 40, 100).astype(np.float32) / 255.0
    win = max(5, int(0.04 * max(sw, sh)) | 1)
    density = cv2.boxFilter(edges, -1, (win, win))
    texture_free = np.clip(1.0 - (density - 0.03) / 0.07, 0.0, 1.0)

    shading = base + detail * texture_free
    shading = cv2.resize(shading, (w, h), interpolation=cv2.INTER_CUBIC)

    inside = alpha > 0.5
    ref = np.percentile(shading[inside], 60) if inside.any() else np.median(shading)
    return np.clip(np.exp(shading - ref), 0.04, 3.0).astype(np.float32)


def compress_highlights(shading, headroom):
    """Smooth shoulder for light above the surface average.

    Shadows (ratio < 1) pass through unchanged. Brightening is rolled off with tanh so
    a sunlit patch tops out at 1 + headroom (e.g. 1.35x) instead of the raw 2-3x ratio
    that bleached tiles to pure white. Order is preserved, so sun patches still read
    brighter than their surroundings and the tile pattern stays visible.
    """
    over = np.maximum(shading - 1.0, 0.0)
    return np.where(shading > 1.0, 1.0 + headroom * np.tanh(over / headroom), shading).astype(np.float32)


def _soft_clip(x, knee=0.8):
    over = x > knee
    out = x.copy()
    out[over] = knee + (1 - knee) * (1 - np.exp(-(x[over] - knee) / (1 - knee)))
    return out


def _noise_sigma(gray):
    """Immerkaer's fast noise estimate, so the clean render matches the photo's grain."""
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    conv = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    h, w = gray.shape
    return float(np.sqrt(np.pi / 2) * np.abs(conv).sum() / (6.0 * (w - 2) * (h - 2)))


# --------------------------------------------------------------------------- gloss

def floor_reflection(room_np, alpha):
    """Planar reflection of everything standing on the floor.

    A plain cv2.flip mirrors around the image centre, which puts reflections in the
    wrong place. Here each column is flipped around its own contact line, the last
    non-floor row above the pixel (a sofa's base, the baseboard). For upright objects
    at constant depth that is exactly where the mirror image lands. Only the
    foreground (inverse mask) is reflected. The result is heavily blurred and faded
    with distance from the contact line; the caller sets the 10-16% opacity.
    Returns (reflection_rgb float32, coverage float32 in [0, 1]).
    """
    h, w = alpha.shape
    floor = alpha > 0.5
    rows = np.arange(h, dtype=np.int32)[:, None]
    occluder_row = np.where(floor, -1, rows)
    axis = np.maximum.accumulate(occluder_row, axis=0)

    valid = floor & (axis >= 0)
    src_y = np.where(valid, 2 * axis - rows, -1).astype(np.float32)
    src_x = np.broadcast_to(np.arange(w, dtype=np.float32)[None, :], (h, w)).copy()
    valid &= src_y >= 0

    reflected = cv2.remap(room_np.astype(np.float32), src_x, src_y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    fg = cv2.remap((1.0 - alpha).astype(np.float32), src_x, src_y, cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    weight = fg * valid

    # Real polished floors reflect sharply at the contact line and blur with distance:
    # blend a light and a strong blur by distance from the mirror axis.
    dist = np.where(valid, rows - axis, h).astype(np.float32)
    layers = []
    for sigma in (0.003 * max(h, w), 0.012 * max(h, w)):
        num = cv2.GaussianBlur(reflected * weight[..., None], (0, 0), sigma)
        den = cv2.GaussianBlur(weight, (0, 0), sigma)
        layers.append((num / np.maximum(den, 1e-4)[..., None], den))
    mix = np.clip(dist / (0.12 * h), 0, 1)[..., None]
    reflected = layers[0][0] * (1 - mix) + layers[1][0] * mix
    coverage = layers[0][1] * (1 - mix[..., 0]) + layers[1][1] * mix[..., 0]

    fade = np.exp(-dist / (0.2 * h))
    return reflected, (np.clip(coverage, 0, 1) * fade).astype(np.float32)


# --------------------------------------------------------------------------- main entry

def render_surface(room_np, tile_np, alpha, tile_w_mm=600.0, tile_h_mm=600.0, scale=1.0,
                   is_glossy=False, surface_type="auto", grout_mm=2.0):
    """Replaces the surface under `alpha` with the tile. Returns (RGB uint8 image, info dict)."""
    h, w = alpha.shape
    mask_bin = (alpha > 0.5).astype(np.uint8)
    if mask_bin.sum() < 0.002 * h * w:
        raise ValueError("The selected surface is too small to tile.")

    disparity = estimate_depth(room_np)
    geo = estimate_geometry(room_np, disparity, mask_bin, surface_type)

    ys, xs = np.nonzero(alpha > 0.01)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    u, v, view_cos = plane_coordinates(geo, x0, y0, x1, y1)

    # Anchor the grid on the nearest part of the surface, so the foreground shows a whole tile.
    my, mx = np.nonzero(mask_bin[y0:y1, x0:x1])
    anchor = np.argmax(my) if geo.surface == "floor" else len(my) // 2
    u -= u[my[anchor], mx[anchor]]
    v -= v[my[anchor], mx[anchor]]

    texture = build_tile_texture(tile_np, tile_w_mm, tile_h_mm, grout_mm)
    th, tw = texture.shape[:2]
    scale = max(0.05, float(scale))
    s = (u * 1000.0 / (tile_w_mm * scale) + 0.5) * tw
    t = (v * 1000.0 / (tile_h_mm * scale)) * th
    tiles = sample_tiled(texture, s, t)
    tiles = cv2.GaussianBlur(tiles, (0, 0), 0.6)   # match camera optics; also tames aliasing

    shading = intrinsic_shading(room_np, alpha)[y0:y1, x0:x1]
    base = srgb_to_linear(tiles)
    if is_glossy:
        # More highlight headroom plus a capped white sheen where the room is brightest.
        lit_shading = compress_highlights(shading, HIGHLIGHT_HEADROOM[1])
        sheen = 0.12 * np.tanh(np.maximum(shading - 1.0, 0) / 0.8)
        lit = base * lit_shading[..., None] + sheen[..., None]
    else:
        lit = base * compress_highlights(shading, HIGHLIGHT_HEADROOM[0])[..., None]
    # Filmic shoulder from 0.65 (linear): bright tiles in sun roll off instead of clipping.
    out = linear_to_srgb(_soft_clip(lit, knee=0.65))

    sigma_n = min(4.0, _noise_sigma(cv2.cvtColor(room_np, cv2.COLOR_RGB2GRAY)))
    if sigma_n > 0.3:
        out += np.random.default_rng(0).normal(0, sigma_n, out.shape[:2])[..., None]

    if is_glossy and geo.surface == "floor":
        refl, coverage = floor_reflection(room_np, alpha)
        fresnel = np.clip(1.0 - view_cos, 0, 1) ** 2
        opacity = ((0.10 + 0.06 * fresnel) * coverage[y0:y1, x0:x1])[..., None]
        out = out * (1 - opacity) + refl[y0:y1, x0:x1] * opacity

    result = room_np.astype(np.float32)
    result[y0:y1, x0:x1] = composite(result[y0:y1, x0:x1], out, alpha[y0:y1, x0:x1])
    return np.clip(result, 0, 255).astype(np.uint8), geo.info


def composite(room, new, alpha):
    """Alpha composite with colour decontamination along the matte edge.

    A fractional-alpha pixel at a chair leg is a mix of the leg and the OLD floor.
    A plain blend keeps that old-floor share, which shows as a halo once the floor
    changes colour. Writing the pixel as I = (1 - a) * F + a * B_old and swapping
    B_old for the new surface gives I + a * (new - B_old), with B_old estimated
    from nearby fully-selected pixels.
    """
    h, w = alpha.shape
    solid = (alpha > 0.95).astype(np.float32)
    sigma = max(1.5, 0.004 * max(h, w))
    den = np.maximum(cv2.GaussianBlur(solid, (0, 0), sigma), 1e-4)
    b_old = cv2.GaussianBlur(room * solid[..., None], (0, 0), sigma) / den[..., None]
    a = alpha[..., None]
    decontaminated = room + a * (new - b_old)
    interior = np.clip((alpha - 0.9) / 0.08, 0, 1)[..., None]
    edge = np.where(a > 0.02, decontaminated, room)
    return edge * (1 - interior) + new * interior
