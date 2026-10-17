"""Photorealistic tile rendering.

Pipeline (all arrays RGB):
  1. Geometry. Depth Anything V2 gives relative disparity. Over a planar surface,
     disparity is exactly an affine function of pixel position, so a RANSAC plane
     fit in (x, y, disparity) space recovers the rigid plane (up to the depth
     model's unknown shift) while ignoring clutter that leaked into the mask.
     Camera roll comes from the room's vertical lines and floors are forced exactly
     horizontal; Hough-line vanishing points (or an eye-level prior) fix the remaining
     shift. Walls skip all of this: a flat 2D mapping, level and plumb (wall_plane).
  2. Every pixel is cast onto that single plane, giving metric (u, v) coordinates.
     Tile courses are therefore straight lines by construction: no depth-driven warp.
  3. Tiles: the tile photo (uniform borders auto-cropped) is sampled with
     trilinear mip-mapping; grout lines are drawn analytically with exact
     box-filter anti-aliasing, so they stay crisp, straight and moire-free from
     the foreground to the horizon.
  4. Lighting: luminance is resampled onto a top-down metric grid of the plane and
     filtered there by physical size, which removes old-tile patterns smaller than
     ~45 cm at every depth while keeping room-scale shadows and sun patches: the
     light level comes from a linear-light local mean, crisp shadow edges from a
     morphological alternating sequential filter. Floors get soft
     contact shadows where objects meet them. Highlights are compressed so
     sunlight never bleaches the tiles.
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
WALL_PERSPECTIVE = False      # walls: estimate yaw (vanishing lines / depth)? Off: flat 2D mapping (wall_plane)
FLOOR_MIN_BOTTOM = 0.8        # a floor selection reaches at least this far down the frame (or sits in its lower half)
WALL_VP_MAX_ANGLE_DEG = 30.0  # only near-horizontal segments can vote for a wall's vanishing point
WALL_VP_EYE_BAND = 0.2        # ...and their crossing must lie within +-20% of image height of eye level
HIGHLIGHT_HEADROOM = (0.5, 0.7)   # max extra brightening above average light: (matte, glossy)
MAX_ROLL_DEG = 8.0            # camera roll estimated from vertical lines is clamped to this
SHADING_FOOTPRINT_M = 0.45    # light/dark detail smaller than this on the real surface is old-tile texture
SHADING_DETAIL_FLOOR = 0.12   # crisp light changes weaker than this (log units, ~12%) are old-surface tone, not light
SHADING_GRID_CELLS = 640      # resolution of the top-down metric grid used for lighting
CONTACT_SHADOW = (0.25, 0.07) # floors: (darkening right at an object's base, decay distance in metres)
BEVEL = (0.10, 1.5)           # tile edges: (darkening next to the grout, width in mm)
SURFACE_EXPOSURE = (0.25, 0.8, 1.15)  # light on the surface vs the rest of the photo: (share of the log ratio, min, max gain)
DETAIL_TRANSFER = (0.10, 0.15)        # small-scale shadows from the photo (log units): (dead zone, max brightening)
DETAIL_TEXTURE = (0.035, 0.07)        # fine-texture energy (log units) up to which they transfer fully / not at all
DETAIL_MIN_M = 0.06                   # shadow detail smaller than this on the surface is grout, seams or veins
GROUT_FADE_PX = (3.0, 14.0)           # floors: on-screen tile size (px) at which grout has faded out / starts to fade
PATTERNS = ("standard", "brick", "herringbone", "checkerboard")
ORIENTATIONS = (0.0, 90.0, 45.0)
MOODS = {"natural": None, "warm": (3000.0, 0.92), "cool": (6500.0, 1.0)}   # (kelvin, exposure)
MOOD_REFERENCE_K = 4800.0     # a white-balanced interior photo, the baseline the moods shift from
MOOD_STRENGTH = 0.45          # share of the full black-body shift applied


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


def estimate_roll(gray):
    """Camera roll (radians) from the room's vertical lines: door frames, wall corners, furniture.

    The returned angle is the tilt of true verticals in the image (positive = leaning
    right going down). Lines near the image centre count most, because camera pitch
    makes off-centre verticals lean. Returns 0 when the evidence is weak or inconsistent.
    """
    h, w = gray.shape
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    min_len = 0.08 * h
    segs = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=int(min_len * 0.5),
                           minLineLength=min_len, maxLineGap=0.01 * max(h, w))
    if segs is None:
        return 0.0
    segs = segs.reshape(-1, 4).astype(np.float64)
    dx, dy = segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]
    flip = dy < 0
    dx[flip], dy[flip] = -dx[flip], -dy[flip]
    tilt = np.degrees(np.arctan2(dx, dy))
    length = np.hypot(dx, dy)
    keep = np.abs(tilt) < 12.0
    if keep.sum() < 4 or length[keep].sum() < 0.6 * h:
        return 0.0
    xm = 0.5 * (segs[keep, 0] + segs[keep, 2])
    weights = length[keep] * np.exp(-((xm - w / 2) / (0.35 * w)) ** 2)
    t = tilt[keep]

    def wmedian(vals):
        order = np.argsort(vals)
        cum = np.cumsum(weights[order])
        return vals[order][np.searchsorted(cum, cum[-1] / 2)]

    med = wmedian(t)
    if wmedian(np.abs(t - med)) > 3.0 or abs(med) < 0.5:
        return 0.0
    return float(np.radians(np.clip(med, -MAX_ROLL_DEG, MAX_ROLL_DEG)))


def _fit_disparity_plane(disparity, mask_bin, normals, axis=None):
    """RANSAC plane fit of disparity = a*x + b*y + c over the surface pixels.

    Depth Anything predicts affine-invariant inverse depth, and the inverse depth of a
    planar surface is exactly affine in pixel position. So this is a rigid 3D plane fit
    (up to the model's unknown scale and shift). RANSAC keeps clutter that leaked into
    the mask (chair legs, bags, feet) from bending it; the inliers are refit by least squares.

    axis: optional unit image direction the disparity gradient is constrained to. Walls
    pass the roll-corrected horizontal, so they come out exactly vertical; floors pass the
    roll-corrected vertical, so they come out exactly horizontal.
    Returns (coef [a, b, c], fitted disparity at the sample points, median normal, inlier share).
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
    d = disparity[ys, xs].astype(np.float64)
    if axis is None:
        design = A
    else:
        design = np.stack([A[:, 0] * axis[0] + A[:, 1] * axis[1], A[:, 2]], axis=1)
    n_par = design.shape[1]

    def to_coef(sol):
        if axis is None:
            return np.asarray(sol, dtype=np.float64)
        return np.array([sol[0] * axis[0], sol[0] * axis[1], sol[1]])

    sol, *_ = np.linalg.lstsq(design, d, rcond=None)
    resid = np.abs(design @ sol - d)
    tau = float(np.clip(2.5 * 1.4826 * np.median(resid), 0.002, 0.05))
    best, best_count = sol, int((resid < tau).sum())
    rng = np.random.default_rng(0)
    for _ in range(200):
        idx = rng.choice(len(d), n_par, replace=False)
        try:
            cand = np.linalg.solve(design[idx], d[idx])
        except np.linalg.LinAlgError:
            continue
        count = int((np.abs(design @ cand - d) < tau).sum())
        if count > best_count:
            best, best_count = cand, count

    inliers = np.abs(design @ best - d) < tau
    for _ in range(3):
        if inliers.sum() < n_par + 10:
            break
        best, *_ = np.linalg.lstsq(design[inliers], d[inliers], rcond=None)
        resid = np.abs(design @ best - d)
        inliers = resid < max(tau, 2.5 * 1.4826 * float(np.median(resid[inliers])))
    coef = to_coef(best)
    return coef, A @ coef, med, float(inliers.mean())


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


def _horizon_shift_from_vanishing_points(segs, coef, d_far, d_near, wall_frame=None, max_rho=0.97):
    """Picks the disparity shift t so the plane's horizon passes through the dominant vanishing point.

    Lines that are parallel in 3D and lie on the surface meet at a vanishing point on its
    horizon, where disparity (extrapolated from the plane fit) equals t. Every pair of
    segments votes; the votes are histogrammed by the perspective strength they imply,
    and the strongest consistent cluster wins.

    For walls (wall_frame = (cx, cy, across, down, image_h)), only horizontal 3D lines
    (baseboards, window heads, tile courses) carry the wall's vanishing point, and they
    converge at eye level. So only segments within WALL_VP_MAX_ANGLE_DEG of the
    roll-corrected horizontal vote, and only crossings near the eye-level line count.
    Window mullions, furniture legs and sofa edges, which made a nearly frontal wall
    look like it was seen at a grazing angle, are ignored.
    """
    if wall_frame is not None:
        cx, cy, across, down, image_h = wall_frame
        seg_dir = np.column_stack([segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]])
        seg_dir /= np.linalg.norm(seg_dir, axis=1, keepdims=True) + 1e-9
        segs = segs[np.abs(seg_dir @ across) >= np.cos(np.radians(WALL_VP_MAX_ANGLE_DEG))]
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
    if wall_frame is not None:
        offset = (v[:, 0] / vz - cx) * down[0] + (v[:, 1] / vz - cy) * down[1]
        ok &= np.abs(offset) < WALL_VP_EYE_BAND * image_h
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


def _grid_rotation(segs, horizon, normal, e1, e2, f, cx, cy, image_w=None):
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
    # Straight (0 deg) unless the room's own lines agree strongly: at least half of the
    # voting length, from long lines (>= 30% of the image width in total). Desk and chair
    # edges in a cluttered office used to tilt grids by 7-13 degrees.
    support = weights[diff < 5].sum()
    if support < 0.5 * weights.sum() or (image_w and support < 0.3 * image_w):
        return 0.0
    rot = peak if peak <= 45 else peak - 90
    # Beyond the limit it is clutter; under 2 degrees it would only read as "almost straight".
    if abs(rot) > MAX_FLOOR_ROTATION_DEG or abs(rot) < 2.0:
        return 0.0
    return float(np.radians(rot))


def wall_plane(h, w, roll=0.0):
    """Flat 2D wall mapping: (u, v) = ((x - cx) * s, (y - cy) * s) metres, s = WALL_DISTANCE_M / f.

    A wall is laid like wallpaper on the image plane: u depends only on x and v only on y,
    with one scale for both axes, so courses are exactly level, columns exactly plumb and a
    600x600 tile stays square everywhere in the frame. No depth model and no vanishing
    points are involved. Wall yaw can't be recovered reliably from one photo: the
    disparity slope across a nearly frontal wall is dominated by Depth Anything's unknown
    shift (an office wall seen head-on came out yawed 36 deg, courses sloping AGAINST the
    room's own lines), the ceiling, AC units and furniture feed stray vanishing points, and
    the roll read from the room's lines tilted a bathroom grid 4.2 deg although its window
    frame is vertical. (WALL_PERSPECTIVE = True restores the 3D estimate.)
    """
    f, cx, cy = FOCAL_FACTOR * max(w, h), w / 2.0, h / 2.0
    info = {
        "surface": "wall",
        "horizon_source": "flat 2D wall",
        "line_segments": 0,
        "grid_rotation_deg": 0.0,
        "roll_deg": round(float(np.degrees(roll)), 1),
        "plane_inliers": None,
        "normal": [0.0, 0.0, 1.0],
    }
    return PlaneGeometry("wall", np.array([0.0, 0.0, 1.0]), WALL_DISTANCE_M,
                         np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), f, cx, cy, info)


def plausible_floor(mask_bin):
    """False when a mask can't be a floor: it stays in the upper part of the frame.

    A floor lies below the horizon and, in a room photo, runs down towards the lower edge
    of the frame. A selection that neither reaches the bottom FLOOR_MIN_BOTTOM of the
    frame nor sits mostly in its lower half is a wall, whatever the request said.
    """
    ys = np.nonzero(mask_bin)[0]
    h = mask_bin.shape[0]
    return bool(np.percentile(ys, 98) / h >= FLOOR_MIN_BOTTOM or ys.mean() / h >= 0.55)


def estimate_geometry(room_np, disparity, mask_bin, surface_type="auto"):
    h, w = mask_bin.shape
    f, cx, cy = FOCAL_FACTOR * max(w, h), w / 2.0, h / 2.0
    gray = cv2.cvtColor(room_np, cv2.COLOR_RGB2GRAY)
    roll = estimate_roll(gray)
    down = np.array([np.sin(roll), np.cos(roll)])      # image direction of gravity
    across = np.array([np.cos(roll), -np.sin(roll)])   # image direction of level lines
    normals = surface_normals(disparity, f)

    k = max(3, int(0.01 * max(h, w)))
    eroded = cv2.erode(mask_bin, np.ones((k, k), np.uint8))
    fit_mask = eroded if eroded.sum() > 500 else mask_bin
    coef, _, median_normal, _ = _fit_disparity_plane(disparity, fit_mask, normals)
    surface = surface_type if surface_type in ("floor", "wall") else _classify_surface(median_normal, coef, mask_bin)
    if surface == "wall" and not WALL_PERSPECTIVE:
        return wall_plane(h, w, roll)

    # Rigid orientation: a wall's inverse depth may only change along level lines (it is
    # vertical); a floor's only along the gravity direction (it is horizontal).
    coef, dhat, _, inlier_share = _fit_disparity_plane(disparity, fit_mask, normals,
                                                       axis=across if surface == "wall" else down)
    a, b, c = coef
    d_far, d_near = np.percentile(dhat, 1), np.percentile(dhat, 99)
    span = d_near - d_far
    if inlier_share < 0.35:
        # Depth disagrees with itself over the mask (mirror, glass, clutter): don't trust the
        # plane's slope; fall back to the eye-level prior (floors) or a frontal wall.
        span = 0.0

    kb = max(5, int(0.02 * max(h, w)))
    band = cv2.dilate(mask_bin, np.ones((kb, kb), np.uint8))
    segs = _detect_segments(gray, band)

    t, source = None, "fronto-parallel"
    # Reached by floors, and by walls only with WALL_PERSPECTIVE (see wall_plane).
    if span > 1e-3:
        # Walls: how much perspective the depth map itself supports. On a nearly frontal
        # wall the disparity gradient is tiny, so any vanishing point gets huge leverage;
        # stray converging lines just off the wall (sofa backs, cushions) must not turn it
        # into a grazing-angle wall.
        wall_cap = float(np.clip(0.15 + 4.0 * span / max(d_near, 1e-6), 0.15, MAX_WALL_PERSPECTIVE))
        if surface == "wall":
            t = _horizon_shift_from_vanishing_points(segs, coef, d_far, d_near,
                                                     wall_frame=(cx, cy, across, down, h), max_rho=wall_cap)
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
        # Level horizon (roll-corrected) through eye level, kept above the floor mask.
        ys, xs = np.nonzero(mask_bin)
        top = float(((xs - cx) * down[0] + (ys - cy) * down[1]).min())
        offset = min(0.0, top - 0.05 * h)
        horizon = np.array([down[0], down[1], -(down[0] * cx + down[1] * cy + offset)])
        source = "eye-level prior"
    else:
        horizon = np.array([0.0, 0.0, 1.0])  # line at infinity: plane faces the camera

    # Orient so horizon . p > 0 on the surface (points in front of the camera).
    ys, xs = np.nonzero(mask_bin)
    if horizon @ np.array([np.median(xs), np.median(ys), 1.0]) < 0:
        horizon = -horizon
    normal = np.array([f * horizon[0], f * horizon[1], cx * horizon[0] + cy * horizon[1] + horizon[2]])
    normal /= np.linalg.norm(normal)

    # Floors: tile courses run away from the camera. Walls (WALL_PERSPECTIVE only): tile
    # columns follow roll-corrected gravity.
    gravity = np.array([down[0], down[1], 0.0])
    refs = [np.array([0.0, 0.0, 1.0]), gravity] if surface == "floor" else [gravity, np.array([0.0, 0.0, 1.0])]
    for ref in refs:
        e2 = ref - (ref @ normal) * normal
        if np.linalg.norm(e2) > 0.2:
            break
    e2 /= np.linalg.norm(e2)
    e1 = np.cross(normal, e2)
    if e1[0] < 0:
        e1 = -e1

    # Grid rotation from the room's lines is for floors only; wall grids are always at 0 deg.
    rot = 0.0
    if surface == "floor" and t is not None:
        rot = _grid_rotation(segs, horizon, normal, e1, e2, f, cx, cy, image_w=w)
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
        "roll_deg": round(float(np.degrees(roll)), 1),
        "plane_inliers": round(inlier_share, 3),
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

def prepare_tile_image(tile_np, tol=18, max_trim=0.2):
    """Crops uniform borders (white frames, photo background) off a catalog tile photo.

    A product shot often has a flat white or grey frame around the tile. Left in, it
    renders as a second, fake grout line around every tile. Rows/columns that are flat
    and match the border colour are trimmed from each side (at most max_trim per side).
    """
    img = tile_np.astype(np.int16)
    h, w = img.shape[:2]
    if h < 16 or w < 16:
        return tile_np
    ring = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
    border = np.median(ring, axis=0)

    def flat(line):
        return (np.abs(line - border).max(axis=1) < tol).mean() > 0.97 and line.std(axis=0).max() < 10

    top, bottom, left, right = 0, 0, 0, 0
    while top < h * max_trim and flat(img[top]):
        top += 1
    while bottom < h * max_trim and flat(img[h - 1 - bottom]):
        bottom += 1
    while left < w * max_trim and flat(img[:, left]):
        left += 1
    while right < w * max_trim and flat(img[:, w - 1 - right]):
        right += 1
    if top + bottom + left + right == 0:
        return tile_np
    return tile_np[top:h - bottom, left:w - right]


def fit_tile_aspect(tile_np, tile_w_mm, tile_h_mm, tolerance=0.15):
    """Centre-crops a tile photo to the physical tile's aspect ratio.

    Stretching a 450x185 terrazzo photo onto a square 600x600 tile distorts every chip
    2.4x. When the photo's aspect differs from the tile's by more than `tolerance`, the
    largest centred crop with the right aspect is used instead.
    """
    h, w = tile_np.shape[:2]
    target = tile_w_mm / tile_h_mm
    actual = w / h
    if abs(np.log(actual / target)) <= np.log(1 + tolerance):
        return tile_np
    if actual > target:
        nw = max(1, int(round(h * target)))
        x0 = (w - nw) // 2
        return tile_np[:, x0:x0 + nw]
    nh = max(1, int(round(w / target)))
    y0 = (h - nh) // 2
    return tile_np[y0:y0 + nh]


def build_tile_texture(tile_np, tile_w_mm, tile_h_mm):
    """Tile image resized to its physical aspect, about 1px per mm (grout is drawn analytically)."""
    ppm = min(2.0, 1024.0 / max(tile_w_mm, tile_h_mm))
    tw, th = max(8, int(round(tile_w_mm * ppm))), max(8, int(round(tile_h_mm * ppm)))
    interp = cv2.INTER_AREA if tile_np.shape[1] > tw else cv2.INTER_CUBIC
    return cv2.resize(tile_np, (tw, th), interpolation=interp).astype(np.float32)


def _line_coverage(a, half, footprint):
    """Exact box-filtered coverage of grout bands [k - half, k + half] (k integer) over a pixel.

    a: tile-index coordinate at the pixel centre; footprint: the pixel's extent along a.
    This is the analytic anti-aliasing of a grid: lines stay perfectly straight and keep
    their true weight at any distance, fading to the average coverage (2 * half) once a
    pixel spans more than a whole tile, where sampled lines would break into moire.
    """
    fw = np.maximum(footprint, 1e-6)
    k = np.round(a)
    lo, hi = a - fw / 2, a + fw / 2
    cov = np.zeros_like(a)
    for dk in (-1.0, 0.0, 1.0):
        c = k + dk
        cov += np.clip(np.minimum(hi, c + half) - np.maximum(lo, c - half), 0, None)
    cov /= fw
    far = np.clip(fw - 1.0, 0, 1)
    return np.clip(cov * (1 - far) + 2 * half * far, 0, 1)


@dataclass
class TileLayout:
    """Where each pixel falls in its tile, for one laying pattern.

    In the pattern's own frame every tile of every supported pattern is axis-aligned, so
    a pixel is described by its position (lu, lv) in mm inside its tile, the tile's extent
    (su, sv) along u and v, the quarter-turns `rot` its texture is laid at (0, 1 = 90 deg,
    2 = 180 deg) and a tone multiplier `shade`. Grout and bevels then stay analytic.
    """
    lu: np.ndarray
    lv: np.ndarray
    su: object
    sv: object
    rot: object = 0
    shade: object = 1.0


def tile_layout(um, vm, tile_w, tile_h, pattern="standard"):
    """Pattern-frame coordinates (mm) -> TileLayout. tile_w runs along u, tile_h along v.

    standard     straight grid
    brick        running bond: every other row shifted half a tile along u
    checkerboard quarter-turn: alternate tiles rotated 90 deg (180 deg for oblong tiles),
                 which a directional texture or sheen shows as a checker
    herringbone  90 deg herringbone of L x W tiles (L = long side, laid along u). Closed
                 form: horizontal tiles sit at i*(W, W) + j*(L, -L); along x - y the rows
                 are 2L apart and each spans less than 2L, so row j is unique, then column i.
                 A pixel outside its candidate horizontal tile is in the vertical one.
    """
    if pattern == "brick":
        row = np.floor(vm / tile_h)
        shift = 0.5 * tile_w * np.mod(row, 2)
        return TileLayout(np.mod(um + shift, tile_w), np.mod(vm, tile_h), tile_w, tile_h)
    if pattern == "checkerboard":
        odd = np.mod(np.floor(um / tile_w) + np.floor(vm / tile_h), 2).astype(bool)
        square = abs(tile_w - tile_h) <= 0.02 * max(tile_w, tile_h)
        return TileLayout(np.mod(um, tile_w), np.mod(vm, tile_h), tile_w, tile_h,
                          rot=np.where(odd, 1 if square else 2, 0), shade=np.where(odd, 0.965, 1.0))
    if pattern == "herringbone":
        L, W = max(tile_w, tile_h), min(tile_w, tile_h)
        s = um - vm
        j = np.floor((s + W) / (2 * L))
        i = np.floor((vm + j * L) / W)
        xh, yh = um - i * W - j * L, vm - i * W + j * L
        in_h = (xh >= 0) & (xh < L) & (yh >= 0) & (yh < W)
        jv = np.floor((s - L + W) / (2 * L))
        iv = np.floor((um - L - jv * L) / W)
        xv, yv = um - L - iv * W - jv * L, vm - (W - L) - iv * W + jv * L
        return TileLayout(np.where(in_h, xh, xv), np.where(in_h, yh, yv),
                          np.where(in_h, L, W), np.where(in_h, W, L), rot=np.where(in_h, 0, 1))
    return TileLayout(np.mod(um, tile_w), np.mod(vm, tile_h), tile_w, tile_h)


def _smoothstep(lo, hi, x):
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def render_tile_layer(tile_img, um, vm, tile_w_mm, tile_h_mm, grout_mm, pattern="standard", grout_rgb=None,
                      fade_far=False):
    """RGB float32 tiles at pattern-frame coordinates (um, vm) in mm: texture, analytic grout and edge bevel.

    grout_rgb: exact grout colour (sRGB 0-255), or None for a neutral grout tinted by the tile.
    fade_far: fade grout and bevels to their average tone where tiles shrink below
    GROUT_FADE_PX on screen (floors receding to the horizon). Exact box filtering keeps
    each line correct, but dozens of 1px lines a few pixels apart still read as a dense,
    shimmering mesh; a camera lens would already have softened them.
    """
    if pattern == "herringbone" and tile_h_mm > tile_w_mm:
        # Herringbone lays the long side along u; turn the photo so its long side matches.
        tile_img = np.ascontiguousarray(np.rot90(tile_img, -1))
        tile_w_mm, tile_h_mm = tile_h_mm, tile_w_mm
    texture = build_tile_texture(tile_img, tile_w_mm, tile_h_mm)
    th, tw = texture.shape[:2]
    lay = tile_layout(um, vm, tile_w_mm, tile_h_mm, pattern)

    # Derivatives of the continuous pattern coordinates (mm per pixel). The per-tile
    # coordinates jump at every joint, so their own gradients would be useless there.
    dux, duy = np.gradient(um, axis=1), np.gradient(um, axis=0)
    dvx, dvy = np.gradient(vm, axis=1), np.gradient(vm, axis=0)

    a, b = lay.lu / lay.su, lay.lv / lay.sv
    rot = lay.rot
    if np.ndim(rot) == 0 and rot == 0:
        s, t = a * tw, b * th
        derivs = (dux / lay.su * tw, dvx / lay.sv * th, duy / lay.su * tw, dvy / lay.sv * th)
    else:
        # Texture laid at a quarter-turn (rot 1: s along v, t against u) or a half-turn.
        r1, r2 = rot == 1, rot == 2

        def pick(quarter, half, straight):
            return np.where(r1, quarter, np.where(r2, half, straight))

        s = pick(b, 1 - a, a) * tw
        t = pick(1 - a, 1 - b, b) * th
        ga_x, ga_y = dux / lay.su, duy / lay.su
        gb_x, gb_y = dvx / lay.sv, dvy / lay.sv
        derivs = (pick(gb_x, -ga_x, ga_x) * tw, pick(-ga_x, -gb_x, gb_x) * th,
                  pick(gb_y, -ga_y, ga_y) * tw, pick(-ga_y, -gb_y, gb_y) * th)
    rgb = sample_tiled(texture, s, t, derivs=derivs)
    if np.ndim(lay.shade) or lay.shade != 1.0:
        rgb *= np.asarray(lay.shade, np.float32)[..., None]

    # Pixel footprint along each tile axis (box extent of the pixel in tile units).
    fa = (np.abs(dux) + np.abs(duy)) / lay.su
    fb = (np.abs(dvx) + np.abs(dvy)) / lay.sv
    ha, hb = 0.5 * grout_mm / lay.su, 0.5 * grout_mm / lay.sv
    # Share of each axis' line contrast that survives (1 = crisp); fa is tiles per pixel.
    keep_a = keep_b = 1.0
    if fade_far:
        lo, hi = GROUT_FADE_PX
        keep_a = _smoothstep(lo, hi, 1.0 / np.maximum(fa, 1e-6))
        keep_b = _smoothstep(lo, hi, 1.0 / np.maximum(fb, 1e-6))

    # Edge bevel: tiles darken slightly towards the grout; invisible once sub-pixel.
    strength, width_mm = BEVEL
    ea = np.maximum((np.abs(a - np.round(a)) - ha) * lay.su, 0)
    eb = np.maximum((np.abs(b - np.round(b)) - hb) * lay.sv, 0)
    va = np.clip(width_mm / np.maximum(fa * lay.su, 1e-6), 0, 1) * keep_a
    vb = np.clip(width_mm / np.maximum(fb * lay.sv, 1e-6), 0, 1) * keep_b
    dark = np.maximum(np.exp(-ea / width_mm) * va, np.exp(-eb / width_mm) * vb)
    rgb *= (1.0 - strength * dark)[..., None].astype(np.float32)

    if grout_mm <= 0:
        return rgb
    ca, cb = _line_coverage(a, ha, fa), _line_coverage(b, hb, fb)
    if fade_far:
        # Towards the average coverage, so the far floor keeps its overall tone.
        avg_a, avg_b = np.minimum(2 * ha, 1.0), np.minimum(2 * hb, 1.0)
        ca = avg_a + (ca - avg_a) * keep_a
        cb = avg_b + (cb - avg_b) * keep_b
    cov = 1.0 - (1.0 - ca) * (1.0 - cb)
    if grout_rgb is None:
        # Grout picks up the tile's tint and sits recessed, so slightly darker.
        grout = (0.35 * texture.reshape(-1, 3).mean(0) + 0.65 * np.array([190.0, 186.0, 180.0])) * 0.72
    else:
        grout = np.asarray(grout_rgb, np.float64)
    cov = cov[..., None].astype(np.float32)
    return rgb * (1 - cov) + grout.astype(np.float32) * cov


def sample_tiled(texture, s, t, lod_bias=-0.25, max_taps=8, derivs=None):
    """Samples an infinitely repeated texture at texel coords (s, t) with anisotropic filtering.

    A floor seen at an angle is squashed far more along one screen direction than the
    other. Plain trilinear mip-mapping picks its blur from the squashed direction and
    applies it in every direction, smearing fine texture (terrazzo chips, wood grain,
    marble veins) into a flat grey. Here the mip level comes from the short axis of the
    pixel footprint, and up to `max_taps` samples are averaged along the long axis, the
    way GPUs do anisotropic filtering. Distant tiles still average out instead of
    shimmering.

    derivs: (ds/dx, dt/dx, ds/dy, dt/dy) when s, t are per-tile coordinates that jump at
    joints (bricks, herringbone); otherwise they come from np.gradient.
    """
    th, tw = texture.shape[:2]
    if derivs is None:
        dsx, dtx = np.gradient(s, axis=1), np.gradient(t, axis=1)
        dsy, dty = np.gradient(s, axis=0), np.gradient(t, axis=0)
    else:
        dsx, dtx, dsy, dty = derivs
    lx, ly = np.hypot(dsx, dtx), np.hypot(dsy, dty)
    x_major = lx >= ly
    major = np.maximum(np.where(x_major, lx, ly), 1e-6)
    minor = np.maximum(np.where(x_major, ly, lx), 1e-6)
    n = np.clip(np.ceil(major / minor), 1, max_taps)
    taps = int(np.clip(np.ceil(np.percentile(n, 95)), 1, max_taps))
    ax_s = np.where(x_major, dsx, dsy)
    ax_t = np.where(x_major, dtx, dty)

    levels = [texture]
    while min(levels[-1].shape[:2]) > 4:
        lh, lw = levels[-1].shape[:2]
        levels.append(cv2.resize(texture, (max(1, lw // 2), max(1, lh // 2)), interpolation=cv2.INTER_AREA))

    lod = np.clip(np.log2(major / n) + lod_bias, 0, len(levels) - 1)
    lo = np.floor(lod).astype(np.int32)
    frac = (lod - lo).astype(np.float32)

    offsets = [((k + 0.5) / taps - 0.5) for k in range(taps)]
    tap_coords = [(np.mod((s + o * ax_s) / tw, 1.0), np.mod((t + o * ax_t) / th, 1.0)) for o in offsets]

    out = np.zeros(s.shape + (3,), dtype=np.float32)
    for k, lvl in enumerate(levels):
        weight = np.where(lo == k, 1 - frac, 0) + np.where(lo + 1 == k, frac, 0)
        if not weight.any():
            continue
        lh, lw = lvl.shape[:2]
        acc = np.zeros(s.shape + (3,), dtype=np.float32)
        for su, tu in tap_coords:
            acc += cv2.remap(lvl, (su * lw).astype(np.float32), (tu * lh).astype(np.float32),
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        out += (acc / len(tap_coords)) * weight[..., None].astype(np.float32)
    return out


# --------------------------------------------------------------------------- lighting

def srgb_to_linear(x):
    x = x / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055) * 255.0


def _fill_invalid(img, valid):
    """Fills invalid cells from nearby valid ones (normalised convolution, coarse to fine)."""
    out = img.astype(np.float32).copy()
    filled = valid.copy()
    vf = valid.astype(np.float32)
    weighted = img.astype(np.float32) * vf
    for sigma in (2, 6, 18, 54, 162):
        num = cv2.GaussianBlur(weighted, (0, 0), sigma)
        den = cv2.GaussianBlur(vf, (0, 0), sigma)
        take = ~filled & (den > 1e-3)
        out[take] = num[take] / den[take]
        filled |= take
    if (~filled).any():
        out[~filled] = float(img[valid].mean()) if valid.any() else 0.0
    return out


def _remove_small_features(img, size):
    """Destroys light and dark features smaller than `size` cells, keeping larger regions intact.

    An alternating sequential filter: at growing scales, a closing fills dark features
    (black diamonds, grout lines, plank gaps) and an opening removes light ones (veins,
    specks). Unlike blurring, it does not smear a strong pattern into a soft ghost, and
    unlike a bilateral filter, a high-contrast pattern edge cannot survive it. A median and
    a light Gaussian then clean up the blocky residue. Shadows and sun patches larger
    than `size` keep their level and edges.
    """
    out = img.astype(np.float32)
    k = 3
    while True:
        kk = min(k, size) | 1
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (kk, kk))
        out = cv2.morphologyEx(cv2.morphologyEx(out, cv2.MORPH_CLOSE, se), cv2.MORPH_OPEN, se)
        if kk >= size:
            break
        k = max(k + 2, int(k * 1.6))
    lo, hi = float(out.min()), float(out.max())
    q = ((out - lo) / max(hi - lo, 1e-6) * 255.0).astype(np.uint8)
    q = cv2.medianBlur(q, max(3, (size // 2) | 1))
    out = q.astype(np.float32) / 255.0 * (hi - lo) + lo
    return cv2.GaussianBlur(out, (0, 0), max(1.0, size / 6.0))


def _shading_from_grid(log_lum, size):
    """Albedo-free shading (log) on the metric grid: two bands from two estimators.

    Low band: the local mean of LINEAR light. A camera averages distant, unresolved
    patterns in linear light too, so this level is the same near and far; the
    morphological filter alone takes the near-field white level but the far-field
    average, which fakes a darkening towards the horizon on checkerboards.
    High band: the morphological filter's crisp edges (sun patches, shadow borders),
    with changes under SHADING_DETAIL_FLOOR dropped as plank-to-plank tone.
    """
    lin_mean = np.log(cv2.GaussianBlur(np.exp(log_lum), (0, 0), max(1.0, size / 2.0)) + 1e-6)
    low = cv2.GaussianBlur(lin_mean, (0, 0), max(1.0, float(size)))
    crisp = _remove_small_features(log_lum, size)
    high = crisp - cv2.GaussianBlur(crisp, (0, 0), max(1.0, float(size)))
    high = np.sign(high) * np.maximum(np.abs(high) - SHADING_DETAIL_FLOOR, 0)
    return (low + high).astype(np.float32)


def illumination_map(room_np, alpha, geo, u, v, bbox, with_detail=False):
    """Shading ratio (1.0 = typical light on the surface) for the bbox, free of old-tile texture.

    Intrinsic split: luminance = albedo x shading. Old tiles shrink with distance, so a
    fixed image-space blur either leaves near-field patterns as ghosts or wipes out
    far-field shadows. Here the photo's luminance is resampled onto a top-down metric
    grid of the actual plane, so every feature appears at its true physical size; there
    everything smaller than SHADING_FOOTPRINT_M is removed and room-scale light (window
    falloff, furniture shadows, sun patches) is kept (see _shading_from_grid). The
    result is mapped back per pixel.

    with_detail: return (shading, detail, weight), adding the small-scale shadows of
    _shadow_detail_grid as a multiplier and their per-pixel transfer weight.
    """
    x0, y0, x1, y1 = bbox
    surface = alpha > 0.5
    sub = surface[y0:y1, x0:x1]
    if sub.sum() < 50:
        flat = np.ones(u.shape, np.float32)
        return (flat, flat, np.zeros(u.shape, np.float32)) if with_detail else flat

    L = cv2.cvtColor(room_np, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32) / 255.0
    log_lum = np.log(np.power((L * 100.0 + 16.0) / 116.0, 3.0) + 1e-3).astype(np.float32)

    us, vs = u[sub], v[sub]
    u_lo, u_hi = np.percentile(us, [0.5, 99.5])
    v_lo, v_hi = np.percentile(vs, [0.5, 99.5])
    # Grid fine enough for crisp shadow edges, coarse enough that the footprint is <= 24 cells.
    cell = max(max(u_hi - u_lo, v_hi - v_lo) / SHADING_GRID_CELLS, SHADING_FOOTPRINT_M / 24)
    gw, gh = int((u_hi - u_lo) / cell) + 2, int((v_hi - v_lo) / cell) + 2
    uu, vv = np.meshgrid(u_lo + cell * np.arange(gw), v_lo + cell * np.arange(gh))
    pts = geo.distance * geo.normal + uu[..., None] * geo.e1 + vv[..., None] * geo.e2
    z = pts[..., 2]
    ahead = z > 1e-6
    zs = np.where(ahead, z, 1.0)
    map_x = np.where(ahead, geo.f * pts[..., 0] / zs + geo.cx, -10).astype(np.float32)
    map_y = np.where(ahead, geo.f * pts[..., 1] / zs + geo.cy, -10).astype(np.float32)

    grid = cv2.remap(log_lum, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # Sample light only away from the mask border, where pixels can still be part
    # object (a bright ceramic base, a dark shoe); the border is filled from the interior.
    k = max(3, int(round(0.003 * max(alpha.shape))) | 1)
    core = cv2.erode(surface.astype(np.uint8), np.ones((k, k), np.uint8))
    if core.sum() < 0.5 * surface.sum():
        core = surface.astype(np.uint8)
    valid = cv2.remap(core.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.5
    filled = _fill_invalid(grid, valid)
    size = max(3, int(round(SHADING_FOOTPRINT_M / cell)))
    filtered = _shading_from_grid(filled, size)

    gx = ((u - u_lo) / cell).astype(np.float32)
    gy = ((v - v_lo) / cell).astype(np.float32)
    shading = cv2.remap(filtered, gx, gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    ref = np.percentile(shading[sub], 60)
    norm = np.clip(np.exp(shading - ref), 0.04, 3.0).astype(np.float32)
    if not with_detail:
        return norm

    h, w = alpha.shape
    on_surface = cv2.remap(surface.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.5
    in_frame = (map_x >= 0) & (map_x <= w - 1) & (map_y >= 0) & (map_y <= h - 1)
    log_detail, weight = _shadow_detail_grid(filled, filtered, valid, on_surface, in_frame & ~on_surface, cell, size)
    detail = np.exp(cv2.remap(log_detail, gx, gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
    weight = cv2.remap(weight, gx, gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return norm, detail.astype(np.float32), np.clip(weight, 0, 1).astype(np.float32)


def _shadow_detail_grid(filled, filtered, valid, on_surface, occluder, cell, size):
    """Small-scale shadows on the metric grid: (log multiplier, transfer weight).

    _shading_from_grid keeps only light patterns larger than SHADING_FOOTPRINT_M, because
    anything smaller could be the old surface's own pattern, so the soft shadow under a
    chair, the darkening into a corner or along a skirting board was lost. Here the band
    between DETAIL_MIN_M and the footprint comes back, as a Multiply blend in linear light,
    but only where it can only be light:
      - features under DETAIL_MIN_M (grout lines, plank seams, veins, specks) are removed
        first with the same morphological filter;
      - `weight` fades the transfer out where the old surface is textured: 1 below
        DETAIL_TEXTURE[0] of fine-texture energy, 0 above DETAIL_TEXTURE[1];
      - only light or dark regions touching an occluder (an object, the wall at the floor's
        edge) are kept. A cast or contact shadow starts at what casts it; a stain or a
        darker old tile in the open floor is an isolated blob, i.e. albedo;
      - changes within DETAIL_TRANSFER's dead zone are mottling; brightening is capped.
    """
    if valid.sum() < 50:
        return np.zeros(filled.shape, np.float32), np.zeros(filled.shape, np.float32)
    mid = _remove_small_features(filled, max(3, int(round(DETAIL_MIN_M / cell))))
    vf = valid.astype(np.float32)
    sigma = max(1.0, size / 2.0)
    energy = (cv2.GaussianBlur(np.abs(filled - mid) * vf, (0, 0), sigma)
              / np.maximum(cv2.GaussianBlur(vf, (0, 0), sigma), 1e-3))
    lo, hi = DETAIL_TEXTURE
    weight = np.clip((hi - energy) / (hi - lo), 0.0, 1.0).astype(np.float32)

    d = mid - filtered
    d = d - np.median(d[valid])                    # the surface's own tone is the baseline
    dead, up = DETAIL_TRANSFER
    d = np.minimum(np.sign(d) * np.maximum(np.abs(d) - dead, 0.0), up)
    reach = cv2.dilate(occluder.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    keep = np.zeros(d.shape, bool)
    for part in (d < 0, d > 0):
        n, labels = cv2.connectedComponents((part & on_surface).astype(np.uint8), connectivity=8)
        if n > 1:
            ids = np.unique(labels[reach & (labels > 0)])
            keep |= np.isin(labels, ids[ids > 0])
    return np.where(keep, weight * d, 0.0).astype(np.float32), weight


def surface_exposure(room_np, alpha):
    """Exposure gain from how much light the selected surface gets compared with the rest of the photo.

    Shading is normalised to the surface's own typical light, so a wall in deep shade and
    one in full sun would otherwise get equally bright tiles. The ratio of the log-mean
    luminance inside the matte to the log-mean outside it measures that light, but also
    the old material's colour (a black floor reads as "dim", a white wall as "bright"),
    so only SURFACE_EXPOSURE's share of the ratio is applied, within its bounds.
    """
    lum = np.log(srgb_to_linear(cv2.cvtColor(room_np, cv2.COLOR_RGB2GRAY).astype(np.float32)) + 1e-3)
    inside, outside = alpha > 0.5, alpha < 0.05
    if inside.sum() < 50 or outside.sum() < 50:
        return 1.0
    share, lo, hi = SURFACE_EXPOSURE
    return float(np.clip(np.exp(share * (lum[inside].mean() - lum[outside].mean())), lo, hi))


def contact_shadow(alpha, u, v, bbox):
    """Soft ambient occlusion where objects meet the floor (chair legs, sofa bases, feet, skirting).

    The texture filter removes small dark blobs, real contact shadows included, so these
    are rebuilt from geometry: darkening that decays with the metric distance to the
    nearest non-floor pixel directly above (the object standing there).
    """
    x0, y0, x1, y1 = bbox
    h = alpha.shape[0]
    rows = np.arange(h, dtype=np.int32)[:, None]
    axis = np.maximum.accumulate(np.where(alpha > 0.5, -1, rows), axis=0)[y0:y1, x0:x1]
    d_px = (rows[y0:y1] - axis).astype(np.float32)
    m_per_px = np.hypot(np.gradient(u, axis=0), np.gradient(v, axis=0)).astype(np.float32)
    strength, decay = CONTACT_SHADOW
    ao = 1.0 - strength * np.exp(-d_px * m_per_px / decay)
    return np.where(axis >= 0, ao, 1.0).astype(np.float32)


def compress_highlights(shading, headroom):
    """Smooth shoulder for light above the surface average.

    Shadows (ratio < 1) pass through unchanged. Brightening is rolled off with tanh so
    a sunlit patch tops out at 1 + headroom (e.g. 1.35x) instead of the raw 2-3x ratio
    that bleached tiles to pure white. Order is preserved, so sun patches still read
    brighter than their surroundings and the tile pattern stays visible.
    """
    over = np.maximum(shading - 1.0, 0.0)
    return np.where(shading > 1.0, 1.0 + headroom * np.tanh(over / headroom), shading).astype(np.float32)


def scene_exposure(room_np):
    """How far the photo is under-exposed (1.0 = normally exposed, down to 0.25).

    Shading is normalised to the surface's own typical light, so on its own a new tile in
    a dark, under-exposed photo would render at full brightness and glow. The photo's
    near-white level (97th percentile of linear luminance) sets the overall exposure;
    normally exposed photos are left untouched (range 0.05-1).
    """
    lum = srgb_to_linear(cv2.cvtColor(room_np, cv2.COLOR_RGB2GRAY).astype(np.float32))
    return float(np.clip(np.percentile(lum, 97) / 0.6, 0.05, 1.0))


def tone_map(lit, knee=0.65):
    """Filmic shoulder on luminance only, so bright, colourful tiles keep their colour.

    Compressing each RGB channel separately pushes bright colours towards white and
    washes out a tile's pattern. Here luminance is compressed and RGB scaled with it;
    only a colour that would still exceed 1.0 is desaturated just enough to fit.
    """
    lum = 0.2126 * lit[..., 0] + 0.7152 * lit[..., 1] + 0.0722 * lit[..., 2]
    target = _soft_clip(lum, knee)
    out = lit * (target / np.maximum(lum, 1e-6))[..., None]
    peak = out.max(axis=-1)
    over = peak > 1.0
    if over.any():
        t = target[over][..., None]
        k = ((1.0 - t) / np.maximum(peak[over] - t[..., 0], 1e-6)[..., None])
        out[over] = t + (out[over] - t) * np.clip(k, 0, 1)
    return np.clip(out, 0, 1)


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

def kelvin_to_linear_rgb(kelvin):
    """Linear-sRGB colour (luminance 1) of a black-body light, via the Planckian locus in CIE xy.

    Kim et al.'s cubic fit of the locus (1667-25000 K), then xyY -> XYZ -> linear sRGB.
    """
    T = float(np.clip(kelvin, 1667.0, 25000.0))
    if T <= 4000:
        x = -0.2661239e9 / T ** 3 - 0.2343589e6 / T ** 2 + 0.8776956e3 / T + 0.179910
    else:
        x = -3.0258469e9 / T ** 3 + 2.1070379e6 / T ** 2 + 0.2226347e3 / T + 0.240390
    if T <= 2222:
        y = -1.1063814 * x ** 3 - 1.34811020 * x ** 2 + 2.18555832 * x - 0.20219683
    elif T <= 4000:
        y = -0.9549476 * x ** 3 - 1.37418593 * x ** 2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x ** 3 - 5.87338670 * x ** 2 + 3.75112997 * x - 0.37001483
    xyz = np.array([x / y, 1.0, (1.0 - x - y) / y])
    m = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
    return np.maximum(m @ xyz, 1e-4)


def mood_gains(mood):
    """Per-channel light gains (linear RGB) for a lighting mood, or None for the photo's own light.

    The mood's colour temperature is taken relative to MOOD_REFERENCE_K, a typical
    white-balanced interior, at MOOD_STRENGTH (a full black-body shift reads as a colour
    filter, not as different light). Gains keep luminance, so only the light's colour changes.
    """
    if mood not in MOODS or MOODS[mood] is None:
        return None
    kelvin, exposure = MOODS[mood]
    g = kelvin_to_linear_rgb(kelvin) / kelvin_to_linear_rgb(MOOD_REFERENCE_K)
    g = 1.0 + MOOD_STRENGTH * (g - 1.0)
    g /= 0.2126 * g[0] + 0.7152 * g[1] + 0.0722 * g[2]
    return (g * exposure).astype(np.float32)


def apply_mood(image, mood):
    """Relights a whole RGB image (uint8 or float 0-255) for a lighting mood.

    The mood changes the light falling on the scene, which multiplies every surface's
    reflected colour by the illuminant (von Kries, in linear light): the new tiles' lighting
    layer and the rest of the photo shift together, so the render stays coherent.
    """
    gains = mood_gains(mood)
    if gains is None:
        return image
    lin = srgb_to_linear(np.asarray(image, np.float32)) * gains
    return linear_to_srgb(tone_map(lin, knee=0.9)).astype(np.float32)


def render_surface(room_np, tile_np, alpha, tile_w_mm=600.0, tile_h_mm=600.0, scale=1.0,
                   is_glossy=False, surface_type="auto", grout_mm=2.0, pattern="standard",
                   orientation_deg=0.0, grout_rgb=None, mood="natural"):
    """Replaces the surface under `alpha` with the tile. Returns (RGB uint8 image, info dict).

    pattern: one of PATTERNS. orientation_deg: the layout's turn on the surface (0, 90 or
    45 for a diamond lay). grout_rgb: exact grout colour or None. mood: one of MOODS.
    """
    if pattern not in PATTERNS:
        raise ValueError(f"Unknown pattern '{pattern}'.")
    h, w = alpha.shape
    mask_bin = (alpha > 0.5).astype(np.uint8)
    if mask_bin.sum() < 0.002 * h * w:
        raise ValueError("The selected surface is too small to tile.")

    override = None
    if surface_type == "floor" and not plausible_floor(mask_bin):
        # A forced "floor" on what can only be a wall gave walls a receding floor projection.
        surface_type, override = "wall", "floor requested, but the selection lies in the upper frame: rendered as a wall"
    if surface_type == "wall" and not WALL_PERSPECTIVE:
        geo = wall_plane(h, w)                 # flat 2D mapping: no depth model, no vanishing points
    else:
        geo = estimate_geometry(room_np, estimate_depth(room_np), mask_bin, surface_type)

    ys, xs = np.nonzero(alpha > 0.01)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    bbox = (x0, y0, x1, y1)
    u, v, view_cos = plane_coordinates(geo, x0, y0, x1, y1)

    shading, detail, transfer = illumination_map(room_np, alpha, geo, u, v, bbox, with_detail=True)
    if geo.surface == "floor":
        # The synthetic contact shadow stands in for the real one only where the photo's
        # own small-scale shadows are not transferred (patterned old floors).
        ao = contact_shadow(alpha, u, v, bbox)
        shading *= 1.0 - (1.0 - ao) * (1.0 - transfer)
    shading *= detail

    # Anchor the grid on the nearest part of the surface, so the foreground shows a whole tile.
    my, mx = np.nonzero(mask_bin[y0:y1, x0:x1])
    anchor = np.argmax(my) if geo.surface == "floor" else len(my) // 2
    u0, v0 = u[my[anchor], mx[anchor]], v[my[anchor], mx[anchor]]

    # Pattern-frame coordinates in mm, turned by the chosen orientation around the anchor,
    # which sits mid-tile along u. Grout scales with the tile so the size slider keeps proportions.
    scale = max(0.05, float(scale))
    tile_w, tile_h = tile_w_mm * scale, tile_h_mm * scale
    du, dv = (u - u0) * 1000.0, (v - v0) * 1000.0
    if orientation_deg:
        c, s = np.cos(np.radians(orientation_deg)), np.sin(np.radians(orientation_deg))
        du, dv = c * du + s * dv, -s * du + c * dv
    um, vm = du + 0.5 * tile_w, dv
    tile_img = fit_tile_aspect(prepare_tile_image(tile_np), tile_w_mm, tile_h_mm)
    tiles = render_tile_layer(tile_img, um, vm, tile_w, tile_h, grout_mm * scale, pattern=pattern, grout_rgb=grout_rgb,
                              fade_far=geo.surface == "floor")
    tiles = cv2.GaussianBlur(tiles, (0, 0), 0.35)   # a touch of camera optics; filtering handles aliasing
    gain = surface_exposure(room_np, alpha)
    base = srgb_to_linear(tiles) * (scene_exposure(room_np) * gain)
    if is_glossy:
        # More highlight headroom plus a capped white sheen where the room is brightest.
        lit_shading = compress_highlights(shading, HIGHLIGHT_HEADROOM[1])
        sheen = 0.12 * np.tanh(np.maximum(shading - 1.0, 0) / 0.8)
        lit = base * lit_shading[..., None] + sheen[..., None]
    else:
        lit = base * compress_highlights(shading, HIGHLIGHT_HEADROOM[0])[..., None]
    # Multiply blend in linear light keeps the tile's own texture; the filmic shoulder on
    # luminance (from 0.65) rolls off sunlit highlights without bleaching colour.
    out = linear_to_srgb(tone_map(lit, knee=0.65))

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
    result = apply_mood(result, mood)
    inside = alpha[y0:y1, x0:x1] > 0.5
    info = {**geo.info, "pattern": pattern, "orientation_deg": orientation_deg, "mood": mood,
            "exposure_gain": round(gain, 3), "shadow_transfer": round(float(transfer[inside].mean()), 2)}
    if override:
        info["surface_override"] = override
    return np.clip(result, 0, 255).astype(np.uint8), info


def composite(room, new, alpha):
    """Alpha composite with colour decontamination along the matte edge.

    A fractional-alpha pixel at a chair leg is a mix of the leg and the OLD floor.
    A plain blend keeps that old-floor share, which shows as a halo once the floor
    changes colour. Writing the pixel as I = (1 - a) * F + a * B_old and swapping
    B_old for the new surface gives I + a * (new - B_old), with B_old estimated
    from nearby fully-selected pixels.

    That estimate is only meaningful where the old surface is locally uniform. On a
    high-contrast old pattern (black/white checker) the local average is a grey that
    no edge pixel actually contains, and the correction paints bright or dark smears
    along object bases; there a plain alpha blend is used instead.
    """
    h, w = alpha.shape
    solid = (alpha > 0.95).astype(np.float32)
    sigma = max(1.5, 0.004 * max(h, w))
    den = np.maximum(cv2.GaussianBlur(solid, (0, 0), sigma), 1e-4)
    b_old = cv2.GaussianBlur(room * solid[..., None], (0, 0), sigma) / den[..., None]
    sq = cv2.GaussianBlur(room * room * solid[..., None], (0, 0), sigma) / den[..., None]
    spread = np.sqrt(np.maximum(sq - b_old * b_old, 0).mean(axis=-1))      # local std of the old surface
    trust = np.clip((40.0 - spread) / 20.0, 0, 1)[..., None]               # uniform below ~20, patterned above ~40
    a = alpha[..., None]
    decontaminated = room + a * (new - b_old)
    blended = room * (1 - a) + new * a
    edge = np.where(a > 0.02, trust * decontaminated + (1 - trust) * blended, room)
    interior = np.clip((alpha - 0.9) / 0.08, 0, 1)[..., None]
    return edge * (1 - interior) + new * interior
