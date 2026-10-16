"""Pre-flight photo checks: sharpness, exposure and camera angle, with user-facing advice.

Runs once when a room photo is opened, before any clicks. Every check is a cheap image
statistic (no model), so the advice appears within a few hundred milliseconds.

Camera angle comes from vanishing lines only. Depth Anything was tried as a fallback, but
its disparity has an unknown shift, so the horizon it implies is unreliable (it read a
level synthetic room as pitched 56 degrees), and a false "steep angle" warning is worse
than none.
"""
import cv2
import numpy as np

from app.services.render_engine import FOCAL_FACTOR

BLUR_MIN_LAPLACIAN_VAR = 80.0    # variance of the Laplacian (at ANALYSIS_SIDE) below this = blurry
ANALYSIS_SIDE = 1024             # metrics are computed at a fixed long side, so thresholds don't drift with upload size
DARK_MAX_MEAN_LUM = 45.0         # mean luminance (0-255) below this = too dark
CLIP_LEVEL = 250                 # a pixel is clipped when its brightest channel reaches this
OVEREXPOSED_MIN_CLIPPED = 0.15   # share of clipped pixels above this = overexposed
STEEP_MIN_PITCH_DEG = 30.0       # camera pitch (up or down) beyond this = steep angle
STEEP_MIN_ROLL_DEG = 5.0         # camera roll beyond this = tilted
MIN_VERTICAL_EVIDENCE = 0.6      # total length of vertical lines (in image heights) needed to trust them

MESSAGES = {
    "BLURRY": ("warning", "⚠️ Photo looks slightly blurry. Fine tile textures might appear soft. "
                          "For best realism, hold your phone steady or tap to focus."),
    "TOO_DARK": ("info", "💡 Room lighting is quite dim. New tiles will render realistically darker to match, "
                         "but details might be subtle."),
    "OVEREXPOSED": ("info", "☀️ Bright glare/overexposure detected. Highlights will be compressed to preserve "
                            "tile details."),
    "STEEP_ANGLE": ("warning", "📐 Steep camera angle detected. For perfect perspective and straight grout lines, "
                               "try capturing at chest height facing the wall or floor directly."),
    "GOOD": ("success", "✅ Excellent lighting and angle for photorealistic visualization!"),
}


def _analysis_gray(image_np):
    """Grey image at ANALYSIS_SIDE with its 1st-99th percentile range stretched to 0-255.

    The stretch makes the checks exposure-independent: Laplacian variance scales with
    contrast squared (a sharp but dim photo would read as blurry) and fixed Canny
    thresholds find no lines at all in a dark photo.
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    k = ANALYSIS_SIDE / max(gray.shape)
    if k < 1:
        gray = cv2.resize(gray, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    p1, p99 = np.percentile(gray, [1, 99])
    stretched = (gray.astype(np.float32) - float(p1)) * (255.0 / max(float(p99 - p1), 16.0))
    return np.clip(stretched, 0, 255).astype(np.uint8)


def sharpness(gray):
    """Variance of the Laplacian: high for crisp edges and texture, low for a soft photo."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def exposure(image_np):
    lum = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
    p1, p99 = np.percentile(lum, [1, 99])
    return {
        "mean_luminance": round(float(lum.mean()), 1),
        "clipped_share": round(float((image_np.max(axis=-1) >= CLIP_LEVEL).mean()), 4),
        "shadow_share": round(float((lum < 25).mean()), 4),
        "dynamic_range": round(float(p99 - p1), 1),
    }


def _segments(gray):
    """(segments (N,4), tilt from vertical in degrees) of the photo's long straight lines."""
    h, w = gray.shape
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    min_len = 0.08 * h
    segs = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=int(min_len * 0.5),
                           minLineLength=min_len, maxLineGap=0.01 * max(h, w))
    if segs is None:
        return np.zeros((0, 4)), np.zeros(0)
    segs = segs.reshape(-1, 4).astype(np.float64)   # (N,1,4) in OpenCV 4, (N,4) in 5
    dx, dy = segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]
    return segs, np.degrees(np.arctan2(np.abs(dx), np.abs(dy)))


def _vp_residual_deg(segs, vp):
    """Angle (deg) between each segment and the direction from its midpoint to the point vp (homogeneous)."""
    mid = 0.5 * (segs[:, :2] + segs[:, 2:])
    seg_dir = segs[:, 2:] - segs[:, :2]
    to_vp = vp[:2][None, :] - mid * vp[2]          # finite or at infinity alike
    cos = np.abs((seg_dir * to_vp).sum(1)) / (np.linalg.norm(seg_dir, axis=1) * np.linalg.norm(to_vp, axis=1) + 1e-12)
    return np.degrees(np.arccos(np.clip(cos, 0, 1)))


def _ransac_vp(segs, lengths, min_support, inlier_deg=1.5, finite=False):
    """Dominant vanishing point (homogeneous 3-vector) of a set of segments, or None.

    RANSAC over segment pairs picks the point most segments (by length) agree on, so
    slanted clutter (chair backs and legs) can't drag it. It is refined by least squares
    over its inliers: the point closest to all their lines (smallest singular vector).
    finite: reject points at infinity (parallel lines carry no horizon information).
    """
    if len(segs) < 3:
        return None
    p1 = np.column_stack([segs[:, :2], np.ones(len(segs))])
    p2 = np.column_stack([segs[:, 2:], np.ones(len(segs))])
    lines = np.cross(p1, p2)
    lines /= np.linalg.norm(lines[:, :2], axis=1, keepdims=True) + 1e-12
    rng = np.random.default_rng(0)
    prob = lengths / lengths.sum()
    best_vp, best_support = None, 0.0
    for _ in range(300):
        i, j = rng.choice(len(segs), 2, replace=False, p=prob)
        vp = np.cross(lines[i], lines[j])
        if not np.isfinite(vp).all() or np.abs(vp).max() < 1e-12:
            continue
        support = lengths[_vp_residual_deg(segs, vp) < inlier_deg].sum()
        if support > best_support:
            best_vp, best_support = vp, support
    if best_vp is None or best_support < min_support or best_support < 0.4 * lengths.sum():
        return None
    inl = _vp_residual_deg(segs, best_vp) < inlier_deg
    if inl.sum() < 2:
        return None
    _, _, vt = np.linalg.svd(lines[inl] * lengths[inl, None])
    vp = vt[-1]
    if finite and abs(vp[2]) < 1e-9 * np.abs(vp[:2]).max():
        return None
    return vp


def angle_from_verticals(gray, segs, tilt):
    """(pitch_deg, roll_deg) from the vertical vanishing point, or None without enough lines.

    Verticals (door frames, wall corners, furniture sides) meet at the vertical vanishing
    point. Back-projected through the camera, it is the direction of gravity, whose tilt
    away from the image's y axis gives the camera pitch (positive = looking down) and roll.
    """
    h, w = gray.shape
    segs = segs[tilt < 25.0]
    lengths = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    if len(segs) < 4 or lengths.sum() < MIN_VERTICAL_EVIDENCE * h:
        return None
    vp = _ransac_vp(segs, lengths, 0.5 * MIN_VERTICAL_EVIDENCE * h)
    if vp is None:
        return None
    f, cx, cy = FOCAL_FACTOR * max(h, w), w / 2.0, h / 2.0
    ray = np.array([(vp[0] - cx * vp[2]) / f, (vp[1] - cy * vp[2]) / f, vp[2]])
    ray /= np.linalg.norm(ray)
    if ray[1] < 0:
        ray = -ray                                   # point "down" the image
    pitch = float(np.degrees(np.arctan2(ray[2], np.hypot(ray[0], ray[1]))))
    roll = float(np.degrees(np.arctan2(ray[0], ray[1])))
    return pitch, roll


def pitch_from_horizon(gray, segs, tilt):
    """Camera pitch (deg, positive = looking down) from the horizon, or None.

    For photos without usable verticals (a bare floor, a raw concrete room). Floor-wall
    junctions, skirting and floor joints are horizontal in 3D, so where they converge lies
    on the horizon; its height above the image centre gives the pitch. Near-horizontal
    image lines meet at infinity and are left out; their horizon could be anywhere.
    """
    h, w = gray.shape
    sel = (tilt > 30.0) & (tilt < 86.0)
    segs = segs[sel]
    lengths = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    if len(segs) < 3 or lengths.sum() < MIN_VERTICAL_EVIDENCE * h:
        return None
    vp = _ransac_vp(segs, lengths, 0.5 * MIN_VERTICAL_EVIDENCE * h, finite=True)
    if vp is None:
        return None
    f, cy = FOCAL_FACTOR * max(h, w), h / 2.0
    y_vp = vp[1] / vp[2]
    if abs(y_vp - cy) > 4 * f:      # nearly parallel: no usable horizon
        return None
    return float(np.degrees(np.arctan2(cy - y_vp, f)))


def assess_quality(image_np):
    """{flags, messages, metrics} for a room photo (RGB uint8, already downscaled)."""
    gray = _analysis_gray(image_np)
    metrics = {"sharpness": round(sharpness(gray), 1), **exposure(image_np)}

    angle, segs, tilt = None, np.zeros((0, 4)), np.zeros(0)
    # A mostly blown-out photo has only stray edges left; any angle read from them is noise.
    if metrics["clipped_share"] < 0.6:
        segs, tilt = _segments(gray)
        angle = angle_from_verticals(gray, segs, tilt)
    if angle is not None:
        metrics.update(pitch_deg=round(angle[0], 1), roll_deg=round(angle[1], 1), angle_source="vertical lines")
    else:
        pitch = pitch_from_horizon(gray, segs, tilt)
        metrics.update(pitch_deg=None if pitch is None else round(pitch, 1), roll_deg=None,
                       angle_source="horizon lines" if pitch is not None else "unknown")

    flags = []
    if metrics["sharpness"] < BLUR_MIN_LAPLACIAN_VAR:
        flags.append("BLURRY")
    if metrics["mean_luminance"] < DARK_MAX_MEAN_LUM:
        flags.append("TOO_DARK")
    if metrics["clipped_share"] > OVEREXPOSED_MIN_CLIPPED:
        flags.append("OVEREXPOSED")
    pitch, roll = metrics["pitch_deg"], metrics["roll_deg"]
    if (pitch is not None and abs(pitch) > STEEP_MIN_PITCH_DEG) or (roll is not None and abs(roll) > STEEP_MIN_ROLL_DEG):
        flags.append("STEEP_ANGLE")

    shown = flags or ["GOOD"]
    messages = [{"code": c, "level": MESSAGES[c][0], "text": MESSAGES[c][1]} for c in shown]
    return {"flags": flags, "messages": messages, "metrics": metrics}
