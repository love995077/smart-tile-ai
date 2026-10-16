"""Room palette extraction and colour-harmony ranking of the tile catalog ("✨ Best Match").

The room's dominant colours come from k-means in CIELAB (perceptually even, so cluster
distances mean something). Each catalog tile is profiled the same way once and cached.
A tile scores well when it:
  - echoes a colour the room already has (tone-on-tone affinity, by Delta E),
  - sits at a pleasing hue relation to the room's main accent colour (analogous or
    complementary), or is a neutral, which goes with any palette,
  - shares the room's warm or cool character.
"""
import os
import threading

import cv2
import numpy as np
from PIL import Image

from app.services.render_engine import prepare_tile_image

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
NEUTRAL_CHROMA = 10.0      # LAB chroma below this reads as a neutral (white / grey / greige)
WEIGHTS = (0.45, 0.35, 0.20)   # affinity, hue harmony, warmth

_profiles = {}
_lock = threading.Lock()


def to_lab(rgb):
    """RGB uint8 (..., 3) -> CIELAB float32 with L in 0-100 and a, b roughly -128..127."""
    arr = np.asarray(rgb, np.float32).reshape(-1, 1, 3) / 255.0
    return cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).reshape(np.shape(rgb)[:-1] + (3,))


def lab_to_hex(lab):
    rgb = cv2.cvtColor(np.float32(lab).reshape(1, 1, 3), cv2.COLOR_LAB2RGB).reshape(3)
    r, g, b = (np.clip(rgb, 0, 1) * 255).round().astype(int)
    return f"#{r:02x}{g:02x}{b:02x}"


def palette(rgb, k=6, side=160, min_share=0.03):
    """Dominant colours as [(lab (3,), share)], most common first."""
    h, w = rgb.shape[:2]
    s = side / max(h, w)
    small = cv2.resize(rgb, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    samples = to_lab(small).reshape(-1, 3).astype(np.float32)
    k = int(min(k, len(samples)))
    cv2.setRNGSeed(0)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2)
    _, labels, centers = cv2.kmeans(samples, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    shares = np.bincount(labels.ravel(), minlength=k) / len(labels)
    order = np.argsort(-shares)
    out = [(centers[i], float(shares[i])) for i in order if shares[i] >= min_share]
    total = sum(sh for _, sh in out) or 1.0
    return [(c, sh / total) for c, sh in out]


def color_name(lab):
    L, a, b = (float(x) for x in lab)
    C = np.hypot(a, b)
    hue = np.degrees(np.arctan2(b, a)) % 360
    if C < NEUTRAL_CHROMA:
        warm = b > 4
        if L > 88:
            return "warm white" if warm else "white"
        if L > 72:
            return "greige" if warm else "light grey"
        if L > 50:
            return "taupe" if warm else "grey"
        return "charcoal" if L < 32 else "dark grey"
    if hue < 25 or hue >= 340:
        return "blush" if L > 70 else "red"
    if hue < 55:
        if C < 28:
            return "beige" if L > 68 else ("tan" if L > 45 else "brown")
        return "peach" if L > 70 else ("terracotta" if L > 42 else "rust")
    if hue < 75:
        return ("beige" if L > 70 else "tan" if L > 48 else "brown") if C < 30 else ("apricot" if L > 70 else "amber")
    if hue < 105:
        return ("sand" if L > 70 else "khaki" if L > 45 else "olive brown") if C < 30 else ("yellow" if L > 70 else "mustard")
    if hue < 170:
        return "sage" if C < 22 and L > 45 else ("forest green" if L < 40 else "green")
    if hue < 230:
        return "aqua" if L > 65 else "teal"
    if hue < 300:
        return "navy" if L < 35 else ("slate blue" if C < 22 else "blue")
    return "mauve" if L > 55 else "plum"


def _profile(path):
    """Cached (dominant lab, palette) for a catalog image, keyed on path + mtime + size."""
    st = os.stat(path)
    key = (path, st.st_mtime_ns, st.st_size)
    with _lock:
        if key in _profiles:
            return _profiles[key]
    img = Image.open(path).convert("RGB")
    img.thumbnail((256, 256))
    rgb = prepare_tile_image(np.array(img))
    pal = palette(rgb, k=3, side=96)
    with _lock:
        _profiles[key] = pal
    return pal


def _hue_harmony(tile_lab, room_pal):
    """1 = analogous or complementary to the room's main accent colour, 0.75 = neutral tile."""
    tL, ta, tb = tile_lab
    tile_c = np.hypot(ta, tb)
    accents = [(c, s) for c, s in room_pal if np.hypot(c[1], c[2]) >= NEUTRAL_CHROMA]
    if tile_c < NEUTRAL_CHROMA:
        return 0.75, "neutral"
    if not accents:
        # Neutral room: a gentle colour is fine, a loud one less so.
        return float(np.clip(1.0 - (tile_c - NEUTRAL_CHROMA) / 60.0, 0.35, 0.8)), "accent"
    best, kind = 0.0, "analogous"
    tile_h = np.degrees(np.arctan2(tb, ta))
    for c, s in accents:
        dh = abs((np.degrees(np.arctan2(c[2], c[1])) - tile_h + 180) % 360 - 180)
        analog = np.exp(-(dh / 30.0) ** 2)
        compl = 0.85 * np.exp(-((dh - 180) / 30.0) ** 2)
        score = max(analog, compl) * (0.6 + 0.4 * min(1.0, s / 0.25))
        if score > best:
            best, kind = score, "analogous" if analog >= compl else "complementary"
    return float(best), kind


def harmony_score(tile_pal, room_pal):
    """(score 0-1, reason) for one tile against the room palette."""
    tile_lab = tile_pal[0][0]
    # Affinity: how closely the tile's main colours echo colours present in the room.
    affinity, echo = 0.0, None
    for tc, ts in tile_pal:
        for rc, rs in room_pal:
            de = float(np.linalg.norm(tc - rc))
            val = ts * np.exp(-de / 22.0) * min(1.0, rs / 0.15)
            if val > affinity:
                affinity, echo = val, rc
    harmony, kind = _hue_harmony(tile_lab, room_pal)
    room_warm = sum(s * (c[2] + 0.5 * c[1]) for c, s in room_pal)
    tile_warm = tile_lab[2] + 0.5 * tile_lab[1]
    warmth = 1.0 - abs(np.tanh(room_warm / 15.0) - np.tanh(tile_warm / 15.0)) / 2.0
    wa, wh, ww = WEIGHTS
    score = wa * min(1.0, affinity * 1.4) + wh * harmony + ww * warmth

    parts = {"affinity": wa * min(1.0, affinity * 1.4), "harmony": wh * harmony, "warmth": ww * warmth}
    top = max(parts, key=parts.get)
    if top == "affinity" and echo is not None:
        reason = f"Echoes the room's {color_name(echo)} tones"
    elif top == "harmony" and kind == "neutral":
        reason = f"{color_name(tile_lab).capitalize()} neutral that suits this palette"
    elif top == "harmony":
        accent = max((c for c, s in room_pal if np.hypot(c[1], c[2]) >= NEUTRAL_CHROMA),
                     key=lambda c: np.hypot(c[1], c[2]), default=None)
        name = color_name(accent) if accent is not None else "room"
        reason = f"{'Complements' if kind == 'complementary' else 'Harmonises with'} the {name} accents"
    else:
        reason = f"Matches the room's {'warm' if room_warm > 0 else 'cool'} character"
    return float(score), reason


def catalog_files(folder):
    for root, _, files in os.walk(folder):
        for filename in sorted(files):
            if filename.lower().endswith(IMAGE_EXTS):
                full = os.path.join(root, filename)
                yield full, os.path.relpath(full, folder).replace("\\", "/")


def recommend(room_rgb, folder, top_n=3):
    """Room palette and the top_n most harmonious catalog tiles."""
    room_pal = palette(room_rgb)
    ranked = []
    for full, rel in catalog_files(folder):
        try:
            tile_pal = _profile(full)
        except Exception as e:     # unreadable or truncated file: skip, don't fail the analysis
            print(f"color_harmony: skipping {rel}: {e}")
            continue
        if not tile_pal:
            continue
        score, reason = harmony_score(tile_pal, room_pal)
        ranked.append({"path": rel, "score": round(score * 100, 1), "reason": reason,
                       "color": lab_to_hex(tile_pal[0][0])})
    ranked.sort(key=lambda r: -r["score"])
    return {
        "palette": [{"hex": lab_to_hex(c), "share": round(s, 3), "name": color_name(c)} for c, s in room_pal],
        "best_matches": ranked[:top_n],
    }


def warm_catalog(folder):
    """Profiles every catalog tile up front (called in a background thread at startup)."""
    for full, rel in catalog_files(folder):
        try:
            _profile(full)
        except Exception as e:
            print(f"color_harmony: skipping {rel}: {e}")
