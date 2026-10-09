"""Build Riccardo's full 3D head from a single frontal photo.

A single photo only shows the face, so the rest of the head is reconstructed:

  1. MediaPipe FaceLandmarker -> 468 face landmarks with relative depth.
  2. MediaPipe multiclass segmenter -> hair / skin / accessories masks.
  3. A signed-distance head (skull + jaw + neck + ears) proportioned from the
     landmarks, sampled as a radial function r(theta, phi) around the skull center.
  4. The real face surface is rasterized into the same radial grid and fused in;
     the difference is diffused outward so skull and face join seamlessly.
  5. Texture: the photo is projected wherever it really shows the head (face, beard,
     front hair) and blended into synthesized hair / skin / beard elsewhere, with
     colors sampled from the photo.

The grid is denser toward the face (warped angles), and UVs follow that warp so the
texture spends its pixels where it matters.

Output (public/face/):
  face.bin   Float32 positions | Float32 uvs | Float32 normals | Uint16 indices
  face.json  counts, byte offsets, chin/neck heights
  face.jpg   texture

Usage: python tools/build_head.py [--photo assets/photos/reference/front.jpg] [--debug]
"""

import argparse
import json
from pathlib import Path

import mediapipe as mp
import numpy as np
import scipy.ndimage as ndi
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    ImageSegmenter,
    ImageSegmenterOptions,
)
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "tools" / "models"
OUT = ROOT / "public" / "face"

N_PHI, N_THETA = 128, 256  # grid rows (top->bottom), columns (around)
WARP_THETA, WARP_PHI = 0.4, 0.35  # >0 concentrates vertices on the face
TEX_W, TEX_H = 2048, 1024

# Segmenter classes
BG, HAIR, BODY_SKIN, FACE_SKIN, CLOTHES, ACCESSORY = range(6)

# MediaPipe face mesh landmark ids
FOREHEAD, CHIN, NOSE_TIP, NOSE_BOTTOM, BROW_MID = 10, 152, 1, 2, 168
CHEEK_R, CHEEK_L, LIP_TOP, BROW_R = 234, 454, 13, 105
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
             152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
SKIN_SPOTS = [50, 280, 101, 330, 108, 337]
# Natural skin tone the front photo is pulled toward (its light is cold and hazy).
TARGET_SKIN = np.array([196.0, 150.0, 123.0]) / 255.0
LUMA = np.array([0.299, 0.587, 0.114])
BEARD_SPOTS = [152, 175, 199, 377, 148, 400, 176]


# --------------------------------------------------------------------------- detection

def detect_landmarks(photo):
    opts = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODELS / "face_landmarker.task")), num_faces=1)
    with FaceLandmarker.create_from_options(opts) as lm:
        res = lm.detect(mp.Image.create_from_file(str(photo)))
    if not res.face_landmarks:
        raise SystemExit(f"No face found in {photo}")
    return np.array([[p.x, p.y, p.z] for p in res.face_landmarks[0][:468]])


def segment(photo):
    opts = ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=str(MODELS / "selfie_multiclass.tflite")),
        output_category_mask=True)
    with ImageSegmenter.create_from_options(opts) as seg:
        # Copy before the segmenter closes: numpy_view() aliases memory it frees.
        mask = seg.segment(mp.Image.create_from_file(str(photo))).category_mask.numpy_view().copy()
    return mask.reshape(mask.shape[0], mask.shape[1])


def load_triangles():
    tris = []
    for line in (MODELS / "canonical_face_model.obj").read_text().splitlines():
        if line.startswith("f "):
            tris.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
    return np.array(tris)


def subdivide(pos, tris):
    edge_mid, pos = {}, list(pos)

    def mid(a, b):
        key = (min(a, b), max(a, b))
        if key not in edge_mid:
            edge_mid[key] = len(pos)
            pos.append((pos[a] + pos[b]) / 2)
        return edge_mid[key]

    out = []
    for a, b, c in tris:
        ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
        out += [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
    return np.array(pos), np.array(out)


# --------------------------------------------------------------------------- grid / warp

def warp_theta(s):  # s in [-1, 1] -> theta in [-pi, pi], dense near 0 (front)
    return np.pi * (s - WARP_THETA * np.sin(np.pi * s) / np.pi)


def warp_phi(t):  # t in [0, 1] -> phi in [0, pi], dense near the equator
    return np.pi * (t + WARP_PHI * np.sin(2 * np.pi * t) / (2 * np.pi))


def unwarp(values, fn, lo, hi):
    table = np.linspace(lo, hi, 4097)
    return np.interp(values, fn(table), table)


def directions(theta, phi):
    return np.stack([np.sin(phi) * np.sin(theta), np.cos(phi), np.sin(phi) * np.cos(theta)], axis=-1)


# --------------------------------------------------------------------------- head SDF

def sd_ellipsoid(q, r):
    k0 = np.linalg.norm(q / r, axis=-1)
    k1 = np.linalg.norm(q / (r * r), axis=-1)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)


def sd_capped_cylinder_y(q, radius, y0, y1):
    d_r = np.linalg.norm(q[..., [0, 2]], axis=-1) - radius
    d_y = np.maximum(y0 - q[..., 1], q[..., 1] - y1)
    outside = np.linalg.norm(np.stack([np.maximum(d_r, 0), np.maximum(d_y, 0)], -1), axis=-1)
    return np.minimum(np.maximum(d_r, d_y), 0.0) + outside


def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + (a - b) * h - k * h * (1.0 - h)


class HeadShape:
    """Head proportions derived from landmarks, in head space (origin = skull center)."""

    def __init__(self, P, face_h, face_w):
        self.face_h, self.face_w = face_h, face_w
        self.a = face_w * 0.60          # half width (with hair over the temples)
        self.c = self.a * 1.24          # half length front-to-back
        self.center = np.array([
            (P[CHEEK_R, 0] + P[CHEEK_L, 0]) / 2,
            P[BROW_MID, 1] + face_h * 0.04,
            P[FOREHEAD, 2] - self.c * 0.95,
        ])
        C = self.center
        self.b_top = (P[FOREHEAD, 1] + face_h * 0.36) - C[1]   # crown incl. hair volume
        self.b_back = self.c * 0.86
        self.chin_y = P[CHIN, 1] - C[1]
        self.jaw_c = np.array([0.0, self.chin_y + face_h * 0.33, P[CHIN, 2] - C[2] - face_w * 0.42])
        self.jaw_r = np.array([face_w * 0.47, face_h * 0.36, face_w * 0.46])
        self.neck_r = face_w * 0.40
        self.neck_z = -self.c * 0.18
        self.neck_bottom = self.chin_y - face_h * 0.85
        ear_y = (P[BROW_MID, 1] + P[NOSE_BOTTOM, 1]) / 2 - C[1]
        ear_z = (P[CHEEK_R, 2] - C[2]) - face_w * 0.30
        self.ears = [np.array([s * self.a * 0.93, ear_y, ear_z]) for s in (-1, 1)]
        self.ear_r = np.array([face_w * 0.07, face_h * 0.2, face_w * 0.12])

    def sdf(self, q):
        radii = np.where(q[..., 1:2] > 0, [self.a, self.b_top, self.c], [self.a, self.b_back, self.c * 0.97])
        d = sd_ellipsoid(q, radii)
        d = smin(d, sd_ellipsoid(q - self.jaw_c, self.jaw_r), self.face_w * 0.12)
        qn = q - np.array([0.0, 0.0, self.neck_z])
        d = smin(d, sd_capped_cylinder_y(qn, self.neck_r, self.neck_bottom, 0.0), self.face_w * 0.18)
        for e in self.ears:
            d = smin(d, sd_ellipsoid(q - e, self.ear_r), self.face_w * 0.03)
        return d

    def radial(self, dirs, r_max):
        """Outermost surface distance along each direction (handles non-convex unions)."""
        steps = 160
        rs = np.linspace(r_max, 0.0, steps)
        out = np.zeros(dirs.shape[:-1])
        for i in range(dirs.shape[0]):  # row by row to bound memory
            d = dirs[i][:, None, :] * rs[None, :, None]
            inside = self.sdf(d) < 0
            first = np.argmax(inside, axis=1)  # first inside sample marching inward
            lo, hi = rs[first], rs[np.maximum(first - 1, 0)]
            for _ in range(20):
                m = (lo + hi) / 2
                ins = self.sdf(dirs[i] * m[:, None]) < 0
                lo, hi = np.where(ins, m, lo), np.where(ins, hi, m)
            out[i] = (lo + hi) / 2
        return out


# --------------------------------------------------------------------------- face fusion

def rasterize_face(face_pos, face_tris, center):
    """Rasterize the face mesh into the warped radial grid. Returns (r, mask)."""
    q = face_pos - center
    r = np.linalg.norm(q, axis=1)
    theta = np.arctan2(q[:, 0], q[:, 2])
    phi = np.arccos(np.clip(q[:, 1] / r, -1, 1))
    gj = (unwarp(theta, warp_theta, -1, 1) + 1) / 2 * N_THETA  # column coordinate
    gi = unwarp(phi, warp_phi, 0, 1) * N_PHI                   # row coordinate

    r_grid = np.full((N_PHI + 1, N_THETA + 1), -np.inf)
    for a, b, c in face_tris:
        xs, ys = gj[[a, b, c]], gi[[a, b, c]]
        j0, j1 = int(np.floor(xs.min())), int(np.ceil(xs.max()))
        i0, i1 = int(np.floor(ys.min())), int(np.ceil(ys.max()))
        jj, ii = np.meshgrid(np.arange(j0, j1 + 1), np.arange(i0, i1 + 1))
        den = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
        if abs(den) < 1e-12:
            continue
        w0 = ((ys[1] - ys[2]) * (jj - xs[2]) + (xs[2] - xs[1]) * (ii - ys[2])) / den
        w1 = ((ys[2] - ys[0]) * (jj - xs[2]) + (xs[0] - xs[2]) * (ii - ys[2])) / den
        w2 = 1 - w0 - w1
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not inside.any():
            continue
        val = w0 * r[a] + w1 * r[b] + w2 * r[c]
        ii, jj, val = ii[inside], jj[inside], val[inside]
        np.maximum.at(r_grid, (ii, jj), val)
    mask = np.isfinite(r_grid)
    # Fill pinholes inside the face region.
    filled = ndi.binary_closing(mask, iterations=2)
    holes = filled & ~mask
    if holes.any():
        r_tmp = np.where(mask, r_grid, 0)
        num = ndi.uniform_filter(r_tmp, 3)
        den = ndi.uniform_filter(mask.astype(float), 3)
        r_grid[holes] = num[holes] / np.maximum(den[holes], 1e-6)
    return np.where(filled, r_grid, 0.0), filled


def fuse(r_base, r_face, mask):
    """Replace the skull with the face where available and diffuse the offset outward."""
    D = np.where(mask, r_face - r_base, 0.0)
    for _ in range(400):
        avg = (np.roll(D, 1, 1) + np.roll(D, -1, 1) + np.vstack([D[:1], D[:-1]]) + np.vstack([D[1:], D[-1:]])) / 4
        D = np.where(mask, D, avg)
    dist_out = ndi.distance_transform_edt(~mask)
    dist_in = ndi.distance_transform_edt(mask)
    r = r_base + D * np.exp(-((dist_out / 14.0) ** 2))
    # Soften the seam band on both sides of the face border.
    band = np.exp(-((np.where(mask, dist_in, dist_out) / 3.0) ** 2))
    smooth = ndi.gaussian_filter(r, 1.5, mode=["nearest", "wrap"])
    return r * (1 - band) + smooth * band


# --------------------------------------------------------------------------- texture

def fractal_noise(shape, scales, rng):
    """Anisotropic value noise: scales = list of (rows, cols) base resolutions."""
    out = np.zeros(shape)
    amp = 1.0
    for sy, sx in scales:
        small = rng.random((sy, sx))
        out += amp * ndi.zoom(small, (shape[0] / sy, shape[1] / sx), order=3, mode="grid-wrap")[: shape[0], : shape[1]]
        amp *= 0.5
    out -= out.mean()
    return out / (np.abs(out).max() + 1e-9)


def median_color(img, points, radius):
    samples = []
    for x, y in points:
        x, y = int(x), int(y)
        patch = img[max(0, y - radius): y + radius, max(0, x - radius): x + radius].reshape(-1, 3)
        samples.append(patch)
    return np.median(np.concatenate(samples), axis=0)


def hairline_y(abs_theta, P, fh):
    """Height of the hairline going around the head: forehead, above the ears, nape."""
    y_front = P[FOREHEAD, 1] + fh * 0.03
    y_side = P[BROW_R, 1] + fh * 0.02
    y_nape = P[NOSE_BOTTOM, 1] - fh * 0.12
    return np.where(
        abs_theta < np.pi / 2,
        y_front + (y_side - y_front) * smoothstep(0.15 * np.pi, 0.5 * np.pi, abs_theta),
        y_side + (y_nape - y_side) * smoothstep(0.5 * np.pi, 0.85 * np.pi, abs_theta),
    )


def add_hair_volume(r_grid, center, face_mask, P, fh, rng):
    """Thicken the scalp where hair grows, with tufts and a fuller fringe on top."""
    S, T = np.meshgrid(np.linspace(-1, 1, N_THETA + 1), np.linspace(0, 1, N_PHI + 1))
    theta, phi = warp_theta(S), warp_phi(T)
    y = center[1] + np.cos(phi) * r_grid
    line = hairline_y(np.abs(theta), P, fh)
    w = smoothstep(line - fh * 0.02, line + fh * 0.1, y)
    w *= 1 - ndi.gaussian_filter(face_mask.astype(float), 3)
    tufts = fractal_noise(r_grid.shape, [(6, 24), (12, 48), (24, 96)], rng)
    tufts[:, -1] = tufts[:, 0]
    crown = smoothstep(P[FOREHEAD, 1], P[FOREHEAD, 1] + fh * 0.4, y)
    thickness = fh * (0.05 + 0.06 * crown + 0.035 * tufts)
    return r_grid + w * np.maximum(thickness, 0)


def white_balance(img_f, seg, strength=0.7):
    """Per-channel gains that move the face-skin median toward TARGET_SKIN, luminance preserved."""
    skin = np.median(img_f[seg == FACE_SKIN], axis=0)
    gains = TARGET_SKIN / skin
    gains *= (skin @ LUMA) / ((skin * gains) @ LUMA)
    gains = 1 + (gains - 1) * strength
    return np.clip(img_f * gains, 0, 1)


def reference_hair_color(photo):
    """Hair hue from a photo where the hair is fully visible and warmly lit."""
    img_f = np.asarray(Image.open(photo).convert("RGB"), dtype=np.float64) / 255.0
    seg = segment(photo)
    return np.median(img_f[seg == HAIR], axis=0)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def bake_texture(img_f, seg, shape, P, r_grid, normals_grid, face_mask, hair_ref, rng):
    H_img, W_img = seg.shape
    C, fh = shape.center, shape.face_h

    # Texel -> warped grid coordinates -> surface point.
    s = (np.arange(TEX_W) + 0.5) / TEX_W * 2 - 1
    t = (np.arange(TEX_H) + 0.5) / TEX_H
    S, T = np.meshgrid(s, t)
    theta, phi = warp_theta(S), warp_phi(T)
    gi, gj = T * N_PHI, (S + 1) / 2 * N_THETA
    r = ndi.map_coordinates(r_grid, [gi, gj], order=1, mode="nearest")
    pos = C + directions(theta, phi) * r[..., None]
    nrm = np.stack([ndi.map_coordinates(normals_grid[..., k], [gi, gj], order=1, mode="nearest") for k in range(3)], -1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-9
    in_face = ndi.map_coordinates(face_mask.astype(float), [gi, gj], order=1) > 0.5

    # ---- photo projection (orthographic: world x/y are photo pixels, y flipped)
    px, py = pos[..., 0], -pos[..., 1]
    # z-buffer of the head as seen by the photo camera
    zbuf = np.full((H_img, W_img), -np.inf)
    gpos = C + directions(*np.meshgrid(warp_theta(np.linspace(-1, 1, N_THETA * 4)),
                                         warp_phi(np.linspace(0, 1, N_PHI * 4)))) * \
        ndi.zoom(r_grid, 4, order=1)[: N_PHI * 4, : N_THETA * 4, None]
    gx = np.clip(gpos[..., 0].astype(int), 0, W_img - 1)
    gy = np.clip((-gpos[..., 1]).astype(int), 0, H_img - 1)
    np.maximum.at(zbuf, (gy, gx), gpos[..., 2])
    zbuf = ndi.maximum_filter(zbuf, 5)
    ix = np.clip(px.astype(int), 0, W_img - 1)
    iy = np.clip(py.astype(int), 0, H_img - 1)
    # The fused face region faces the camera by construction; only test the rest.
    visible = (pos[..., 2] >= zbuf[iy, ix] - shape.face_w * 0.03) | in_face

    # Which photo pixels may be used: hair, face skin, glasses inside the face oval, neck below chin.
    oval = Image.new("L", (W_img, H_img), 0)
    ImageDraw.Draw(oval).polygon([(P[i, 0], -P[i, 1]) for i in FACE_OVAL], fill=255)
    oval = np.asarray(oval) > 0
    chin_px = -P[CHIN, 1]
    rows = np.arange(H_img)[:, None]
    allowed = (seg == HAIR) | (seg == FACE_SKIN) | (oval & (seg != BG) & (seg != BODY_SKIN)) | \
              ((seg == BODY_SKIN) & (rows > chin_px) & (rows < chin_px + fh * 0.5))
    allowed = ndi.binary_erosion(allowed, iterations=4)
    allowed = ndi.gaussian_filter(allowed.astype(float), 4)

    photo = np.stack([ndi.map_coordinates(img_f[..., k], [py, px], order=1, mode="nearest") for k in range(3)], -1)
    allowed_tex = ndi.map_coordinates(allowed, [py, px], order=1)
    w_photo = allowed_tex * smoothstep(0.3, 0.6, nrm[..., 2]) * visible
    # Inside the fused face the projection is exact, whatever the surface angle.
    face_soft = ndi.gaussian_filter(in_face.astype(float), 6)
    w_photo = np.maximum(w_photo, face_soft * allowed_tex)

    # ---- synthesized hair / skin / beard
    photo_hair = np.median(img_f[(seg == HAIR) & ~oval], axis=0)
    # Hue from the reference photo, softened; brightness from the darker of the two
    # (warm and cold light both wash dark brown hair out).
    hue = hair_ref / (hair_ref @ LUMA)
    hue = 1 + (hue - 1) * 0.7
    hair_lum = 0.8 * min(photo_hair @ LUMA, hair_ref @ LUMA)
    hair_col = hue * hair_lum
    skin_col = median_color(img_f, [(P[i, 0], -P[i, 1]) for i in SKIN_SPOTS], 6)
    beard_raw = median_color(img_f, [(P[i, 0], -P[i, 1]) for i in BEARD_SPOTS], 6)
    beard_col = (beard_raw + hue * (beard_raw @ LUMA)) / 2

    # Recolor projected hair to the target hue, keeping per-pixel luminance (detail).
    hair_px = ndi.gaussian_filter((seg == HAIR).astype(float), 3)
    w_hair_px = ndi.map_coordinates(hair_px, [py, px], order=1)[..., None]
    rel = (photo @ LUMA)[..., None] / (photo_hair @ LUMA)
    photo = photo * (1 - w_hair_px) + hair_col * rel * w_hair_px
    print("colors hair/skin/beard:", *(np.round(c * 255).astype(int) for c in (hair_col, skin_col, beard_col)))

    y = pos[..., 1]
    ath = np.abs(theta)
    hairline = hairline_y(ath, P, fh)
    w_hair = smoothstep(hairline - fh * 0.03, hairline + fh * 0.03, y)

    mouth_y = P[LIP_TOP, 1]
    beard_top = mouth_y + (hairline - mouth_y) * smoothstep(0.22 * np.pi, 0.42 * np.pi, ath)
    w_beard = smoothstep(beard_top + fh * 0.04, beard_top - fh * 0.02, y) * \
        smoothstep(P[CHIN, 1] - fh * 0.45, P[CHIN, 1] - fh * 0.25, y) * \
        smoothstep(0.62 * np.pi, 0.5 * np.pi, ath)

    strands = fractal_noise((TEX_H, TEX_W), [(16, 96), (48, 384), (128, 1024)], rng)
    grain = fractal_noise((TEX_H, TEX_W), [(256, 512), (512, 1024)], rng)
    hair = hair_col[None, None] * (1 + 0.35 * strands[..., None])
    skin = skin_col[None, None] * (1 + 0.04 * grain[..., None])
    beard = beard_col[None, None] * (1 + 0.18 * grain[..., None])
    synth = skin * (1 - w_beard[..., None]) + beard * w_beard[..., None]
    synth = synth * (1 - w_hair[..., None]) + hair * w_hair[..., None]
    # Soft top-light so the synthesized parts are not flat next to the lit photo.
    synth *= (0.78 + 0.22 * np.clip(nrm[..., 1] * 0.5 + 0.5, 0, 1))[..., None]

    tex = synth * (1 - w_photo[..., None]) + photo * w_photo[..., None]
    tex = np.clip(tex, 0, 1)
    return Image.fromarray((tex * 255).astype(np.uint8)), w_photo


# --------------------------------------------------------------------------- mesh

def grid_mesh(center, r_grid):
    s = np.linspace(-1, 1, N_THETA + 1)
    t = np.linspace(0, 1, N_PHI + 1)
    S, T = np.meshgrid(s, t)
    pos = center + directions(warp_theta(S), warp_phi(T)) * r_grid[..., None]
    uv = np.stack([(S + 1) / 2, 1 - T], -1)

    # Normals with wrap-around in theta (column N_THETA duplicates column 0 for the UV seam).
    p = pos[:, :N_THETA]
    d_theta = np.roll(p, -1, 1) - np.roll(p, 1, 1)
    d_phi = np.vstack([p[1:2] - p[0:1], p[2:] - p[:-2], p[-1:] - p[-2:-1]])
    n = np.cross(d_phi, d_theta)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    # Poles: average the ring next to them.
    n[0] = n[1].mean(0) / np.linalg.norm(n[1].mean(0))
    n[-1] = n[-2].mean(0) / np.linalg.norm(n[-2].mean(0))
    normals = np.concatenate([n, n[:, :1]], axis=1)

    idx = np.arange((N_PHI + 1) * (N_THETA + 1)).reshape(N_PHI + 1, N_THETA + 1)
    a, b = idx[:-1, :-1], idx[:-1, 1:]
    c, d = idx[1:, :-1], idx[1:, 1:]
    tris = np.concatenate([np.stack([a, c, b], -1).reshape(-1, 3), np.stack([b, c, d], -1).reshape(-1, 3)])
    return pos, uv, normals, tris


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", default=str(ROOT / "assets/photos/reference/front.jpg"))
    ap.add_argument("--hair-photo", default=str(ROOT / "assets/photos/reference/three-quarter.jpg"))
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(7)

    photo = Path(args.photo)
    seg = segment(photo)
    img_f = white_balance(np.asarray(Image.open(photo).convert("RGB"), dtype=np.float64) / 255.0, seg)
    hair_ref = reference_hair_color(Path(args.hair_photo))
    # Work in segmentation-mask pixel units (same aspect as the photo).
    Hs, Ws = seg.shape
    lm = detect_landmarks(photo)
    P = np.stack([lm[:, 0] * Ws, -lm[:, 1] * Hs, -lm[:, 2] * Ws], axis=1)  # x right, y up, z to camera

    face_h = P[FOREHEAD, 1] - P[CHIN, 1]
    face_w = abs(P[CHEEK_L, 0] - P[CHEEK_R, 0])
    shape = HeadShape(P, face_h, face_w)
    C = shape.center

    # Base head as radial function on the warped grid.
    s = np.linspace(-1, 1, N_THETA + 1)
    t = np.linspace(0, 1, N_PHI + 1)
    S, T = np.meshgrid(s, t)
    dirs = directions(warp_theta(S), warp_phi(T))
    r_base = shape.radial(dirs, r_max=face_h * 3)

    # Real face, smoothed, fused into the skull.
    face_pos, face_tris = P.copy(), load_triangles()
    for _ in range(2):
        face_pos, face_tris = subdivide(face_pos, face_tris)
    r_face, face_mask = rasterize_face(face_pos, face_tris, C)
    r_grid = fuse(r_base, r_face, face_mask)
    r_grid = add_hair_volume(r_grid, C, face_mask, P, face_h, rng)
    r_grid[:, -1] = r_grid[:, 0]  # close the seam exactly

    pos, uv, normals, tris = grid_mesh(C, r_grid)
    tex, w_photo = bake_texture(img_f, seg, shape, P, r_grid, normals, face_mask, hair_ref, rng)

    # World space: origin at the skull center, face ~2 units tall, +Y up, +Z to camera.
    scale = 2.0 / face_h
    pos_w = (pos - C) * scale
    meta_extra = {
        "chinY": float(shape.chin_y * scale),
        "neckY": float(shape.neck_bottom * scale),
        "minY": float(pos_w[..., 1].min()),
        "maxY": float(pos_w[..., 1].max()),
    }

    pos_f, uv_f, n_f = pos_w.reshape(-1, 3), uv.reshape(-1, 2), normals.reshape(-1, 3)
    OUT.mkdir(parents=True, exist_ok=True)
    tex.save(OUT / "face.jpg", quality=86)
    blobs = [pos_f.astype("<f4"), uv_f.astype("<f4"), n_f.astype("<f4"), tris.astype("<u2")]
    meta, offset = {"vertexCount": len(pos_f), "indexCount": tris.size, **meta_extra}, 0
    with open(OUT / "face.bin", "wb") as fh:
        for name, b in zip(["position", "uv", "normal", "index"], blobs):
            meta[name] = offset
            fh.write(b.tobytes())
            offset += b.nbytes
    (OUT / "face.json").write_text(json.dumps(meta))
    print(f"{photo.name}: {len(pos_f)} verts, {len(tris)} tris, {offset / 1024:.0f} KB, meta={meta_extra}")

    if args.debug:
        Image.fromarray((w_photo * 255).astype(np.uint8)).save(ROOT / "tools" / "debug-photo-weight.png")
        Image.fromarray((face_mask * 255).astype(np.uint8)).save(ROOT / "tools" / "debug-face-mask.png")


if __name__ == "__main__":
    main()
