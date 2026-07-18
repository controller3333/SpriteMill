#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pixelize_sheet.py -- turn a finished LT sprite sheet into genuine pixel art.

Two modes (--quantize):

  dither (default): 普通のディザ減色。opaque加重ボックス縮小 ->
     Floyd-Steinberg 誤差拡散でパレットへ量子化 -> 黒系1ドット縁取り。
     フルカラーはめ込み (顔プレート転写) や特殊な後処理は一切しない
     (2026-07-10 要望: idle顔の全コマ転写でつながりが破綻するため)。
  legacy: 従来の特殊減色。stages map one-to-one onto what makes
     hand-made ドット絵 read as pixel art:

  1. Palette: either extracted from the character (median-cut + k-means) or
     a FIXED designer palette (--palette-file, e.g. the RPG Maker VX RTP
     256; EDGE2 .pal and JSON supported). Fixed palettes are organized as
     hue-coherent luma ramps -- detected automatically.
  2. Block reduce: hybrid mean/majority vote on palette indices (smooth
     blocks take the quantized mean so shading survives; edgy blocks take a
     dark-weighted vote so linework survives).
  3. Ramp discipline (fixed palette): keep only the handful of ramps the
     character really uses, majority-smooth the ramp field (kills dither
     mottle), then re-tone every dot from SOURCE luma inside its ramp --
     shading returns exactly where the original had it.
  4. Cleanup: silhouette stair regularization, luma-guarded despeckle,
     hanging-dot absorption (eye highlights preserved).
  5. Outline: consistent genuinely-dark 1-dot enclosing contour (hue kept).
  6. Face symmetry (front cells): axis from the EYE PAIR (a tail in the
     silhouette cannot skew it), the larger eye is flip-copied over the
     other (the pixel-artist move), lone under-eye protrusion columns are
     trimmed, and the brow-to-chin band is strictly mirrored with source
     luma deciding borderline dots. Back cells get a conservative
     source-truth mirror pass. The contour ring is frozen throughout.
  7. Hard alpha: a block is opaque iff >=50% covered. No AA halos.

Usage:
    python engine/pixelize_sheet.py --round-dir <round> [--colors 24]
        [--pixel-scale 1] [--cell-w 64 --cell-h 128]
        [--palette-file palettes/rtp_vx_256.json]
Outputs into <round>/08_pixel_art/:
    <char>T_pixel.png      pixel-art sheet at T resolution
    <char>T_pixel@2x.png   nearest-neighbor 2x for eyeballing
    cells/<dir>/*.png      per-direction cells (preview-compatible)
    <char>_pixel_*.png     round template (run_config "template") re-rendered
                           from the pixel cells (t_spec 以外のとき)
    pixel_params.json      parameters used
"""
from __future__ import annotations

import argparse
import colorsys
import json
import sys
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

DIR_PLACEMENT = {
    "front": (0, 0), "left": (1, 0), "right": (2, 0), "back": (3, 0),
    "front_left": (0, 6), "front_right": (1, 6),
    "back_left": (2, 6), "back_right": (3, 6),
}
FRAME_NAMES = ["idle", "walk1", "walk2", "walk3", "walk4", "walk5"]
LUMW = np.array([0.299, 0.587, 0.114])


def _luma(pal: np.ndarray) -> np.ndarray:
    return pal.astype(np.float32) @ LUMW.astype(np.float32)


# ---------------------------------------------------------------- palette

def extract_palette(img: Image.Image, colors: int,
                    merge_dist: float = 14.0) -> np.ndarray:
    """Global palette from the character's opaque pixels (median cut + kmeans),
    then near-duplicates merged -- k-means often emits 2-3 almost-equal darks
    that turn hair into speckle; a hand palette would use one."""
    small = img.resize((max(1, img.width // 4), max(1, img.height // 4)),
                       Image.LANCZOS)
    px = np.asarray(small.convert("RGBA"))
    opaque = px[px[..., 3] > 128][:, :3]
    if len(opaque) == 0:
        raise SystemExit("sheet has no opaque pixels")
    strip = Image.new("RGB", (len(opaque), 1))
    strip.putdata([tuple(p) for p in opaque])
    q = strip.quantize(colors=colors, method=Image.MEDIANCUT, kmeans=colors)
    pal = np.array(q.getpalette()[:colors * 3],
                   dtype=np.float32).reshape(-1, 3)
    kept: list[np.ndarray] = []
    for c in pal:
        if all(np.linalg.norm(c - k) >= merge_dist for k in kept):
            kept.append(c)
    return np.array(kept, dtype=np.float32)


def load_palette_file(path: str | Path) -> np.ndarray:
    """Fixed palette from .json ([[r,g,b],...]) or EDGE2 .pal (zlib-packed
    records; the 768-byte record is RGB x 256). Order is preserved --
    consecutive entries form the designer's shading ramps."""
    p = Path(path)
    if p.suffix.lower() == ".json":
        return np.array(json.loads(p.read_text(encoding="utf-8")),
                        dtype=np.float32)
    raw = p.read_bytes()
    if not raw.startswith(b"EDGE2 PAL"):
        raise SystemExit(f"unsupported palette format: {p}")
    d = zlib.decompress(raw[16:])
    off = 0
    while off + 6 <= len(d):
        size = int.from_bytes(d[off + 2:off + 6], "little")
        data = d[off + 6:off + 6 + size]
        if size and size % 3 == 0 and size >= 48:
            n = size // 3
            return np.array([[data[i * 3], data[i * 3 + 1], data[i * 3 + 2]]
                             for i in range(n)], dtype=np.float32)
        off += 6 + size
    raise SystemExit(f"no RGB record found in {p}")


def detect_ramps(pal: np.ndarray) -> list[list[int]]:
    """Maximal consecutive palette runs with coherent hue and monotonic
    luma = the designer's shading ramps (RTP VX: mostly 5-step)."""
    n = len(pal)
    lum = _luma(pal)
    hsv = [colorsys.rgb_to_hsv(*(c / 255.0)) for c in pal]
    ramps = []
    i = 0
    while i < n:
        run = [i]
        direction = 0
        j = i + 1
        while j < n:
            a, b = run[-1], j
            dl = float(lum[b] - lum[a])
            if abs(dl) < 4 or abs(dl) > 95:
                break
            d = 1 if dl > 0 else -1
            if direction and d != direction:
                break
            ha, sa, _ = hsv[a]
            hb, sb, _ = hsv[b]
            if sa > 0.12 and sb > 0.12:
                hd = abs(ha - hb)
                if min(hd, 1 - hd) > 0.09:
                    break
            elif max(sa, sb) > 0.35 and (sa > 0.25) != (sb > 0.25):
                break
            direction = d
            run.append(j)
            j += 1
        ramps.append(sorted(run, key=lambda k: float(lum[k])))
        i = run[-1] + 1
    return ramps


def synth_ramps(pal: np.ndarray) -> list[list[int]]:
    """Ramps for an EXTRACTED palette (arbitrary order): group colors into
    hue families, luma-sort each family. Gives the character's own vivid
    colors the same ramp discipline as a designer palette -- 'ドット感だけ
    出して色は維持' mode."""
    n = len(pal)
    lum = _luma(pal)
    hsv = [colorsys.rgb_to_hsv(*(c / 255.0)) for c in pal]
    sat_idx = sorted((i for i in range(n) if hsv[i][1] > 0.15),
                     key=lambda i: hsv[i][0])
    used: set[int] = set()
    ramps: list[list[int]] = []
    for i in sat_idx:
        if i in used:
            continue
        fam = [i]
        used.add(i)
        for j in sat_idx:
            if j in used:
                continue
            hd = abs(hsv[i][0] - hsv[j][0])
            hd = min(hd, 1 - hd)
            # saturation gate: a crimson coat and warm dark-brown hair sit
            # 0.05 apart on the hue wheel -- without it they merge into
            # one family and coat-red speckles survive in the hair
            if hd <= 0.045 and abs(hsv[i][1] - hsv[j][1]) <= 0.30:
                fam.append(j)
                used.add(j)
        ramps.append(sorted(fam, key=lambda k: float(lum[k])))
    grays = [i for i in range(n) if hsv[i][1] <= 0.15]
    if grays:
        ramps.append(sorted(grays, key=lambda k: float(lum[k])))
    return ramps


# ----------------------------------------------------------------- reduce

def _diffuse_fs(rgb: np.ndarray, pal: np.ndarray,
                strength: float) -> np.ndarray:
    """Floyd-Steinberg 誤差拡散 (強さ可変)。拡散する誤差に strength
    (0..1) を掛ける: 1.0 でフルFS (PILのFLOYDSTEINBERG相当)、小さく
    するほど市松のザラつきが減って平坦なベタ塗りに近づく。逐次処理は
    Python ループになるので、最近傍パレット検索は 32^3 の RGB LUT
    (8階調ビン) で引く -- 全ドット x 全色の距離計算を回すより桁違いに
    速く、ビン量子化の誤差はディザが均してしまう。"""
    g = (np.arange(32, dtype=np.float32) * 8 + 4)
    grid = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    lut = np.empty(len(grid), dtype=np.uint16)
    step = 4096
    for i in range(0, len(grid), step):
        d = ((grid[i:i + step, None, :] - pal[None, :, :]) ** 2).sum(-1)
        lut[i:i + step] = d.argmin(1)
    lut = lut.reshape(32, 32, 32)

    h, w = rgb.shape[:2]
    buf = rgb.astype(np.float32).copy()
    out = np.empty((h, w), dtype=np.uint16)
    for y in range(h):
        row = buf[y]
        nrow = buf[y + 1] if y + 1 < h else None
        for x in range(w):
            px = row[x]
            r = 0 if px[0] < 0 else (31 if px[0] >= 255 else int(px[0]) >> 3)
            gg = 0 if px[1] < 0 else (31 if px[1] >= 255 else int(px[1]) >> 3)
            b = 0 if px[2] < 0 else (31 if px[2] >= 255 else int(px[2]) >> 3)
            k = lut[r, gg, b]
            out[y, x] = k
            if strength <= 0.0:
                continue
            err = (px - pal[k]) * strength
            if x + 1 < w:
                row[x + 1] += err * (7 / 16)
            if nrow is not None:
                if x > 0:
                    nrow[x - 1] += err * (3 / 16)
                nrow[x] += err * (5 / 16)
                if x + 1 < w:
                    nrow[x + 1] += err * (1 / 16)
    return out


def pixelize_dither(src: Image.Image, colors: int, factor: int,
                    fixed_pal: np.ndarray | None = None,
                    dither: float = 1.0):
    """普通のディザ減色 (--quantize dither): opaque加重ボックス縮小の後、
    Floyd-Steinberg 誤差拡散でパレットへ量子化するだけ。ブロック多数決も
    ランプ矯正も顔処理もしない。アルファは50%被覆のハードマスク。
    dither は拡散誤差のスケール (0=ディザなしの最近傍減色 .. 1=フルFS)。
    Returns (index array (th,tw), alpha mask (th,tw) bool, palette)."""
    pal = fixed_pal if fixed_pal is not None else extract_palette(src, colors)
    rgba = np.asarray(src.convert("RGBA"), dtype=np.float32)
    h, w = rgba.shape[:2]
    th, tw = h // factor, w // factor
    rgba = rgba[:th * factor, :tw * factor]
    a = rgba[..., 3] > 128
    blocks_a = a.reshape(th, factor, tw, factor)
    blocks_rgb = rgba[..., :3].reshape(th, factor, tw, factor, 3)
    am = blocks_a[..., None]
    asum = np.maximum(am.sum(axis=(1, 3)), 1)
    mean_rgb = (blocks_rgb * am).sum(axis=(1, 3)) / asum
    mask = blocks_a.mean(axis=(1, 3)) >= 0.5

    # 誤差拡散はマスクを知らない: 縁の外が黒(0,0,0)のままだとシルエット際に
    # 黒誤差が流れ込むので、不透明色を数回外側へ膨張させて敷いておく
    # (量子化後にマスクで捨てる)。残りは不透明部の平均色。
    have = blocks_a.any(axis=(1, 3))
    fill = mean_rgb.copy()
    for _ in range(4):
        if bool(have.all()):
            break
        ph = np.pad(have, 1, mode="constant")
        pf = np.pad(fill, ((1, 1), (1, 1), (0, 0)), mode="constant")
        acc = np.zeros_like(fill)
        cnt = np.zeros(have.shape, dtype=np.float32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                nh = ph[1 + dy:1 + dy + th, 1 + dx:1 + dx + tw]
                cnt += nh
                acc += pf[1 + dy:1 + dy + th, 1 + dx:1 + dx + tw] \
                    * nh[..., None]
        new = ~have & (cnt > 0)
        if not new.any():
            break
        fill[new] = acc[new] / cnt[new][..., None]
        have |= new
    if not bool(have.all()):
        base = mean_rgb[mask].mean(axis=0) if mask.any() else 128.0
        fill[~have] = base

    n = len(pal)
    dither = max(0.0, min(1.0, float(dither)))
    if dither >= 0.999:
        # フルFSはPILに任せる (速く、可変実装と見た目は同等)
        pal_u8 = np.clip(pal, 0, 255).astype(np.uint8)
        if n < 256:
            # 端数はパレット末尾色の複製で埋める: 黒(0,0,0)で埋めると
            # 誤差拡散がそこへ量子化してしまう。複製が選ばれても色は
            # 同一なので後段のクリップで正規のインデックスに戻せる。
            pal_u8 = np.vstack([pal_u8,
                                np.repeat(pal_u8[-1:], 256 - n, axis=0)])
        pimg = Image.new("P", (1, 1))
        pimg.putpalette(pal_u8.reshape(-1).tolist())
        q = Image.fromarray(np.clip(fill, 0, 255).astype(np.uint8), "RGB") \
            .quantize(palette=pimg, dither=Image.FLOYDSTEINBERG)
        idx = np.minimum(np.asarray(q, dtype=np.uint8), n - 1)
    elif dither <= 0.0:
        idx = nearest_index(np.clip(fill, 0, 255), pal.astype(np.float32))
    else:
        idx = _diffuse_fs(np.clip(fill, 0, 255),
                          pal.astype(np.float32), dither)
    return idx, mask, pal.astype(np.float32)


def nearest_index(rgb: np.ndarray, pal: np.ndarray) -> np.ndarray:
    """Nearest palette index, row-chunked (a 256-color palette against a
    full LT sheet would otherwise broadcast tens of GB)."""
    h = rgb.shape[0]
    out = np.empty(rgb.shape[:2], dtype=np.uint8)
    step = max(1, int(4e7 / (max(1, rgb.shape[1]) * len(pal))))
    for y in range(0, h, step):
        blk = rgb[y:y + step]
        d = ((blk[:, :, None, :] - pal[None, None, :, :]) ** 2).sum(-1)
        out[y:y + step] = d.argmin(-1).astype(np.uint8)
    return out


def pixelize(src: Image.Image, colors: int, factor: int,
             line_weight: float = 1.25,
             fixed_pal: np.ndarray | None = None):
    """Returns (index array (th,tw), alpha mask (th,tw) bool, palette,
    source block-mean luma (th,tw), source block-mean RGB (th,tw,3)) --
    the source means are the ground truth for re-toning and eye colors."""
    from PIL import ImageEnhance
    boosted = ImageEnhance.Contrast(
        ImageEnhance.Color(src.convert("RGBA")).enhance(1.12)).enhance(1.06)
    src = boosted
    pal = fixed_pal if fixed_pal is not None else extract_palette(src, colors)
    colors = len(pal)
    luma = _luma(pal)
    dark = luma < np.percentile(luma, 30)  # line/shadow colors

    rgba = np.asarray(src.convert("RGBA"), dtype=np.float32)
    h, w = rgba.shape[:2]
    th, tw = h // factor, w // factor
    rgba = rgba[:th * factor, :tw * factor]

    idx = nearest_index(rgba[..., :3], pal)
    a = rgba[..., 3] > 128

    blocks_i = idx.reshape(th, factor, tw, factor)
    blocks_a = a.reshape(th, factor, tw, factor)
    blocks_rgb = rgba[..., :3].reshape(th, factor, tw, factor, 3)

    counts = np.zeros((colors, th, tw), dtype=np.float32)
    for c in range(colors):
        cnt = ((blocks_i == c) & blocks_a).sum(axis=(1, 3))
        counts[c] = cnt * (line_weight if dark[c] else 1.0)
    vote_idx = counts.argmax(axis=0).astype(np.uint8)
    out_a = blocks_a.mean(axis=(1, 3)) >= 0.5

    am = blocks_a[..., None]
    asum = np.maximum(am.sum(axis=(1, 3)), 1)
    mean_rgb = (blocks_rgb * am).sum(axis=(1, 3)) / asum
    lum = blocks_rgb @ LUMW.astype(np.float32)
    mlum = (lum * blocks_a).sum(axis=(1, 3)) / asum[..., 0]
    var = (((lum - mlum[:, None, :, None]) ** 2) * blocks_a
           ).sum(axis=(1, 3)) / asum[..., 0]
    smooth = (var < 18.0 ** 2) & out_a
    mean_idx = nearest_index(mean_rgb, pal)
    out_idx = np.where(smooth, mean_idx, vote_idx).astype(np.uint8)

    # silhouette stair regularization
    for _ in range(3):
        pada = np.pad(out_a, 1, mode="constant")
        votes = np.zeros((th, tw), dtype=np.int8)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                votes += pada[1 + dy:1 + dy + th, 1 + dx:1 + dx + tw]
        new_a = (votes + out_a.astype(np.int8)) >= 5
        new_a &= ~(out_a & (votes <= 1))
        new_a |= (~out_a & (votes >= 7))
        if bool((new_a == out_a).all()):
            break
        out_a = new_a

    # luma-guarded despeckle
    for _ in range(2):
        neigh_counts = np.zeros((colors, th, tw), dtype=np.int16)
        padded = np.pad(out_idx, 1, mode="edge")
        pada = np.pad(out_a, 1, mode="constant")
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ni = padded[1 + dy:1 + dy + th, 1 + dx:1 + dx + tw]
                na = pada[1 + dy:1 + dy + th, 1 + dx:1 + dx + tw]
                for c in range(colors):
                    neigh_counts[c] += ((ni == c) & na)
        own = np.take_along_axis(neigh_counts,
                                 out_idx[None].astype(np.int64),
                                 axis=0)[0]
        majority = neigh_counts.argmax(axis=0).astype(np.uint8)
        gap = np.abs(luma[out_idx] - luma[majority])
        weak = out_a & (own <= 1) & (gap < 45)
        if not weak.any():
            break
        out_idx[weak] = majority[weak]

    return out_idx, out_a, pal.astype(np.float32), mlum, mean_rgb


# ------------------------------------------------------- ramp discipline

def whitelist_ramps(idx, mask, pal, ramp_of, ramps, min_frac=0.005):
    """A hand-made sprite uses a handful of ramps, not all 64. Keep only
    ramps carrying real coverage; strays (a green dot in a brown fox) get
    remapped to the nearest kept color."""
    rid = ramp_of[idx]
    counts = np.bincount(rid[mask].ravel(), minlength=len(ramps))
    total = int(mask.sum())
    keep = counts >= max(8, int(total * min_frac))
    if not keep.any():
        return
    kept_steps = np.array([i for k, r in enumerate(ramps) if keep[k]
                           for i in r], dtype=np.int64)
    bad = mask & ~keep[rid]
    if bad.any():
        sub = pal[idx[bad]]
        d = ((sub[:, None, :] - pal[kept_steps][None, :, :]) ** 2).sum(-1)
        idx[bad] = kept_steps[d.argmin(1)].astype(idx.dtype)
    print(f"ramp whitelist: kept {int(keep.sum())}/{len(ramps)} ramps, "
          f"remapped {int(bad.sum())} stray dots")


def smooth_ramp_field(idx, mask, pal, ramp_of, ramps, mlum, passes=3):
    """Majority-vote the ramp id so materials are contiguous, restoring the
    dot's tone from SOURCE luma inside the winning ramp."""
    lum = _luma(pal)
    th, tw = idx.shape
    nramp = len(ramps)
    rid = ramp_of[idx]
    for _ in range(passes):
        pr = np.pad(rid, 1, mode="edge")
        pm = np.pad(mask, 1, mode="constant")
        votes = np.zeros((nramp, th, tw), dtype=np.int8)
        yy = np.tile(np.arange(th)[:, None], (1, tw)).ravel()
        xx = np.tile(np.arange(tw)[None, :], (th, 1)).ravel()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                nr = pr[1 + dy:1 + dy + th, 1 + dx:1 + dx + tw]
                na = pm[1 + dy:1 + dy + th, 1 + dx:1 + dx + tw]
                np.add.at(votes, (nr.ravel(), yy, xx),
                          na.ravel().astype(np.int8))
        own = np.take_along_axis(votes, rid[None].astype(np.int64), 0)[0]
        best = votes.argmax(axis=0)
        best_ct = votes.max(axis=0)
        move = mask & (own <= 2) & (best_ct >= 5) & (best != rid)
        if not move.any():
            break
        ys, xs = np.where(move)
        for y, x in zip(ys, xs):
            steps = ramps[best[y, x]]
            sl = np.array([lum[s] for s in steps])
            idx[y, x] = steps[int(np.abs(sl - mlum[y, x]).argmin())]
        rid = ramp_of[idx]
    return rid


def retone_from_source(idx, mask, pal, rid, ramps, mlum, hysteresis=6.0):
    """Absolute re-toning: each dot takes the step of ITS ramp whose luma is
    nearest to the SOURCE luma -- shading reappears exactly where the source
    has it, with hysteresis so near-ties don't flip (no noise amplification;
    quantile stretching turned fur texture into stripes)."""
    lum = _luma(pal)
    for k, ramp in enumerate(ramps):
        if len(ramp) < 2:
            continue
        sel = mask & (rid == k)
        if not sel.any():
            continue
        rl = np.array([lum[s] for s in ramp])
        sl = mlum[sel]
        d = np.abs(rl[None, :] - sl[:, None])
        new_idx = np.array(ramp, dtype=idx.dtype)[d.argmin(1)]
        cur = idx[sel]
        switch = (np.abs(lum[cur] - sl) - d.min(1)) > hysteresis
        cur[switch] = new_idx[switch]
        idx[sel] = cur


# ----------------------------------------------------------- cleanup

def absorb_hanging_dots(idx: np.ndarray, mask: np.ndarray,
                        pal: np.ndarray) -> None:
    """Remove lone dots with NO orthogonal same-color neighbor (the classic
    'reduced-image' tell), absorbing them into the dominant surrounding
    color. A bright dot sitting inside a DARK cluster is a deliberate eye
    highlight and is preserved."""
    colors = len(pal)
    lum = _luma(pal)
    th, tw = idx.shape
    for _ in range(2):
        pi = np.pad(idx, 1, mode="edge")
        pm = np.pad(mask, 1, mode="constant")
        orth_same = np.zeros((th, tw), dtype=np.int8)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni = pi[1 + dy:1 + dy + th, 1 + dx:1 + dx + tw]
            na = pm[1 + dy:1 + dy + th, 1 + dx:1 + dx + tw]
            orth_same += ((ni == idx) & na)
        counts = np.zeros((colors, th, tw), dtype=np.int8)
        lum_sum = np.zeros((th, tw), dtype=np.float32)
        lum_n = np.zeros((th, tw), dtype=np.float32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ni = pi[1 + dy:1 + dy + th, 1 + dx:1 + dx + tw]
                na = pm[1 + dy:1 + dy + th, 1 + dx:1 + dx + tw]
                for c in range(colors):
                    counts[c] += ((ni == c) & na)
                lum_sum += np.where(na, lum[ni], 0)
                lum_n += na
        dom = counts.argmax(axis=0).astype(np.uint8)
        dom_ct = counts.max(axis=0)
        mean_nl = lum_sum / np.maximum(lum_n, 1)
        highlight = (lum[idx] - mean_nl > 40) & (mean_nl < 90)
        conspic = np.abs(lum[idx] - mean_nl) > 25
        hanging = (mask & (orth_same == 0)
                   & ((dom_ct >= 5) | conspic) & ~highlight)
        if not hanging.any():
            break
        idx[hanging] = dom[hanging]


def outline_pass(idx: np.ndarray, mask: np.ndarray,
                 pal: np.ndarray) -> None:
    """Consistent 1-dot dark enclosing contour: each silhouette-edge dot
    takes the palette color nearest to 0.38x its own RGB, constrained to be
    genuinely dark -- hue is kept, mid-tone mushy edges are not."""
    colors = len(pal)
    lum = _luma(pal)
    global_dark = int(lum.argmin())
    omap = np.empty(colors, dtype=np.uint8)
    for i in range(colors):
        target = pal[i].astype(np.float32) * 0.38
        d = ((pal.astype(np.float32) - target) ** 2).sum(-1)
        d[lum > 70] = 1e12
        j = int(d.argmin())
        omap[i] = j if lum[j] <= 70 else global_dark
    pm = np.pad(mask, 1, mode="constant")
    edge = mask & ~(pm[:-2, 1:-1] & pm[2:, 1:-1]
                    & pm[1:-1, :-2] & pm[1:-1, 2:])
    idx[edge] = omap[idx[edge]]


# ------------------------------------------------------- face symmetry

def dot_symmetrize_src(idx, mask, pal, mlum, x0, y0, w, h, frozen=None):
    """Conservative mirror pass (back cells): mismatched mirror pairs whose
    SOURCE luma agrees get the color matching the shared source tone;
    bilateral bright-in-dark highlights are copied to both sides."""
    si = idx[y0:y0 + h, x0:x0 + w]
    sm = mask[y0:y0 + h, x0:x0 + w]
    sl = mlum[y0:y0 + h, x0:x0 + w]
    if not sm.any():
        return
    lum = _luma(pal)
    cols = np.where(sm.any(axis=0))[0]
    c2 = int(cols.min()) + int(cols.max())
    xs = np.arange(w)
    best, bscore = c2, None
    for ax2 in range(c2 - 3, c2 + 4):
        mx = ax2 - xs
        valid = (mx >= 0) & (mx < w)
        mm = np.zeros_like(sm)
        mm[:, valid] = sm[:, mx[valid]]
        s = int((sm ^ mm).sum())
        if bscore is None or s < bscore:
            bscore, best = s, ax2
    mx = best - xs
    valid = (mx >= 0) & (mx < w)
    mi = np.zeros_like(si)
    mm = np.zeros_like(sm)
    ml = np.zeros_like(sl)
    mi[:, valid] = si[:, mx[valid]]
    mm[:, valid] = sm[:, mx[valid]]
    ml[:, valid] = sl[:, mx[valid]]

    both = sm & mm
    mism = both & (si != mi)
    src_agree = np.abs(sl - ml) < 14.0
    fix = mism & src_agree
    if frozen is not None:
        fz = frozen[y0:y0 + h, x0:x0 + w]
        fzm = np.zeros_like(fz)
        fzm[:, valid] = fz[:, mx[valid]]
        fix &= ~fz & ~fzm
    tgt = (sl + ml) / 2.0
    d_own = np.abs(lum[si] - tgt)
    d_mir = np.abs(lum[mi] - tgt)
    pick = np.where(d_mir + 1.0 < d_own, mi, si)
    si[fix] = pick[fix]

    # bilateral highlight enforcement
    pl = np.pad(lum[si] * sm, 1, mode="constant")
    pm8 = np.pad(sm, 1, mode="constant")
    dark_n = np.zeros(si.shape, dtype=np.int8)
    lsum = np.zeros(si.shape, dtype=np.float32)
    lcnt = np.zeros(si.shape, dtype=np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            nl = pl[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
            na = pm8[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
            dark_n += (na & (nl < 80))
            lsum += np.where(na, nl, 0)
            lcnt += na
    nmean = lsum / np.maximum(lcnt, 1)
    hi = sm & (dark_n >= 5) & (lum[si] - nmean > 35)
    if frozen is not None:
        hi &= ~frozen[y0:y0 + h, x0:x0 + w]
    ys, xs2 = np.where(hi)
    for y, x in zip(ys, xs2):
        qx = best - x
        if 0 <= qx < w and sm[y, qx] and dark_n[y, qx] >= 5 \
                and lum[si[y, qx]] < lum[si[y, x]] - 35 \
                and (frozen is None or not frozen[y0 + y, x0 + qx]):
            si[y, qx] = si[y, x]

    if frozen is None:
        agree = np.zeros(si.shape, dtype=np.int8)
        pm_ = np.pad(sm, 1, mode="constant")
        qm_ = np.pad(mm, 1, mode="constant")
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                agree += (pm_[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
                          == qm_[1 + dy:1 + dy + h, 1 + dx:1 + dx + w])
        on4 = np.zeros(si.shape, dtype=np.int8)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            on4 += pm_[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        kill = (sm ^ mm) & sm & (agree >= 7) & (on4 <= 2)
        sm[kill] = False


def load_eye_lib(lib_dir) -> list[dict]:
    """Eye template library: eyes/*.json with a role-letter grid.
    Roles: O=outline, S=iris shadow, I=iris, H=highlight, W=sclera,
    .=passthrough. Colors are sampled from the character, the SHAPE is
    the fixed format."""
    out = []
    d = Path(lib_dir) if lib_dir else None
    if d is None or not d.is_dir():
        return out
    for p in sorted(d.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            grid = data["grid"]
            if not grid or len({len(r) for r in grid}) != 1:
                continue
            if any(ch not in ".OSIHW" for r in grid for ch in r):
                continue
            out.append({"name": data.get("name", p.stem), "grid": grid,
                        "h": len(grid), "w": len(grid[0]),
                        "mirror": bool(data.get("mirror", True))})
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            continue
    return out


def _pick_eye_template(eye_lib, ew: int, eh: int, prefer=None):
    """Nearest template by bbox size; None when nothing fits within
    +-2 dots per axis (fall back to organic stamping). An explicit style
    (name substring) wins regardless of size tolerance."""
    if prefer:
        p = prefer.lower()
        pool = [t for t in eye_lib if p in t["name"].lower()]
        if pool:
            t = min(pool,
                    key=lambda t: abs(t["w"] - ew) + abs(t["h"] - eh))
            # even a forced style must roughly fit -- stamping a 9x12 eye
            # over a 4x5 detection makes saucer eyes
            if abs(t["w"] - ew) + abs(t["h"] - eh) <= 4:
                return t
            return None
    best, score = None, None
    for t in eye_lib:
        dw, dh = abs(t["w"] - ew), abs(t["h"] - eh)
        if dw > 2 or dh > 2:
            continue
        s = dw + dh
        if score is None or s < score:
            best, score = t, s
    return best


def _eye_role_colors(si, sm, pal, sel, srgb=None) -> dict:
    """Sample the character's own colors for each template role. S and I
    must be VISIBLY lighter steps of the eye's hue family -- an all-dark
    organic eye (dark iris crushed into the frame) would otherwise map
    every role to near-black and any template renders as a black bean."""
    lum = _luma(pal)

    def mode_of(vals, fallback):
        vals = list(vals)
        if not vals:
            return fallback
        return int(np.bincount(np.array(vals)).argmax())

    cluster = [int(v) for v in si[sel]]
    darkest = min(cluster, key=lambda i: lum[i])
    cell_cols = np.unique(si[sm])
    brightest = int(cell_cols[np.argmax(lum[cell_cols])])
    o = mode_of([i for i in cluster if lum[i] < 45], darkest)
    # iris hue from the SOURCE image, not the quantized cluster -- a dark
    # organic eye quantizes to blacks/grays and would gray out the iris
    if srgb is not None:
        sl = srgb[sel]
        slum = sl @ LUMW.astype(np.float32)
        mid = sl[slum >= 50]
        if not len(mid):
            k = max(1, len(sl) * 3 // 10)
            mid = sl[np.argsort(slum)[-k:]]  # brightest 30% of a dark eye
        ref_rgb = mid.mean(0)
    else:
        ref_rgb = pal[mode_of([i for i in cluster if lum[i] >= 45], o)]
    hr, sr, _ = colorsys.rgb_to_hsv(
        *(np.clip(ref_rgb, 0, 255) / 255.0))

    def hue_step(target_l: float) -> int:
        best, bd = o, None
        for i in range(len(pal)):
            hi_, si_, _ = colorsys.rgb_to_hsv(*(pal[i] / 255.0))
            if si_ > 0.15 and sr > 0.15:
                hd = abs(hi_ - hr)
                hd = min(hd, 1 - hd)
                if hd > 0.09:
                    continue
            else:
                hd = 0.0 if si_ <= 0.15 and sr <= 0.15 else 0.05
            d = abs(float(lum[i]) - target_l) + hd * 180
            if bd is None or d < bd:
                best, bd = int(i), d
        return best

    s = hue_step(max(60.0, float(lum[o]) + 40.0))
    i_ = hue_step(max(105.0, float(lum[o]) + 85.0))
    return {"O": o, "S": s, "I": i_, "H": brightest, "W": brightest}


def _stamp_eye_template(si, sm, fz, tpl, colors, cy, cx, mirror) -> None:
    grid = tpl["grid"]
    gh, gw = tpl["h"], tpl["w"]
    y0, x0 = cy - gh // 2, cx - gw // 2
    h, w = si.shape
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch == ".":
                continue
            y = y0 + r
            x = x0 + (gw - 1 - c if mirror else c)
            if 0 <= y < h and 0 <= x < w and sm[y, x] \
                    and (fz is None or not fz[y, x]):
                si[y, x] = colors[ch]


def _cluster_with_holes(sel, sm):
    """The cluster plus its interior holes (glints live in the holes),
    found by flood-filling the bbox from its border."""
    ys3, xs3 = np.where(sel)
    if not len(ys3):
        return sel & sm
    y0b, y1b = int(ys3.min()), int(ys3.max())
    x0b, x1b = int(xs3.min()), int(xs3.max())
    bb = sel[y0b:y1b + 1, x0b:x1b + 1]
    bh, bw = bb.shape
    reached = np.zeros_like(bb)
    stack = [(yy, xx) for yy in range(bh) for xx in range(bw)
             if (yy in (0, bh - 1) or xx in (0, bw - 1)) and not bb[yy, xx]]
    for yy, xx in stack:
        reached[yy, xx] = True
    while stack:
        yy, xx = stack.pop()
        for dy2, dx2 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            y3, x3 = yy + dy2, xx + dx2
            if 0 <= y3 < bh and 0 <= x3 < bw and not bb[y3, x3] \
                    and not reached[y3, x3]:
                reached[y3, x3] = True
                stack.append((y3, x3))
    out = np.zeros_like(sel)
    out[y0b:y1b + 1, x0b:x1b + 1] = bb | ~reached
    return out & sm


def _plate_register_paste(si, sm, fzc, pal, p, so=None, som=None) -> bool:
    """Paste a captured face plate, positioned by image correlation on the
    plate's LIGHT (skin/fur) dots -- no per-frame feature detection.
    When the plate carries a full-color eye overlay (orgb/om), it is
    pasted into the frame's overlay slices (so/som) at the same spot."""
    rect, rmask, wmask = p["rect"], p["rmask"], p["wmask"]
    ph, pw = rect.shape
    h, w = si.shape
    pr = pal[rect]
    bs, boff = None, (0, 0)
    for dy in range(-5, 6):
        ty = p["py0"] + dy
        if ty < 0 or ty + ph > h:
            continue
        for dx in range(-3, 4):
            tx = p["px0"] + dx
            if tx < 0 or tx + pw > w:
                continue
            m = wmask & sm[ty:ty + ph, tx:tx + pw]
            if int(m.sum()) < int(wmask.sum()) * 0.5:
                continue
            sub = si[ty:ty + ph, tx:tx + pw]
            score = float(np.abs(pal[sub] - pr).sum(-1)[m].mean())
            if bs is None or score < bs:
                bs, boff = score, (dy, dx)
    if bs is None:
        return False
    ty0, tx0 = p["py0"] + boff[0], p["px0"] + boff[1]
    for yy in range(ph):
        for xx in range(pw):
            if not rmask[yy, xx]:
                continue
            y2, x2 = ty0 + yy, tx0 + xx
            if 0 <= y2 < h and 0 <= x2 < w and sm[y2, x2] \
                    and (fzc is None or not fzc[y2, x2]):
                si[y2, x2] = rect[yy, xx]
    if so is not None and p.get("om") is not None:
        prgb, pom = p["orgb"], p["om"]
        for yy in range(ph):
            for xx in range(pw):
                if not pom[yy, xx]:
                    continue
                y2, x2 = ty0 + yy, tx0 + xx
                if 0 <= y2 < h and 0 <= x2 < w and sm[y2, x2] \
                        and (fzc is None or not fzc[y2, x2]):
                    so[y2, x2] = prgb[yy, xx]
                    som[y2, x2] = True
    return True


def face_montage_fullcolor(idx, mask, pal, mlum, x0, y0, w, h, frozen,
                           lock, mrgb, ov_rgb, ov_m):
    """顔全体を減色しない (要望「顔全体を減色しないようにできる?」):
    頭部ゾーンの明るい肌領域 (最大の明色連結成分) と、その内部に閉じ
    込められた暗部 (目・口・眉・顔にかかる眼帯) を元のフルカラーで
    オーバーレイする。領域境界は実在の輪郭 (肌と髪の境) に一致するので
    矩形のような継ぎ目が出ない。idle で顔プレートとして捕獲し、walk
    コマへはレジストレーションで配布 -- 目検出に一切依存しないので、
    髪と目が融合するアルバートでも顔が丸ごと守られる。"""
    si = idx[y0:y0 + h, x0:x0 + w]
    sm = mask[y0:y0 + h, x0:x0 + w]
    if not sm.any():
        return
    fzc = frozen[y0:y0 + h, x0:x0 + w] if frozen is not None else None
    so = ov_rgb[y0:y0 + h, x0:x0 + w]
    som = ov_m[y0:y0 + h, x0:x0 + w]
    if "plate" in lock:
        _plate_register_paste(si, sm, fzc, pal, lock["plate"], so, som)
        return
    sl = mlum[y0:y0 + h, x0:x0 + w]
    srgb = mrgb[y0:y0 + h, x0:x0 + w]
    ys, xs = np.where(sm)
    top, bot = int(ys.min()), int(ys.max())
    zone_end = top + int((bot - top) * 0.55)
    bright = sm & (sl > 140)
    bright[zone_end + 1:] = False
    lab, n = _components(bright)
    if n == 0:
        return
    best_k, best_s = -1, 0
    for k in range(n):
        s = int((lab == k).sum())
        if s > best_s:
            best_k, best_s = k, s
    if best_s < 20:
        return  # no meaningful face region in this view
    face = lab == best_k
    # close small leaks (an eye touching the hairline is not a sealed
    # hole) by dilating the face twice before hole-filling; the dilation
    # ring itself is NOT kept, so the boundary stays on the real edge
    f2 = face.copy()
    for _ in range(2):
        pf = np.pad(f2, 1, mode="constant")
        f2 = f2 | pf[:-2, 1:-1] | pf[2:, 1:-1] | pf[1:-1, :-2] \
            | pf[1:-1, 2:]
    f2 &= sm
    filled2 = _cluster_with_holes(f2, sm)
    face_full = face | (filled2 & ~f2)
    if fzc is not None:
        face_full &= ~fzc
    if not face_full.any():
        return
    so[face_full] = srgb[face_full]
    som[face_full] = True
    fy, fx = np.where(face_full)
    py0 = max(0, int(fy.min()) - 1)
    py1 = min(h - 1, int(fy.max()) + 1)
    px0 = max(0, int(fx.min()) - 1)
    px1 = min(w - 1, int(fx.max()) + 1)
    rect = si[py0:py1 + 1, px0:px1 + 1].copy()
    rmask = sm[py0:py1 + 1, px0:px1 + 1].copy()
    lum = _luma(pal)
    lock["plate"] = {"rect": rect, "rmask": rmask,
                     "wmask": rmask & (lum[rect] >= 95),
                     "py0": py0, "px0": px0,
                     "orgb": so[py0:py1 + 1, px0:px1 + 1].copy(),
                     "om": som[py0:py1 + 1, px0:px1 + 1].copy()}


def montage_direction(idx, mask, pal, mlum, x0, y0, w, h, frozen, lock,
                      mrgb=None, ov_rgb=None, ov_m=None,
                      eye_color="palette"):
    """Face montage for the OTHER eye-visible views (profiles and front
    diagonals): no symmetry logic applies there, so the idle frame simply
    donates its eye zone -- the union bbox of compact dark clusters in
    the upper head -- and walk frames take the plate via registration."""
    si = idx[y0:y0 + h, x0:x0 + w]
    sm = mask[y0:y0 + h, x0:x0 + w]
    if not sm.any():
        return
    fzc = frozen[y0:y0 + h, x0:x0 + w] if frozen is not None else None
    so = ov_rgb[y0:y0 + h, x0:x0 + w] if ov_rgb is not None else None
    som = ov_m[y0:y0 + h, x0:x0 + w] if ov_m is not None else None
    if "plate" in lock:
        _plate_register_paste(si, sm, fzc, pal, lock["plate"], so, som)
        return
    lum = _luma(pal)
    ys, xs = np.where(sm)
    top, bot = int(ys.min()), int(ys.max())
    span = bot - top
    z0 = top + int(span * 0.22)
    z1 = top + int(span * 0.62)
    dark = sm & (lum[si] < 85)
    dark[:z0] = False
    dark[z1 + 1:] = False
    if fzc is not None:
        dark &= ~fzc

    def compact_clusters(m):
        lab, n = _components(m)
        out = []
        for k in range(n):
            sel = lab == k
            s = int(sel.sum())
            if not (4 <= s <= 80):
                continue
            cy2, cx2 = np.where(sel)
            bb_area = (cy2.max() - cy2.min() + 1) * (cx2.max() - cx2.min() + 1)
            if s / bb_area < 0.35:
                continue  # stringy hair strand, not an eye
            out.append(sel)
        return out

    clusters = compact_clusters(dark)
    if not clusters:
        # eye may have merged with the fringe: erode to cut thin bridges
        pd = np.pad(dark, 1, mode="constant")
        core = dark & pd[:-2, 1:-1] & pd[2:, 1:-1] & pd[1:-1, :-2] \
            & pd[1:-1, 2:]
        clusters = compact_clusters(core)
    if not clusters:
        return
    # 目ヂカラ mode: the eye clusters (glint holes included) keep their
    # ORIGINAL full colors -- written to the overlay BEFORE the plate is
    # captured so every frame inherits them
    if eye_color == "original" and mrgb is not None and so is not None:
        srgb_full = mrgb[y0:y0 + h, x0:x0 + w]
        for sel in clusters:
            # bounding box, not the dark cluster: iris/glint colors live
            # inside the box, the cluster is just the lash line
            ys5, xs5 = np.where(sel)
            box = np.zeros_like(sel)
            box[max(0, int(ys5.min()) - 1):min(h, int(ys5.max()) + 2),
                max(0, int(xs5.min())):int(xs5.max()) + 1] = True
            box &= sm
            if fzc is not None:
                box &= ~fzc
            so[box] = srgb_full[box]
            som[box] = True
    ally, allx = np.where(np.logical_or.reduce(clusters))
    py0 = max(0, int(ally.min()) - 2)
    py1 = min(h - 1, int(ally.max()) + 3)
    px0 = max(0, int(allx.min()) - 3)
    px1 = min(w - 1, int(allx.max()) + 3)
    rect = si[py0:py1 + 1, px0:px1 + 1].copy()
    rmask = sm[py0:py1 + 1, px0:px1 + 1].copy()
    lock["plate"] = {"rect": rect, "rmask": rmask,
                     "wmask": rmask & (lum[rect] >= 95),
                     "py0": py0, "px0": px0,
                     "orgb": (so[py0:py1 + 1, px0:px1 + 1].copy()
                              if so is not None else None),
                     "om": (som[py0:py1 + 1, px0:px1 + 1].copy()
                            if som is not None else None)}


def _components(m):
    lab = -np.ones(m.shape, dtype=np.int32)
    n = 0
    for y0, x0 in zip(*np.where(m)):
        if lab[y0, x0] >= 0:
            continue
        stack = [(y0, x0)]
        lab[y0, x0] = n
        while stack:
            y, x = stack.pop()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < m.shape[0] and 0 <= xx < m.shape[1] \
                        and m[yy, xx] and lab[yy, xx] < 0:
                    lab[yy, xx] = n
                    stack.append((yy, xx))
        n += 1
    return lab, n


def face_stamp_symmetrize(idx, mask, pal, mlum, x0, y0, w, h, frozen=None,
                          eye_lib=None, eye_style=None, tpl_lock=None,
                          mrgb=None, ov_rgb=None, ov_m=None,
                          eye_color="palette"):
    """The pixel-artist move for front cells: find the eye pair, take its
    midpoint as the face axis (a tail in the silhouette cannot skew it),
    flip-copy the larger eye over the other, trim lone under-eye protrusion
    columns, then strictly mirror the brow-to-chin band (source luma decides
    borderline dots, highlights win outright, contour ring frozen)."""
    si = idx[y0:y0 + h, x0:x0 + w]
    sm = mask[y0:y0 + h, x0:x0 + w]
    sl = mlum[y0:y0 + h, x0:x0 + w]
    if not sm.any():
        return False
    lum = _luma(pal)
    ys, xs = np.where(sm)
    top, bot = int(ys.min()), int(ys.max())

    # montage mode: once the idle frame has donated its face plate, later
    # frames just take it wholesale -- position found by image
    # correlation on the plate's LIGHT (skin) dots, so no per-frame
    # feature detection can pick bang shadows and wreck the paste
    if tpl_lock is not None and "plate" in tpl_lock:
        fzc = frozen[y0:y0 + h, x0:x0 + w] if frozen is not None else None
        so2 = ov_rgb[y0:y0 + h, x0:x0 + w] if ov_rgb is not None else None
        som2 = ov_m[y0:y0 + h, x0:x0 + w] if ov_m is not None else None
        return _plate_register_paste(si, sm, fzc, pal, tpl_lock["plate"],
                                     so2, som2)

    dark = sm & (lum[si] < 85)
    dark[int(top + (bot - top) * 0.55):] = False
    fz = frozen[y0:y0 + h, x0:x0 + w] if frozen is not None else None
    if fz is not None:
        dark &= ~fz  # the contour ring is dark too: don't let the eye
        # component leak into it and drag the bbox to the ears
    # erode once: 1-2 dot lash/bang bridges snap off, so a fringe shadow
    # cannot merge with an eye and wreck the pair constraints (C17)
    pd = np.pad(dark, 1, mode="constant")
    core = dark & pd[:-2, 1:-1] & pd[2:, 1:-1] & pd[1:-1, :-2] \
        & pd[1:-1, 2:]
    if core.sum() >= 8:
        lab, n = _components(core)
        core_mode = True
    else:
        lab, n = _components(dark)  # tiny eyes: erosion would kill them
        core_mode = False
    comps = []
    for k in range(n):
        sel = lab == k
        if core_mode:
            # restore the true eye extent: dilate the core by 1 within
            # the dark mask
            ps = np.pad(sel, 1, mode="constant")
            sel = dark & (ps[:-2, 1:-1] | ps[2:, 1:-1] | ps[1:-1, :-2]
                          | ps[1:-1, 2:] | ps[1:-1, 1:-1])
        s = int(sel.sum())
        if s < (4 if core_mode else 6):
            continue
        if s > 200:
            continue  # eyes are compact; a merged hair/collar mass is not
        # 目候補は「肌に浮いている」こと: クラスタ周囲1pxリングの明色率。
        # 暗髪キャラは髪・眼帯・襟が一塊の巨大成分になり、ペア検出が左右の
        # 髪の房を目と誤認していた (アルバート)。髪の房のリングは暗いので
        # ここで弾ける。
        ps2 = np.pad(sel, 1, mode="constant")
        ring = (~sel) & sm & (ps2[:-2, 1:-1] | ps2[2:, 1:-1]
                              | ps2[1:-1, :-2] | ps2[1:-1, 2:])
        rvals = sl[ring]
        if len(rvals) == 0 or float((rvals > 150).mean()) < 0.30:
            continue
        cy, cx = np.where(sel)
        # 目は丸く密、眼帯のストラップ断片は細長い: bbox充填率で弾く
        # (アルバートは額のベルト断片2つがペアにされ顔が万華鏡化した)
        bb_area = (int(cy.max()) - int(cy.min()) + 1) \
            * (int(cx.max()) - int(cx.min()) + 1)
        if s / bb_area < 0.40:
            continue
        comps.append((s, float(cx.mean()), float(cy.mean()), sel))
    best = None
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            a, b = comps[i], comps[j]
            if abs(a[2] - b[2]) > 4 or not (4 <= abs(a[1] - b[1]) <= w * 0.6):
                continue
            if max(a[0], b[0]) / max(1, min(a[0], b[0])) > 3.5:
                continue
            if best is None or a[0] + b[0] > best[0][0] + best[1][0]:
                best = (a, b)
    if best is None:
        return False
    eL, eR = sorted(best, key=lambda e: e[1])
    axis2 = int(round(eL[1] + eR[1]))
    ey0 = int(min(np.where(eL[3])[0].min(), np.where(eR[3])[0].min()))
    ey1 = int(max(np.where(eL[3])[0].max(), np.where(eR[3])[0].max()))

    # 非対称顔ガード (指摘事例: アルバートの眼帯が対称化で崩壊):
    # 目ペア矩形を SOURCE luma で軸ミラー比較し、最良軸 (±2探索: 検出
    # 起因の軸1pxズレで対称顔が誤爆しない) でも一致しないなら、デザイン
    # として非対称 (眼帯・傷) と判定して一切ミラーしない。単純なボックス
    # 輝度平均比較は両方向に誤判定した (眼帯ボックスΔ20 vs 対称顔Δ35)。
    # 判定基準 (3代目): 「目は色を持つ、眼帯は無彩色の暗塊」。クラスタ
    # (+虹彩の下がり2行) 内で元絵クロマが高いドットの割合を測る。虹彩は
    # 暗くても彩度が高い (紫・茶・緑)、黒革の眼帯はほぼ無彩色。輝度系の
    # 比較は 暗vs暗の盲点 (18.8) と位置ズレノイズ (81.2) の両方で死んだ。
    srgb_asym = mrgb[y0:y0 + h, x0:x0 + w] if mrgb is not None else None

    def _color_frac(sel):
        if srgb_asym is None:
            return 1.0
        grown = sel.copy()
        ys5, xs5 = np.where(sel)
        for yy, xx in zip(ys5, xs5):
            for dy2 in (1, 2):  # iris hangs under the lash line
                if yy + dy2 < h and sm[yy + dy2, xx]:
                    grown[yy + dy2, xx] = True
        vals = srgb_asym[grown & sm].astype(np.int16)
        if not len(vals):
            return 0.0
        chroma = vals.max(axis=1) - vals.min(axis=1)
        return float((chroma > 35).mean())

    fL, fR = _color_frac(eL[3]), _color_frac(eR[3])
    lo_f, hi_f = min(fL, fR), max(fL, fR)
    asym = hi_f >= 0.12 and lo_f <= 0.05
    print(f"    目クラスタの有彩色率 L={fL:.2f} R={fR:.2f} -> "
          f"{'非対称 (片側が無彩色の塊): 対称化スキップ・実絵のまま' if asym else '対称'}")

    import os as _os
    if _os.environ.get("SM_DEBUG_EYE"):
        dbg = np.stack([sl, sl, sl], axis=-1).astype(np.uint8)
        for comp in comps:
            dbg[comp[3]] = (80, 80, 255)
        dbg[eL[3]] = (255, 80, 80)
        dbg[eR[3]] = (80, 255, 80)
        Image.fromarray(dbg, "RGB").resize((w * 6, h * 6), Image.NEAREST) \
            .save(_os.environ["SM_DEBUG_EYE"])
        print(f"    debug eye map -> {_os.environ['SM_DEBUG_EYE']}")

    # ---- fixed eye FORMAT: replace both organic eyes with the nearest
    # library template, recolored from the character, mirror-stamped
    # about the face axis (user request: 目のフォーマット化)
    master = eL if eL[0] >= eR[0] else eR
    tpl = None
    if eye_lib and not asym:
        # true eye size: the UNERODED dark component around each core --
        # erosion+dilation underestimates a big lashed eye and a small
        # template would win. Use the smaller of the two components (the
        # other may have merged with the fringe shadow).
        dlab, _dn = _components(dark)
        cand = []
        for eye in (eL, eR):
            ys3, xs3 = np.where(eye[3])
            k2 = int(dlab[ys3[0], xs3[0]])
            if k2 >= 0:
                dys, dxs = np.where(dlab == k2)
                cand.append((len(dys),
                             int(dxs.max() - dxs.min() + 1),
                             int(dys.max() - dys.min() + 1),
                             int(dys.min()), int(dys.max())))
        if cand:
            cand.sort()
            _, ew, eh0, e_top, e_bot = cand[0]
        else:
            mys0, mxs0 = np.where(master[3])
            ew = int(mxs0.max() - mxs0.min() + 1)
            e_top = int(mys0.min())
            e_bot = int(mys0.max())
        mys0, mxs0 = np.where(master[3])
        # the row cutoff clips the dark mask: extend height over dark
        # dots, per column from its OWN bottom, capped -- a shared
        # running bottom ratchets down dark clothing row by row
        for xx in range(int(mxs0.min()), int(mxs0.max()) + 1):
            col = np.where(master[3][:, xx])[0]
            if not len(col):
                continue
            yb = int(col.max())
            grow = 0
            while yb + 1 < h and grow < 2 and sm[yb + 1, xx] \
                    and lum[si[yb + 1, xx]] < 85 \
                    and (fz is None or not fz[yb + 1, xx]):
                yb += 1
                grow += 1
            e_bot = max(e_bot, yb)
        eh = e_bot - e_top + 1
        # one character = one eye format: the first (idle) decision locks
        # the template for every walk frame
        if tpl_lock is not None and "tpl" in tpl_lock:
            tpl = tpl_lock["tpl"]
        else:
            tpl = _pick_eye_template(eye_lib, ew, eh, eye_style)
            if tpl_lock is not None:
                tpl_lock["tpl"] = tpl
            if tpl is not None:
                print(f"    eye format: {tpl['name']} "
                      f"(detected {ew}x{eh})")
    if tpl is not None:
        srgb = mrgb[y0:y0 + h, x0:x0 + w] if mrgb is not None else None
        colors = _eye_role_colors(si, sm, pal, master[3], srgb)
        # templates are authored as the LEFT eye; anchor position off the
        # master (better centroid), orientation off the side. mirror:false
        # templates copy the art unflipped (same-side highlights).
        if master is eL:
            cxl = int(round(eL[1]))
            cxr = axis2 - cxl
        else:
            cxr = int(round(eR[1]))
            cxl = axis2 - cxr
        gh, gw = tpl["h"], tpl["w"]
        cy = e_top + gh // 2
        # clear ONLY old-eye dots inside the stamp footprint (+1): bangs
        # and lashes that merged into the detected cluster but hang
        # OUTSIDE the new eye must survive (C17's fringe got skin-wiped
        # when the whole cluster was cleared)
        for eye, cxc in ((eL, cxl), (eR, cxr)):
            sy0 = max(0, cy - gh // 2 - 1)
            sy1 = min(h - 1, cy - gh // 2 + gh)
            sx0 = max(0, cxc - gw // 2 - 1)
            sx1 = min(w - 1, cxc - gw // 2 + gw)
            sel = eye[3].copy()
            ys2, xs2 = np.where(sel)
            for xx in range(int(xs2.min()), int(xs2.max()) + 1):
                col = np.where(sel[:, xx])[0]
                if not len(col):
                    continue
                yb = int(col.max())
                grow = 0
                while yb + 1 < h and grow < 4 and sm[yb + 1, xx] \
                        and lum[si[yb + 1, xx]] < 85 \
                        and (fz is None or not fz[yb + 1, xx]):
                    yb += 1
                    grow += 1
                    sel[yb, xx] = True
            ring = [int(si[yy, xx])
                    for yy in range(max(0, sy0 - 1), min(h, sy1 + 2))
                    for xx in range(max(0, sx0 - 1), min(w, sx1 + 2))
                    if sm[yy, xx] and not sel[yy, xx]
                    and lum[si[yy, xx]] >= 95
                    and (fz is None or not fz[yy, xx])]
            if not ring:
                continue
            fill = int(np.bincount(np.array(ring)).argmax())
            for yy in range(sy0, sy1 + 1):
                for xx in range(sx0, sx1 + 1):
                    if sel[yy, xx] and sm[yy, xx] \
                            and (fz is None or not fz[yy, xx]):
                        si[yy, xx] = fill
        _stamp_eye_template(si, sm, fz, tpl, colors, cy, cxl, mirror=False)
        _stamp_eye_template(si, sm, fz, tpl, colors, cy, cxr,
                            mirror=tpl["mirror"])
        ry0 = max(0, e_top - 3)
        ry1 = min(h - 1, e_top + gh + 2)
        rx0 = max(0, cxl - gw // 2 - 3)
        rx1 = min(w - 1, cxl + gw // 2 + 3)

    # eye-shape cleanup: a pixel-art eye's bottom edge never has a lone
    # column poking deeper than both its neighbors (tear-line remnants).
    for eye in (() if tpl is not None else (eL, eR)):
        sel = eye[3]
        ys2, xs2 = np.where(sel)
        if xs2.max() - xs2.min() + 1 < 4:
            continue
        bottom = {}
        for yy, xx in zip(ys2, xs2):
            bottom[xx] = max(bottom.get(xx, -1), yy)
        # the detection row-cutoff clips the cluster: follow dark dots
        # downward (max 4) so the bottom map sees the real eye bottom
        for xx in list(bottom):
            start = yb = bottom[xx]
            while yb + 1 < h and yb - start < 4 and sm[yb + 1, xx] \
                    and lum[si[yb + 1, xx]] < 85 \
                    and (fz is None or not fz[yb + 1, xx]):
                yb += 1
            bottom[xx] = yb
        for xx, yb in bottom.items():
            if xx - 1 not in bottom or xx + 1 not in bottom:
                continue
            ref = max(bottom[xx - 1], bottom[xx + 1])
            if 0 < yb - ref <= 2:
                for yy in range(ref + 1, yb + 1):
                    if lum[si[yy, xx]] >= 85:
                        continue
                    for y3, x3 in ((yy + 1, xx), (yy, xx - 1), (yy, xx + 1)):
                        if 0 <= y3 < h and 0 <= x3 < w and sm[y3, x3] \
                                and lum[si[y3, x3]] >= 85:
                            si[yy, xx] = si[y3, x3]
                            break

    # eye stamp fallback (no template fit): larger eye is master, its
    # patch (+3 margin) mirrored over
    if tpl is None and not asym:
        mys, mxs = np.where(master[3])
        ry0, ry1 = max(0, mys.min() - 3), min(h - 1, mys.max() + 3)
        rx0, rx1 = max(0, mxs.min() - 3), min(w - 1, mxs.max() + 3)
        for y in range(ry0, ry1 + 1):
            for x in range(rx0, rx1 + 1):
                qx = axis2 - x
                if 0 <= qx < w and sm[y, x] and sm[y, qx] \
                        and (fz is None or not (fz[y, x] or fz[y, qx])):
                    si[y, qx] = si[y, x]

    # face band mirror: eye top -> muzzle bottom, eye span + margin wide.
    # The bottom is content-driven: extend while the SOURCE itself is
    # mirror-symmetric, stop where it stops being so (a fox's chest tuft
    # is symmetric -> included; a satchel strap across a coat is
    # asymmetric BY DESIGN -> the band ends at the chin, not the belt).
    if asym:
        rx0 = rx1 = 0  # unused; band mirror skipped entirely
    ex_out = max(abs(eL[1] - axis2 / 2.0), abs(eR[1] - axis2 / 2.0)) \
        + (rx1 - rx0) / 2.0
    by0 = max(0, ey0 - 4)
    c = axis2 / 2.0
    x_lo = max(0, int(c - ex_out - 4))
    by1 = min(h - 1, ey1 + 4)
    for y in range(ey1 + 5, min(h - 1, ey1 + 20) + 1):
        if asym:
            break
        pairs = disagree = 0
        for x in range(x_lo, int(c) + 1):
            qx = axis2 - x
            if 0 <= qx < w and qx > x and sm[y, x] and sm[y, qx]:
                pairs += 1
                if abs(sl[y, x] - sl[y, qx]) > 25:
                    disagree += 1
        if pairs >= 4 and disagree / pairs > 0.4:
            break
        by1 = y
    for y in range(by0, by1 + 1):
        if asym:
            break
        for x in range(x_lo, int(c) + 1):
            qx = axis2 - x
            if not (0 <= qx < w) or qx <= x:
                continue
            if not (sm[y, x] and sm[y, qx]) or si[y, x] == si[y, qx]:
                continue
            if fz is None or not (fz[y, x] or fz[y, qx]):
                la, lb = lum[si[y, x]], lum[si[y, qx]]
                if abs(la - lb) > 55:
                    bx = x if la > lb else qx
                    dn = 0
                    for dy2 in (-1, 0, 1):
                        for dx2 in (-1, 0, 1):
                            if dy2 == 0 and dx2 == 0:
                                continue
                            yy, xx = y + dy2, bx + dx2
                            if 0 <= yy < h and 0 <= xx < w and sm[yy, xx] \
                                    and lum[si[yy, xx]] < 80:
                                dn += 1
                    if dn >= 4:
                        pick = si[y, x] if la > lb else si[y, qx]
                        si[y, x] = pick
                        si[y, qx] = pick
                        continue
                tgt = (sl[y, x] + sl[y, qx]) / 2.0
                pick = si[y, x] if abs(la - tgt) <= abs(lb - tgt) \
                    else si[y, qx]
                si[y, x] = pick
                si[y, qx] = pick

    # vivid-mode rescue (黒豆修正): an auto-extracted palette starves the
    # tiny eye of colors and the vote/despeckle crush it into a flat dark
    # bean. If the finished eye region is DEAD (no light dot, low
    # contrast), re-develop it from the SOURCE luma distribution:
    # quantiles of the eye's source luma become frame/shadow/iris/
    # highlight, hues sampled from the source. Left eye only, mirrored to
    # the right; the face plate then carries it to every frame.
    if tpl is None and mrgb is not None and not asym:
        sel_e = eL[3]
        # deadness is judged over the eye BBOX (glint holes included) --
        # the dark cluster itself is dark by definition and would always
        # test as dead, relighting healthy eyes too
        lys3, lxs3 = np.where(sel_e)
        # deadness region = the cluster PLUS its interior holes (glints
        # live in holes). A plain bbox always grazes some bright cheek
        # dot and made true beans look healthy; flood-fill from the bbox
        # border finds what is genuinely inside the eye.
        y0b, y1b = int(lys3.min()), int(lys3.max())
        x0b, x1b = int(lxs3.min()), int(lxs3.max())
        bb = sel_e[y0b:y1b + 1, x0b:x1b + 1]
        bh, bw = bb.shape
        reached = np.zeros_like(bb)
        stack = [(yy, xx) for yy in range(bh) for xx in range(bw)
                 if (yy in (0, bh - 1) or xx in (0, bw - 1))
                 and not bb[yy, xx]]
        for yy, xx in stack:
            reached[yy, xx] = True
        while stack:
            yy, xx = stack.pop()
            for dy2, dx2 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                y3, x3 = yy + dy2, xx + dx2
                if 0 <= y3 < bh and 0 <= x3 < bw and not bb[y3, x3] \
                        and not reached[y3, x3]:
                    reached[y3, x3] = True
                    stack.append((y3, x3))
        regm = np.zeros_like(sm)
        regm[y0b:y1b + 1, x0b:x1b + 1] = bb | ~reached
        regm &= sm
        clum = lum[si[regm]]
        cur = si[sel_e]
        if len(cur) >= 8 and (float(clum.max()) - float(clum.min()) < 60.0
                              or bool((clum < 140).all())):
            srgb2 = mrgb[y0:y0 + h, x0:x0 + w]
            roles = _eye_role_colors(si, sm, pal, sel_e, srgb2)
            slr = mlum[y0:y0 + h, x0:x0 + w][sel_e]
            q1, q2, q3 = np.percentile(slr, (45, 75, 88))
            med = float(np.median(slr))
            new = np.full(len(slr), roles["O"], dtype=si.dtype)
            new[slr >= q1] = roles["S"]
            new[slr >= q2] = roles["I"]
            hi_ok = (slr >= q3) & (slr > med + 20)
            new[hi_ok] = roles["H"]
            if fz is not None:
                keep = fz[sel_e]
                new[keep] = cur[keep]
            si[sel_e] = new
            # mirror the redeveloped eye onto the right side
            lys2, lxs2 = np.where(sel_e)
            for yy in range(int(lys2.min()) - 1, int(lys2.max()) + 2):
                for xx in range(int(lxs2.min()) - 1, int(lxs2.max()) + 2):
                    qx = axis2 - xx
                    if 0 <= yy < h and 0 <= xx < w and 0 <= qx < w \
                            and sm[yy, xx] and sm[yy, qx] \
                            and (fz is None
                                 or not (fz[yy, xx] or fz[yy, qx])):
                        si[yy, qx] = si[yy, xx]

    # 目ヂカラ mode: the eye regions keep their ORIGINAL
    # full colors -- the quantize/vote/outline passes flatten small eyes,
    # and the source's rich iris colors read better at dot scale. Written
    # to the overlay, composited over the palette image at the very end.
    so = ov_rgb[y0:y0 + h, x0:x0 + w] if ov_rgb is not None else None
    som = ov_m[y0:y0 + h, x0:x0 + w] if ov_m is not None else None
    if (eye_color == "original" and mrgb is not None and so is not None
            and tpl is None):
        srgb_full = mrgb[y0:y0 + h, x0:x0 + w]

        def _eye_box(sel):
            # the dark cluster is only the lash/frame line; iris, white
            # and glint live INSIDE its bounding box -- take the box
            ys5, xs5 = np.where(sel)
            box = np.zeros_like(sel)
            box[max(0, ys5.min() - 1):min(h, ys5.max() + 2),
                max(0, xs5.min()):xs5.max() + 1] = True
            box &= sm
            if fz is not None:
                box &= ~fz
            return box

        if asym:
            # 非対称顔: 各目 (眼帯含む) は自分自身の元色をそのまま
            for eye in (eL, eR):
                box = _eye_box(eye[3])
                so[box] = srgb_full[box]
                som[box] = True
        else:
            # 対称顔: マスター側の元色を、反対側へは鏡像コピー
            box = _eye_box(master[3])
            ys4, xs4 = np.where(box)
            for yy, xx in zip(ys4, xs4):
                so[yy, xx] = srgb_full[yy, xx]
                som[yy, xx] = True
                qx = axis2 - xx
                if 0 <= qx < w and sm[yy, qx] \
                        and (fz is None or not fz[yy, qx]):
                    so[yy, qx] = srgb_full[yy, xx]
                    som[yy, qx] = True

    # montage capture (user request: 下地ごと上書き): after the idle face
    # is fully finished, keep both eyes plus the surrounding skin base as
    # the character's face plate; every other frame pastes it wholesale.
    if tpl_lock is not None and tpl is None and "plate" not in tpl_lock:
        lys, lxs = np.where(eL[3])
        rys, rxs = np.where(eR[3])
        py0 = max(0, ey0 - 3)
        py1 = min(h - 1, ey1 + 4)
        px0 = max(0, int(lxs.min()) - 3)
        px1 = min(w - 1, int(rxs.max()) + 3)
        rect = si[py0:py1 + 1, px0:px1 + 1].copy()
        rmask = sm[py0:py1 + 1, px0:px1 + 1].copy()
        tpl_lock["plate"] = {
            "rect": rect, "rmask": rmask,
            "wmask": rmask & (lum[rect] >= 95),
            "py0": py0, "px0": px0,
            "orgb": (so[py0:py1 + 1, px0:px1 + 1].copy()
                     if so is not None else None),
            "om": (som[py0:py1 + 1, px0:px1 + 1].copy()
                   if som is not None else None),
        }
    return True


# ------------------------------------------------------------------ main

def to_image(idx: np.ndarray, mask: np.ndarray, pal: np.ndarray) -> Image.Image:
    rgb = pal.astype(np.uint8)[idx]
    out = np.dstack([rgb, np.where(mask, 255, 0).astype(np.uint8)])
    return Image.fromarray(out.astype(np.uint8), "RGBA")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--round-dir", required=True)
    ap.add_argument("--colors", type=int, default=24,
                    help="palette size when auto-extracting (16-32)")
    ap.add_argument("--palette-file", default=None,
                    help="fixed palette (.json or EDGE2 .pal); enables "
                         "ramp discipline (e.g. palettes/rtp_vx_256.json)")
    ap.add_argument("--quantize", choices=("dither", "legacy"),
                    default="dither",
                    help="dither (既定): 普通のディザ減色 (Floyd-"
                         "Steinberg)。縁取り以外の特殊処理・フルカラー"
                         "はめ込みなし; legacy: 従来の特殊減色+顔処理")
    ap.add_argument("--dither-strength", type=float, default=1.0,
                    help="ディザの強さ 0..1 (0=なし/ベタ塗り, 0.3=弱, "
                         "0.6=中, 1=フル・既定)。--quantize dither のみ")
    ap.add_argument("--pixel-scale", type=int, default=1,
                    help="1 = dot per T pixel; 2 = chunkier dots")
    ap.add_argument("--cell-w", type=int, default=64)
    ap.add_argument("--cell-h", type=int, default=128)
    ap.add_argument("--eye-lib", default="none",
                    help="eye template dir; 'auto' looks next to the app. "
                         "Default OFF -- symmetrizing the character's own "
                         "eyes beat template stamping (position/size fit)")
    ap.add_argument("--eye-style", default=None,
                    help="prefer this eye template (name substring), "
                         "e.g. round_cute / almond / fox; implies "
                         "--eye-lib auto")
    ap.add_argument("--eye-color", choices=("face", "original", "palette"),
                    default="face",
                    help="face (既定): 顔全体 (肌領域+内部の目・口) を"
                         "減色せず元のフルカラーで残す; original: 目"
                         "ボックスのみ元色 (目ヂカラ); palette: 全て"
                         "パレットに量子化 (従来)")
    a = ap.parse_args()
    if a.eye_style and a.eye_lib.lower() == "none":
        a.eye_lib = "auto"

    rd = Path(a.round_dir)
    lt = next(iter(rd.glob("*LT.png")), None)
    if lt is None:
        raise SystemExit(f"no *LT.png in {rd}")
    char_id = lt.name[:-len("LT.png")]
    src = Image.open(lt)
    factor = (src.height // (a.cell_h * 4)) * a.pixel_scale

    fixed_pal = None
    if a.palette_file:
        fixed_pal = load_palette_file(a.palette_file)
        print(f"fixed palette: {len(fixed_pal)} colors from "
              f"{Path(a.palette_file).name}")

    eye_templates: list[dict] = []
    if a.eye_lib.lower() != "none":
        if a.eye_lib == "auto":
            cands = [Path(sys.argv[0]).resolve().parent / "eyes",
                     TOOLS.parent / "eyes"]
            lib_dir = next((c for c in cands if c.is_dir()), None)
        else:
            lib_dir = Path(a.eye_lib)
        eye_templates = load_eye_lib(lib_dir)
        if eye_templates:
            print(f"eye templates: {len(eye_templates)} "
                  f"({', '.join(t['name'] for t in eye_templates)})")

    if a.quantize == "dither":
        # 普通のディザ減色: 縁取り以外の特殊処理なし・フルカラーはめ込みなし
        print(f"quantize: Floyd-Steinberg dither 強さ{a.dither_strength:g} "
              "(特殊減色・フルカラーはめ込みなし)")
        idx, mask, pal = pixelize_dither(src, a.colors, factor,
                                         fixed_pal=fixed_pal,
                                         dither=a.dither_strength)
        outline_pass(idx, mask, pal)  # 黒系1ドット縁取りは維持
        img = to_image(idx, mask, pal)
    else:
        idx, mask, pal, mlum, mrgb = pixelize(src, a.colors, factor,
                                              fixed_pal=fixed_pal)

        # ramp discipline for BOTH modes: designer palettes carry their
        # ramps in entry order; extracted palettes get ramps synthesized
        # from hue families ('色は維持してドット感だけ' -- the vivid
        # original colors with hand-dot shading structure)
        ramps = detect_ramps(pal) if fixed_pal is not None \
            else synth_ramps(pal)
        ramp_of = np.zeros(len(pal), dtype=np.int16)
        for k, r in enumerate(ramps):
            for i in r:
                ramp_of[i] = k
        whitelist_ramps(idx, mask, pal, ramp_of, ramps)
        rid = smooth_ramp_field(idx, mask, pal, ramp_of, ramps, mlum)
        retone_from_source(idx, mask, pal, rid, ramps, mlum)

        absorb_hanging_dots(idx, mask, pal)
        outline_pass(idx, mask, pal)

        # symmetrize LAST so absorb/outline cannot re-break pairs; the
        # contour ring is frozen (silhouette asymmetry is legitimate)
        pada = np.pad(mask, 1, mode="constant")
        frozen = mask & ~(pada[:-2, 1:-1] & pada[2:, 1:-1]
                          & pada[1:-1, :-2] & pada[1:-1, 2:])
        dcw, dch = a.cell_w // a.pixel_scale, a.cell_h // a.pixel_scale
        eye_lock: dict = {}
        dir_locks: dict = {d: {} for d in ("left", "right",
                                           "front_left", "front_right")}
        # 目ヂカラ overlay: full-color eye dots collected here and
        # composited over the palette image at the very end (no later
        # pass can crush them)
        ov_rgb = np.zeros((*idx.shape, 3), dtype=np.uint8)
        ov_m = np.zeros(idx.shape, dtype=bool)
        face_locks: dict = {d: {} for d in ("front", "left", "right",
                                            "front_left", "front_right")}
        for d, (row, block) in DIR_PLACEMENT.items():
            if d in ("back_left", "back_right"):
                continue
            for k in range(len(FRAME_NAMES)):
                cx0, cy0 = (block + k) * dcw, row * dch
                if a.eye_color == "face" and d != "back":
                    # 顔まるごと元色モード: 目検出に依存しない領域ベース
                    face_montage_fullcolor(idx, mask, pal, mlum,
                                           cx0, cy0, dcw, dch, frozen,
                                           face_locks[d], mrgb,
                                           ov_rgb, ov_m)
                elif d == "front":
                    face_stamp_symmetrize(idx, mask, pal, mlum,
                                          cx0, cy0, dcw, dch, frozen,
                                          eye_templates, a.eye_style,
                                          eye_lock, mrgb, ov_rgb, ov_m,
                                          a.eye_color)
                elif d == "back":
                    dot_symmetrize_src(idx, mask, pal, mlum,
                                       cx0, cy0, dcw, dch, frozen)
                else:
                    # profiles and front diagonals: idle donates its eye
                    # zone, walk frames take it (montage)
                    montage_direction(idx, mask, pal, mlum,
                                      cx0, cy0, dcw, dch, frozen,
                                      dir_locks[d], mrgb, ov_rgb, ov_m,
                                      a.eye_color)

        img = to_image(idx, mask, pal)
        if a.eye_color in ("original", "face") and ov_m.any():
            arr = np.array(img)
            sel_ov = ov_m & mask
            arr[..., :3][sel_ov] = ov_rgb[sel_ov]
            img = Image.fromarray(arr, "RGBA")
            print(f"目ヂカラ: {int(sel_ov.sum())} dots kept "
                  f"in original colors")
    if a.pixel_scale > 1:
        img = img.resize((img.width * a.pixel_scale,
                          img.height * a.pixel_scale), Image.NEAREST)

    out_dir = rd / "08_pixel_art"
    cells_dir = out_dir / "cells"
    out_dir.mkdir(exist_ok=True)
    sheet_p = out_dir / f"{char_id}T_pixel.png"
    img.save(sheet_p)
    img.resize((img.width * 2, img.height * 2),
               Image.NEAREST).save(out_dir / f"{char_id}T_pixel@2x.png")

    cw, ch = a.cell_w, a.cell_h
    for d, (row, block) in DIR_PLACEMENT.items():
        ddir = cells_dir / d
        ddir.mkdir(parents=True, exist_ok=True)
        for k, name in enumerate(FRAME_NAMES):
            c = (block + k) * cw
            img.crop((c, row * ch, c + cw, (row + 1) * ch)).save(
                ddir / f"{d}_{name}T.png")

    # ---- 巡回のテンプレ (t_spec 以外) もドット絵セルから組み直す ----
    # (2026-07-18ユーザー報告「ドット絵生成でT規格じゃないやつが適用され
    #  ない」: 従来はT_pixelとセルだけで、run_configのtemplate (wolf_8dir
    #  等) のシートは元絵のまま置き去りだった。本線と同じ
    #  templates_render をドット絵セルに当てて 08_pixel_art へ出力する)
    tpl_stem = ""
    rcp = rd / "run_config.json"
    if rcp.is_file():
        try:
            tpl_stem = str(json.loads(rcp.read_text(encoding="utf-8"))
                           .get("template") or "")
        except (OSError, json.JSONDecodeError):
            tpl_stem = ""
    if tpl_stem and tpl_stem != "t_spec":
        tpl_path = TOOLS.parent / "templates" / f"{tpl_stem}.json"
        if not tpl_path.is_file():
            tpl_path = (Path(getattr(sys, "_MEIPASS", str(TOOLS.parent)))
                        / "templates" / f"{tpl_stem}.json")
        if tpl_path.is_file():
            try:
                # walkA/walkB の検証済みペアがあれば本線と同じ選択にする
                hint = rd / "06_sheet_build" / "cells" / "walkAB.json"
                if hint.is_file():
                    import shutil
                    shutil.copy2(hint, cells_dir / "walkAB.json")
                from templates_render import render_template
                out = render_template(tpl_path, cells_dir, out_dir,
                                      f"{char_id}_pixel")
                print(f"PIXEL_TEMPLATE_SHEET: {out}")
            except Exception as e:  # noqa: BLE001
                # テンプレ再構成の失敗でドット絵本体を道連れにしない
                print(f"template re-render failed (T_pixelは有効): {e}")
        else:
            print(f"template not found (skipped): {tpl_stem}")

    used = int(len(np.unique(idx[mask])))
    (out_dir / "pixel_params.json").write_text(json.dumps(
        {"colors": a.colors, "pixel_scale": a.pixel_scale,
         "palette_file": a.palette_file, "colors_used": used,
         "source": lt.name, "factor": factor, "quantize": a.quantize,
         "dither_strength": a.dither_strength, "template": tpl_stem},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"colors used: {used}")
    print(f"PIXEL_SHEET: {sheet_p}")
    print(f"pixel cells -> {cells_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
