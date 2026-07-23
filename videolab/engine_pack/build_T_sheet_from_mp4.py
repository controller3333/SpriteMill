#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_T_sheet_from_mp4.py  (Claude seat, roundtable relay 2026-07-02)

Turn Grok's 8 walking MP4s (one per direction, magenta chroma-key background)
into index-format T sprite sheets:

    <char>_walkT.png    768 x 512    12 cols x 4 rows   cell 64 x 128   RGBA
    <char>_walkLT.png  3840 x 2560   12 cols x 4 rows   cell 320 x 640  RGBA

Pipeline per direction (SPRITE_SHEET_MASS_PRODUCTION_GOAL.md, Claude steps):
  1. ffmpeg-split the MP4 into frames.
  2. idle  = frame 0 (the static Codex reference the i2v was conditioned on).
  3. fA    = frame most different from idle (leg most forward / extreme pose).
  4. fB    = the later frame most similar to fA  -> one gait cycle = [fA, fB).
  5. pick 5 walk frames evenly across that cycle (natural 1-loop foot cadence).
  6. chroma-key magenta -> transparent (pure PIL: bg where min(R,B)-G >= thr).
  7. place idle + 5 walk into the T grid with a shared scale by default, then
     idle bbox-centered per cell so the 8 idle centers coincide.

Memory-safe: processes one MP4 at a time and never holds more than a couple of
full frames in RAM (scans are single-frame; only the 6 chosen frames are kept).

Layout (matches tools/inspect_T_sheet.py):
    rows  : left block(cols 0-5) front/left/right/back
            right block(cols 6-11) front_left/front_right/back_left/back_right
    cols  : [ idle | walk1 walk2 walk3 walk4 walk5 ] per 6-wide block
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

import numpy as np
from PIL import Image, ImageChops, ImageStat

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inspect_walk_mp4 import (  # noqa: E402
    bg_magenta_mask, direction_of, key_magenta_alpha, load_references,
    normalize_foreground, norm_diff,
)

COLS, ROWS = 12, 4

# direction -> (row, block_start_col)
DIR_PLACEMENT = {
    "front": (0, 0), "left": (1, 0), "right": (2, 0), "back": (3, 0),
    "front_left": (0, 6), "front_right": (1, 6),
    "back_left": (2, 6), "back_right": (3, 6),
}

_WINGET_FF = (r"C:\Users\contr\AppData\Local\Microsoft\WinGet\Packages"
              r"\Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe"
              r"\ffmpeg-8.1.1-essentials_build\bin\ffmpeg.exe")
DEFAULT_FF = _WINGET_FF if os.path.isfile(_WINGET_FF) else "ffmpeg"


# --------------------------------------------------------------------- keying
def key_alpha(rgb, thr):
    """Return an L-mode alpha: opaque where the pixel is NOT magenta
    background. 連結成分キーイング (bg_magenta_mask) -- キャラ内部の
    マゼンタ寄りの色 (髪のメッシュ等) は抜かない。"""
    return ImageChops.invert(bg_magenta_mask(rgb, thr))


_ALPHA_CACHE: dict = {}


def load_keyed(path, thr):
    """キー済みRGBAを返す。アルファはパス別にキャッシュ: ビルダーは
    コマ選定・向き検査・サイズ計測で同じフレームを複数回読むため、
    連結キーイング (数十ms/回) を都度やり直すと数分の後退になる。

    キーにはmtime+sizeを含める(2026-07-05 村人A事故): 凍結EXEはツールを
    インプロセス実行するため、検査FAIL→同名パスでコマ再生成→再ビルドの
    リトライで旧コマのαが新コマのRGBに適用され、旧ポーズ形の不透明
    マゼンタ残像がシートに焼き込まれた。"""
    im = Image.open(path).convert("RGB")
    try:
        st = os.stat(path)
        k = (path, thr, st.st_mtime_ns, st.st_size)
    except OSError:
        k = (path, thr, None, None)
    a = _ALPHA_CACHE.get(k)
    if a is None:
        a = key_alpha(im, thr)
        if len(_ALPHA_CACHE) >= 320:  # ~2方向ぶん (アルファLのみ約100MB)
            _ALPHA_CACHE.clear()
        _ALPHA_CACHE[k] = a
    out = im.convert("RGBA")
    out.putalpha(a)
    # ★デスピル (2026-07-23): アルファを立てるだけではRGBのマゼンタ被りが
    # 残り、輪郭1pxと細い髪束がピンクのままシートへ焼き込まれる。実測で
    # sheet_c9.png の縁の純マゼンタが 200px -> 7px。SM_DESPILL=off で無効
    if os.environ.get("SM_DESPILL", "on").strip().lower() not in (
            "0", "off", "false", "no"):
        try:
            from inspect_walk_mp4 import despill_magenta
            out = despill_magenta(out, a)
        except Exception:                     # noqa: BLE001
            pass                              # 効かなくてもシートは作る
    return out


def mean_diff(a, b):
    d = ImageChops.difference(a, b)
    s = sum(ImageStat.Stat(d).sum)
    n = a.size[0] * a.size[1] * len(a.getbands())
    return s / n if n else 0.0


# ----------------------------------------------------------------- selection
def orientation_clean_mask(frame_paths, direction, refs, thr):
    """clean[i] = frame i matches its own direction reference best.

    i2v walk cycles tend to yaw the body a few degrees off the reference
    facing mid-clip. Frame picking maximizes difference-from-idle, which
    preferentially selects those yawed frames -- so restrict candidates to
    frames whose best-matching reference is still their own direction.
    """
    own = refs.get(direction)
    if own is None:
        return None, None
    others = [v for k, v in refs.items() if k != direction]
    if not others:
        return None, None
    clean, lean = [], []
    for fp in frame_paths:
        n = normalize_foreground(Image.open(fp), thr)
        d_own = norm_diff(n, own)
        d_best_other = min(norm_diff(n, o) for o in others)
        lean.append(round(d_own - d_best_other, 3))  # negative = own is best
        clean.append(d_own <= d_best_other)
    return clean, lean


def _snap_clean(idx, clean, lo, hi, radius=4):
    """Move idx to the nearest orientation-clean index within radius."""
    if clean is None or (0 <= idx < len(clean) and clean[idx]):
        return idx
    for off in range(1, radius + 1):
        for cand in (idx + off, idx - off):
            if lo <= cand <= hi and cand < len(clean) and clean[cand]:
                return cand
    return idx


def _thumb(path, thr):
    im = load_keyed(path, thr).convert("RGB")
    im.thumbnail((96, 128))
    return im


def _tdiff(a, b):
    if a.size != b.size:
        b = b.resize(a.size)
    d = ImageChops.difference(a, b)
    s = sum(ImageStat.Stat(d).sum)
    return s / (a.size[0] * a.size[1] * 3)


def _estimate_period(thumbs, lo, hi, a_lo):
    """Gait period via self-similarity: argmin_p mean_i d(f[i], f[i+p])."""
    n = len(thumbs)
    idxs = list(range(a_lo, min(n - hi - 1, a_lo + 70), 3))
    best_p, best_v = None, None
    for p in range(lo, hi + 1):
        vals = [_tdiff(thumbs[i], thumbs[i + p]) for i in idxs]
        v = sum(vals) / len(vals)
        if best_v is None or v < best_v:
            best_p, best_v = p, v
    return best_p


def _best_cycle_start(thumbs, period, a_lo, clean, cdiffs, rank=0):
    """Pick the segment whose motion repeats best one period later, sampling
    frames by MOTION arc-length rather than by time.

    A clean gait loops: pose at s+k matches pose at s+k+P. Stutters break
    that repetition (high residual E). Separately, anime-style i2v often
    DWELLS at the stride extreme then snaps -- equal-time sampling grabs the
    dwell twice and skips the in-betweens ("warp"). Equal-arc sampling picks
    frames at equal cumulative-motion steps, so a held pose contributes one
    frame and the fast transition keeps its in-betweens. The remaining
    unevenness of the picked frames (pick_uniformity U) then measures
    whether in-betweens exist in the footage at all.
    """
    n = len(thumbs)
    hi_s = n - 2 * period - 1
    prefix = [0.0]
    for v in cdiffs:
        prefix.append(prefix[-1] + v)

    def arc(i, j):
        return prefix[j] - prefix[i]

    def picks_for(s):
        total = arc(s, s + period)
        if total <= 0:
            return [s + round(k * period / 5) for k in range(5)]
        picks, j = [], s
        for k in range(5):
            tgt = total * k / 5
            while j < s + period - 1 and arc(s, j) < tgt:
                j += 1
            picks.append(j)
        for i in range(1, 5):
            if picks[i] <= picks[i - 1]:
                picks[i] = min(picks[i - 1] + 1, s + period - 1)
        return picks

    def score(require_clean):
        out = []
        for s in range(a_lo, max(a_lo + 1, hi_s), 2):
            picks = picks_for(s)
            if require_clean and clean is not None and any(
                    p < len(clean) and not clean[p] for p in picks):
                continue
            E = sum(_tdiff(thumbs[p], thumbs[p + period])
                    for p in picks) / len(picks)
            adj = [_tdiff(thumbs[picks[i]], thumbs[picks[i + 1]])
                   for i in range(4)]
            adj.append(_tdiff(thumbs[picks[4]], thumbs[picks[0]]))  # wrap
            A = sum(adj) / len(adj)
            med = sorted(adj)[len(adj) // 2]
            U = max(adj) / max(med, 0.05)
            out.append((E / max(A, 0.05), E, A, s, picks, U))
        return out

    results = score(True) or score(False)
    if not results:
        return None
    amax = max(r[2] for r in results)
    eligible = [r for r in results if r[2] >= 0.5 * amax] or results
    # prefer smooth (no-warp) segments first, then best periodicity
    eligible.sort(key=lambda r: (r[5] > 2.8, r[0]))
    if rank <= 0:
        return eligible[0]
    # rank > 0 = the free re-pick ("puzzle") retry: when the final sheet
    # rejects the first choice, hand out a genuinely DIFFERENT cycle from
    # the same footage. Adjacent starts (step 2) are near-duplicates, so
    # distinct picks must sit at least half a period apart.
    picked: list = []
    for r in eligible:
        if all(abs(r[3] - q[3]) >= max(period // 2, 3) for q in picked):
            picked.append(r)
    return picked[min(rank, len(picked) - 1)]


def select_frames(frame_paths, thr, cy_lo=14, cy_hi=40, a_lo=6, clean=None,
                  rank=0, period=None, start=0, end=0):
    """idle=frame0; estimate the gait period, then sample 5 frames from the
    most consistently-repeating cycle in the clip (see _best_cycle_start).

    With `clean` (orientation mask), candidate segments and the sampled walk
    frames are constrained/snapped to frames that still face the right way.
    `rank` selects the rank-th DISTINCT candidate cycle (free re-pick when
    the sheet inspector rejects the previous choice).
    `period` (float): 自己相関推定をスキップして既知の歩行周期を使う。
    VACE骨格駆動の動画は周期が既知な上、左右対称の
    骨格歩行+上下動(半周期で反復)のせいで自己相関が半周期にロックし
    「片足だけ出す」シートになる (2026-07-13実障害) — 既知周期の明示が根治。
    `start` (int): 歩行が始まるフレーム番号。骨格動画の先頭には直立区間
    (pose_video.walk_layout) があり、静止区間は残差ゼロで周期スコアが
    最良になってしまう — 探索と歩行コマの下限を start に引き上げて
    直立コマ (frame 0..start-1) を歩行選出から締め出す。
    `end` (int): 歩行が終わるフレーム番号 (これ以降は末尾静止)。先頭直立と
    対称に、探索・スナップ・重複解消の上限を end に切り下げて末尾静止を
    歩行コマから締め出す (残差計算 thumbs[p+period] が静止フレームと
    歩行ポーズを比較して汚染される越境も防ぐ)。0=末尾静止なし。
    Falls back to the legacy extreme-pose picker on very short clips.
    """
    n = len(frame_paths)
    # 末尾静止つきクリップは「歩行終端+1」を実効クリップ長として扱う
    # (以降の探索・スナップ・残差が全て末尾静止を見ない)
    if end and 0 < int(end) + 1 < n:
        n = int(end) + 1
    forced = bool(period)
    # 直立コマ: 骨格駆動 (start>0) では直立区間の最後のフレームを採用する。
    # f0はVACE経路の既知の「先頭だけ暗い」段差の上、素材ごとに色差が違う
    # (2026-07-13実測: 髪+27G/体+12G) ため単一アフィンのカラーアンカーでは
    # 歩行コマと揃わない。直立区間の最後=立ち姿のまま安定パレットへ回復
    # 済みの標本で、これを直立コマにすると色が歩行コマと構造的に揃う。
    idle_pick = max(0, min(n - 1, int(start or 0) - 1))
    a_lo = max(a_lo, int(start or 0))
    lo_pick = max(1, int(start or 0))
    if forced:
        period = max(4, int(round(float(period))))
        # 推定不要なら探索余白は要らない: _best_cycle_start が動く最小長
        # (81f動画は従来ゲートn>=90未満で常にレガシー選出に落ちていた=
        #  半周期ロックの温床。既知周期ならここで周期パスに乗せる)。
        # ★+2ではなく+1: walk_layoutの2周期ちょうど割り (n=a_lo+2p+1、
        # 49f=直立6+21f×2周期) が常に1フレーム差でsingle_cycleへ落ち、
        # 第2周期が5点残差検証に使われなかった (2026-07-13検証で発見)。
        # n=2p+a_lo+1ならs=a_loの全ピックでp+periodが範囲内に収まる。
        gate = 2 * period + a_lo + 1
    else:
        gate = a_lo + 2 * cy_hi + 4
    # 33f既定は「直立6f + 歩行1周期26f + 終端」の最小クリップ。
    # _best_cycle_start は周期残差を測るため2周期を要求するが、骨格側から
    # start/periodが確定している場合は1周期そのものを等移動量で5分割できる。
    # これでシートに不要な2周期を生成せず、時間と外見ドリフトを減らす。
    single_cycle = (forced and n >= a_lo + period + 1 and n < gate)
    if single_cycle:
        thumbs = [_thumb(p, thr) for p in frame_paths[:n]]
        end = min(n - 1, a_lo + period)
        cdiffs = [_tdiff(thumbs[i], thumbs[i + 1])
                  for i in range(a_lo, end)]
        prefix = [0.0]
        for v in cdiffs:
            prefix.append(prefix[-1] + v)
        total = prefix[-1]
        phase = min(max(int(rank), 0), 2) * 0.10
        picks = []
        j = 0
        for k in range(5):
            target = total * (k + phase) / 5.0
            while j < len(cdiffs) - 1 and prefix[j] < target:
                j += 1
            picks.append(a_lo + j)
        if total <= 0:
            picks = [a_lo + round((k + phase) * period / 5.0)
                     for k in range(5)]
        picks = [_snap_clean(idx, clean, lo_pick, end - 1)
                 for idx in picks]
        # cleanスナップが同じフレームへ寄っても、5コマは必ず別にする。
        used: set[int] = set()
        fixed = []
        candidates = list(range(lo_pick, end))
        for idx in picks:
            available = [v for v in candidates if v not in used]
            if not available:
                break
            idx = min(available, key=lambda v: abs(v - idx))
            used.add(idx)
            fixed.append(idx)
        fixed.sort()
        if len(fixed) == 5:
            adj = [_tdiff(thumbs[fixed[i]], thumbs[fixed[i + 1]])
                   for i in range(4)]
            adj.append(_tdiff(thumbs[fixed[4]], thumbs[end]))
            amplitude = sum(adj) / len(adj)
            median = sorted(adj)[len(adj) // 2]
            residual = _tdiff(thumbs[a_lo], thumbs[end])
            return {"idle": idle_pick, "fa": a_lo, "fb": end, "walk": fixed,
                    "method": "periodic_v4_single_cycle",
                    "period": period, "period_forced": True,
                    "cycle_residual": round(residual, 3),
                    "amplitude": round(amplitude, 3),
                    "gait_ratio": round(residual / max(amplitude, 0.05), 3),
                    "pick_uniformity": round(
                        max(adj) / max(median, 0.05), 3),
                    "pick_rank": rank,
                    "orientation_filtered": bool(clean) and not all(clean)}
    if n >= gate:
        thumbs = [_thumb(p, thr) for p in frame_paths[:n]]
        if not forced:
            period = _estimate_period(thumbs, cy_lo, cy_hi, a_lo)
        cdiffs = [_tdiff(thumbs[i], thumbs[i + 1]) for i in range(n - 1)]
        best = _best_cycle_start(thumbs, period, a_lo, clean, cdiffs,
                                 rank=rank)
        if best is not None:
            ratio, E, A, s, picks, uniformity = best
            walk = [_snap_clean(idx, clean, lo_pick, n - 1) for idx in picks]
            seen, fixed = set(), []
            for idx in walk:
                idx = max(lo_pick, min(n - 1, idx))
                while idx in seen and idx < n - 1:
                    idx += 1
                seen.add(idx)
                fixed.append(idx)
            return {"idle": idle_pick, "fa": s, "fb": s + period,
                    "walk": fixed,
                    "method": "periodic_v4_arclen", "period": period,
                    "period_forced": forced,
                    "cycle_residual": round(E, 3),
                    "amplitude": round(A, 3),
                    "gait_ratio": round(ratio, 3),
                    "pick_uniformity": round(uniformity, 3),
                    "pick_rank": rank,
                    "orientation_filtered": bool(clean) and not all(clean)}
    return _select_frames_legacy(frame_paths[:n], thr, cy_lo, cy_hi, a_lo,
                                 clean, idle_pick=idle_pick)


def _select_frames_legacy(frame_paths, thr, cy_lo=14, cy_hi=40, a_lo=6,
                          clean=None, idle_pick=0):
    """Original extreme-pose picker (fallback for short clips)."""
    n = len(frame_paths)
    idle = load_keyed(frame_paths[idle_pick], thr)

    def is_clean(i):
        return clean is None or (i < len(clean) and clean[i])

    a_hi = min(n // 2 + 5, n - cy_lo - 1)
    a_hi = max(a_hi, a_lo + 1)
    candidates = [i for i in range(a_lo, a_hi) if is_clean(i)]
    orientation_filtered = clean is not None and len(candidates) < (a_hi - a_lo)
    if not candidates:
        candidates = list(range(a_lo, a_hi))
        orientation_filtered = False
    best_a, best_a_val = candidates[0], -1.0
    for i in candidates:
        f = load_keyed(frame_paths[i], thr)
        d = mean_diff(f, idle)
        if d > best_a_val:
            best_a_val, best_a = d, i
    fa = best_a

    hi = min(fa + cy_hi, n - 1)
    lo = min(fa + cy_lo, n - 1)
    fb, fb_val, method = None, 1e9, "cycle"
    if hi > lo:
        fa_img = load_keyed(frame_paths[fa], thr)
        b_candidates = [j for j in range(lo, hi + 1) if is_clean(j)]
        if not b_candidates:
            b_candidates = list(range(lo, hi + 1))
        for j in b_candidates:
            f = load_keyed(frame_paths[j], thr)
            d = mean_diff(f, fa_img)
            if d < fb_val:
                fb_val, fb = d, j
    if fb is None or fb - fa < 5:
        # fallback: evenly space 5 across the safe middle span
        method = "even_fallback"
        lo2, hi2 = a_lo, n - a_lo
        walk = [round(lo2 + k * (hi2 - lo2) / 5.0) for k in range(5)]
    else:
        walk = [round(fa + k * (fb - fa) / 5.0) for k in range(5)]
    walk = [_snap_clean(idx, clean, 1, n - 1) for idx in walk]

    # make strictly increasing & in-range & distinct
    seen, fixed = set(), []
    for idx in walk:
        idx = max(1, min(n - 1, idx))
        while idx in seen and idx < n - 1:
            idx += 1
        seen.add(idx)
        fixed.append(idx)
    return {"idle": idle_pick, "fa": fa, "fb": fb, "walk": fixed,
            "method": method,
            "fa_val": round(best_a_val, 3), "fb_val": round(fb_val, 3),
            "orientation_filtered": orientation_filtered}


# --------------------------------------------------------------- cell builder
def fit_scale(max_w, max_h, cell_w, cell_h, margin):
    return min((cell_w - 2 * margin) / max_w, (cell_h - 2 * margin) / max_h)


def head_height_px(keyed_rgba):
    """Head height (bbox top -> neck notch) in source-frame pixels.

    The neck is the first row (below the top 15%, within 20-68% of the
    body) whose width drops under 55% of the running max width. Chibi
    characters read as same-sized when their HEADS match, not their
    bounding boxes -- wide coats/sleeves shrink a max-fit character."""
    a = keyed_rgba.getchannel("A")
    bb = a.getbbox()
    if not bb:
        return None
    crop = a.crop(bb)
    scale = 1.0
    if crop.size[1] > 240:
        scale = 240 / crop.size[1]
        crop = crop.resize((max(1, round(crop.size[0] * scale)), 240))
    w, h = crop.size
    px = crop.load()
    maxw = 0
    for y in range(h):
        row_l = row_r = None
        for x in range(w):
            if px[x, y] > 16:
                if row_l is None:
                    row_l = x
                row_r = x
        rw = (row_r - row_l + 1) if row_l is not None else 0
        fy = y / h
        if (fy > 0.15 and maxw > 0 and rw < 0.55 * maxw
                and 0.2 <= fy <= 0.68):
            # plausibility bound: a chibi head is 22-48% of body height.
            # Wide sleeves keep the width above the 55% line all the way
            # to the HAKAMA WAIST (C22: "neck" at 54% -> head=310px ->
            # the whole character shrank); such a hit is a costume notch,
            # not a neck -- better no anchor than a wrong one.
            if not (0.22 <= fy <= 0.48):
                return None
            return y / scale
        maxw = max(maxw, rw)
    return None


def stretch_below_neck(img, f: float, keep_frac: float = 0.45):
    """Vertically stretch (f>1) or squash (f<1) the body region below
    `keep_frac` of the figure, keeping the head untouched -- the final3
    recipe (身長は首下のみ伸縮、頭は聖域) ported to video cells. Lifts
    directions drawn shorter (C22 diagonals) and re-equalizes height
    after a manual per-direction 頭身 rescale (--dir-scale)."""
    if abs(f - 1.0) <= 0.001:
        return img
    a2 = img.getchannel("A")
    bb = a2.getbbox()
    if not bb:
        return img
    y0, y1 = bb[1], bb[3]
    split = y0 + int((y1 - y0) * keep_frac)
    lower = img.crop((0, split, img.width, img.height))
    new_h = max(1, int(round(lower.height * f)))
    lower = lower.resize((img.width, new_h), Image.LANCZOS)
    out = Image.new("RGBA", (img.width, split + new_h), (0, 0, 0, 0))
    out.paste(img.crop((0, 0, img.width, split)), (0, 0))
    out.paste(lower, (0, split))
    return out


def head_hump_metric(keyed_rgba):
    """(head_h, head_w) via the width profile's head hump: the head's max
    width in the top 45% and the first LOCAL DIP below it (head bottom /
    neck). A local minimum cannot slide to the hakama waist the way the
    55%-of-running-max notch did, and long hair without any dip simply
    returns None (no anchor beats a wrong anchor).

    Calibrated 2026-07-04 against human-tuned C27 (fr=1.25, fl/bl/br=1.08):
    auto measured 1.23 / 1.06 / 1.07 / 1.08 -- within 2%."""
    a = keyed_rgba.getchannel("A")
    bb = a.getbbox()
    if not bb:
        return None
    crop = a.crop(bb)
    w, h = crop.size
    if h < 40:
        return None
    px = crop.load()
    W = [0.0] * h
    for y in range(h):
        row_l = row_r = None
        for x in range(w):
            if px[x, y] > 16:
                if row_l is None:
                    row_l = x
                row_r = x
        W[y] = (row_r - row_l + 1) if row_l is not None else 0.0
    k = max(1, h // 100)
    Ws = [sum(W[max(0, y - k):y + k + 1]) / len(W[max(0, y - k):y + k + 1])
          for y in range(h)]
    top45 = int(h * 0.45)
    hm_i = max(range(top45), key=lambda y: Ws[y])
    hm = Ws[hm_i]
    lim = int(h * 0.60)
    for y in range(hm_i + 1, lim):
        if Ws[y] <= Ws[y - 1] and (y + 1 >= lim or Ws[y] <= Ws[y + 1]) \
                and Ws[y] < hm * 0.85:
            ahead = Ws[y + 1:min(lim, y + max(3, h // 40))]
            if ahead and max(ahead) > Ws[y] * 1.04:
                return float(y), float(max(Ws[:y]))
    return None


def cap_area_metric(keyed_rgba, band_frac: float = 0.28):
    """Head-size proxy robust to fluffy hair and hats: sqrt of the
    foreground area inside the top `band_frac` of the tight crop.

    The neck-notch metric returns None when hair hides the neck
    (ウルファール: all 8 directions) and wobbles on side views (C21
    left/right read 27% high); the cap-band area tracks the perceived
    head/hat mass instead. Calibration 2026-07-03: characters the user
    accepts as uniform (C17, C21) measure within ±2% across directions;
    ウルファール's front sat 11-15% under its sides -- exactly the
    complaint."""
    a = keyed_rgba.getchannel("A")
    bb = a.getbbox()
    if not bb:
        return None
    crop = a.crop(bb)
    w, h = crop.size
    band = max(1, int(h * band_frac))
    px = crop.load()
    area = sum(1 for y in range(band) for x in range(w) if px[x, y] > 16)
    return area ** 0.5 if area else None


def selected_bbox_size(frame_paths, picks, thr):
    max_w, max_h = 1, 1
    for idx in picks:
        im = load_keyed(frame_paths[idx], thr)
        bb = im.getchannel("A").getbbox()
        if not bb:
            continue
        max_w = max(max_w, bb[2] - bb[0])
        max_h = max(max_h, bb[3] - bb[1])
    return max_w, max_h


def _torso_axis_x(im):
    """体軸x = 前景上部20%(頭バンド)のアルファ重心。ポーズ(脚の開き)に
    不変な横アンカー (hybrid_walk._head_cx と同じ発想)。ソース座標で返す。"""
    a = np.asarray(im.getchannel("A"))
    ys, _ = np.where(a > 16)
    if len(ys) == 0:
        return None
    y0 = int(ys.min())
    ch = int(ys.max()) - y0 + 1
    cols = (a[y0:y0 + max(1, int(0.20 * ch))] > 16).sum(axis=0).astype(float)
    tot = cols.sum()
    if tot <= 0:
        return None
    return float((cols * np.arange(len(cols))).sum() / tot)


def build_cells(frame_imgs, cell_w, cell_h, margin, scale=None):
    """frame_imgs = [idle, w1..w5] keyed RGBA (full frame). Return 6 cells,
    all sharing one scale+offset derived so the IDLE bbox centers at cell mid.

    体軸そろえ (2026-07-16「ウディタ形式で前後がくがく」実測対策):
    生成動画は直立区間=参照+latent固定に、歩行区間=骨格に錨止めされる
    ため、動画内の立ち位置がidle vs 歩行で2〜3px割れる (真ロップ実測:
    front +2.3 / front_right -2.5px。骨格軸は足元アンカー化で根治済みだが
    生成ドリフト分は残る)。3パターン形式 (walkA→idle→walkB→idle) はidleを
    1コマおきに挟むので、この差がそのまま毎フレームの横ホップ=体軸ブレに
    なる。対策=共有スケール・共有オフセットは維持したまま、各コマの体軸
    (頭バンド重心=ポーズ不変) をidleの体軸へ整数pxだけ横補正する
    (±4pxクランプ=誤計測の暴走防止)。"""
    boxes = [im.getchannel("A").getbbox() for im in frame_imgs]
    max_w = max(bb[2] - bb[0] for bb in boxes)
    max_h = max(bb[3] - bb[1] for bb in boxes)
    s = scale if scale is not None else fit_scale(max_w, max_h, cell_w, cell_h, margin)

    fw, fh = frame_imgs[0].size
    sw, sh = max(1, round(fw * s)), max(1, round(fh * s))
    # idle scaled center -> cell center
    ib = boxes[0]
    icx = (ib[0] + ib[2]) / 2.0 * s
    icy = (ib[1] + ib[3]) / 2.0 * s
    ox = round(cell_w / 2.0 - icx)
    oy = round(cell_h / 2.0 - icy)
    ax0 = _torso_axis_x(frame_imgs[0])

    cells, overflow = [], 0
    for im in frame_imgs:
        dx = 0
        if ax0 is not None:
            axi = _torso_axis_x(im)
            if axi is not None:
                dx = max(-4, min(4, round((ax0 - axi) * s)))
        scaled = im.resize((sw, sh), Image.LANCZOS)
        cell = Image.new("RGBA", (cell_w, cell_h), (0, 0, 0, 0))
        cell.alpha_composite(scaled, (ox + dx, oy))
        cells.append(cell)
        bb = im.getchannel("A").getbbox()
        # track worst-case clip beyond cell edges (scaled coords)
        l = ox + dx + bb[0] * s
        r = ox + dx + bb[2] * s
        t = oy + bb[1] * s
        bo = oy + bb[3] * s
        overflow = max(overflow, -l, r - cell_w, -t, bo - cell_h, 0)
    return cells, round(s, 4), (ox, oy), round(overflow, 2)


# --------------------------------------------------------------------- driver
def discover(mp4_dir):
    pat = re.compile(r"_(\d+)_(?P<dir>.+?)_walkT?\.mp4$", re.IGNORECASE)
    found = {}
    for p in sorted(glob.glob(os.path.join(mp4_dir, "*.mp4"))):
        m = pat.search(os.path.basename(p))
        if m and m.group("dir") in DIR_PLACEMENT:
            found[m.group("dir")] = p
    return found


def extract(ff, mp4, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # 残骸掃除: 前回の長いクリップのf*.pngが残っていると、短い新クリップの
    # 再抽出後に旧コマが混入する(インプロセス再実行・レジューム対策)
    for old in glob.glob(os.path.join(out_dir, "f*.png")):
        os.remove(old)
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", mp4, "-vsync", "0",
                    os.path.join(out_dir, "f%04d.png")], check=True,
                   creationflags=0x08000000 if sys.platform == "win32" else 0)
    return sorted(glob.glob(os.path.join(out_dir, "f*.png")))


def paste_block(sheet, cells, row, block, cw, ch):
    for i, cell in enumerate(cells):
        sheet.alpha_composite(cell, ((block + i) * cw, row * ch))


def main():
    # インプロセス再実行(凍結EXE)対策: 前回実行のα持ち越しを捨てる
    _ALPHA_CACHE.clear()
    ap = argparse.ArgumentParser()
    ap.add_argument("--mp4-dir", default=None)
    ap.add_argument("--frames-dir", default=None,
                    help="use pre-made frame PNGs instead of videos: "
                         "<dir>/<direction>/*.png sorted = idle + walk "
                         "frames (magenta background). Skips extraction, "
                         "orientation cleaning and period selection.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--char", default="C01")
    ap.add_argument("--thr", type=int, default=70)
    ap.add_argument("--ffmpeg", default=DEFAULT_FF)
    ap.add_argument("--keep-frames", action="store_true")
    ap.add_argument("--scale-mode", choices=("global", "per-direction"),
                    default="global",
                    help="global keeps all directions at one pixel scale; "
                         "per-direction preserves the legacy auto-fit behavior")
    ap.add_argument("--reference-dir",
                    help="split_centered direction references; when given, "
                         "walk frames are picked only from frames that still "
                         "match their own facing (prevents mid-clip yaw from "
                         "being selected)")
    ap.add_argument("--gait-ratio-max", type=float, default=1.2,
                    help="flag a direction when even its best cycle has "
                         "residual/amplitude above this (the walk motion "
                         "never repeats cleanly -> stutter/irregular gait). "
                         "Clean videos measure 0.06-0.79; ~1.0+ means the "
                         "motion is essentially aperiodic")
    ap.add_argument("--cell-w", type=int, default=64,
                    help="sheet cell width (LT = 5x); use wider cells for "
                         "quadrupeds etc.")
    ap.add_argument("--cell-h", type=int, default=128,
                    help="sheet cell height (LT = 5x)")
    ap.add_argument("--scale-anchor", choices=("fit", "head"), default="fit",
                    help="fit: max bbox fills the cell (legacy). head: scale "
                         "so the character's head height hits --head-frac of "
                         "the cell (uniform look across a character set), "
                         "capped by fit so nothing overflows")
    ap.add_argument("--head-frac", type=float, default=0.34,
                    help="head-anchor target: head height as a fraction of "
                         "cell height (0.34 is reachable by wide-costume "
                         "characters within the overflow allowance)")
    ap.add_argument("--max-overflow", type=float, default=0.12,
                    help="head anchor may exceed the all-frames fit by this "
                         "fraction: extreme WALK poses may clip slightly at "
                         "the cell edge, the IDLE pose never clips")
    ap.add_argument("--pick-uniformity-max", type=float, default=2.8,
                    help="flag a direction when the arc-length-sampled "
                         "frames still step unevenly (max/median adjacent "
                         "diff above this -> footage lacks in-betweens, "
                         "pose-dwell-then-snap timing)")
    ap.add_argument("--pick-rank", default="",
                    help="per-direction alternate-cycle choice, e.g. "
                         "'front=1,back_left=2': use the rank-th DISTINCT "
                         "candidate cycle from the same footage (free "
                         "re-pick instead of regenerating the video)")
    ap.add_argument("--gait-period", type=float, default=0.0,
                    help="既知の歩行周期(フレーム数)を明示して自己相関推定を"
                         "スキップ。VACE骨格駆動の動画は81f=3周期が既知で、"
                         "対称歩行+上下動が半周期にロックする誤検出の根治 "
                         "(0=従来の自動推定)")
    ap.add_argument("--gait-period-dirs", default="all",
                    help="--gait-periodを適用する方向 (カンマ区切り or all)")
    ap.add_argument("--gait-start", type=int, default=0,
                    help="歩行が始まるフレーム番号 (骨格動画の先頭直立区間の"
                         "長さ=pose_video.walk_layoutのidle_n)。直立区間は"
                         "残差ゼロで周期スコア最良になるため、歩行コマ探索"
                         "から除外する (--gait-period-dirs と同じ方向に適用)")
    ap.add_argument("--gait-end", type=int, default=0,
                    help="歩行が終わるフレーム番号 (walk_layoutの歩行終端="
                         "idle+cycles*period)。これ以降は末尾静止 (立ち"
                         "止まり) 区間で、歩行コマ探索から除外し、直立コマは"
                         "末尾静止窓 [gait_end+2, n) から実測選出する "
                         "(2026-07-17「立ち姿シーンを直立コマに」: 末尾静止は"
                         "歩行コマと同じ骨格アンカーに立つため、参照と骨格が"
                         "ズレたキャラでも直立↔歩行の継ぎ目が出ない)。"
                         "0=末尾静止なし (直立コマは従来どおり先頭窓)")
    ap.add_argument("--head-norm-max", type=float, default=0.15,
                    help="per-direction head-size normalization clamp "
                         "(scale-anchor head only): equalize the top-band "
                         "cap metric to the median within ±this fraction; "
                         "0 disables")
    ap.add_argument("--height-norm-max", type=float, default=0.08,
                    help="per-direction height rescue (scale-anchor head "
                         "only): stretch a short direction's below-neck "
                         "region up to +this fraction toward the median "
                         "idle height (enlarge-only, head untouched); "
                         "0 disables")
    ap.add_argument("--dir-scale", default="",
                    help="manual per-direction 頭身 knob, e.g. "
                         "'front_right=1.15': uniformly rescale that "
                         "direction's figure; the height equalizer then "
                         "squashes/stretches below the neck so only the "
                         "head-to-body ratio changes")
    ap.add_argument("--auto-dir-scale", type=int, default=1,
                    help="automatic per-direction 頭身 equalization via the "
                         "head-hump dip metric (1=on, 0=off). Enlarge-only, "
                         "deadband 5%%, cap 1.10, and fires on the FRONT "
                         "DIAGONALS only -- the one direction class where "
                         "the metric survived full-set human calibration "
                         "(hair/scarf silhouettes break it elsewhere); "
                         "manual --dir-scale directions are never touched "
                         "and handle everything the auto path declines")
    ap.add_argument("--force-picks", default="",
                    help="explicit walk frames per direction, e.g. "
                         "'front=78:84:90:95:101' (5 frame indices, idle "
                         "stays frame 0; directions separated by commas). "
                         "Bypasses automatic selection -- the manual "
                         "escape hatch for blink/artifact frames")
    a = ap.parse_args()
    pick_rank: dict[str, int] = {}
    for tok in (t for t in a.pick_rank.split(",") if t.strip()):
        d, _, k = tok.partition("=")
        try:
            pick_rank[d.strip()] = int(k)
        except ValueError:
            raise SystemExit(f"--pick-rank: bad token {tok!r}")
    dir_scale: dict[str, float] = {}
    for tok in (t for t in a.dir_scale.split(",") if t.strip()):
        d, _, k = tok.partition("=")
        try:
            dir_scale[d.strip()] = float(k)
        except ValueError:
            raise SystemExit(f"--dir-scale: bad token {tok!r}")
    force_picks: dict[str, list[int]] = {}
    for tok in (t for t in a.force_picks.split(",") if t.strip()):
        d, _, k = tok.partition("=")
        try:
            idxs = [int(x) for x in k.split(":")]
        except ValueError:
            raise SystemExit(f"--force-picks: bad token {tok!r}")
        if len(idxs) != 5:
            raise SystemExit(f"--force-picks: need exactly 5 frames ({tok!r})")
        force_picks[d.strip()] = idxs

    refs = {}
    if a.reference_dir:
        from pathlib import Path as _P
        refs = load_references(_P(a.reference_dir), a.thr)
        print(f"orientation references loaded: {sorted(refs)}")

    os.makedirs(a.out_dir, exist_ok=True)
    scratch = os.path.join(a.out_dir, "_frames")
    cells_root = os.path.join(a.out_dir, "cells")
    webp_dir = os.path.join(a.out_dir, "webp")
    for d in (scratch, cells_root, webp_dir):
        os.makedirs(d, exist_ok=True)

    if not a.mp4_dir and not a.frames_dir:
        raise SystemExit("need --mp4-dir or --frames-dir")
    if a.frames_dir:
        # 3パターン形式(歩A/歩B)の正解ペアのヒント: posesetの構造は
        # s1/s3が反対接地(検証済み_limb_labels: 全方向でs1とs5は同側)。
        # 画素差の最大ペア探索は位相を知らないため同側ペアを掴み得る
        # (2026-07-07監査: backでw2/w5=同側が選ばれ片足歩きになった)。
        # templates_render / GUIアニメはこのヒントを優先する。
        with open(os.path.join(cells_root, "walkAB.json"), "w",
                  encoding="utf-8") as fj:
            json.dump({"walkA": "walk1", "walkB": "walk3",
                       "_source": "poseset _limb_labels (s1/s3=反対接地) "
                                  "2026-07-07"}, fj, ensure_ascii=False)
        found = {}
        for direction in DIR_PLACEMENT:
            dd = os.path.join(a.frames_dir, direction)
            if os.path.isdir(dd):
                found[direction] = dd
    else:
        found = discover(a.mp4_dir)
    missing = [d for d in DIR_PLACEMENT if d not in found]
    print(f"directions found: {sorted(found)}")
    if missing:
        print(f"!! missing directions: {missing}")

    cw, ch = a.cell_w, a.cell_h
    cwl, chl = cw * 5, ch * 5
    sheet64 = Image.new("RGBA", (COLS * cw, ROWS * ch), (0, 0, 0, 0))
    sheetLT = Image.new("RGBA", (COLS * cwl, ROWS * chl), (0, 0, 0, 0))

    log = {"char": a.char, "thr": a.thr, "scale_mode": a.scale_mode,
           "cell_w": cw, "cell_h": ch,
           "directions": {}}
    work_items = []
    for direction, mp4 in found.items():
        row, block = DIR_PLACEMENT[direction]
        if a.frames_dir:
            # pre-made frames (walk-source codex): sorted PNGs are
            # idle + walk1..N, already phase-ordered and normalized
            frames = [os.path.join(mp4, f) for f in sorted(os.listdir(mp4))
                      if f.lower().endswith(".png")]
            if len(frames) < 6:
                print(f"!! {direction}: only {len(frames)} frame files; "
                      f"skipping")
                continue
            frames = frames[:6]
            clean = None
            # fa/fb=接地の両極(3パターン形式の歩A/歩B)。posesetの構造は
            # s1/s3が反対接地(検証済みラベル: 全方向でs1とs5は同側、
            # s3のみ逆側)——旧値fa=1/fb=5は「同じ脚が前の2コマ」を
            # 歩A/歩Bにしてしまい、ツクール/ウディタ形式で片足歩きになる
            # (2026-07-07全方向監査で発覚)。
            sel = {"idle": 0, "walk": list(range(1, 6)), "period": None,
                   "fa": 1, "fb": 3, "method": "frames-dir",
                   "source": "frames-dir"}
        else:
            fdir = os.path.join(scratch, direction)
            frames = extract(a.ffmpeg, mp4, fdir)
            clean = lean = None
            if refs:
                clean, lean = orientation_clean_mask(frames, direction,
                                                     refs, a.thr)
                if clean is not None:
                    print(f"  {direction:<12} orientation-clean frames: "
                          f"{sum(clean)}/{len(clean)}")
            _gp_dirs = [s.strip() for s in
                        (a.gait_period_dirs or "all").split(",")]
            _gp = (a.gait_period
                   if a.gait_period > 0
                   and ("all" in _gp_dirs or direction in _gp_dirs)
                   else None)
            # 末尾静止窓の有効性はクリップ実長で判定 (49f旧資産と57f新規が
            # 混在するラウンドでは、静止を持たない旧mp4にendを適用しない)
            _ge = int(a.gait_end or 0)
            _has_tail = bool(_gp) and 0 < _ge + 2 < len(frames)
            if direction in force_picks:
                walk = [max(0, min(len(frames) - 1, i))
                        for i in force_picks[direction]]
                # 骨格駆動では直立コマも select_frames と同じ規則で採る:
                # 末尾静止があれば最終フレーム (latent錨止め済み=最安定)、
                # 無ければ直立区間の最後 (f0の色段差を持ち込まない)
                if _has_tail:
                    _ip = len(frames) - 1
                elif _gp:
                    _ip = max(0, min(len(frames) - 1, int(a.gait_start) - 1))
                else:
                    _ip = 0
                sel = {"idle": _ip, "walk": walk, "period": None,
                       "fa": walk[0], "fb": walk[-1],
                       "method": "forced-picks"}
                print(f"  {direction:<12} forced picks: {walk}")
            else:
                sel = select_frames(frames, a.thr, clean=clean,
                                    rank=pick_rank.get(direction, 0),
                                    period=_gp,
                                    start=(a.gait_start if _gp else 0),
                                    end=(_ge if _has_tail else 0))
                if _has_tail:
                    # 既定の直立コマも末尾静止側へ (実測選択が後で上書き
                    # しうるが、refs無し実行でも末尾静止を使わせる)
                    sel["idle"] = len(frames) - 1
            if lean is not None:
                sel["picked_lean"] = [lean[i] for i in sel["walk"]]
        picks = [sel["idle"]] + sel["walk"]
        max_w, max_h = selected_bbox_size(frames, picks, a.thr)
        work_items.append({
            "direction": direction, "mp4": mp4, "row": row, "block": block,
            "frames": frames, "sel": sel, "picks": picks, "clean": clean,
            "max_bbox_w": max_w, "max_bbox_h": max_h,
        })

    # ---- period consensus ---------------------------------------------------
    # One character, one cadence: the gait period must agree across the eight
    # directions. Symmetric views (back, profiles) can lock onto k*P + P/2
    # (mirrored poses look identical from behind: C11 back chose 36 while the
    # other seven agreed on 24-25 -> 1.5 cycles in the loop). Re-estimate
    # outliers with the search constrained near the consensus.
    periods = sorted(it["sel"]["period"] for it in work_items
                     if it["sel"].get("period"))
    if len(periods) >= 3:
        consensus = periods[len(periods) // 2]
        tol = max(3, round(consensus * 0.2))
        for it in work_items:
            p = it["sel"].get("period")
            if it["sel"].get("period_forced"):
                continue          # 既知周期の明示指定は合議で覆さない
            if p and abs(p - consensus) > tol:
                print(f"  {it['direction']:<12} period {p} deviates from "
                      f"consensus {consensus}; re-selecting within "
                      f"[{consensus - 3}, {consensus + 3}]")
                it["sel"] = select_frames(
                    it["frames"], a.thr,
                    cy_lo=max(8, consensus - 3), cy_hi=consensus + 3,
                    clean=it["clean"],
                    rank=pick_rank.get(it["direction"], 0))
                it["picks"] = [it["sel"]["idle"]] + it["sel"]["walk"]
                it["max_bbox_w"], it["max_bbox_h"] = selected_bbox_size(
                    it["frames"], it["picks"], a.thr)
                it["sel"]["period_consensus_applied"] = True

    # 直立コマの実測選択 (2026-07-13ゴン実障害「直立フレームで直立して
    # ません」): モデルが骨格の直立指示を守らず歩き続ける方向がある
    # (ゴン左向きはf2で既に歩行)。固定フレームでは歩行中コマを掴むため、
    # 直立窓から参照立ち絵に最も近いフレームを実測で選ぶ (向き照合の
    # normalize_foreground/norm_diff を流用。色はカラーアンカーが直立
    # フレームをフレーム毎に補正するのでどのコマでも揃う)。
    # 窓は末尾静止 (--gait-end、2026-07-17) があれば [gait_end+2, n)
    # — 歩行→静止の切替直後1コマはモーフしがちなので+2から。末尾静止は
    # 歩行コマと同じ骨格アンカーに立つため、骨格が参照とズレたキャラでも
    # 直立↔歩行の位置・角度の継ぎ目が出ない。無ければ従来の先頭窓
    # [0, gait_start)。
    if a.gait_start > 0 and a.gait_period > 0 and refs:
        _gpd = [s.strip() for s in (a.gait_period_dirs or "all").split(",")]
        for it in work_items:
            d = it["direction"]
            if not ("all" in _gpd or d in _gpd):
                continue
            rimg = refs.get(d)
            if rimg is None:
                continue
            _cl = it.get("clean")
            _nf = len(it["frames"])
            _ge = int(a.gait_end or 0)
            if 0 < _ge + 2 < _nf:
                _win = range(_ge + 2, _nf)          # 末尾静止窓
            else:
                _win = range(min(int(a.gait_start), _nf))   # 先頭直立窓
            scores = []
            for i in _win:
                if _cl is not None and i < len(_cl) and not _cl[i]:
                    continue
                scores.append((norm_diff(normalize_foreground(
                    Image.open(it["frames"][i]), a.thr), rimg), i))
            best = bv = None
            if scores:
                bv = min(v for v, _ in scores)
                # 同着 (最良+15%以内) なら最も遅いフレームを採る: 先頭窓
                # ではf0のVACE色段差から遠いほど安全 (2026-07-13新型ロップ
                # 「直立だけ色が揃ってない」)、末尾窓では最終フレームが
                # latent錨止め (σ=0でstage1へ厳密着地) 済みで最も安定。
                # 歩き続ける方向はポーズ差が大きく同着にならないので、
                # 実測が勝つのは従来どおり
                best = max(i for v, i in scores if v <= bv * 1.15)
                bv = next(v for v, i in scores if i == best)
            if best is not None:
                if best != it["sel"].get("idle"):
                    print(f"  {d:<12} 直立コマ実測選択: f{best} "
                          f"(既定f{it['sel'].get('idle')}, 参照差 {bv:.3f})")
                it["sel"]["idle"] = best
                it["sel"]["idle_measured"] = True
                it["picks"] = [best] + it["sel"]["walk"]
                it["max_bbox_w"], it["max_bbox_h"] = selected_bbox_size(
                    it["frames"], it["picks"], a.thr)

    global_s64 = global_sLT = None
    if a.scale_mode == "global" and work_items:
        global_w = max(item["max_bbox_w"] for item in work_items)
        global_h = max(item["max_bbox_h"] for item in work_items)
        global_s64 = fit_scale(global_w, global_h, cw, ch, 2)
        global_sLT = fit_scale(global_w, global_h, cwl, chl, 10)
        log["global_fit"] = {
            "max_bbox_w": global_w, "max_bbox_h": global_h,
            "scale64": round(global_s64, 4), "scaleLT": round(global_sLT, 4),
        }
        print(f"global scale from selected frames: bbox={global_w}x{global_h} "
              f"s64={global_s64:.4f} sLT={global_sLT:.4f}")

        if a.scale_anchor == "head":
            ref_item = next((it for it in work_items
                             if it["direction"] == "front"), work_items[0])
            idle_img = load_keyed(ref_item["frames"][0], a.thr)
            hh = head_height_px(idle_img)
            if hh:
                # idle pose must never clip; extreme walk poses may exceed
                # the cell by up to --max-overflow (clipped at the edge),
                # which is what frees wide-costume characters to reach the
                # common head size at all
                iw = ih = 1
                for it in work_items:
                    ib = load_keyed(it["frames"][0], a.thr).getchannel(
                        "A").getbbox()
                    if ib:
                        iw = max(iw, ib[2] - ib[0])
                        ih = max(ih, ib[3] - ib[1])
                fit_idle = fit_scale(iw, ih, cw, ch, 2)
                head_scale = (a.head_frac * ch) / hh
                s = min(head_scale, fit_idle,
                        global_s64 * (1 + a.max_overflow))
                capped = s < head_scale
                log["head_anchor"] = {
                    "head_px": round(hh, 1), "head_frac": a.head_frac,
                    "scale64": round(s, 4), "capped": capped,
                    "fit_all": round(global_s64, 4),
                    "fit_idle": round(fit_idle, 4)}
                global_s64 = s
                global_sLT = s * 5
                print(f"head-anchored scale: head={hh:.0f}px -> target "
                      f"{a.head_frac * ch:.0f}px, s64={s:.4f}"
                      + (" (capped)" if capped else ""))
            else:
                # no trustworthy neck: still grant the head-anchor's
                # overflow allowance -- fit the IDLE fully and let extreme
                # walk poses clip up to --max-overflow, so a wide-sleeved
                # character is not shrunk by its own arm swing
                iw = ih = 1
                for it in work_items:
                    ib = load_keyed(it["frames"][0], a.thr).getchannel(
                        "A").getbbox()
                    if ib:
                        iw = max(iw, ib[2] - ib[0])
                        ih = max(ih, ib[3] - ib[1])
                fit_idle = fit_scale(iw, ih, cw, ch, 2)
                s = min(fit_idle, global_s64 * (1 + a.max_overflow))
                log["head_anchor"] = {"head_px": None,
                                      "scale64": round(s, 4),
                                      "fit_all": round(global_s64, 4),
                                      "fit_idle": round(fit_idle, 4)}
                global_s64 = s
                global_sLT = s * 5
                print("head anchor requested but head not detected; "
                      f"idle-fit scale with overflow: s64={s:.4f}")

        # ---- 自動頭身そろえ (dip法) -- 手動 --dir-scale の自動化 ----
        # C27で人間が目視確定した値 (fr=1.25, fl/bl/br=1.08) を、頭こぶ
        # メトリックが 1.23/1.06/1.07/1.08 で再現し、承認済みC21では
        # 無発動 -- で校正済み (2026-07-04)。
        #   発動条件: 複合メトリック M=sqrt(頭高x頭幅) が中央値より
        #   5%以上小さい (デッドバンド兼拡大専用)、かつ片側プロキシが
        #   強く反対しない (>10%「大きい」と言う側があれば髪・飾りの
        #   計測事故とみなし棄却)。中央値はこぶ検出が5方向以上ある
        #   ときだけ信頼する。上限1.10 (2026-07-04 人間校正:
        #   C01-C28一斉ビフォアフで×1.10超の自動補正は横幅の太り--
        #   身長そろえは縦しか戻さない--でバランス崩壊に見えた。
        #   C27右前のような1.25級の真の欠損は自動1.1+手動--dir-scale
        #   で仕上げる運用)。
        auto_scale: dict[str, float] = {}
        dip_measured: set[str] = set()
        if a.scale_anchor == "head" and a.auto_dir_scale:
            hums = {}
            for it in work_items:
                m = head_hump_metric(load_keyed(it["frames"][0], a.thr))
                if m:
                    hums[it["direction"]] = m
            if len(hums) >= 5:
                dip_measured = set(hums)
                Ms = {d: (h * w) ** 0.5 for d, (h, w) in hums.items()}
                med = sorted(Ms.values())[len(Ms) // 2]
                med_h = sorted(h for h, _ in hums.values())[len(hums) // 2]
                med_w = sorted(w for _, w in hums.values())[len(hums) // 2]
                for d, (hh, hw) in hums.items():
                    if d in dir_scale:
                        continue  # 手動 --dir-scale が常に優先
                    if d not in ("front_left", "front_right"):
                        # 自動発動は前斜め2方向のみ (2026-07-04、C01〜C28
                        # 一斉ビフォアフ3周の人間校正)。方向クラス別
                        # 成績: 前斜め6/6正解、横0/7 (おさげ・もみあげ・
                        # 結い髪がプロファイルのくぼみを崩す)、前0/1
                        # (マフラーであご下が埋まる)、後ろ2/7 (襟・フード
                        # の首隠れ+後頭部の髪塊デッカチ化)。首隠れ・髪
                        # 盛りは幅プロファイル形状でも肌色でも機械判別
                        # できなかった (C14右後とC27右前が同一形状、C06
                        # は背負い物が肌色に化ける)。前斜めだけは3/4角で
                        # あご・首が素直に出る上、描き直し/ペア生成で頭が
                        # 小さくなる事故が起きるのも前斜め (C27右前・
                        # アルバート右前)。他方向の頭身は手動 --dir-scale。
                        continue
                    f = med / Ms[d]
                    if f < 1.05:
                        continue
                    if min(med_h / hh, med_w / hw) < 0.90:
                        continue
                    auto_scale[d] = round(min(1.10, f), 4)
                    print(f"  {d:<12} 自動頭身そろえ x{auto_scale[d]:.3f} "
                          f"(頭こぶ M={Ms[d]:.0f} -> 中央値 {med:.0f})")
                if auto_scale:
                    log["auto_dir_scale"] = dict(auto_scale)
            else:
                print(f"自動頭身そろえ: 頭こぶ検出 {len(hums)}/8 方向 -- "
                      "中央値が不安定なので不介入 (capA正規化に委任)")

        # 方向間サイズ正規化 (2026-07-13ユーザー報告「横向きがおっきい」):
        # コンパスキャンバスのレターボックスはmin比率で立ち絵をセルへ
        # 広げるため、細身の横向きが正面より大きく生成される (ロップ実測
        # +13%)。参照立ち絵 (生画像) の方向間プロポーションへ一様スケール
        # で戻す。生成側は直さない — 横向きが大きい=ソース解像度が高い
        # のは縮小時にむしろ得。normalize_foreground済みのrefsは比率が
        # 消えているため、生の立ち絵から測り直す。
        if a.reference_dir and a.scale_anchor == "head" \
                and len(work_items) >= 3:
            ref_h: dict = {}
            for pattern in ("*centered*.png", "*.png"):
                for path in sorted(_P(a.reference_dir).glob(pattern)):
                    d = direction_of(path.stem)
                    if d is None or d in ref_h:
                        continue
                    bb = key_magenta_alpha(
                        Image.open(path), a.thr).getchannel("A").getbbox()
                    if bb:
                        ref_h[d] = bb[3] - bb[1]
            src_h: dict = {}
            for it in work_items:
                ib = load_keyed(it["frames"][it["sel"].get("idle", 0)],
                                a.thr).getchannel("A").getbbox()
                if ib:
                    src_h[it["direction"]] = ib[3] - ib[1]
            common = [d for d in src_h if d in ref_h]
            if len(common) >= 3:
                rmed = sorted(ref_h[d] for d in common)[len(common) // 2]
                smed = sorted(src_h[d] for d in common)[len(common) // 2]
                for it in work_items:
                    d = it["direction"]
                    f = 1.0
                    if d in ref_h and d in src_h and rmed and smed:
                        f = ((ref_h[d] / rmed)
                             / max(0.01, src_h[d] / smed))
                        f = max(0.80, min(1.15, f))
                        if abs(f - 1.0) < 0.02:
                            f = 1.0
                    it["size_norm"] = round(f, 4)
                    if f != 1.0:
                        print(f"  {d:<12} 方向間サイズ正規化 x{f:.3f} "
                              f"(参照 {ref_h.get(d)}/{rmed} vs "
                              f"実測 {src_h.get(d)}/{smed})")
                log["size_norm"] = {
                    it["direction"]: it.get("size_norm", 1.0)
                    for it in work_items}

        if a.scale_anchor == "head" and a.head_norm_max > 0:
            # per-DIRECTION head-size normalization: the still generators
            # draw some views with a visibly smaller head, and a shared
            # global scale carries that straight into the sheet -- 頭身が
            # 方向で違って見える. Equalize the cap metric to the median,
            # gently (clamped), keeping the idle no-clip rule.
            caps = {}
            for it in work_items:
                m = cap_area_metric(load_keyed(it["frames"][0], a.thr))
                if m:
                    # サイズ正規化後の見かけで比較する (capは√面積=線形
                    # スケールなので size_norm を1乗で掛ける)
                    caps[it["direction"]] = (
                        m * it.get("size_norm", 1.0))
            # 参照立ち絵の方向別頭サイズ (比較の物差し)。横顔の頭
            # シルエットが正面より小さく写るのは幾何的に自然で、参照にも
            # 等しく現れるため比を取ると相殺される。旧・方向間中央値
            # そろえはこの幾何差を「小頭」と誤認して横向きを+11%誤拡大
            # していた (2026-07-13ユーザー報告「横向きがおっきい」の真因。
            # ソース動画と参照はどちらも横+2%で正常だった)
            ref_caps = {}
            if a.reference_dir:
                for pattern in ("*centered*.png", "*.png"):
                    for path in sorted(_P(a.reference_dir).glob(pattern)):
                        d = direction_of(path.stem)
                        if d is None or d in ref_caps:
                            continue
                        m0 = cap_area_metric(
                            key_magenta_alpha(Image.open(path), a.thr))
                        if m0:
                            ref_caps[d] = m0
            if len(caps) >= 3:
                med = sorted(caps.values())[len(caps) // 2]
                _common = [d for d in caps if d in ref_caps]
                use_ref = len(_common) >= 3
                rmed = (sorted(ref_caps[d] for d in _common)[len(_common) // 2]
                        if use_ref else None)
                for it in work_items:
                    if it["direction"] in dir_scale:
                        continue  # 手動頭身スケールが自動補正より優先
                    if it["direction"] in dip_measured:
                        continue  # dip法が計測できた方向はその判定を採用
                    c = caps.get(it["direction"])
                    if use_ref and c and it["direction"] in ref_caps \
                            and rmed and med:
                        # 参照比そろえ: (参照での頭サイズ比) / (実測の
                        # 頭サイズ比)。参照が物差しなので縮小側も対称に
                        # 信用できる (旧法の拡大専用床は不要)
                        m = ((ref_caps[it["direction"]] / rmed)
                             / max(0.01, c / med))
                        m = max(1 - a.head_norm_max,
                                min(1 + a.head_norm_max, m))
                    else:
                        m = med / c if c else 1.0
                        # enlarge-only (床0.97): capAが大きい方向は「頭が
                        # 大きい」のではなく結い髪・ポニーテールが帯に
                        # 入っただけのことが多い (C22は横向きが13%縮んだ)。
                        # 参照無しのときだけの保守則
                        m = max(0.97, min(1 + a.head_norm_max, m))
                    ib = load_keyed(it["frames"][0],
                                    a.thr).getchannel("A").getbbox()
                    if ib and global_s64:
                        f_idle = fit_scale(ib[2] - ib[0], ib[3] - ib[1],
                                           cw, ch, 2)
                        m = min(m, f_idle / global_s64)
                    it["head_norm"] = round(m, 4)
                    if abs(m - 1) >= 0.015:
                        print(f"  {it['direction']:<12} 頭サイズ正規化 "
                              f"x{m:.3f} (capA {c:.0f} -> 中央値 {med:.0f})")

        # 方向別頭身ノブ: 図ごと拡大し、直後の身長そろえ (squash可) が
        # 首下を縮めて身長を戻す = 頭身だけが変わる。手動 --dir-scale が
        # 最優先、なければ dip法の自動係数。(顎シルエット法は方向で
        # 暴れるため不採用: C27で前0.47/横0.32/斜め0.18。dip法は局所
        # くぼみ基準なのでこの暴れ方をしない -- head_hump_metric 参照。)
        for it in work_items:
            ds = dir_scale.get(it["direction"])
            if ds:
                it["head_norm"] = round(it.get("head_norm", 1.0) * ds, 4)
                print(f"  {it['direction']:<12} 手動頭身スケール x{ds:.3f}")
            elif it["direction"] in auto_scale:
                asf = auto_scale[it["direction"]]
                it["head_norm"] = round(it.get("head_norm", 1.0) * asf, 4)
        if a.scale_anchor == "head":
            log["head_norm"] = {it["direction"]: it.get("head_norm", 1.0)
                                for it in work_items}

        if a.scale_anchor == "head" and a.height_norm_max > 0:
            # 身長の方向間そろえ (頭は聖域): idle高の中央値と違う方向は
            # 首下だけ縦に伸縮する。伸長は上限 --height-norm-max、squash
            # は 0.90 まで (手動頭身スケールで大きくした方向の身長戻し)。
            ihs = {}
            raw_h = {}
            for it in work_items:
                ib = load_keyed(it["frames"][0], a.thr).getchannel(
                    "A").getbbox()
                if ib:
                    raw_h[it["direction"]] = ib[3] - ib[1]
                    ihs[it["direction"]] = ((ib[3] - ib[1])
                                            * it.get("head_norm", 1.0)
                                            * it.get("size_norm", 1.0))
            if len(ihs) >= 3:
                # 目標身長=スケール前の自然な中央値。スケール後の値で
                # 中央値を取ると、手動拡大した方向が基準を押し上げて
                # 無関係な方向まで伸ばされる (C27で前後左右が+7%伸びた)
                med_h = sorted(raw_h.values())[len(raw_h) // 2]
                for it in work_items:
                    hcur = ihs.get(it["direction"])
                    if hcur:
                        # f は「首下 (下部55%) にだけ掛かる」係数なので、
                        # 全身比 med/hcur では効き不足になる (C27で発覚:
                        # -22%要求が-12%しか効かずゲートFAIL)。keep分を
                        # 除いた正しい換算:
                        keep = 0.45 * hcur
                        f = (med_h - keep) / max(1.0, hcur - keep)
                    else:
                        f = 1.0
                    # 頭身スケールが掛かった方向 (手動/自動どちらも) は
                    # 拡大した身長を戻すため深squashを許す (頭パリティに
                    # f~0.65 が要る絵がある: C27右前)
                    lo = 0.65 if (dir_scale.get(it["direction"])
                                  or it["direction"] in auto_scale) else 1.0
                    f = max(lo, min(1 + a.height_norm_max, f))
                    it["height_stretch"] = round(f, 4)
                    if abs(f - 1.0) >= 0.015:
                        print(f"  {it['direction']:<12} 首下伸縮 x{f:.3f} "
                              f"(idle高 {hcur:.0f} -> 中央値 {med_h:.0f})")
                log["height_stretch"] = {
                    it["direction"]: it.get("height_stretch", 1.0)
                    for it in work_items}

    for item in work_items:
        direction = item["direction"]
        mp4 = item["mp4"]
        row = item["row"]
        block = item["block"]
        frames = item["frames"]
        sel = item["sel"]
        picks = item["picks"]
        imgs = [load_keyed(frames[i], a.thr) for i in picks]

        hn = item.get("head_norm", 1.0) * item.get("size_norm", 1.0)
        hs = item.get("height_stretch", 1.0)
        if abs(hs - 1.0) > 0.001:  # stretch AND squash (>1 gate ate squash)
            imgs = [stretch_below_neck(im, hs) for im in imgs]
        cells64, s64, off64, ov64 = build_cells(
            imgs, cw, ch, 2, global_s64 * hn if global_s64 else global_s64)
        cellsLT, sLT, offLT, ovLT = build_cells(
            imgs, cwl, chl, 10, global_sLT * hn if global_sLT else global_sLT)
        paste_block(sheet64, cells64, row, block, cw, ch)
        paste_block(sheetLT, cellsLT, row, block, cwl, chl)

        # save per-direction cells (64px) + animated walk WebP (LT frames)
        cdir = os.path.join(cells_root, direction)
        os.makedirs(cdir, exist_ok=True)
        names = ["idle", "walk1", "walk2", "walk3", "walk4", "walk5"]
        for nm, c in zip(names, cells64):
            c.save(os.path.join(cdir, f"{direction}_{nm}T.png"))
        cellsLT[1].save(os.path.join(webp_dir, f"{a.char}_{direction}_walkT.webp"),
                        save_all=True, append_images=cellsLT[2:6],
                        duration=140, loop=0, disposal=2)

        log["directions"][direction] = {
            "mp4": os.path.basename(mp4), "row": row, "block": block,
            "n_frames": len(frames), "picks": picks, "selection": sel,
            "selected_max_bbox": [item["max_bbox_w"], item["max_bbox_h"]],
            "scale64": s64, "offset64": off64, "overflow64_px": ov64,
            "scaleLT": sLT, "offsetLT": offLT, "overflowLT_px": ovLT,
        }
        gait_note = ""
        if "gait_ratio" in sel:
            stable = sel["gait_ratio"] <= a.gait_ratio_max
            smooth = sel.get("pick_uniformity", 0) <= a.pick_uniformity_max
            sel["gait_ok"] = stable and smooth
            gait_note = (f" gait P={sel['period']} "
                         f"E/A={sel['gait_ratio']} "
                         f"U={sel.get('pick_uniformity')}"
                         + ("" if stable else " !! UNSTABLE GAIT")
                         + ("" if smooth else " !! MISSING IN-BETWEENS"))
        print(f"  {direction:<12} n={len(frames)} idle={sel.get('idle', 0)} "
              f"fa={sel['fa']} "
              f"fb={sel['fb']} walk={sel['walk']} [{sel['method']}] "
              f"s64={s64} ov64={ov64}px{gait_note}")

        if not a.keep_frames and not a.frames_dir:
            # mp4 mode only: the extracted frames are scratch. In
            # frames-dir mode these files ARE the caller's inputs.
            for f in frames:
                os.remove(f)
            try:
                os.rmdir(fdir)
            except OSError:
                pass

    out64 = os.path.join(a.out_dir, f"{a.char}_walkT.png")
    outLT = os.path.join(a.out_dir, f"{a.char}_walkLT.png")
    sheet64.save(out64)
    sheetLT.save(outLT)
    with open(os.path.join(a.out_dir, f"{a.char}_build_logT.json"), "w",
              encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    if not a.keep_frames:
        try:
            os.rmdir(scratch)
        except OSError:
            pass
    print(f"\nwrote {out64}  {sheet64.size}")
    print(f"wrote {outLT}  {sheetLT.size}")
    print(f"webp per direction -> {webp_dir}")


if __name__ == "__main__":
    main()
