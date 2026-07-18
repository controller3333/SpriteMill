#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""color_anchor.py — hybrid生成セルの色を参照立ち絵へ決定的に再アンカーする。

VACE→AniSora latent直結にはリファイン段が無く、handoff後の無制御区間で
全体が黄色へ漂う (2026-07-13実測: 純マゼンタ背景の青が255→183〜207。
steps/boundaryノブではB207が上限だった)。

補正 (v4, 2026-07-13。実測したドリフト構造に基づく):
  hybridの色ズレは漸進ドリフトではなく「f0だけ暗く、f6以降は安定した別
  パレット」という段差 (VACE経路の既知の先頭フレーム特性。ロップ左横実測:
  髪 f0=(129,80,64) → f6以降=(152〜155,105〜107,61)で安定)。よって:
  ①直立区間 (先頭idle_nフレーム、立ち姿=参照と色構成が1:1): フレーム毎に
    (背景→純マゼンタ, キャラ平均→参照キャラ平均) の2点アフィンでフィット
    → 遷移中の各直立コマ (シートのidleコマ=f0含む) が参照色に揃う。
  ②歩行区間 (idle_n以降、パレット内部一貫): 歩行統計に最も近い直立
    フレームの写像を1本だけ全歩行フレームへ共有適用。パレットが等しい
    フレームは画素値も等しいので、立ち姿でフィットした写像がそのまま
    正しく働き、姿勢による色構成の違いが写像に一切入らない。
★v1の教訓: フレーム毎の「キャラ全体平均→参照平均」は姿勢で色構成が変わる
たび写像が揺れ、直立コマと歩行コマで髪の色がズレる (ユーザー報告、髪G差
14.5実測)。★v2の教訓: 背景基準のフレーム毎補正+固定キャラ写像は、キャラ
自体の段差 (背景と形が違う) を見逃し悪化 (G差29)。★v3の教訓: 頭部バンドを
フレーム毎アンカーにすると、頭に花があるキャラで花のピンク≈マゼンタと
なりアンカー2点が縮退してR/Bが暴れる (R/B差10超)。

ノブ: config videolab_color_anchor (既定on) / 環境変数 SM_COLOR_ANCHOR 優先。
off/0/false/none で無効。直立区間の長さは SM_POSE_IDLE (pose_videoと同じ、
既定6) を参照する。
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

MAGENTA = (255.0, 0.0, 255.0)
# windowed EXEからのffmpeg起動で子コンソールが一瞬開く「チカチカ」対策
# (2026-07-13報告: 方向別8セル×2呼び出しで顕著)
CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def bg_mask(a: np.ndarray) -> np.ndarray:
    """寛容なマゼンタ背景検出 (黄変後の背景も拾えるよう青の下限を緩める)。"""
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return (r > 140) & (b > 90) & (g < 110) & (np.abs(r - b) < 130)


def compose_ref_cell(ref: Image.Image, width: int, height: int) -> np.ndarray:
    """参照立ち絵をセルと同じレターボックス写像 (min比率+中央) でマゼンタ地へ。

    compass_vace.compose_reference / pose_video._char_box と同一の写像で
    あること — キャラ占有率が生成セルと揃わないと平均の対応が崩れる。"""
    ref = ref.convert("RGB")
    sc = min(width / ref.width, height / ref.height)
    rs = ref.resize((max(1, round(ref.width * sc)),
                     max(1, round(ref.height * sc))), Image.LANCZOS)
    canvas = Image.new("RGB", (width, height),
                       tuple(int(v) for v in MAGENTA))
    canvas.paste(rs, ((width - rs.width) // 2, (height - rs.height) // 2))
    return np.asarray(canvas, dtype=np.float32)


def char_mean(ref_cell: np.ndarray) -> np.ndarray:
    """参照セルのキャラ領域 (非マゼンタ) の平均色。"""
    m = ~bg_mask(ref_cell)
    if m.sum() < 16:
        raise ValueError("参照セルにキャラ領域が見つかりません")
    return ref_cell[m].mean(axis=0)


def _apply_affine(a: np.ndarray, maps) -> np.ndarray:
    out = np.empty_like(a)
    for c, (s, b) in enumerate(maps):
        out[..., c] = a[..., c] * s + b
    return np.clip(out, 0.0, 255.0)


def _bg_normalize_map(bg) -> tuple:
    """背景実測→純マゼンタ: R,B=ゲイン / G=オフセット (キャラ不可視時の保険)。"""
    return ((255.0 / max(float(bg[0]), 1.0), 0.0),
            (1.0, -float(bg[1])),
            (255.0 / max(float(bg[2]), 1.0), 0.0))


def _idle_n() -> int:
    """直立プレフィックス長 (pose_video.walk_layoutのSM_POSE_IDLEと同じ既定)。"""
    try:
        return max(1, min(24, int(float(os.environ.get("SM_POSE_IDLE", "")
                                        or 6))))
    except (TypeError, ValueError):
        return 6


def _fit_two_point(bg, char, char_target) -> tuple:
    """(背景実測→純マゼンタ, キャラ実測平均→目標) を通るチャネル別アフィン。"""
    maps = []
    for c in range(3):
        x1, y1 = float(bg[c]), MAGENTA[c]
        x2, y2 = float(char[c]), float(char_target[c])
        if abs(x2 - x1) < 5.0:
            if y1 > 0:
                maps.append((y1 / max(x1, 1.0), 0.0))
            else:
                maps.append((1.0, -x1))
            continue
        s = (y2 - y1) / (x2 - x1)
        s = min(max(s, 0.5), 2.0)
        maps.append((s, y1 - s * x1))
    return tuple(maps)


def _idle_cluster_align(out: list, k: int, ref_cell: np.ndarray,
                        tail_start: int | None = None) -> None:
    """直立フレームの素材別色そろえ (in-place)。

    f0はVACE経路の色段差が素材ごとに違い (髪+27G/体+12G)、アフィン1本の
    v4補正では歩行コマと揃わない (2026-07-13新型ロップ「直立だけ色が
    揃ってない」— ポーズ実測選択がf0しか選べない方向で再発)。参照立ち絵の
    色クラスタ (髪/肌/服…) で画素を分類し、直立フレームの各クラスタ平均を
    「歩行フレームの同クラスタ平均」へシフトする = 素材別の色転写。
    シェーディングはクラスタ内相対値として保持される。
    tail_start: 末尾静止区間の開始 (walk_layoutのtail>0時)。歩行代表の
    サンプルを [k, tail_start) に限定し、末尾静止フレームにも直立側と
    同じ整列を適用する (シートの直立コマは末尾から採られるため)。"""
    if not out or k <= 0:
        return
    ts = (tail_start if tail_start is not None
          and k < tail_start <= len(out) else len(out))
    if ts - k <= 3:
        return                      # 歩行区間が短すぎて代表が取れない
    rm = ~bg_mask(ref_cell)
    if rm.sum() < 500:
        return
    ref_px = ref_cell[rm].astype(np.float32)
    if len(ref_px) > 40000:
        ref_px = ref_px[:: len(ref_px) // 40000 + 1]
    # 決定論kmeans: 輝度分位で初期化 (乱数不使用・毎回同じ結果)
    n_k = 6
    order = np.argsort(ref_px.mean(axis=1))
    centers = ref_px[order[np.linspace(0, len(order) - 1, n_k)
                           .astype(int)]].copy()
    for _ in range(8):
        d = ((ref_px[:, None, :] - centers[None]) ** 2).sum(axis=2)
        lab = d.argmin(axis=1)
        for j in range(n_k):
            sel = lab == j
            if sel.any():
                centers[j] = ref_px[sel].mean(axis=0)

    def _members(a):
        m = ~bg_mask(a)
        px = a[m].astype(np.float32)
        if len(px) < 500:
            return None, None, None
        d = ((px[:, None, :] - centers[None]) ** 2).sum(axis=2)
        return m, px, d.argmin(axis=1)

    # 歩行区間 [k, ts) の代表フレーム3枚からクラスタ別の目標色を作る
    walk_idx = [k + 1, k + (ts - k) // 2, ts - 2]
    targets = {}
    counts = {}
    for wi in walk_idx:
        m, px, lab = _members(out[wi])
        if px is None:
            continue
        for j in range(n_k):
            sel = lab == j
            if sel.sum() < 50:
                continue
            targets[j] = targets.get(j, 0.0) + px[sel].mean(axis=0)
            counts[j] = counts.get(j, 0) + 1
    for j in list(targets):
        targets[j] = targets[j] / counts[j]
    if not targets:
        return
    # 先頭直立 + 末尾静止の両方を歩行の色へ整列 (直立コマの供給源が
    # 末尾に移っても「直立だけ色が違う」を再発させない)
    for i in list(range(k)) + list(range(ts, len(out))):
        m, px, lab = _members(out[i])
        if px is None:
            continue
        fixed = px.copy()
        for j, tgt in targets.items():
            sel = lab == j
            if sel.sum() < 50:
                continue
            fixed[sel] += tgt - px[sel].mean(axis=0)
        a = out[i]
        a[m] = np.clip(fixed, 0.0, 255.0)


def anchor_frames(frames: list, ref_char, idle_n: int | None = None,
                  ref_cell: np.ndarray | None = None,
                  gait_end: int | None = None) -> list:
    """全フレームを補正して返す (入出力 float32 HxWx3 のリスト)。

    直立区間=フレーム毎フィット (立ち姿なので参照と色構成が1:1)、
    歩行区間=歩行統計に最も近い直立フレームの写像を共有適用 (パレットが
    内部一貫なので1本で正しく、姿勢の色構成が写像に入らない)。
    gait_end: 歩行終端フレーム (walk_layoutのtail>0時)。gait_end+1以降は
    末尾静止=立ち姿なので直立区間と同じフレーム毎フィットを適用し、
    歩行統計 (中央値) にも混ぜない。シートの直立コマは末尾静止から
    採られる (2026-07-17) ため、ここの色一貫性が直立コマの色を決める。"""
    if not frames:
        return frames
    n_px = frames[0].shape[0] * frames[0].shape[1]
    bgs, chars = [], []
    for f in frames:
        m = bg_mask(f)
        has_bg = m.sum() >= n_px * 0.01
        cm = ~m
        bgs.append(f[m].mean(axis=0) if has_bg else None)
        chars.append(f[cm].mean(axis=0)
                     if has_bg and cm.sum() >= n_px * 0.01 else None)
    if all(b is None for b in bgs):
        return [np.clip(f, 0.0, 255.0) for f in frames]   # 背景不明: 触らない
    # 統計が取れないフレームは近傍から補間 (背景/キャラ平均は数万画素の
    # 平均でノイズ極小のため、取れたフレームの値はそのまま使う)
    for seq in (bgs, chars):
        last = None
        for i, v in enumerate(seq):
            if v is not None:
                last = v
            seq[i] = last
        nxt = None
        for i in range(len(seq) - 1, -1, -1):
            if seq[i] is not None:
                nxt = seq[i]
            else:
                seq[i] = nxt
    if any(c is None for c in chars):
        # キャラが一度も見えない: 背景正規化のみ (安全側)
        return [_apply_affine(f, _bg_normalize_map(bgs[i]))
                for i, f in enumerate(frames)]

    k = min(idle_n if idle_n is not None else _idle_n(), len(frames))
    ts = len(frames)                      # 末尾静止の開始 (無ければ末尾)
    if gait_end is not None and k <= int(gait_end) + 1 < len(frames):
        ts = int(gait_end) + 1
    idle_maps = [_fit_two_point(bgs[i], chars[i], ref_char)
                 for i in range(k)]
    out = [_apply_affine(frames[i], idle_maps[i]) for i in range(k)]
    if k < ts:
        # 歩行区間 [k, ts) の代表統計 (中央値) に最も近い直立フレームの
        # 写像を共有 (末尾静止は立ち姿なので統計に混ぜない)
        wbg = np.median(np.stack(bgs[k:ts]), axis=0)
        wch = np.median(np.stack(chars[k:ts]), axis=0)
        a = min(range(k), key=lambda i: float(
            np.abs(bgs[i] - wbg).sum() + np.abs(chars[i] - wch).sum()))
        wmap = idle_maps[a]
        out += [_apply_affine(f, wmap) for f in frames[k:ts]]
    # 末尾静止: 直立区間と同じフレーム毎フィット (立ち姿=参照と1:1)
    out += [_apply_affine(frames[i],
                          _fit_two_point(bgs[i], chars[i], ref_char))
            for i in range(ts, len(frames))]
    if ref_cell is not None:
        _idle_cluster_align(out, k, ref_cell, tail_start=ts)
    return out


def _crop_to_aspect(im: Image.Image, tw: int, th: int) -> Image.Image:
    """ターゲットアス比へ中央クロップ (レターボックスの余白側だけ削る)。

    セルは参照立ち絵をmin比率で中央配置したレターボックスなので、
    参照アス比へのクロップはキャラを欠けさせない。丸ごとresizeで
    アス比を変えるとキャラが潰れる (2026-07-13新型ロップ実障害:
    セル0.556→参照0.633へ引き伸ばされ縦14%潰れ、シートまで汚染)。"""
    ca, ta = im.width / im.height, tw / th
    if abs(ca - ta) < 1e-3:
        return im
    if ta > ca:          # ターゲットの方が横長 -> 高さ(上下の余白)を削る
        nh = max(1, round(im.width / ta))
        y = (im.height - nh) // 2
        return im.crop((0, y, im.width, y + nh))
    nw = max(1, round(im.height * ta))   # 縦長 -> 幅(左右の余白)を削る
    x = (im.width - nw) // 2
    return im.crop((x, 0, x + nw, im.height))


def split_anchor_scale(ffmpeg: str, canvas_mp4: Path, jobs: list,
                       fps: int = 16, idle_n: int | None = None,
                       gait_end: int | None = None) -> None:
    """キャンバス動画を1回だけデコードし、セルごとに 切出し→アンカー→拡大→
    エンコード を単一パスで行う (2026-07-13ユーザー要望「ピクセル化する前に
    色を直したい」の実装形: セルmp4を一度焼いてから開き直す中間再エンコードを
    1世代削減し、ffmpeg起動も 1+セル数 回に減る)。

    jobs: [{"crop": (x, y, w, h), "ref": Path|Image.Image,
            "dest": Path, "size": (tw, th)}, ...]
    キャンバス実寸がcropの前提と合わない場合は例外 (呼び出し側が旧経路=
    分割→セル毎アンカーへフォールバック)。"""
    with tempfile.TemporaryDirectory() as td:
        fd = Path(td)
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i",
                        str(canvas_mp4), str(fd / "in_%05d.png")],
                       check=True, capture_output=True,
                       creationflags=CREATE_NO_WINDOW)
        paths = sorted(fd.glob("in_*.png"))
        if not paths:
            raise RuntimeError(f"フレームを抽出できません: {canvas_mp4}")
        frames = [np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
                  for p in paths]
        fh, fw = frames[0].shape[:2]
        for job in jobs:
            x, y, w, h = job["crop"]
            if x + w > fw or y + h > fh:
                raise RuntimeError(
                    f"キャンバス実寸 {fw}x{fh} がセル切出し {job['crop']} と"
                    "整合しません")
        for j, job in enumerate(jobs):
            x, y, w, h = job["crop"]
            ref = (job["ref"] if isinstance(job["ref"], Image.Image)
                   else Image.open(job["ref"]))
            ref_cell = compose_ref_cell(ref, w, h)
            cells = [f[y:y + h, x:x + w].astype(np.float32) for f in frames]
            fixed = anchor_frames(cells, char_mean(ref_cell),
                                  idle_n=idle_n, ref_cell=ref_cell,
                                  gait_end=gait_end)
            tw, th = job["size"]
            for i, a in enumerate(fixed):
                im = Image.fromarray(a.astype(np.uint8))
                if (im.width, im.height) != (tw, th):
                    im = _crop_to_aspect(im, tw, th)
                    im = im.resize((tw, th), Image.LANCZOS)
                im.save(fd / f"c{j}_{i:05d}.png")
            subprocess.run([ffmpeg, "-y", "-loglevel", "error",
                            "-framerate", str(fps),
                            "-i", str(fd / f"c{j}_%05d.png"),
                            "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-crf", "16", "-an", str(job["dest"])],
                           check=True, capture_output=True,
                           creationflags=CREATE_NO_WINDOW)


def anchor_and_scale(ffmpeg: str, src_mp4: Path, ref_still: Path | Image.Image,
                     dest_mp4: Path, width: int, height: int,
                     fps: int = 16, idle_n: int | None = None,
                     gait_end: int | None = None) -> None:
    """セル動画を フレーム毎カラーアンカー + lanczos拡大 して書き出す。

    失敗時は例外を投げる (呼び出し側が無補正スケールへフォールバック)。"""
    ref = (ref_still if isinstance(ref_still, Image.Image)
           else Image.open(ref_still))
    with tempfile.TemporaryDirectory() as td:
        fd = Path(td)
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i",
                        str(src_mp4), str(fd / "in_%05d.png")],
                       check=True, capture_output=True,
                       creationflags=CREATE_NO_WINDOW)
        paths = sorted(fd.glob("in_*.png"))
        if not paths:
            raise RuntimeError(f"フレームを抽出できません: {src_mp4}")
        frames = [np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)
                  for p in paths]
        ref_cell = compose_ref_cell(ref, frames[0].shape[1],
                                    frames[0].shape[0])
        fixed = anchor_frames(frames, char_mean(ref_cell),
                              idle_n=idle_n, ref_cell=ref_cell,
                              gait_end=gait_end)
        for i, a in enumerate(fixed):
            im = Image.fromarray(a.astype(np.uint8))
            if (im.width, im.height) != (width, height):
                im = _crop_to_aspect(im, width, height)
                im = im.resize((width, height), Image.LANCZOS)
            im.save(fd / f"out_{i:05d}.png")
        subprocess.run([ffmpeg, "-y", "-loglevel", "error",
                        "-framerate", str(fps),
                        "-i", str(fd / "out_%05d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "16", "-an", str(dest_mp4)],
                       check=True, capture_output=True,
                       creationflags=CREATE_NO_WINDOW)


if __name__ == "__main__":
    raise SystemExit("compass_vace/pipeline から使ってください")
