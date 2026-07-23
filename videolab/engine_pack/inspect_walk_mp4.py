"""QC for eight-direction walking MP4s: motion, center drift, direction/orientation sanity.

Generalized from the C01 round-local script
(output/rounds/20260702_1243_C01_field_page/04_grok_video_qc/inspect_walk_mp4.py)
so every round uses the same reference-alignment gate.

The orientation check compares sampled MP4 frames against the labeled Codex
direction references (split_centered/*_<direction>_centered.png). A direction
fails when enough sampled frames are closer to its left/right partner
reference than to its own -- this is what catches mid-clip mirror flips that
motion/center checks miss.

Usage:
    python tools/inspect_walk_mp4.py \
        --mp4-dir  ROUND/03_grok_walk_mp4/mp4 \
        --reference-dir ROUND/01_codex_8dir_generation/split_centered \
        --out-dir  ROUND/04_grok_video_qc

Exit code: 0 = gate_pass true (all 8 directions pass), 1 = FAIL.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    from PIL import Image, ImageChops, ImageStat
except ImportError:
    print("Pillow required: pip install pillow", file=sys.stderr)
    raise

DEFAULT_FF = (r"C:\Users\contr\AppData\Local\Microsoft\WinGet\Packages"
              r"\Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe"
              r"\ffmpeg-8.1.1-essentials_build\bin\ffmpeg.exe")

DIRECTIONS = [
    "front", "left", "right", "back",
    "front_left", "front_right", "back_left", "back_right",
]
# Longest names first so "front_left" is matched before "left".
MATCH_ORDER = sorted(DIRECTIONS, key=len, reverse=True)

PARTNERS = {
    "left": "right",
    "right": "left",
    "front_left": "front_right",
    "front_right": "front_left",
    "back_left": "back_right",
    "back_right": "back_left",
}

# 180-degree / point-opposite directions. The diagonal pairs
# front_right<->back_left and front_left<->back_right differ on BOTH axes, so
# the left/right PARTNERS check alone cannot detect them swapped.
OPPOSITE = {
    "front": "back", "back": "front",
    "left": "right", "right": "left",
    "front_left": "back_right", "back_right": "front_left",
    "front_right": "back_left", "back_left": "front_right",
}


def relation(expected: str, other: str) -> str:
    if other == expected:
        return "same"
    if other == OPPOSITE.get(expected):
        return "OPPOSITE (180 reversal)"
    if other == PARTNERS.get(expected):
        return "L/R mirror"
    return "wrong direction"

KEY_RGB = (255, 0, 255)
NORM_SIZE = (96, 160)
NORM_MARGIN = 4


def direction_of(stem: str) -> str | None:
    for d in MATCH_ORDER:
        if re.search(rf"(^|_){re.escape(d)}(_|$)", stem):
            return d
    return None


def is_key(px: tuple[int, int, int], tol: int) -> bool:
    return all(abs(px[c] - KEY_RGB[c]) <= tol for c in range(3))


def fg_center(img: Image.Image, tol: int) -> tuple[float, float] | None:
    w, h = img.size
    xs: list[int] = []
    ys: list[int] = []
    px = img.convert("RGB").load()
    for y in range(h):
        for x in range(w):
            if not is_key(px[x, y], tol):
                xs.append(x)
                ys.append(y)
    if not xs:
        return None
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def frame_diff(a: Image.Image, b: Image.Image) -> float:
    pa = a.convert("RGB").resize((128, 128))
    pb = b.convert("RGB").resize((128, 128))
    da = pa.tobytes()
    db = pb.tobytes()
    if len(da) != len(db):
        return 0.0
    total = 0
    for i in range(0, len(da), 3):
        total += abs(da[i] - db[i]) + abs(da[i + 1] - db[i + 1]) + abs(da[i + 2] - db[i + 2])
    return total / (len(da) // 3 * 3)


def bg_magenta_mask(img: Image.Image, magenta_thr: int) -> Image.Image:
    """背景マスク (L画像、255=背景) -- 連結成分キーイング。

    「マゼンタっぽさ」だけで全画素を抜くと、キャラ内部のマゼンタ寄りの
    色 (髪のメッシュ等) まで背景扱いで抜ける (イチカ, 2026-07-04)。
    背景と判定するのは:
      ① 画面の縁から連結しているマゼンタっぽい領域 (輪郭ハロー含む)
      ② ほぼ純マゼンタ (腕組み等でキャラに囲まれた閉じた背景穴)
    のみ。キャラ内部のマゼンタ寄りは縁と連結せず①②とも外れるので残る。
    numpyが無い環境では従来の全画素しきい値にフォールバック。
    """
    r, g, b = img.convert("RGB").split()
    rb_min = ImageChops.darker(r, b)
    magentaness = ImageChops.subtract(rb_min, g)
    try:
        import numpy as np
    except ImportError:
        return magentaness.point(lambda v: 255 if v >= magenta_thr else 0)
    m = np.asarray(magentaness)
    loose = m >= magenta_thr
    tight = m >= 200  # ほぼ純マゼンタ = 確実に背景 (閉じた穴の中身も)
    # 種は「画面の縁のloose」+「純マゼンタ領域ぜんぶ」。後者を種に
    # 含めることで、腕組み・股下のような閉じた背景穴の縁のにじみ
    # (70〜199) も穴の核から連結で回収できる (縁だけを種にすると
    # 閉じた穴の1pxにじみリングが残る理論的隙間があった)
    reach = tight.copy()
    reach[0, :] |= loose[0, :]
    reach[-1, :] |= loose[-1, :]
    reach[:, 0] |= loose[:, 0]
    reach[:, -1] |= loose[:, -1]
    # 測地拡張のベクトル化: 列/行ごとの「連続したloose区間 (run)」に
    # seedが1つでもあれば区間全体が到達可能。非loose画素のcumsumが
    # run idになるので、seedのrun idをmaximum.accumulateで流せば
    # 1方向まるごと1スキャンで埋まる。半解像度近似は袖と体の隙間の
    # ような細い背景水路を取りこぼし紫フリンジが残った (C27) ので、
    # 満解像度で回す。呼び出し側のアルファキャッシュが実効コストを
    # 吸収する (builder load_keyed)。
    rid_v = np.cumsum(~loose, axis=0, dtype=np.int32)
    rid_h = np.cumsum(~loose, axis=1, dtype=np.int32)

    def _fill(rc, rid, axis, rev):
        if rev:
            rc = rc[::-1] if axis == 0 else rc[:, ::-1]
            rid = rid[::-1] if axis == 0 else rid[:, ::-1]
        s = np.maximum.accumulate(np.where(rc, rid, -1), axis=axis)
        f = s == rid
        if rev:
            f = f[::-1] if axis == 0 else f[:, ::-1]
        return f

    prev = -1
    for _ in range(8):  # 実画像は2〜3周で収束 (凹んだポケット対策の反復)
        cur = int(reach.sum())
        if cur == prev:
            break
        prev = cur
        reach |= _fill(reach, rid_v, 0, False) & loose   # 下へ
        reach |= _fill(reach, rid_v, 0, True) & loose    # 上へ
        reach |= _fill(reach, rid_h, 1, False) & loose   # 右へ
        reach |= _fill(reach, rid_h, 1, True) & loose    # 左へ
    # 細い閉じ隙間の回収 (イチカの股下, 2026-07-04): 幅数pxの股間・脇の
    # 隙間は動画圧縮で両側の服と混ざり「純マゼンタの核」を持たない
    # (全部にじみ値70-199) ため、連結でも純核種でも取れない。実測では
    #   髪メッシュ: 最大m 91-99 (頭部)
    #   股下の隙間: 最大m 117-196 (脚部)
    # と分離できるので、残存blobを「最大m>=120」または「体の下半分かつ
    # m>=85」を種に丸ごと回収する。頭部の低彩度メッシュだけが生き残る。
    kept = loose & ~reach
    if kept.any():
        fg_rows = np.where((~loose).any(axis=1))[0]
        split = (fg_rows[0] + int(0.55 * (fg_rows[-1] - fg_rows[0]))
                 if len(fg_rows) else m.shape[0])
        rows = np.arange(m.shape[0])[:, None]
        seed2 = kept & ((m >= 120) | ((m >= 85) & (rows >= split)))
        if seed2.any():
            k_v = np.cumsum(~kept, axis=0, dtype=np.int32)
            k_h = np.cumsum(~kept, axis=1, dtype=np.int32)
            grow = seed2
            prev2 = -1
            for _ in range(8):
                c2 = int(grow.sum())
                if c2 == prev2:
                    break
                prev2 = c2
                grow = grow | (_fill(grow, k_v, 0, False) & kept)
                grow = grow | (_fill(grow, k_v, 0, True) & kept)
                grow = grow | (_fill(grow, k_h, 1, False) & kept)
                grow = grow | (_fill(grow, k_h, 1, True) & kept)
            reach = reach | grow
    return Image.fromarray(np.where(reach, 255, 0).astype(np.uint8))


def despill_magenta(rgba: Image.Image, alpha: Image.Image,
                    strength: float = 1.0) -> Image.Image:
    """前景に残ったマゼンタ被り(スピル)を抜く。

    キーイングはアルファを二値で立てるだけでRGBを直さないので、背景と
    前景が混ざった画素 — magentaness が閾値(70)未満の 40〜69 の帯 — が
    **不透明のまま**残る。その帯はちょうど輪郭1pxと細い髪束の幅に一致
    するため、髪束がピンクに染まり、睫毛や瞳の縁にも紫が乗る
    (2026-07-23の並列監査で確定。最新シートの半透明縁に純マゼンタが
    272px=旧方式の9.4倍あった)。

    式は標準的なVlahos: m = min(R,B) - G が正なら R,B を G 側へ寄せる。
    **大きな面積のマゼンタ/ピンクの衣装や髪は壊さない** — 対象を
    「アルファの境界から2px以内」に限る (内部の意図的なピンクは無傷)。
    numpy が無い環境では素通し。"""
    try:
        import numpy as np
    except ImportError:
        return rgba
    a = np.asarray(rgba.convert("RGBA")).astype(np.int16)
    al = np.asarray(alpha)
    fg = al > 0
    if not fg.any():
        return rgba
    # アルファ境界の2px帯 (前景側)。膨張は shift-OR で足りる
    edge = np.zeros_like(fg)
    bgm = ~fg
    for _ in range(2):
        g = bgm.copy()
        g[1:, :] |= bgm[:-1, :]
        g[:-1, :] |= bgm[1:, :]
        g[:, 1:] |= bgm[:, :-1]
        g[:, :-1] |= bgm[:, 1:]
        bgm = g
    edge = bgm & fg
    m = np.minimum(a[..., 0], a[..., 2]) - a[..., 1]
    hit = edge & (m > 0)
    if hit.any():
        cut = (m * float(strength)).clip(0, 255).astype(np.int16)
        a[..., 0] = np.where(hit, a[..., 0] - cut, a[..., 0])
        a[..., 2] = np.where(hit, a[..., 2] - cut, a[..., 2])
    out = Image.fromarray(a.clip(0, 255).astype("uint8"), "RGBA")
    out.putalpha(alpha)
    return out


def key_magenta_alpha(img: Image.Image, magenta_thr: int) -> Image.Image:
    """Convert magenta-background frames/references to RGBA foreground images."""
    rgba = img.convert("RGBA")
    a = rgba.getchannel("A")
    bg = bg_magenta_mask(rgba, magenta_thr)
    alpha = ImageChops.multiply(a, ImageChops.invert(bg))
    if os.environ.get("SM_DESPILL", "on").strip().lower() not in (
            "0", "off", "false", "no"):
        return despill_magenta(rgba, alpha)
    out = rgba.copy()
    out.putalpha(alpha)
    return out


def normalize_foreground(img: Image.Image, magenta_thr: int) -> Image.Image:
    """Crop foreground and fit it to a stable canvas for direction comparison."""
    rgba = key_magenta_alpha(img, magenta_thr)
    bb = rgba.getchannel("A").getbbox()
    canvas = Image.new("RGBA", NORM_SIZE, (0, 0, 0, 0))
    if not bb:
        return canvas
    crop = rgba.crop(bb)
    scale = min(
        (NORM_SIZE[0] - 2 * NORM_MARGIN) / crop.size[0],
        (NORM_SIZE[1] - 2 * NORM_MARGIN) / crop.size[1],
    )
    nw = max(1, round(crop.size[0] * scale))
    nh = max(1, round(crop.size[1] * scale))
    crop = crop.resize((nw, nh), Image.LANCZOS)
    canvas.alpha_composite(crop, ((NORM_SIZE[0] - nw) // 2, (NORM_SIZE[1] - nh) // 2))
    return canvas


def norm_diff(a: Image.Image, b: Image.Image) -> float:
    d = ImageChops.difference(a, b)
    s = sum(ImageStat.Stat(d).sum)
    n = a.size[0] * a.size[1] * len(a.getbands())
    return s / n if n else 0.0


def load_references(ref_dir: Path, magenta_thr: int) -> dict[str, Image.Image]:
    refs: dict[str, Image.Image] = {}
    # Prefer *_centered.png over raw splits when both exist.
    for pattern in ("*centered*.png", "*.png"):
        for path in sorted(ref_dir.glob(pattern)):
            d = direction_of(path.stem)
            if d is None or d in refs:
                continue
            refs[d] = normalize_foreground(Image.open(path), magenta_thr)
    return refs


def find_mp4s(mp4_dir: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in sorted(mp4_dir.glob("*.mp4")):
        d = direction_of(path.stem)
        if d is not None and d not in found:
            found[d] = path
    return found


def inspect_orientation(label: str, frames: list[Path], refs: dict[str, Image.Image],
                        args) -> dict:
    rec: dict = {
        "checked": label in refs,
        "partner": PARTNERS.get(label),
        "margin": args.orientation_margin,
        "min_bad_frames": args.orientation_min_bad_frames,
        "partner_better_frames": [],
        "best_mismatch_frames": [],
        "pass": True,
    }
    if label not in refs:
        rec["error"] = "missing_expected_reference"
        rec["pass"] = False
        return rec

    partner = PARTNERS.get(label)
    expected_ref = refs[label]
    partner_ref = refs.get(partner) if partner else None

    for fp in frames:
        norm = normalize_foreground(Image.open(fp), args.magenta_thr)
        expected = norm_diff(norm, expected_ref)
        ranked = sorted((norm_diff(norm, ref), name) for name, ref in refs.items())
        best_diff, best_name = ranked[0]
        if best_name != label and best_diff + args.orientation_margin < expected:
            rec["best_mismatch_frames"].append({
                "frame": fp.name,
                "best": best_name,
                "best_diff": round(best_diff, 3),
                "expected_diff": round(expected, 3),
            })
        if partner_ref is not None:
            partner_diff = norm_diff(norm, partner_ref)
            if partner_diff + args.orientation_margin < expected:
                rec["partner_better_frames"].append({
                    "frame": fp.name,
                    "partner_diff": round(partner_diff, 3),
                    "expected_diff": round(expected, 3),
                    "best": best_name,
                    "best_diff": round(best_diff, 3),
                })

    rec["best_mismatch_count"] = len(rec["best_mismatch_frames"])
    rec["partner_better_count"] = len(rec["partner_better_frames"])

    # Which OTHER direction do the mismatching frames resemble most? Failing on
    # this (not only the L/R partner) is what catches diagonal OPPOSITE swaps
    # such as front_right<->back_left that mirror-only checks miss.
    best_tally: dict[str, int] = {}
    for hit in rec["best_mismatch_frames"]:
        best_tally[hit["best"]] = best_tally.get(hit["best"], 0) + 1
    dominant = max(best_tally, key=best_tally.get) if best_tally else None
    dominant_ct = best_tally.get(dominant, 0) if dominant else 0
    rec["dominant_best"] = dominant
    rec["dominant_best_count"] = dominant_ct
    rec["dominant_best_relation"] = relation(label, dominant) if dominant else None

    # Relation-aware gating (same lesson as inspect_T_sheet): mirror/opposite
    # matches are the real swap modes and use the tight margin; marginal
    # resemblance to an UNRELATED facing (boxy/symmetric characters: profile
    # vs 3/4 sits within ~1.0) is silhouette noise and needs a much larger
    # margin before it may fail a video.
    mismatch_bad = False
    if dominant and dominant_ct >= args.orientation_min_bad_frames:
        needed = (args.orientation_wrong_dir_margin
                  if relation(label, dominant) == "wrong direction"
                  else args.orientation_margin)
        strong = [h for h in rec["best_mismatch_frames"]
                  if h["best"] == dominant
                  and h["best_diff"] + needed <= h["expected_diff"]]
        mismatch_bad = len(strong) >= args.orientation_min_bad_frames
    partner_bad = rec["partner_better_count"] >= args.orientation_min_bad_frames
    rec["pass"] = not (mismatch_bad or partner_bad)
    return rec


# 顔消失ゲートの対象 (前向き系のみ。横顔は顔面積が小さく、後ろ系は
# 顔が無いのが正しい)
FACE_DIRS = ("front", "front_left", "front_right")


def head_band(norm: Image.Image, frac: float = 0.32) -> Image.Image:
    """正規化済み前景 (normalize_foreground) の頭部帯 (fgバウンディング
    ボックス上端から高さ frac)。頭の向き反転は全身diffでは髪・服に
    埋もれて見えない (20260717_2232 真ロップ: 後頭部化した斜め前が
    orientationゲートを素通り) — 帯を頭に絞って調べる。"""
    bb = norm.getchannel("A").getbbox()
    if not bb:
        return norm.crop((0, 0, norm.size[0], 1))
    x0, y0, x1, y1 = bb
    return norm.crop((x0, y0, x1,
                      min(y1, y0 + max(2, round((y1 - y0) * frac)))))


def head_diversity(band: Image.Image) -> float | None:
    """頭部帯の色多様性 (支配色クラスタから遠い前景画素の比率%)。

    顔が見えている頭 = 髪+肌 (+目) の複数色人口で多様性が高い。
    後頭部 = ほぼ髪一色で低い。テンプレ比較や目検出と違い、髪の
    ハイライト・陰影・キャラ画風に依存しない (顔テンプレ比較と
    目暗画素検出は真ロップの髪陰影が支配して弁別不能だった実測
    2026-07-18)。numpy 無し環境は None (検査スキップ)。"""
    try:
        import numpy as np
    except ImportError:
        return None
    a = np.asarray(band.convert("RGBA")).astype(int)
    fg = a[..., 3] > 128
    if fg.sum() < 20:
        return None
    rgb = a[..., :3][fg]
    q = (rgb // 24).astype(np.int32)
    key = q[:, 0] * 10000 + q[:, 1] * 100 + q[:, 2]
    vals, counts = np.unique(key, return_counts=True)
    dom = rgb[key == vals[np.argmax(counts)]].mean(axis=0)
    far = np.abs(rgb - dom).sum(axis=1) > 90
    return 100.0 * float(far.sum()) / float(len(rgb))


def inspect_face(label: str, frames: list[Path], args) -> dict:
    """歩行区間の「顔消失 (後頭部化)」検査 (2026-07-18)。

    VACE骨格+i2vの斜め前は、直立区間 (参照立ち絵に錨止め) では顔が
    出るのに、歩行開始と同時に頭だけ後頭部へ反転して歩行区間の顔が
    消える故障がある (20260717_2232 真ロップ: 8試行全滅なのに従来
    ゲートは全身diffのため素通り)。顔が消えると頭部帯の色人口から
    肌が抜けて色多様性が落ちる — 同一動画の直立サンプル (正解保証)
    を基準に、歩行窓サンプルの多様性の相対ドロップで判定する。
    実測 (真ロップ/ろーら/C43/ロップオリジン 32方向): 後頭部化のみ
    22%ドロップ・健全は全て3%以内 — 既定しきい12%は両側に3倍の余裕。
    歩行窓 (--gait-*) が無ければ先頭サンプルを基準に残り全部を検査。"""
    rec: dict = {"checked": False, "pass": True}
    if str(getattr(args, "face_check", "on")).strip().lower() in (
            "off", "0", "false", "no"):
        rec["skipped"] = "disabled"
        return rec
    if label not in FACE_DIRS:
        return rec
    gs = int(getattr(args, "gait_start_frame", 0) or 0)
    ge = int(getattr(args, "gait_end_frame", 0) or 0)
    idle_divs: list[float] = []
    samples: list[dict] = []
    for i, fp in enumerate(frames):
        # vf_fps は各出力スロットへ丸め込まれた入力フレームの「最後の
        # 1枚」を残す (ffmpeg 8.1.1 実測: 16fps→2fpsでサンプルi=フレーム
        # 8i+3。「最寄り」仮定の round(i*src/fps)=8i は+3ズレで、境界の
        # サンプルが末尾静止をまたいで歩行扱いになる — 敵対的レビュー
        # 2026-07-18で実機反証) → スロット末尾で換算する
        src = round((i + 0.5) * args.src_fps / args.fps) - 1
        is_idle = (src < gs) if ge > 0 else (i == 0)
        in_walk = (gs <= src <= ge) if ge > 0 else (i > 0)
        if not (is_idle or in_walk):
            continue
        band = head_band(normalize_foreground(Image.open(fp),
                                              args.magenta_thr))
        div = head_diversity(band)
        if div is None:
            continue
        if is_idle:
            idle_divs.append(div)
        else:
            samples.append({"frame": fp.name, "src_frame": src,
                            "diversity": round(div, 2)})
    if not idle_divs or len(samples) < 2:
        rec["skipped"] = "no_idle_anchor_or_samples"
        return rec
    base = sum(idle_divs) / len(idle_divs)
    rec["idle_diversity"] = round(base, 2)
    if base < args.face_div_min:
        # 頭部帯がほぼ単色のキャラ (フード等): ドロップが測れない
        rec["skipped"] = "head_monochrome"
        return rec
    rec["checked"] = True
    flips = []
    for s in samples:
        s["drop"] = round(1.0 - s["diversity"] / base, 3)
        if s["drop"] >= args.face_drop:
            flips.append(s)
    rec["samples"] = samples
    rec["flipped_count"] = len(flips)
    rec["flipped_frames"] = [s["frame"] for s in flips]
    rec["pass"] = len(flips) < args.face_min_bad
    return rec


def extract_frames(ffmpeg: str, mp4: Path, out_dir: Path, fps: float,
                   max_frames: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("f_*.png"):
        old.unlink()
    pattern = str(out_dir / "f_%04d.png")
    subprocess.run(
        [ffmpeg, "-y", "-i", str(mp4), "-vf", f"fps={fps},scale=384:-1", pattern],
        check=True,
        capture_output=True,
        creationflags=0x08000000 if sys.platform == "win32" else 0,
    )
    frames = sorted(out_dir.glob("f_*.png"))
    return frames[:max_frames]


def gait_motion_indices(n_samples: int, fps: float, src_fps: float,
                        gait_start: int, gait_end: int) -> list:
    """動き量diff (隣接サンプル差) のうち歩行区間に収まるインデックス。

    サンプル i はソースフレーム round(i*src_fps/fps) 相当として扱う。
    ※実測 (ffmpeg 8.1.1、2026-07-18): vf_fpsはスロット内の「最後の」
    入力フレームを残すため実体は約 +src_fps/(2*fps) 後ろ (16→2fpsで
    8i+3)。この窓は当時の挙動込みで較正・実績PASS済みのため式は
    据え置く (境界1サンプルが静止をまたぐが、ポーズ差が大きく実害
    なしを確認済み)。diff i = サンプル i と i+1 の差なので、両端の
    ソースフレームが [gait_start, gait_end] に入るものだけ残す。
    末尾静止つき配分 (57f=直立6+歩行+静止8) では静止区間のdiff≈0が
    motion_mean を約25-30%希釈して決定論的な偽FAILになる (2026-07-17)
    — 歩行区間だけで測る。"""
    keep = []
    for i in range(max(0, int(n_samples) - 1)):
        a = round(i * src_fps / fps)
        b = round((i + 1) * src_fps / fps)
        if gait_start <= a and b <= gait_end:
            keep.append(i)
    return keep


def inspect_one(label: str, mp4: Path | None, refs: dict[str, Image.Image],
                frames_dir: Path, args) -> dict:
    rec: dict = {"direction": label, "file": mp4.name if mp4 else None, "pass": False}
    if mp4 is None or not mp4.is_file():
        rec["error"] = "missing"
        return rec
    rec["bytes"] = mp4.stat().st_size
    if rec["bytes"] < 50000:
        rec["error"] = "too_small"
        return rec

    try:
        frames = extract_frames(args.ffmpeg, mp4, frames_dir / label,
                                args.fps, args.max_frames)
    except subprocess.CalledProcessError as e:
        rec["error"] = f"ffmpeg_failed: {e.stderr.decode(errors='replace')[:200]}"
        return rec

    rec["frame_count"] = len(frames)
    if len(frames) < 4:
        rec["error"] = "too_few_frames"
        return rec

    centers = []
    diffs = []
    prev = None
    for fp in frames:
        img = Image.open(fp)
        c = fg_center(img, args.key_tol)
        if c:
            centers.append(c)
        if prev is not None:
            diffs.append(frame_diff(prev, img))
        prev = img

    if len(centers) < 3:
        rec["error"] = "no_foreground"
        return rec

    cx = [c[0] for c in centers]
    cy = [c[1] for c in centers]
    rec["center_x_mean"] = round(sum(cx) / len(cx), 2)
    rec["center_y_mean"] = round(sum(cy) / len(cy), 2)
    rec["center_x_spread"] = round(max(cx) - min(cx), 2)
    rec["center_y_spread"] = round(max(cy) - min(cy), 2)
    rec["motion_mean"] = round(sum(diffs) / len(diffs), 3) if diffs else 0.0
    rec["motion_max"] = round(max(diffs), 3) if diffs else 0.0

    # 末尾静止つき配分では動き量を歩行区間 [gait_start, gait_end] だけで
    # 測る (--gait-end-frame>0 かつ --gait-dirs が合う方向のみ。窓が
    # 小さすぎるときは従来の全区間へフォールバック)
    _gd = [s.strip() for s in
           (getattr(args, "gait_dirs", "all") or "all").split(",")]
    if (int(getattr(args, "gait_end_frame", 0) or 0) > 0
            and ("all" in _gd or label in _gd) and diffs):
        keep = gait_motion_indices(len(frames), args.fps, args.src_fps,
                                   int(args.gait_start_frame),
                                   int(args.gait_end_frame))
        keep = [i for i in keep if i < len(diffs)]
        if len(keep) >= 2:
            rec["motion_mean_all"] = rec["motion_mean"]
            rec["motion_mean"] = round(
                sum(diffs[i] for i in keep) / len(keep), 3)
            rec["motion_window"] = [int(args.gait_start_frame),
                                    int(args.gait_end_frame)]

    rec["pass_motion"] = rec["motion_mean"] >= args.motion_min
    rec["pass_center"] = (rec["center_x_spread"] <= args.center_x_max
                          and rec["center_y_spread"] <= args.center_y_max)
    rec["orientation"] = inspect_orientation(label, frames, refs, args)
    rec["pass_orientation"] = bool(rec["orientation"].get("pass"))
    rec["face"] = inspect_face(label, frames, args)
    rec["pass_face"] = bool(rec["face"].get("pass", True))
    rec["pass"] = (rec["pass_motion"] and rec["pass_center"]
                   and rec["pass_orientation"] and rec["pass_face"])
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mp4-dir", required=True,
                    help="directory with the eight *_<direction>_walk*.mp4 files")
    ap.add_argument("--reference-dir", required=True,
                    help="Codex split_centered directory with labeled *_<direction>_centered.png")
    ap.add_argument("--out-dir", required=True,
                    help="QC output directory (manifest + qc_frames/)")
    ap.add_argument("--ffmpeg", default=DEFAULT_FF if Path(DEFAULT_FF).is_file() else "ffmpeg")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--max-frames", type=int, default=12)
    ap.add_argument("--key-tol", type=int, default=40)
    ap.add_argument("--magenta-thr", type=int, default=70)
    ap.add_argument("--motion-min", type=float, default=1.5)
    ap.add_argument("--gait-start-frame", type=int, default=0,
                    help="歩行開始のソースフレーム番号 (walk_layoutのidle_n)。"
                         "--gait-end-frame とセットで動き量を歩行区間だけで"
                         "測る (末尾静止8fが動き量を希釈する偽FAIL対策)")
    ap.add_argument("--gait-end-frame", type=int, default=0,
                    help="歩行終端のソースフレーム番号 (0=窓なし=従来)")
    ap.add_argument("--gait-dirs", default="all",
                    help="動き量窓を適用する方向 (カンマ区切り or all)")
    ap.add_argument("--src-fps", type=float, default=16.0,
                    help="ソース動画のfps (サンプル→ソースフレーム換算用)")
    ap.add_argument("--center-x-max", type=float, default=40.0)
    ap.add_argument("--center-y-max", type=float, default=50.0)
    ap.add_argument("--face-check", default="on",
                    help="歩行中の顔消失 (後頭部化) ゲート on/off (既定on。"
                         "front/front_left/front_rightのみ検査)")
    ap.add_argument("--face-drop", type=float, default=0.12,
                    help="直立基準からの頭部帯色多様性の相対ドロップが"
                         "これ以上のサンプルを後頭部化とみなす (実測: "
                         "後頭部化=0.22 / 健全=0.03以内)")
    ap.add_argument("--face-min-bad", type=int, default=2,
                    help="後頭部化サンプルがこの数に達したらFAIL")
    ap.add_argument("--face-div-min", type=float, default=8.0,
                    help="直立の頭部帯多様性がこれ未満 (ほぼ単色頭) の"
                         "キャラは検査をスキップ")
    ap.add_argument("--orientation-margin", type=float, default=0.75,
                    help="partner/reference diff margin before counting a mismatch frame")
    ap.add_argument("--orientation-wrong-dir-margin", type=float, default=8.0,
                    help="stronger margin for NON-mirror/NON-opposite 'wrong "
                         "direction' matches: walking poses vs standing refs "
                         "drift up to ~7.6 toward an adjacent 45-degree "
                         "facing on clean videos (C07). Mirror/opposite "
                         "swaps still gate at --orientation-margin")
    ap.add_argument("--orientation-min-bad-frames", type=int, default=2,
                    help="fail a direction when this many frames prefer the partner reference")
    args = ap.parse_args()

    mp4_dir = Path(args.mp4_dir)
    ref_dir = Path(args.reference_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "qc_frames"

    refs = load_references(ref_dir, args.magenta_thr)
    mp4s = find_mp4s(mp4_dir)
    results = [inspect_one(d, mp4s.get(d), refs, frames_dir, args) for d in DIRECTIONS]
    passed = sum(1 for r in results if r.get("pass"))

    summary = {
        "qc_version": "walk_mp4_direction_v5_faceband",
        "mp4_dir": str(mp4_dir.resolve()),
        "reference_dir": str(ref_dir.resolve()),
        "references_loaded": sorted(refs),
        "references_missing": [d for d in DIRECTIONS if d not in refs],
        "orientation_margin": args.orientation_margin,
        "orientation_min_bad_frames": args.orientation_min_bad_frames,
        "criteria": {
            "motion_min": args.motion_min,
            "center_x_max": args.center_x_max,
            "center_y_max": args.center_y_max,
            "face_check": args.face_check,
            "face_drop": args.face_drop,
            "face_min_bad": args.face_min_bad,
            "face_div_min": args.face_div_min,
        },
        "directions_total": len(DIRECTIONS),
        "directions_pass": passed,
        "gate_pass": passed == len(DIRECTIONS),
        "results": results,
    }
    manifest = out_dir / "walk_qc_manifest.json"
    manifest.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    for r in results:
        flags = []
        if not r.get("pass_motion", False):
            flags.append("motion")
        if not r.get("pass_center", False):
            flags.append("center")
        if not r.get("pass_orientation", False):
            flags.append("orientation")
        if not r.get("pass_face", True):
            flags.append("face_lost")
        state = "PASS" if r.get("pass") else "FAIL(" + (",".join(flags) or r.get("error", "?")) + ")"
        ori = r.get("orientation", {})
        looks = ori.get("dominant_best")
        looks_txt = (f" looks_like={looks}[{ori.get('dominant_best_relation')}]x{ori.get('dominant_best_count')}"
                     if looks else "")
        face = r.get("face", {})
        face_txt = (f" face_flip={face.get('flipped_count')}"
                    if face.get("checked") else "")
        print(f"  {r['direction']:<12} {state:<24} motion={r.get('motion_mean', '-')} "
              f"cx={r.get('center_x_spread', '-')} cy={r.get('center_y_spread', '-')} "
              f"partner_better={ori.get('partner_better_count', '-')}{looks_txt}{face_txt}")
    print(f"manifest: {manifest}")
    print(f"WALK MP4 GATE: {'PASS' if summary['gate_pass'] else 'FAIL'} "
          f"({passed}/{len(DIRECTIONS)})")
    return 0 if summary["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
