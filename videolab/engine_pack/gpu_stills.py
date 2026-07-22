#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gpu_stills.py -- Codex枠切れ時のGPU立ち絵フォールバック用ヘルパ。

母艦がシードパック (概念文+任意の参考画像) を返すと、GPU側が:
  1. 正面立ち絵を Illustrious (または Qwen) で生成
  2. AniSora Low の360°ターンテーブル i2v で回転動画を作る
  3. 8方向フレームを切り出し・マゼンタキー・センタリング
  4. 既存 walkpack (8方向歩行) に渡す

このモジュールは GPU/diffusers を import しない。角度表・切り出し・
センタリング・プロンプト組み立てだけを持ち、videolab_server と
share_pack / mothership_relay から共有する。
"""
from __future__ import annotations

import io
import json
import math
import re
import zipfile
from pathlib import Path

# 時計回り360° (上から見て front→right→back→left)。
# AniSoraターンテーブル用プロンプトと同じ規約。
TURNTABLE_CLOCKWISE = (
    "front",
    "front_right",
    "right",
    "back_right",
    "back",
    "back_left",
    "left",
    "front_left",
)
DIR_INDEX = {
    "front": 1, "left": 2, "right": 3, "back": 4,
    "front_left": 5, "front_right": 6,
    "back_left": 7, "back_right": 8,
}
MAGENTA = (255, 0, 255)

TURNTABLE_PROMPT = (
    "Full-body turntable animation of exactly the same single anime "
    "character as the reference. Starting from the straight front view, "
    "the character rotates smoothly clockwise around their own vertical "
    "axis through one exact 360-degree revolution and returns to the "
    "straight front view. Their standing pose remains completely unchanged: "
    "arms relaxed down, both feet planted together, no walking and no limb "
    "motion. Show clear evenly spaced front, three-quarter, profile and "
    "back views during the rotation. Preserve face, hair, outfit, colors "
    "and proportions. The entire body, head and both feet remain inside "
    "the frame at the original constant scale. Static locked camera, no "
    "zoom, no dolly, no pan, no orbit. The flat solid magenta background "
    "remains exactly unchanged."
)
TURNTABLE_NEGATIVE = (
    "walking, running, limb motion, pose change, camera movement, zoom, "
    "close-up, crop, body out of frame, scale change, duplicate character, "
    "extra limbs, deformed face, motion blur, background change"
)


def stills_model_default() -> str:
    import os
    m = (os.environ.get("SM_GPU_STILLS_MODEL")
         or os.environ.get("VIDEOLAB_STILLS_MODEL")
         or "illustrious").strip().lower()
    return m if m in ("illustrious", "qwen") else "illustrious"


def _en_meta(meta: dict) -> dict:
    """Codex無しでも日本語依頼を英語化 (mt_en)。失敗時は原文のまま。"""
    try:
        from mt_en import ensure_meta_english
        return ensure_meta_english(dict(meta or {}))
    except Exception:  # noqa: BLE001
        return dict(meta or {})


def proportion_tags(leg_scale) -> tuple:
    """頭身ノブ → (肯定タグ, 否定タグ)。

    SpriteMillの leg_scale は「1.0=2頭身、4.0=8頭身」(head_frac=1/(2×leg))。
    ControlNetへ渡す骨格は必ずこのノブどおりに描かれるので、プロンプト側が
    "chibi" 決め打ちだと骨と言葉が正面衝突する — 2026-07-22の実走で
    「8頭身ローラ」依頼が chibi頭+棒脚のキメラになった実障害の原因。"""
    try:
        heads = 2.0 * float(leg_scale or 1.0)
    except (TypeError, ValueError):
        heads = 2.0
    heads = max(2.0, min(8.0, heads))
    n = int(round(heads))
    if heads <= 3.0:
        return ("chibi, super deformed, 2 heads tall, big head, short limbs",
                "realistic proportions, long legs, tall")
    if heads <= 4.5:
        return (f"{n} heads tall, deformed proportions, child body, "
                "short limbs", "super deformed, tall adult proportions")
    if heads <= 6.0:
        return (f"{n} heads tall, teenage body proportions, slender limbs",
                "chibi, super deformed, big head, short legs")
    return (f"{n} heads tall, tall slender adult proportions, long legs, "
            "small head, realistic body proportions",
            "chibi, super deformed, big head, short legs")


def stills_negative(meta: dict, base: str = "") -> str:
    """立ち絵用ネガティブ = アダプタ既定 + 頭身の否定タグ。"""
    neg = proportion_tags((meta or {}).get("leg_scale"))[1]
    base = (base or "").strip().rstrip(",")
    return f"{base}, {neg}" if base else neg


def concept_to_illustrious_prompt(meta: dict) -> str:
    """依頼メタから Illustrious 向けタグ寄りプロンプトを組む (英語化済み)。"""
    m = _en_meta(meta)
    prop = proportion_tags(m.get("leg_scale"))[0]
    tags = str(m.get("illustrious_tags") or (
        f"masterpiece, best quality, {prop}, full body, standing, "
        "front view, front lighting, flat color, simple background, "
        "magenta background")).strip()
    concept = str(m.get("concept_en") or m.get("concept") or "").strip()
    palette = str(m.get("palette_en") or m.get("palette") or "").strip()
    sil = str(m.get("silhouette_en") or m.get("silhouette") or "").strip()
    notes = str(m.get("notes_en") or m.get("notes") or "").strip()
    parts = [tags]
    if concept:
        parts.append(concept)
    if palette:
        parts.append(f"color palette: {palette}")
    if sil:
        parts.append(f"silhouette: {sil}")
    if notes:
        parts.append(notes)
    return ", ".join(parts)


def concept_to_qwen_prompt(meta: dict) -> str:
    """Qwen-Image 向けの自然文プロンプト (英語化済み)。"""
    m = _en_meta(meta)
    concept = str(m.get("concept_en") or m.get("concept")
                  or "anime game character").strip()
    palette = str(m.get("palette_en") or m.get("palette") or "").strip()
    sil = str(m.get("silhouette_en") or m.get("silhouette") or "").strip()
    notes = str(m.get("notes_en") or m.get("notes") or "").strip()
    prop = proportion_tags(m.get("leg_scale"))[0]
    bits = [
        "Single full-body front-facing idle anime game character sprite, "
        f"{prop}, crisp outlines, flat colors, one soft shadow, "
        "perfectly flat solid chroma-key magenta #FF00FF background, "
        "no floor, no text, no watermark, no border.",
        f"Character: {concept}.",
    ]
    if palette:
        bits.append(f"Palette: {palette}.")
    if sil:
        bits.append(f"Silhouette cues: {sil}.")
    if notes:
        bits.append(notes)
    return " ".join(bits)


def turntable_frame_index(n_frames: int, direction: str) -> int:
    """81f等のターンテーブルから方向に対応するフレーム添字を返す。

    先頭=front、時計回りに一周して末尾もfrontに戻る前提。末尾を除いた
    (n-1) 区間を8等分する。
    """
    if n_frames < 2:
        return 0
    try:
        slot = TURNTABLE_CLOCKWISE.index(direction)
    except ValueError:
        slot = 0
    # slot 0..7 → 0 .. (n-1) の 0/8, 1/8, ...
    return int(round(slot * (n_frames - 1) / 8.0))


def key_to_magenta(img, thr: int = 60):
    """一様背景をマゼンタへ置換 (SDXLがベージュ地を塗る実測への後処理)。"""
    import numpy as np
    from PIL import Image
    a = np.asarray(img.convert("RGB")).astype(int)
    border = np.concatenate([a[0, :], a[-1, :], a[:, 0], a[:, -1]])
    bgc = np.median(border, axis=0)
    mask = np.abs(a - bgc).sum(axis=2) < thr
    a[mask] = MAGENTA
    return Image.fromarray(a.astype("uint8"), "RGB")


def tight_crop_magenta(img, thr: int = 70):
    """マゼンタ背景をキーアウトして前景bboxで切り出す。"""
    import numpy as np
    from PIL import Image
    a = np.asarray(img.convert("RGB")).astype(int)
    # マゼンタ近傍 = 背景
    dist = (np.abs(a[..., 0] - 255) + np.abs(a[..., 1] - 0)
            + np.abs(a[..., 2] - 255))
    fg = dist >= thr
    ys, xs = np.where(fg)
    if len(xs) == 0:
        return img.convert("RGB")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return img.convert("RGB").crop((x0, y0, x1, y1))


def write_centered_crops(crops: dict, out_dir: Path, char_id: str,
                         margin: int = 40) -> dict:
    """方向名→PIL画像を split_centered 契約 (*_NN_dir_centered.png) で保存。"""
    from PIL import Image
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*_centered.png"):
        old.unlink()
    cw = max(c.size[0] for c in crops.values()) + margin * 2
    ch = max(c.size[1] for c in crops.values()) + margin * 2
    paths = {}
    for d, crop in crops.items():
        idx = DIR_INDEX.get(d, 0)
        canvas = Image.new("RGB", (cw, ch), MAGENTA)
        canvas.paste(crop, ((cw - crop.size[0]) // 2,
                            (ch - crop.size[1]) // 2))
        p = out_dir / f"{char_id}_{idx:02d}_{d}_centered.png"
        canvas.save(p)
        paths[d] = p
    return paths


def extract_turntable_dirs(frames: list, thr: int = 70) -> dict:
    """連番フレーム (PIL) から8方向の tight crop を返す。"""
    n = len(frames)
    crops = {}
    for d in TURNTABLE_CLOCKWISE:
        fi = turntable_frame_index(n, d)
        fi = max(0, min(n - 1, fi))
        keyed = key_to_magenta(frames[fi])
        crops[d] = tight_crop_magenta(keyed, thr=thr)
    return crops


def ascii_pack_id(name: str, rid: str = "") -> str:
    """pack_id を ASCII 英数._- だけにする。rid があれば req_<rid8>_ を前置。"""
    import hashlib
    ascii_part = "".join(
        c if (c.isascii() and (c.isalnum() or c in "._-")) else ""
        for c in name).strip("._-")
    if not ascii_part or ascii_part != name:
        h = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
        ascii_part = (ascii_part + "_" if ascii_part else "seed_") + h
    if rid:
        prefix = f"req_{rid[:8]}_"
        if ascii_part.startswith(prefix):
            return ascii_part[:64]
        return (prefix + ascii_part)[:64]
    return ascii_part[:64]


def build_seed_pack_zip(req: dict, rid: str = "",
                        templates_dir: Path | None = None,
                        ref_bytes: bytes | None = None,
                        stills_model: str | None = None,
                        ) -> tuple[str, bytes, dict]:
    """Codex無しでGPUに渡すシードパック (8方向なし) を作る。

    zip:
      meta.json  (stills_source=gpu_turntable, concept 等)
      ref.png    (任意)
      template.json (任意)
    """
    name = str(req.get("name") or "キャラ")[:40]
    cell = str(req.get("cell_size") or "64x128")
    try:
        cw, ch = (int(x) for x in cell.lower().split("x"))
    except ValueError:
        cw, ch = 64, 128
    model = (stills_model or stills_model_default()).strip().lower()
    if model not in ("illustrious", "qwen"):
        model = "illustrious"
    tmpl = str(req.get("template") or "t_spec").strip() or "t_spec"
    meta = {
        "name": name,
        "char_id": f"R{(rid or 'seed')[:6]}",
        "leg_scale": float(req.get("leg_scale") or 1.0),
        "cell_w": cw,
        "cell_h": ch,
        "template": tmpl,
        "body_plan": str(req.get("body_plan") or "ai").strip() or "ai",
        "concept": str(req.get("concept") or "").strip(),
        "palette": str(req.get("palette") or "").strip(),
        "silhouette": str(req.get("silhouette") or "").strip(),
        "notes": str(req.get("notes") or "").strip(),
        "stills_source": "gpu_turntable",
        "stills_model": model,
        "request_id": rid,
        "source_round": f"seed_{rid}" if rid else "seed",
    }
    if req.get("_codex_error"):
        meta["codex_error"] = str(req["_codex_error"])[:400]
    # Codex無し: 機械翻訳で英語化し、歩行用 motion_prompt も同梱
    try:
        from mt_en import ensure_meta_english
        meta = ensure_meta_english(meta)
        meta["mt_source"] = "mt_en"
    except Exception as e:  # noqa: BLE001
        meta["mt_source"] = f"skip:{str(e)[:80]}"
    motion_en = str(meta.get("motion_prompt") or "").strip()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("meta.json",
                   json.dumps(meta, ensure_ascii=False, indent=1))
        if motion_en:
            z.writestr("motion_prompt.txt", motion_en)
        if ref_bytes:
            z.writestr("ref.png", ref_bytes)
        if templates_dir is None:
            templates_dir = Path(__file__).resolve().parent.parent / "templates"
        tp = Path(templates_dir) / f"{tmpl}.json"
        if tp.is_file():
            z.write(tp, "template.json")
    # 表示名だけを ascii 化し、rid 前置は一回だけ
    pid = ascii_pack_id(name, rid=rid)
    return pid, buf.getvalue(), meta


def is_quota_or_stills_error(msg: str) -> bool:
    """Codex枠切れ・認証・一般的な立ち絵失敗をGPUフォールバック対象と判定。"""
    t = (msg or "").lower()
    keys = (
        "usage limit", "quota", "rate limit", "429",
        "try again at", "hit your usage",
        "立ち絵生成が完了しませんでした",
        "codex", "refresh token", "unauthorized", "401",
        "stills_ready", "image generation failed",
        "contact-sheet", "generation failed",
    )
    return any(k in t for k in keys)


_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
