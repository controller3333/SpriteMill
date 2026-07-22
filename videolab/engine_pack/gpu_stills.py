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

# ターンテーブルのスロット順。**どちらに回るかはAniSoraに任せきりに
# できない** — プロンプトで clockwise と指示しても逆に回ることがあり、
# 2026-07-22の実走では "right" スロットに画面左向きの絵が入った
# (SpriteMillの right = 画面右向き。posesets/right_idle.png が正)。
# 実測 (measure_rotation_sense) で選ぶこと。
TURNTABLE_RIGHT_FIRST = (        # 画面右へ回る: front→front_right→right…
    "front",
    "front_right",
    "right",
    "back_right",
    "back",
    "back_left",
    "left",
    "front_left",
)
TURNTABLE_LEFT_FIRST = (         # 画面左へ回る (上の鏡像)
    "front",
    "front_left",
    "left",
    "back_left",
    "back",
    "back_right",
    "right",
    "front_right",
)
TURNTABLE_CLOCKWISE = TURNTABLE_RIGHT_FIRST      # 旧名 (後方互換)
DIR_INDEX = {
    "front": 1, "left": 2, "right": 3, "back": 4,
    "front_left": 5, "front_right": 6,
    "back_left": 7, "back_right": 8,
}
MAGENTA = (255, 0, 255)

# 立ち絵に注文する背景色。**マゼンタと言ってはいけない** — 2026-07-22/23の
# 実走で、キャラの肌・服まで背景と同じ紅色に塗られ、キーアウトで body ごと
# 消える事故が2回起きた (色名がパレット全体を引っ張る)。切り抜きは
# key_to_magenta が「縁の中央値と近い画素」を落とす方式で背景色を選ばない
# ので、注文するのは「均一であること」だけでよい。
BG_TAG = "plain flat solid light gray background"
BG_NEG = "gray clothes, gray skin, gradient background, scenery, floor"

# 1体だけ描かせる。ControlNetに骨格を1体分しか渡していなくても、SDXLは
# 縦長キャンバスを「ターンアラウンド表」で埋めることがある (2026-07-23実走:
# 5体並びが出て、うち4体は顔が空白だった)。danbooru語の solo が最も効く。
SOLO_TAG = "solo, 1 character, single character, full body visible"
SOLO_NEG = ("multiple characters, duplicate character, character sheet, "
            "multiple views, turnaround sheet, collage, 2girls, 3girls, "
            "crowd, cropped")

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
    """立ち絵用ネガティブ = アダプタ既定 + 頭身の否定 + 背景同化の否定。"""
    neg = (proportion_tags((meta or {}).get("leg_scale"))[1]
           + ", " + BG_NEG + ", " + SOLO_NEG)
    base = (base or "").strip().rstrip(",")
    return f"{base}, {neg}" if base else neg


def subject_coverage(img) -> tuple:
    """キー済み立ち絵の被写体量 (面積比, 縦の占有比)。

    背景と同化した絵を「立ち絵」として通さないための実測。2026-07-23の
    実走で、背景色をプロンプトで名指ししたせいで体まで同色に塗られ、
    キーアウト後に髪と目しか残らない絵がそのまま歩行段まで流れた。"""
    import numpy as np
    a = np.asarray(img.convert("RGB")).astype(int)
    dist = (np.abs(a[..., 0] - MAGENTA[0]) + np.abs(a[..., 1] - MAGENTA[1])
            + np.abs(a[..., 2] - MAGENTA[2]))
    fg = dist >= 70
    h, w = fg.shape
    if not fg.any():
        return 0.0, 0.0
    ys = np.where(fg.any(axis=1))[0]
    return float(fg.sum()) / float(h * w), float(
        ys.max() - ys.min() + 1) / float(h)


def stills_ok(img, min_area: float = 0.04, min_height: float = 0.45,
              max_area: float = 0.45) -> tuple:
    """立ち絵として使えるか (ok, 理由)。閾値は実走4枚の実測から:

      正常 = 面積21.8% / 22.7% (縦90%・97%)
      背景と同化して髪だけ残った絵 = 3.7%
      5体並びのターンアラウンド表 = 56.1%

    下限で「消えた絵」、上限で「複数体で埋めた絵」を弾く。1体の全身は
    縦長キャンバスの細い柱にしかならないので、この窓で分離できる。"""
    area, height = subject_coverage(img)
    if area < min_area:
        return False, f"被写体が小さすぎます (面積{area * 100:.1f}%)"
    if height < min_height:
        return False, f"全身が写っていません (縦{height * 100:.0f}%)"
    if area > max_area:
        return False, (f"画面を埋めすぎです (面積{area * 100:.0f}% — "
                       "複数体を並べた可能性)")
    return True, f"面積{area * 100:.1f}% 縦{height * 100:.0f}%"


def concept_to_illustrious_prompt(meta: dict) -> str:
    """依頼メタから Illustrious 向けタグ寄りプロンプトを組む (英語化済み)。"""
    m = _en_meta(meta)
    prop = proportion_tags(m.get("leg_scale"))[0]
    tags = str(m.get("illustrious_tags") or (
        f"masterpiece, best quality, {SOLO_TAG}, {prop}, "
        "full body, standing, front view, front lighting, flat color, "
        f"simple background, {BG_TAG}")).strip()
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
        "Exactly one single full-body front-facing idle anime game "
        f"character sprite, {prop}, crisp outlines, flat colors, "
        f"one soft shadow, {BG_TAG}, uniform background with no gradient, "
        "no floor, no text, no watermark, no border, "
        "not a turnaround sheet.",
        f"Character: {concept}.",
    ]
    if palette:
        bits.append(f"Palette: {palette}.")
    if sil:
        bits.append(f"Silhouette cues: {sil}.")
    if notes:
        bits.append(notes)
    return " ".join(bits)


def facing_ref_path(posesets_dir=None):
    """向き判定の参照 (C17 の right 指示書) のパス。無ければ None。

    較正の出典がC17なので、差し替え式の現行poseset (色分け人形・左右が
    ほぼ対称) では代用しない — 感度が落ちて誤判定側に倒れる
    (pipeline._facing_margin_vs_poseset の注記と同じ理由)。"""
    root = Path(posesets_dir) if posesets_dir else (
        Path(__file__).resolve().parent.parent / "posesets")
    here = Path(__file__).resolve().parent
    for p in (root / "_old_C17_v2" / "right_1.png",
              root / "_old_C17" / "right_1.png",
              # GPU VMには posesets が無いので engine_pack へ同梱した実体
              # (gce_drive.push が code/engine_pack/ へ送る)
              here / "facing_ref_right.png"):
        if p.is_file():
            return p
    return None


def _norm64(img, flip: bool = False, thr: int = 70):
    """マゼンタ地を抜いて前景bboxで切り、64x128へ正規化した配列。

    pipeline._facing_margin_vs_poseset (C17較正の向き判定) と同一の
    前処理 — 較正値をそのまま引き継ぐため式を変えないこと。"""
    import numpy as np
    from PIL import Image
    a = np.asarray(img.convert("RGB")).astype(int)
    mag = (np.minimum(a[:, :, 0], a[:, :, 2]) - a[:, :, 1]) >= thr
    ys, xs = np.nonzero(~mag)
    if len(ys) < 50:
        return None
    crop = img.convert("RGB").crop((int(xs.min()), int(ys.min()),
                                    int(xs.max()) + 1, int(ys.max()) + 1))
    if flip:
        crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
    return np.asarray(crop.resize((64, 128), Image.LANCZOS)).astype(float)


def facing_margin(img, ref_right):
    """画面右向きらしさ。正=右向き / 負=左向き / None=測れず。

    C17の right 指示書との鏡映比較 (pipeline._facing_margin_vs_poseset と
    同じ計量。較正: 正常 +9.7以上 / 逆 -6.7以下でプロファイルは完全分離)。
    目の明度に依存しないので、平坦な塗りのAI出力でも効く
    (_estimate_yaw はこの実データで flat_lum 全滅だった)。"""
    import numpy as np
    g = _norm64(ref_right)
    own = _norm64(img)
    flp = _norm64(img, flip=True)
    if g is None or own is None or flp is None:
        return None
    return float(np.abs(flp - g).mean()) - float(np.abs(own - g).mean())


def measure_rotation_sense(frames: list, ref_right=None,
                           min_margin: float = 4.0) -> tuple:
    """ターンテーブルの回り方を実測する。

    戻り値 (sense, score): sense=+1 なら画面右へ回る
    (front→front_right→right…)、-1 なら左へ、0 なら判定不能。
    1/4周と3/4周のコマは必ず互いに反対の横向きなので、両方を「右向き
    らしさ」で採点し、その差で向きを決める (片方だけより頑健)。
    2026-07-22の実走データでは 1/4周=-9.46 / 3/4周=+11.63 (差-21.1) と
    はっきり左回りに出た。"""
    n = len(frames or [])
    if n < 4 or ref_right is None:
        return 0, 0.0
    q = max(0, min(n - 1, int(round((n - 1) * 0.25))))
    t = max(0, min(n - 1, int(round((n - 1) * 0.75))))
    mq = facing_margin(key_to_magenta(frames[q]), ref_right)
    mt = facing_margin(key_to_magenta(frames[t]), ref_right)
    if mq is None or mt is None:
        return 0, 0.0
    score = mq - mt                    # 正: 1/4周が右向き = 右回り
    if abs(score) < min_margin:
        return 0, score
    return (1 if score > 0 else -1), score


def turntable_order(sense: int) -> tuple:
    """実測した回り方 → スロット順。判定不能(0)は右回り既定。"""
    return TURNTABLE_LEFT_FIRST if sense < 0 else TURNTABLE_RIGHT_FIRST


def turntable_frame_index(n_frames: int, direction: str,
                          order: tuple = None) -> int:
    """81f等のターンテーブルから方向に対応するフレーム添字を返す。

    先頭=front、一周して末尾もfrontに戻る前提。末尾を除いた (n-1) 区間を
    8等分する。order は measure_rotation_sense の実測から作った
    スロット順 (省略時は右回り既定)。
    """
    if n_frames < 2:
        return 0
    order = order or TURNTABLE_RIGHT_FIRST
    try:
        slot = order.index(direction)
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


def extract_turntable_dirs(frames: list, thr: int = 70,
                           order: tuple = None, ref_right=None) -> dict:
    """連番フレーム (PIL) から8方向の tight crop を返す。

    order 省略時は ref_right (C17 right指示書) で回り方を実測する
    (指示どおりの向きに回るとは限らない)。参照が無ければ右回り既定。
    """
    n = len(frames)
    if order is None:
        order = turntable_order(
            measure_rotation_sense(frames, ref_right)[0])
    crops = {}
    for d in order:
        fi = turntable_frame_index(n, d, order=order)
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
    # 向き判定の物差し (C17 right指示書)。GPU側はこれでターンテーブルの
    # 回り方を実測し、左右取り違えを防ぐ。母艦にしか posesets が無いので
    # パックへ同梱する
    fr = facing_ref_path()
    if fr is not None:
        meta["facing_ref"] = fr.name
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("meta.json",
                   json.dumps(meta, ensure_ascii=False, indent=1))
        if fr is not None:
            z.write(fr, "facing_ref_right.png")
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
