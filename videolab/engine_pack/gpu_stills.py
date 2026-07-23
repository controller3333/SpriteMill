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
# ★背景と同系色の服を禁じる。key_to_magenta は「縁の中央値に近い画素」を
# 背景として塗り潰すので、背景色に近い衣装は**穴になる** (2026-07-23実走:
# 灰背景×灰系チャイナ→胴が丸ごとRGB(255,0,255)に化け、歩行シートでは
# 下着姿に見えた)。ユーザー指摘「衣装が消えたのはマゼンタの服だから」。
BG_NEG = ("gradient background, scenery, floor, "
          "gray clothes, gray dress, magenta clothes, purple clothes, "
          "clothes same color as background")

# 1体だけ描かせる。ControlNetに骨格を1体分しか渡していなくても、SDXLは
# 縦長キャンバスを「ターンアラウンド表」で埋めることがある (2026-07-23実走:
# 5体並びが出て、うち4体は顔が空白だった)。danbooru語の solo が最も効く。
SOLO_TAG = "solo"
SOLO_NEG = ("multiple characters, duplicate character, character sheet, "
            "multiple views, turnaround sheet, collage, 2girls, 3girls, "
            "crowd, cropped")

# 服を必ず着せる。依頼の衣装が訳で落ちると (「青チャイナ」→"blue china" は
# 服として読まれない) 素体が出る — 2026-07-23実走で実際に裸の立ち絵が
# 生成され、そのまま回転→歩行へ流れかけた。配布物なので下限として常に付ける。
CLOTHED_TAG = "fully clothed"
NSFW_NEG = ("nude, naked, nsfw, topless, bottomless, underwear, lingerie, "
            "bare chest, exposed skin, nipples, swimsuit")

# 彩色を必ず要求する。パレット指定が無い依頼では色の手がかりが定型句
# だけになり、白地に輪郭線だけの線画が出る (2026-07-23実障害: 白い線画は
# 縁の中央値キーで丸ごと背景送りになり被写体面積0%で落ちた)。
COLOR_TAG = "colored illustration, solid color fill"
COLOR_NEG = ("monochrome, greyscale, grayscale, lineart, line art, sketch, "
             "uncolored, white background")

# 依頼文でよく出る衣装語の対訳 (機械翻訳が服だと解さない語を補う)
COSTUME_GLOSSARY = (
    ("チャイナドレス", "cheongsam qipao dress"),
    ("チャイナ", "cheongsam qipao dress"),
    ("セーラー", "sailor uniform"),
    ("巫女", "miko shrine maiden outfit"),
    ("袴", "hakama"),
    ("浴衣", "yukata"),
    ("着物", "kimono"),
    ("学ラン", "gakuran school uniform"),
    ("ブレザー", "blazer school uniform"),
    ("メイド", "maid outfit"),
    ("白衣", "lab coat"),
    ("甲冑", "plate armor"),
    ("鎧", "armor"),
    ("法衣", "robe"),
    ("忍装束", "ninja outfit"),
)


# 参考画像から拾ったタグのうち、立ち絵生成に持ち込んではいけないもの。
# ★背景・構図・画風メタは必ず落とす: 参考画像の白背景をそのまま持ち込むと
# 生成物も白背景になり、マゼンタ抜きが成立しない (2026-07-23ユーザー指摘
# 「denoise 0.72では背景がマゼンタにならない」)。頭身・単独性・服の有無は
# こちらが別に指定しているので、それらも捨てて衝突を避ける。
# ★"background" を含むタグは色によらず**全部**落とす (2026-07-23ユーザー
# 指摘「黒でも透明でもついてたら困る」)。部分一致にしてあるので
# black/transparent/two-tone/checkered… どれが来ても確実に消える。
CAPTION_DROP_SUBSTR = ("background", "backdrop", "wallpaper")
# 背景の語が無くても「後ろに何かを描かせる」タグも塞ぐ (場所・天候・
# 光・小道具)。キャラの見た目だけを残すのがこの関数の役目。
CAPTION_DROP = (
    "border", "frame", "letterboxed",
    "indoors", "outdoors", "scenery", "sky", "cloud", "clouds", "sun",
    "moon", "star", "stars", "tree", "trees", "grass", "flower field",
    "wall", "floor", "ceiling", "window", "door", "room", "road",
    "水", "night", "day", "sunset", "sunlight", "lens flare",
    "shadow", "cast shadow", "reflection", "gradient", "vignette",
    "chibi", "solo", "1girl", "1boy", "2girls", "multiple",
    "full body", "fullbody", "upper body", "portrait", "close-up",
    "looking at viewer", "standing", "sitting", "walking",
    "monochrome", "greyscale", "grayscale", "lineart", "sketch",
    "highres", "absurdres", "commentary", "artist name", "signature",
    "watermark", "username", "web address", "text", "english text",
    "nude", "naked", "nsfw",
)


def clean_caption_tags(tags, limit: int = 24) -> str:
    """参考画像のタグ/説明文 → 立ち絵プロンプトに混ぜてよい断片。

    背景・構図・画風のメタを落とし、見た目 (髪・目・服・色・種族) だけを
    残す。tagsは list でも カンマ区切り文字列でもよい。"""
    if isinstance(tags, str):
        items = [t.strip() for t in tags.replace(chr(10), ",").split(",")]
    else:
        items = [str(t).strip() for t in (tags or [])]
    out = []
    for t in items:
        low = t.lower().strip(" ._").replace("_", " ")
        if not low or len(low) > 40:
            continue
        if any(d in low for d in CAPTION_DROP_SUBSTR):
            continue                       # 色を問わず背景タグは全部捨てる
        if any(d == low or d in low.split() or low.endswith(" " + d)
               for d in CAPTION_DROP):
            continue
        if low in [o.lower() for o in out]:
            continue
        out.append(low)
        if len(out) >= limit:
            break
    return ", ".join(out)


def costume_hint(meta: dict) -> str:
    """依頼の原文 (日本語) から衣装語を拾って英語タグにする。

    palette欄に「青チャイナ」のように書かれた衣装は、機械翻訳では
    "blue china" になって服として効かない。原文を直接見て補う。"""
    src = " ".join(str((meta or {}).get(k) or "") for k in (
        "concept_ja", "palette_ja", "silhouette_ja", "notes_ja",
        "concept", "palette", "silhouette", "notes"))
    hits = []
    for ja, en in COSTUME_GLOSSARY:
        if ja in src and en not in hits:
            hits.append(en)
    return ", ".join(hits)

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
    "extra limbs, deformed face, motion blur, background change, "
    "nude, naked, undressing, clothing change"
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
           + ", " + BG_NEG + ", " + SOLO_NEG + ", " + NSFW_NEG
           + ", " + COLOR_NEG)
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


def torso_solid(img) -> float:
    """キー済み立ち絵で、胴の中心線が残っている割合 (0..1)。

    背景と同系色の服はキーで穴になる。面積だけ見ていると髪と手足が残る
    ので通ってしまい、歩行段まで「胴が透明なキャラ」が流れる
    (2026-07-23実走: 灰背景×灰系チャイナで胴がRGB(255,0,255)に化けた)。
    立位の胴は必ず中心線上にあるので、そこを直接見るのが一番確実。"""
    import numpy as np
    a = np.asarray(img.convert("RGB")).astype(int)
    # 下流と同じキー判定 (build_T_sheet / pipeline._keyed と同一式)
    mag = (np.minimum(a[:, :, 0], a[:, :, 2]) - a[:, :, 1]) >= 70
    fg = ~mag
    if not fg.any():
        return 0.0
    ys = np.where(fg.any(axis=1))[0]
    xs = np.where(fg.any(axis=0))[0]
    y0, y1 = int(ys.min()), int(ys.max())
    cx = int((xs.min() + xs.max()) / 2)
    h = y1 - y0 + 1
    band = fg[y0 + int(0.25 * h): y0 + int(0.55 * h),
              max(0, cx - max(2, (xs.max() - xs.min()) // 12)):
              cx + max(2, (xs.max() - xs.min()) // 12) + 1]
    if band.size == 0:
        return 0.0
    return float(band.mean())


def max_area_for(leg_scale) -> float:
    """頭身ごとの「画面占有の上限」。

    ★低頭身ほど大きく写るのが正常: 同じ全高でも 1.2頭身のずんぐり体型は
    8頭身の細身よりずっと面積を食う。上限を8頭身基準(45%)で固定していた
    ため、1.2頭身の依頼(ロップ・leg_scale 0.6)が面積57%で「複数体を並べた
    可能性」と誤判定され、3回とも落ちて依頼ごと失敗していた
    (2026-07-23実障害)。実測アンカー: 8頭身の正常値 16-26%。"""
    try:
        heads = 2.0 * float(leg_scale or 1.0)
    except (TypeError, ValueError):
        heads = 2.0
    heads = max(1.0, min(8.0, heads))
    if heads <= 3.0:
        return 0.72
    if heads <= 4.5:
        return 0.62
    if heads <= 6.0:
        return 0.52
    return 0.45


def subject_parts(img, min_frac: float = 0.02) -> tuple:
    """キー済み立ち絵の前景を連結成分に分ける。

    戻り値 (最大塊の占有率, min_frac以上の塊の数, 縁に接している辺の数)。
    ★ターンテーブルは渡された絵を増幅する。分身・浮いた欠片・見切れを
    抱えたまま回すと8方向すべてがクリーチャーになる (2026-07-23実障害:
    背面セルが顔2つ・耳4本・宙に浮いた耳の欠片つきで出た)。回す前に
    ここで止める。scipy無しで済むよう、縮小マスク上の反復伝播で数える。"""
    import numpy as np
    from PIL import Image as _I
    a = np.asarray(img.convert("RGB")).astype(int)
    dist = (np.abs(a[..., 0] - MAGENTA[0]) + np.abs(a[..., 1] - MAGENTA[1])
            + np.abs(a[..., 2] - MAGENTA[2]))
    fg = dist >= 70
    if not fg.any():
        return 0.0, 0, 0
    h, w = fg.shape
    b = max(1, int(min(h, w) * 0.01))
    edges = sum(bool(x) for x in (
        fg[:b].any(), fg[-b:].any(), fg[:, :b].any(), fg[:, -b:].any()))
    small = np.asarray(_I.fromarray((fg * 255).astype("uint8")).resize(
        (96, 160), _I.NEAREST)) > 127
    lab = np.zeros(small.shape, dtype=np.int32)
    lab[small] = np.arange(1, int(small.sum()) + 1)
    for _ in range(240):
        prev = lab
        m = lab.copy()
        m[1:, :] = np.maximum(m[1:, :], lab[:-1, :])
        m[:-1, :] = np.maximum(m[:-1, :], lab[1:, :])
        m[:, 1:] = np.maximum(m[:, 1:], lab[:, :-1])
        m[:, :-1] = np.maximum(m[:, :-1], lab[:, 1:])
        lab = np.where(small, m, 0)
        if np.array_equal(lab, prev):
            break
    ids, counts = np.unique(lab[lab > 0], return_counts=True)
    if not len(counts):
        return 0.0, 0, edges
    tot = float(counts.sum())
    big = int((counts >= tot * min_frac).sum())
    return float(counts.max()) / tot, big, edges


def stills_ok(img, min_area: float = 0.04, min_height: float = 0.45,
              max_area: float = 0.45, min_torso: float = 0.6,
              max_height: float = 0.97, min_main: float = 0.82,
              max_parts: int = 3) -> tuple:
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
    # ★ターンテーブルに回す前の「成立している見た目」検査 (2026-07-23)
    if height > max_height:
        return False, (f"上下が見切れています (縦{height * 100:.0f}% — "
                       "頭か足がキャンバス外に出ている)")
    main, parts, edges = subject_parts(img)
    if parts > max_parts:
        return False, (f"体がばらけています (大きな塊が{parts}個 — "
                       "分身や浮いた欠片の可能性)")
    if main < min_main:
        return False, (f"胴体が一塊になっていません (最大塊{main * 100:.0f}%"
                       " — 分身や欠片が混ざっている)")
    if edges >= 3:
        return False, f"四方が見切れています (縁に接する辺{edges}/4)"
    torso = torso_solid(img)
    if torso < min_torso:
        return False, (f"胴が抜けています (中心線の残り{torso * 100:.0f}% — "
                       "背景と同系色の服がキーで消えた可能性)")
    return True, (f"面積{area * 100:.1f}% 縦{height * 100:.0f}% "
                  f"胴{torso * 100:.0f}% 一塊{main * 100:.0f}%")


def concept_to_illustrious_prompt(meta: dict) -> str:
    """依頼メタから Illustrious 向けタグ寄りプロンプトを組む (英語化済み)。

    ★語順が効く: SDXLのCLIPは77トークンで切れ、前方ほど強い。定型句を
    先に積むと依頼のキャラ描写が後方へ押し出されて効かなくなる —
    2026-07-23の実走で、定型句を増やした版が「顔なし・無彩色・後ろ向き」
    を連発した (増やす前の短い版は一発で金髪・青チャイナ・緑目が出た)。
    **キャラ描写を先、技術タグを後**に置くこと。"""
    m = _en_meta(meta)
    prop = proportion_tags(m.get("leg_scale"))[0]
    if m.get("illustrious_tags"):
        head = str(m["illustrious_tags"]).strip()
    else:
        head = f"masterpiece, best quality, solo, {prop}"
    concept = str(m.get("concept_en") or m.get("concept") or "").strip()
    palette = str(m.get("palette_en") or m.get("palette") or "").strip()
    sil = str(m.get("silhouette_en") or m.get("silhouette") or "").strip()
    notes = str(m.get("notes_en") or m.get("notes") or "").strip()
    cos = costume_hint(meta)
    parts = [head]                     # ① キャラの中身 (依頼の言葉)
    cap = str(m.get("caption_tags") or "").strip()
    if cap:                            # 参考画像から読んだ見た目 (背景は除去済)
        parts.append(cap)
    if concept:
        parts.append(concept)
    if palette:
        parts.append(palette)
    if cos:
        parts.append(cos)
    if sil:
        parts.append(sil)
    parts.append(CLOTHED_TAG)          # ② 構図と作画の指定
    parts.append(COLOR_TAG)
    parts.append("full body, standing, front view, front lighting, "
                 f"flat color, simple background, {BG_TAG}")
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
        f"character sprite, {prop}, {CLOTHED_TAG}, crisp outlines, "
        f"flat colors, one soft shadow, {BG_TAG}, uniform background with "
        "no gradient, no floor, no text, no watermark, no border, "
        "not a turnaround sheet.",
        f"Character: {concept}.",
    ]
    cos = costume_hint(meta)
    if cos:
        bits.append(f"Outfit: {cos}.")
    if palette:
        bits.append(f"Palette: {palette}.")
    if sil:
        bits.append(f"Silhouette cues: {sil}.")
    if notes:
        bits.append(notes)
    return " ".join(bits)


def facing_ref_path(posesets_dir=None, kind: str = "right"):
    """向き判定の参照 (C17 の指示書) のパス。無ければ None。

    kind="right" = 左右の判定用 / kind="front" = 正面コマ探しの物差し。
    較正の出典がC17なので、差し替え式の現行poseset (色分け人形・左右が
    ほぼ対称) では代用しない — 感度が落ちて誤判定側に倒れる
    (pipeline._facing_margin_vs_poseset の注記と同じ理由)。"""
    name = "front_1.png" if kind == "front" else "right_1.png"
    root = Path(posesets_dir) if posesets_dir else (
        Path(__file__).resolve().parent.parent / "posesets")
    here = Path(__file__).resolve().parent
    for p in (root / "_old_C17_v2" / name,
              root / "_old_C17" / name,
              # GPU VMには posesets が無いので engine_pack へ同梱した実体
              # (gce_drive.push が code/engine_pack/ へ送る)
              here / f"facing_ref_{kind}.png"):
        if p.is_file():
            return p
    return None


def find_front_index(frames, ref_front=None) -> int:
    """360°回転の中から**本当の正面コマ**を選ぶ。

    ★立ち絵が正面向きに描かれるとは限らない (実走で後ろ姿・斜めが出た)。
    その場合フレーム0を正面として8等分すると、全方向が丸ごとずれる
    (2026-07-23ユーザー指示「最初に生成された絵が正面向きでなかったときは
    0フレーム目ではなく、360度の中から正面の絵を選ぶ」)。
    判定は2つの合成:
      ① 左右対称性 — 正面と背面だけが鏡映対称に近い
      ② C17 front指示書との近さ — 正面と背面を分ける
    測れないときは 0 (従来動作) を返す。"""
    import numpy as np
    n = len(frames or [])
    if n < 4:
        return 0
    g = _norm64(ref_front) if ref_front is not None else None
    best, best_score = 0, None
    for i in range(n):
        a = _norm64(key_to_magenta(frames[i]))
        if a is None:
            continue
        sym = float(np.abs(a - a[:, ::-1]).mean())      # 小さいほど対称
        score = sym
        if g is not None:
            score = sym + 0.6 * float(np.abs(a - g).mean())
        if best_score is None or score < best_score:
            best, best_score = i, score
    return best


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
                          order: tuple = None, front_idx: int = 0) -> int:
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
    # slot 0..7 → 0 .. (n-1) の 0/8, 1/8, ... を front_idx から数える
    # (正面コマが先頭とは限らないため。周回なので mod で巻き取る)
    off = int(round(slot * (n_frames - 1) / 8.0))
    return int((int(front_idx or 0) + off) % max(1, n_frames - 1))


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
                           order: tuple = None, ref_right=None,
                           front_idx: int = 0) -> dict:
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
        fi = turntable_frame_index(n, d, order=order, front_idx=front_idx)
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
                        front_bytes: bytes | None = None,
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
        "stills_source": ("front_turntable" if front_bytes
                          else "gpu_turntable"),
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
    fr_front = facing_ref_path(kind="front")
    if fr is not None:
        meta["facing_ref"] = fr.name
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("meta.json",
                   json.dumps(meta, ensure_ascii=False, indent=1))
        if fr is not None:
            z.write(fr, "facing_ref_right.png")
        if fr_front is not None:
            z.write(fr_front, "facing_ref_front.png")
        if motion_en:
            z.writestr("motion_prompt.txt", motion_en)
        if ref_bytes:
            z.writestr("ref.png", ref_bytes)
        # ★出来合いの正面立ち絵 (Codexが描いたもの等)。これが入っていれば
        # GPUは生成せず、そのまま360°ターンテーブルへ回す
        # (2026-07-23ユーザー提案「Codexルートも正面だけ出させて、AniSoraに
        # 360度ローテートさせれば連続性維持した8方向を安定して作れる」)。
        if front_bytes:
            z.writestr("front.png", front_bytes)
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
