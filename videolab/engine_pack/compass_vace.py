#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""compass_vace.py -- VACE骨格グリッドで8方向を1本にまとめるコンパス生成(実験)。

旧コンパス (canvas_walk mode=compass + anisora i2v) は「セル155px相当では
プロンプトの向きロックが効かず、時間経過で複数セルが回転」して不採用
(2026-07-11)。敗因の回転はVACE骨格が構造的に殺すため、勝ちスタック
(体型フィット骨格 + VACE High→AniSora Low latent直結、2026-07-13
実装) で再挑戦する。方向まとめなら動画生成が 8本 -> 1本。完成動画を
方向別AniSoraへ再入力する旧refineは、hybrid時には行わない。

使い方 (GUIのエンジンコマンドに --compass-test を足すだけ):
  SpriteMill.exe --engine --compass-test
      --round-dir <完了済みラウンド> --video-provider videolab
      --videolab-url ... --videolab-token ... [--compass-size 720x1296]

出力: <round>/compass_test/ に
  canvas_ref.png                       3x3立ち絵キャンバス (生成の参照画)
  pose_grid_f000.png / _f040.png       グリッド骨格の目視確認用フレーム
  canvas_walk.mp4                      生成されたlatent直結の1本
  mp4/{char}_{idx:02d}_{dir}_walkT.mp4 方向別分割 (既存ゲート/シート互換)
  04_video_qc/                         inspect_walk_mp4 の結果
"""
from __future__ import annotations

import base64
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

import pose_video
from canvas_walk import (CANVAS_PROMPT, IDX, LAYOUT_COMPASS, MODES,
                         split_canvas_video)

MAGENTA = (255, 0, 255)
# 方向名のマッチは長い順 (back_right が right に食われないように)
DIRS8 = sorted([d for d in LAYOUT_COMPASS[2] if d], key=len, reverse=True)
# ★"the cape" を名指ししてはいけない (2026-07-23、ロップの実障害で確定)。
# この文は否定ですらなく「マントが自然に垂れている」という**肯定の断言**
# で、しかも歩行経路は guidance=1.0 = CFG無効なので打ち消す術がない。
# 実際の出力は「静止コマにはマント無し・歩行コマは全部マント」——まさに
# 「動きに応じて揺れる布」としてモデルが忠実に描いた結果だった。
# 参照キャンバス側にはマントが1本も無いことも確認済み。
# ★"wind" と書いてはいけない (2026-07-23ユーザー報告「やけに風が吹いて
# いる感じの動画」)。guidance=1.0=CFG無効なので "no wind" は否定にならず、
# wind という語を条件へ撒くだけ — "the cape" で踏んだのと同じ罠。
# 望む状態を、その語を使わずに肯定文で書く。
NO_WIND = (" The hair and clothes hang down and settle against the body, "
           "moving only as much as the character's own steps carry them. "
           "The air around the character is perfectly calm and empty.")


def planned_direction_jobs(mode: str,
                           dirs_subset: list | None = None) -> list[list[str]]:
    """方向の撮り直しをGUIの「方向まとめ」単位へ束ねる。

    compass/8x1 は選択数にかかわらず1キャンバス、4x2 は選択方向が
    前半球・後半球のどちらに属するかで1～2キャンバス、all だけ方向ごとの
    個別ジョブにする。各まとめキャンバスには元の組全体を載せ、出力の
    差し替えだけを dirs_subset に限定するため、部分的な空セルは作らない。
    """
    requested = ({str(d) for d in dirs_subset if d}
                 if dirs_subset else None)
    if mode == "all":
        ordered = [d for d in IDX if requested is None or d in requested]
        return [[d] for d in ordered]
    if mode not in MODES:
        raise ValueError(f"未知の方向まとめモード: {mode}")
    jobs: list[list[str]] = []
    for layout in MODES[mode]:
        group = [d for d in layout[2] if d]
        if requested is None or requested.intersection(group):
            jobs.append(group)
    return jobs


def _frame_count(value: object) -> int:
    """AniSora/Wanの制約 (8k+1) を満たす短いフレーム数へ丸める。"""
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = 49
    return max(9, ((n - 1) // 8) * 8 + 1)


def _load_dir_refs(round_dir: Path) -> tuple[str, dict]:
    """split_centered から 方向->立ち絵パス とキャラ名を取る。"""
    src = round_dir / "01_generation" / "split_centered"
    refs: dict = {}
    char = None
    for p in sorted(src.glob("*_centered.png")):
        stem = p.stem[: -len("_centered")]
        for d in DIRS8:
            if stem.endswith("_" + d):
                refs[d] = p
                base = stem[: -(len(d) + 1)]           # {char}_{idx:02d}
                char = base.rsplit("_", 1)[0]
                break
    missing = [d for d in LAYOUT_COMPASS[2] if d and d not in refs]
    if missing:
        raise SystemExit(f"split_centered に不足方向 {missing} ({src})")
    return char or "char", refs


def compose_reference(refs: dict, width: int, height: int,
                      layout=LAYOUT_COMPASS) -> Image.Image:
    """立ち絵をセルへレターボックス配置した参照キャンバスを作る。

    ★ pose_video._char_box と同一の写像 (min比率スケール+中央寄せ) で
    置くこと — グリッド骨格と参照キャンバスのキャラ位置が画素単位で
    一致するのがコンパス成立の前提 (骨格の幾何矛盾は全体崩壊を招く、
    2026-07-12の教訓)。"""
    cols, rows, dirs = layout
    cw, ch = width // cols, height // rows
    canvas = Image.new("RGB", (width, height), MAGENTA)
    for i, d in enumerate(dirs):
        if d is None:
            continue
        im = refs[d]
        if not hasattr(im, "convert"):
            im = Image.open(im)
        im = im.convert("RGB")
        sc = min(cw / im.width, ch / im.height)
        rs = im.resize((max(1, round(im.width * sc)),
                        max(1, round(im.height * sc))), Image.LANCZOS)
        ox = (i % cols) * cw + (cw - rs.width) // 2
        oy = (i // cols) * ch + (ch - rs.height) // 2
        canvas.paste(rs, (ox, oy))
    return canvas


def _b64_png(img: Image.Image) -> str:
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _knobs_extra() -> dict:
    """videolab_pose_* / SM_VACE_* の実験ノブを extra へ (pipelineと同順)。"""
    import pipeline as pl
    extra = {}
    for k, env, ck in (("vace_base", "SM_VACE_BASE", "videolab_pose_base"),
                       ("vace_experts", "SM_VACE_EXPERTS",
                        "videolab_pose_experts"),
                       ("vace_patch", "SM_VACE_PATCH", "videolab_pose_patch"),
                       ("vace_end", "SM_VACE_END", "videolab_pose_end"),
                       ("vace_lora", "SM_VACE_LORA", "videolab_pose_lora"),
                       ("vace_high_steps", "SM_VACE_HIGH_STEPS",
                        "videolab_pose_high_steps"),
                       ("hybrid_boundary", "SM_VACE_HYBRID_BOUNDARY",
                        "videolab_pose_hybrid_boundary")):
        v = (os.environ.get(env, "").strip()
             or str(pl.CONFIG.get(ck, "") or "").strip())
        if v:
            extra[k] = v
    return extra


def _poll_job(pl, vl: dict, job: str, dest: Path, tag: str,
              span: tuple = (2, 90)) -> None:
    """ジョブをポーリングして dest へ回収 (pipeline._run_job の縮約版)。

    span=(lo, hi): このジョブの進捗0..1を動画フェーズ全体の lo..hi % に
    写像して [VIDEOLAB_PCT] マーカーで出力する (2026-07-14要望「進捗
    ゲージが全く進まないまま止まる」— GUIは段階マーカーでしか動けず、
    方向まとめの動画フェーズには段階が無かった)。"""
    url, token = vl["url"], vl.get("token")
    print(f"[VIDEOLAB_JOB] {job}", flush=True)
    deadline = time.time() + 3600
    seen = 0
    last = ""
    last_pct = -1
    try:
        while time.time() < deadline:
            try:
                st = pl.api_get(f"{url}/status/{job}", token, timeout=60)
            except Exception as e:      # noqa: BLE001
                print(f"  poll 一時エラー(継続): {e}")
                time.sleep(10)
                continue
            for line in (st.get("log") or [])[seen:]:
                print(f"  videolab┃ {line}", flush=True)
            seen = len(st.get("log") or [])
            s = st.get("status")
            try:
                jp = max(0.0, min(1.0, float(st.get("progress") or 0.0)))
            except (TypeError, ValueError):
                jp = 0.0
            pct = int(span[0] + (span[1] - span[0]) * jp)
            if pct > last_pct:
                print(f"[VIDEOLAB_PCT] {pct}", flush=True)
                last_pct = pct
            tagline = f"{s} {st.get('detail', '')}".strip()
            if tagline != last:
                print(f"  {tag} {job}: {tagline}", flush=True)
                last = tagline
            if s == "done":
                pl.download(f"{url}/result/{job}", dest,
                            headers=({"Authorization": f"Bearer {token}"}
                                     if token else None))
                return
            if s in ("error", "cancelled"):
                raise RuntimeError(f"videolab job {s}: {st.get('detail')}")
            time.sleep(8)
        raise TimeoutError(f"videolab job {job} timed out")
    finally:
        print(f"[VIDEOLAB_JOB_END] {job}", flush=True)


def _auto_size(ref) -> tuple:
    """立ち絵の縦横比を維持して総画素≈39万・16の倍数へ (pipelineの
    --videolab-size auto と同一式)。グリッドのセルを方向別生成と同じ
    解像度の土俵へ拡大するために使う (2026-07-13 混成シートでグリッド側
    だけ半分サイズになった対策)。"""
    im = ref if hasattr(ref, "size") else Image.open(ref)
    iw, ih = im.size
    sc = math.sqrt((464 * 848) / float(iw * ih))
    return (max(224, int(round(iw * sc / 16)) * 16),
            max(224, int(round(ih * sc / 16)) * 16))


def _canvas_size(layout, cell: str = "") -> tuple:
    """レイアウトとセル寸法予算 -> キャンバス寸法 (16の倍数)。"""
    cols, rows, _ = layout
    try:
        cw, ch = (int(x) for x in (cell or "240x432").lower().split("x"))
    except ValueError:
        cw, ch = 240, 432
    cw = max(96, round(cw / 16) * 16)
    ch = max(96, round(ch / 16) * 16)
    return cw * cols, ch * rows


# ---- VRAM由来の動的キャンバス設計 (2026-07-13ユーザー要望「ご家庭の
# ゲーミングPCでも8方向まとめ生成をオミットしない」) --------------------
# 係数はA100-40実機のpeak VRAMログから較正 (720x1296 Q4 hybrid):
#   33f=17.7GB / 49f=20.7GB / 81f=26.6GB → モデル約9GB +
#   活性化 ≈ 0.85GB × latentフレーム数 × (画素数/720x1296)
_A_REF = 0.85                 # GB / latentフレーム @720x1296
_PX_REF = 720 * 1296
_MODEL_RES_GB = 9.0           # Q4 GGUF DiT 1体 (handoffは常に1体ずつ)
_MODEL_OFF_GB = 2.0           # block offload時の常駐分 (概算・実験的)
_SAFETY_GB = 1.5              # CUDAコンテキスト・断片化の口銭


def _lat_frames(nf: int) -> int:
    return (nf - 1) // 4 + 2      # +1=VACE参照タイムスロット


def plan_canvas(free_gb: float, nf_req: int, cols: int, rows: int,
                base_cell: tuple = (240, 432), floor_w: int = 160,
                min_w: int = 128) -> dict:
    """空きVRAMからセル寸法・フレーム数・オフロード要否を決める純関数。

    後退ラダー (速度優先: 常駐のまま縮める → フレームを削る → 最終手段
    としてblock offload):
      ①常駐でセル縮小 (下限floor_w=160、品質バー155px教訓の上)
      ②フレーム 57→41→33 (41=直立6+1周期26f+末尾静止8: walk_layoutの
        末尾静止対応 2026-07-17 で「period34スロモ」だった41fが
        直立コマ供給つきの中間段として復権。33=直立6+1周期26fの
        実績構成で最終段=末尾静止なし・直立コマは先頭窓へ後退)
      ③block offload (遅いが8方向1本は維持) で同ラダー
      ④どうしても入らなければ min_w まで縮めて警告付き続行
    戻り値: {cell:(w,h), nf, offload:bool, note:str}"""
    bw, bh = base_cell

    def _snap(wpx: float) -> tuple:
        w16 = max(min_w, int(wpx // 16) * 16)
        h16 = max(96, round(w16 * bh / bw / 16) * 16)
        return w16, h16

    def _act(w: int, h: int, nf: int) -> float:
        return _A_REF * _lat_frames(nf) * (w * cols * h * rows) / _PX_REF

    ladder_nf = ([nf_req] + [x for x in (41, 33) if nf_req > x])
    for offload in (False, True):
        budget = free_gb - _SAFETY_GB - (_MODEL_OFF_GB if offload
                                         else _MODEL_RES_GB)
        if budget <= 0.3:
            continue
        for nf in ladder_nf:
            base_act = _act(bw, bh, nf)
            s = budget / base_act
            if s >= 1.0:
                return {"cell": (bw, bh), "nf": nf, "offload": offload,
                        "note": "フル解像度"}
            w16, h16 = _snap(bw * math.sqrt(s))
            if w16 >= floor_w and _act(w16, h16, nf) <= budget:
                return {"cell": (w16, h16), "nf": nf, "offload": offload,
                        "note": "セル縮小"}
    # 最終フォールバック: 最小セル+33f+offloadで警告付き続行
    w16, h16 = _snap(min_w)
    return {"cell": (w16, h16), "nf": min(33, nf_req), "offload": True,
            "note": "警告: VRAMが極小のため最小構成 (品質低下・OOMの可能性)"}


def _mirror_dirs() -> set:
    """ミラー生成する方向の集合 (2026-07-15ユーザー発案「右上だけ左右反転
    して左上向きのボーンで対応」)。

    VACEは back_right の脚交差だけを崩すモデル側のクセがある (2キャラ・
    レイアウト非依存で再現、骨格側の左右対称性は数値検証済み — 手前脚の
    OpenPose色の学習偏りが有力)。得意な back_left 骨格で生成し、入口
    (参照立ち絵) と出口 (セル動画) を左右反転して返す。左右非対称な
    デザイン (髪の分け目等) は鏡像になる点に注意。
    ノブ: SM_POSE_MIRROR_DIRS > config videolab_pose_mirror_dirs >
    ★既定 off (2026-07-16実走: ONだと「背中から顔が生えて向きが
    ハチャメチャ」の破綻。参照反転+骨格すげ替えの組み合わせがモデルの
    向き解釈を壊す事例が出たため、素のコンパス生成を既定に戻した。
    ノブは実験用に残す)。"back_right back_left" 等の複数指定可。"""
    import pipeline as pl
    v = str(os.environ.get("SM_POSE_MIRROR_DIRS", "").strip()
            or pl.CONFIG.get("videolab_pose_mirror_dirs", "off")
            or "off").lower()
    if v in ("off", "0", "false", "no", "none"):
        return set()
    return {t for t in v.replace(",", " ").split()
            if t in pose_video.MIRROR_NAME}


def _lr_pin_frames(nf: int, conf: str = "on") -> list:
    """latent固定するフレーム番号 (中間=周期境界+末尾静止/最終) を導く。

    stage2のSDEdit再デノイズは全フレームを自由に動かすため、歩行位相が
    stage1からわずかに流れて「先頭=終端同位相」(コマ選出の前提) が崩れる
    — 周期境界 (2周期目以降の頭) と最終フレームをstage1へ錨止めする
    (2026-07-15要望「中間フレームと最終フレームを固定化」)。末尾静止つき
    配分 (walk_layout の tail>0) では静止フレーム全部を錨止めし、stage2が
    「立ち止まり」を歩き出させないための保険にする (2026-07-17「立ち姿
    シーンを直立コマに」)。フレーム配分は walk_layout が単一情報源なので
    必ずそこから導く。
    49f (直立6+21f×2周期・tail0) -> [27, 48] / 33f -> [32] /
    81f -> [31, 55, 80] / 57f (直立6+21f×2周期+静止8) -> [27, 48..56]。
    conf: "on"=自動 / "off"=無効 / "27,48"=明示リスト (範囲外は落とす)。"""
    nf = int(nf)
    conf = str(conf or "on").strip().lower()
    if conf in ("off", "0", "false", "no", "none"):
        return []
    if conf not in ("on", "1", "true", "yes", "auto", ""):
        out = set()
        for tok in conf.replace(",", " ").split():
            try:
                f = int(tok)
            except ValueError:
                continue
            if 0 < f < nf:
                out.add(f)
        return sorted(out)
    idle, cyc, period, tail = pose_video.walk_layout(nf)
    pins = {int(round(idle + period * k)) for k in range(1, cyc + 1)}
    if tail:
        # 歩行終端 (=最後の周期境界) は上の集合に含まれる。静止区間は
        # 4フレーム=1latentスロットの粒度でサーバが固定する
        pins.update(range(nf - tail, nf))
    else:
        pins.add(nf - 1)
    return sorted(f for f in pins if 0 < f < nf)


# 方向ごとのプロンプト節 (肯定文のみ: 本経路はguidance=1.0=CFGなしで
# ネガティブは無効)。斜め前の文面はcompass角セルで実績のあるC43系。
# rear系の3/4語彙は前半球キャンバスに絶対に同居させないこと (2026-07-18
# 実測: compass上段へrear 3/4語彙を足した版は下段の角まで後ろ向きに化けた
# — 方向タグ全併記が1向きへ収束する既知の汚染と同型)。
_DIR_CLAUSES = {
    "front": ("faces the camera square-on and keeps the whole face "
              "visible while marching in place"),
    # ★左右を必ず明記する (2026-07-19)。以前この2つは一字一句同じ文面で、
    # 左右を示す語が1つも無かった。1レイアウト=1ノイズの同時サンプルなので、
    # テキスト上で交換可能な2セルは対称性の破れをノイズに委ねることになり、
    # 「どちらか片方の斜め前だけが崩れ、どちらが崩れるかは回ごとに移る」
    # という症状になる (実際にロップで左前→右前と移動した)。純横の
    # left/right が "facing left"/"facing right" で崩れないのと同じ語彙に
    # 斜めを紐づけて、非対称性を文面で固定する。
    "front_left": ("is a three-quarter front view turned 45 degrees to the "
                   "left, exactly midway between facing the camera and "
                   "facing left, keeping both eyes visible for the entire "
                   "walk, never flattening into a straight-on front view "
                   "and never turning into a pure side profile"),
    "front_right": ("is a three-quarter front view turned 45 degrees to the "
                    "right, exactly midway between facing the camera and "
                    "facing right, keeping both eyes visible for the entire "
                    "walk, never flattening into a straight-on front view "
                    "and never turning into a pure side profile"),
    "left": ("is an exact left profile facing left, with one eye visible "
             "on the side of the face"),
    "right": ("is an exact right profile facing right, with one eye "
              "visible on the side of the face"),
    "back": ("is seen from directly behind: only the back of the head and "
             "outfit is visible, each knee bends away from the viewer, and "
             "whenever a foot lifts, its heel and sole rise toward the "
             "camera"),
    # 後ろ斜めも同じ理由で左右を明記する (前斜めと同型の対称性の罠)
    "back_left": ("is a three-quarter rear view turned 45 degrees to the "
                  "left, exactly midway between facing away from the camera "
                  "and facing left: the face stays hidden behind the head, "
                  "each knee bends away from the viewer, and lifted feet "
                  "show heel and sole toward the camera"),
    "back_right": ("is a three-quarter rear view turned 45 degrees to the "
                   "right, exactly midway between facing away from the "
                   "camera and facing right: the face stays hidden behind "
                   "the head, each knee bends away from the viewer, and "
                   "lifted feet show heel and sole toward the camera"),
}
_FRONT_FAMILY = {"front", "front_left", "front_right", "left", "right"}
# 顔正面化 (pose_video._face_front_mode) が発動したセルの斜め前節:
# 骨格が顔0°+両耳を宣言するのに文面が「45°・正面化禁止」のままだと
# 骨格とテキストが綱引きする — 「体は45°・顔はカメラへ」で一致させる
# ★{side} で左右を必ず埋める。差し替え節も左右を書かないと、せっかく
# _DIR_CLAUSES を非対称にしても発動時に両斜めが同一文面へ戻ってしまう
_DIAG_FACE_FRONT_CLAUSE = (
    "keeps its body angled 45 degrees to the {side} while its face "
    "stays turned toward the camera with both eyes fully visible for the "
    "entire walk, exactly as in the very first frame, never turning away "
    "and never becoming a pure side profile")
# 体ヨー追従の発動セル用: 骨格の体が浅い斜め (床20°) なので「45度」を
# 言わせない — 骨格・参照・テキストの三者を「ほぼ正面のまま歩く」で一致
_DIAG_BODY_FOLLOW_CLAUSE = (
    "walks facing the camera, the body turned only slightly to the {side}, "
    "keeping the face turned straight to the camera with both eyes fully "
    "visible for the entire walk, exactly as in the very first frame, "
    "never turning away and never becoming a pure side profile")


def _diag_side(d: str) -> str:
    """斜め方向 -> "left"/"right"。差し替え節の {side} を埋めるため。"""
    return "right" if d.endswith("_right") else "left"


# COMPASS専用の左右中立版 (上の {side} 版と本文は同一で、側の指定だけを
# 元の "toward the camera" / "to the side" に戻したもの)。compassは角2セルを
# 「the two corner figures」と1文でまとめて指すので個別の左右を差し込めない。
# ★この文面は触らないこと: 角セルに語彙を足した版は下段の角まで後ろ向きへ
# 化けた実測がある (2026-07-18 真ロップ実走)。
_DIAG_FACE_FRONT_CLAUSE_COMPASS = (
    "keeps its body angled 45 degrees toward the camera while its face "
    "stays turned toward the camera with both eyes fully visible for the "
    "entire walk, exactly as in the very first frame, never turning away "
    "and never becoming a pure side profile")
_DIAG_BODY_FOLLOW_CLAUSE_COMPASS = (
    "walks facing the camera, the body turned only slightly to the side, "
    "keeping the face turned straight to the camera with both eyes fully "
    "visible for the entire walk, exactly as in the very first frame, "
    "never turning away and never becoming a pure side profile")


def _cell_phrase(i: int, cols: int, rows: int) -> str:
    """グリッド位置 -> 英語の位置句 ("the top-left figure" 等)。"""
    r, c = i // cols, i % cols
    row = ("top" if r == 0 else "bottom") if rows == 2 else f"row {r + 1}"
    if cols == 2:
        return f"the {row}-{'left' if c == 0 else 'right'} figure"
    ordinal = ("first", "second", "third", "fourth",
               "fifth", "sixth")[c] if c < 6 else f"{c + 1}th"
    return f"the {ordinal} figure in the {row} row"


def _direction_text(layout, face_front_diag: bool = False,
                    body_follow_diag: bool = False) -> str:
    """レイアウト -> 方向の明文宣言 (プロンプト追記)。

    stage2 (AniSora SDEdit) は骨格を見ないため、歩行中の向きの言い分は
    テキストと参照キャンバスだけ — 全レイアウトに必ず方向文を付ける
    (旧実装はcompass限定で、4x2キャンバスは方向文ゼロ=stage2が完全に
    事前分布任せだった。20260717_2232 真ロップ: 斜め前が歩行開始と同時に
    後頭部化)。compassは実績文面を維持 (行単位の宣言が既にA/B済み)。
    半球キャンバス (canvas_walk.LAYOUT_F4/B4) では「全員の顔が最後まで
    見える」を汚染なしで宣言できる — これが半球再編の本旨。
    face_front_diag: 顔正面化の発動時、斜め前の節を「体45°・顔はカメラへ」
    版に差し替える (骨格の顔0°宣言と文面を一致させる)。
    body_follow_diag: 体ヨー追従の発動時はさらに「体もほぼ正面」版へ
    (骨格の体20°宣言と一致。face_front_diagより優先)。"""
    if layout is LAYOUT_COMPASS:
        # ★上段は実績のあるC43文面のまま拡張しない: 角セルに rear 系の
        # 3/4語彙を足した版は、下段の角まで後ろ向きに化けた (2026-07-18
        # 真ロップ実走。方向タグの全併記が1向きへ収束する既知実測
        # 「全タグ=全部後ろ姿」と同型の汚染)
        corner = ("and each of the two corner figures "
                  f"{_DIAG_BODY_FOLLOW_CLAUSE_COMPASS}" if body_follow_diag else
                  "and each of the two corner figures "
                  f"{_DIAG_FACE_FRONT_CLAUSE_COMPASS}" if face_front_diag else
                  "and the two corner figures are "
                  "three-quarter front views that keep the body and "
                  "face angled 45 degrees toward the camera with both "
                  "eyes visible for the entire walk, never flattening "
                  "into a straight-on front view and never turning into "
                  "a pure side profile")
        return (" In the top row all three figures are seen from "
                "directly behind and walk away from the camera: only "
                "the backs of their heads and outfits are visible, "
                "each knee bends away from the viewer, and whenever a "
                "foot lifts, its heel and sole rise toward the camera. "
                "In the bottom row the three figures walk toward the "
                "camera: the center figure faces the camera square-on, "
                f"{corner}. The middle row shows exact left "
                "and right profiles. The center cell stays empty "
                "magenta.")
    cols, rows, dirs = layout
    parts = []
    present = [d for d in dirs if d]
    pure_front = bool(present) and all(d in _FRONT_FAMILY for d in present)
    if pure_front:
        # 前半球キャンバスだけの特権: 後ろ姿セルがいないので初めて
        # 「顔は常に見える」を全セルへ無差別に宣言できる
        parts.append(" Every figure on this sheet keeps its face visible "
                     "toward the camera for the entire video; no figure "
                     "ever shows the back of its head.")
    # 前向きセルと斜め後ろが同居するキャンバス (8x1等の旧レイアウト) では
    # rear系3/4語彙を落として素の「真後ろ」節に差し替える: rear 3/4語彙と
    # 前向きセルの同居は前の角まで後ろ向きに化ける実測毒 (2026-07-18
    # 真ロップ、compass上段実験)。compassが斜め後ろをセル別でなく行文で
    # 済ませているのも同じ理由。純後半球 (B4) は前向きセルが無いので
    # rear 3/4語彙のままで安全
    poison = (any(d in ("front", "front_left", "front_right")
                  for d in present)
              and any(d in ("back_left", "back_right") for d in present))
    for i, d in enumerate(dirs):
        if d is None or d not in _DIR_CLAUSES:
            continue
        clause = _DIR_CLAUSES[d]
        if poison and d in ("back_left", "back_right"):
            clause = _DIR_CLAUSES["back"]
        if body_follow_diag and d in ("front_left", "front_right"):
            clause = _DIAG_BODY_FOLLOW_CLAUSE.format(side=_diag_side(d))
        elif face_front_diag and d in ("front_left", "front_right"):
            clause = _DIAG_FACE_FRONT_CLAUSE.format(side=_diag_side(d))
        parts.append(f" On this sheet {_cell_phrase(i, cols, rows)} "
                     f"{clause}.")
    return "".join(parts)


def _run_layout(args, vl: dict, char: str, refs: dict, layout,
                mp4_dir: Path, work_dir: Path, tag: str,
                dirs_subset: list | None = None, seed: int = 42) -> list:
    """1レイアウト分: 参照キャンバス+骨格グリッド -> 1本 -> 方向別分割。

    hybrid時は1本の内部でVACE High→AniSora Lowを直結し、分割後の再生成は
    しない。旧方式だけ任意のAniSoraリファインを残す。
    """
    import pipeline as pl
    hybrid = bool(getattr(args, "videolab_pose_hybrid", False))
    work_dir.mkdir(parents=True, exist_ok=True)
    cell = str(pl.CONFIG.get("videolab_canvas_cell", "") or "")
    w, h = _canvas_size(layout, cell)
    dirs = [d for d in layout[2] if d]
    nf = _frame_count(getattr(args, "videolab_frames", 57))
    # フレーム配分 (直立/歩行/末尾静止) — プロンプト・カラーアンカー区間・
    # latent固定が全て同じ配分を見る (単一情報源はwalk_layout)
    lay_idle, lay_cyc, lay_per, lay_tail = pose_video.walk_layout(nf)
    lay_gend = lay_idle + int(round(lay_cyc * lay_per))   # 歩行終端フレーム
    print(f"  骨格グリッド[{tag}]: {len(dirs)}方向 -> 1本 {w}x{h} "
          f"(セル {w // layout[0]}x{h // layout[1]}, {nf}f"
          + (f"=直立{lay_idle}+歩行{lay_gend - lay_idle + 1}"
             f"+静止{lay_tail}" if lay_tail else "") + ")")

    # ミラー生成 (詳細は _mirror_dirs docstring): 対象方向は参照を反転して
    # 鏡像方向の骨格で生成し、分割後にセル動画を反転して戻す
    mdirs = [d for d in dirs if d in _mirror_dirs()]
    if mdirs:
        print(f"  ミラー生成: {','.join(mdirs)} は反対側の骨格+左右反転で "
              "(非対称デザインは鏡像になります。SM_POSE_MIRROR_DIRS=off"
              "で無効)")
        refs = dict(refs)
        for d in mdirs:
            _mim = Image.open(refs[d]).transpose(Image.FLIP_LEFT_RIGHT)
            _mp = work_dir / f"mirror_ref_{tag}_{d}.png"
            _mim.save(_mp)
            refs[d] = str(_mp)
    canvas = compose_reference(refs, w, h, layout)
    canvas.save(work_dir / f"canvas_ref_{tag}.png")
    # 振り倍率は素の1.0が既定 (1.5はhandoff減衰補償の遺物、2026-07-15撤去)
    aswing = float(os.environ.get("SM_POSE_ARM_SWING", "").strip()
                   or pl.CONFIG.get("videolab_pose_arm_swing", 1.0))
    lswing = float(os.environ.get("SM_POSE_LEG_SWING", "").strip()
                   or pl.CONFIG.get("videolab_pose_leg_swing", 1.0))
    bob = float(os.environ.get("SM_POSE_BOB", "").strip()
                or pl.CONFIG.get("videolab_pose_bob", 1.0))
    lcross = float(os.environ.get("SM_POSE_LEG_CROSS", "").strip()
                   or pl.CONFIG.get("videolab_pose_leg_cross", 1.0))
    # 顔68点 (DWPose白ドット)。既定auto=高頭身(スライダー1.8以上)のみon
    # (2026-07-16チビ顔歪みでoff → 2026-07-18真ロップ実害で高頭身は
    # 5点顔が判読不能=向こう向き化と判明し頭身連動へ。詳細_face68_on)
    _f68v = (os.environ.get("SM_POSE_FACE68", "").strip()
             or str(pl.CONFIG.get("videolab_pose_face68", "auto"))).lower()
    f68 = (True if _f68v in ("on", "1", "true", "yes")
           else False if _f68v in ("off", "0", "false", "no")
           else None)   # None=auto: pose_video._face68_on が頭身で判定
    # ヨー追従 (斜め前の骨格を立ち絵の実測ヨーへ寄り添わせる)。既定on
    yadapt = (os.environ.get("SM_POSE_YAW_ADAPT", "").strip()
              or str(pl.CONFIG.get("videolab_pose_yaw_adapt", "on"))
              ).lower() not in ("0", "off", "false", "no")
    # 顔正面化 (浅ヨー立ち絵の斜め前は顔を0°+両耳で宣言。2026-07-18
    # ユーザー発案、詳細 pose_video._face_front_mode)。既定auto
    _ffv = (os.environ.get("SM_POSE_FACE_FRONT", "").strip()
            or str(pl.CONFIG.get("videolab_pose_face_front", "auto"))
            ).lower()
    ffront = _ffv if _ffv in ("on", "off") else "auto"
    # 体ヨー追従 (浅ヨー立ち絵の斜め前は体骨格ごと実測へ・床20°。45°歩行
    # が分布外で真横に膠着した2026-07-18の対策。詳細 _diag_body_mode)。
    # 既定off — configで明示投入 (2026-07-16「体は45°」裁定の見直しのため)
    _dbv = (os.environ.get("SM_POSE_DIAG_BODY", "").strip()
            or str(pl.CONFIG.get("videolab_pose_diag_body", "off"))
            ).lower()
    dbody = "auto" if _dbv in ("auto", "on", "1", "true", "yes") else "off"
    frames = pose_video.build_canvas_pose_frames(refs, nf, w, h, layout,
                                                 arm_swing=aswing,
                                                 leg_swing=lswing, bob=bob,
                                                 leg_cross=lcross,
                                                 mirror_dirs=mdirs,
                                                 face68=f68,
                                                 yaw_adapt=yadapt,
                                                 face_front=ffront,
                                                 diag_body=dbody)
    frames[0].save(work_dir / f"pose_grid_{tag}_f000.png")

    # GPUのVRAMを見てオフロード方式を決める (720x1296級はA100-40の常駐だと
    # 活性化~21GBで溢れる。model_cpu_offloadは遅いが確実)
    extra = {"motion_score": float(getattr(args, "videolab_motion", 3.0)),
             "pose_frames_b64": pose_video.encode_frames_b64(frames),
             "conditioning_scale": float(
                 os.environ.get("SM_VACE_COND")
                 or pl.CONFIG.get("videolab_pose_cond", 1.0))}
    extra.update(_knobs_extra())
    # HFトークン (任意): 認証つきDLでColab IPの帯域制限を回避 (v0.8.5)
    _hft = str(pl.CONFIG.get("hf_token") or "").strip()
    if _hft:
        extra["hf_token"] = _hft
    q = (getattr(args, "videolab_quant", "") or "").strip()
    explicit_low_offload = q.endswith("-low")
    if explicit_low_offload:
        extra["offload"] = "seq"
        q = q[:-4]
    if q:
        extra["quant"] = q
    # 動的キャンバス設計 (plan_canvas) がoffload必須と判断した弱GPU向け。
    # v0.8.2から素のvace/anisoraも "block" を解釈する (handoffは従来から
    # 任意のoffload値をblockとして扱う。GGUF×seqはdiffusers既知バグの
    # ため値もblockへ変更 2026-07-14。旧サーバはblock未知=自動判定へ
    # 落ちるだけで壊れない)
    if getattr(args, "_canvas_offload", False) and "offload" not in extra:
        extra["offload"] = "block"
    if not hybrid:
        try:
            hlt = pl.api_get(
                f"{vl['url']}/health", vl.get("token"), timeout=30)
            vram = float((hlt.get("gpu") or {}).get("vram_gb") or 0)
            if 0 < vram < 60 and "offload" not in extra:
                extra["offload"] = "model"
                print(f"  VRAM {vram:.0f}GB (<60) のためオフロード運転にします "
                      "(遅いが確実。A100-80なら常駐で高速)")
        except Exception:
            pass

    # steps/cfg は pipeline と同じ既定ロジック (lightning=6/1.0 など)
    lora = str(extra.get("vace_lora", "")).lower()
    base = str(extra.get("vace_base", "")).lower()
    plain_fun = base == "fun" or q.lower().startswith("bf16")
    lit = lora == "lightning"
    steps = int(os.environ.get("SM_VACE_STEPS")
                or pl.CONFIG.get("videolab_pose_steps",
                                 6 if lit else (30 if plain_fun else 8)))
    guidance = float(os.environ.get("SM_VACE_GUIDANCE")
                     or pl.CONFIG.get("videolab_pose_guidance",
                                      1.0 if lit
                                      else (5.0 if plain_fun else 1.0)))
    if hybrid:
        # lightning時も8step固定: 6stepでは境界0.90後のAniSora Low区間が
        # 足りず全体が黄変する (2026-07-13 ロップ実測) — pipelineと同一既定
        steps = int(os.environ.get("SM_VACE_HYBRID_STEPS")
                    or pl.CONFIG.get("videolab_pose_hybrid_steps", 8))
        guidance = float(os.environ.get("SM_VACE_HYBRID_GUIDANCE")
                         or pl.CONFIG.get("videolab_pose_hybrid_guidance",
                                          1.0))
    prompt = CANVAS_PROMPT + NO_WIND
    # ロングスカート: 裾実測が立てば「不透明・スリット禁止」を宣言
    # (骨格側は裾より上の膝・足首を遮蔽=欠損化 — pose_video.skirt_hem_y。
    # 透け・スリット発明の対策 2026-07-17、プロンプトと両輪)
    _hem_front = None
    try:
        _fr = refs.get("front")
        if _fr is not None:
            _fim = _fr if hasattr(_fr, "convert") else Image.open(_fr)
            _hem_front = pose_video.skirt_hem_y(_fim, 240, 432)
    except Exception:                     # noqa: BLE001
        _hem_front = None
    if _hem_front is not None:
        prompt += (" The character wears a long skirt that stays "
                   "completely opaque: the legs never show through the "
                   "fabric, no slit ever opens in it, and only the feet "
                   "appear below the hem while walking.")
    if lay_tail:
        # 末尾静止: CANVAS_PROMPTの「歩き続ける」宣言と骨格の立ち止まりが
        # 矛盾しないよう明示 (特にstage2=AniSora SDEditは骨格を知らない
        # ため、プロンプトだけが静止区間の言い分になる)
        prompt += (" At the very end of the video every figure stops "
                   "walking and stands perfectly still in the same "
                   "relaxed upright standing pose as the first frames, "
                   "arms hanging naturally at the sides.")
    # 方向の明文宣言 (2026-07-17 C43実害: 背面の脚だけ前後反転、
    # 2026-07-17 真ロップ実害: 斜めセルの真横化 — 経緯は _direction_text)。
    # 2026-07-18から全レイアウト必須: stage2は骨格を見ないため、
    # 方向文の無いキャンバス (旧4x2) は歩行区間が事前分布任せだった。
    # 顔正面化が発動するキャンバスでは斜め前の節を「体45°・顔はカメラへ」
    # 版に差し替え、骨格の顔0°宣言とテキストの言い分を一致させる
    # (quiet探針: 実際の発動判定は build_canvas_pose_frames 内の
    # _adapted_yaw と同一関数・同一ゲート)
    _ff_diag = False
    _db_diag = False
    if ffront != "off" or dbody != "off":
        try:
            _pfig = pose_video.Figure(
                head_frac=pose_video.head_frac_for_leg_scale(
                    pose_video._leg_scale_env()))
            _pfr = refs.get("front")
            if _pfr is not None:
                _pim = _pfr if hasattr(_pfr, "convert") else Image.open(_pfr)
                _pfig = pose_video._fit_figure_to_char(_pfig, _pim)
            else:
                _pim = None
            for _d in ("front_left", "front_right"):
                if _d not in refs or _d not in [x for x in layout[2] if x]:
                    continue
                _rim = refs[_d] if hasattr(refs[_d], "convert") \
                    else Image.open(refs[_d])
                if (ffront != "off"
                        and pose_video._adapted_yaw(
                            _d, _rim, _pfig, front_ref=_pim,
                            face_front=ffront, quiet=True) == 0.0):
                    _ff_diag = True
                if (dbody != "off"
                        and pose_video._adapted_body_yaw(
                            _d, _rim, _pfig, front_ref=_pim,
                            mode=dbody, quiet=True)
                        != float(pose_video.DIR_YAW[_d])):
                    _db_diag = True
        except Exception:                     # noqa: BLE001
            _ff_diag = _db_diag = False   # 探針失敗時は従来文面
    prompt += _direction_text(layout, face_front_diag=_ff_diag,
                              body_follow_diag=_db_diag)
    if _db_diag:
        print("  体ヨー追従: 斜め前セルは体骨格も浅ヨー(床20°)で宣言し、"
              "プロンプトも「体もほぼ正面のまま歩く」版に切替 "
              "(videolab_pose_diag_body=offで従来動作)")
    elif _ff_diag:
        print("  顔正面化: 斜め前セルは顔0°+両耳で宣言し、プロンプトも"
              "「体45°・顔はカメラへ」版に切替 (SM_POSE_FACE_FRONT=off"
              "で従来動作)")
    cvid = work_dir / f"canvas_walk_{tag}.mp4"
    # latent_refine (2026-07-13ユーザー発案「VACEをフルで4ステップ当てて、
    # その潜在をVAEを通す前にanisoraで再加工」):
    # ①素のVACE-Fun+Lightning両エキスパートがフル軌道で骨格を完全制御
    # (handoffのように途中で制御を手放さない=動き減衰の根本対策)
    # ②最終latentをVAE未通過のままAniSoraがσ=0.45から再デノイズ
    # (旧2段構えのdecode→encode往復劣化とVRAMスパイクが消える)。
    # A100実測で全指標が過去最高 (動き20.91/上下動3.0%/背景B250) を受けて
    # 2026-07-14に既定化 (ユーザー指示「L4でいけるように調整して、これを
    # 既定に」)。旧handoffへは videolab_pose_mode="handoff" で戻せる。
    # サーバv0.8.2以降 (L4はUMT5遅延+block offloadで運転)。
    lr_mode = str(os.environ.get("SM_VACE_MODE", "").strip()
                  or pl.CONFIG.get("videolab_pose_mode", "")
                  or "latent_refine").lower()
    if hybrid and lr_mode in ("latent_refine", "latent", "vace_full"):
        # L4級 (VRAM<30GB) は両ステージともblock offloadで運転する:
        # 素のGGUFはseq offload不可 (diffusers既知バグ)、常駐/model
        # offloadでは再加工の活性化(~13GB)+重みが22.5GBに収まらない
        if "offload" not in extra:
            try:
                hlt = pl.api_get(f"{vl['url']}/health", vl.get("token"),
                                 timeout=30)
                _vram = float((hlt.get("gpu") or {}).get("vram_gb") or 0)
            except Exception:
                _vram = 0.0
            if 0 < _vram < 30:
                extra["offload"] = "block"
                print(f"  VRAM {_vram:.0f}GB (<30) のため両ステージを"
                      "block offload運転にします (重み転送ぶん低速・確実)")
        lr_steps = int(os.environ.get("SM_VACE_LR_STEPS")
                       or pl.CONFIG.get("videolab_pose_lr_steps", 4))
        lr_sigma = float(os.environ.get("SM_VACE_LR_REFINE")
                         or pl.CONFIG.get("videolab_pose_lr_refine", 0.45))
        # 既定24: A/Bで32と品質同等を確認済み (2026-07-14、尻尾3step)。
        # 旧refine経路・GUI表示とも24で統一 (2026-07-15レビュー指摘:
        # ここだけ32で、GUIの「既定24」表示と食い違っていた)
        lr_total = int(os.environ.get("SM_VACE_REFINE_TOTAL", "").strip()
                       or pl.CONFIG.get("videolab_pose_refine_total", 24))
        # 中間・最終フレームのlatent固定 (v0.9.8サーバ、旧サーバは黙って
        # 無視)。SM_VACE_LR_PIN: on(既定)=walk_layoutから自動 / off /
        # "27,48"明示。SM_VACE_LR_PIN_RELEASE: σ<この値で描き戻しを止めて
        # 質感を馴染ませる (既定0.0=終端まで固定。固定コマとリファイン済み
        # コマの質感差が見えたらここでA/B)
        pin_conf = str(os.environ.get("SM_VACE_LR_PIN", "").strip()
                       or pl.CONFIG.get("videolab_pose_lr_pin", "on")
                       or "on")   # config null値でも既定on (色アンカーと同規約)
        pins = _lr_pin_frames(nf, pin_conf)
        pin_rel = float(os.environ.get("SM_VACE_LR_PIN_RELEASE", "").strip()
                        or pl.CONFIG.get("videolab_pose_lr_pin_release", 0)
                        or 0)
        ex1 = dict(extra)
        ex1["vace_base"] = "fun"      # 勝ちスタック=素のVACE-Fun+Lightning
        ex1["vace_lora"] = "lightning"
        ex1["emit_latent"] = 1
        ex1.pop("hybrid_boundary", None)
        print(f"  生成[1/2]: VACEフル骨格制御 {nf}f {w}x{h} "
              f"steps={lr_steps} cfg=1.0 (latent直出し)")
        p1 = pl.api_post(
            f"{vl['url']}/api/generate", vl.get("token"),
            {"model": "vace", "mode": "i2v", "prompt": prompt,
             "images_b64": [_b64_png(canvas)], "key_positions": [],
             "width": w, "height": h, "num_frames": nf, "fps": 16,
             "steps": lr_steps, "guidance": 1.0, "seed": int(seed),
             "cancel_if_unpolled": True, "extra": ex1}, timeout=300)
        _poll_job(pl, vl, p1["job"],
                  work_dir / f"canvas_stage1_{tag}.mp4", f"vace1:{tag}",
                  span=(2, 48))
        ex2 = {"latent_from": p1["job"], "refine_strength": lr_sigma,
               "refine_cond_still": True,
               "motion_score": extra.get("motion_score", 3.0)}
        if pins:
            ex2["latent_pin_frames"] = pins
            if pin_rel > 0:
                ex2["latent_pin_release"] = pin_rel
            # 旧サーバ(<=0.9.7)はextra未知鍵を黙って無視するため、効いて
            # いないのに「錨止め」と読めるログは偽PASSの温床 — /healthの
            # versionで実効を確かめてから印字する (レビュー指摘 2026-07-15)
            try:
                _h = pl.api_get(f"{vl['url']}/health", vl.get("token"),
                                timeout=30) or {}
                _ver = tuple(int(x) for x in
                             str(_h.get("version") or "").split(".")[:3])
            except Exception:
                _ver = ()
            if _ver and _ver < (0, 9, 8):
                print("  ⚠ latent固定はサーバv0.9.8+が必要 — 現行サーバは"
                      "黙って無視します (Colabを張り直して更新してください)")
            else:
                _pin_kind = ("周期境界+末尾静止" if lay_tail
                             else "周期境界+最終フレーム")
                print(f"  latent固定: {_pin_kind} {pins} を"
                      "stage1へ錨止め (SM_VACE_LR_PIN=offで無効)")
        if "quant" in extra:
            ex2["quant"] = extra["quant"]
        if extra.get("offload"):
            ex2["offload"] = extra["offload"]
        if extra.get("hf_token"):
            ex2["hf_token"] = extra["hf_token"]   # AniSora側のDLにも必要
        print(f"  生成[2/2]: AniSora latent再加工 σ={lr_sigma} "
              f"(スケジュール{lr_total}step・VAE未通過)")
        p2 = pl.api_post(
            f"{vl['url']}/api/generate", vl.get("token"),
            {"model": "anisora", "mode": "i2v", "prompt": prompt,
             "images_b64": [_b64_png(canvas)], "key_positions": [],
             "width": w, "height": h, "num_frames": nf, "fps": 16,
             "steps": lr_total, "guidance": 1.0, "seed": int(seed),
             "cancel_if_unpolled": True, "extra": ex2}, timeout=300)
        _poll_job(pl, vl, p2["job"], cvid, f"anisora2:{tag}",
                  span=(48, 92))
    else:
        model = "vace_anisora_handoff" if hybrid else "vace"
        route = ("VACE High→AniSora Low latent直結" if hybrid
                 else "vace骨格グリッド")
        print(f"  生成: {route} {nf}f {w}x{h} steps={steps} cfg={guidance}")
        payload = pl.api_post(
            f"{vl['url']}/api/generate", vl.get("token"),
            {"model": model, "mode": "i2v", "prompt": prompt,
             "images_b64": [_b64_png(canvas)], "key_positions": [],
             "width": w, "height": h, "num_frames": nf, "fps": 16,
             "steps": steps, "guidance": guidance, "seed": int(seed),
             "cancel_if_unpolled": True, "extra": extra}, timeout=300)
        _poll_job(pl, vl, payload["job"], cvid, f"canvas:{tag}")

    # 2段目: 分割 -> セル拡大 -> セルごとAniSoraリファイン。
    # ★グリッドのセル(240x432級)は方向別生成(auto-size 448x880級)より
    # 物理解像度が低く、そのままシートに混ざると scale-mode global の
    # 「全方向同一ソース解像度」前提が壊れて該当方向だけ小さく写る
    # (2026-07-13実障害: 混成シートで8方向一括側が半分サイズ)。
    # セルを方向別と同じサイズへ拡大してからリファインすることで、
    # リファイン段が解像度イコライザを兼ねる (失敗時も拡大済みセルを
    # 使うためスケールは常に一致する)
    if hybrid:
        # AniSora Lowはすでに同じlatent軌道の後半を担当済み。分割後に
        # 完成動画を再入力する旧refineは絶対に重ねない。
        rs = 0.0
    else:
        try:
            rs = float(os.environ.get("SM_VACE_REFINE", "").strip()
                       or pl.CONFIG.get("videolab_pose_refine", 0) or 0)
        except (TypeError, ValueError):
            rs = 0.0
    rtotal = int(os.environ.get("SM_VACE_REFINE_TOTAL", "").strip()
                 or pl.CONFIG.get("videolab_pose_refine_total", 24))
    done_dirs = [d for d in dirs if (not dirs_subset or d in dirs_subset)]
    mp4_dir.mkdir(parents=True, exist_ok=True)
    # hybridはリファイン段が無く、handoff後の無制御区間で全体が黄色へ漂う
    # (2026-07-13実測: マゼンタ背景の青255→183-207、ノブでは詰め切れない)
    # → カラーアンカー (color_anchor.py) で参照立ち絵の色へ決定的に補正。
    # 本線=キャンバスを1回だけデコードし 切出し+アンカー+拡大 を単一パス
    # (2026-07-13ユーザー要望「ピクセル化する前に色を直したい」: セルmp4を
    # 焼いてから開き直す中間再エンコードを1世代削減)。失敗時は旧経路
    # (分割→セル毎アンカー→無補正スケール) へ後退。
    anchor_on = str(os.environ.get("SM_COLOR_ANCHOR", "").strip()
                    or pl.CONFIG.get("videolab_color_anchor", "on") or "on"
                    ).lower() not in ("off", "0", "false", "none")
    cols, rows_n = layout[0], layout[1]
    cw_cell, ch_cell = w // cols, h // rows_n
    fused = False
    if anchor_on:
        try:
            import color_anchor
            jobs = []
            for i, d in enumerate(layout[2]):
                if d is None or d not in done_dirs:
                    continue
                jobs.append({"crop": ((i % cols) * cw_cell,
                                      (i // cols) * ch_cell,
                                      cw_cell, ch_cell),
                             "ref": refs[d],
                             "dest": mp4_dir /
                             f"{char}_{IDX[d]:02d}_{d}_walkT.mp4",
                             "size": _auto_size(refs[d])})
            color_anchor.split_anchor_scale(
                pl._ffmpeg_exe(), cvid, jobs, idle_n=lay_idle,
                gait_end=(lay_gend if lay_tail else None))
            fused = True
            print(f"  分割+カラーアンカー+拡大を単一パスで実行 "
                  f"({len(jobs)}セル、中間再エンコードなし)")
        except Exception as e:      # noqa: BLE001
            print(f"  ⚠ 単一パス分割に失敗 — 旧経路(分割→セル毎)へ後退: "
                  f"{str(e)[:200]}")
    if not fused:
        cells_dir = work_dir / f"cells_{tag}"
        split_canvas_video(pl._ffmpeg_exe(), cvid, layout, cells_dir, char,
                           IDX, dirs_subset=dirs_subset)
        for d in done_dirs:
            cell_mp4 = cells_dir / f"{char}_{IDX[d]:02d}_{d}_walkT.mp4"
            final = mp4_dir / f"{char}_{IDX[d]:02d}_{d}_walkT.mp4"
            tw, th = _auto_size(refs[d])
            scaled = False
            if anchor_on:
                try:
                    import color_anchor
                    color_anchor.anchor_and_scale(
                        pl._ffmpeg_exe(), cell_mp4, refs[d], final, tw, th,
                        idle_n=lay_idle,
                        gait_end=(lay_gend if lay_tail else None))
                    scaled = True
                except Exception as e:      # noqa: BLE001
                    print(f"  ⚠ カラーアンカー失敗[{d}] — 無補正で続行: "
                          f"{str(e)[:200]}")
            if not scaled:
                # アス比の異なる拡大は余白側を先に中央クロップ (丸ごと
                # scaleするとキャラが潰れる 2026-07-13新型ロップ実障害)
                _ca, _ta = cw_cell / ch_cell, tw / th
                if _ta > _ca + 1e-3:
                    _crh = max(2, int(round(cw_cell / _ta)) // 2 * 2)
                    _vf = (f"crop={cw_cell}:{_crh},"
                           f"scale={tw}:{th}:flags=lanczos")
                elif _ta < _ca - 1e-3:
                    _crw = max(2, int(round(ch_cell * _ta)) // 2 * 2)
                    _vf = (f"crop={_crw}:{ch_cell},"
                           f"scale={tw}:{th}:flags=lanczos")
                else:
                    _vf = f"scale={tw}:{th}:flags=lanczos"
                subprocess.run(
                    [pl._ffmpeg_exe(), "-y", "-loglevel", "error",
                     "-i", str(cell_mp4),
                     "-vf", _vf, "-an",
                     str(final)],
                    check=True, capture_output=True,
                    creationflags=pl.CREATE_NO_WINDOW)
    for d in done_dirs:
        final = mp4_dir / f"{char}_{IDX[d]:02d}_{d}_walkT.mp4"
        tw, th = _auto_size(refs[d])
        if rs <= 0:
            continue
        print(f"  仕上げ[{tag}/{d}]: AniSoraリファイン {tw}x{th} "
              f"(σ={rs} / {rtotal}step)")
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            fd = Path(td)
            subprocess.run(
                [pl._ffmpeg_exe(), "-y", "-loglevel", "error", "-i",
                 str(final), "-qscale:v", "2", str(fd / "%05d.jpg")],
                check=True, capture_output=True,
                creationflags=pl.CREATE_NO_WINDOW)
            rb = [base64.b64encode(p.read_bytes()).decode("ascii")
                  for p in sorted(fd.glob("*.jpg"))]
        extra2 = {"motion_score": 3.0, "refine_frames_b64": rb,
                  "refine_strength": rs,
                  # 条件画像=原画立ち絵 (直立プレフィックス前提)。粘土を
                  # 「正常」と誤認させないための参照注入 (2026-07-13)
                  "refine_cond_still": True}
        if "quant" in extra:
            extra2["quant"] = extra["quant"]
        # 720x1296級キャンバス向けにhealth判定で足したmodel offloadを、
        # 464x848級の方向別AniSoraへ引き継ぐと、各stepで巨大なDiTが
        # CPU<->GPUを往復してA100でも数分/本になる。Q4は約27GBなので
        # A100-40GBへ常駐可能。明示的な-low、または専用ノブだけを尊重し、
        # それ以外はAniSoraAdapter自身のVRAM判定へ戻す。
        refine_offload = (os.environ.get("SM_VACE_REFINE_OFFLOAD", "").strip()
                          or str(pl.CONFIG.get(
                              "videolab_pose_refine_offload", "") or "").strip())
        if explicit_low_offload:
            refine_offload = "seq"
        if refine_offload in ("seq", "model"):
            extra2["offload"] = refine_offload
        elif extra.get("offload"):
            print("    仕上げ: 大キャンバス用offloadは引き継がず、"
                  "AniSoraのVRAM自動判定を使用")
        try:
            payload = pl.api_post(
                f"{vl['url']}/api/generate", vl.get("token"),
                {"model": "anisora", "mode": "i2v", "prompt": prompt,
                 "images_b64": [_b64_png(Image.open(refs[d]))],
                 "key_positions": [],
                 "width": tw, "height": th, "num_frames": len(rb),
                 "fps": 16, "steps": rtotal, "guidance": 1.0,
                 "seed": int(seed),
                 "cancel_if_unpolled": True, "extra": extra2}, timeout=300)
            _poll_job(pl, vl, payload["job"], final, f"refine:{d}")
        except Exception as e:      # noqa: BLE001
            print(f"  ⚠⚠ 仕上げ失敗[{d}] — 拡大済みセルのまま続行 "
                  f"(スケールは正しい): {str(e)[:200]}")
    # ミラー生成の出口: 対象方向のセル動画を左右反転して本来の向きへ戻す
    for d in mdirs:
        if d not in done_dirs:
            continue
        _mf = mp4_dir / f"{char}_{IDX[d]:02d}_{d}_walkT.mp4"
        if not _mf.is_file():
            continue
        _mt = _mf.with_name(_mf.stem + ".fliptmp.mp4")
        try:
            subprocess.run(
                [pl._ffmpeg_exe(), "-y", "-loglevel", "error",
                 "-i", str(_mf), "-vf", "hflip", "-c:v", "libx264",
                 "-pix_fmt", "yuv420p", "-crf", "12", "-an", str(_mt)],
                check=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            os.replace(_mt, _mf)
            print(f"  ミラー出口反転: {d}")
        except Exception as e:      # noqa: BLE001
            print(f"  ⚠ ミラー出口反転に失敗[{d}] — 鏡像のまま残っています"
                  f" (要作り直し): {str(e)[:200]}")
    action = "分割（再生成なし）" if hybrid else "分割+仕上げ"
    print(f"  {action}[{tag}]: {len(done_dirs)}本 -> {mp4_dir}")
    return done_dirs


def _apply_canvas_plan(args, vl: dict, mode: str) -> None:
    """空きVRAMを/healthで実測し、動的キャンバス設計を args/CONFIG へ反映。

    ご家庭のゲーミングPC (16GB/12GB級) でも8方向まとめ生成をオミット
    しないための自動ダウンサイズ (2026-07-13ユーザー要望)。手動指定
    (config videolab_canvas_cell) がある場合と videolab_canvas_auto=off /
    SM_CANVAS_AUTO=off では何もしない。1ラン1回だけ実行 (FAIL再生成でも
    構成が揺れないように)。"""
    import pipeline as pl
    if getattr(args, "_canvas_planned", False):
        return
    args._canvas_planned = True
    args._canvas_offload = False
    if str(pl.CONFIG.get("videolab_canvas_cell", "") or "").strip():
        return                          # 手動指定を尊重
    auto = str(os.environ.get("SM_CANVAS_AUTO", "").strip()
               or pl.CONFIG.get("videolab_canvas_auto", "on") or "on"
               ).lower() not in ("off", "0", "false", "none")
    if not auto:
        return
    try:
        hlt = pl.api_get(f"{vl['url']}/health", vl.get("token"), timeout=30)
        gpu = hlt.get("gpu") or {}
        free = float(gpu.get("free_gb") or gpu.get("vram_gb") or 0)
    except Exception:
        return                          # 測れなければ静的既定のまま
    if free <= 0:
        return
    nf_req = _frame_count(getattr(args, "videolab_frames", 57))
    # モード内で最大のキャンバス (=最悪ケース) を基準に1回だけ設計する
    cols, rows = max(((la[0], la[1]) for la in MODES[mode]),
                     key=lambda cr: cr[0] * cr[1])
    plan = plan_canvas(free, nf_req, cols, rows)
    cw, ch = plan["cell"]
    if (cw, ch) == (240, 432) and plan["nf"] == nf_req and not plan["offload"]:
        return                          # フル構成が入るGPU: 何も変えない
    pl.CONFIG["videolab_canvas_cell"] = f"{cw}x{ch}"   # 実行時のみ・保存なし
    if plan["nf"] != nf_req:
        args.videolab_frames = plan["nf"]   # 骨格生成と--gait-startヒントが
        #                                     walk_layout経由で自動追従する
    args._canvas_offload = bool(plan["offload"])
    print(f"  動的キャンバス設計: 空きVRAM {free:.1f}GB -> "
          f"セル{cw}x{ch} {plan['nf']}f"
          f"{' +block offload(低速)' if plan['offload'] else ''}"
          f" [{plan['note']}]")


def run_canvas_vace(args, vl: dict, char: str, refs: dict, mp4_dir: Path,
                    mode: str, work_dir: Path,
                    dirs_subset: list | None = None,
                    seed: int = 42) -> list:
    """GUI「方向まとめ」のvideolab版 (2026-07-13コンパス実験PASSを受けて
    正式配線)。canvas_walk.MODES のレイアウト群を骨格グリッドで生成する。
    書き出した方向名の一覧を返す。"""
    if mode not in MODES:
        raise SystemExit(f"未知の方向まとめモード: {mode}")
    _apply_canvas_plan(args, vl, mode)
    if mode == "compass":
        # 斜め前の立ち絵が正面寄り (Codexの斜め45°の壁) のキャラは、
        # compassの1生成では斜め前セルが後ろ姿セル群と見た目を共有し、
        # 歩行開始と同時に後頭部化しやすい (20260717_2232 真ロップ:
        # 実測ヨー-13°/+11°で8試行全滅)。半球分離できる 4x2 を推奨。
        try:
            _fig = pose_video.Figure(
                head_frac=pose_video.head_frac_for_leg_scale(
                    pose_video._leg_scale_env()))
            _fr = refs.get("front")
            if _fr is not None:
                _fig = pose_video._fit_figure_to_char(
                    _fig, Image.open(_fr))
            _shallow = []
            for _d in ("front_left", "front_right"):
                if _d not in refs:
                    continue
                _est, _ = pose_video._estimate_yaw(Image.open(refs[_d]),
                                                   _fig)
                if _est is not None and abs(_est) < 20.0:
                    _shallow.append(f"{_d}={_est:+.0f}°")
            if _shallow:
                print("  ⚠ 斜め前の立ち絵が正面寄りです ("
                      + ", ".join(_shallow) + "): compassは斜め前が後ろ姿"
                      "セルと同じ1生成に同居するため、歩行中の後頭部化が"
                      "出やすい構成です (2026-07-18 真ロップ実測)。方向"
                      "まとめ=4x2 (前後半球の2生成) への切替を推奨します",
                      flush=True)
        except Exception:                     # noqa: BLE001
            pass                              # 警告はベストエフォート
    requested = ({str(d) for d in dirs_subset if d}
                 if dirs_subset else None)
    planned = planned_direction_jobs(mode, dirs_subset)
    print("  方向まとめ計画: "
          + " / ".join("+".join(group) for group in planned)
          + f" ({len(planned)}ジョブ)")
    done: list = []
    for li, layout in enumerate(MODES[mode]):
        dirs = [d for d in layout[2] if d]
        if requested and not (set(dirs) & requested):
            continue                     # このレイアウトに対象方向がない
        tag = f"{mode}{li + 1 if len(MODES[mode]) > 1 else ''}"
        done += _run_layout(args, vl, char, refs, layout, mp4_dir,
                            work_dir, tag, dirs_subset=dirs_subset,
                            seed=seed)
    missing = sorted(requested.difference(done)) if requested else []
    if missing:
        raise RuntimeError(
            f"方向まとめモード {mode} では選択方向を生成できません: "
            + ", ".join(missing))
    return done


def run_compass_test(args, vl: dict) -> int:
    """CLI --compass-test: 完了済みラウンドで3x3コンパスを1本生成し、
    既存ゲートまで通す実験ハーネス (2026-07-13 実GPUでPASS済み)。"""
    import pipeline as pl
    if not args.round_dir:
        raise SystemExit("--compass-test には --round-dir (完了済みラウンド) "
                         "が必要です")
    round_dir = Path(args.round_dir)
    out = round_dir / "compass_test"
    char, refs = _load_dir_refs(round_dir)
    # --compass-size はフルキャンバス指定 -> セル寸法へ換算して共通経路へ。
    # ★既定値(720x1296)のままなら config を埋めない: setdefaultで埋めると
    # _apply_canvas_plan が「手動指定あり」と誤認してVRAM動的設計を
    # スキップし、L4等でフル解像度が走ってしまう (2026-07-13 L4実走で発覚)
    _cs = str(getattr(args, "compass_size", "") or "720x1296")
    if _cs != "720x1296":
        try:
            w, h = (int(x) for x in _cs.lower().split("x"))
        except ValueError:
            raise SystemExit("--compass-size は WxH 形式 (例 720x1296)")
        import pipeline as _pl
        _pl.CONFIG.setdefault("videolab_canvas_cell",
                              f"{max(96, w // 3)}x{max(96, h // 3)}")
    print(f"コンパス生成: {char} 8方向 -> 1本")
    run_canvas_vace(args, vl, char, refs, out / "mp4", "compass", out)

    qc_dir = out / "04_video_qc"
    rc = pl.run_logged(
        "inspect_walk_mp4",
        ["--mp4-dir", str(out / "mp4"),
         "--reference-dir",
         str(round_dir / "01_generation" / "split_centered"),
         "--out-dir", str(qc_dir),
         "--ffmpeg", pl._ffmpeg_exe()], [])
    print(f"\nコンパス実験 完了: ゲート{'PASS' if rc == 0 else 'FAIL'} "
          f"(成果物: {out})")
    print("  目視ポイント: 各セルの回転なし・セル間のにじみ/混線・"
          "セル解像度での画質劣化")
    return 0


def run_illustrious_test(args, vl: dict) -> int:
    """CLI --illustrious-test: Illustrious-XL+OpenPoseで8方向シートを
    1枚生成する実験ハーネス (2026-07-14ユーザー指示「イラストリアス搭載」
    のパイロット。compass-testと同型)。

    骨格グリッド=マネキン3Dの直立8方向を4x2に並べてOpenPose化 (dir_refs=
    マネキンセルなので骨格の位置・体型が頭身ノブに追従)。生成シートは
    既存の pipeline.split_to_crops (4x2契約) で8分割して保存する。"""
    import pipeline as pl
    import mannequin3d
    if not args.round_dir:
        raise SystemExit("--illustrious-test には --round-dir が必要です")
    round_dir = Path(args.round_dir)
    out = round_dir / "illustrious_test"
    out.mkdir(parents=True, exist_ok=True)
    rc = {}
    rcp = round_dir / "run_config.json"
    if rcp.is_file():
        import json
        rc = json.loads(rcp.read_text(encoding="utf-8"))
    leg = float(pl.CONFIG.get("leg_scale", 1.0) or 1.0)
    dirs8 = [d for _i, d, _v, _l in pl.DIRECTIONS]
    # 1枚のコンパス配置 (3x3・中央空白) で8方向を一括生成する。方向別の
    # 個別生成は生成間で画風が割れて同一性が保てない (2026-07-14ユーザー
    # 知見「1枚で八方向コンパス出しがもっとも方向を守りやすい」— Codex
    # コースの向き取り違え対策 compose_idle_guide と同じ配置)。向きタグの
    # 全併記も1向きへ収束する実測 (タグ無し=全部正面/全タグ=全部後ろ姿)
    # のためタグでは押さず、下地+コンパス骨格+turnaround系タグで指定する
    import canvas_walk
    layout = canvas_walk.LAYOUT_COMPASS
    _sz = os.environ.get("SM_ILL_SIZE", "1536x1536")
    try:
        W, H = (int(x) for x in _sz.lower().split("x"))
    except ValueError:
        W, H = 1536, 1536                # セル512x512 (v2.0は1536ネイティブ)
    # 斜め4方向のヨー誇張ノブ (SDXLは45°を正面/背面へ丸める実測 2026-
    # 07-14への対抗策。既定45=素の角度)。マネキン絵と骨格が同じヨーを
    # 共有する必要があるため render_cells の前に差し替える (DIR_YAWは
    # mannequin3d/pose_videoが共有する辞書)
    dy = float(os.environ.get("SM_ILL_DIAG_YAW", "45"))
    _yaw_orig = dict(mannequin3d.DIR_YAW)
    mannequin3d.DIR_YAW.update(
        {"front_left": -dy, "front_right": dy,
         "back_left": -(180 - dy), "back_right": 180 - dy})

    def _to_rgb(im):
        bg = Image.new("RGB", im.size, MAGENTA)
        if im.mode == "RGBA":
            bg.paste(im, (0, 0), im)
            return bg
        return im.convert("RGB")

    try:
        cells = mannequin3d.render_cells(leg)
        dir_refs = {d: _to_rgb(cells[(d, "idle")]) for d in dirs8}
        grid = pose_video.build_canvas_pose_frames(
            dir_refs, 9, W, H, layout, adapt=True, arms=True)[0]
        # i2i下地 = マネキンコンパスそのもの (マゼンタ地・骨格と画素整列)
        init = compose_reference(dir_refs, W, H, layout)
    finally:
        mannequin3d.DIR_YAW.clear()
        mannequin3d.DIR_YAW.update(_yaw_orig)
    concept = (os.environ.get("SM_ILL_CONCEPT", "").strip()
               or (rc.get("concept") or "").strip())
    if not concept:
        print("⚠ キャラ説明(concept)が空です — 画像参照だけのラウンドは"
              "プロンプトにキャラ描写が乗らず同一性を担保できません "
              "(SM_ILL_CONCEPT=タグ列 で補えます)")
    tags = str(pl.CONFIG.get(
        "illustrious_tags",
        "masterpiece, best quality, chibi, full body, standing, "
        "front lighting, flat color, simple background"))
    # シート系タグはノブ化 (2026-07-14: 「reference sheet/turnaround」は
    # 正面・横・背面が正典の学習分布を引き込み、斜め4方向を正面/背面へ
    # 丸める容疑 — SM_ILL_SHEET_TAGSで消去実験できる)
    sheet_tags = os.environ.get(
        "SM_ILL_SHEET_TAGS",
        "character turnaround, reference sheet, "
        "multiple views of the same character, same outfit")
    prompt = ", ".join(x for x in (tags, sheet_tags, concept) if x)
    steps = int(os.environ.get("SM_ILL_STEPS", "28"))
    cfg = float(os.environ.get("SM_ILL_CFG", "6.0"))
    cn = float(os.environ.get("SM_ILL_CN", "1.2"))
    den = float(os.environ.get("SM_ILL_DENOISE", "0.7"))
    # 生成モード (2026-07-14スイープ実測): i2iは強度を上げるほど
    # 「設定資料は上段に顔」の学習priorが骨格を上書きし向きが崩れる
    # (0.9=上下逆転、1.0=崩壊)。向きの信頼性は純t2iが最良のため
    # SM_ILL_MODEで切替可 (i2iの勝ち分=背景維持・マネキン頭身の直接誘導)
    mode = os.environ.get("SM_ILL_MODE", "i2i").strip().lower()
    print(f"Illustrious {mode}: {W}x{H} マネキンコンパス"
          f"{'下地' if mode == 'i2i' else '骨格のみ'}で8方向一括 "
          f"(steps={steps} cfg={cfg} CN={cn} denoise={den} "
          f"diag_yaw={dy:g})")
    grid.save(out / "pose_grid.png")
    init.save(out / "init_canvas.png")
    for f in out.glob("gen_*.png"):
        f.unlink()                       # 旧・方向別生成の残骸を掃除
    ex = {"pose_image_b64": _b64_png(grid), "controlnet_scale": cn,
          "denoise": den}
    _hft = str(pl.CONFIG.get("hf_token") or "").strip()
    if _hft:
        ex["hf_token"] = _hft
    # i2i: マゼンタ背景・セル配置・斜めの向きをマネキン下地から引き継ぐ
    # (2026-07-14ユーザー発案「マゼンダ背景をi2iで維持できませんか?」)。
    # 骨格CNはt2iと同じく併用
    payload = pl.api_post(
        f"{vl['url']}/api/generate", vl.get("token"),
        {"model": "illustrious", "mode": mode, "prompt": prompt,
         "images_b64": [_b64_png(init)] if mode == "i2i" else [],
         "key_positions": [],
         "width": W, "height": H, "num_frames": 1, "fps": 1,
         "steps": steps, "guidance": cfg, "seed": 42,
         "cancel_if_unpolled": True, "extra": ex}, timeout=300)
    sheet = out / "sheet.png"
    _poll_job(pl, vl, payload["job"], sheet, "ill:t2i")
    # 背景をマゼンタへ塗り替え (SDXLは「magenta background」を無視して
    # ベージュ等で塗る実測 — 一様背景なら後処理キーイングが確実)
    import numpy as np
    a = np.asarray(Image.open(sheet).convert("RGB")).astype(int)
    border = np.concatenate([a[0, :], a[-1, :], a[:, 0], a[:, -1]])
    bgc = np.median(border, axis=0)
    mask = np.abs(a - bgc).sum(axis=2) < 60
    a[mask] = (255, 0, 255)
    keyed_img = Image.fromarray(a.astype("uint8"))
    # コンパス3x3 -> 既存契約の4x2へ組み替え (分割・QCゲートは4x2前提。
    # 骨格グリッドと画素整列しているのでセルは幾何で切れる)
    cols, rows_, ldirs = layout
    cw, ch = W // cols, H // rows_
    pos = {d: ((i % cols) * cw, (i // cols) * ch)
           for i, d in enumerate(ldirs) if d}
    sheet42 = Image.new("RGB", (cw * 4, ch * 2), MAGENTA)
    for i, d in enumerate(dirs8):
        x, y = pos[d]
        sheet42.paste(keyed_img.crop((x, y, x + cw, y + ch)),
                      ((i % 4) * cw, (i // 4) * ch))
    keyed = out / "sheet_keyed.png"
    sheet42.save(keyed)
    crops = pl.split_to_crops(keyed, getattr(args, "magenta_thr", 70))
    for d, im in crops.items():
        im.save(out / f"crop_{d}.png")
    print(f"\nIllustrious実験 完了: 分割{len(crops)}方向 (成果物: {out})")
    print("  目視ポイント: 8体が同一キャラか・向きと左右・マゼンタ背景の"
          "純度・チビ体型の破綻")
    return 0


if __name__ == "__main__":
    raise SystemExit("pipeline経由で --compass-test を使ってください")
