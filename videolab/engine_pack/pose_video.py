#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pose_video.py -- mannequin3d の歩行サイクルから OpenPose 骨格制御動画を作る。

VACE (Wan VACE / VACE-Fun) のポーズ駆動用。AniSora i2v の斜め後ろ
(back_left/back_right) は、プロンプトロック・motion減速・終端アンカー・
中間キーフレーム5点拘束の全てを試しても「拘束点の合間で一回転する」抜け道を
塞げなかった (2026-07-12 ロップで実証)。全フレームのポーズを骨格で指定すれば
回転は定義上起こり得ない、が本モジュールの目的。

設計:
  - WALK_POSES 1..4 (接地A→通過→接地B→通過) の関節角を位相で線形補間し、
    連続歩行サイクルにする (5=中間コマはサイクル外なので使わない)。
    81フレーム/16fpsで3周期 = AniSora実証済みのケイデンスと同じ。
    先頭と終端は同位相 (ループアンカーと同じ始点=終点拘束)。
  - 関節3D位置は mannequin3d.Figure.joints() をそのまま使い、同じカメラ
    (DIR_YAW の8方向ヨー + 俯角 ELEV_DEG=15) で2D投影する。
  - 描画は OpenPose BODY_18 規格 (黒地・標準の18色・肢は楕円0.6輝度+
    関節は円)。VACE のポーズ制御は OpenPose 形式で学習されているため、
    colored blob マネキンではなくこの形式で描くこと。
  - 顔キーポイント (鼻・目・耳) は視線カリング付き: 前後の視点弁別は
    OpenPose では「どの顔点が見えるか+左右色の画面配置」で符号化される。
    back系で鼻・目が消えることが「後ろ向き」の構造的な宣言になる。
  - DWPoseの顔68点白ドットは実装済みだが既定off (SM_POSE_FACE68=onで
    有効。2026-07-16実走でチビ顔が歪んだため反転 — 経緯は_face68_onの
    コメント)。足6点は全系譜で描画されない語彙外記号なので描かない —
    詳細は _face68_template のコメント。
  - ヨー追従 (SM_POSE_YAW_ADAPT=offで無効): 斜め前2方向は「頭だけ」
    立ち絵の実測ヨー (目の重心横オフセット) で描き、体は公称45°を維持
    する。参照立ち絵が正面寄りなのに骨格が45°を主張する綱引きが
    「斜め前の引っ張られ(奥向き)」の真因で、頭の追従がそれを消す。
    体まで追従させると足運びがゲームの移動角度と合わなくなる
    (2026-07-16実走裁定) — 詳細は _adapted_yaw / _keypoints のコメント。

左右の対応 (重要):
  mannequin3d はキャラ前方+z・yaw=0でカメラ正対。右手系 y-up では
  「解剖学的な右 = 前方x上 = -x側」なので、mannequin の tag "R" (+x側) は
  解剖学的には左。OpenPose の R* キーポイントには mannequin の *_L (-x側) を、
  L* には *_R (+x側) を割り当てる。これを取り違えると全方向の骨格が
  「左右反転した人」になり、前向きと後ろ向きの弁別が逆転する。

Usage (目視検査用):
    python engine/pose_video.py --out scratch_pose --size 464x848 \
        --dirs back_left,back_right [--ref 立ち絵.png] [--gif]
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from mannequin3d import (DIR_YAW, ELEV_DEG, HEAD_WORLD, IDLE_POSE,
                         WALK_POSES, Figure, head_frac_for_leg_scale,
                         _rotx, _roty)

# ---------------------------------------------------------- OpenPose 規格
# BODY_18 キーポイント (0-based):
#  0 Nose  1 Neck  2 RSho  3 RElb  4 RWri  5 LSho  6 LElb  7 LWri
#  8 RHip  9 RKne 10 RAnk 11 LHip 12 LKne 13 LAnk
# 14 REye 15 LEye 16 REar 17 LEar
# 肢の接続と18色は controlnet_aux / openpose 公式 draw_bodypose と同一
# (VACE が学習した見た目に合わせるため、配列は原典のまま使うこと)。
OP_LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9),
            (9, 10), (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16),
            (0, 15), (15, 17)]
OP_COLORS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
             (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
             (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
             (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
             (255, 0, 170), (255, 0, 85)]

# mannequin joints() -> OpenPose 体幹12点 (左右スワップに注意: 冒頭コメント)
_BODY_MAP = {2: "sh_L", 3: "elbow_L", 4: "wrist_L",
             5: "sh_R", 6: "elbow_R", 7: "wrist_R",
             8: "hip_L", 9: "knee_L", 10: "ankle_L",
             11: "hip_R", 12: "knee_R", 13: "ankle_R"}

# 顔キーポイント: 頭中心からのオフセット (頭半径r単位, 人物座標系 +z=前方,
# 解剖学的右=-x) と外向き法線・可視閾値。法線をヨー+俯角で回し、カメラ向き
# 成分 z > thr のときだけ描く。8方向での見え方は実写OpenPoseの正準
# パターンに一致させる (VACEの教師データと同じ語彙で喋る):
#   front: 鼻+両目+両耳 / ★front_3/4: 鼻+両目+手前耳のみ /
#   横: 鼻+片目+片耳 / back_3/4: 手前耳のみ / back: 両耳のみ
# ★斜め前の奥耳は2026-07-16に再び非可視へ (実測で確定した実害:
#   可視化した奥耳 (±0.92,-0.45,-0.05) は45°回転後、鼻とほぼ同座標に
#   投影される — front_leftで鼻-16.3px/奥耳-15.4px・高さもほぼ同じ
#   (-0.46r vs -0.45r)。「顔の前縁に耳」は実写では後ろ3/4の署名なので、
#   頭を奥向きに解釈する強い引き金になる (ロップ実走で確定: 参照絵に
#   人間の顔ランドマークが無いキャラほど耳点に引かれ、斜め前が奥向きに
#   描かれた)。2026-07-15の可視化はユーザー診断「奥耳が無いと頭が
#   回りすぎる」に基づいたが、その時点では目・耳の高さ較正も誤っていた。
#   高さ較正済みの今は目2点+鼻の正準3/4パターン (奥目が顔縁ぎわ・
#   鼻寄り、手前目が中央寄り) で角度が立つ — 奥耳で補う必要は無い。
#   法線は純横向き (±1,0,0)・閾値-0.35: front0°=nz≈0で両耳可視 /
#   45°=奥nz-0.68で消え・手前+0.68で残る / 90°=同様 / back135°=手前のみ /
#   back180°=nz≈0で両耳可視、が単一閾値で全部成立 (俯角15°はnzを
#   cos15≈0.97倍するだけで符号を変えない)。
# 顔点のオフセットはチビ絵の実測に較正 (2026-07-15、真ロップの基準絵
# オーバーレイで確定した実害: 旧値は頭「中心」基準でチビの顔には高すぎ、
# 特に耳が生え際の高さ×髪幅の外周に浮いていた。目・耳が実絵より高い
# 骨格を渡すと、正面・斜め前の顔がやや奥(上方)へ引っ張られ、シート化で
# 直立(原画)と頭部の角度が繋がらない)。チビ顔は頭の下半分に寄る:
# 目=瞳中心の高さ・広さ、耳=目線の高さの顔の縁、鼻=口元。
# 鼻のz=前方突出は0.78 (2026-07-16「歩きが猫背」対策: 旧0.92=球面仮定は
# チビの平らな顔には深すぎ、横・斜めで首→鼻リンクが前下がりの長い
# 斜めバーになり「頭を前に突き出した歩き」を宣言していた。直立キャラの
# 足元軸からの鼻先実測=ろーら0.70/真ロップ0.77/AI妹0.84の中央で較正)
_FACE_PTS = {
    0:  ((0.00, -0.46, 0.78), (0.0, -0.10, 0.995), -0.05),   # Nose
    14: ((-0.47, -0.15, 0.75), (-0.35, 0.05, 0.93), 0.0),    # REye (-x)
    15: ((+0.47, -0.15, 0.75), (+0.35, 0.05, 0.93), 0.0),    # LEye (+x)
    16: ((-0.92, -0.45, -0.05), (-1.0, 0.0, 0.0), -0.35),    # REar (-x)
    17: ((+0.92, -0.45, -0.05), (+1.0, 0.0, 0.0), -0.35),    # LEar (+x)
}

# 左右鏡像の方向名 (ミラー生成用: back_rightをback_left骨格で作る等)
MIRROR_NAME = {"left": "right", "right": "left",
               "front_left": "front_right", "front_right": "front_left",
               "back_left": "back_right", "back_right": "back_left",
               "front": "front", "back": "back"}


def _face68_template() -> list:
    """iBUG-68配置の顔ランドマーク (face_r単位オフセット, 外向き法線, 閾値)。

    VACE既定のposeタスク (PoseBodyFaceVideoAnnotator) は BODY_18 に加えて
    DWPoseの顔68点を「白(255,255,255)・半径3・線なし」でBODY_18の上に描く
    (ali-vilab/VACE vace/annotators/dwpose/util.py draw_facepose。
    VideoX-Fun=Wan-Fun本家も同一描画)。一方、足6点は全系譜
    (CMU BODY_25以外: ControlNet/DWPose/MusePose/VACE/VideoX-Fun) で
    「検出後にスライスもされず捨てられる」死にコードで、足の描画関数自体が
    存在しない=語彙外 (2026-07-16 実ソース裏取り)。よって足は描かず、
    顔だけを正準どおり詳細化する。
    配置は _FACE_PTS のチビ較正アンカー (目±0.47/-0.15・鼻0/-0.46・
    耳±0.92/-0.45) と整合する頭球面上の68点を機械生成する。順序は
    iBUG準拠 (顎0-16/眉17-26/鼻27-35/目36-47/口48-67) だが描画は
    無ラベルの白点なので視覚上は分布だけが意味を持つ。"""
    pts: list = []

    def sph(x: float, y: float) -> tuple:
        # x,y から前面 (z>=0) の単位球面点。輪郭に近いほど z が小さい
        z2 = 1.0 - x * x - y * y
        return (x, y, math.sqrt(z2) if z2 > 0.0 else 0.0)

    def add(p: tuple, thr: float) -> None:
        n = math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2) or 1.0
        pts.append((p, (p[0] / n, p[1] / n, p[2] / n), thr))

    # 顎輪郭 0-16: 右耳際→顎先→左耳際の下面弧 (端は耳±0.92/-0.45に接続、
    # 顎先は頭球の底前面)。輪郭点は遮蔽輪郭として横からも見えるので
    # 閾値は耳と同じ-0.35
    for i in range(17):
        t = i / 16.0
        phi = math.radians(-100.0 + 200.0 * t)
        y = -0.45 - 0.47 * math.sin(math.pi * t)
        rho = math.sqrt(max(0.0, 1.0 - y * y))
        add((rho * math.sin(phi), y, rho * math.cos(phi)), -0.35)
    # 眉 17-21(右=-x)/22-26(左=+x): 目の上のゆるいアーチ
    for side in (-1, +1):
        for k in range(5):
            x = side * (0.70 - 0.12 * k)
            y = 0.08 + 0.05 * math.sin(math.pi * k / 4.0)
            add(sph(x, y), 0.0)
    # 鼻すじ 27-30: 眉間→鼻先 (鼻先はBODY鼻点の高さ-0.46へ降りる)
    for k in range(4):
        add(sph(0.0, -0.18 - 0.08 * k), 0.0)
    # 鼻底 31-35: BODY鼻点 (0,-0.46) と同じ高さの横列
    for k in range(5):
        add(sph(-0.14 + 0.07 * k, -0.46), 0.0)
    # 目 36-41(右)/42-47(左): 較正済み目中心±0.47/-0.15を囲む6点リング
    for side in (-1, +1):
        for k in range(6):
            a = math.pi * k / 3.0
            add(sph(side * 0.47 + 0.13 * math.cos(a),
                    -0.15 + 0.055 * math.sin(a)), 0.0)
    # 口 48-59(外唇12点)/60-67(内唇8点): 鼻底の下・顎先の上
    for k in range(12):
        a = math.pi * k / 6.0
        add(sph(0.26 * math.cos(a), -0.62 + 0.075 * math.sin(a)), 0.0)
    for k in range(8):
        a = math.pi * k / 4.0
        add(sph(0.16 * math.cos(a), -0.62 + 0.035 * math.sin(a)), 0.0)
    assert len(pts) == 68
    return pts


_FACE68 = _face68_template()


def _face68_on() -> bool:
    """顔68点描画のノブ (SM_POSE_FACE68、既定auto=高頭身のみon)。

    ★2026-07-16実走で既定offへ反転: ロップで顔が歪んだ。68点は実写顔の
    配置で学習された強い処方箋で、チビ顔との不一致ぶん顔が引っ張られる
    (DWPoseはアニメ顔で検出不安定=教師データのアニメ系クリップには
    顔68点がほぼ無い可能性が高く、「実写では語彙内・アニメでは実質
    語彙外」)。向きの宣言は5点(鼻目耳)の可視パターンが担っており
    (arXiv 1812.00739)、VACE本家にもpose_body=顔なしバリアントが正式に
    あるため、68点を消しても視線・頭の向きの符号化は失われない。

    ★2026-07-18 auto追加 (真ロップ実害): 高頭身 (スライダー1.8=3.6頭身
    以上) は頭がセル内で小さく、5点の顔がブロブ化して実質判読不能 —
    片耳3/4語彙+プロンプト明文でも斜め前の歩行が向こう向きに化けた。
    高頭身は実写プロポーションに近く68点の学習分布そのものなので、
    スライダー連動で自動有効化する (チビ量産=スライダー1.0-1.3は
    従来通りoff)。on/off の明示指定は従来通り最優先。"""
    v = os.environ.get("SM_POSE_FACE68", "auto").strip().lower()
    if v in ("on", "1", "true", "yes"):
        return True
    if v in ("off", "0", "false", "no"):
        return False
    return _leg_scale_env() >= 1.8

_ANGLE_KEYS = ("hipR", "hipL", "kneeR", "kneeL", "shR", "shL", "elbR", "elbL")
# 歩行サイクル = 接地A→通過→接地B→通過→(接地Aへ戻る)。5番(中間)は
# walk_codex のシート組み立て用のポーズでサイクルの一部ではない。
_CYCLE = (1, 2, 3, 4)


def _leg_scale_env() -> float:
    """頭身ノブ (hybrid_walk._leg_scale と同じ意味・同じ既定)。
    2026-07-16レンジ開放: 0〜4=最大8頭身 (実効はhead_frac_for_leg_scaleが
    飽和)。"""
    try:
        return max(0.6, min(4.0, float(os.environ.get("SM_LEG_SCALE",
                                                      "1.0"))))
    except ValueError:
        return 1.0


def _f_env(name: str, default: float, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, float(os.environ.get(name, "") or default)))
    except (TypeError, ValueError):
        return default


def walk_layout(n: int, idle: int | None = None,
                tail: int | None = None) -> tuple:
    """フレーム配分 (idle_n, cycles, period, tail_n) の単一情報源。

    先頭 idle_n フレームは直立 (IDLE_POSE) で静止し、続いて歩行サイクル
    (歩行区間の先頭と終端=idle_n+cycles*period は同位相)、末尾 tail_n
    フレームは再び直立で立ち止まる。シートの直立コマは末尾静止から採る
    (2026-07-17ユーザー発案「2周期したら骨を立ち止まらせて、その安定した
    フレームを直立フレームに」): 先頭の直立はi2v条件+latent固定で参照の
    位置・角度に錨止めされるが、歩行コマは骨格の言い分で立つため、骨格が
    参照とズレたキャラでは直立コマだけ別の場所・角度になっていた (7/16
    実測: 歩行同士±1px、idleだけ体軸2〜3px別)。末尾静止は歩行と同じ
    骨格アンカーに立つのでシート内の継ぎ目が発生源から消える。
    先頭 idle は据え置き (i2vはframe0=参照立ち絵から始まるため、いきなり
    歩行骨格だと出だしで参照と骨格が衝突する助走として必要)。
    末尾静止は「歩行のケイデンスを壊さない場合のみ」確保する: 旧来の
    n (33/49/81) は tail=0 で従来配分と完全一致=既存動画・低VRAM後退・
    旧テスト資産と互換。57f -> (6, 2, 21.0, 8) = 直立6+歩行2周期43+静止8。
    41f -> (6, 1, 26.0, 8) (かつてスロモ理由でラダーから外した41fが復権)。
    骨格生成と pipeline のビルダーヒント (--gait-period/--gait-start/
    --gait-end) の両方が必ずこの関数を使うこと (配分のズレ=コマ選出の
    ズレ)。"""
    n = max(2, int(n))
    if idle is None:
        idle = int(_f_env("SM_POSE_IDLE", 6, 0, 24))
    idle = max(0, min(int(idle), n // 4))

    def _core(nn: int) -> tuple:
        m = nn - idle
        cyc = max(1, round((m - 1) / 80.0 * 3))
        return cyc, (m - 1) / float(cyc)

    if tail is None:
        tail = int(_f_env("SM_POSE_TAIL", 8, 0, 24))
    tail = max(0, min(int(tail), n // 4))
    if tail:
        cyc, period = _core(n - tail)
        # ケイデンス圏 (実証済み21〜26中心) を外れるなら末尾静止を諦めて
        # 従来配分へ: 49f=(6,2,21)のまま (静止を確保すると period34 の
        # スローモーション歩行になる)、33f=(6,1,26)のまま (period18の
        # 早歩きになる)。既存資産の互換もこのガードが守る。
        if not (20.0 <= period <= 27.5):
            tail = 0
    if not tail:
        cyc, period = _core(n)
    return idle, cyc, period, tail


def walk_angles_at(phase: float, arm_swing: float = 1.0,
                   leg_swing: float = 1.0) -> dict:
    """位相 0..1 -> 関節角。WALK_POSES を周回線形補間する。

    arm_swing: 肩の振り角の倍率。かつては腕短縮 (最大50%) の補償として
    既定1.5倍だったが、腕長を実寸へ戻した2026-07-15以降は素の1.0が既定
    (latent_refineは骨格に忠実で、補償値のままだと振りが大げさに写る —
    ユーザー裁定「1.0既定にして」)。
    leg_swing: 股・膝の振り角の倍率。歩幅・腿上げを一括で抑える/強める
    A/Bレバー (SM_POSE_LEG_SWING、既定1.0。「足の動きが大げさ」への
    調整はまず bob=1.0 復帰、それでも過剰なら 0.8 あたりから)。"""
    p = (phase % 1.0) * len(_CYCLE)
    j = int(p) % len(_CYCLE)
    t = p - int(p)
    a = WALK_POSES[_CYCLE[j]]
    b = WALK_POSES[_CYCLE[(j + 1) % len(_CYCLE)]]
    ang = {k: a[k] + (b[k] - a[k]) * t for k in _ANGLE_KEYS}
    if arm_swing != 1.0:
        ang["shR"] *= arm_swing
        ang["shL"] *= arm_swing
    if leg_swing != 1.0:
        for k in ("hipR", "hipL", "kneeR", "kneeL"):
            ang[k] *= leg_swing
    return ang


def _gait_mode() -> str:
    """SM_POSE_GAIT: walk (既定) / crawl = 四つん這い (体格メニュー
    「四足歩行(骨格固定)」2026-07-21ユーザー裁定「人型骨格を四つん這いに
    して四足に対応」)。"""
    v = os.environ.get("SM_POSE_GAIT", "walk").strip().lower()
    return "crawl" if v == "crawl" else "walk"


def _crawl_J(fig: Figure, angles: dict) -> dict:
    """四つん這い(ハイハイ)の関節ワールド座標。

    人型骨格をクロール姿勢に組み直す: 膝と手が接地・胴ほぼ水平・頭は
    前方で持ち上げ。歩容は WALK_POSES の振り角を流用し、対角肢 (右手+
    左膝が同時に前) が自然に成立する (WALK_POSESが元々 shR=-hipR 系の
    対角位相)。寸法は fig.total (立ち身長) 基準の比率で組む — 参照の
    ハイハイ立ち絵の bbox に写像されるのはセル側のレターボックスなので、
    ここでは解剖学的な比率よりクロールらしい輪郭を優先する。
    座標系: y=上, z=前 (進行方向), 接地面 y=0。"""
    import math as _m
    T = fig.total
    hipx = fig.sh_x * 0.9
    J: dict = {}
    # 脚: 腰(後方)から腿が下へ、振り角で前後スイング。膝が接地点
    for side, tag in ((+1, "R"), (-1, "L")):
        phi = _m.radians(float(angles.get(f"hip{tag}", 0.0)) * 0.55)
        hip = (side * hipx, 0.36 * T, -0.30 * T)
        knee = (hip[0], hip[1] - 0.36 * T * _m.cos(phi),
                hip[2] + 0.36 * T * _m.sin(phi))
        # 脛は膝から後方へ寝かせる (すね接地・足は後ろで軽く浮く)
        ankle = (knee[0], knee[1] + 0.04 * T, knee[2] - 0.30 * T)
        toe = (ankle[0], ankle[1] + 0.02 * T, ankle[2] - 0.10 * T)
        J[f"hip_{tag}"] = hip
        J[f"knee_{tag}"] = knee
        J[f"ankle_{tag}"] = ankle
        J[f"toe_{tag}"] = toe
    # 腕=前脚: 肩(前方)から真下へ、振り角で前後スイング。手が接地点
    for side, tag in ((+1, "R"), (-1, "L")):
        psi = _m.radians(float(angles.get(f"sh{tag}", 0.0)) * 0.55)
        sh = (side * fig.sh_x, 0.42 * T, 0.18 * T)
        elbow = (sh[0], sh[1] - 0.22 * T * _m.cos(psi),
                 sh[2] + 0.22 * T * _m.sin(psi))
        psi2 = psi * 1.15
        wrist = (elbow[0], elbow[1] - 0.20 * T * _m.cos(psi2),
                 elbow[2] + 0.20 * T * _m.sin(psi2))
        J[f"sh_{tag}"] = sh
        J[f"elbow_{tag}"] = elbow
        J[f"wrist_{tag}"] = wrist
    # 胴・首・頭: 頭は前方で持ち上げて前を見る (ハイハイの赤ちゃん)
    J["neck"] = (0.0, 0.46 * T, 0.22 * T)
    hr = getattr(fig, "face_r", fig.head_r)
    J["head_c"] = (0.0, 0.46 * T + hr * 0.9, 0.30 * T)
    J["chin"] = (J["head_c"][0], J["head_c"][1] - hr * 0.6,
                 J["head_c"][2] + hr * 0.5)
    J["head_top"] = (0.0, J["head_c"][1] + hr * 0.9, J["head_c"][2])
    J["torso"] = (0.0, 0.40 * T, -0.05 * T)
    return J


def _ground_shift(fig: Figure, angles: dict, leg_cross: float = 1.0,
                  gait: bool = False) -> float:
    """接地補正量 (ワールドy)。低い方の足首を接地線に着けるための沈み込み。

    従来は腰の高さが固定で、脚を開く接地ポーズほど足首が浮いていた
    (=歩行の上下動が無い。2026-07-13ユーザー指摘)。低い方の足首と
    「直立時の足首高さ」の差だけ全身を沈めると、接地で最低・通過で最高の
    自然な上下動 (振幅≈脚長×(1-cos振り角)≈脚長の1割) が幾何から生まれ、
    足も毎フレーム接地する。"""
    if _gait_mode() == "crawl":
        return 0.0        # 四つん這いは膝・手の接地が幾何から常に成立
    J = fig.joints(**{k: angles[k] for k in _ANGLE_KEYS},
                   leg_cross=leg_cross, gait=gait)
    nominal = fig.hip_y - fig.leg_upper - fig.leg_lower   # 直立時の足首y
    return max(0.0, min(J["ankle_R"][1], J["ankle_L"][1]) - nominal)


def _persp_dist() -> float:
    """弱透視の視距離 (キャラ全高の倍数)。0/off=平行投影 (旧挙動)。

    OpenPose/DWPoseの教師データは透視カメラの実写に由来するため、奥行きは
    「近い点ほど画面中心から離れて大きく写る」形で2Dパターンに刻まれて
    いる。平行投影の骨格は前後の動き (腕振り・脚のストライド) が正面・
    背面でほぼ潰れ、モデルが前後を読めず振り子状の腕・脚の前後反転を
    起こす (2026-07-17 C43実害)。既定3.0=全高の3倍の視距離 (全身実写の
    標準的な撮影距離感、前後の足で±5%前後の拡縮差)。"""
    v = os.environ.get("SM_POSE_PERSP", "").strip().lower()
    if v in ("off", "0", "false", "no", "none"):
        return 0.0
    try:
        return max(1.5, min(20.0, float(v))) if v else 3.0
    except ValueError:
        return 3.0


def _project(p, yaw: float, scale: float, cx: float, base_y: float,
             total: float = 0.0):
    """mannequin3d.project_point と同じ投影 + キャンバス配置。
    ワールド原点(接地面中心)が (cx, base_y) に写る。

    total>0 のとき弱透視 (_persp_dist): ヨー回転後のz (体の立ち平面=0、
    +がカメラ寄り) に応じて、体中心 (全高の半分) を軸に画面オフセットを
    拡縮する。直立の体幹はz≈0で不変=参照との整列は崩れず、前後に
    振り出した手足だけが「近い=大きく低く / 遠い=小さく高く」の
    実写と同じ奥行き署名を得る。"""
    q = _rotx(_roty(p, yaw), ELEV_DEG)
    if total > 0.0:
        f = _persp_dist()
        if f > 0.0:
            c_yaw = math.cos(math.radians(yaw))
            if c_yaw < -0.5:
                # 背面系は視距離を詰めて奥行き署名を増幅する: 2D骨格の脚は
                # 奥行きが両義的 (膝が手前でも奥でも同じ画素位置を満たす)
                # で、アニメ学習データに稀な「背面歩行」は事前分布が
                # 正面の膝を選びがち (2026-07-17 C43実害: 背面の脚だけ
                # 前後反転が残った)。SM_POSE_PERSP_BACK=倍率 (1.0で無効)
                f = max(1.5, f * _f_env("SM_POSE_PERSP_BACK",
                                        0.65, 0.3, 1.0))
            elif (c_yaw > 0.35
                  and abs(math.sin(math.radians(yaw))) >= 0.35):
                # 斜め前も視距離を詰める: 高頭身は顔が小さく3/4シグナルが
                # 弱いため、歩行中に「横歩き」事前分布へ吸われて一旦
                # 真横を向く (2026-07-17 真ロップ実害。チビは巨大な顔が
                # 45°を保持するので非発現)。近い足ほど大きく低く写る
                # 署名で「斜め前進」を保つ。SM_POSE_PERSP_DIAG=倍率
                # ★既定0.75→0.6 (2026-07-18): 半球分離+顔正面化を入れても
                # 真ロップ(ls1.5)の斜め前歩行が真横で妥協した — 後頭部は
                # 消え顔は残ったので、残る横歩き事前分布への対抗署名を
                # 一段強める (背面のPERSP_BACK=0.65と同水準)
                f = max(1.5, f * _f_env("SM_POSE_PERSP_DIAG",
                                        0.60, 0.3, 1.0))
            zc = _roty(p, yaw)[2]           # 奥行き: 体平面からの偏差
            k = (f * total) / max(f * total - zc, 1e-6)
            yc = (total / 2.0) * math.cos(math.radians(ELEV_DEG))
            q = (q[0] * k, yc + (q[1] - yc) * k, q[2])
    return (cx + q[0] * scale, base_y - q[1] * scale)


def _keypoints(fig: Figure, angles: dict, yaw: float, scale: float,
               cx: float, base_y: float, leg_cross: float = 1.0,
               gait: bool = False, yaw_head: float | None = None,
               face_fwd: float = 1.0) -> list:
    """OpenPose 18点の (x,y) or None (非可視) を返す。

    yaw_head: 顔点(鼻目耳)だけ別ヨーで描く (ヨー追従の頭専用化、
    2026-07-16ユーザー裁定「斜め前の体は45度でいい。足運びが絵の角度と
    あってない」— 体=公称ヨーで歩行方向を守り、頭=立ち絵の実測ヨーに
    寄り添う。頭中心は体軸上にあるため両ヨーで同位置に写り、顔点だけが
    頭中心まわりで回る=実写の「歩きながらの頭部回転」と同じ語彙)。
    face_fwd: 顔点の前方突出(zオフセット)の倍率 (_face_fwd_scale が
    参照シルエットの前縁から実測した0..1。髪ボリュームでface_rが過大な
    キャラの鼻が実際の顔より前に出る=首前傾スマホ首宣言の矯正)。"""
    if yaw_head is None:
        yaw_head = yaw
    _crawl = _gait_mode() == "crawl"
    if _crawl:
        # 頭中心が体軸から前方 (z=0.30T) に外れているため、頭別ヨーだと
        # 顔クラスタが首から分離する — crawlは頭=体ヨーに固定 (ハイハイの
        # 頭は進行方向を向くので語彙上も正しい)
        yaw_head = yaw
    J = (_crawl_J(fig, angles) if _crawl
         else fig.joints(**angles, leg_cross=leg_cross, gait=gait))
    kps: list = [None] * 18
    # Neck = 肩の中点 (mannequin の肩と同じ高さ・中心)。crawlは前方の首
    kps[1] = _project(J["neck"] if _crawl
                      else (0.0, fig.shoulder_y - 0.01, 0.0),
                      yaw, scale, cx, base_y, total=fig.total)
    for op_i, name in _BODY_MAP.items():
        kps[op_i] = _project(J[name], yaw, scale, cx, base_y,
                             total=fig.total)
    # 顔: 頭中心 + r単位オフセット、法線カリング。face_r はキャラ実測の
    # 頭幅から入る上書き (無ければ絶対系の頭半径)。耳が髪の外に浮くと
    # VACEがそこに点を描く (2026-07-12 ロップ実害) ため輪郭内に収める。
    # crawlは前方へ持ち上げた頭中心を使う (顔は進行方向を向く)
    hc = J["head_c"] if _crawl else (0.0, fig.head_cy, 0.0)
    r = getattr(fig, "face_r", fig.head_r)
    for op_i, (off, n, thr) in _FACE_PTS.items():
        nn = math.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
        nq = _rotx(_roty((n[0] / nn, n[1] / nn, n[2] / nn), yaw_head),
                   ELEV_DEG)
        if nq[2] < thr:
            continue
        # z(奥行き)だけ face_fwd 倍: 鼻・目の前方突出を実顔に合わせて
        # 縮める (法線=可視判定は角度の宣言なので触らない)。
        # face_dy = 顔クラスタの縦スナップ (立ち絵の目の行への実測合わせ、
        # _fit_figure_to_char が設定。鼻目耳が一体で動き顔の形は不変)
        zoff = off[2]
        if op_i in (14, 15) and abs(
                math.sin(math.radians(yaw_head))) >= 0.85:
            # 横顔の目は鼻先より奥 (2026-07-18 C01_hs08実害「横顔の目が
            # 鼻に近すぎ」: 目z0.75r≒鼻z0.78rで横投影がほぼ重なる。
            # 実顔の瞳の奥行き≈0.58rへ、横顔時のみ描画を補正 — 正面/斜めの
            # 較正とヨー推定式 (0.75前提) は触らない)
            zoff = 0.58
        p = (hc[0] + off[0] * r,
             hc[1] + off[1] * r + getattr(fig, "face_dy", 0.0)
             + (getattr(fig, "nose_dy", 0.0) if op_i == 0 else 0.0),
             hc[2] + zoff * r * face_fwd)
        kps[op_i] = _project(p, yaw_head, scale, cx, base_y,
                             total=fig.total)
    # ---- 奥耳の遮蔽 (2026-07-18 真ロップ実害、ユーザー診断「奥の耳の
    # ポイントが目より手前にあるから向こう向いた動画になる」):
    # 体が斜め以上に回ったセルで両耳が描かれると、奥耳の投影が鼻より
    # 進行方向側に落ちる —「耳が顔より前」は後頭部ビューの署名で、
    # 生成が向こう向きに化ける。両耳が出るのはヨー追従が立ち絵実測で
    # 頭を正面寄り (下限10°) へ寄せたときに、45°較正の法線ゲートを
    # 奥耳がすり抜けるため。実写DWPoseの3/4は片耳が正準 (奥耳は頭に
    # 遮蔽され低スコア欠損) — 頭が中間ヨーに適応しても、体のヨーが
    # 斜めなら耳は3/4語彙 (片耳) を守る。
    # ★例外=顔正面化 (yaw_head≈0、2026-07-18): 完全正面の頭の正準は
    # 「鼻+両目+両耳」の対称パターンで、両耳は±0.92rの最外側に写り
    # 鼻と重ならない — 片耳へ削ると意図した正面語彙がまた中間の
    # 3/4語彙へ弱まるため、遮蔽はスキップして両耳を残す ----
    if (abs(math.sin(math.radians(yaw))) >= 0.35
            and abs(math.sin(math.radians(yaw_head))) >= 0.10
            and kps[16] is not None and kps[17] is not None):
        # 奥=頭ヨーで回した位置zが深い側 (頭正面 |sin|<0.10 は上の
        # 条件で遮蔽自体をスキップ済みなので頭ヨーで確定できる)
        z16 = _roty(_FACE_PTS[16][0], yaw_head)[2]
        z17 = _roty(_FACE_PTS[17][0], yaw_head)[2]
        kps[16 if z16 < z17 else 17] = None
    # ---- 脚交差の遮蔽カリング (2026-07-15 ロップオリジンback_right実害:
    # 交差の瞬間、奥脚のキーポイントが画面上で手前脚とほぼ重なり
    # 「同じ画素に2本の脚を描け」という矛盾指示になって奥脚が手前脚と
    # 粘土融合する。実写のOpenPose教師データならこの瞬間の奥脚は
    # 遮蔽=欠損なので、膝・足首を欠損扱いにして手前脚だけを宣言する。
    # 腰(hip)は胴に付くため残す。SM_POSE_LEG_OCCLUDE=offで無効) ----
    if (os.environ.get("SM_POSE_LEG_OCCLUDE", "on").strip().lower()
            not in ("off", "0", "false", "no")
            and all(kps[i] is not None for i in (9, 10, 12, 13))
            # 奥行き差が画面に出る向き (横・斜め) だけ。正面/背面は
            # 交差しても左右に並んで見えるため隠さない
            and abs(math.sin(math.radians(yaw))) >= 0.5
            # すれ違い位相: 股角がほぼ揃う瞬間 (前後スイングの交点)。
            # 直立区間 (両膝伸び) は膝屈曲ゲートで除外する
            and abs(angles.get("hipR", 0.0) - angles.get("hipL", 0.0)) < 10.0
            and max(angles.get("kneeR", 0.0),
                    angles.get("kneeL", 0.0)) > 8.0):
        # 奥=カメラ回転後のzが小さい側 (+z=カメラ向き)。
        # op 8-10 は mannequin *_L、11-13 は *_R (左右スワップ規約)
        z_l = _rotx(_roty(J["knee_L"], yaw), ELEV_DEG)[2]
        z_r = _rotx(_roty(J["knee_R"], yaw), ELEV_DEG)[2]
        far = (9, 10) if z_l < z_r else (12, 13)
        for i in far:
            kps[i] = None
    return kps


def _face68_pts(fig: Figure, yaw: float, scale: float, cx: float,
                base_y: float) -> list:
    """可視な顔68点の画面座標 [(x,y), ...] を返す。背面系は空リスト。

    実写のDWPoseは背面〜斜め後ろで顔推定が崩れ、68点全部がスコアゲート
    (0.3) で消える — 同じ挙動を鼻法線のグローバルゲートで再現する
    (正面〜横までは発火、back系3方向は全滅)。横・斜めは点ごとの
    radial法線カリングで可視半面だけ残す (2D回帰の実写でも奥半面は
    低スコアで欠けるため、部分顔は語彙内)。"""
    n0 = _FACE_PTS[0][1]
    nn = math.sqrt(n0[0] ** 2 + n0[1] ** 2 + n0[2] ** 2)
    gate = _rotx(_roty((n0[0] / nn, n0[1] / nn, n0[2] / nn), yaw), ELEV_DEG)
    if gate[2] < -0.15:
        return []
    hc = (0.0, fig.head_cy, 0.0)
    r = getattr(fig, "face_r", fig.head_r)
    out = []
    for off, nrm, thr in _FACE68:
        nq = _rotx(_roty(nrm, yaw), ELEV_DEG)
        if nq[2] < thr:
            continue
        p = (hc[0] + off[0] * r,
             hc[1] + off[1] * r + getattr(fig, "face_dy", 0.0),
             hc[2] + off[2] * r)
        out.append(_project(p, yaw, scale, cx, base_y, total=fig.total))
    return out


def _ellipse_poly(p0, p1, half_w: float, seg: int = 24) -> list:
    """肢を表す回転楕円 (OpenPose の cv2.ellipse2Poly 相当) の頂点列。"""
    mx, my = (p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0
    half_l = max(half_w, math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / 2.0)
    ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    ca, sa = math.cos(ang), math.sin(ang)
    pts = []
    for i in range(seg):
        t = 2.0 * math.pi * i / seg
        x, y = half_l * math.cos(t), half_w * math.sin(t)
        pts.append((mx + x * ca - y * sa, my + x * sa + y * ca))
    return pts


_FACE_IDX = (0, 14, 15, 16, 17)      # 鼻・両目・両耳 (BODY_18)
_EYE_IDX = (14, 15)                  # 両目のみ


def _face_points_mode() -> str:
    """骨格に描く顔点の範囲 (SM_POSE_FACE_POINTS)。

    on (既定) = 従来どおり鼻目耳+face68。
    noeyes    = 目とface68だけ消し、鼻・耳は残す。2026-07-20実走の教訓:
                目の誤配置が顔を壊す一方、鼻・耳は「頭の位置と向き」の
                宣言で、全部消すと猫背が復活した (頭の所在が語られなく
                なる)。顔の細部の権威は頭部エッジへ委ねる折衷。
    off       = 体のみ (全顔点なし)。実験用。"""
    v = os.environ.get("SM_POSE_FACE_POINTS", "on").strip().lower()
    if v in ("off", "0", "false", "no"):
        return "off"
    if v in ("noeyes", "no_eyes", "body+nose"):
        return "noeyes"
    return "on"


def _face_skip_idx() -> tuple:
    m = _face_points_mode()
    if m == "off":
        return _FACE_IDX
    if m == "noeyes":
        return _EYE_IDX
    return ()


_LEGS_KEEP = (8, 9, 10, 11, 12, 13)   # 腰・膝・足首 (左右)


def _body_parts_mode() -> str:
    """SM_POSE_BODY_PARTS: full (既定) / legs = 腰から下 (8-13) だけ描く。

    2026-07-21実験h (走る忍者実障害): 参照立ち絵が全方向「走りの途中」
    (前傾・拳・片脚上げ) の依頼に直立歩行骨格の上半身を毎回被せると、
    参照と制御の全面矛盾で二重人格化・ブロブ発明・頭身潰れが起きた。
    脚だけ骨格で誘導し、上半身の姿勢権威を参照立ち絵へ返す折衷モード。"""
    v = os.environ.get("SM_POSE_BODY_PARTS", "full").strip().lower()
    return "legs" if v in ("legs", "legs_only") else "full"


def _draw_openpose_onto(img: Image.Image, kps: list,
                        face68: list | None = None) -> None:
    """既存キャンバスへ1人分のOpenPose骨格を重ね描きする (グリッド用)。

    face68 を渡すと BODY_18 の上に白い顔ドットを重ねる (正準の描画順:
    draw_bodypose→draw_facepose。白はOP_COLORSに無い色なので骨格側と
    衝突しない)。SM_POSE_FACE_POINTS=noeyes は目とface68だけ・off は
    顔点全部を描かない (_face_points_mode 参照)。"""
    dr = ImageDraw.Draw(img)
    skip = set(_face_skip_idx())
    if _body_parts_mode() == "legs":
        skip.update(i for i in range(18) if i not in _LEGS_KEEP)
    sw = max(2, round(4 * min(img.width, img.height) / 512.0))  # 標準4px@512
    for li, (a, b) in enumerate(OP_LIMBS):
        if kps[a] is None or kps[b] is None:
            continue
        if a in skip or b in skip:
            continue
        col = tuple(int(c * 0.6) for c in OP_COLORS[li])
        dr.polygon(_ellipse_poly(kps[a], kps[b], sw), fill=col)
    for i in range(18):
        if kps[i] is None:
            continue
        if i in skip:
            continue
        x, y = kps[i]
        dr.ellipse((x - sw, y - sw, x + sw, y + sw), fill=OP_COLORS[i])
    if face68 and not skip:
        # 正準比: 関節円r4に対し顔ドットr3 (draw_facepose)
        fr = max(1, round(sw * 0.75))
        for x, y in face68:
            dr.ellipse((x - fr, y - fr, x + fr, y + fr),
                       fill=(255, 255, 255))


def draw_openpose(kps: list, width: int, height: int,
                  face68: list | None = None) -> Image.Image:
    """OpenPose 標準描画: 黒地、肢=0.6輝度の楕円、関節=全輝度の円。"""
    img = Image.new("RGB", (width, height), (0, 0, 0))
    _draw_openpose_onto(img, kps, face68)
    return img


def _fg_mask(ref_image):
    """立ち絵(マゼンタ地)の前景マスク (bool ndarray)。"""
    a = np.asarray(ref_image.convert("RGB")).astype(int)
    return ~((np.abs(a[..., 0] - 255) < 40) & (a[..., 1] < 60)
             & (np.abs(a[..., 2] - 255) < 40))


def _feet_axis_x(fxs):
    """足元バンドの体軸x (ソース画像列座標)。fxsが空ならNone。

    正面・背面は両足が「クラスタ2つ+間の空白」に分かれ、全画素の素朴な
    中央値は50%分位が落ちる側の足の内側エッジに張り付く (両足の画素質量差
    わずか1.5%で±半歩幅ぶれる不安定点。2026-07-16実害: ろーらちゃんの
    正面/背面で骨格だけが約7px横ズレ→参照キャンバスと骨格の綱引きで
    生成キャラが横に引っ張られて歩く)。列ランに分解して形で選ぶ:
      1ラン (横・斜め=両足が画面上で重なる) -> そのランの中央値 (従来挙動)
      2ラン以上 (正面・背面の左右の足)      -> 最外2ラン (=両足) の
                                               各中央値の中点 (股の真下)
    頑健化3点 (2026-07-16敵対的レビューの確認済み指摘4件への対策):
      ①微小列プレフィルタ: 質量が占有列中央値の2割未満の列 (接地影・
        裾ドット・AAフリンジ) をギャップ扱いに落とす — 両足が微小画素の
        橋渡しで1ランに融合し素朴中央値へ退化する再発経路を塞ぐ
      ②保持フィルタ=質量25%かつ幅40% (いずれも最大ラン比): 質量だけでは
        足首丈の髪毛先・杖先 (縦に長く質量はあるが細い) が「第2の足」を
        騙れる。実測: 正当な両足ペアの幅比は最小0.62 (量産61体)、
        毛先0.37/杖0.29は落ちる。幅も質量も足並みの太い接地尻尾だけは
        原理的に弁別不能 (既知の限界、SM_POSE_FEET_AXIS=offが逃げ道)
      ③最外2ラン (位置基準): 質量順は第3の接地ランがあると1px質量差で
        軸が半歩幅飛び、同点では安定ソートの右バイアスが出る。両足は
        スタンスの両端=最外が構造的に正しい。
    ランの区切りは空白2列以上のギャップ (占有列距離>2)。"""
    fxs = np.asarray(fxs)
    if len(fxs) == 0:
        return None
    counts = np.bincount(fxs)
    occ = np.where(counts > 0)[0]
    col_med = float(np.median(counts[occ]))
    occ = occ[counts[occ] >= max(1.0, 0.2 * col_med)]
    if len(occ) == 0:
        return float(np.median(fxs))
    runs = np.split(occ, np.where(np.diff(occ) > 2)[0] + 1)
    mass = np.array([int(counts[r].sum()) for r in runs])
    width = np.array([int(r[-1] - r[0] + 1) for r in runs])
    keep = [i for i in range(len(runs))
            if mass[i] >= 0.25 * mass.max()
            and width[i] >= 0.40 * width.max()]
    if not keep:                  # 最大質量ランが幅落ち∧最大幅ランが質量
        keep = [int(np.argmax(mass))]   # 落ちの病的形状: 最重ランに帰着

    def med(i):
        r = runs[i]
        return float(np.median(fxs[(fxs >= r[0]) & (fxs <= r[-1])]))

    if len(keep) == 1:
        return med(keep[0])
    return (med(keep[0]) + med(keep[-1])) / 2.0


def _char_box(ref_image, width: int, height: int):
    """立ち絵(マゼンタ地)の前景bboxを、サーバ _fit_image と同じ比率維持
    レターボックスで出力キャンバス座標へ写す -> (体軸x, 接地y, 高さpx)。
    骨格をキャラの画面上の大きさ・位置に合わせるため (方向間でスケールが
    揃い、他方向のAniSora生成とも整合する)。

    体軸x=足元バンド(下10%行)の前景から _feet_axis_x で推定 (2026-07-16
    「歩きが猫背」対策): bbox中心を体軸にすると、横・斜めでは後ろ髪の
    ボリュームでbboxが後方に膨らみ、骨格全体が実測7〜20px(0.1〜0.3頭)
    後ろへズレる。すると骨格の足・腰は参照の体より後ろ、鼻は前方に届く=
    「足を引いて頭を突き出した人」という猫背指示になる (直立コマはlatent
    固定が守るが歩行コマで露呈、ろーらちゃん実走で確定)。足元は接地点
    そのもの=視点非依存の真の体軸。正面・背面の「両足の間の空白」で
    中央値が片足の内側エッジへ張り付く罠は _feet_axis_x が2クラスタの
    中点を取って回避する。異常時(bbox中心からキャラ幅40%超=足元バンドが
    キャラ本体を捉えていない疑い)はbbox中心へフォールバック (大きな
    後ろ髪の正当な補正は幅の3〜4割に達し得るためガードはこれ以上
    締めない)。"""
    im = ref_image.convert("RGB")
    iw, ih = im.size
    sc = min(width / iw, height / ih)
    ox, oy = (width - iw * sc) / 2.0, (height - ih * sc) / 2.0
    fg = _fg_mask(im)
    ys, xs = np.where(fg)
    if len(xs) == 0:            # 前景なし: 中央に高さ86%で置くフォールバック
        return width / 2.0, height * 0.93, height * 0.86
    cx = ox + (int(xs.min()) + int(xs.max()) + 1) / 2.0 * sc
    y0i, y1i = int(ys.min()), int(ys.max())
    ch_img = y1i - y0i + 1
    feet = fg[max(y0i, y1i - max(1, int(0.10 * ch_img))):y1i + 1]
    fxs = np.where(feet)[1]
    if (os.environ.get("SM_POSE_FEET_AXIS", "on").strip().lower()
            in ("off", "0", "false", "no")):
        fxs = fxs[:0]           # 旧挙動 (bbox中心=体軸) へ戻す逃げ道
    if len(fxs) > 0:
        fx = ox + (_feet_axis_x(fxs) + 0.5) * sc
        bw = (int(xs.max()) - int(xs.min()) + 1) * sc
        if abs(fx - cx) <= 0.40 * bw:
            cx = fx
    bottom = oy + (y1i + 1) * sc
    ch = ch_img * sc
    return cx, bottom, ch


# ---- 目ペア検出による頭身の検証・救済 (2026-07-18、ユーザー指摘
# 「頭身の算出がかなり甘い」の対策。66体の立ち絵で較正) ----
# 背景: シルエットのくびれだけでは頭身は原理的に決まらないキャラがいる。
# ウルファール級 (髪カーテンが首を覆う) は本物の首がくびれ条件を満たせず、
# 腰下の偽くびれ (hf0.581) がスライダー由来の信頼窓をすり抜ける
# (スライダー1.5なら窓[0.185,0.60]に収まってしまう=窓はスライダーが
# 絵に合っている時しか守れない)。目は絵柄に依らず「明るい肌の中の
# コンパクトな暗ブロブ2つ」で、スライダー非依存の物差しになる。
# 較正 (66体): 健全なくびれ実測は pinch/hf_eye = 0.40..1.17 に収まり、
# 偽くびれは 2.21 — 閾値1.5で完全分離。さぶり(猫耳付け根)・C47(前髪の影)
# の偽ペアは「ブロブ間が明るい肌」条件で棄却され本物の目に落ち着く。
_EYE_VETO_RATIO = 1.5      # pinch/hf_eye がこれ超 = 顔より下の偽くびれ
# 目行→頭身の逆算はモデル幾何から解く (旧: 0.575定数はチビで荒れた —
# 26体診断 2026-07-18夜: 「目は検出済みなのにフィット未反映」13体):
#   目の画面行/ch = hf * (0.5 + (0.15 + 0.75*tanE) * face_r/H)
#   (目y=head_cy-0.15r・z=0.75rの俯角投影。face_rはw_ear実測とhfの
#   相互依存なので2-3回反復)
_EYE_SOLVE_BAND = (0.70, 1.60)   # くびれ実測に対する目補正の許容倍率
_EYE_DY_MAX = 0.40               # 顔クラスタ残差スナップの上限 (head_r比)


def _cc_label_runs(mask: np.ndarray) -> list:
    """numpyのみの連結成分 (4近傍・行ランのunion-find)。
    戻り値: [x0, x1, y0, y1, area] のリスト。"""
    H, W = mask.shape
    parent: dict = {}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    labels = np.zeros((H, W), dtype=np.int32)
    nxt = 1
    prev_runs: list = []
    for y in range(H):
        row = mask[y]
        runs = []
        x = 0
        while x < W:
            if row[x]:
                x2 = x
                while x2 + 1 < W and row[x2 + 1]:
                    x2 += 1
                lab = None
                for (px0, px1, pl) in prev_runs:
                    if px0 <= x2 and x <= px1:
                        rp = find(pl)
                        if lab is None:
                            lab = rp
                        elif rp != lab:
                            parent[rp] = lab
                if lab is None:
                    lab = nxt
                    parent[lab] = lab
                    nxt += 1
                labels[y, x:x2 + 1] = lab
                runs.append((x, x2, lab))
                x = x2 + 1
            else:
                x += 1
        prev_runs = runs
    comps: dict = {}
    ys, xs = np.nonzero(labels)
    for y, x in zip(ys, xs):
        r = find(int(labels[y, x]))
        c = comps.setdefault(r, [10 ** 9, -1, 10 ** 9, -1, 0])
        c[0] = min(c[0], int(x))
        c[1] = max(c[1], int(x))
        c[2] = min(c[2], int(y))
        c[3] = max(c[3], int(y))
        c[4] += 1
    return list(comps.values())


def _detect_eye_pair(ref_image, hf_hint: float | None = None) -> float | None:
    """front立ち絵から目ペアを検出し、目の行 (fg上端基準px) を返す。

    目=「明るい肌の中に横に並ぶコンパクトな暗ブロブ2つ」。条件:
    上端55%帯 / 面積・アスペクト・充填率 / 左右のy揃い / 間隔がブロブ幅
    相応 / 中心線がfg中央帯 / ★ブロブ間の非ブロブ画素が明るい肌
    (median>=135 かつ ブロブ+30以上 — 髪の中の暗部ペア (さぶりの猫耳
    付け根・C47の前髪影) はここで落ちる)。複数ペアは最上位を採用。
    検出できなければ None (呼び出し側は従来挙動のまま)。"""
    try:
        if not hasattr(ref_image, "convert"):
            ref_image = Image.open(ref_image)
        fg = _fg_mask(ref_image)
        ys, xs = np.where(fg)
        if len(ys) < 100:
            return None
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        ch = y1 - y0 + 1
        bw = x1 - x0 + 1
        a = np.asarray(ref_image.convert("RGB")).astype(int)
        lum = 0.30 * a[..., 0] + 0.59 * a[..., 1] + 0.11 * a[..., 2]
        # 窓0.62 (旧0.55): 超チビの巨大目が下端で切れてfill棄却されていた
        # (26体診断: 0714ロップ)
        top = slice(y0, y0 + int(0.62 * ch))
        reg_fg = fg[top]
        reg_lum = lum[top]
        if reg_fg.sum() < 50:
            return None
        from PIL import ImageFilter

        def _fill_ratio(mask, cx0, cx1, cy0, cy1, area) -> float:
            """穴埋め後の充填率。アニメの虹彩はハイライトで中空になり、
            素の面積/箱ではfill棄却される (26体診断: H/S/ピンク/0714)。
            箱内の非マスク画素のうち箱の縁から到達できないもの=穴。"""
            sub = mask[cy0:cy1 + 1, cx0:cx1 + 1]
            hh, ww = sub.shape
            if hh * ww <= 0:
                return 0.0
            outside = np.zeros_like(sub, dtype=bool)
            stack = ([(0, x) for x in range(ww)]
                     + [(hh - 1, x) for x in range(ww)]
                     + [(y, 0) for y in range(hh)]
                     + [(y, ww - 1) for y in range(hh)])
            for (sy, sx) in stack:
                if not sub[sy, sx] and not outside[sy, sx]:
                    q = [(sy, sx)]
                    outside[sy, sx] = True
                    while q:
                        yq, xq = q.pop()
                        for ny, nx in ((yq-1, xq), (yq+1, xq),
                                       (yq, xq-1), (yq, xq+1)):
                            if (0 <= ny < hh and 0 <= nx < ww
                                    and not sub[ny, nx]
                                    and not outside[ny, nx]):
                                outside[ny, nx] = True
                                q.append((ny, nx))
            holes = int((~sub & ~outside).sum())
            return (area + holes) / float(hh * ww)

        def _try_thr(thr: float):
            dark = (reg_fg & (reg_lum < thr)).astype(np.uint8) * 255
            dm = Image.fromarray(dark).filter(ImageFilter.MaxFilter(3)) \
                .filter(ImageFilter.MinFilter(3))
            mask = np.asarray(dm) > 128
            cands = []
            for cx0, cx1, cy0, cy1, area in _cc_label_runs(mask):
                w = cx1 - cx0 + 1
                h = cy1 - cy0 + 1
                if area < max(10, 3e-5 * ch * ch) or area > 0.02 * ch * ch:
                    continue
                if not (0.35 <= h / max(1, w) <= 3.0):
                    continue
                if _fill_ratio(mask, cx0, cx1, cy0, cy1, area) < 0.35:
                    continue
                cands.append((cx0 + w / 2.0, cy0 + h / 2.0, w, h, area,
                              cx0, cx1, cy0, cy1))
            pairs = []
            for i in range(len(cands)):
                for j in range(i + 1, len(cands)):
                    A, B = cands[i], cands[j]
                    if A[0] > B[0]:
                        A, B = B, A
                    xa, ya, wa, ha, aa = A[:5]
                    xb, yb, wb, hb, ab = B[:5]
                    if abs(ya - yb) > 0.6 * max(ha, hb):
                        continue
                    sep = xb - xa
                    avg_w = (wa + wb) / 2.0
                    if not (0.9 * avg_w <= sep <= 5.0 * avg_w):
                        continue
                    if max(aa, ab) / max(1.0, min(aa, ab)) > 3.0:
                        continue
                    midx = (xa + xb) / 2.0
                    if not (x0 + 0.25 * bw <= midx <= x0 + 0.75 * bw):
                        continue
                    ymid = (ya + yb) / 2.0
                    # 目の深さゲート: 通常0.42・(0.42, 0.50]は「巨大目」
                    # (超チビ) の証拠がある時だけ許す — C39の服上の黒点対
                    # (6px) を弾きつつ、1.6頭身の低い目は通す
                    if ymid > 0.50 * ch:
                        continue
                    if (ymid > 0.42 * ch
                            and min(aa, ab) < 8e-4 * ch * ch):
                        continue
                    gx0, gx1 = int(A[6]) + 1, int(B[5])
                    if gx1 <= gx0:
                        continue
                    band = slice(max(0, int(ymid - max(ha, hb))),
                                 int(ymid + max(ha, hb)) + 1)
                    gap_mask = (~mask[band, gx0:gx1]
                                & reg_fg[band, gx0:gx1])
                    if gap_mask.sum() < 4:
                        continue
                    between = float(np.median(
                        reg_lum[band, gx0:gx1][gap_mask]))
                    span = slice(int(A[5]), int(B[6]) + 1)
                    bm = mask[band, span]
                    if not bm.any():
                        continue
                    blob_lum = float(np.median(reg_lum[band, span][bm]))
                    if between < 135 or between < blob_lum + 30:
                        continue
                    pairs.append((ymid, -(aa + ab), midx, max(ha, hb)))
            if not pairs:
                return None
            # ペア選択v3 (2026-07-18ラベルv5検収32/37broken、支配クラス=
            # 眉×前髪の誤認27件): 深さヒント方式は頭身推定の誤差を継承して
            # 不発だった (ヒント過小→眉ペアが「正準深度に最も近い」まま)。
            # 頭身非依存の判別へ: ①同じx位置に縦積みのペア (眉影の下に
            # 瞳) があれば常に下を採る ②残候補は合計面積の最大 (瞳は
            # 眉影より大きく描かれる) を採る
            def _midx(pr):
                return pr[2]
            survivors = []
            for cand in pairs:
                shadowed = False
                for other in pairs:
                    if other is cand:
                        continue
                    if (abs(_midx(other) - _midx(cand)) <= cand[3]
                            and other[0] > cand[0] + cand[3] * 0.6):
                        shadowed = True   # 下に同軸ペアあり=候補は眉影
                        break
                if not shadowed:
                    survivors.append(cand)
            survivors.sort(key=lambda t: t[1])   # t[1] = -(面積和)
            return float(survivors[0][0])

        # しきい値ラダー: 基準p10 → 濃色前髪と目が融合するキャラは
        # より暗い段で瞳だけが分離する (26体診断: C12/C13/C16/C17)。
        # 明るめの段は薄色瞳の救済
        p = np.percentile(reg_lum[reg_fg], [3, 5, 10, 18])
        thrs = []
        for t in (max(55.0, p[2]), max(40.0, p[1]), max(40.0, p[0]),
                  max(60.0, p[3])):
            if all(abs(t - u) > 2.0 for u in thrs):
                thrs.append(t)
        for thr in thrs:
            row = _try_thr(thr)
            if row is not None:
                return row
        return None
    except Exception:                     # noqa: BLE001
        return None                       # 検出はベストエフォート


def _fit_figure_to_char(fig: Figure, ref_image) -> Figure:
    """骨格をキャラの実測プロポーションに合わせる (2026-07-12 実障害対策)。

    mannequin3d.Figure の幅 (肩幅0.1625・頭半径0.25) はTシートワープ用の
    固定値で、チビキャラでは肩リムが体の外・耳が髪の外に置かれ、VACEが
    そこに「腕/点」を描いてしまう (ロップで骨丸見え・幽霊腕の実害)。
    立ち絵の前景幅プロファイルから頭身・肩幅・腰幅・頭幅を実測し、
    figの該当属性を上書きして骨格を輪郭の内側に収める。

    - 頭身: 幅プロファイルの首のくびれを頭/体の境界とし head_frac を
      再構築。走査は二段: まずチビ帯 (上から30..72%)、不発なら高頭身帯
      (max(0.125, 想定/1.8)..30%。2026-07-18: 4頭身超の首は30%より上で
      未走査だった)。くびれが浅い(<最大幅の82%)キャラは非チビ体型と
      みなし既定のまま。
    - 肩x = 顎下バンド (肩行〜下へ15%) 最大幅の32% (拡幅も許す。肩行
      そのものは首のくびれで過小になる 2026-07-17 C43実害)。腕鎖が
      シルエットからはみ出さないガードつき。腕の長さは短縮しない
      (体高比の既定のまま=手先≒拳の位置。旧実装の「肩幅と同率短縮」は
      SM_POSE_ARM_FIT=off でのみ残る 2026-07-15)。
    - 腰x = 腰高さの実測幅の30% (上限は既定値)。
    - 頭(耳・目・鼻の広がり) = 頭部最大幅の47% (耳0.92倍が輪郭内に
      収まる)。頭は縦横比が違うため face 用の半径のみ上書き。
    計測に失敗したら既定のまま返す (フォールバック安全)。"""
    try:
        fg = _fg_mask(ref_image)
        ys, xs = np.where(fg)
        if len(xs) < 100:
            return fig
        y0, y1 = int(ys.min()), int(ys.max())
        ch_img = y1 - y0 + 1                     # 画像座標での全高px
        prof = fg[y0:y1 + 1].sum(axis=1).astype(float)   # 行ごとの前景幅
        if len(prof) < 20:
            return fig
        # 移動平均で毛先ノイズをならす
        k = max(3, ch_img // 40)
        kern = np.ones(k) / k
        smooth = np.convolve(prof, kern, mode="same")
        # ---- 頭身: 「首のくびれ」を探す。単なる帯内最小幅は脚の細まりを
        #      拾う (2026-07-12レビューで数値実証: 脚が首より細いキャラで
        #      hfが0.72に張り付き、肩バーが脚に落ちる)。首の条件:
        #        (a) 局所最小 (上から順に走査し最上位を採用)
        #        (b) くびれ幅 < 上側最大幅(=頭)の82%
        #        (c) 上側最大幅の行がくびれ位置の半分より近い かつ 全高の
        #            28%以内 (チビの頭はくびれ直上で最も広い。6頭身の
        #            「肩下の胴」「腰のくびれ」に加え、ロングヘアの
        #            「髪の終わり+ドレス裾の再拡幅」を首と誤認する事故
        #            (2026-07-13 AI妹実障害: head_frac0.62誤検出で肩が腰に
        #            落ち、脚が圧縮されて歩幅が消えた) を棄却する)
        #        (d) 直下12%行以内に1.12倍以上へ再拡幅 (肩・胴の膨らみ。
        #            脚の細まりは直下で再拡幅しないので棄却される)
        #        (e) 直上12%行以内にも1.12倍以上 (頭からのテーパーで
        #            くびれに落ちる形。等幅胴の平坦帯を弾く)
        #      条件を満たす行が無ければ頭身は触らない。頭上のアクセサリ
        #      (ロップの花など)は幅が細いので(b)(c)に影響しない。スカート
        #      裾広がりのチビを誤棄却していた旧 below_max ゲートは廃止 ----
        lo, hi = int(0.30 * ch_img), int(0.72 * ch_img)
        rw_span = max(3, int(0.12 * ch_img))
        hf_knob = float(fig.head_frac)   # スライダー由来の想定頭身

        def _scan_pinch(rlo: int, rhi: int) -> int:
            for r in range(max(rlo, 1), min(rhi, len(smooth) - 2)):
                v = float(smooth[r])
                if not (v <= smooth[r - 1] and v <= smooth[r + 1]):
                    continue
                above = smooth[:r]
                am = float(above.max())
                am_row = int(np.argmax(above))
                rw = smooth[r + 1: r + 1 + rw_span]
                up = smooth[max(0, r - rw_span): r]
                if (v < 0.82 * am
                        and (r - am_row) * 2 <= r
                        and (r - am_row) <= 0.28 * ch_img
                        and len(rw)
                        and float(rw.max()) >= 1.12 * max(v, 1.0)
                        and len(up)
                        and float(up.max()) >= 1.12 * max(v, 1.0)):
                    return r
            return -1

        # ---- 実測の信頼窓 (2026-07-17 ウルファール実害): リアル頭身の
        # 立ち絵は髪カーテンがウエストのくぼみへ流れ、上の全条件を満たす
        # 「偽の首」を作る (hf0.58の超巨頭チビとして骨格を組み崩壊)。
        # シルエットだけではロングヘアチビ (AI妹級) と原理的に区別
        # できない (実測: AI妹の本物の首は深さ0.81/上側最大0.88rで、
        # ウルファールの偽首0.71/0.82より「偽物らしい」= 深さ・位置とも
        # 分離不能)。よって頭身スライダー由来の想定から1.8倍を超えて
        # 外れる実測は棄却し、スライダー値のまま組む。リアル頭身の子は
        # スライダーを絵に合わせるのが前提 (チビ量産はスライダー1.0前後×
        # くびれ実測0.35-0.6が全て窓内=挙動不変)。
        # SM_POSE_TALL_FIT=off で旧挙動 (無条件に実測が勝つ) ----
        def _trusted(p: int) -> int:
            if (p <= 0
                    or os.environ.get("SM_POSE_TALL_FIT", "on")
                    .strip().lower() in ("off", "0", "false", "no")):
                return p
            hf_meas = p / float(ch_img)
            return p if (hf_knob / 1.8 <= hf_meas <= hf_knob * 1.8) else -1

        pinch = _trusted(_scan_pinch(lo, hi))
        if pinch <= 0:
            # ---- 高頭身帯の第2走査 (2026-07-18 真ロップ実害、ユーザー
            # 診断「顔のパーツの位置取得ミスってる(肩も全然違う)」):
            # 4頭身超の立ち絵は首が上端30%より上 (実測24.2%・5条件全合格)
            # にあり、第1走査帯 [30%,72%] では走査すらされず、フィット
            # 不発でスライダー値のまま組んでいた — 顎・肩・顔点が全高の
            # 約9%下へ一斉にズレる (肩が腰の位置)。2026-07-16の頭身
            # レンジ開放でクランプ下限は0.125へ開放済みだったが走査帯の
            # 下限0.30が置き去りだった。チビの挙動を1bitも変えないため
            # 第1走査が不発のときだけ [max(0.125, 想定/1.8), 30%) を
            # 追加走査する (信頼窓は同一適用) ----
            lo2 = int(max(0.125, hf_knob / 1.8) * ch_img)
            if lo2 < lo:
                pinch = _trusted(_scan_pinch(lo2, lo))
        # ---- 目ペアによる頭身の逆算・検証 (2026-07-18「頭身の算出が
        # 甘い」対策、同夜の26体診断で「目は検出済みなのにフィット未反映」
        # が13体=最多故障だったため一次情報へ昇格)。
        # ①逆算: 目の画面行 = hf*(0.5+(0.15+0.75tanE)*fr_rel)*ch を
        #   hfについて解く (face_rはw_ear実測と相互依存→3回反復)
        # ②ベト: くびれ実測が目由来の1.5倍超=「顔より下の偽くびれ」
        #   (ウルファール腰下0.581 vs 目0.26) として棄却
        # ③選択: くびれあり=目解が許容帯 (0.70..1.60倍) 内なら目解で
        #   微修正、帯外なら くびれ維持 / くびれ無し=目解のみで組む
        # SM_POSE_EYE_FIT=off で無効 ----
        _eye_row = None
        hf_solved = -1.0
        if (os.environ.get("SM_POSE_EYE_FIT", "on").strip().lower()
                not in ("off", "0", "false", "no")):
            _eye_row = _detect_eye_pair(
                ref_image,
                hf_hint=(pinch / float(ch_img) if pinch > 0 else None))
        if _eye_row is not None:
            _tanE = math.tan(math.radians(ELEV_DEG))
            _kfr = 0.15 + 0.75 * _tanE
            _cosE = math.cos(math.radians(ELEV_DEG))
            fr_rel = 0.35
            hf_c = 0.0
            for _ in range(3):
                hf_c = max(0.125, min(
                    0.60, (_eye_row / float(ch_img))
                    / (0.5 + _kfr * fr_rel)))
                r_row = int(hf_c / 2.0 * ch_img)      # head_cy の画像行
                m = max(2, ch_img // 50)
                seg = smooth[max(0, r_row - m):
                             min(len(smooth), r_row + m + 1)]
                w_px = float(np.median(seg)) if len(seg) else 0.0
                if w_px <= 0:
                    break
                fr_w = min(HEAD_WORLD / 2.0,
                           0.45 * w_px * (HEAD_WORLD / hf_c) / ch_img
                           * _cosE)
                fr_rel = fr_w / HEAD_WORLD
            hf_solved = hf_c
            if pinch > 0 and pinch / float(ch_img) \
                    > _EYE_VETO_RATIO * hf_solved:
                print(f"[pose] 頭身: くびれ実測{pinch / ch_img:.2f}は"
                      f"目由来{hf_solved:.2f}の{_EYE_VETO_RATIO}倍超"
                      "=顔より下の偽くびれとして棄却")
                pinch = -1
        # ★第2周回の見直し (2026-07-18夜、ユーザーラベルv2で退行4体):
        # 目→頭身の換算は「目の深さ=頭高の57%」仮定が画風依存で
        # (さぶり=目が高い巨頭0.477→0.375、真ロップ長身0.242→0.191と
        # 正しいくびれ解を上書きした)、帯内上書きは撤回。頭身はくびれ
        # 優先・目解はくびれ不在/ベト時のみ。目の行の厳密合わせは
        # 換算不要の顔クラスタスナップ (face_dy) が担う
        if pinch <= 0 and hf_solved > 0:
            print(f"[pose] 頭身: 目の位置から実測 head_frac="
                  f"{hf_solved:.3f} (くびれ検出不能/棄却。"
                  "SM_POSE_EYE_FIT=offで従来)")
            fig = Figure(head_frac=hf_solved)
        if pinch > 0:
            # 上限0.60: ロングヘア+ワンピース等でくびれを深く誤検出しても
            # 脚に最低4割を確保する保険 (2026-07-13 AI妹実障害: hf0.71で
            # 脚が圧縮され歩幅消失。ロップ級の巨頭も0.60で封じ込め可を
            # 実データのオーバーレイで確認済み)。下限は0.30→0.155へ開放
            # (2026-07-16頭身レンジ0〜3: 6頭身級の立ち絵の実測に追従する)
            hf = max(0.125, min(0.60, pinch / float(ch_img)))
            fig = Figure(head_frac=hf)

        # (肩立ち上がり行の実測アンカーは2026-07-18ラベルv3で撤回:
        #  1.18倍・3行連続の未較正しきいが髪カーテンの終端等を肩と誤認し
        #  「全体的に肩の検出精度が落ちた」— 目逆算上書きと同じ教訓で、
        #  コーパス較正なしの新ヒューリスティックを既定onにしない)

        # ---- Codexビジョン実測ランドマーク (2026-07-18方針転換、最優先):
        # measure_landmarks.py が <round>/01_generation/landmarks.json へ
        # 瞳y・顎y・肩y を保存する。存在すれば頭身=顎行・肩=実測行・
        # 目snap先=瞳行の直接アンカーで、上の暗ブロブ系推定を上書きする
        # (前髪・メガネ・ヘルメット・マスクに意味論で頑健。暗ブロブ系は
        # ランドマーク未測定ラウンドのフォールバックへ降格) ----
        _lm_eye = None
        try:
            _fn = getattr(ref_image, "filename", None)
            if _fn and os.environ.get("SM_POSE_LANDMARKS", "on") \
                    .strip().lower() not in ("off", "0", "false", "no"):
                _lp = Path(_fn).resolve().parent.parent / "landmarks.json"
                if _lp.is_file():
                    _lmj = json.loads(
                        _lp.read_text(encoding="utf-8"))["front"]
                    _chin_row = float(_lmj["chin_y"]) - y0
                    _sh_row = float(_lmj["shoulder_y"]) - y0
                    _lm_eye = float(_lmj["pupil_y"]) - y0
                    hf_lm = max(0.125, min(0.60,
                                           _chin_row / float(ch_img)))
                    fig = Figure(head_frac=hf_lm)
                    _shy = (1.0 - _sh_row / float(ch_img)) * fig.total
                    fig.shoulder_y = max(
                        fig.hip_y + 0.55 * (fig.chin - fig.hip_y),
                        min(fig.chin - 0.005, _shy))
                    # 鼻行 = 瞳と顎の実測の内分 (55%)。検収r5cで「鼻点が
                    # 顎に落ちる」軽度残存 (さぶり=目が高い巨頭で固定
                    # オフセット-0.46rが過大) の対策
                    _lm_nose_row = _lm_eye + 0.55 * (_chin_row - _lm_eye)
                    print(f"[pose] ランドマーク実測: 顎行→head_frac="
                          f"{hf_lm:.3f}・肩行・瞳行・鼻行を直接アンカー "
                          "(SM_POSE_LANDMARKS=offで従来)")
        except Exception:                 # noqa: BLE001
            _lm_eye = None                # 破損jsonは無視して従来経路

        # ---- 各部位の高さ(世界y)を画像行へ写して実測幅を取る。
        #      骨格スケールは全高で合わせるため「世界yの全高比 = 画像bbox
        #      内の高さ比」(俯角の縦縮みは分子分母で相殺)。帯は±2%中央値 ----
        def width_at(y_world: float) -> float:
            r = int(round(ch_img - y_world / fig.total * ch_img))
            m = max(2, ch_img // 50)
            seg = smooth[max(0, r - m): min(len(smooth), r + m + 1)]
            return float(np.median(seg)) if len(seg) else 0.0

        # 世界単位への換算: 全高(世界) fig.total = ch_img px。幅は俯角補正
        # (scale=ch/(total·cosE)) で 1/cosE 倍に描かれるため、実測幅を
        # cosE で先に縮めて相殺する (これが無いと収まり係数0.32/0.45の
        # 余白を3.5%食う。2026-07-12レビュー指摘)
        px2w = fig.total / float(ch_img) \
            * math.cos(math.radians(ELEV_DEG))
        # ---- 肩幅: 顎下バンド (肩行〜下へ全高15%) の最大幅で測る。
        # 肩行そのもの (顎直下) は首のくびれ=シルエット最狭部に当たり、
        # 0.32*首幅では肩が胴の中へ埋まって腕鎖ごと体にめり込む
        # (2026-07-17 C43実害: 腕が体めり込み+前後感ゼロの振り子腕。
        # 実測 首行0.289 vs 肩バンド0.416)。バンド最大=肩・上胴の張り出し。
        # 旧実装は既定0.1625からの縮小専用だったが、幅広チビは既定より
        # 広い肩が正解のため拡幅も許す。SM_POSE_SHOULDER_FIT=off で
        # 旧挙動 (肩行×縮小のみ) へ戻せる ----
        _sh_band = (os.environ.get("SM_POSE_SHOULDER_FIT", "on")
                    .strip().lower() not in ("off", "0", "false", "no"))
        w_sh = width_at(fig.shoulder_y) * px2w
        if _sh_band:
            r_hi = int(round(ch_img - fig.shoulder_y / fig.total * ch_img))
            r_lo = int(round(ch_img - (fig.shoulder_y - 0.15 * fig.total)
                             / fig.total * ch_img))
            seg = smooth[max(0, min(r_hi, r_lo)):
                         min(len(smooth), max(r_hi, r_lo) + 1)]
            if len(seg):
                w_sh = float(seg.max()) * px2w
        w_hip = width_at(fig.hip_y) * px2w
        w_ear = width_at(fig.head_cy) * px2w      # 耳の行の実測幅
        if w_sh > 0:
            if _sh_band:
                new_sh = max(0.05 * fig.total, min(0.26, 0.32 * w_sh))
                # 腕鎖ガード: 直立の肘・手首 (arm_out≈11°で外へ開く) が
                # その行のシルエットからはみ出さない肩幅まで戻す (骨が
                # 輪郭外に出るとVACEがそこに腕を描く 2026-07-12実害)
                _sin_ao = math.sin(math.radians(11.0))
                for _ln, _off in ((fig.arm_upper, _sin_ao * fig.arm_upper),
                                  (fig.arm_upper + fig.arm_lower,
                                   _sin_ao * (fig.arm_upper
                                              + fig.arm_lower))):
                    _wy = fig.shoulder_y - 0.01 - _ln
                    _half = width_at(_wy) * px2w / 2.0
                    if _half > 0:
                        new_sh = min(new_sh,
                                     max(0.05 * fig.total,
                                         _half - fig.arm_r - _off))
            else:
                new_sh = max(0.05 * fig.total, min(fig.sh_x, 0.32 * w_sh))
            fig.sh_x = new_sh
            # ---- 腕の長さは短縮しない (2026-07-15ユーザー報告「腕がすごく
            # 短い」)。旧実装は肩幅の縮小率と同率で腕も短縮 (下限50%) —
            # 幅の数字で長さを決めるため、チビキャラで実際の腕の約半分に
            # なり、VACEが骨どおりの短い腕を描いていた。Figure の腕は体高比
            # (0.313b) で頭身実測 (head_frac) に既に追従しており、手首は
            # 手先≒拳の位置に着地する (C01/C10/C30/C42/C50 の立ち絵
            # オーバーレイで実証済み)。立ち絵から手先を直接実測する案
            # (腕|胴|腕の3ラン行) は試作の実測で 5体中4体不発 (腕が胴に
            # 密着)・1体誤発火 (袖とドレスの隙間を手先と誤認して逆に短縮)
            # だったため不採用。SM_POSE_ARM_FIT=off で旧挙動へ戻せる。
            if str(os.environ.get("SM_POSE_ARM_FIT", "on")
                   ).strip().lower() in ("off", "0", "false", "no"):
                ratio = fig.sh_x / 0.1625
                fig.arm_upper *= max(0.5, ratio)
                fig.arm_lower *= max(0.5, ratio)
        if w_hip > 0:
            fig.hip_x = max(0.03 * fig.total, min(fig.hip_x, 0.30 * w_hip))
        if w_ear > 0:
            # face点の広がり専用 (頭の縦寸は Figure の絶対系のまま)。
            # 耳=±0.92face_r が耳の行の輪郭 (±0.5w_ear) の内側に入る係数
            fig.face_r = max(0.08 * fig.total,
                             min(fig.head_r, 0.45 * w_ear))
        # ---- 顔クラスタの残差スナップ (2026-07-18夜): hfのクランプ
        # (0.60上限=超チビ) や許容帯ではみ出た誤差を、顔点クラスタ
        # (鼻目耳) ごと縦シフトして目の行を実測へ厳密一致させる。
        # 上限 _EYE_DY_MAX*head_r — 骨格の頭球から顔がはみ出ない範囲 ----
        if _lm_eye is not None:
            _eye_row = _lm_eye           # ランドマーク実測が最優先
        if _eye_row is not None:
            # 深さ妥当性ゲート (2026-07-18ラベルv3「顎を鼻と捉える・
            # 口隠し/ヘルメットの目ズレ」対策): 検出行が採用頭身に対して
            # 目としてあり得る深さ (頭高の40..80%) のときだけスナップ。
            # マスクの口元やヘルメット影の偽ペアが顔クラスタごと下へ
            # 引きずる連鎖を遮断する。ランドマーク実測はゲート免除 (信頼)
            _depth = _eye_row / max(1.0, fig.head_frac * ch_img)
            if _lm_eye is not None or 0.40 <= _depth <= 0.80:
                frw = getattr(fig, "face_r", fig.head_r)
                _tanE = math.tan(math.radians(ELEV_DEG))
                row_pred = ch_img * (1.0 - (fig.head_cy - 0.15 * frw
                                            - 0.75 * frw * _tanE)
                                     / fig.total)
                y_shift = -(_eye_row - row_pred) / float(ch_img) * fig.total
                lim = _EYE_DY_MAX * fig.head_r
                dy = max(-lim, min(lim, y_shift))
                if abs(dy) > 0.01 * fig.total:
                    fig.face_dy = dy
        # 鼻の個別アンカー (ランドマークがある場合のみ): クラスタ一括
        # スナップ後の鼻残差を実測鼻行へ合わせる
        if _lm_eye is not None and "_lm_nose_row" in locals():
            frw2 = getattr(fig, "face_r", fig.head_r)
            _tanE2 = math.tan(math.radians(ELEV_DEG))
            _fd = getattr(fig, "face_dy", 0.0)
            nose_pred = ch_img * (1.0 - (fig.head_cy - 0.46 * frw2 + _fd
                                         - 0.78 * frw2 * _tanE2)
                                  / fig.total)
            nshift = -(_lm_nose_row - nose_pred) / float(ch_img) * fig.total
            nlim = 0.30 * fig.head_r
            ndy = max(-nlim, min(nlim, nshift))
            if abs(ndy) > 0.008 * fig.total:
                fig.nose_dy = ndy
        # ---- スタンス実測 → 外転 leg_out (2026-07-20 骨合わせ第1弾) ----
        # 足開きの立ち絵 (神爺さん実例) に足閉じ標準骨格を当てると、
        # 静止窓はfree_idleで守れても歩行窓に姿勢矛盾が残る。足元バンドの
        # 足クラスタ間隔から立ち姿の開きを実測し、脚鎖全体の外転角へ写す。
        # スカート等で足が1塊のとき・足が写らないときは不発=既定0度
        # (安全側)。SM_POSE_STANCE_FIT=off で無効化。
        if os.environ.get("SM_POSE_STANCE_FIT", "").strip().lower()                 not in ("off", "0", "false"):
            try:
                band = fg[max(y0, y1 - max(2, int(0.10 * ch_img))):y1 + 1]
                cols = band.sum(axis=0)
                on = cols > max(1, int(0.25 * band.shape[0]))
                runs, st = [], None
                for x, v_ in enumerate(on.tolist() + [False]):
                    if v_ and st is None:
                        st = x
                    elif not v_ and st is not None:
                        runs.append((st, x - 1))
                        st = None
                runs = [r for r in runs if r[1] - r[0] >= 2]
                if len(runs) >= 2:
                    c1 = (runs[0][0] + runs[0][1]) / 2.0
                    c2 = (runs[-1][0] + runs[-1][1]) / 2.0
                    half_w = (c2 - c1) / 2.0 / ch_img * fig.total
                    leg_len = fig.leg_upper + fig.leg_lower
                    x_off = half_w - fig.hip_x
                    if x_off > 0 and leg_len > 0:
                        out_deg = math.degrees(
                            math.asin(min(0.42, x_off / leg_len)))
                        if out_deg >= 3.0:
                            fig.leg_out = min(22.0, out_deg)
                            print(f"[pose] スタンス実測: 足間隔"
                                  f"{c2 - c1:.0f}px → 外転"
                                  f"{fig.leg_out:.1f}°")
            except Exception:                     # noqa: BLE001
                pass
        # ---- 腕の実測フィット (2026-07-20 骨合わせ第2弾: 腕) ----
        # 手首バンドの外縁半幅から arm_out の必要角を逆算し、既定より
        # 外へ張るぶんだけ加算する (袖広ローブ等)。内側方向は既存の
        # 「腕鎖がシルエットからはみ出さないガード」の持ち場なので触らず、
        # 外側のみ。腕上げ・持ち物ポーズの誤検知はクランプ+閾値で減衰。
        # SM_POSE_ARM_OUT_FIT=off で無効化。
        if os.environ.get("SM_POSE_ARM_OUT_FIT", "").strip().lower()                 not in ("off", "0", "false"):
            try:
                wrist_y = (fig.shoulder_y - 0.01
                           - fig.arm_upper - fig.arm_lower)
                wr = int(round(ch_img * (1.0 - wrist_y / fig.total)))
                half_band = max(2, int(0.05 * ch_img))
                b0 = max(0, wr - half_band)
                b1 = min(ch_img, wr + half_band)
                band = fg[y0 + b0:y0 + b1]
                cols = np.where(band.any(axis=0))[0]
                if len(cols) > 4:
                    half_w = (cols[-1] - cols[0]) / 2.0 / ch_img * fig.total
                    target_x = half_w - fig.arm_r * 1.6
                    base_ao = 11.0            # IDLE_POSEの既定arm_out
                    cur_x = (fig.sh_x + math.sin(math.radians(base_ao))
                             * fig.arm_upper)
                    if target_x > cur_x and fig.arm_upper > 0:
                        s = min(0.85, (target_x - fig.sh_x)
                                / fig.arm_upper)
                        add = math.degrees(math.asin(max(0.0, s))) - base_ao
                        if add >= 3.0:
                            fig.arm_out_add = min(25.0, add)
                            print(f"[pose] 腕実測: 手首バンド半幅→"
                                  f"arm_out +{fig.arm_out_add:.1f}°")
            except Exception:                     # noqa: BLE001
                pass
        return fig
    except Exception:
        return fig


def _face_fwd_scale(ref_image, fig: Figure, yaw_head: float,
                    scale: float, cx: float, base_y: float,
                    width: int, height: int) -> float:
    """顔点の前方突出(z)倍率を参照シルエットの前縁から実測する (0.25..1.0)。

    鼻キーポイントの前方突出は 0.78*face_r だが、face_r は頭中心行の
    実測幅45% (髪を含む) 由来のため、ツインテール・ロングウェーブ等で
    髪が横に広いキャラでは実際の顔幅を大きく超えて膨らむ。すると横・
    斜めで鼻が顔の前縁より前に写り、首→鼻リンクが前下がりの長い斜め
    バー=「頭を前に突き出した歩き」(スマホ首) を宣言してしまう
    (2026-07-17 ろーらちゃん実害: z=0.78の全体較正だけでは髪幅キャラを
    救えない)。その方向の立ち絵で「鼻の高さの行帯」の前景前縁を実測し、
    投影上の鼻先がそこへ収まる倍率を返す。倍率は _keypoints が顔点全部の
    zに掛ける (鼻・目の奥行きが同率で縮み顔の凸形状は保たれる)。
    正面・背面 (|sin(yaw_head)|<0.35) は突出が画面に出ないので1.0。
    計測不能・前縁の方が遠い(=絵の顔が骨格より前に出ている)場合も1.0
    (縮める方向にしか働かない=旧挙動より悪化しない)。
    SM_POSE_NOSE_CLAMP=off で無効。"""
    try:
        if (os.environ.get("SM_POSE_NOSE_CLAMP", "on").strip().lower()
                in ("off", "0", "false", "no")):
            return 1.0
        s = math.sin(math.radians(float(yaw_head)))
        if abs(s) < 0.35:
            return 1.0
        if not hasattr(ref_image, "convert"):
            ref_image = Image.open(ref_image)
        off = _FACE_PTS[0][0]                     # 鼻オフセット (0,-0.46,0.78)
        r = getattr(fig, "face_r", fig.head_r)
        p = (off[0] * r, fig.head_cy + off[1] * r, off[2] * r)
        nx, ny = _project(p, yaw_head, scale, cx, base_y, total=fig.total)
        fwd = nx - cx                             # 投影上の前方突出 (符号つき)
        if abs(fwd) < 1.0:
            return 1.0
        # 参照画像 -> キャンバスのレターボックス写像 (_char_box と同一式)
        im = ref_image.convert("RGB")
        iw, ih = im.size
        sc = min(width / iw, height / ih)
        ox, oy = (width - iw * sc) / 2.0, (height - ih * sc) / 2.0
        fg = _fg_mask(im)
        row = int(round((ny - oy) / sc))
        band = max(2, int(0.02 * ih))
        seg = fg[max(0, row - band): row + band + 1]
        if seg.size == 0 or not seg.any():
            return 1.0
        # 帯内の全行の合併シルエット端 = 最も前へ出ている画素 (顎・襟が
        # 混ざっても「より前の縁」を採る=控えめにしかクランプしない)
        cols = np.where(seg.any(axis=0))[0]
        edge_col = float(cols.max()) if fwd > 0 else float(cols.min())
        edge_fwd = (ox + (edge_col + 0.5) * sc) - cx
        if fwd > 0 and 0 < edge_fwd < fwd:
            return max(0.25, edge_fwd / fwd)
        if fwd < 0 and fwd < edge_fwd < 0:
            return max(0.25, edge_fwd / fwd)
        return 1.0
    except Exception:
        return 1.0


def _skirt_hem_frac(ref_image, side: bool = False) -> float | None:
    """ロングスカートの裾 (bbox内の高さ比率 0..1) を実測する。

    骨格がスカートの中に膝・足首を宣言すると、VACEは「そこに脚が見える」
    契約を果たそうとしてスカートを透過させたりスリットを発明する
    (2026-07-17ユーザー報告)。実写のOpenPose教師データではロングスカートの
    膝・足首は遮蔽=欠損なので、裾より上の膝・足首キーポイントを
    フレームごとに欠損化するのが語彙内の対策 — その裾を実測する。

    スカート行の判定 (65体×3方向の全数較正+目視3周 2026-07-17):
      ①単一ラン かつ 幅≥0.80×腰行幅 (横向きは0.85×。チビの密着脚+
        ブーツ束は0.5〜0.77×にしかならない。0.55×では C01生足半ズボン/
        C34ズボンを誤検出した)
      ②裾<0.80 (膝丈・チュニック・ズボン) は対象外
      ③裾下の段差検査 (下の関数末尾): 同色ズボンの脚柱を弾く
    ※初版は「色の連続性」も条件にしていたが、白トリム・ベルトつき
    ドレスで裾を実際より上に誤測し、その帯に足首ドットが残って
    「スカートの布の上に靴が実体化」した (2026-07-17 1640実害) — 色は
    廃止し幾何+段差のみに統一 (横向きの床丈ドレスも拾えるようになる)。
    腰下(0.58)から連続ブロックを下へ辿り、途切れたら終了。
    SM_POSE_SKIRT_OCCLUDE=off で無効。"""
    try:
        if (os.environ.get("SM_POSE_SKIRT_OCCLUDE", "on").strip().lower()
                in ("off", "0", "false", "no")):
            return None
        im = ref_image.convert("RGB")
        fg = _fg_mask(im)
        ys, xs = np.where(fg)
        if len(ys) < 100:
            return None
        y0, y1 = int(ys.min()), int(ys.max())
        ch = y1 - y0 + 1
        cw = int(xs.max()) - int(xs.min()) + 1

        def runs_at(r):
            band = fg[max(y0, r - 1): min(y1 + 1, r + 2)]
            cols = np.where(band.any(axis=0))[0]
            if len(cols) == 0:
                return []
            rr = np.split(cols, np.where(np.diff(cols) > 2)[0] + 1)
            return [x for x in rr if (x[-1] - x[0] + 1) >= 0.04 * cw]

        def width_of(rr):
            return sum(int(x[-1] - x[0] + 1) for x in rr)

        hip_w = width_of(runs_at(y0 + int(0.52 * ch)))
        if hip_w <= 0:
            return None
        need = (0.85 if side else 0.80) * hip_w
        hem = None
        miss = 0
        for r in range(y0 + int(0.58 * ch), y1 + 1):
            rr = runs_at(r)
            ok = (len(rr) == 1
                  and (rr[0][-1] - rr[0][0] + 1) >= need)
            if ok:
                hem = r
                miss = 0
            else:
                miss += 1
                if miss >= max(3, int(0.02 * ch)):
                    break
        if hem is None or (hem - y0) / ch < 0.80:
            return None
        # 裾下の段差検査: 本物の裾は下で幅が細る (足・靴だけになる)。
        # 同色ズボンの脚柱はブーツまで同じ太さが続く (るか・C19・勇者back
        # の実測誤検出 2026-07-17 目視較正)。床丈 (下4%以内) は免除
        if hem < y1 - max(3, int(0.04 * ch)):
            below = [width_of(runs_at(r))
                     for r in range(hem + 2, y1 + 1, 2)]
            below = [w for w in below if w > 0]
            hem_w = width_of(runs_at(hem))
            if below and hem_w > 0 and \
                    float(np.median(below)) > 0.72 * hem_w:
                return None
        return (hem - y0) / float(ch)
    except Exception:
        return None


def skirt_hem_y(ref_image, width: int, height: int, side: bool = False,
                frac: float | None = None) -> float | None:
    """裾のy (キャンバス座標、_char_boxと同じレターボックス写像) を返す。

    frac指定時は検出をスキップし、その比率 (bbox内 0..1) をこの参照の
    bboxへ写像するだけ — スカートはキャラの属性なので、垂れ袖などで
    自方向の検出が立たない方向にも front/back の実測比率を適用する
    「方向間コンセンサス」用 (2026-07-17 1640実害: front/backだけ検出され
    横・斜め6方向が遮蔽なし=全方向に布の上の靴)。"""
    try:
        if frac is None:
            frac = _skirt_hem_frac(ref_image, side=side)
        if frac is None:
            return None
        im = ref_image.convert("RGB")
        fg = _fg_mask(im)
        ys, _xs = np.where(fg)
        if len(ys) < 100:
            return None
        y0, y1 = int(ys.min()), int(ys.max())
        hem = y0 + frac * (y1 - y0 + 1)
        # 参照画像 -> キャンバスのレターボックス写像 (_char_box と同一式)
        iw, ih = im.size
        sc = min(width / iw, height / ih)
        oy = (height - ih * sc) / 2.0
        return oy + (hem + 0.5) * sc
    except Exception:
        return None


def _yaw_adapt_on() -> bool:
    """ヨー追従のノブ (SM_POSE_YAW_ADAPT、既定on)。"""
    return os.environ.get("SM_POSE_YAW_ADAPT", "on").strip().lower() \
        not in ("off", "0", "false", "no")


# 顔正面化の発動しきい (実測ヨーの絶対値がこれ未満=「正面寄り立ち絵」)。
# 真ロップ-13°/+11°・ハム猫-10°が対象クラス、AI妹-42°(真の45°絵)は対象外。
# compass_vace の浅ヨー警告と同じ値
FACE_FRONT_THR = 20.0


def _face_front_mode() -> str:
    """斜め前の顔正面化ノブ (SM_POSE_FACE_FRONT、既定auto)。

    2026-07-18ユーザー発案「顔だけ正面向かせてみます?」: 立ち絵が正面寄り
    (Codexの斜め45°の壁) のキャラは、ヨー追従で顔を実測13°に寄せても
    「ほぼ正面なのに片耳」の弱い中間語彙にしかならず、歩行開始と同時に
    後頭部へ吸われた (真ロップ8+2試行全滅)。いっそ完全正面 (0°) で宣言
    すれば、立ち絵との綱引きゼロのまま、鼻+両目+両耳の対称パターン=
    「後頭部」と最も相容れない最強の顔語彙になる。
    auto=実測ヨーが浅い (|est|<FACE_FRONT_THR) キャラのみ発動 /
    on=斜め前は常に正面宣言 / off=従来 (ヨー追従のみ)。"""
    v = os.environ.get("SM_POSE_FACE_FRONT", "auto").strip().lower()
    if v in ("on", "1", "true", "yes"):
        return "on"
    if v in ("off", "0", "false", "no"):
        return "off"
    return "auto"


def _estimate_yaw(ref_image, fig: Figure):
    """立ち絵の実ヨー(度)を目の重心横オフセットから推定する。

    返り値 (yaw_deg, debug文字列) / 推定不能は (None, 理由)。
    原理: 骨格モデルの目は (±0.47, -0.15, 0.75)·face_r — 両目の平均は
    頭中心から z=0.75·face_r 前方なので、ヨーθで横に 0.75·face_r·sinθ
    ずれて写る。目=「目バンド内の明るい顔ラン(髪・耳の暗い両脇を除外した
    最長明カラム帯)の中の暗画素」の重心。単色キャラ(輝度コントラスト
    無し)・目が見えない向き・非人型は各ゲートでNoneに落ちる。
    較正 (2026-07-16、量産済み61体×8方向の実測): 清書キャラは
    front≈0°/斜め≈∓35〜50°の教科書パターン、真ロップの斜め前は
    -14°/+17°=「正面寄り立ち絵」の実走診断と一致、目視3体照合で
    振幅も正確 (AI妹-42=真の45°斜め、ハム猫-10=本当に浅い顔)。"""
    try:
        a = np.asarray(ref_image.convert("RGB")).astype(int)
        fg = _fg_mask(ref_image)
        ys, xs = np.where(fg)
        if len(ys) == 0:
            return None, "no_fg"
        y0 = int(ys.min())
        ch_img = int(ys.max()) - y0 + 1
        # 頭の高さ: 頭はワールド上端の 2·head_r/total (HEAD_WORLD固定系)
        head_h = ch_img * (2.0 * fig.head_r / fig.total)
        # 目バンド: チビ較正で目=頭中心の少し下 (-0.15r) → 頭高の42..78%
        b0 = int(y0 + 0.42 * head_h)
        b1 = int(y0 + 0.78 * head_h)
        band_fg = fg[b0:b1 + 1]
        band_rgb = a[b0:b1 + 1]
        if band_fg.sum() < 30:
            return None, "band_empty"
        lum = (0.30 * band_rgb[..., 0] + 0.59 * band_rgb[..., 1]
               + 0.11 * band_rgb[..., 2])
        col_ok = band_fg.sum(axis=0) >= max(2, int(0.15 * (b1 - b0 + 1)))
        if not col_ok.any():
            return None, "no_cols"
        col_lum = np.zeros(band_fg.shape[1])
        for x in range(band_fg.shape[1]):
            if col_ok[x] and band_fg[:, x].any():
                col_lum[x] = float(np.median(lum[:, x][band_fg[:, x]]))
        lo, hi = np.percentile(col_lum[col_ok], [15, 85])
        if hi - lo < 12:                       # 単色キャラ (毛玉等)
            return None, "flat_lum"
        bright = col_lum >= (lo + 0.55 * (hi - lo))
        best_l = best_r = -1
        cur = None
        for x in range(len(bright) + 1):
            v = bool(bright[x]) if x < len(bright) else False
            if v and cur is None:
                cur = x
            elif not v and cur is not None:
                if best_l < 0 or (x - cur) > (best_r - best_l):
                    best_l, best_r = cur, x
                cur = None
        if best_l < 0 or (best_r - best_l) < 6:
            return None, "no_face_run"
        flum = lum[:, best_l:best_r]
        ffg = band_fg[:, best_l:best_r]
        if not ffg.any():
            return None, "face_empty"
        skin_med = float(np.median(flum[ffg]))
        eye = ffg & (flum < skin_med - max(25.0, 0.30 * skin_med))
        n_eye = int(eye.sum())
        if n_eye < 8:
            return None, "no_eyes"
        eye_cx = float((np.where(eye)[1] + best_l).mean())
        h1 = int(y0 + head_h)
        hxs = np.where(fg[y0:h1 + 1])[1]
        head_cx = (float(hxs.min()) + float(hxs.max())) / 2.0
        r_px = getattr(fig, "face_r", fig.head_r) / fig.total * ch_img
        s = (eye_cx - head_cx) / (0.75 * r_px)
        if abs(s) > 1.2:
            return None, "off_wild"
        return (math.degrees(math.asin(max(-1.0, min(1.0, s)))),
                f"off={eye_cx - head_cx:+.1f}px eye={n_eye}px")
    except Exception as e:                     # 推定は常にベストエフォート
        return None, f"error:{e}"


# ヨー追従の対象方向 (斜め前のみ。横・後ろは目が測れない)
_YAW_ADAPT_DIRS = ("front_left", "front_right")

# 体ヨー追従の床 (完全に正面へ潰さず「斜め」の弁別を残す最小角)
DIAG_BODY_FLOOR = 20.0


def _diag_body_mode() -> str:
    """斜め前の体ヨー追従ノブ (SM_POSE_DIAG_BODY、既定off)。

    2026-07-18の実測系列: 斜め前(公称45°)の歩行は、奥耳遮蔽→ヨー追従→
    半球分離→顔正面化→σ0.45+VACE6step+透視0.6 の全対策後も
    後頭部(180°)→真横(90°)までしか押し戻せず膠着。正面セルと真横セルは
    毎回完璧 — つまりAniSoraの歩行分布は前/横/後ろの3モードが支配的で、
    「45°を向いて歩く顔つきの人」はほぼ分布外。制御を強めても隣のモードに
    スナップする。auto=浅ヨー立ち絵 (実測|est|<FACE_FRONT_THR) のキャラは
    体骨格ごと実測ヨー (床DIAG_BODY_FLOOR=20°) へ追従させ、勝てる
    「前歩きモード」の盆地内で生成する。立ち絵自体がほぼ正面 (斜め45°の
    壁) なので、シートの絵とも整合する。2026-07-16裁定「体は45度でいい」
    の明示的な見直し (裁定当時は45°が到達可能と思われていた) — 必ず
    ユーザー可視のログを出し、offで従来へ戻せること。"""
    v = os.environ.get("SM_POSE_DIAG_BODY", "off").strip().lower()
    if v in ("auto", "1", "true", "yes", "on"):
        return "auto"
    return "off"


def _adapted_body_yaw(direction: str, ref_image, fig: Figure,
                      front_ref=None, mode: str | None = None,
                      quiet: bool = False) -> float:
    """斜め前セルの体ヨー: 浅ヨー立ち絵なら実測へ追従 (床20°)。

    信頼ゲートは _adapted_yaw と同一 (front自己較正・符号一致・実測不能は
    公称へフォールバック)。発動条件も顔正面化と同じ浅ヨークラス —
    「絵が45°で描けているキャラ」(AI妹級) は公称45°のまま。"""
    nominal = float(DIR_YAW[direction])
    m = mode if mode in ("auto", "off") else _diag_body_mode()
    if (m == "off" or direction not in _YAW_ADAPT_DIRS
            or ref_image is None):
        return nominal
    lm = _landmark_yaw(ref_image, direction)
    if lm is not None:
        if lm >= FACE_FRONT_THR:
            return nominal                 # 真の45°絵は公称のまま
        sign = 1.0 if nominal > 0 else -1.0
        yawv = sign * max(DIAG_BODY_FLOOR, lm)
        if not quiet:
            print(f"[pose] 体ヨー追従(VLM実測): {direction} 体"
                  f"{nominal:+.0f}°→{yawv:+.0f}°")
        return yawv
    gate_src = front_ref if front_ref is not None else ref_image
    g, _ = _estimate_yaw(gate_src, fig)
    if g is None or abs(g) > 12.0:
        return nominal
    est, dbg = _estimate_yaw(ref_image, fig)
    if est is None:
        return nominal
    sign = 1.0 if nominal > 0 else -1.0
    if est * sign < 0 and abs(est) >= 10.0:
        return nominal                          # 符号逆転=不信
    if abs(est) >= FACE_FRONT_THR:
        return nominal                          # 真の45°絵は公称のまま
    yaw = sign * max(DIAG_BODY_FLOOR, abs(est))
    if not quiet:
        print(f"[pose] 体ヨー追従: {direction} 体{nominal:+.0f}°→"
              f"{yaw:+.0f}° (浅ヨー立ち絵 {dbg} — 45°歩行は分布外のため"
              "前歩きモード内で生成。SM_POSE_DIAG_BODY=offで従来)")
    return yaw


def _landmark_yaw(ref_image, direction: str) -> float | None:
    """landmarks.json の斜め前VLM実測ヨー (度・絶対値)。無ければNone。

    2026-07-18 C01_hs08実害: 目重心ヒューリスティックのヨーが境界ぎわで
    左右バラバラ (front_left=-21/front_right=+20) になり顔正面化の誤発動→
    首グルン再発。measure_landmarks.measure_diag のVLM実測を最優先する。"""
    try:
        if os.environ.get("SM_POSE_LANDMARKS", "on").strip().lower() \
                in ("off", "0", "false", "no"):
            return None
        fn = getattr(ref_image, "filename", None)
        if not fn:
            return None
        lp = Path(fn).resolve().parent.parent / "landmarks.json"
        if not lp.is_file():
            return None
        d = json.loads(lp.read_text(encoding="utf-8")).get(direction)
        return float(d["head_yaw_deg"]) if d else None
    except Exception:                     # noqa: BLE001
        return None


def _adapted_yaw(direction: str, ref_image, fig: Figure,
                 front_ref=None, face_front: str | None = None,
                 quiet: bool = False) -> float:
    """頭の描画ヨー: 立ち絵の実測ヨーに寄り添う (斜め前のみ)。
    呼び出し側は戻り値を _keypoints の yaw_head にだけ渡すこと
    (体のヨーは公称のまま=足運びが移動方向を守る)。

    「参照立ち絵が正面寄りなのに骨格が45°固定」の決定的な綱引きが、
    斜め前の引っ張られの真因 (2026-07-16実走: シード違い2回で同一の
    崩れ方=非ランダム)。骨格が立ち絵と同じ角度を指せば綱引き自体が
    消える。安全装置3段:
      ①front自己較正ゲート — frontの実測が|12°|超のキャラは推定器が
        そのキャラで壊れている (眼鏡・帽子・非対称前髪) ので全面不適用
      ②符号一致 — 公称と逆向きの推定は不信でフォールバック
        (|推定|<10°だけは「ほぼ正面の立ち絵」とみなし公称符号で10°)
      ③振幅クランプ [10°, 75°]
    推定不能 (単色・非人型等) は公称へフォールバック。"""
    nominal = float(DIR_YAW[direction])
    if direction not in _YAW_ADAPT_DIRS or ref_image is None:
        return nominal
    ffm = face_front if face_front in ("on", "off", "auto") \
        else _face_front_mode()
    if ffm == "on":
        # 顔正面化(強制): 実測に依らず斜め前の顔を完全正面で宣言
        if not quiet:
            print(f"[pose] 顔正面化(on): {direction} 顔0°で宣言・"
                  "体は公称維持")
        return 0.0
    lm = _landmark_yaw(ref_image, direction)
    if lm is not None:
        sign = 1.0 if nominal > 0 else -1.0
        if ffm == "auto" and lm < FACE_FRONT_THR:
            if not quiet:
                print(f"[pose] 顔正面化(VLM実測): {direction} 実測{lm:.0f}°"
                      f"<{FACE_FRONT_THR:.0f}° → 顔0°で宣言")
            return 0.0
        yawv = sign * max(10.0, min(75.0, lm))
        if not quiet and abs(yawv - nominal) >= 3.0:
            print(f"[pose] ヨー追従(VLM実測): {direction} 顔"
                  f"{nominal:+.0f}°→{yawv:+.0f}°")
        return yawv
    gate_src = front_ref if front_ref is not None else ref_image
    g, _ = _estimate_yaw(gate_src, fig)
    if g is None or abs(g) > 12.0:
        return nominal
    est, dbg = _estimate_yaw(ref_image, fig)
    if est is None:
        return nominal
    sign = 1.0 if nominal > 0 else -1.0
    if est * sign < 0 and abs(est) >= 10.0:
        return nominal                          # 符号逆転=不信
    if ffm == "auto" and abs(est) < FACE_FRONT_THR:
        # 顔正面化(auto): 正面寄り立ち絵は中間ヨーの弱い語彙をやめ、
        # 完全正面 (両耳可視の対称パターン) へ振り切る (詳細は
        # _face_front_mode。従来の10°床の中間宣言は歩行プライアに負けた)
        if not quiet:
            print(f"[pose] 顔正面化(auto): {direction} 立ち絵実測"
                  f"{est:+.0f}°<{FACE_FRONT_THR:.0f}° → 顔0°で宣言・"
                  f"体は公称維持 ({dbg})")
        return 0.0
    yaw = sign * max(10.0, min(75.0, abs(est)))
    if abs(yaw - nominal) >= 3.0:
        print(f"[pose] ヨー追従(頭): {direction} 顔{nominal:+.0f}°→"
              f"{yaw:+.0f}°・体は公称維持 (立ち絵実測 {dbg})")
    return yaw


def build_walk_pose_frames(direction: str, num_frames: int,
                           width: int, height: int,
                           ref_image=None, leg_scale: float | None = None,
                           cycles: float | None = None,
                           adapt: bool | None = None,
                           arms: bool | None = None,
                           arm_swing: float | None = None,
                           leg_swing: float | None = None,
                           bob: float | None = None,
                           leg_cross: float | None = None,
                           fit_ref=None,
                           face68: bool | None = None,
                           yaw_adapt: bool | None = None,
                           face_front: str | None = None,
                           diag_body: str | None = None) -> list:
    """方向 direction の歩行サイクル骨格制御フレーム列 (PIL RGB) を返す。

    ref_image (PILまたはパス) を渡すと、キャラの画面上のbboxに骨格の
    大きさ・足元位置を合わせ、さらに前景プロファイルから頭身・肩幅・
    頭幅を実測して骨格を輪郭の内側に収める (adapt=False か環境変数
    SM_POSE_ADAPT=off で無効)。実測フィットが首のくびれを検出した場合、
    leg_scale (SM_LEG_SCALE) 由来の頭身は上書きされて不使用になる
    (実測が勝つ。非チビ判定のキャラだけノブが効く)。arms=False
    (SM_POSE_ARMS=off) で腕のキーポイントを出さない (OpenPoseは欠損肢=
    オクルージョンとして学習されており、チビキャラの見えない腕を無理に
    契約させない逃げ道)。フレーム配分は walk_layout に従う: 先頭に直立
    (IDLE_POSE) 区間、続いて歩行サイクル (歩行区間の先頭と終端は同位相)、
    末尾に立ち止まり静止 (シートの直立コマはここから採る)。
    cycles 明示時は歩行区間の周回数として使う。"""
    if direction not in DIR_YAW:
        raise ValueError(f"unknown direction: {direction}")
    yaw = DIR_YAW[direction]
    ls = _leg_scale_env() if leg_scale is None else \
        max(0.6, min(4.0, float(leg_scale)))
    _off = ("0", "off", "false")
    if adapt is None:
        adapt = os.environ.get("SM_POSE_ADAPT", "1").strip().lower() \
            not in _off
    if arms is None:
        arms = os.environ.get("SM_POSE_ARMS", "on").strip().lower() \
            not in _off
    if arm_swing is None:
        arm_swing = _f_env("SM_POSE_ARM_SWING", 1.0, 0.5, 3.0)
    if leg_swing is None:
        leg_swing = _f_env("SM_POSE_LEG_SWING", 1.0, 0.3, 2.0)
    if bob is None:
        bob = _f_env("SM_POSE_BOB", 1.0, 0.0, 2.0)
    if leg_cross is None:
        # モデル歩き (前脚の中央寄せ)。詳細は mannequin3d._leg_chain
        leg_cross = _f_env("SM_POSE_LEG_CROSS", 1.0, 0.0, 1.5)
    if face68 is None:
        face68 = _face68_on()
    if yaw_adapt is None:
        yaw_adapt = _yaw_adapt_on()
    fig = Figure(head_frac=head_frac_for_leg_scale(ls))
    n = max(2, int(num_frames))
    idle_n, cyc, _, tail_n = walk_layout(n)
    if cycles:
        cyc = float(cycles)
    if ref_image is not None and not hasattr(ref_image, "convert"):
        ref_image = Image.open(ref_image)
    if ref_image is not None:
        cx, base_y, ch = _char_box(ref_image, width, height)
        if adapt:
            # fit_ref: 体型計測に使う立ち絵 (省略時はref_image)。方向ごとに
            # 計測すると誤り方が方向間で食い違い、同一キャラの骨格解剖が
            # 矛盾する (2026-07-13 AI妹実障害: front/backで別の頭身になり
            # VACEが静止や大型化で妥協) → 呼び出し側はfront立ち絵を全方向
            # 共通のfit_refとして渡すこと
            fr = fit_ref if fit_ref is not None else ref_image
            if not hasattr(fr, "convert"):
                fr = Image.open(fr)
            fig = _fit_figure_to_char(fig, fr)
    else:
        cx, base_y, ch = width / 2.0, height * 0.93, height * 0.86
    yaw_h = yaw
    if yaw_adapt and ref_image is not None:
        # ヨー追従(頭専用): 顔だけ立ち絵の実測ヨーで描く。体は公称ヨー=
        # 歩行方向を維持 (2026-07-16裁定「体は45度でいい。足運びが絵の
        # 角度とあってない」)
        fr2 = fit_ref if fit_ref is not None else ref_image
        if not hasattr(fr2, "convert"):
            fr2 = Image.open(fr2)
        yaw_h = _adapted_yaw(direction, ref_image, fig, front_ref=fr2,
                             face_front=face_front)
        # 体ヨー追従 (浅ヨー立ち絵の斜め前は45°歩行が分布外 — 詳細は
        # _diag_body_mode)。yaw変数はこの後の全消費者 (キーポイント・
        # 透視・遮蔽・裾side判定) に一貫して効く
        yaw = _adapted_body_yaw(direction, ref_image, fig, front_ref=fr2,
                                mode=diag_body)
    # 図の全高 (接地面〜頭頂) を立ち絵の高さに合わせる。俯角で縦は
    # cos(ELEV) に縮んで写るぶんを補正。
    scale = ch / (fig.total * math.cos(math.radians(ELEV_DEG)))
    cos_e = math.cos(math.radians(ELEV_DEG))
    face_fwd = _face_fwd_scale(ref_image, fig, yaw_h, scale, cx, base_y,
                               width, height) if ref_image is not None \
        else 1.0
    # ロングスカートの裾 (裾より上の膝・足首は遮蔽=欠損化。透け・スリット
    # 発明の対策 2026-07-17)。自方向で検出できなければ front (fit_ref) の
    # 比率を継承 — スカートはキャラの属性 (方向間コンセンサス)
    hem_y = None
    if ref_image is not None:
        _hfrac = _skirt_hem_frac(
            ref_image, side=abs(math.sin(math.radians(yaw))) >= 0.7)
        if _hfrac is None and fit_ref is not None:
            _fr3 = fit_ref if hasattr(fit_ref, "convert") \
                else Image.open(fit_ref)
            _hfrac = _skirt_hem_frac(_fr3)
        if _hfrac is not None:
            hem_y = skirt_hem_y(ref_image, width, height, frac=_hfrac)
    ankle_hide = False
    if hem_y is not None:
        # 足首は「裾+3pxより下にいる瞬間だけ実位置で出す」— 接地時だけ
        # 裾の下から足が覗く実写の床丈ドレス歩行と同じ表現。裾は正しい
        # 位置に全方向コンセンサス済みなので、ドットが布に乗ることはない。
        # ※全フレーム欠損 (frac>=0.90) は撤回 (2026-07-17実走: 脚信号ゼロ
        # だと歩行がふわふわ上下浮遊になる)。クランプ方式も撤回済み
        # (固定裾ラインに乗ったドットが横滑りして布の上の靴化)。
        # SM_POSE_SKIRT_FEET=off で常時欠損 (浮遊してもよいから足を
        # 完全に消したい場合の逃げ道)
        ankle_hide = (os.environ.get("SM_POSE_SKIRT_FEET", "auto")
                      .strip().lower() in ("off", "0", "false", "no"))
    frames = []
    m = n - idle_n - tail_n
    for i in range(n):
        if i < idle_n or i >= idle_n + m:
            # 先頭=直立助走 / 末尾=立ち止まり静止 (シートの直立コマ供給源)
            ang = dict(IDLE_POSE)
        else:
            ph = ((i - idle_n) * cyc / max(1, m - 1)) % 1.0
            ang = walk_angles_at(ph, arm_swing=arm_swing,
                                 leg_swing=leg_swing)
        # 接地補正: 低い方の足首を接地線へ -> 体側が沈む上下動が生まれる
        gait = idle_n <= i < idle_n + m   # 直立区間は内転しない
        by = base_y + (_ground_shift(fig, ang, leg_cross, gait)
                       * bob * scale * cos_e)
        kps = _keypoints(fig, ang, yaw, scale, cx, by, leg_cross, gait,
                         yaw_head=yaw_h, face_fwd=face_fwd)
        if not arms:
            for j in (2, 3, 4, 5, 6, 7):      # 肩・肘・手首を欠損扱いに
                kps[j] = None
        if hem_y is not None:
            # 膝: 裾より上は遮蔽 (実写のDWPoseと同じ欠損表現)
            for j in (9, 12):
                if kps[j] is not None and kps[j][1] < hem_y:
                    kps[j] = None
            # 足首: 裾+3pxより下にいる瞬間だけ実位置で出す (接地の足)
            for j in (10, 13):
                if kps[j] is not None and (ankle_hide
                                           or kps[j][1] < hem_y + 3.0):
                    kps[j] = None
        f68 = _face68_pts(fig, yaw_h, scale, cx, by) if face68 else None
        frames.append(draw_openpose(kps, width, height, f68))
    return frames


def build_canvas_pose_frames(dir_refs: dict, num_frames: int,
                             width: int, height: int, layout,
                             cycles: float | None = None,
                             adapt: bool = True,
                             arms: bool = True,
                             arm_swing: float | None = None,
                             leg_swing: float | None = None,
                             bob: float | None = None,
                             leg_cross: float | None = None,
                             mirror_dirs=(),
                             face68: bool | None = None,
                             yaw_adapt: bool | None = None,
                             face_front: str | None = None,
                             diag_body: str | None = None,
                             kps_out: list | None = None) -> list:
    """グリッド(コンパス)配置の骨格制御フレーム列を返す (8方向1発生成用)。

    layout = (cols, rows, [方向 or None, ...]) — canvas_walk.LAYOUT_COMPASS等。
    dir_refs = 方向 -> 立ち絵(PILまたはパス)。各セルはその方向の立ち絵で
    体型フィットし、compass_vace.compose_reference と同一のレターボックス
    写像でセル内に配置する (骨格と参照キャンバスの位置が画素単位で一致)。
    全セル同位相 (シートのコマ位置が方向間で揃う)。None セルは空(黒)。"""
    if arm_swing is None:
        arm_swing = _f_env("SM_POSE_ARM_SWING", 1.0, 0.5, 3.0)
    if leg_swing is None:
        leg_swing = _f_env("SM_POSE_LEG_SWING", 1.0, 0.3, 2.0)
    if bob is None:
        bob = _f_env("SM_POSE_BOB", 1.0, 0.0, 2.0)
    if leg_cross is None:
        # モデル歩き (前脚の中央寄せ)。詳細は mannequin3d._leg_chain
        leg_cross = _f_env("SM_POSE_LEG_CROSS", 1.0, 0.0, 1.5)
    if face68 is None:
        face68 = _face68_on()
    if yaw_adapt is None:
        yaw_adapt = _yaw_adapt_on()
    cols, rows, dirs = layout
    cw, ch = width // cols, height // rows
    cos_e = math.cos(math.radians(ELEV_DEG))
    # 体型フィットはキャラごとに1回 (front優先)。セルごとに計測すると
    # 誤り方が方向間で食い違い、同一キャラの骨格解剖が矛盾してVACEが
    # 「静止」「大型化」で妥協する (2026-07-13 AI妹実障害)
    fig0 = Figure(head_frac=head_frac_for_leg_scale(_leg_scale_env()))
    if adapt:
        fit_src = dir_refs.get("front") or next(iter(dir_refs.values()))
        if not hasattr(fit_src, "convert"):
            fit_src = Image.open(fit_src)
        fig0 = _fit_figure_to_char(fig0, fit_src)
    ycal = None
    if yaw_adapt:
        # ヨー追従の自己較正ゲート用 front 立ち絵 (詳細は_adapted_yaw)
        ycal = dir_refs.get("front")
        if ycal is not None and not hasattr(ycal, "convert"):
            ycal = Image.open(ycal)
    # ロングスカートの方向間コンセンサス: スカートはキャラの属性なので、
    # front/back で裾が実測できたら全方向へ同じ比率を適用する (垂れ袖で
    # 横・斜めの検出が立たず6方向だけ遮蔽なし=布の上の靴、の対策)
    hem_fracs: dict = {}
    for d in dirs:
        if d is None or d not in dir_refs:
            continue
        rr = dir_refs[d]
        if not hasattr(rr, "convert"):
            rr = Image.open(rr)
        yv0 = float(DIR_YAW[MIRROR_NAME.get(d, d)
                            if d in (mirror_dirs or ()) else d])
        f = _skirt_hem_frac(rr,
                            side=abs(math.sin(math.radians(yv0))) >= 0.7)
        if f is not None:
            hem_fracs[d] = f
    hem_cons = None
    if "front" in hem_fracs or "back" in hem_fracs:
        hem_cons = float(np.median(list(hem_fracs.values())))
    cells = []
    for i, d in enumerate(dirs):
        if d is None or d not in dir_refs:
            continue
        ref = dir_refs[d]
        if not hasattr(ref, "convert"):
            ref = Image.open(ref)
        ox, oy = (i % cols) * cw, (i // cols) * ch
        cx, base_y, chh = _char_box(ref, cw, ch)
        scale = chh / (fig0.total * math.cos(math.radians(ELEV_DEG)))
        # ミラー生成セル: 骨格は鏡像方向のヨーで描く (参照は呼び出し側が
        # 反転済み・出力セルは呼び出し側が反転して戻す)
        yaw_d = MIRROR_NAME.get(d, d) if d in (mirror_dirs or ()) else d
        yaw_v = float(DIR_YAW[yaw_d])
        yaw_h = yaw_v
        if yaw_adapt:
            # ヨー追従(頭専用): 顔だけ立ち絵の実測ヨー。体=公称で歩行方向
            yaw_h = _adapted_yaw(yaw_d, ref, fig0, front_ref=ycal,
                                 face_front=face_front)
            # 体ヨー追従 (浅ヨー立ち絵の斜め前は45°歩行が分布外 —
            # 詳細 _diag_body_mode)。yaw_v はセルの全消費者に一貫
            yaw_v = _adapted_body_yaw(yaw_d, ref, fig0, front_ref=ycal,
                                      mode=diag_body)
        # 鼻前縁クランプ: セル座標系で測り、配置はセル内オフセットのまま
        ff = _face_fwd_scale(ref, fig0, yaw_h, scale, cx, base_y, cw, ch)
        # ロングスカートの裾 (自方向の実測 > 方向間コンセンサス)
        _f_use = hem_fracs.get(d, hem_cons)
        hem = (skirt_hem_y(ref, cw, ch, frac=_f_use)
               if _f_use is not None else None)
        hem_abs = (oy + hem) if hem is not None else None
        # 足首は裾下の瞬間だけ実位置 (常時欠損は浮遊化のため撤回)。
        # SM_POSE_SKIRT_FEET=off で常時欠損の逃げ道のみ
        a_hide = False
        if hem_abs is not None:
            a_hide = (os.environ.get("SM_POSE_SKIRT_FEET", "auto")
                      .strip().lower() in ("off", "0", "false", "no"))
        if _gait_mode() == "crawl":
            # クロール骨格は標準写像 (直立身長→bbox高) だと食み出す:
            # z奥行き0.85T×俯瞰投影で前後ビューが縦に膨らみ、巨頭チビは
            # 頭基準スケールで四肢が参照外へ出る (2026-07-21実測)。
            # idleポーズの投影bboxを実測し、参照キャラbboxへレターボックス
            # 収容する縮尺・接地・中心へ補正する
            _kps0 = _keypoints(fig0, dict(IDLE_POSE), yaw_v, scale,
                               0.0, 0.0, 1.0, False, yaw_head=yaw_v)
            _pts0 = [p for p in _kps0 if p is not None]
            _ma = _fg_mask(ref)
            _mys, _mxs = np.nonzero(_ma)
            if len(_pts0) >= 6 and _mxs.size:
                _xs0 = [p[0] for p in _pts0]
                _ys0 = [p[1] for p in _pts0]
                _bw = max(_xs0) - min(_xs0)
                _bh = max(_ys0) - min(_ys0)
                _asp = ((_mxs.max() - _mxs.min() + 1)
                        / max(1, _mys.max() - _mys.min() + 1))
                if _bw > 1 and _bh > 1:
                    _k = min(chh / _bh, (chh * _asp) / _bw) * 0.96
                    cells.append((yaw_v, yaw_h, fig0, scale * _k,
                                  (ox + cx) - (min(_xs0) + max(_xs0))
                                  / 2.0 * _k,
                                  (oy + base_y) - max(_ys0) * _k, ff,
                                  hem_abs, a_hide))
                    continue
        cells.append((yaw_v, yaw_h, fig0, scale, ox + cx, oy + base_y, ff,
                      hem_abs, a_hide))
    n = max(2, int(num_frames))
    idle_n, cyc, _, tail_n = walk_layout(n)
    if cycles:
        cyc = float(cycles)
    m = n - idle_n - tail_n
    frames = []
    for i in range(n):
        if i < idle_n or i >= idle_n + m:
            # 先頭=直立助走 / 末尾=立ち止まり静止 (シートの直立コマ供給源)
            ang = dict(IDLE_POSE)
        else:
            ph = ((i - idle_n) * cyc / max(1, m - 1)) % 1.0
            ang = walk_angles_at(ph, arm_swing=arm_swing,
                                 leg_swing=leg_swing)
        img = Image.new("RGB", (width, height), (0, 0, 0))
        _rec = [] if kps_out is not None else None
        for yaw, yaw_h, fig, scale, cx, by, ff, hem_abs, a_hide in cells:
            # 接地補正 (体型フィットでfigがセル毎に違うため個別に計算)
            gait = idle_n <= i < idle_n + m   # 直立区間は内転しない
            by2 = by + (_ground_shift(fig, ang, leg_cross, gait)
                        * bob * scale * cos_e)
            kps = _keypoints(fig, ang, yaw, scale, cx, by2, leg_cross,
                             gait, yaw_head=yaw_h, face_fwd=ff)
            if not arms:
                for j in (2, 3, 4, 5, 6, 7):
                    kps[j] = None
            if hem_abs is not None:
                # 膝: 裾より上は遮蔽 (透け・スリット発明の対策)
                for j in (9, 12):
                    if kps[j] is not None and kps[j][1] < hem_abs:
                        kps[j] = None
                # 足首: 裾+3pxより下にいる瞬間だけ実位置で出す (接地の足)
                for j in (10, 13):
                    if kps[j] is not None and (a_hide
                                               or kps[j][1]
                                               < hem_abs + 3.0):
                        kps[j] = None
            f68 = _face68_pts(fig, yaw_h, scale, cx, by2) if face68 else None
            if _rec is not None:
                # パペット用の生キーポイント (2026-07-20ユーザー発案
                # 「骨に線画を貼り付けて動かす」): 描画と同じ座標系の
                # BODY_18リストをセル順に記録する
                _rec.append(list(kps))
            _draw_openpose_onto(img, kps, f68)
        if kps_out is not None:
            kps_out.append(_rec)
        frames.append(img)
    return frames


def encode_frames_b64(frames) -> list:
    """PILフレーム列 -> base64 PNG文字列のリスト (videolab extra用)。"""
    out = []
    for f in frames:
        buf = io.BytesIO()
        f.save(buf, format="PNG", optimize=True)
        out.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return out


# ------------------------------------------------------------- 目視検査CLI
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="pose_video_check")
    ap.add_argument("--size", default="464x848", help="WxH")
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--dirs", default="all",
                    help="カンマ区切り or all (既定)")
    ap.add_argument("--ref", default=None, help="立ち絵 (bbox合わせの確認用)")
    ap.add_argument("--leg-scale", type=float, default=None)
    ap.add_argument("--gif", action="store_true", help="方向ごとにGIFも書く")
    ap.add_argument("--no-adapt", action="store_true",
                    help="体型実測フィットを無効化 (旧挙動)")
    ap.add_argument("--arms", default="on", choices=("on", "off"))
    ap.add_argument("--arm-swing", type=float, default=None,
                    help="肩振り角の倍率 (既定1.0)")
    ap.add_argument("--leg-swing", type=float, default=None,
                    help="股・膝振り角の倍率 (既定1.0)")
    ap.add_argument("--bob", type=float, default=None,
                    help="上下動の倍率 (既定1.0、0で無効=旧挙動)")
    ap.add_argument("--leg-cross", type=float, default=None,
                    help="前脚の中央寄せ=モデル歩き (既定1.0、0で振り子)")
    ap.add_argument("--face68", default="off", choices=("on", "off"),
                    help="顔68点の白ドット (既定off=実走で顔歪み)")
    ap.add_argument("--yaw-adapt", default="on", choices=("on", "off"),
                    help="斜め前のヨー追従 (既定on)")
    ap.add_argument("--face-front", default="auto",
                    choices=("auto", "on", "off"),
                    help="斜め前の顔正面化 (auto=浅ヨー立ち絵のみ0°宣言)")
    ap.add_argument("--diag-body", default="off",
                    choices=("auto", "off"),
                    help="斜め前の体ヨー追従 (auto=浅ヨー立ち絵は体も"
                         "実測ヨー・床20°。45°歩行の分布外対策)")
    ap.add_argument("--overlay", action="store_true",
                    help="立ち絵の上に骨格を半透明合成した検証画像も書く")
    a = ap.parse_args()
    w, h = (int(x) for x in a.size.lower().split("x"))
    dirs = list(DIR_YAW) if a.dirs == "all" else a.dirs.split(",")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ref = Image.open(a.ref) if a.ref else None
    strip_rows = []
    for d in dirs:
        frames = build_walk_pose_frames(d, a.frames, w, h, ref_image=ref,
                                        leg_scale=a.leg_scale,
                                        adapt=not a.no_adapt,
                                        arms=(a.arms != "off"),
                                        arm_swing=a.arm_swing,
                                        leg_swing=a.leg_swing, bob=a.bob,
                                        leg_cross=a.leg_cross,
                                        face68=(a.face68 == "on"),
                                        yaw_adapt=(a.yaw_adapt != "off"),
                                        face_front=a.face_front,
                                        diag_body=a.diag_body)
        frames[0].save(out / f"{d}_f000.png")
        if a.overlay and ref is not None:
            # 立ち絵をキャンバスへレターボックスし、骨格を加算合成。
            # マゼンタ地(255,0,255)はR/Bが飽和済みで、赤/青系の骨格色が
            # 加算で丸ごと消える (=輪郭の外に出た点こそ見たいのに背景上
            # では不可視になる罠。2026-07-12レビュー指摘) → 前景以外を
            # 暗色に落としてから合成する
            iw, ih = ref.size
            sc = min(w / iw, h / ih)
            bg = Image.new("RGB", (w, h), (24, 24, 24))
            rs = ref.convert("RGB").resize(
                (max(1, round(iw * sc)), max(1, round(ih * sc))),
                Image.LANCZOS)
            arr = np.asarray(rs).copy()
            arr[~_fg_mask(rs)] = (24, 24, 24)
            bg.paste(Image.fromarray(arr),
                     ((w - rs.width) // 2, (h - rs.height) // 2))
            from PIL import ImageChops
            ImageChops.add(bg, frames[0]).save(out / f"{d}_overlay.png")
        # 1周期を8サンプルした横並びストリップ (位相の連続性確認用)
        cyc = max(1, round((a.frames - 1) / 80.0 * 3))
        per = (a.frames - 1) / cyc
        idxs = [min(a.frames - 1, round(k * per / 8)) for k in range(8)]
        strip = Image.new("RGB", (w * 8 // 2, h // 2), (0, 0, 0))
        for k, ix in enumerate(idxs):
            strip.paste(frames[ix].resize((w // 2, h // 2)), (k * w // 2, 0))
        strip.save(out / f"{d}_strip.png")
        strip_rows.append((d, strip))
        if a.gif:
            frames[0].save(out / f"{d}.gif", save_all=True,
                           append_images=frames[1:], duration=63, loop=0)
        print(f"{d}: {len(frames)}f OK")
    mont = Image.new("RGB", (w * 8 // 2, h // 2 * len(strip_rows)), (0, 0, 0))
    for r, (d, strip) in enumerate(strip_rows):
        mont.paste(strip, (0, r * h // 2))
    mont.save(out / "_montage.png")
    print(f"-> {out}/_montage.png (行={','.join(d for d, _ in strip_rows)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
