#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mannequin3d.py -- 3Dマネキンから poseset (方向x位相の見本画像) を生成する.

従来の poseset は動画AI経由 (マネキンキャラを i2v で歩かせて切り出し) で
作られており、方向ごとに別生成のため「同じ瞬間・同じ角度」が保証されず、
ガイド自体が方向間でブレていた (2026-07-11 ユーザー指摘「ポーズマネキン
だと、ちゃんと出力されない」)。本ツールはパラメトリックな3D人形を
コードでポーズ付け・レンダリングするので:

  - 8方向すべてが同一ポーズの真の回転視点 (位相ズレ・角度ブレがゼロ)
  - カメラ俯角は全方向・全コマで厳密に同一 (STILL_PROMPT の 15-20度)
  - 四肢の色ラベルはどの視点でも同じ側 (右腕=赤・左腕=青・右脚=橙・
    左脚=緑) で、前脚ラベル (_limb_labels.json) は幾何学から自動生成
  - 頭身などの体型はパラメータで直接指定できる

レンダラは numpy+PIL のみ (球連鎖のペインターアルゴリズム): ボーンに
沿って球をサンプリングし、奥から順に円を描く。関節は球で自然に繋がる。

色は hybrid_walk._guide_hue_fracs の HSV 窓 (赤h<=9/>=246, 青125-195,
橙12-32, 緑45-110) の中心に入る値を使い、既存の色判定群と互換。

Usage:
    python engine/mannequin3d.py --out posesets_3d [--cell-h 520]
Outputs: <out>/{dir}_{idle,1..5}.png, _montage.png, _limb_labels.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

MAGENTA = (255, 0, 255)
# hybrid_walk の HSV 窓中心に収まるラベル色 (右腕=赤, 左腕=青, 右脚=橙,
# 左脚=緑 -- 3Dなのでどの視点でも同じ側が同じ色)
C_RED = (225, 45, 45)
C_BLUE = (60, 120, 230)
C_ORANGE = (240, 160, 40)
C_GREEN = (110, 200, 60)
C_SKIN = (238, 210, 185)     # 低彩度: 色判定窓に入らない
C_DARK = (70, 50, 40)        # 目・つむじ

# 方向 -> ヨー角 (度)。front=カメラへ正対。left=画面左向き (現行poseset
# と同じ見え方)。カメラは右手系 y-up、キャラ前方 +z、ヨーは +y 軸回り。
DIR_YAW = {
    "front": 0, "back": 180,
    "left": -90, "right": 90,
    "front_left": -45, "front_right": 45,
    "back_left": -135, "back_right": 135,
}
ELEV_DEG = 20     # 俯角 (STILL_PROMPT の 15-20度)。★2026-07-16に15→20:
                  # 俯瞰を強めると前に出た脚 (カメラ寄り) が画面で下へ
                  # 分離し、正面・背面でも「どちらの脚が前か」が読める
                  # (ユーザー要望)。深度による足元の画面高ズレは増えるが、
                  # 方向判読性を優先。参照立ち絵のカメラ帯 (15-20) の内側


# ------------------------------------------------------------ 骨格と姿勢

def _rotx(p, deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    x, y, z = p
    return (x, c * y - s * z, s * y + c * z)


def _roty(p, deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    x, y, z = p
    return (c * x + s * z, y, -s * x + c * z)


def _rotz(p, deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    x, y, z = p
    return (c * x - s * y, s * x + c * y, z)


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


# 頭のワールド高さ (絶対系の基準)。Figure は頭身によらずこの頭サイズで
# 組み立てられ、描画スケールも共通なので、頭も体の横幅もピクセル固定・
# 変わるのは体の高さだけになる (2026-07-11 ユーザー指定「頭は1.25の
# サイズで固定」「体の横幅は変えないで」)。
HEAD_WORLD = 0.5


class Figure:
    """チビ頭身のパラメトリック人形。接地面 y=0、前方 +z。

    ワールド座標は頭基準の絶対系: 頭の高さ=HEAD_WORLD 固定、全高=
    HEAD_WORLD/head_frac (頭身で変化)。幅 (肩幅・腰幅・胴/手足の太さ・
    足の長さ) は頭に対する固定値なので、共通スケールで描けば頭身を
    変えても体の横幅は1pxも変わらない。標準ノブ1.0=2頭身、ノブ1.25
    (=2.5頭身) が従来の見た目アンカー。"""

    def __init__(self, head_frac: float = 0.40):
        # 下限0.125=8頭身 (head_frac_for_leg_scaleの床と一致)。旧下限0.30は
        # チビ時代の名残で、2026-07-16の頭身レンジ開放 (スライダー4.0=
        # 8頭身) がここだけ漏れており、リアル頭身の歩行骨格が黙って
        # 3.3頭身に潰れていた (2026-07-17 ウルファール6頭身実害で発覚)
        hf = max(0.125, min(0.72, head_frac))
        self.head_frac = hf
        H = HEAD_WORLD
        b = H * (1.0 / hf - 1.0)                  # 体の高さ (頭身で変化)
        self.total = b + H                        # 全高 (ワールド)
        self.chin = b                             # あご下端
        self.head_r = H / 2.0
        self.head_cy = b + H / 2.0
        s = b / 0.50              # 高さ寸法の縮尺 (2頭身の体=0.5が基準)
        self.shoulder_y = self.chin - 0.02 * s
        self.hip_y = 0.56 * b
        # ---- 幅は頭身に依存しない固定値 (アンカー=旧ノブ1.25の見た目。
        #      旧実装比 x1.25 が同じピクセル幅に相当) ----
        self.sh_x = 0.1625                        # 肩の半幅
        self.hip_x = 0.0625                       # 股関節の半幅 (広いと俯瞰の
        # 深度差で手前脚が沈み、遊脚クリアランスが画面上で相殺される)
        self.torso_r_top = 0.106
        self.torso_r_bot = 0.125
        self.arm_r = 0.040
        self.leg_r = 0.0525
        self.foot_len = 0.1125
        # ---- 高さ方向の寸法だけ体の縮尺に追従 ----
        self.arm_upper = 0.165 * b
        self.arm_lower = 0.148 * b
        # スタンス実測の外転角 (度・pose_video._fit_figure_to_charが設定。
        # 足開き立ち絵のキャラだけ>0になる。2026-07-20 骨合わせ)
        self.leg_out = 0.0
        # 腕の開き実測の加算角 (度・同上第2弾: 袖広ローブ等で腕の質量が
        # 外に張るキャラは骨格の腕も外へ=袖の中に骨を収める)
        self.arm_out_add = 0.0
        ankle = 0.070 * b
        self.leg_upper = (self.hip_y - ankle) / 2.0
        self.leg_lower = (self.hip_y - ankle) / 2.0

    def _leg_chain(self, side, hp, kn, leg_cross=1.0, gait=False):
        """股→膝→足首→つま先 (ワールド3D)。joints/pose 共用の単一情報源。

        ★モデル歩き (2026-07-16要望): 歩行中は両足を体の中央線の上へ
        内転させる。純粋な振り子 (矢状面のみ) だと正面・背面では前後
        スイングが画面上ほぼ潰れ、歩きが伝わらない。
        ★v2 (同日実走「カニ歩き/酔っ払い」修正): 初版は「前に振る脚だけ」
        内転させたが、接地した足は現実には横へ滑れないのに、体の下を
        通過する間 (前振り→後振り) に中央→股幅へスライドして見え、
        千鳥足になった。モデル歩きの本質は「両足が同じ一本線に着地し、
        ★接地中は線上に留まる」こと。角度だけでは接地/遊脚を区別
        できない (通過中の支持脚と直立が同じ角度署名) ため、gait=True
        (歩行フレーム) のときだけ★膝の屈曲で判別する: 膝が伸びている=
        接地系→中央線上、膝が大きく曲がる=遊脚の通過→線の脇 (股幅) を
        すり抜ける (自然な回し込み。正面で支持脚と重ならない)。
        内転角の上限=伸ばした脚がちょうど中央線に届く角度
        (asin(股半幅/脚長))。leg_cross は倍率ノブ (0=従来の振り子)。
        直立 (gait=False) は内転しない (普通の立ち姿勢)。"""
        hip = (side * self.hip_x, self.hip_y, 0.0)
        v = _rotx((0, -self.leg_upper, 0), -hp)
        w = _rotx((0, -self.leg_lower, 0), -(hp - kn))
        # 足: 前振り脚はつま先上げ (踵接地)、後ろ脚はつま先下げ (蹴り出し)。
        # すね角の35%だけ追従、±28度で頭打ち
        fp = max(-28.0, min(28.0, (hp - kn) * 0.35))
        fdir = _rotx((0, 0, 1), -fp)
        tv = (fdir[0] * self.foot_len, fdir[1] * self.foot_len,
              fdir[2] * self.foot_len)
        ad = 0.0
        if gait and leg_cross > 0.0:
            reach = math.degrees(math.asin(min(1.0, self.hip_x /
                                               (self.leg_upper
                                                + self.leg_lower))))
            ad = (reach * max(0.0, 1.0 - kn / 45.0)
                  * min(1.5, leg_cross))
        # ★スタンス実測の外転 (2026-07-20 骨合わせ): 足開き立ち絵の
        # キャラは常時 leg_out 度だけ脚鎖全体を外へ。歩行の内転(ad)と
        # 同軸の逆符号なので正味角で1回だけ回す (回転経路はモデル歩きで
        # 実証済みの _rotz を共用 = 幾何の新規リスクなし)
        net = ad - float(getattr(self, "leg_out", 0.0))
        if abs(net) > 1e-6:
            v = _rotz(v, -side * net)
            w = _rotz(w, -side * net)
            tv = _rotz(tv, -side * net)
        knee = _add(hip, v)
        ankle = _add(knee, w)
        toe = _add(ankle, tv)
        return hip, knee, ankle, toe

    def joints(self, hipR=0.0, hipL=0.0, kneeR=0.0, kneeL=0.0,
               shR=0.0, shL=0.0, elbR=12.0, elbL=12.0,
               arm_out=9.0, leg_cross=1.0,
               gait=False) -> dict[str, tuple]:
        """ポーズの関節位置 (ワールド3D)。pose() と同じ式で計算する
        (脚は _leg_chain 共用で構造的に一致)。walk_warp のMLS制御点用。"""
        arm_out = arm_out + float(getattr(self, "arm_out_add", 0.0))
        J: dict[str, tuple] = {
            "head_c": (0.0, self.head_cy, 0.0),
            "head_top": (0.0, self.head_cy + self.head_r * 0.9, 0.0),
            "chin": (0.0, self.chin + 0.01, 0.0),
            "torso": (0.0, (self.shoulder_y + self.hip_y) / 2.0, 0.0),
        }
        for side, sh_pitch, elb, tag in ((+1, shR, elbR, "R"),
                                         (-1, shL, elbL, "L")):
            sh = (side * self.sh_x, self.shoulder_y - 0.01, 0.0)
            v = _rotx((0, -self.arm_upper, 0), -sh_pitch)
            v = (v[0] + side * math.sin(math.radians(arm_out))
                 * self.arm_upper, v[1], v[2])
            elbow = _add(sh, v)
            w = _rotx((0, -self.arm_lower, 0), -(sh_pitch + elb))
            wrist = _add(elbow, (w[0] + side * 0.004, w[1], w[2]))
            J[f"sh_{tag}"] = sh
            J[f"elbow_{tag}"] = elbow
            J[f"wrist_{tag}"] = wrist
        for side, hp, kn, tag in ((+1, hipR, kneeR, "R"),
                                  (-1, hipL, kneeL, "L")):
            hip, knee, ankle, toe = self._leg_chain(side, hp, kn, leg_cross, gait)
            J[f"hip_{tag}"] = hip
            J[f"knee_{tag}"] = knee
            J[f"ankle_{tag}"] = ankle
            J[f"toe_{tag}"] = toe
        return J

    # 姿勢 = 関節角 (度)。hip/shoulder は矢状面ピッチ (+=前方へ振る)、
    # knee/elbow は屈曲 (0=伸展)。
    def pose(self, hipR=0.0, hipL=0.0, kneeR=0.0, kneeL=0.0,
             shR=0.0, shL=0.0, elbR=12.0, elbL=12.0,
             arm_out=9.0, leg_cross=1.0,
             gait=False) -> list[tuple]:
        """(point3, radius, color[, normal]) の球リストを返す。normal は
        目・つむじ等の貼り付き装飾の外向き法線 (視線カリング用)。"""
        spheres: list[tuple] = []

        def bone(p0, p1, r0, r1, color, n=None):
            seg = math.dist(p0, p1)
            n = n or max(4, int(seg / (min(r0, r1) * 0.35) + 1))
            for i in range(n + 1):
                t = i / n
                p = (p0[0] + (p1[0] - p0[0]) * t,
                     p0[1] + (p1[1] - p0[1]) * t,
                     p0[2] + (p1[2] - p0[2]) * t)
                spheres.append((p, r0 + (r1 - r0) * t, color))

        # 胴
        bone((0, self.shoulder_y, 0), (0, self.hip_y, 0),
             self.torso_r_top, self.torso_r_bot, C_SKIN)
        # 頭 (大玉1つ + つむじ + 目)
        spheres.append(((0, self.head_cy, 0), self.head_r * 0.98, C_SKIN))
        # つむじ (頭頂やや後ろ): 後ろ向きの判別マーク。カリング閾値を高く
        # して「明確に後ろ寄りの視点」のみ描く (検証指摘: 横顔で頭頂の点が
        # 第2の目に誤読される — 幾何学上は見える位置だが紛らわしい)
        wn = (0, 0.55, -0.84)
        whorl = _add((0, self.head_cy, 0),
                     (0, self.head_r * 0.68, -self.head_r * 0.42))
        spheres.append((whorl, self.head_r * 0.15, C_DARK, wn, 0.30))
        # 目 (顔前面): 前向きの判別マーク。視線カリング付き。頭球の内側に
        # 収める (検証指摘: 3/4視点で奥目が輪郭から数pxはみ出す)
        for ex in (-0.34, 0.34):
            en = (ex * 0.55, -0.08, 0.83)
            eye = _add((0, self.head_cy, 0),
                       (self.head_r * ex, -self.head_r * 0.10,
                        self.head_r * 0.76))
            spheres.append((eye, self.head_r * 0.14, C_DARK, en))

        # 腕 (+x側=赤, -x側=青)。arm_out で軽く外へ開く (胴との分離)。
        # 角度の符号は「+=前方(+z)へ振る」(_rotxは+角で-z側へ倒すため負符号。
        # 検証指摘 2026-07-11: 符号が逆で接地コマの前脚色が全滅していた)
        arm_out = arm_out + float(getattr(self, "arm_out_add", 0.0))
        for side, sh_pitch, elb, color in (
                (+1, shR, elbR, C_RED), (-1, shL, elbL, C_BLUE)):
            sh = (side * self.sh_x, self.shoulder_y - 0.01, 0)
            v = _rotx((0, -self.arm_upper, 0), -sh_pitch)
            v = (v[0] + side * math.sin(math.radians(arm_out))
                 * self.arm_upper, v[1], v[2])
            elbow = _add(sh, v)
            # 前腕: 肘は前方向へのみ屈曲 (人体)
            w = _rotx((0, -self.arm_lower, 0), -(sh_pitch + elb))
            wrist = _add(elbow, (w[0] + side * 0.004, w[1], w[2]))
            bone(sh, elbow, self.arm_r, self.arm_r * 0.9, color)
            bone(elbow, wrist, self.arm_r * 0.9, self.arm_r * 0.82, color)
            spheres.append((wrist, self.arm_r * 1.05, color))  # 手

        # 脚 (+x側=橙, -x側=緑) + 足先。膝は後方へのみ屈曲 (人体)。
        # 幾何は _leg_chain 共用 (モデル歩きの内転もここで揃う)
        for side, hp, kn, color in (
                (+1, hipR, kneeR, C_ORANGE), (-1, hipL, kneeL, C_GREEN)):
            hip, knee, ankle, toe = self._leg_chain(side, hp, kn, leg_cross, gait)
            bone(hip, knee, self.leg_r, self.leg_r * 0.88, color)
            bone(knee, ankle, self.leg_r * 0.88, self.leg_r * 0.8, color)
            bone(ankle, toe, self.leg_r * 0.8, self.leg_r * 0.62, color)
        return spheres


# 歩行5位相 (walk_codex.assemble5 の期待順: 接地A, 通過, 接地B, 通過, 中間)
# +x側の脚前=位相1。数値は関節角(度)。接地コマは前後の足裏高さが揃うよう
# 前脚に軽い膝曲げ (前31/膝10 vs 後-28/膝16 で足裏がほぼ同高)。通過コマは
# 遊脚の膝を高く上げてクリアランスを出す (検証指摘: 膝上げ不足。
# ★2026-07-16実走で66→74へ強化: 「ひざを曲げて後ろから前へ脚を出す」
# 動きが前進感の要 — 遊脚が伸びたまま滑ると skating に見える)。
WALK_POSES = {
    1: dict(hipR=+31, hipL=-26, kneeR=10, kneeL=10,
            shR=-28, shL=+28, elbR=16, elbL=26),
    2: dict(hipR=-2, hipL=+32, kneeR=2, kneeL=74,
            shR=+10, shL=-10, elbR=14, elbL=18),
    3: dict(hipR=-26, hipL=+31, kneeR=10, kneeL=10,
            shR=+28, shL=-28, elbR=26, elbL=16),
    4: dict(hipR=+32, hipL=-2, kneeR=74, kneeL=2,
            shR=-10, shL=+10, elbR=18, elbL=14),
    5: dict(hipR=+10, hipL=-9, kneeR=4, kneeL=8,
            shR=-8, shL=+8, elbR=13, elbL=13),
}
# idle は僅かに脚をずらす: 完全な直立は横顔で両脚が完全重なりして
# 1本脚に見える (検証指摘)
IDLE_POSE = dict(hipR=+2, hipL=-2, kneeR=0, kneeL=0,
                 shR=0, shL=0, elbR=8, elbL=8, arm_out=11)

# 前に出ている脚 (幾何学の正解): hip ピッチが大きい方
FWD_LEG = {k: ("orange" if p["hipR"] > p["hipL"] else "green")
           for k, p in WALK_POSES.items()}


# ------------------------------------------------------------- レンダラ

def render(spheres: list[tuple], yaw: float, out_h: int = 520,
           ss: int = 2, scale_px: float | None = None) -> Image.Image:
    """ヨー+俯角の平行投影でペインター描画し、タイトクロップした
    マゼンタ地 RGB を返す。ss=スーパーサンプル倍率。

    scale_px=None (従来): 図の高さを out_h へ正規化する。
    scale_px 指定: ワールド1.0=scale_px ピクセルの絶対スケールで描き、
    リサイズしない (頭サイズ固定モード用。2026-07-11 ユーザー要望
    「頭のサイズは固定し、体の高さだけで頭身の変化をつけて」)。"""
    proj = []
    for item in spheres:
        p, r, color = item[0], item[1], item[2]
        normal = item[3] if len(item) > 3 else None
        thr = item[4] if len(item) > 4 else 0.15
        if normal is not None:
            # 貼り付き装飾 (目・つむじ) はカメラを向いている時だけ描く:
            # 裏に回ると頭の輪郭からはみ出た黒点になる
            nq = _rotx(_roty(normal, yaw), ELEV_DEG)
            if nq[2] < thr:
                continue
        q = _roty(p, yaw)          # キャラを回す (=カメラのヨー)
        # 俯角: 見下ろしカメラでは手前(カメラ側)が画面の下・奥が上に写る。
        # +ELEV が正方向 (検証エージェント指摘「接地コマの前足が浮く」の
        # 原因は符号逆で手前の前足が持ち上がって写っていたこと 2026-07-11)
        q = _rotx(q, ELEV_DEG)
        proj.append((q[2], q[0], q[1], r, color))
    proj.sort(key=lambda t: t[0])  # 奥 (z小) から順に描く

    S = (out_h * 1.35 if scale_px is None else scale_px) * ss  # world -> px
    # キャンバスは実際の投影範囲から決める (頭身で全高が変わるため固定
    # サイズだと高頭身がはみ出す)
    pad = 4 * ss
    ymin = min(y - r for _, _, y, r, _ in proj)
    ymax = max(y + r for _, _, y, r, _ in proj)
    xext = max(abs(x) + r for _, x, _, r, _ in proj)
    W = int(2 * xext * S) + pad * 2
    H = int((ymax - ymin) * S) + pad * 2
    cx, cy = W / 2.0, pad + ymax * S
    img = Image.new("RGB", (W, H), MAGENTA)
    dr = ImageDraw.Draw(img)
    zs = [t[0] for t in proj]
    z0, z1 = min(zs), max(zs) or 1
    for z, x, y, r, color in proj:
        # 奥行きで僅かに暗く: 立体感の手がかり (ラベル色相は保つ)
        f = 0.82 + 0.18 * ((z - z0) / max(1e-6, z1 - z0))
        col = tuple(min(255, int(c * f)) for c in color)
        px, py, pr = cx + x * S, cy - y * S, max(1.0, r * S)
        dr.ellipse((px - pr, py - pr, px + pr, py + pr), fill=col)
    if ss > 1:
        img = img.resize((W // ss, H // ss), Image.LANCZOS)
    # タイトクロップ
    a = np.asarray(img).astype(int)
    fg = ~((np.abs(a[..., 0] - 255) < 40) & (a[..., 1] < 60)
           & (np.abs(a[..., 2] - 255) < 40))
    ys, xs = np.where(fg)
    img = img.crop((int(xs.min()), int(ys.min()),
                    int(xs.max()) + 1, int(ys.max()) + 1))
    if scale_px is not None:
        return img                 # 絶対スケール: リサイズしない
    # 従来モード: 高さ正規化 (静的posesetのフォーマット: h=520)
    sc = out_h / img.height
    return img.resize((max(2, round(img.width * sc)), out_h), Image.LANCZOS)


def project_point(p: tuple, yaw: float, scale_px: float) -> tuple:
    """render() と同じヨー+俯角の平行投影で 3D点 -> (px, py)。
    y は画面下向き、原点はワールド原点 (接地面)。"""
    q = _rotx(_roty(p, yaw), ELEV_DEG)
    return (q[0] * scale_px, -q[1] * scale_px)


def silhouette_extent(spheres: list[tuple], yaw: float,
                      scale_px: float) -> tuple:
    """render() のタイトクロップと同じ投影範囲 (xmin, ymin, xmax, ymax)
    を球群から解析的に求める (ピクセル座標、project_point と同系)。"""
    xs0 = ys0 = 1e9
    xs1 = ys1 = -1e9
    for item in spheres:
        p, r = item[0], item[1]
        if len(item) > 3:
            continue  # 目・つむじ等の装飾は輪郭に寄与しない扱いで十分
        px, py = project_point(p, yaw, scale_px)
        rr = r * scale_px
        xs0 = min(xs0, px - rr)
        xs1 = max(xs1, px + rr)
        ys0 = min(ys0, py - rr)
        ys1 = max(ys1, py + rr)
    return xs0, ys0, xs1, ys1


def head_frac_for_leg_scale(ls: float) -> float:
    """頭身ノブ (SM_LEG_SCALE) -> head_frac。
    対応則は【頭身数 = 2 × ノブ】(2026-07-11 ユーザー指定「標準は2頭身」):
    ノブ0.6=1.2頭身, 1.0=2頭身, 1.5=3頭身, 3.0=6頭身, 4.0=8頭身。
    2026-07-16「0〜3まで選べるように」→「むしろ8頭身まで」でレンジ開放し、
    さらに同日「見せかけの0は紛らわしい」で**下限を実効飽和点0.6へ統一**
    (旧: ノブ0.575以下は全て1.15頭身に張り付く死にレンジだった)。
    有効域は 0.6〜4.0 = 1.2〜8頭身で、全域が実際に効く。"""
    ls = max(0.6, min(4.0, float(ls)))
    return max(0.125, min(0.834, 1.0 / (2.0 * ls)))


# 頭サイズのアンカー: スライダー1.25 (=2.5頭身) のときの頭の大きさで固定し、
# 頭身は体の高さだけで変える (2026-07-11 ユーザー指定「1.25のサイズで頭を
# 固定。でないと低頭身の時に頭が大きくなりすぎる」)。
# 基準セル高520のとき、1.25時の頭 = head_frac(1.25)=0.40 × 520 = 208px。
HEAD_ANCHOR_LS = 1.25


def head_px_for_cell(cell_h: int) -> float:
    """基準セル高に対する固定頭サイズ (px)。"""
    return head_frac_for_leg_scale(HEAD_ANCHOR_LS) * cell_h


_CELL_CACHE: dict = {}


def render_cells(leg_scale: float = 1.0,
                 cell_h: int = 520) -> dict[tuple[str, str], Image.Image]:
    """(direction, 'idle'|'1'..'5') -> セル画像 を、指定の頭身ノブで
    その場レンダリングして返す (プロセス内キャッシュ)。ガイド合成が
    画像加工 (_headmatch_guide) の代わりに使う: 3Dモデルのプロポーション
    を直接変えるので、潰れ・伸びの加工アーティファクトが出ない。

    頭サイズ固定モード: 頭は常に head_px_for_cell(cell_h) の大きさで
    描かれ、頭身は体の高さ=セルの全高だけで変わる (ノブ1.25で全高が
    ちょうど cell_h、低頭身ほど背が低くなる)。"""
    key = (round(float(leg_scale), 3), int(cell_h))
    if key not in _CELL_CACHE:
        hf = head_frac_for_leg_scale(leg_scale)
        fig = Figure(head_frac=hf)
        # 絶対系: 頭=HEAD_WORLD なので、このスケールは頭身によらず一定
        # = 頭も体の横幅もピクセル固定、体の高さだけが変わる
        spx = head_px_for_cell(cell_h) / HEAD_WORLD
        cells: dict[tuple[str, str], Image.Image] = {}
        for d, yaw in DIR_YAW.items():
            for k2, pose in [("idle", IDLE_POSE)] + [
                    (str(k), WALK_POSES[k]) for k in range(1, 6)]:
                cells[(d, k2)] = render(fig.pose(**pose), yaw,
                                        scale_px=spx)
        _CELL_CACHE[key] = cells
    return _CELL_CACHE[key]


def build_poseset(out_dir: Path, cell_h: int = 520,
                  head_frac: float = 0.50) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = Figure(head_frac=head_frac)
    cells: dict[tuple, Image.Image] = {}
    for d, yaw in DIR_YAW.items():
        for key, pose in [("idle", IDLE_POSE)] + [
                (str(k), WALK_POSES[k]) for k in range(1, 6)]:
            im = render(fig.pose(**pose), yaw, out_h=cell_h)
            im.save(out_dir / f"{d}_{key}.png")
            cells[(d, key)] = im
        print(f"{d}: idle + walk1-5 OK")

    # モンタージュ (目視確認用): 行=方向, 列=idle+5位相
    dirs = list(DIR_YAW)
    cw = max(im.width for im in cells.values()) + 12
    ch = cell_h // 2 + 12
    mont = Image.new("RGB", (cw * 6, ch * len(dirs)), MAGENTA)
    for r, d in enumerate(dirs):
        for c, key in enumerate(["idle"] + [str(k) for k in range(1, 6)]):
            im = cells[(d, key)]
            im2 = im.resize((round(im.width / 2), round(im.height / 2)))
            mont.paste(im2, (c * cw + (cw - im2.width) // 2, r * ch + 6))
    mont.save(out_dir / "_montage.png")

    # 前脚ラベル (幾何学の正解を全方向へ)。sha_v2 = 8方向 x walk1-5
    dirs2 = ("front", "left", "right", "back", "front_left",
             "front_right", "back_left", "back_right")
    h = hashlib.sha1()
    for d2 in dirs2:
        for k in range(1, 6):
            h.update((out_dir / f"{d2}_{k}.png").read_bytes())
    labels = {
        "_provenance": "mannequin3d.py による自動生成 (2026-07-11)。"
                       "3Dモデルの関節角から幾何学的に導出した前脚ラベル。"
                       "右脚=橙, 左脚=緑, 右腕=赤, 左腕=青 (全方向共通)。",
        "poseset_sha1_v2": h.hexdigest(),
        "fwd_legs": {d2: {str(k): FWD_LEG[k] for k in range(1, 6)}
                     for d2 in dirs2},
    }
    (out_dir / "_limb_labels.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"poseset -> {out_dir} (montage: _montage.png)")


def install(src: Path) -> None:
    """生成済みセットを posesets/ へ導入する。旧セット (セル48枚 +
    _limb_labels.json + プレビュー) は posesets/_old_<n>/ へ退避し、
    dist/posesets/ にも同期する (tools_make_posesets_C21 と同じ流儀)。"""
    import shutil
    root = Path(__file__).resolve().parent.parent / "posesets"
    dist = root.parent / "dist" / "posesets"
    n = 1
    while (root / f"_old_{n:02d}").exists():
        n += 1
    old = root / f"_old_{n:02d}"
    old.mkdir(parents=True)
    moved = 0
    for d in DIR_YAW:
        for key in ("idle", "1", "2", "3", "4", "5"):
            p = root / f"{d}_{key}.png"
            if p.is_file():
                shutil.move(str(p), old / p.name)
                moved += 1
    for name in ("_limb_labels.json", "_preview_unified.png"):
        p = root / name
        if p.is_file():
            shutil.move(str(p), old / name)
    print(f"旧poseset {moved}枚 -> {old}")
    copied = 0
    for d in DIR_YAW:
        for key in ("idle", "1", "2", "3", "4", "5"):
            shutil.copy2(src / f"{d}_{key}.png", root / f"{d}_{key}.png")
            copied += 1
    shutil.copy2(src / "_limb_labels.json", root / "_limb_labels.json")
    shutil.copy2(src / "_montage.png", root / "_preview_unified.png")
    print(f"新poseset {copied}枚 -> {root}")
    if dist.is_dir():
        for p in root.glob("*.png"):
            shutil.copy2(p, dist / p.name)
        shutil.copy2(root / "_limb_labels.json", dist / "_limb_labels.json")
        print(f"dist同期 -> {dist}")
    print(f"戻すには: _old_{n:02d}/ の中身を posesets/ (とdist/posesets/) "
          "へ書き戻してください")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="posesets_3d")
    ap.add_argument("--cell-h", type=int, default=520)
    ap.add_argument("--head-frac", type=float, default=0.50,
                    help="頭の高さ/全高 (既定0.50 = 標準2頭身)")
    ap.add_argument("--install", action="store_true",
                    help="生成後、posesets/ へ導入 (旧セットは _old_NN/ へ"
                         "退避、dist/posesets にも同期)")
    a = ap.parse_args()
    build_poseset(Path(a.out), a.cell_h, a.head_frac)
    if a.install:
        install(Path(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
