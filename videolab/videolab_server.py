#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SpriteMill VideoLab — モデル差し替え式 動画生成サーバ + webUI。

1つのファイルで「ローカルGPU実行」と「Colab実行」の両方に対応する。
ブラウザの webUI からモデルを選んで t2v / i2v / キーフレーム動画を試せるほか、
SpriteMill 本体の canvas_walk.py が使う旧 FramePack 契約
(POST /generate_multikey -> GET /status/{job} -> GET /result/{job}, Bearer)
と完全互換なので、SpriteMill のパイプラインからもそのまま呼べる。

使い方(ローカル / GPUのある環境):
    pip install -r requirements.txt        # fastapi uvicorn pillow (+モデル依存)
    python videolab_server.py              # http://127.0.0.1:7860 が開く
    python videolab_server.py --token XXX  # LAN公開時はトークン必須にする

使い方(Colab): colab/SpriteMill_video_lab.ipynb を「すべてのセルを実行」。
最後に表示される URL と TOKEN をブラウザ / SpriteMill に貼り付ける。

モデルアダプタは ADAPTERS への register() 追加だけで増やせる。重い import は
すべて ensure_loaded() 内(遅延)なので、GPUなし環境でも mock で全機能を試せる。

注意: `from __future__ import annotations` は使わないこと。FastAPI のルート関数を
ネストスコープで定義しているため、文字列化された型注釈(Request等)が解決できず
全エンドポイントが 422 になる。Python 3.10+ 前提。
"""
import argparse
import base64
import io
import json
import os
import queue
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

# CUDAの断片化緩和(torchの初回import前に効かせる必要があるためここで設定)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

__version__ = "0.11.0"  # 0.11.0: 動きの型4択=AI経路の一本化 (2026-07-21ユーザー裁定「体型という名前は撤退。二足歩行や四足歩行も含めてVACE依存を完全に捨て、動画AI処理は共通してAniSoraのみの頭部固定インペイント方式に」— 新値ai=AI生成(既定)をanisora_inpaint A型へ、stretch_v/stretch_h/move_vはGPU到達時も手続き完結。VACE骨格系(biped等)はレガシー互換で残置=旧依頼のpack_ready戻し専用。★同時に実穴修正: share_packのpack metaにエイリアス解決後のベース体格しか載らず、*_aiのGPU自動ルーティングが実配線では一度も発火していなかった — run_configにbody_plan_rawを永続化しmetaは生値を書く) / 0.10.56: 背面の頭部固定 (A型インペイント: 顔ボックス無し方向=背面系は従来bbox全体生成で頭部が無防備=振り向き顔の発明を許していた。実測方向の顔帯中央値から頭部帯を推定して凍結・横幅=シルエット実測。2026-07-21ユーザー指摘「こっち向く危険」) / 0.10.55: 管理ノブσ/再加工stepsをAI通過(anisora_inpaint)経路へ配線 (σ=0.85・steps=24ハードコードの死にノブ解消。「AI通過の動き量レバー=σとmotion」が両方とも管理GUIから回せるように。優先順位=API明示>管理ノブ>経路既定0.85) / 0.10.54: キャラ別モーション文 (母艦Codexが依頼の日本語コンセプトを英語モーション記述へ翻案→pack同梱→動画プロンプト末尾へ注入。「日本語がモデルに届かない」問題の正面解決) / 0.10.53: A型インペイント既定化 (ユーザー図解「Aにしないと体は動かない」— 凍結=顔窓+bbox外背景・生成=体全体。B型=下部帯は inpaint_mode:"bottom" で残置) / 0.10.52: インペイント系マスクを顔認識対応 (切断線=実測顔ボックス下端+4%より下にクランプ。赤さんAI通過の瞬き実障害: 巨頭チビはbbox比率の切断線が顔を横切る)。pipelineは*_ai依頼で顔ボックスをVLM実測しshare_packが同梱 / 0.10.51: 体格12択のAI通過 (*_ai=anisora_inpaint既定・体格別frac。素のやわらか値は母艦の手続き経路へ=このサーバには来ない) / 0.10.50: biped_legsの正式経路にlegs_mask=0.55を既定組込 (r3昇格: 忍者実走で黒い塊=参照の上げた脚の残存が完全消滅・上半身は実画素凍結・脚2本のストライド健全) / 0.10.49: 実験r3=legs_mask (VACEマスク同居: 上半身=実画素凍結・脚領域=骨格入り生成・idle/静止=全身実画素。忍者の「参照の上げた脚が第3の脚として残存」の構造的排除) / 0.10.48: 実験r2=anisora_inpaint (手続きアニメ土台+空間latent固定=RePaint式。凍結領域は手続き運動ごと画素完全、四肢だけAniSoraがσ再加工。VACE不使用=ユーザー発案「固定はマスク済みならAniSora単体で完結」) / 0.10.47: 実験r=region_mask (キャラ下部だけ空の潜在=標準インペイントで部分生成、上部=実画素保持) + 実験p=procedural (参照絵へ手続き運動を直接適用・生成なし=同一性画素完全。2026-07-21ユーザー発案「ゆらゆら・上下・変形だけならAI不要では」) / 0.10.46: 四肢二重化対策 (biped_legs=参照の上げた脚を「動く2本のうちの1本」と正宣言 / quadruped_bone=手2つ膝2つの正宣言+crawl腕振り0.55→0.35で骨格と参照の手の権威距離を縮小。2026-07-21実走FB「手が分裂・脚が増える」) / 0.10.45: 体格メニュー8種 (biped_legs=脚のみ骨格+姿勢維持文面の正式化 / quadruped_bone=四つん這い骨格(SM_POSE_GAIT=crawl)+骨格系ゲート共有 / other=キー錨+文面のみへ変更。2026-07-21ユーザー裁定) / 0.10.44: 実験h=legs_only+gait_run (走る忍者実障害: 参照が走り姿勢の依頼に直立歩行骨格の上半身が全面矛盾→二重人格化。骨格=脚(8-13)のみ+走りサイクル正宣言で上半身の姿勢権威を参照へ返す) / 0.10.43: 実験g3=face_line (非人型の間引きギャップに顔限定線画。VLM実測face_boxes.jsonの顔ボックスで線画リファレンスをマスクし、同じ手続き運動に追従させる。顔=見た目の権威を毎フレーム維持しつつ体の中割をVACEに委ねる) / 0.10.42: 実験g2=非人型のpose_every (線画制御の間引き。API明示のみ・既定は毎フレーム線画のまま・flying対象外) / 0.10.41: pose_everyを既定昇格+管理ノブ化 (既定3=3フレームごと骨格・間はVACE中割。1=従来フル制御。優先順位: API明示>管理ノブ>SM_WP_POSE_EVERY>既定。神爺さん/裏ファール/岡田の3体A/Bで中割実動・同一性向上を確認) / 0.10.40: 実験f=pose_every (純制御モードのまま骨格をNフレームごと間引き・間=黒。中割をVACEに委ねる) / 0.10.39: 実験e2=key_interp_pose (実画像アンカー+Nフレームに1回の骨格道しるべ+空潜在) / 0.10.38: 実験e=key_interp (実画像アンカー+灰色空潜在のVACE補間、maskでアンカー保持。VACE-Fun Extension/Loop同型) / 0.10.37: 実験d=scribble_mix (頭=立ち絵線画ボブ追従+体=白棒人間スクリブル。服のヒダを変形させないポーズ指示) / 0.10.36: 頭部完全固定(0.10.35)を撤回=頭はボブ追従へ戻す (ユーザー裁定「上下移動はしてほしい」) / 0.10.34: パペットの後頭部割れ対策 (首より上=頭の無条件所有+頭距離0.5バイアス) / 0.10.33: 実験c=line_puppet (骨格キーポイントで線画をパーツ分割・ボーン相似変換で駆動=キャラ自身の線が歩く制御。二足の骨格代替候補) / 0.10.32: 非二足の既定を線画制御へ反転 (ユーザー目視判定: 深度=ディテール崩れ・線画=完璧維持。スライム娘の前髪で実証) / 0.10.31: 管理ノブnat_control (非二足の制御方式 depth/line/none をGUIから切替) / 0.10.30: 非二足の既定を深度制御へ昇格 (赤さんr2/r3の3-way同条件比較で確定: キー錨のみ=静止 vs depth=這行維持+実動+発明ゼロ)。SM_WP_NAT_CONTROL=none/lineで切替可 / 0.10.29: depth/line_moveの制御からマゼンタを黒正規化 (分布外対策)+姿勢ゲート1.5→1.35 (チビ直立すり抜け対策) / 0.10.28: 実験b=depth_move/line_move (立ち絵実測の疑似深度/線画を体格別の手続き運動で動かして制御へ。骨格語彙に依存しない任意形状対応の布石) / 0.10.27: 顔エッジv2 — 二足はnoeyes (目とface68だけ消し鼻耳=頭アンカー維持=猫背対策)・flyingはボブ骨格維持のままエッジ同期重ね (静止化対策)。既定はoffのまま=SM_WP_EDGE_FACE=onで検証 / 0.10.26: 発明抑制第1弾 — 骨格なし経路に「空白は空白のまま」節 (_WP_NO_PROPS、guidance=1.0でネガ無効のため正宣言)+NO_WINDの歩行前提文を体格整合+motion scoreを管理ノブ化 (既定3.0=V3.2公式標準・レンジ2.0-4.0)+キー錨σの管理ノブ死活修正 / 0.10.25: 顔エッジ固定を既定off (実走で二足=猫背回帰・flying=静止化。動き量適正化と一体で再設計) / 0.10.24: 顔エッジ固定の既定昇格 (骨格の顔点が目を外して顔を壊す対策 — 二足=体のみ骨格+歩行窓に頭部キャニー、flying/非二足=骨格なし+全域頭部キャニー(flyingはsinボブ同期)。SM_WP_EDGE_FACE=offで旧動作) / 0.10.23: 管理ノブ (受付台/adminのGCS config/walkpack_knobs.jsonを依頼ごとに読む=再起動不要。σ/steps/振り/latent固定) / 0.10.22: 非二足の自然移動ルート (赤さん実障害「ハイハイを無理やり二足歩行に」) — quadruped/serpentine/amorphous/otherは二足骨格を出さず、キー錨既定+体格別文面で誘導 / 0.10.21: 隣セル見切れ欠片の除去 / 0.10.20: 取り残し根治3点 (ハートビート・SIGTERM請負解放・停止TOCTOU封じ)
# 0.11.33: 工房aiの未採用末尾4fを次方向への旋回・共有終端錨にした。
# 0.11.32: latent refineでも終端画像アンカー(last_image)を接続。
# 0.11.31: AniSora公式型の疎画像条件を任意stride (sparse24等)へ拡張。
# 0.11.30: 二条件HighのPose/画像比をフレーム別に指定可能にした。
# 0.11.29: 蒸留AniSoraの生成step下限を管理画面どおり4へ接続。
# 4--7を指定しても内部で8へ戻していた隠れクランプを除去。
# 0.11.28: 回転+歩行同時プローブ用に、二条件Highの画像側へ
# anisora_image_guidance_frames_b64の方向別画像列を別入力できるように。
# 0.11.27: 固定のキャラbbox maskを撤廃し、元絵の実シルエット+
# 各時刻のPose周囲だけを開ける動的maskへ。生成余白の黒い帯を根治。
# 0.11.26: 実走で歩いた二条件forwardを工房ai本線へ接続。
# frame0=全面元絵固定、以後=顔/背景固定+体純ノイズ、High=
# OpenPose25%+画像75%の別forward、Low=通常I2V。やりなおしも同経路。
# 0.11.25: 二条件HighのPose画像で黒背景を中立灰へ置換。
# 黒地が骨情報と25%混ざり、体mask全体を黒ずませる実走修正。
# 0.11.24: AniSoraの骨条件と画像条件を同じ36chへ交互に
# 詰めず、Highの同一stepを別forwardしてノイズ予測だけを
# 合成するanisora_dual_condition実験を追加。Lowは通常I2V画像条件。
# 0.11.23: Pose解放後のLow条件で、未知の体域に元絵
# latentを残さず中立灰にする。mask=0でも条件latentの形が
# 参照され、毎フレーム元姿勢へ引き戻していた実走への修正。
# 0.11.22: 空間インペイントの既知/未知maskをWanの
# 36ch条件側にも接続。frame0=全面元絵既知、以後=顔と背景は
# 元絵既知・体はPose条件+未知とし、Low解放後も同じ
# 空間maskを維持する。latent側の体は純ノイズのまま。
# 0.11.21: latent_spatial_start_at_high_edgeを追加。σ1.0からでは
# なく、スケジュール上の最後のHigh-noise expert 1stepから開始し、
# 体=純ノイズのまま骨を1回効かせた直後にLowで人物化する。
# 0.11.20: 生成領域を純ノイズにすると衣装/四肢の所在を失い、
# マスク形の紫ノイズに収束する実走への比較用に、元latentを薄く
# 残すlatent_spatial_source_mix(0.0既定、0.15実験)を追加。
# 0.11.19: HighのPose条件/Low解放を空間latentインペイントへ接続。
# 生成領域=純ノイズ、顔と背景=参照latent軌道の毎step固定を
# 保ったまま、Highにだけ歩行骨を見せてLowで通常条件へ戻す。
# 0.11.18: V3.2のHighだけPose条件、Lowは通常I2V条件へ解放する
# anisora_guidance_release_low実験を追加。骨で大きな姿勢を決めた後、
# Low側に骨線を描かせず先頭絵+文章で仕上げる。
# 0.11.17: AniSora V3/V3.2公式の任意時刻画像条件を忠実移植。
# 全骨フレームを連続動画としてVAEへ入れるのではなく、8フレーム
# ごとの指定時刻だけゼロ動画へ配置し、同じ時刻のmaskを1にする。
# V3 image2video_any.pyとV3.2 image2video.pyで同一の条件構成を確認済み。
# 0.11.16: Multimodal Poseの条件マスクを選択可能に。公式デモ入力は
# frame0=実画像、frame1以降=OpenPose動画なので、firstモードでは
# 条件latentは全時刻保持しつつmask4chは先頭slotだけ1・以後0にする。
# 0.11.15: V3公式configは旧WanModel形式(in_dim)のためDiffusersが既定
# 16chで初期化して36ch重みを拒否する差を修正。Wan2.1 I2V基盤の
# transformer/config.json (in_channels=36)を明示して読み込む。
# 0.11.14: README実演対象そのもののAniSora V3 (Wan2.1系) を読み、
# V3.2と同じ全フレームPose潜在で比較する隔離アダプタを追加。
# 0.11.13: AniSora公式READMEのMultimodal Guidance実配線を検証する
# 隔離プローブ。extra.anisora_guidance_frames_b64へPose/Depth/Line動画を
# 渡した場合だけ、全フレームをVAE encodeして通常I2Vの20ch条件
# (mask4+latent16)へ一括注入する。工房本線からは未使用。
# 0.11.12: Wan-Animate実験の斜め前で、AniSora/VACE用の「正面寄り
# 立ち絵なら顔骨だけ0度へ戻す」ヨー救済が誤発火していたのを分離。
# 正面絵1枚から全方位を作る本経路は参照絵の実測ヨーを使わず、頭・胴とも
# DIR_YAWの公称角度（斜め=45度）を厳守する。
# 0.11.11: 正面立ち絵1枚から、45度刻み8方向×各2歩行周期を1本の
# Wan-Animate動画として生成するdirection_sequence実験を追加。49f区間を
# 1fずつ重ねて385f/8セグメントとし、方向間の同一性を時間軸で共有する。
# 0.11.10: Wan2.2-Animate実験にface_mode=blankを追加。正面立ち絵1枚へ
# 背面骨を与える検証で、静止正面顔のmotion条件が背面化を妨げないようにする。
# 0.11.9: Wan2.2-Animateの前向き歩行を本番経路から隔離して実走できる
# 実験アダプタを追加。工房のOpenPose歩容を直接pose動画にし、顔入力は
# 静止クロップを反復して「歩くか」をプロンプト頼みでなく切り分ける。
# 0.11.8: 管理ノブの0を未設定扱いする共通読取バグを修正。
# head_release_steps=0が既定5へ戻り、head_bob=0も既定1へ戻っていた。
# None/欠落だけを既定へ落とし、数値0はそのままエンジンへ通す。
# 0.11.7: 頭部固定を外す時点を「最後のNステップ」で管理ノブ化。
# 背景固定は終端まで維持し、0=頭部も最後まで固定、既定5=従来の
# 8step時High 3step固定→終盤5step開放と同じ。作品manifestにも実効値を保存。
# 0.11.6: 完成manifestへ実生成設定のスナップショットを保存。管理ノブを
# 後から変えても、作品詳細には生成時のsteps/motion/head_bob/方式が残る。
# 0.11.5: 現行AI生成の管理ノブを整理し、頭部マスクの上下追従量
# (head_bob、既定1.0)を実配線。管理画面はsteps/motion/head_bobだけで
# 新方式を調整できる。
# 0.11.4: High/Low段階別インペイント。Highでは歩行ボブ追従の顔/頭部+
# 背景を参照潜在へ固定し、Lowでは顔/頭部だけ開放して境界を描き切らせる。
# 背景のlatent固定は継続し、decode後の矩形画素貼り戻しは行わない。
# 0.11.3: 顔/背面頭部の固定画素と固定マスクを歩行ボブへ追従させる。
# 静止した顔窓と動く胴の境界だけが伸縮する「うみょん」現象を除去。
# 時間可変マスクをVAE latent時間スロットまで運び、デコード後も各時刻の
# 移動済み原画で画素固定する。
# 0.11.2: AI生成を1方向ずつ8回・各8stepへ分離。複数キャラを同じ
# キャンバスへ置いたときの注意分散をなくし、連続歩行の文面へ変更。
# 0.11.1: AI生成を真のAniSora空間インペイントへ。体マスク内は
# 開始σ1.0の純ノイズ潜在、顔/推定頭部帯/bbox外背景は静止参照を
# 毎step潜在固定+デコード後画素固定。ai->otherの姿勢固定文も撤去。
__version__ = "0.11.44"
# 0.11.44: 分室GCSパックを8方向walkpack検査へ入れる前に分岐し、原画PNGと
# 英訳txtを専用構造へ展開。annex_ pack_idも補助判定にして旧/欠損requestを救済。
# 0.11.43: Codex motion_profileのgait/limb_modeをAI骨格へ配線し、走行語は
# run骨格、脚のみ/腕のみ/四肢なしはOpenPose描画部位を自動切替。GCS分室
# packは通常walkpackと分岐してAniSora単純I2V→MP4+原画posterを納品。
# 0.11.42: Lightning 4stepのVACE High/Lowを経由して、最終1stepだけ
# native AniSora Lowへ渡す三段latent handoffを追加。
# 0.11.41: KijaiのAniSora High要素抽出LoRAをVACE Highへ適用し、
# native AniSora Lowは回転priorを抑えるため最終1--2stepだけに限定する
# latent直結実験へ更新。AniSora High本体の移植とLightning LoRAは使わない。
# 0.11.40: 歩行グラ用に量子化Wan-Animate素体＋AniSora High LoRAを
# 使える隔離口を追加（回転を持つAniSora Low本体は混ぜない）。
# 0.11.39: Wan-Animate制御枝を持つAniSora High/Low二体を3/3stepで
# 切り替える完全二段の隔離実験を追加。
# 0.11.38: AniSora High一体を全ノイズ域へ通す対照実験口を追加。
# 0.11.37: Wan-Animate量子化本体の共通36ch/40層へAniSora V3.2 Lowを
# 丸ごと移植し、pose/motion/face制御専用層だけ温存する隔離実験を追加。
# 0.11.36: Low単独Pose実験に、原画latentとPose latentを同じ20ch
# conditioning内で混合する一回forward方式と、別forward予測合成方式を追加。
# 0.11.35: AniSora V3.2 Low単独の純I2V実験口を追加。
# extra.anisora_low_only=trueで通常生成もLow一体だけをロードし、Highの
# ノイズ域も同じLowへ通す。extra.anisora_flow_shiftで実験ジョブだけ
# scheduler shiftを変更し、次の通常ジョブでは既定5.0へ自動復帰する。
# 0.11.34: aiの内部モーション制御を母艦Codexの画像判定で自動分岐。
# 二足だけOpenPose、四足/飛行/蛇行/不定形/その他はキャラ原画自身を
# 手続き変形した画像列でAniSoraを誘導する。UIの動きの型4択は不変。
# 0.10.3: 監査4件修正 — _snap_valid の空JSON誤判定(無限再DL)、.complete を書き順の最後へ、キャッシュ下限割れの無言フォールバックを可視化、AniSoraドナーconfigを実体dirへ (Hub直参照の迂回を封じる)
# 0.10.1: 依頼リレー — webUIの生成依頼を母艦がclaim/completeし、パック到着でwalkpack自動投入
# 0.10.0: 工房モード — キャラパック+walk_pack API+お友だち用webUI (旧UIは/advanced)
# 0.9.14: ディスク退避キャッシュのquant切替取り違え根治
# 0.9.13: 保証機構のレビュー12件修正 (詳細はcommit)
# 0.9.12: OOM/フリーズしない保証+quant開通 (P0-P2)
# 0.9.11: T4のanisora切替段RAM山対策 — 低RAM VM×block
#         ではDiTを1体ロードするごとに即ディスク退避 (2体+18GBの同時保持を
#         排除。anisoraはLoRA非搭載で安全、vaceはLightning適用があるため
#         従来順)。UMT5遅延ロードにlow_cpu_mem_usage (二重確保の排除)。
# 0.9.10: 低RAM VMのblock offloadをディスク退避へ —
#         T4(RAM51GB)はDiT2体のRAM常駐でロードピーク50GB到達→VMごと
#         OOM killされた実障害の根治 (offload_to_disk_path、モデル別dir、
#         旧diffusersへはTypeErrorフォールバック)。
# 0.9.9: T4級(bf16非対応/sm<80)対応 — Wan系アダプタの
#         bf16直書き19箇所を_pick_dtype()へ (compute capability厳密判定で
#         fp16に自動フォールバック、VAEはfp32へ上げて数値安全)。
#         無料T4でも生成可能に (速度はL4比3〜5倍遅・要実機検証)。
# 0.9.8: latent固定 — 再加工(SDEdit)中に指定フレーム
#         のlatentスロットを毎step stage1へ描き戻す (extra.latent_pin_frames
#         / latent_pin_release)。歩行周期の位相流れ (先頭=終端同位相の崩れ)
#         を錨止めする (2026-07-15要望「中間フレームと最終フレームを固定化」)。
# 0.9.7: Illustrious i2iのOOM根治 — VAEタイリング
#         (1536のinit encode / 1152のdecodeがL4 22GBを超える実測) +
#         ジョブ末のVRAM掃除 (断片化の持ち越しで2回目以降が落ちる)。
#         i2iの画像必須ガード・modes広告・denoiseクランプ・進捗eff計算
#         (静的検証ワークフローの指摘4件)。
# 0.9.6: Drive経由スナップショットに.completeを
#         打ち忘れ — セッション内の2回目以降のロードでもHFチャレンジ+
#         Drive再コピーが走っていた (実障害 2026-07-14)。あわせてHF先行
#         チャレンジは1セッション1回だけ (停滞実績が付いたら以後スキップ)。
#         Illustriousにi2iモード追加 (マネキンコンパス下地+骨格CN併用 —
#         マゼンタ背景とセル配置・斜め向きを下地から引き継ぐ)。
#         0.9.5: Illustrious-XL t2iアダプタ (SDXL+ControlNet
#         OpenPose・8方向シート1枚生成・PNG結果) — ローカル完結の画像生成
#         (次点プロバイダ)。0.9.4: HF先行チャレンジ — Driveの保険がある
#         ときだけHFを1回試し (10秒毎監視)、停滞30秒で即Driveへ切替。
#         健康なHF(トークン認証)はDrive読みの約2倍速い。0.9.3: セッション
#         ディスクの既存コピーを再利用 — モデル切替のたびにDriveから
#         ~19GBを再コピーしていた。0.9.2: スナップショットのマニフェスト照合 —
#         Drive同期未完だと大きいシャードが丸ごと欠ける (実障害:
#         text_encoderのsafetensorsがFileNotFound)。.manifest.jsonの
#         パス+サイズ台帳と照合し、欠けは「同期未完了の疑い」で明示。
#         0.9.1: Drive読みの健全性検査 (空JSON/サイズ不一致)。
#         0.9.0: /healthにdrive状態 (only/mounted/ready) —
#         ⚡自動運転のマウント承諾待ちゲート用。0.8.9: 初回セットアップセル
#         (populate_drive:
#         配置済みなら数秒スキップ・不足分だけHFから取得してDriveへ配置。
#         Run All毎回で安全)。0.8.8: Colabは完全Drive固定 (HF直DL封止)。
#         0.8.7: Driveキャッシュ+curl -f修正。0.8.6: DL経路多段。
#         0.8.5: 停滞自動再接続+HFトークン。0.8.4: hf_transfer撤去。
#         0.8.3: ロード鼓動。0.8.2: L4対応。latent_refine既定
# 0.7.5: 低RAM VM対策 (UMT5ジョブ毎解放+handoff先載せ順)
# 0.7.4: handoffにblock offload (弱GPUで8方向まとめ維持・実験的)
# 0.7.3: 既定量子化をQ4_0へ (16GB未満級上限方針・Q8はGUI撤廃)
# 0.7.2: hybrid既定8step/49f (6step黄変対策) +peakVRAMログ
# 0.7.1: inference_mode/DiT明示退避でVAE 80GB OOM修正
# 0.7.0: VACE High -> native AniSora Low latent直結
# 0.6.9: '-'始まりトークンを再生成 (CLI誤認対策) +0.6.8 refine_cond_still
# 0.6.6: hf_transferでモデルDL高速化(60GB初回が1-2分へ)
# 0.6.5: リファインencodeのno_grad化(勾配グラフでOOMの真因)
# 0.6.4: /api/shutdown=ランタイム自動解放(終了時の片付け)
# 0.6.3: リファインのVRAM退避(A100-40のOOM対策)
# 0.6.2: Colab同梱の古いtorchaoでLoRA適用が落ちる問題の回避
# 0.6.1: vace用Lightning 4step LoRA + High段間引き
# 0.6.0: AniSoraリファイン(SDEdit式・Funポーズ+AniSora質感)
# 0.5.1: vace_end=骨格制御の序盤限定適用(骨転写対策)
# 0.5.0: vace=AniSoraベース移植(アニメprior+8step蒸留で骨格制御)
# 0.4.3: トンネル生存判定をDNS(DoH)に+死産の作り直し
# 0.4.2: トンネル据え置き判定をプロセス生存ベースに
# 0.4.1: トンネルURL据え置き(再実行でURLが変わらない)
# 0.4.0: VACEをGGUF化(共通quant設定が効く・OOM根治)
# 0.3.9: VACEのVAEタイリング(A100-80のencode OOM対策)
# 0.3.8: VACE骨格制御アダプタ(ポーズ駆動・回転根絶)
# 0.3.7: 中間キーフレーム注入(往復回転の対策)
# 0.3.6: 終端画像アンカー(last_image、後ろ向き回転対策)
# 0.3.5: VRAM余裕時はGPU常駐(オフロード無し)=A100高速化
# 0.3.4: ジョブextraで量子化/オフロード指定(共通設定対応)
# 0.3.3: 孤児ジョブ自動中止(cancel_if_unpolled)
# 0.3.2: 入力画像の比率維持パディング(縦伸び事故対策)
# 0.3.1: 依存バージョン検査(古いdiffusers対策)、/healthにlibs
# 0.3.0: AniSora/Wan-A14B+LoRAアダプタ、extra欄、ディスク自動解放

# Colab セットアップセルが入れる依存 (make_notebook.py がここを読む)。
# torch は Colab に CUDA 版が同梱されているため入れない。
# diffusers は 0.39.0 で LTX-2.3 対応が安定版入り (2026-07-03)。
COLAB_PIP = [
    "fastapi uvicorn pillow imageio-ffmpeg",
    "-U \"diffusers>=0.39.0\" transformers accelerate safetensors sentencepiece",
    # ftfy=Wan系プロンプト前処理 / gguf=GGUF量子化ロード / peft=LoRA
    # hf_xet=HFのXet転送経路 (v0.8.6: Colab→HF CDN経路の窒息障害への
    # 迂回路。従来経路とは別インフラ)
    # (hf_transferはv0.8.4で撤去 — 無限ハング障害。下のコメント参照)
    "ftfy gguf peft hf_xet",
]

# hf_transferは撤去 (v0.8.4、2026-07-14実障害): 接続が途中停止すると
# エラーもタイムアウトも出さず永久に固まる (ユーザー実機で再現3回以上:
# GGUF DLのログ行から進まず、ディスク使用量も60秒間±0で完全停止)。
# 標準のPythonダウンローダはHF_HUB_DOWNLOAD_TIMEOUT(既定10秒)の
# 読み取りタイムアウト→自動リトライ+.incompleteからの再開があるので
# 無限ハングしない。速度は実測280MB/s (60GB初回でも数分) で十分。
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# ---------------------------------------------------------------- 基本設定
HERE = Path(__file__).resolve().parent
WORK_ROOT = Path(os.environ.get("VIDEOLAB_WORK", tempfile.gettempdir())) / "videolab_jobs"
DEFAULT_PORT = 7860


def find_ffmpeg() -> str | None:
    """ffmpeg 実行ファイルを探す(環境変数 > PATH > SpriteMill config > imageio)。"""
    cand = os.environ.get("VIDEOLAB_FFMPEG")
    if cand and Path(cand).is_file():
        return cand
    w = shutil.which("ffmpeg")
    if w:
        return w
    # SpriteMill 同梱時: 隣の config.json に ffmpeg パスが入っていることがある
    for cfg in (HERE.parent / "config.json", HERE / "config.json"):
        try:
            p = json.loads(cfg.read_text(encoding="utf-8")).get("ffmpeg")
            if p and Path(p).is_file():
                return p
        except Exception:
            pass
    try:
        import imageio_ffmpeg  # type: ignore
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def encode_mp4(ffmpeg: str, frames_dir: Path, fps: int, dest: Path) -> None:
    """連番PNG -> H.264 mp4(ブラウザ再生互換の yuv420p)。

    timeout必須 (P1): ffmpegの無限ハングは単一workerの全キューを永久停止
    させる (「フリーズしない保証」の穴の一つ、2026-07-16調査)。81f級の
    エンコードは数十秒なので既定15分は十分すぎる余白。"""
    cmd = [ffmpeg, "-y", "-framerate", str(fps),
           "-i", str(frames_dir / "%05d.png"),
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
           "-movflags", "+faststart", str(dest)]
    try:
        to = int(os.environ.get("VIDEOLAB_FFMPEG_TIMEOUT", "900") or 900)
    except ValueError:
        to = 900
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=to)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ffmpegが{to}秒でタイムアウトしました "
                           "(ハング検知 — VIDEOLAB_FFMPEG_TIMEOUTで調整可)")
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr[-800:]}")


# ---------------------------------------------------------------- リクエスト
@dataclass
class GenRequest:
    """アダプタに渡す正規化済みリクエスト。"""
    mode: str = "i2v"                  # t2v | i2v | multikey
    prompt: str = ""
    negative: str = ""
    images: list = field(default_factory=list)   # PIL.Image のリスト(入力順)
    key_positions: list = field(default_factory=list)  # 0.0-1.0 の位置(imagesと同数)
    width: int = 768
    height: int = 512
    num_frames: int = 97
    fps: int = 24
    steps: int = 30
    seed: int = 0
    guidance: float = 3.0
    extra: dict = field(default_factory=dict)     # モデル固有パラメータ


def load_images_b64(items: list[str]):
    from PIL import Image
    out = []
    for s in items:
        if "," in s[:80] and s.lstrip().startswith("data:"):
            s = s.split(",", 1)[1]           # data URL 形式を許容
        img = Image.open(io.BytesIO(base64.b64decode(s)))
        out.append(img.convert("RGB"))
    return out


# ---------------------------------------------------------------- アダプタ
class VideoAdapter:
    """モデルアダプタの基底。1モデル=1クラス。重い import は ensure_loaded 内で。"""
    id = "base"
    label = "base"
    desc = ""
    requires = ""            # 必要環境の目安(webUIに表示)
    modes = ("t2v", "i2v", "multikey")
    defaults: dict = {}      # webUI 初期値の上書き {width,height,num_frames,fps,steps,guidance}

    def __init__(self):
        self.loaded = False

    def ensure_loaded(self, log) -> None:
        self.loaded = True

    def unload(self, log) -> None:
        self.loaded = False

    def generate(self, req: GenRequest, workdir: Path, log, progress) -> Path:
        """動画を生成して mp4 のパスを返す。progress(0..1) を随時呼ぶこと。"""
        raise NotImplementedError

    def info(self) -> dict:
        return {"id": self.id, "label": self.label, "desc": self.desc,
                "requires": self.requires, "modes": list(self.modes),
                "loaded": self.loaded, "defaults": self.defaults}


ADAPTERS: dict[str, VideoAdapter] = {}


def register(cls):
    ADAPTERS[cls.id] = cls()
    return cls


def _free_cuda(log):
    try:
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        log("VRAM解放: torch.cuda.empty_cache()")
    except Exception:
        pass


def _log_cuda_state(log, label: str) -> None:
    """GPU実機ログへ現在のallocated/reserved/freeを残す。"""
    try:
        import torch
        free, total = torch.cuda.mem_get_info()
        alloc = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        log(f"VRAM[{label}]: alloc={alloc / 2**30:.2f}GB / "
            f"reserved={reserved / 2**30:.2f}GB / "
            f"free={free / 2**30:.2f}GB / total={total / 2**30:.2f}GB")
    except Exception:
        pass


def _disk_used_gb() -> float:
    try:
        return shutil.disk_usage(str(Path.home())).used / 2**30
    except Exception:
        return -1.0


# ---------------------------------------------------------------- 保証機構
# 「OOM・フリーズしない保証」(2026-07-16調査に基づくv0.9.12):
# クライアントのplan_canvasは最適寸法の設計、サーバ側は拒否/降格の最終
# 防衛線という役割分担。係数は/healthで公開して単一ソース化する。
ACT_GB_PER_LAT = 0.85     # 活性化GB/latentフレーム @720x1296 (A100実測、
#                           クライアントplan_canvasと同一係数)
ACT_SAFETY_GB = 1.5       # CUDAコンテキスト・断片化の口銭 (同上)


def _avail_ram_gb() -> float:
    """空きシステムRAM (GB)。測れなければ -1。"""
    try:
        import psutil
        return psutil.virtual_memory().available / 2**30
    except Exception:
        try:
            with open("/proc/meminfo", encoding="ascii") as f:
                for line in f:
                    if line.startswith("MemAvailable"):
                        return int(line.split()[1]) * 1024 / 2**30
        except Exception:
            pass
        return -1.0


def _gguf_gb(*paths) -> float:
    """GGUFファイル群の実サイズ合計 (GB)。読めないものは0扱い。"""
    total = 0.0
    for p in paths:
        try:
            total += os.path.getsize(str(p)) / 2**30
        except Exception:
            pass
    return total


def _ram_gate(log, need_gb: float, what: str) -> None:
    """ロード前RAMゲート (P0-2): RAM OOM=VM killを実行前の明示エラーへ。

    RAM側OOMはカーネルのOOM killでプロセスごと消え、クライアントには
    「無応答のまま課金」として見える (2026-07-13/16に実障害5件)。
    ロード開始前に空きRAMと見積もりを突き合わせ、不足なら対処法つきで
    ジョブエラーにする。VIDEOLAB_RAM_GATE=off で無効。"""
    if os.environ.get("VIDEOLAB_RAM_GATE", "on").strip().lower() in (
            "off", "0", "false", "no"):
        return
    if need_gb <= 0:
        return
    avail = _avail_ram_gb()
    if avail < 0:
        return                      # 測れない環境ではゲートしない
    if avail >= need_gb:
        log(f"RAMゲート[{what}]: 必要≈{need_gb:.0f}GB / 空き{avail:.0f}GB OK")
        return
    raise RuntimeError(
        f"システムRAM不足の見込みのため{what}を中止しました "
        f"(必要≈{need_gb:.0f}GB > 空き{avail:.0f}GB)。このまま進むと"
        "VMごとOOM killされ「無応答のまま課金」になります。対処: "
        "①quantを下げる (例 Q3_K_S) ②高RAM VM/上位GPUを引き直す "
        "③リファイン専用ならLow専用ロード (latent_from指定で自動) "
        "④どうしても試すなら VIDEOLAB_RAM_GATE=off")


def _norm_quant(q, default: str = "Q4_0", allow_bf16: bool = False) -> str:
    """量子化指定の正規化+検証 (P2: quantラダー開通)。

    旧実装は未知値を黙ってQ4_0へ矯正しており、Q3_K_S等の下位quantを
    渡しても「エラーすら出ずQ4_0で走る」偽成功だった (2026-07-16調査)。
    GGUF命名 (Q4_0/Q8_0/Q3_K_S/Q5_K_M...) に合う値だけ通し、合わないものは
    明示エラー。在庫の有無はHF側の404が教えてくれる (_hf_preflightで前倒し)。"""
    s = str(q or default).strip()
    if allow_bf16 and s.lower() == "bf16":
        return "bf16"
    import re as _re
    sn = s.upper().replace("-", "_")
    if _re.fullmatch(r"Q\d(_\d|_K(_[SML])?)", sn):
        return sn
    raise ValueError(
        f"未知の量子化指定: {q!r} (例: Q4_0 / Q8_0 / Q3_K_S / Q5_K_M"
        + (" / bf16" if allow_bf16 else "") + ")")


def _hf_file_missing(repo: str, filename: str):
    """HFリポにファイルが無いことが確定したらTrue。判定不能はNone。
    (オフライン/Drive固定運転/テストの偽hubでは None -> 従来どおり
    ダウンロード側に任せる)"""
    try:
        from huggingface_hub import file_exists
        return not bool(file_exists(repo, filename))
    except Exception:
        return None


def _anisora_high_quant(quant: str) -> str:
    """AniSora High側のGGUF棚在庫はQ4_0/Q8_0のみ (2026-07-16 HF tree API
    実測の静的知識)。下位quant指定時はHighだけQ4_0へ代替する。

    v0.9.13: 当初はfile_existsで動的判定していたが、Drive固定運転や429で
    HF APIが死んでいる環境では判定不能=代替が無言不発になり、存在しない
    High-Q3_K_S.ggufを追う袋小路DLループに落ちた (レビュー指摘)。棚在庫は
    既知の静的事実なのでコードに焼く。上流に下位quantのHighが追加されたら
    VIDEOLAB_ANISORA_HIGH_QUANT で差し替えられる。"""
    if quant in ("Q4_0", "Q8_0"):
        return quant
    return (os.environ.get("VIDEOLAB_ANISORA_HIGH_QUANT", "Q4_0").strip()
            or "Q4_0")


def _act_estimate_gb(w: int, h: int, n: int) -> float:
    """生成活性化のピークVRAM見積もり (GB)。plan_canvasと同一式。"""
    lat = (max(2, int(n)) - 1) // 4 + 2      # +1=VACE参照タイムスロット
    return ACT_GB_PER_LAT * lat * (float(w) * float(h)) / (720.0 * 1296.0)


def _admit_vram(w: int, h: int, n: int, log, resident_extra_gb: float = 0.0,
                downgrade=None, tag: str = "") -> None:
    """生成直前VRAMゲート (P0-3): 空きVRAMと活性化見積もりを突き合わせ、
    不足なら downgrade() (block offloadへの組み替え等) を1回試し、それでも
    不足なら数十分燃やす前に明示エラーで返す。GPUの無い環境 (テスト等) は
    何もしない。VIDEOLAB_ADMIT=off で無効。"""
    if os.environ.get("VIDEOLAB_ADMIT", "on").strip().lower() in (
            "off", "0", "false", "no"):
        return
    try:
        import torch
        if not torch.cuda.is_available():
            return
        free = torch.cuda.mem_get_info()[0] / 2**30
    except Exception:
        return
    act = _act_estimate_gb(w, h, n)
    need = act + resident_extra_gb + ACT_SAFETY_GB
    if free >= need:
        log(f"VRAM admission[{tag}]: 必要≈{need:.1f}GB "
            f"(活性化{act:.1f}+常駐余地{resident_extra_gb:.1f}) / "
            f"空き{free:.1f}GB OK")
        return
    if downgrade is not None:
        log(f"⚠ VRAM不足見込み (必要≈{need:.1f}GB > 空き{free:.1f}GB) "
            "-> 省メモリ構成へ自動降格します (低速・確実)")
        _ok = False
        try:
            _ok = bool(downgrade())
        except Exception as e:      # noqa: BLE001
            log(f"自動降格に失敗 (そのまま判定続行): {str(e)[:120]}")
        try:
            import torch
            free = torch.cuda.mem_get_info()[0] / 2**30
        except Exception:
            return
        # 降格が実際に成功したときだけ「重みはVRAM外」前提へ切り替える。
        # 失敗時に前提を切り替えると偽PASS→生成中OOMになる (レビュー指摘)
        need = act + ACT_SAFETY_GB + (0.5 if _ok else resident_extra_gb)
        if free >= need:
            log(f"VRAM admission[{tag}]: {'降格後' if _ok else '再判定'}OK "
                f"(必要≈{need:.1f}GB / 空き{free:.1f}GB)")
            return
    import math as _math
    scale = max(0.05, (free - ACT_SAFETY_GB - resident_extra_gb)
                / max(0.05, act))
    sw = max(96, int(w * _math.sqrt(scale)) // 16 * 16)
    sh = max(96, int(h * _math.sqrt(scale)) // 16 * 16)
    raise RuntimeError(
        f"VRAM不足の見込みのため生成前に中止しました "
        f"(必要≈{need:.1f}GB > 空き{free:.1f}GB, {w}x{h}/{n}f)。"
        f"対処: ①解像度を下げる (目安 {sw}x{sh} 以下) ②フレームを減らす "
        "(49→33) ③extra.offload=block ④VIDEOLAB_ADMIT=off (非推奨)")


def _run_with_stall_watch(cmd, env, log, tag, stall_secs=90) -> int:
    """子プロセスを起動し、ディスク使用量が stall_secs 増えなければkill。
    転送プロトコルによらず「何かが書かれているか」で生死判定する
    (v0.8.6: リポキャッシュdir監視だとXet等のチャンクキャッシュを
    見落とすため、ディスク全体のusedへ変更)。戻り値=returncode
    (killしたら None 扱いで -1)。"""
    p = subprocess.Popen(cmd, env=env)
    last, stall = -1.0, 0.0
    while True:
        time.sleep(10)
        if p.poll() is not None:
            return p.returncode
        cur = _disk_used_gb()
        if cur < 0 or cur - last > 0.01:
            stall = 0.0
        else:
            stall += 10
            if stall >= stall_secs:
                log(f"⚠ DL停滞{int(stall)}秒 -> 打ち切って別経路で再試行 "
                    f"({tag})")
                p.kill()
                p.wait()
                return -1
        last = cur


def _drive_cache_dir() -> Path | None:
    """Google Driveのモデルキャッシュ (v0.8.7)。ノートのDRIVE_CACHEセルが
    マウント成功時に VIDEOLAB_DRIVE_CACHE を設定する。HFへ一切触らずに
    毎セッション数分でモデルを積める恒久脱出路 (2026-07-14 HF側の
    429拒否がIP/アカウント割当で解けなくなった際のユーザー発案)。"""
    p = os.environ.get("VIDEOLAB_DRIVE_CACHE", "").strip()
    return Path(p) if p else None


def _drive_only() -> bool:
    """Colabはモデル取得を完全Drive固定 (v0.8.8、2026-07-14ユーザー指示
    「毎回DLは詰まることがわかったので」)。ノートのcell2が
    VIDEOLAB_DRIVE_ONLY=1 を立てる。ローカルGPUモード (家庭回線) は
    HF直DLが健全なので従来どおり多段DL。"""
    return os.environ.get("VIDEOLAB_DRIVE_ONLY", "").strip() == "1"


def _drive_only_error(what: str, expect) -> RuntimeError:
    hint = (f"期待する配置: {expect}" if expect is not None
            else "Drive未マウント: セル2を再実行して許可してください")
    return RuntimeError(
        f"Drive固定運転: {what} がDriveキャッシュにありません — {hint} "
        "(PCの G:\\マイドライブ\\SpriteMill_models へ配置するか、"
        "セル2のDriveマウント許可を確認)")


_HF_STALLED = False   # このセッションでHF先行チャレンジが停滞した実績
#                       (一度死んだHFは部品ごとに再挑戦しない — 30秒/部品
#                       の無駄待ちを1セッション1回に抑える v0.9.6)

_MIN_CACHE_BYTES = 2**30   # キャッシュ採用の下限 (429エラーページ・中断
#                            コピーを「完備」と誤認しないための粗いふるい)


def _cache_size_ok(p: Path, log, label: str, expect: int = 0) -> bool:
    """キャッシュファイルのサイズゲート。**落ちたら必ず声を出す** (v0.10.3)。

    2026-07-19監査: 旧コードはこのゲートを無言のフォールスルーで書いて
    いたため、下限に届かないキャッシュがあると「ネットワーク不要」の
    はずの運転が黙ってHF実DLへ落ちていた (ログにも残らない)。原因が
    キャッシュ側にあることを必ずログに出す。

    expect (Drive原本のサイズ等・分かる場合) を渡せば実サイズ照合を
    優先する — 1GiB固定はLightning LoRA(約1.2GB)のように余裕が200MB
    しかない部品には粗すぎる。"""
    if not p.is_file():
        return False
    sz = p.stat().st_size
    if expect > 0 and sz != expect:
        log(f"⚠ キャッシュ {label} がサイズ不一致 ({sz / 2**20:.0f}MB /"
            f" 期待 {expect / 2**20:.0f}MB) — 使いません")
        return False
    if sz <= _MIN_CACHE_BYTES:
        log(f"⚠ キャッシュ {label} が下限 "
            f"{_MIN_CACHE_BYTES / 2**30:.1f}GB 未満 ({sz / 2**20:.0f}MB) — "
            "壊れた/途中のコピーとみなして使いません。HFからの実DLへ"
            "フォールバックします (ネットワークが必要になります)")
        return False
    return True


def _hf_download(repo: str, filename: str, log, attempts: int = 6,
                 stall_secs: int = 90) -> str:
    """モデルDLの多段フォールバック (v0.8.5-0.8.7)。

    実測 (2026-07-14): Colab VM→HFが429で即拒否 (IP/無料アカウントの
    ダウンロード割当を再試行の繰り返しで使い切り)。hf標準DLの「停止」の
    正体は429の無言バックオフ。優先順:
      0) Google Driveキャッシュにあればローカルへコピーして即返す
      1,3,5) hf_hub_download (Xetが入っていればXet経路)
      2,4) HF_HUB_DISABLE_XET=1 (従来CDN経路へ強制)
      6) curl直DL (-f必須: エラーページを成功扱いした実バグの再発防止。
         サイズ1GB未満も失敗扱い)
    成功したらDriveキャッシュへ自動書き戻し (次セッションからHF不要)。"""
    from huggingface_hub import hf_hub_download
    dc = _drive_cache_dir()
    dsrc = (dc / repo.replace("/", "--") / filename) if dc else None

    def _writeback(src: str) -> None:
        """Drive書き戻し (populate再実行でも収束するようキャッシュ命中時も
        行う)。"""
        if dsrc is None or dsrc.is_file():
            return
        try:
            log(f"Driveキャッシュへ書き戻し: {filename} (次セッションから"
                "HF不要になります)")
            dsrc.parent.mkdir(parents=True, exist_ok=True)
            tmp = dsrc.with_suffix(dsrc.suffix + ".part")
            shutil.copy2(src, tmp)
            tmp.rename(dsrc)
        except OSError as e:
            log(f"Drive書き戻しスキップ: {str(e)[:120]}")

    try:
        got0 = hf_hub_download(repo, filename, local_files_only=True)
        _writeback(got0)
        return got0
    except TypeError:
        # テストの偽hf_hub_download(位置引数のみ)はそのまま呼ぶ
        return hf_hub_download(repo, filename)
    except Exception:
        pass
    dst = WORK_ROOT / "_dl" / repo.replace("/", "--") / filename
    dst.parent.mkdir(parents=True, exist_ok=True)
    # セッションディスクの前回コピーを再利用 (v0.9.3、2026-07-14ユーザー
    # 指摘「2回目以降の生成もドライブから毎回DLしてる」— モデル切替の
    # たびに~19GBを再コピーしていた)。Driveの原本とサイズ一致なら採用
    _want = (dsrc.stat().st_size
             if (dsrc is not None and dsrc.is_file()) else 0)
    if dst.is_file() and _cache_size_ok(
            dst, log, f"セッションディスク {filename}", _want):
        log(f"セッションディスクの既存コピーを再利用: {filename}")
        return str(dst)
    code = ("from huggingface_hub import hf_hub_download; "
            f"hf_hub_download({repo!r}, {filename!r})")
    if dsrc is not None and _cache_size_ok(dsrc, log, f"Drive {filename}"):
        # HF先行チャレンジ (v0.9.4、ユーザー要望「トークンせっかく用意
        # してるから最初1回だけHFからDLチャレンジして、10秒ごとに監視、
        # 駄目ならドライブへ」)。健康なHFはDrive FUSE読み(~150MB/s)の
        # 約2倍速い。Driveの保険があるときだけ試し、30秒停滞で即切替。
        # 停滞実績が付いたセッションでは以後スキップ (v0.9.6)
        global _HF_STALLED
        if not _HF_STALLED:
            tag = "HF先行チャレンジ (停滞30秒でDriveへ切替)"
            log(f"DL開始: {tag}")
            rc = _run_with_stall_watch([sys.executable, "-c", code],
                                       dict(os.environ), log, tag,
                                       stall_secs=30)
            if rc == 0:
                try:
                    return hf_hub_download(repo, filename,
                                           local_files_only=True)
                except Exception:
                    pass
            _HF_STALLED = True
            log("HFが進まないためDriveキャッシュへ切り替えます")
        else:
            log("HF先行チャレンジをスキップ (このセッションで停滞済み) — "
                "Driveキャッシュから取得します")
        want_sz = dsrc.stat().st_size
        for att in (1, 2):
            log(f"Driveキャッシュからコピー: {filename} "
                f"({want_sz / 2**30:.1f}GB・HF不要)")
            shutil.copy2(dsrc, dst)
            got_sz = dst.stat().st_size if dst.is_file() else 0
            if got_sz == want_sz:
                return str(dst)
            # Drive FUSEは同期未完のファイルを短く読ませることがある
            # (2026-07-14実障害と同族) — サイズ照合で検出して再試行
            log(f"⚠ コピーがサイズ不一致 ({got_sz}/{want_sz}) — Drive同期"
                f"未完の疑い。30秒待って再コピー ({att}/2)")
            dst.unlink(missing_ok=True)
            time.sleep(30)
        raise RuntimeError(
            f"Driveの {filename} が読み切れません (同期未完了の疑い)。"
            "PC側のGoogle Drive同期の完了を待ってからやり直してください")
    if _drive_only() and dc is None:
        # Drive固定でマウントも無い=取得手段なし (v0.9.5改: マウント済みで
        # Driveに無いだけなら、HFから取得してDriveへ書き戻す=その場populate。
        # Illustrious等の新モデルをDrive固定が永久ブロックしない)
        raise _drive_only_error(filename, dsrc)

    def _ok(path: Path) -> bool:
        return path.is_file() and path.stat().st_size > _MIN_CACHE_BYTES

    got = None
    code = ("from huggingface_hub import hf_hub_download; "
            f"hf_hub_download({repo!r}, {filename!r})")
    for att in range(1, attempts + 1):
        env = dict(os.environ)
        if att >= attempts:
            url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
            # -f: HTTPエラー時は失敗扱い (無いと429のエラーページを保存して
            # 成功と誤認する — 2026-07-14実バグ)
            cmd = ["curl", "-fsSL", "-C", "-", "--speed-limit", "500000",
                   "--speed-time", "30", "-o", str(dst), url]
            tok = os.environ.get("HF_TOKEN", "")
            if tok:
                cmd[1:1] = ["-H", f"Authorization: Bearer {tok}"]
            tag = f"curl直DL 試行{att}/{attempts}"
        else:
            if att % 2 == 0:
                env["HF_HUB_DISABLE_XET"] = "1"   # 従来CDN経路へ強制
                tag = f"hf(従来経路) 試行{att}/{attempts}"
            else:
                tag = f"hf(既定経路) 試行{att}/{attempts}"
            cmd = [sys.executable, "-c", code]
        log(f"DL開始: {tag}")
        rc = _run_with_stall_watch(cmd, env, log, tag,
                                   stall_secs=stall_secs)
        if rc == 0:
            if att >= attempts:
                if _ok(dst):
                    got = str(dst)
                    break
                log("curl直DL: サイズ不足 (エラー応答の可能性) -> 失敗扱い")
                try:
                    dst.unlink(missing_ok=True)
                except OSError:
                    pass
            else:
                try:
                    got = hf_hub_download(repo, filename,
                                          local_files_only=True)
                    break
                except Exception:
                    pass
        time.sleep(min(60, 5 * att))
    if not got:
        raise RuntimeError(
            f"モデルDLが{attempts}回とも失敗: {repo}/{filename} — HF側の"
            "429拒否 (IP/アカウントのDL割当) の可能性。①ランタイムを終了して"
            "別VMを引き直す ②DriveキャッシュにGGUFを置く (ノートの"
            "DRIVE_CACHEセル参照) のどちらかで回避できます")
    _writeback(got)
    return got


def _snap_valid(root: Path) -> str:
    """スナップショットの健全性検査 (v0.9.1-0.9.2、v0.10.3で誤判定を修正)。

    ①.manifest.json (PC配置時に生成した 相対パス→サイズ の台帳) と
    照合 — Drive同期が未完だと大きいシャードが丸ごと欠ける実障害
    (2026-07-14: text_encoderのsafetensorsがFileNotFound)。
    ②全JSONのパース — FUSEは同期未完ファイルを『サイズはあるのに
    読むと空』で返す実障害 (config.json 0バイト)。
    問題があればその相対パスを返す (""=健全)。

    v0.10.3: ②は `if not json.loads(...)` と書かれていて、**ファイル**
    ではなく**パース結果の真偽値**を見ていた。{} / [] / 0 / false / ""
    に正当にパースされるJSON (空のindexやフラグ類) が毎回「破損」判定に
    なり、呼び出し側がrmtree→再コピー→同じ判定…と抜けられないループに
    落ちて「同期未完了の疑い」で終わる。パースが通れば健全とみなし、
    本当に読めない/壊れているものだけを弾く (0バイトは明確に異常)。"""
    import json as _json
    mf = root / ".manifest.json"
    if mf.is_file():
        try:
            man = _json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            return ".manifest.json"
        for rel, size in man.items():
            p = root / rel
            if not p.is_file() or p.stat().st_size != int(size):
                return rel
    for jp in root.rglob("*.json"):
        if ".cache" in jp.parts or jp.name == ".manifest.json":
            continue
        try:
            if jp.stat().st_size == 0:
                return str(jp.relative_to(root))
            _json.loads(jp.read_text(encoding="utf-8"))   # 値は問わない
        except Exception:
            return str(jp.relative_to(root))
    return ""


def _snap_writeback(local: Path, dsrc: Path, log) -> None:
    """スナップショットをDriveキャッシュへ書き戻す (v0.10.3で順序を修正)。

    サイドカーの書き順が肝: `.manifest.json` を書き切ってから
    `.complete` を**最後の1バイト**として打つ。旧コードは copytree の
    中身として `.complete` が (名前順で先に) 流れ込み、manifestは最後
    だったため、中断すると Driveに「.complete はあるが manifest 無し
    /ファイル欠け」という状態が残った。drive_cache_ready は .complete
    しか見ないので「配置済み」と報告する一方、_snap_valid は落ちるので
    毎ブート再DLになる。"""
    log(f"Driveキャッシュへベース部品を書き戻し: {local.name}")
    # .complete は copytree に運ばせない (完成の宣言は最後に自分で打つ)
    shutil.copytree(local, dsrc, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(".cache", ".complete"))
    # マニフェスト (v0.9.2): 次セッションが同期未完を検出できる
    import json as _json
    man = {f.relative_to(local).as_posix(): f.stat().st_size
           for f in local.rglob("*")
           if f.is_file() and ".cache" not in f.parts
           and f.name not in (".complete", ".manifest.json")}
    (dsrc / ".manifest.json").write_text(
        _json.dumps(man, ensure_ascii=False), encoding="utf-8")
    (dsrc / ".complete").touch()   # ← 木に対する最後の書き込み


def _snapshot_local(repo: str, log, attempts: int = 5) -> str:
    """ベース部品リポ(VAE/UMT5/config類)をsymlink無しのプレーンdirへ取得
    (v0.8.7)。transformer重み(GGUF側で持つ)は除外して約12GBに抑える。
    Driveキャッシュがあれば最優先でコピー、DL成功時は書き戻し。
    from_pretrained/from_single_file(config=)にはこのdirを渡す。"""
    import huggingface_hub as _hh
    if not hasattr(_hh, "snapshot_download"):
        # テストの偽huggingface_hub (hf_hub_downloadのみ): リポ名を素通し
        # (from_pretrained側も偽なので実DLは起きない)
        return repo
    name = repo.replace("/", "--")
    local = WORK_ROOT / "_snap" / name
    marker = local / ".complete"

    if marker.is_file():
        bad = _snap_valid(local)
        if not bad:
            return str(local)
        log(f"⚠ ローカルスナップショット破損 ({bad}) — 取り直します")
        shutil.rmtree(local, ignore_errors=True)
    dc = _drive_cache_dir()
    dsrc = (dc / "_snap" / name) if dc else None
    _ig = ("['transformer/*.safetensors','transformer_2/*.safetensors',"
           "'*.bin','*.md','.git*']")
    _code = ("from huggingface_hub import snapshot_download; "
             f"snapshot_download({repo!r}, local_dir={str(local)!r}, "
             f"ignore_patterns={_ig})")
    if dsrc is not None and (dsrc / ".complete").is_file():
        # HF先行チャレンジ (v0.9.4): Driveの保険があるときだけ1回試す。
        # 停滞30秒で即Driveへ切替 (10秒ごと監視は_run_with_stall_watch)。
        # 停滞実績が付いたセッションでは以後スキップ (v0.9.6)
        global _HF_STALLED
        if not _HF_STALLED:
            tag = "ベース部品HF先行チャレンジ (停滞30秒でDriveへ切替)"
            log(f"DL開始: {tag} ({repo})")
            rc = _run_with_stall_watch([sys.executable, "-c", _code],
                                       dict(os.environ), log, tag,
                                       stall_secs=30)
            if rc == 0 and not _snap_valid(local):
                marker.touch()
                return str(local)
            _HF_STALLED = True
            log("HFが進まないためDriveキャッシュへ切り替えます")
        else:
            log("HF先行チャレンジをスキップ (このセッションで停滞済み) — "
                "Driveキャッシュから取得します")
        for att in (1, 2):
            log(f"Driveキャッシュからベース部品をコピー: {repo} (HF不要)")
            shutil.copytree(dsrc, local, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns(".cache"))
            bad = _snap_valid(local)
            if not bad:
                # Drive経由でも完成マーカーを打つ (v0.9.6: 打ち忘れで
                # 2回目以降のロードが毎回ここへ落ちて再コピーしていた)
                marker.touch()
                return str(local)
            log(f"⚠ Driveから空のファイルを掴みました ({bad}) — Drive同期が"
                f"未完の可能性。30秒待って再コピーします ({att}/2)")
            shutil.rmtree(local, ignore_errors=True)
            time.sleep(30)
        raise RuntimeError(
            f"Driveのベース部品 {repo} が読み切れません (同期未完了の疑い)。"
            "PC側のGoogle Drive同期の完了を待ってからやり直してください")
    if _drive_only() and dc is None:
        # マウント無しのDrive固定のみブロック (マウント済みならHFから
        # 取得してDriveへ書き戻す=その場populate)
        raise _drive_only_error(f"ベース部品 {repo}",
                                dsrc / ".complete" if dsrc else None)
    ig = "['transformer/*.safetensors','transformer_2/*.safetensors','*.bin','*.md','.git*']"
    code = ("from huggingface_hub import snapshot_download; "
            f"snapshot_download({repo!r}, local_dir={str(local)!r}, "
            f"ignore_patterns={ig})")
    ok = False
    for att in range(1, attempts + 1):
        env = dict(os.environ)
        if att % 2 == 0:
            env["HF_HUB_DISABLE_XET"] = "1"
        tag = f"ベース部品snapshot 試行{att}/{attempts}"
        log(f"DL開始: {tag} ({repo})")
        rc = _run_with_stall_watch([sys.executable, "-c", code], env, log,
                                   tag)
        if rc == 0:
            ok = True
            break
        time.sleep(min(60, 5 * att))
    if not ok:
        raise RuntimeError(
            f"ベース部品のDLが{attempts}回とも失敗: {repo} — HF側の429拒否の"
            "可能性。①ランタイム終了→別VM ②Driveキャッシュ (ノートの"
            "DRIVE_CACHEセル) で回避できます")
    marker.touch()
    if dsrc is not None and not (dsrc / ".complete").is_file():
        try:
            _snap_writeback(local, dsrc, log)
        except OSError as e:
            log(f"Drive書き戻しスキップ: {str(e)[:120]}")
    return str(local)


def _drive_manifest() -> list:
    """Drive固定運転が要求する全部品 (Q4_0既定構成、v0.8.9)。
    リポ名はアダプタ定数から導出してドリフトを防ぐ。
    v0.9.12: quantをVIDEOLAB_DRIVE_QUANTで差し替え可能に (Q3_K_S運用等。
    AniSora High側の棚在庫はQ4_0/Q8_0のみなのでHighだけ別ノブ)。"""
    va = ADAPTERS.get("vace")
    an = ADAPTERS.get("anisora")
    rep = os.environ.get("VIDEOLAB_VACE_LORA_REPO",
                         "lightx2v/Wan2.2-Lightning")
    fold = os.environ.get("VIDEOLAB_VACE_LORA_DIR",
                          "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0")
    dq = os.environ.get("VIDEOLAB_DRIVE_QUANT", "Q4_0").strip() or "Q4_0"
    hq = (os.environ.get("VIDEOLAB_DRIVE_QUANT_HIGH", "").strip()
          or (dq if dq in ("Q4_0", "Q8_0") else "Q4_0"))
    return [
        ("file", va.gguf_repo,
         f"HighNoise/Wan2.2-VACE-Fun-A14B-high-noise-{dq}.gguf"),
        ("file", va.gguf_repo,
         f"LowNoise/Wan2.2-VACE-Fun-A14B-low-noise-{dq}.gguf"),
        ("file", an.gguf_repo, f"High/Index-Anisora-V3.2-High-{hq}.gguf"),
        ("file", an.gguf_repo, f"Low/Index-Anisora-V3.2-Low-{dq}.gguf"),
        ("file", rep, f"{fold}/high_noise_model.safetensors"),
        ("file", rep, f"{fold}/low_noise_model.safetensors"),
        ("snap", va.repo, None),
        ("snap", an.base_repo, None),
    ]


def drive_cache_ready() -> tuple:
    """(全部品が揃っているか, 不足リスト)。LoRAは1.2GBなので1GiB閾値、
    それ以外のGGUFも同閾値 (エラーページ等のゴミを完備と誤認しない)。"""
    dc = _drive_cache_dir()
    if dc is None:
        return False, ["(Drive未マウント)"]
    missing = []
    for kind, repo, fname in _drive_manifest():
        if kind == "file":
            p = dc / repo.replace("/", "--") / fname
            if not (p.is_file() and p.stat().st_size > _MIN_CACHE_BYTES):
                missing.append(f"{repo}/{fname}")
        elif not (dc / "_snap" / repo.replace("/", "--")
                  / ".complete").is_file():
            missing.append(f"_snap/{repo}")
    return not missing, missing


def populate_drive(log=print, wait_mount_secs: int = 180) -> bool:
    """初回セットアップ: Driveに不足しているモデルをHFから一度だけ取得して
    配置する (ノートのセル3.5)。**配置済みなら数秒でスキップ**するので、
    ⚡自動運転のRun Allで毎回実行されても安全 (2026-07-14ユーザー要件
    「DL済みならやらない感じの」)。取得中だけDrive固定を解除し、
    _hf_download/_snapshot_localの自動書き戻しがDriveへ配置する。"""
    deadline = time.time() + wait_mount_secs
    while _drive_cache_dir() is None and time.time() < deadline:
        time.sleep(3)   # セル2のマウンドスレッド完了待ち
    if _drive_cache_dir() is None:
        log("⚠ Driveが未マウントです — セル2の認可ポップアップに「許可」して"
            "から、このセルを再実行してください")
        return False
    ok, missing = drive_cache_ready()
    if ok:
        log("モデル配置済み — セットアップをスキップ (Drive固定運転OK)")
        return True
    log(f"Driveに未配置のモデル {len(missing)}件 — HFから一度だけ取得して"
        "配置します (初回のみ・30〜60分。以後のセッションはスキップ)")
    dc = _drive_cache_dir()
    prev = os.environ.pop("VIDEOLAB_DRIVE_ONLY", None)
    try:
        for kind, repo, fname in _drive_manifest():
            if kind == "file":
                p = dc / repo.replace("/", "--") / fname
                if p.is_file() and p.stat().st_size > _MIN_CACHE_BYTES:
                    continue
                _hf_download(repo, fname, log)
            else:
                if (dc / "_snap" / repo.replace("/", "--")
                        / ".complete").is_file():
                    continue
                _snapshot_local(repo, log)
    finally:
        if prev is not None:
            os.environ["VIDEOLAB_DRIVE_ONLY"] = prev
    ok, missing = drive_cache_ready()
    log("配置完了 — Drive固定運転OK" if ok
        else f"⚠ 配置しきれませんでした (HF側の渋滞の可能性): {missing} — "
             "このセルの再実行で続きから再開できます")
    return ok


def _repo_cache_dir(repo: str) -> Path:
    hub = Path(os.environ.get("HF_HOME",
                              str(Path.home() / ".cache" / "huggingface"))) / "hub"
    return hub / ("models--" + repo.replace("/", "--"))


def _repo_cache_gb(repo: str) -> float:
    """リポのHFキャッシュ実サイズ(GB)。snapshots/ はblobsへのsymlinkなので
    実体のみ数える(二重カウント防止)。"""
    d = _repo_cache_dir(repo)
    if not repo or not d.is_dir():
        return 0.0
    try:
        return sum(f.lstat().st_size for f in d.rglob("*")
                   if f.is_file() and not f.is_symlink()) / 2**30
    except Exception:
        return 0.0


def _purge_model_cache(repo: str, log):
    """指定リポのHFキャッシュを削除してディスクを空ける(Colabの容量対策)。

    VIDEOLAB_PURGE_ON_SWITCH=1 のとき、モデル切替でアンロードした側に適用。
    切替で戻ると再ダウンロードになるが、Colabの~200GBディスクでは
    60GB級モデルを2つ置いた時点で枯渇するため削除優先。
    """
    if not repo:
        return
    d = _repo_cache_dir(repo)
    if d.is_dir():
        sz = _repo_cache_gb(repo)
        shutil.rmtree(d, ignore_errors=True)
        log(f"ディスク解放: {repo} のキャッシュ削除 ({sz:.0f}GB)")


def _can_keep_cache(next_adapter, log) -> bool:
    """モデル切替時、前モデルのHFキャッシュを温存できるか判定する。

    従来は無条件削除だったが、anisora(45GB)⇔vace(70GB)を方向別に交互運用
    すると切替のたびに数十GBを再DLしてしまう。次モデルの未DL分を差し引いて
    も空きが残る(既定25GB、VIDEOLAB_PURGE_KEEP_FREE_GB)なら温存する。"""
    try:
        repos = (getattr(next_adapter, "cache_repos", None)
                 or [getattr(next_adapter, "repo", "")])
        cached = sum(_repo_cache_gb(r) for r in repos if r)
        need = max(0.0, float(getattr(next_adapter, "disk_gb", 60)) - cached)
        free = shutil.disk_usage(Path.home()).free / 2**30
        margin = float(os.environ.get("VIDEOLAB_PURGE_KEEP_FREE_GB", "25"))
        log(f"ディスク判定: 空き{free:.0f}GB / 次モデルの未DL分 約{need:.0f}GB")
        return free - need >= margin
    except Exception:
        return False


# ------------------------------------------------ mock: GPU不要の疎通確認用
@register
class MockAdapter(VideoAdapter):
    id = "mock"
    label = "Mock (疎通テスト・GPU不要)"
    desc = ("実モデルを使わず合成アニメを返す。webUI・SpriteMill接続・"
            "トンネルの疎通確認用。")
    requires = "ffmpeg のみ"
    defaults = {"num_frames": 49, "fps": 24, "steps": 10}

    def generate(self, req: GenRequest, workdir: Path, log, progress) -> Path:
        from PIL import Image, ImageDraw
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            raise RuntimeError("ffmpeg が見つかりません (PATH か VIDEOLAB_FFMPEG を設定)")
        frames_dir = workdir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        w, h, n = req.width, req.height, req.num_frames
        keys = req.images
        # キーフレーム位置(未指定なら等間隔)
        pos = req.key_positions or (
            [i / max(1, len(keys) - 1) for i in range(len(keys))] if len(keys) > 1
            else [0.0] * len(keys))
        log(f"mock: {w}x{h} x{n}f, keys={len(keys)}")
        for i in range(n):
            t = i / max(1, n - 1)
            if keys:
                # キーフレーム間クロスフェード(multikey の位置指定を目視確認できる)
                j = 0
                while j + 1 < len(pos) and pos[j + 1] <= t:
                    j += 1
                a = keys[j].resize((w, h))
                if j + 1 < len(keys) and pos[j + 1] > pos[j]:
                    b = keys[j + 1].resize((w, h))
                    mix = (t - pos[j]) / (pos[j + 1] - pos[j])
                    frame = Image.blend(a, b, max(0.0, min(1.0, mix)))
                else:
                    frame = a
            else:
                # t2v: プロンプト文字列とグラデーションだけの合成映像
                frame = Image.new("RGB", (w, h),
                                  (int(40 + 60 * t), 40, int(90 - 50 * t)))
            d = ImageDraw.Draw(frame)
            cx = int(w * 0.1 + (w * 0.8) * t)
            cy = int(h * 0.5 + h * 0.25 * __import__("math").sin(t * 12.56))
            d.ellipse([cx - 14, cy - 14, cx + 14, cy + 14],
                      fill=(255, 210, 60), outline=(20, 20, 20), width=3)
            d.text((8, 8), f"MOCK {i + 1}/{n}  {req.prompt[:60]}", fill=(255, 255, 255))
            frame.save(frames_dir / f"{i:05d}.png")
            if (i + 1) % 10 == 0 or i == n - 1:
                progress((i + 1) / n * 0.9)
            time.sleep(0.02)   # 進捗表示の動作確認用に少しだけ遅くする
        dest = workdir / "out.mp4"
        encode_mp4(ffmpeg, frames_dir, req.fps, dest)
        progress(1.0)
        return dest


# ------------------------------------------------ 拡散モデル共通ヘルパ
def _snap(v: int, mult: int, lo: int) -> int:
    return max(lo, int(round(v / mult)) * mult)


def _snap_min_short_edge(width: int, height: int, short: int = 480,
                         mult: int = 16) -> tuple[int, int]:
    """縦横比を保ちつつ短辺の下限を保証し、両辺をモデル倍数へ丸める。"""
    w = _snap(width, mult, mult)
    h = _snap(height, mult, mult)
    if min(w, h) >= short:
        return w, h
    if w <= h:
        return _snap(short, mult, short), _snap(h * short / w, mult, short)
    return _snap(w * short / h, mult, short), _snap(short, mult, short)


def _snap_frames(n: int) -> int:
    """LTX/Wan 系のフレーム数制約 8k+1 に丸める。"""
    return max(9, ((int(n) - 1) // 8) * 8 + 1)


def _pick_dtype():
    import torch
    # bf16はAmpere(sm80)以上のネイティブ対応のみ採用。T4(Turing/sm75)は
    # is_bf16_supported()がエミュレーション込みでTrueを返す版があり、
    # bf16のまま走ると激遅/一部カーネル不発になる (2026-07-16「T4で
    # 生成できる?」対応: fp16へ落とし、VAEは_finalize_pipeがfp32へ上げて
    # 数値安全を取る。UMT5のfp16はT5系の桁あふれリスクがあるが16GB級に
    # fp32は載らないためWan系コミュニティ実績に合わせて許容)
    try:
        if torch.cuda.get_device_capability()[0] >= 8:
            return torch.bfloat16
        return torch.float16
    except Exception:
        return (torch.bfloat16 if torch.cuda.is_bf16_supported()
                else torch.float16)


def _fit_image(img, w: int, h: int):
    """縦横比を維持して (w,h) に収める。余白は四隅の色でパディング。

    単純な resize((w,h)) は比率が違う入力を引き伸ばす(キャラが縦長になる
    実障害 2026-07-12)。スプライト用途は背景が単色(マゼンタ等)なので、
    四隅の色で letterbox するのが安全。"""
    from PIL import Image
    iw, ih = img.size
    if (iw, ih) == (w, h):
        return img
    sc = min(w / iw, h / ih)
    nw, nh = max(1, round(iw * sc)), max(1, round(ih * sc))
    resized = img.resize((nw, nh), Image.LANCZOS)
    bg = Image.new("RGB", (w, h), img.convert("RGB").getpixel((0, 0)))
    bg.paste(resized, ((w - nw) // 2, (h - nh) // 2))
    return bg


def _require_deps(log):
    """実モデルに必要なライブラリのバージョン検査。

    「WanImageToVideoPipeline がない」= 古い diffusers が既に入っている
    PCで起きた実障害(2026-07-12)への対策。足りない場合は対処コマンド
    入りの日本語エラーにして webUI のジョブログに出す。"""
    def _tup(s):
        out = []
        for x in str(s).split("+")[0].split(".")[:3]:
            out.append(int(x) if x.isdigit() else 0)
        return tuple(out)
    try:
        import diffusers
    except ImportError:
        raise RuntimeError(
            'diffusers が入っていません。pip install -U "diffusers>=0.39.0" '
            "transformers accelerate safetensors sentencepiece ftfy gguf peft "
            "を実行してからサーバを再起動してください")
    v = getattr(diffusers, "__version__", "0")
    if _tup(v) < (0, 39, 0):
        raise RuntimeError(
            f"diffusers {v} は古すぎます (0.39.0 以上が必要。WanImageToVideo"
            f'Pipeline / LTX2系はこの版から)。pip install -U "diffusers>='
            f'0.39.0" を実行してからサーバを再起動してください')
    import torch
    tv = getattr(torch, "__version__", "0")
    if _tup(tv) < (2, 4, 0):
        raise RuntimeError(
            f"torch {tv} は古すぎます (2.4 以上推奨)。pip install -U torch "
            f"--index-url https://download.pytorch.org/whl/cu130 を実行して"
            f"からサーバを再起動してください")
    log(f"deps OK: diffusers {v} / torch {tv}")


def _apply_offload(pipe, log):
    """VRAM 量でオフロード戦略を選ぶ。VAE tiling は常時有効(OOM対策の定石)。"""
    import torch
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    if total >= 60:
        pipe.enable_model_cpu_offload()
        log(f"VRAM {total:.0f}GB: model_cpu_offload")
    else:
        pipe.enable_sequential_cpu_offload()
        log(f"VRAM {total:.0f}GB: sequential_cpu_offload (低VRAM・低速だが確実)")
    try:
        pipe.vae.enable_tiling()
    except Exception:
        pass


def _frames_to_mp4(frames, fps: int, workdir: Path, log) -> Path:
    """パイプライン出力(np配列 or PIL列)を連番PNG->mp4 に。音声は使わないので破棄。"""
    import numpy as np
    from PIL import Image
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg が見つかりません")
    fdir = workdir / "frames"
    fdir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        if isinstance(f, np.ndarray):
            arr = f
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0.0, 1.0) * 255).round().astype("uint8")
            img = Image.fromarray(arr)
        else:
            img = f
        img.save(fdir / f"{i:05d}.png")
    dest = workdir / "out.mp4"
    encode_mp4(ffmpeg, fdir, fps, dest)
    log(f"{len(frames)}コマ -> {dest.name}")
    return dest


def _step_callback(progress, steps: int):
    """diffusers の callback_on_step_end 形式。進捗更新+キャンセル検知。"""
    def cb(pipe, step_index, timestep, callback_kwargs):
        progress(0.05 + 0.85 * (step_index + 1) / max(1, steps))
        return callback_kwargs
    return cb


def _call_with_optional_kwargs(fn, kwargs: dict, optional: list, log):
    """古い diffusers 版が未対応の引数は落として再試行する保険。"""
    kw = dict(kwargs)
    for _ in range(len(optional) + 1):
        try:
            return fn(**kw)
        except TypeError as e:
            hit = next((o for o in optional if o in str(e) and o in kw), None)
            if not hit:
                raise
            kw.pop(hit)
            log(f"引数 {hit} は未対応のため外して再試行")
    return fn(**kw)


def _latent_pin_slots(frames, t_dim: int) -> list:
    """動画フレーム番号 -> Wan VAEのlatent時間スロット番号。

    Wanの時間圧縮は 1+4+4+…: slot0=frame0、slot s>=1 が frames 4s-3..4s
    を担う。frame0はI2Vの条件画像側で既に固定されるため slot0 は対象外
    (条件注入と喧嘩させない)。範囲外・重複・非数は黙って落とす。"""
    out = set()
    for f in (frames or []):
        try:
            fi = int(f)
        except (TypeError, ValueError):
            continue
        if fi <= 0:
            continue
        s = (fi - 1) // 4 + 1
        if 0 < s < int(t_dim):
            out.add(s)
    return sorted(out)


def _pin_step_callback(inner, sched, x0, noise, slots: list, release: float,
                       smask=None, smask_low=None,
                       smask_release_last_steps=None):
    """リファインの毎step後、固定スロットを (1-σ)x0 + σε へ描き戻す。

    SDEdit再デノイズは全フレームを自由に動かすため、歩行周期の位相が
    stage1からわずかに流れて「先頭=終端同位相」(コマ選出の前提) が崩れる。
    描き戻しは初期化と同じ noise を使い、固定スロットにflow matchingの
    直線補間路そのものを歩かせる — モデルから見て軌道上の点なので予測が
    暴れず、終端σ=0で厳密にstage1へ着地する。release>0 ならσ<release の
    終盤stepは描き戻しを止め、質感の馴染ませに開放する。

    smask (2026-07-21ユーザー発案「固定すべきところはマスク済みなら
    AniSoraのみで完結できるのでは」): 空間方向のlatent固定 (RePaint式)。
    bool tensor [Hlat, Wlat] (全時刻共通) または [Tlat,Hlat,Wlat]
    (時刻別)、True=固定。VACEのマスク条件付けに依存しない
    インペイントがstage2単体で成立する。

    smask_lowを渡した場合、smask_release_last_steps=Nなら末尾N stepだけ
    smask_lowへ切り替える。0なら終端までsmaskを維持する。値を省略した旧依頼は
    Wan2.2のboundary_ratioでHigh=smask / Low=smask_lowを選ぶ互換動作。"""
    def cb(pipe, step_index, timestep, callback_kwargs):
        callback_kwargs = (inner(pipe, step_index, timestep, callback_kwargs)
                           or callback_kwargs)
        lat = callback_kwargs.get("latents")
        if lat is None:
            return callback_kwargs
        sig = sched.sigmas          # 切詰め済み(尻尾+終端σ)
        s = float(sig[min(step_index + 1, len(sig) - 1)])
        if s < release:
            return callback_kwargs
        ref = (1.0 - s) * x0 + s * noise
        for sl in slots:
            lat[:, :, sl] = ref[:, :, sl].to(lat.device, lat.dtype)
        stage_mask = smask
        if smask_low is not None:
            if smask_release_last_steps is not None:
                try:
                    total_steps = len(sched.timesteps)
                    release_steps = max(0, min(
                        total_steps, int(smask_release_last_steps)))
                    if (release_steps > 0
                            and step_index >= total_steps - release_steps):
                        stage_mask = smask_low
                except (TypeError, ValueError):
                    # 不正値は安全側=頭部を含む全固定を維持。
                    stage_mask = smask
            else:
                try:
                    boundary_ratio = float(pipe.config.boundary_ratio)
                    boundary_t = (boundary_ratio
                                  * float(sched.config.num_train_timesteps))
                    if float(timestep) < boundary_t:
                        stage_mask = smask_low
                except (AttributeError, TypeError, ValueError):
                    # boundary情報が無い旧pipeでは安全側=全固定を維持。
                    stage_mask = smask
        if stage_mask is not None:
            import torch as _t
            m = stage_mask.to(lat.device)
            if m.ndim == 2:
                m = m.view(1, 1, 1, *m.shape)
            elif m.ndim == 3:
                if m.shape[0] != lat.shape[2]:
                    raise ValueError("時間可変maskのT次元がlatentと不一致")
                m = m.view(1, 1, *m.shape)
            else:
                raise ValueError("spatial maskは[H,W]または[T,H,W]が必要")
            refl = ref.to(lat.device, lat.dtype)
            lat = _t.where(m, refl, lat)
        callback_kwargs["latents"] = lat
        return callback_kwargs
    return cb


def _spatial_inpaint_latents(x0, noise, sigma: float, fixed_mask=None,
                             empty_generate: bool = False,
                             generate_source_mix: float = 0.0):
    """AniSoraリファインの初期潜在を作る。

    fixed_maskは [Hlat,Wlat] または [Tlat,Hlat,Wlat] bool、True=固定。通常は従来の
    SDEdit初期化。empty_generate=Trueの空間インペイントでは、
    生成領域に x0 を一切混ぜず純ノイズを置く。固定領域は
    flow-matchingの参照軌道 (1-σ)x0+σε から始める。"""
    import torch
    s = float(sigma)
    ref = (1.0 - s) * x0 + s * noise
    if not empty_generate or fixed_mask is None:
        return ref
    m = fixed_mask.to(x0.device)
    if m.ndim == 2:
        m = m.view(1, 1, 1, *m.shape)
    elif m.ndim == 3:
        if m.shape[0] != x0.shape[2]:
            raise ValueError("時間可変maskのT次元がlatentと不一致")
        m = m.view(1, 1, *m.shape)
    else:
        raise ValueError("spatial maskは[H,W]または[T,H,W]が必要")
    mix = max(0.0, min(0.5, float(generate_source_mix)))
    generated = noise if mix <= 0 else mix * x0 + (1.0 - mix) * noise
    return torch.where(m, ref, generated)


def _pixel_lock_spatial_frames(frames, reference, generate_mask):
    """デコード後もマスク外を参照画素へ戻す。

    latent固定だけではVAEの受容野跨ぎで顔/背景が数画素流れる
    余地がある。標準的なインペイントと同じく、0=固定側を
    最後に元画像で合成し、凍結の意味を画素側でも保証する。"""
    import numpy as np
    from PIL import Image
    refs = ([reference] if isinstance(reference, Image.Image)
            else list(reference))
    masks = ([generate_mask]
             if isinstance(generate_mask, (Image.Image, np.ndarray))
             else list(generate_mask))
    if len(refs) not in (1, len(frames)) or len(masks) not in (1, len(frames)):
        raise ValueError("pixel lockの参照/mask枚数が動画フレーム数と不一致")
    locked = []
    for i, frame in enumerate(frames):
        ref = refs[0 if len(refs) == 1 else i].convert("RGB")
        size = ref.size
        raw_mi = masks[0 if len(masks) == 1 else i]
        if isinstance(raw_mi, Image.Image):
            mi = raw_mi.convert("L")
        else:
            mi = Image.fromarray(
                np.asarray(raw_mi).astype("uint8"), "L")
        if mi.size != size:
            mi = mi.resize(size, resample=Image.Resampling.NEAREST)
        keep = np.asarray(mi) < 128
        ref_u8 = np.asarray(ref, dtype=np.uint8)
        arr = np.asarray(frame).copy()
        if arr.shape[:2] != (size[1], size[0]):
            raise ValueError("pixel lockのフレーム寸法が参照と不一致")
        ref_arr = (ref_u8 if arr.dtype == np.uint8
                   else ref_u8.astype(arr.dtype) / 255.0)
        arr[keep] = ref_arr[keep]
        locked.append(arr)
    return locked


# ------------------------------------------------ LTX-2.3 (diffusers)
class _LTX2Base(VideoAdapter):
    """LTX-2 系共通実装。t2v/i2v/multikey すべて LTX2ConditionPipeline で賄う。

    - 解像度は 32 の倍数、フレーム数は 8k+1 に自動で丸める。
    - multikey: LTX2VideoCondition(frames=画像, index=latent位置) のリスト。
      index は latent 単位(実フレーム8枚ぶん)なので位置は 8 フレーム粒度。
      条件画像の周辺8フレームは静止しやすい(公式ドキュメント記載)ため、
      strength を extra {"cond_strength": 0.9} 等で下げる実験ができる。
    - 音声も同時生成されるが破棄する。
    - 2段階生成(latent upsample)は品質向上の余地として未実装(README参照)。
    """
    repo = ""            # サブクラスで指定
    distilled = True
    disk_gb = 46         # bf16重みの目安 (キャッシュ温存判定用)

    def __init__(self):
        super().__init__()
        self.pipe = None

    def ensure_loaded(self, log):
        _require_deps(log)
        import torch
        from diffusers import LTX2ConditionPipeline
        log(f"読み込み開始: {self.repo} (bf16・初回は数十GBのDL)")
        self.pipe = LTX2ConditionPipeline.from_pretrained(
            self.repo, torch_dtype=_pick_dtype())
        _apply_offload(self.pipe, log)
        self.loaded = True

    def unload(self, log):
        self.pipe = None
        self.loaded = False

    def generate(self, req: GenRequest, workdir: Path, log, progress) -> Path:
        import torch
        from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
        try:
            from diffusers.pipelines.ltx2.utils import (DEFAULT_NEGATIVE_PROMPT,
                                                        DISTILLED_SIGMA_VALUES)
        except ImportError:
            DEFAULT_NEGATIVE_PROMPT, DISTILLED_SIGMA_VALUES = "", None

        w = _snap(req.width, 32, 256)
        h = _snap(req.height, 32, 256)
        n = _snap_frames(req.num_frames)
        if (w, h, n) != (req.width, req.height, req.num_frames):
            log(f"制約丸め: {req.width}x{req.height}x{req.num_frames} -> {w}x{h}x{n} (32px/8k+1)")

        conds = []
        if req.images:
            pos = req.key_positions or (
                [i / max(1, len(req.images) - 1) for i in range(len(req.images))]
                if len(req.images) > 1 else [0.0])
            strength = float(req.extra.get("cond_strength", 1.0))
            max_latent = (n - 1) // 8
            for img, p in zip(req.images, pos):
                idx = min(max_latent, round(p * max_latent))
                conds.append(LTX2VideoCondition(
                    frames=_fit_image(img, w, h), index=int(idx),
                    strength=strength))
                log(f"キーフレーム: pos={p:.2f} -> latent index {idx} (frame~{idx * 8})")

        steps = int(req.steps)
        kw = dict(prompt=req.prompt,
                  negative_prompt=req.negative or DEFAULT_NEGATIVE_PROMPT,
                  width=w, height=h, num_frames=n, frame_rate=float(req.fps),
                  num_inference_steps=steps,
                  guidance_scale=float(req.guidance),
                  generator=torch.Generator("cpu").manual_seed(req.seed),
                  output_type="np", return_dict=False,
                  callback_on_step_end=_step_callback(progress, steps))
        if conds:
            kw["conditions"] = conds
        if self.distilled and DISTILLED_SIGMA_VALUES is not None:
            kw["sigmas"] = DISTILLED_SIGMA_VALUES
            kw["num_inference_steps"] = len(DISTILLED_SIGMA_VALUES) - 1 \
                if steps == 8 else steps
        log(f"生成開始: {w}x{h} {n}f steps={kw['num_inference_steps']} "
            f"cfg={req.guidance} keys={len(conds)}")
        video, _audio = _call_with_optional_kwargs(
            self.pipe, kw, ["callback_on_step_end", "sigmas", "frame_rate"], log)
        progress(0.92)
        return _frames_to_mp4(list(video[0]), req.fps, workdir, log)


@register
class LTX23Distilled(_LTX2Base):
    id = "ltx23"
    label = "LTX-2.3 22B distilled (本命・高速8step)"
    desc = ("Lightricks LTX-2.3 の蒸留版。8ステップで高速。t2v/i2v/複数キーフレーム"
            "対応。キーフレーム位置は8フレーム粒度。")
    requires = "Colab A100推奨 (L4はRAM次第で低速動作)・bf16重み約46GB"
    repo = os.environ.get("VIDEOLAB_LTX23_REPO", "diffusers/LTX-2.3-Distilled-Diffusers")
    distilled = True
    defaults = {"width": 768, "height": 512, "num_frames": 121, "fps": 24,
                "steps": 8, "guidance": 1.0}


@register
class LTX23Dev(_LTX2Base):
    id = "ltx23dev"
    label = "LTX-2.3 22B dev (品質重視・40step)"
    desc = ("LTX-2.3 のフル版。蒸留版より高品質だが5倍遅い。"
            "guidance 3〜4 推奨。")
    requires = "Colab A100 推奨・bf16重み約46GB"
    repo = os.environ.get("VIDEOLAB_LTX23DEV_REPO", "diffusers/LTX-2.3-Diffusers")
    distilled = False
    defaults = {"width": 768, "height": 512, "num_frames": 121, "fps": 24,
                "steps": 30, "guidance": 3.0}


# ------------------------------------------------ 旧 LTX-Video 0.9.x (低VRAM)
@register
class LTX09Adapter(VideoAdapter):
    """旧世代 LTX-Video 0.9.x。13Bで軽く、fp8化で約10GB — 無料T4でも視野。

    diffusers の LTXConditionPipeline は LTXVideoCondition(image=, frame_index=)
    で「任意フレーム位置」への複数キーフレーム条件付けを公式サポートしており、
    LTX-2系(8フレーム粒度)より細かい位置指定ができる。
    """
    id = "ltx098"
    label = "LTX-Video 0.9.7 distilled 13B (低VRAM枠)"
    desc = ("旧世代だが軽量。fp8化で約10GB=Colab T4でも視野。任意フレーム位置への"
            "キーフレーム条件付け対応。")
    requires = "VRAM 10〜16GB (T4/L4/ローカル12GB+)"
    repo = os.environ.get("VIDEOLAB_LTX098_REPO", "Lightricks/LTX-Video-0.9.7-distilled")
    disk_gb = 30
    defaults = {"width": 704, "height": 480, "num_frames": 97, "fps": 24,
                "steps": 8, "guidance": 1.0}

    def __init__(self):
        super().__init__()
        self.pipe = None

    def ensure_loaded(self, log):
        _require_deps(log)
        import torch
        from diffusers import LTXConditionPipeline
        log(f"読み込み開始: {self.repo}")
        self.pipe = LTXConditionPipeline.from_pretrained(
            self.repo, torch_dtype=_pick_dtype())
        # 低VRAM向け: fp8 レイヤーワイズキャスト(対応環境のみ・失敗しても続行)
        try:
            total = torch.cuda.get_device_properties(0).total_memory / 2**30
            if total < 20:
                self.pipe.transformer.enable_layerwise_casting(
                    storage_dtype=torch.float8_e4m3fn, compute_dtype=_pick_dtype())
                log("fp8 layerwise casting 有効 (VRAM<20GB)")
        except Exception as e:
            log(f"fp8 casting 不可(続行): {e}")
        _apply_offload(self.pipe, log)
        self.loaded = True

    def unload(self, log):
        self.pipe = None
        self.loaded = False

    def generate(self, req: GenRequest, workdir: Path, log, progress) -> Path:
        import torch
        from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition

        w = _snap(req.width, 32, 256)
        h = _snap(req.height, 32, 256)
        n = _snap_frames(req.num_frames)
        conds = []
        if req.images:
            pos = req.key_positions or (
                [i / max(1, len(req.images) - 1) for i in range(len(req.images))]
                if len(req.images) > 1 else [0.0])
            for img, p in zip(req.images, pos):
                fi = min(n - 1, int(round(p * (n - 1) / 8)) * 8)  # 8の倍数推奨
                conds.append(LTXVideoCondition(image=_fit_image(img, w, h),
                                               frame_index=fi))
                log(f"キーフレーム: pos={p:.2f} -> frame_index {fi}")
        steps = int(req.steps)
        kw = dict(prompt=req.prompt, negative_prompt=req.negative or None,
                  width=w, height=h, num_frames=n,
                  num_inference_steps=steps,
                  guidance_scale=float(req.guidance),
                  generator=torch.Generator("cpu").manual_seed(req.seed),
                  output_type="np", return_dict=False,
                  callback_on_step_end=_step_callback(progress, steps))
        if conds:
            kw["conditions"] = conds
        log(f"生成開始: {w}x{h} {n}f steps={steps} keys={len(conds)}")
        out = _call_with_optional_kwargs(
            self.pipe, kw, ["callback_on_step_end"], log)
        video = out[0]
        progress(0.92)
        return _frames_to_mp4(list(video[0]), req.fps, workdir, log)


# ------------------------------------------------ Wan 2.2 TI2V-5B (比較枠)
@register
class Wan22Adapter(VideoAdapter):
    """Wan2.2 TI2V-5B。i2v の同一性・動作一貫性の評判が高い比較用モデル。

    キーフレーム条件付けは無いので i2v のみ(multikey ジョブには使えない)。
    """
    id = "wan22"
    label = "Wan 2.2 TI2V-5B (品質比較枠・i2vのみ)"
    desc = ("Alibaba Wan2.2 の 5B 版。キャラ同一性の維持に定評。単一画像 i2v のみ。"
            "Apache 2.0。")
    requires = "VRAM 24GB (RTX4090/L4/A100)・オフロードで低VRAM可"
    modes = ("i2v",)
    repo = os.environ.get("VIDEOLAB_WAN22_REPO", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    disk_gb = 30
    defaults = {"width": 704, "height": 704, "num_frames": 81, "fps": 24,
                "steps": 40, "guidance": 5.0}

    def __init__(self):
        super().__init__()
        self.pipe = None

    def ensure_loaded(self, log):
        _require_deps(log)
        from diffusers import WanImageToVideoPipeline
        log(f"読み込み開始: {self.repo}")
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            self.repo, torch_dtype=_pick_dtype())
        _apply_offload(self.pipe, log)
        self.loaded = True

    def unload(self, log):
        self.pipe = None
        self.loaded = False

    def generate(self, req: GenRequest, workdir: Path, log, progress) -> Path:
        import torch
        w = _snap(req.width, 32, 256)
        h = _snap(req.height, 32, 256)
        n = max(5, ((int(req.num_frames) - 1) // 4) * 4 + 1)   # Wan は 4k+1
        steps = int(req.steps)
        img = _fit_image(req.images[0], w, h)
        kw = dict(image=img, prompt=req.prompt,
                  negative_prompt=req.negative or None,
                  width=w, height=h, num_frames=n,
                  num_inference_steps=steps,
                  guidance_scale=float(req.guidance),
                  generator=torch.Generator("cpu").manual_seed(req.seed),
                  output_type="np", return_dict=False,
                  callback_on_step_end=_step_callback(progress, steps))
        log(f"生成開始: {w}x{h} {n}f steps={steps}")
        out = _call_with_optional_kwargs(
            self.pipe, kw, ["callback_on_step_end"], log)
        video = out[0]
        progress(0.92)
        return _frames_to_mp4(list(video[0]), req.fps, workdir, log)


# ------------------------------------------------ Wan 2.2 A14B 系 共通
class _WanA14BBase(VideoAdapter):
    """Wan2.2 I2V-A14B (MoE 2エキスパート) 系の共通実装。i2v専用。

    - transformer=高ノイズ / transformer_2=低ノイズ。boundary_ratio=0.9 で
      パイプラインが自動切替するため手動制御は不要。
    - フレーム数は 4k+1 (81既定)、解像度16の倍数、fps=16、shift=5。
    """
    modes = ("i2v",)
    base_repo = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    flow_shift = 5.0
    prompt_suffix = ""

    def __init__(self):
        super().__init__()
        self.pipe = None
        self._te_lazy = False    # 低RAM VM: UMT5を持たずにロードしたか
        self._te_from = None     # UMT5遅延ロード元リポ
        self._offload_mode = ""  # _finalize_pipeが確定した実効モード
        self._dit_gb = 0.0       # DiT1体の実サイズ (admission用)

    def unload(self, log):
        self.pipe = None
        self.loaded = False

    def ensure_loaded(self, log):
        """ロード失敗時の部分pipe残留を根絶する共通ガード (P0-1)。

        途中まで構築されたpipe (DiT数GB) が残ったまま次のジョブが再ロード
        すると二重常駐になり、RAM OOM=VM killに至る (2026-07-16調査で
        現存経路と確認: LoRA適用失敗・DL途中エラー→リトライの型)。"""
        try:
            self._ensure_loaded_impl(log)
        except Exception:
            try:
                self.unload(log)
            except Exception:
                pass
            try:
                import gc
                gc.collect()
            except Exception:
                pass
            _free_cuda(log)
            raise

    def _ensure_loaded_impl(self, log):
        raise NotImplementedError

    def _admit(self, w: int, h: int, n: int, log, tag: str = "") -> None:
        """生成直前VRAMゲート (P0-3) のアダプタ向け窓口。実効オフロード
        モードから常駐余地と降格手段を組み立てて _admit_vram へ渡す。"""
        mode = getattr(self, "_offload_mode", "")
        _admit_vram(
            w, h, n, log,
            resident_extra_gb=((getattr(self, "_dit_gb", 0.0) or 9.3)
                               if mode == "model"
                               else (0.5 if mode in ("block", "seq")
                                     else 0.0)),
            downgrade=((lambda: self._downgrade_to_block(log))
                       if mode in ("cuda", "model") else None),
            tag=tag or self.id)

    def _downgrade_to_block(self, log) -> bool:
        """常駐/モデルオフロード構成をblock offloadへ組み替える (P0-3の
        自動降格)。成功したらTrue。失敗時はモードを変えずFalse (admission
        側が元の常駐余地で再判定する — 無条件block化は偽PASS→生成中OOM
        の温床、レビュー指摘で修正 v0.9.13)。

        注意: 降格は片道 (このセッションの以後のジョブもblock速度)。戻す
        には extra.offload を明示して積み替える。plan_canvasが先に寸法を
        身の丈へ落とすため、ここへ来るのは想定外サイズのジョブだけ。"""
        pipe = self.pipe
        if pipe is None:
            return False
        # model_cpu_offload由来のaccelerateフックはgroup offloadと競合する
        # (近年版diffusersは適用拒否・旧版は.to禁止例外) — 先に剥がす
        try:
            pipe.remove_all_hooks()
        except Exception:
            pass
        seen = set()
        ok = True
        for name in ("transformer", "transformer_2"):
            m = getattr(pipe, name, None)
            if m is None or id(m) in seen:      # liteモードはHigh=Lowの別名
                continue
            seen.add(id(m))
            if self._group_offloaded(m):
                continue
            try:
                m.to("cpu")
            except Exception:
                pass
            if not self._try_group_offload(m, log, f"{name}(自動降格)"):
                ok = False
                try:
                    m.to("cuda")     # CPU取り残し=device不一致連鎖を防ぐ
                except Exception:
                    pass
        if not ok:
            log("自動降格に失敗 — 現構成のまま再判定します")
            _free_cuda(log)
            return False
        # フック剥がしでCPUに残った非DiT部品 (VAE/UMT5等) をGPUへ
        # (_finalize_pipeのblock分岐と同じ配置)
        try:
            import torch as _t
            for cname, comp in pipe.components.items():
                if cname in ("transformer", "transformer_2"):
                    continue
                if isinstance(comp, _t.nn.Module):
                    comp.to("cuda")
        except Exception:
            pass
        try:
            pipe.vae.enable_tiling()
        except Exception:
            pass
        _free_cuda(log)
        self._offload_mode = "block"
        log("自動降格完了: 以後のジョブもblock運転になります "
            "(戻すには offload を明示指定して積み替え)")
        return True

    @staticmethod
    def _try_group_offload(model, log, label) -> bool:
        """extra.offload=block のblock単位ストリーミング退避。

        16GB未満級のご家庭GPUで8方向まとめ生成をオミットしないための
        手段 (2026-07-13方針)。常駐9GB -> 約2GB+転送に置き換わる。
        GGUF×enable_sequential_cpu_offloadはdiffusersの既知バグ
        (quant_type喪失のKeyError)で使えないため、GGUFの省メモリは
        こちらが本線 (v0.7系で実GPU検証済み: 720x1296/49fピーク11.3GB)。
        適用に失敗したら従来経路へ戻す。"""
        import torch
        kw = dict(onload_device=torch.device("cuda"),
                  offload_device=torch.device("cpu"),
                  offload_type="block_level",
                  num_blocks_per_group=1,
                  use_stream=True)
        # RAMが少ないVM (T4=51GB級) では退避先をディスクへ。block offload
        # はDiT2体(約17GB)をRAM常駐させるため、L4で33GBだったロードピーク
        # が T4では 33+17≒50GB になりVMごとOOM killされる (2026-07-16
        # 実障害「RAMに50GBくらい読み込んで落ちる」)。/tmpはローカルSSD
        # なので転送は遅くなるが生存が最優先。旧diffusersは引数ごと
        # フォールバック
        _disk = None
        if _low_ram_vm(60.0):
            # モデルごとに別dir (同居させるとblocks.0等のファイル名が衝突)
            _disk = str(WORK_ROOT / "_offload"
                        / f"{abs(hash(label)) & 0xffffff:06x}")
            # ★既存dirは必ず消してから使う (v0.9.14): diffusersのディスク
            # 退避は group_*.safetensors が存在すると書き込みをスキップし
            # 「前回ロードしたモデルのバイト列」を無言で読む (group_
            # offloading.py の既存ファイル早期スキップ)。同一プロセスで
            # quantを切り替えると前quantの重みを掴む — 2026-07-17実障害:
            # Q4_0→Q3_K_S切替でQ3_KラベルのままQ4_0バイト(5120x5120=
            # 14,745,600B)を復元し「shape [134050,110] invalid」で確定
            # クラッシュ。サイズが偶然同じ組合せなら無言の重み化けになる
            # ため、退避ファイルはロード毎に作り直す (書き直しコストは
            # 数十秒・正しさ優先)。旧プロセスの残骸dirも同時に掃除される
            try:
                if os.path.isdir(_disk):
                    shutil.rmtree(_disk, ignore_errors=True)
                    log(f"{label}: 前回モデルのディスク退避キャッシュを削除 "
                        "(quant切替の取り違え防止)")
            except Exception:
                pass
            kw["offload_to_disk_path"] = _disk
        try:
            try:
                model.enable_group_offload(**kw)
            except TypeError:
                if "offload_to_disk_path" not in kw:
                    raise
                kw.pop("offload_to_disk_path")   # 旧diffusers非対応
                _disk = None
                model.enable_group_offload(**kw)
            log(f"{label}: block offload適用 "
                + (f"(重みディスク退避: {_disk} — 低RAM VM対策)" if _disk
                   else "(重みCPU常駐+ブロック転送)"))
            return True
        except Exception as e:      # noqa: BLE001
            log(f"{label}: block offload不可 -> 従来経路へ ({str(e)[:120]})")
            return False

    @staticmethod
    def _group_offloaded(model) -> bool:
        """グループオフロードhookが載っているか。載っているモデルへの
        .to() はdiffusersが例外を出すため、手動退避の前に必ず確認する。"""
        try:
            from diffusers.hooks.group_offloading import (
                _get_group_onload_device)
            _get_group_onload_device(model)
            return True
        except Exception:
            return False

    def _encode_prompt_lazy(self, prompt, negative, guidance, log):
        """UMT5遅延運転 (低RAM VM): text_encoder=Noneでロードしたpipe用に
        UMT5をジョブ内で一時ロードしてprompt embedsを作り、即解放する。

        v0.8.2: handoff(v0.7.6)と同じ低RAM対策を素のvace/anisoraにも適用。
        L4 VM (RAM53GB) はUMT5(11GB)を常駐させたままモデル切替すると
        RAMスパイクでランタイムごとOOM killされる (2026-07-13実障害)。"""
        import gc
        import torch
        from transformers import UMT5EncoderModel
        pipe = self.pipe
        src = self._te_from or self.base_repo
        log("UMT5を一時ロード (低RAM運転: encode後に即解放・約30秒)")
        try:
            # low_cpu_mem_usage: 初期化+読込の二重確保を避けRAM山を半減
            # (T4=51GB VMのanisora段RAM黄色対策の一部 2026-07-16)
            te = UMT5EncoderModel.from_pretrained(
                src, subfolder="text_encoder", torch_dtype=_pick_dtype(),
                low_cpu_mem_usage=True)
        except TypeError:
            te = UMT5EncoderModel.from_pretrained(
                src, subfolder="text_encoder", torch_dtype=_pick_dtype())
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.text_encoder = te
        try:
            te.to(dev)
            with torch.no_grad():
                pe, ne = pipe.encode_prompt(
                    prompt=prompt, negative_prompt=negative,
                    do_classifier_free_guidance=float(guidance) > 1.0,
                    num_videos_per_prompt=1, max_sequence_length=512,
                    device=torch.device(dev))
        finally:
            pipe.text_encoder = None
            del te
            gc.collect()
            _free_cuda(log)
        return pe, ne

    def _prompt_kwargs(self, prompt, negative, guidance, log) -> dict:
        """pipe呼び出し用のプロンプト系kwargs。UMT5遅延運転ではembedを
        先に作って文字列の代わりに渡す (pipeline内のencodeは
        text_encoder=Noneで失敗するため)。判定は _te_lazy フラグ基準:
        text_encoder属性の有無だけで判定するとテストのモックpipeまで
        遅延経路へ入ってしまう。"""
        if (not getattr(self, "_te_lazy", False)
                or getattr(self.pipe, "text_encoder", None) is not None):
            return {"prompt": prompt, "negative_prompt": negative}
        pe, ne = self._encode_prompt_lazy(prompt, negative, guidance, log)
        kw = {"prompt_embeds": pe}
        if ne is not None:
            kw["negative_prompt_embeds"] = ne
        return kw

    def _finalize_pipe(self, log, offload: str = "", footprint_gb=None,
                       vae_tiling: bool = False,
                       resident_margin_gb: float = 6,
                       gguf: bool = False):
        """vae_tiling=True: GPU常駐でもVAEタイリングを有効にする。
        動画を丸ごとVAEでencodeするアダプタ(VACEの81f条件動画など)は、
        タイリング無しだと中間テンソルが数十GBになりA100-80でもOOMする
        (2026-07-12実障害: 常駐70GB+encodeでVRAM 79GB超過)。
        静止画しかencodeしないアダプタは False のまま(デコードが速い)。

        resident_margin_gb: 常駐判定の余白。既定6GB。アクティベーションが
        大きいモデル(VACE bf16は81f attentionで~9GB)は大きめを渡すこと
        (bf16 70GB+6の判定でA100-80常駐→残り324MiBでOOMした実績あり)。"""
        import torch
        from diffusers import UniPCMultistepScheduler
        try:
            self.pipe.scheduler = UniPCMultistepScheduler.from_config(
                self.pipe.scheduler.config, flow_shift=self.flow_shift)
            log(f"scheduler: UniPC flow_shift={self.flow_shift}")
        except Exception as e:
            log(f"flow_shift設定スキップ: {e}")

        if _pick_dtype() is torch.float16:
            # bf16非対応GPU (T4級/sm<80): DiTはfp16運転、VAEだけfp32へ
            # 上げてデコードのNaN/白飛びを防ぐ (重み約0.5GB→1GBで許容)
            try:
                if getattr(self.pipe, "vae", None) is not None:
                    self.pipe.vae.to(torch.float32)
                log("bf16非対応GPU: fp16運転 + VAE fp32 (T4級対応)")
            except Exception as e:      # noqa: BLE001
                log(f"VAE fp32化に失敗 (fp16のまま続行): {str(e)[:80]}")

        # ---- オフロード戦略の決定 (速度の要。2026-07-12 A100で3分問題) ----
        # cuda  = 全モデルをGPU常駐(オフロード無し=最速。A100等でVRAMに
        #         余裕があるとき)。model_cpu_offloadはCPU<->GPU転送が挟まり
        #         A100の性能を殺すため、載るなら常駐が正解。
        # model = 主要モデルを順次GPUへ(中VRAM)。 seq = 逐次(12GB級)。
        mode = (offload or os.environ.get("VIDEOLAB_OFFLOAD", "")).lower()
        mode = {"sequential": "seq", "offload": "model", "full": "cuda",
                "none": "cuda", "group": "block"}.get(mode, mode)
        if mode not in ("seq", "model", "cuda", "block"):
            # auto: VRAM総量とモデル実測サイズから常駐可否を判定
            mode = "model"
            try:
                total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
                if footprint_gb and total_gb >= footprint_gb + resident_margin_gb:
                    mode = "cuda"
                elif total_gb < 18:
                    mode = "seq"
                log(f"オフロード自動判定: VRAM {total_gb:.0f}GB / 想定"
                    f"{footprint_gb}GB -> {mode}")
            except Exception:
                pass
        if gguf and mode == "seq":
            # GGUF×sequential offloadはdiffusers既知バグ (quant_type喪失の
            # KeyError) -> block offloadへ振替 (v0.8.2)
            log("GGUF量子化はsequential offload不可 -> block offloadへ振替")
            mode = "block"
        _t_hi = getattr(self.pipe, "transformer", None)
        if (_t_hi is not None
                and _t_hi is getattr(self.pipe, "transformer_2", None)
                and mode in ("model", "seq")):
            # Low単体(lite)の別名構成: model/seq offloadは同一DiTを2スロット
            # 登録し、フック連鎖が自分自身を指して毎stepフルDiTのCPU↔GPU
            # 往復になる (レビュー指摘 v0.9.13) — blockへ振替
            log("Low単体(別名)構成はmodel/seq offloadと相性が悪い -> "
                "block offloadへ振替")
            mode = "block"

        if mode == "block":
            ok = True
            for name in ("transformer", "transformer_2"):
                m = getattr(self.pipe, name, None)
                if m is not None and not self._group_offloaded(m):
                    # 早期退避済み (低RAM VMのロード順対策) はスキップ
                    ok = self._try_group_offload(m, log, name) and ok
            if ok:
                # DiT以外の小物 (VAE等) はGPUへ。text_encoderは遅延運転
                # なら不在 (None)。tokenizer/schedulerはnn.Moduleでない
                for cname, comp in self.pipe.components.items():
                    if cname in ("transformer", "transformer_2"):
                        continue
                    if isinstance(comp, torch.nn.Module):
                        comp.to("cuda")
                try:
                    self.pipe.vae.enable_tiling()
                except Exception:
                    pass
                log("block offload: VRAM使用=活性化ぶんのみ "
                    "(重み転送のぶん低速・16GB未満級GPU向け)")
                self._offload_mode = "block"
                return self._log_attn_backend(log)
            mode = "model"   # 適用不可 -> 従来経路へ

        if mode == "cuda":
            try:
                self.pipe.to("cuda")
                log("全モデルをGPU常駐 (オフロード無し=最速)")
                if vae_tiling:
                    try:
                        self.pipe.vae.enable_tiling()
                        log("VAEタイリング有効 (動画encodeのVRAM削減)")
                    except Exception as e:
                        log(f"VAEタイリング設定スキップ: {e}")
                # (静止画encodeのみのモデルは常駐時タイリング不要=デコード最速)
                self._offload_mode = "cuda"
                return self._log_attn_backend(log)
            except Exception as e:   # OOM等 -> オフロードへ退避
                log(f"GPU常駐に失敗({str(e)[:120]}) -> model_cpu_offloadへ")
                mode = "model"
        if mode == "seq":
            self.pipe.enable_sequential_cpu_offload()
            log("enable_sequential_cpu_offload (省メモリ・低速。12GB級GPU向け)")
        else:
            self.pipe.enable_model_cpu_offload()
            log("enable_model_cpu_offload")
        self._offload_mode = mode
        try:
            self.pipe.vae.enable_tiling()
        except Exception:
            pass
        self._log_attn_backend(log)

    def _log_attn_backend(self, log) -> None:
        # 透明性: diffusers/torchは既定でSDPA(A100ではFlashAttention/
        # メモリ効率カーネルを自動選択)。fp32でもxformersでもない。
        try:
            import torch
            has = hasattr(torch.nn.functional, "scaled_dot_product_attention")
            log(f"attention: PyTorch SDPA {'有効' if has else '不明'} "
                f"(A100ではFlashAttentionカーネルを自動選択)")
        except Exception:
            pass

    def _build_prompt(self, req: GenRequest, log) -> str:
        p = req.prompt.strip()
        if self.prompt_suffix and "motion score" not in p:
            ms = float(req.extra.get("motion_score", 3.0))
            suffix = self.prompt_suffix.format(motion=ms)
            p = f"{p} {suffix}"
            log(f"プロンプト接尾辞を自動付与: {suffix}")
        return p

    def _inject_mid_keyframes(self, condition, mids, positions, num_frames,
                              w, h, log):
        """条件テンソルの中間latentフレームにキーフレームを注入する。

        diffusersのWan i2v条件付けは [マスク4ch + 条件latent 16ch] の
        チャネル連結 (先頭=1、中間=0、終端=1)。AniSora V3.2は任意時間位置の
        画像ガイド(時空間マスク)で学習されているため、中間位置のマスクを
        立てて同じ立ち絵のlatentを書き込めば「真ん中も固定」できる
        (2026-07-12 ユーザー発案: 終端だけでは中盤で一回転して戻る
        「往復回転」が起きた=ロップで実証)。"""
        import torch
        pipe = self.pipe
        tvs = int(getattr(pipe, "vae_scale_factor_temporal", 4))
        mask_ch = condition.shape[1] - int(pipe.vae.config.z_dim)
        t_lat = condition.shape[2]
        lm_mean = torch.tensor(pipe.vae.config.latents_mean).view(
            1, -1, 1, 1, 1).to(condition.device, condition.dtype)
        lm_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
            1, -1, 1, 1, 1).to(condition.device, condition.dtype)
        for img, pos in zip(mids, positions):
            m = int(round(float(pos) * (num_frames - 1)))
            lm = 0 if m <= 0 else (m - 1) // tvs + 1
            lm = min(max(lm, 0), t_lat - 1)
            t = pipe.video_processor.preprocess(
                img, height=h, width=w).unsqueeze(2)
            t = t.to(device=condition.device, dtype=pipe.vae.dtype)
            enc = pipe.vae.encode(t)
            lat = (enc.latent_dist.mode() if hasattr(enc, "latent_dist")
                   else enc.latents)
            lat = lat.to(condition.dtype)
            lat = (lat - lm_mean) * lm_std
            condition[:, mask_ch:, lm] = lat[:, :, 0]
            condition[:, :mask_ch, lm] = 1.0
            log(f"中間キーフレーム注入: pos={float(pos):.2f} -> "
                f"画素フレーム{m} (latent {lm}/{t_lat})")

    def _inject_guidance_video(self, condition, frames, num_frames,
                               w, h, log, mask_mode="known",
                               generate_masks=None, fixed_frames=None,
                               neutralize_black=False):
        """Pose/Depth/Line等の全フレーム条件をI2V条件latentへ入れる。

        AniSora V3の公開チェックポイントは通常Wan I2Vと同じ36ch入力
        (noise16 + mask4 + condition latent16)で、専用ControlNetを持たない。
        公式READMEのMultimodal Guidanceがこの条件動画を学習済みかを
        実機で判定するための隔離プローブ。通常生成では呼ばれない。

        キーフレームを1枚ずつencodeすると時間畳み込みの文脈が失われるため、
        公式anymask実装と同様に動画全体を一度にVAEへ通す。
        """
        import torch

        if not frames:
            raise ValueError("AniSora guidance動画が空です")
        pipe = self.pipe
        if len(frames) != num_frames:
            got = len(frames)
            idx = [round(i * (got - 1) / max(1, num_frames - 1))
                   for i in range(num_frames)]
            frames = [frames[i] for i in idx]
            log(f"AniSora guidanceフレーム数を調整: {got} -> {num_frames}")

        # 空間インペイント条件。Wan I2Vの36ch入力は
        # mask4 + condition latent16を持つため、後段callbackで
        # latentを描き戻すだけでなく、モデル側にも「固定画素/
        # 未知画素」を明示する。generate_masksは255=生成/0=固定。
        spatial = bool(generate_masks and fixed_frames)
        spatial_masks = []
        if spatial:
            from PIL import Image as _ImgGuide

            def _resample(items):
                items = list(items)
                if not items:
                    raise ValueError("AniSora空間条件が空です")
                if len(items) == 1:
                    return items * num_frames
                if len(items) == num_frames:
                    return items
                got = len(items)
                ids = [round(i * (got - 1) / max(1, num_frames - 1))
                       for i in range(num_frames)]
                return [items[i] for i in ids]

            spatial_masks = _resample(generate_masks)
            fixed_frames = _resample(fixed_frames)
            composed = []
            for guide, ref, gen_mask in zip(
                    frames, fixed_frames, spatial_masks):
                guide = (guide if guide.size == (w, h)
                         else _fit_image(guide, w, h)).convert("RGB")
                if neutralize_black:
                    # OpenPoseの黒背景は「空の条件」ではなく
                    # 黒い面の描画指示として予測に混ざる。RGB全て
                    # 32未満の画素だけ正規化後0付近の中立灰へ。
                    import numpy as _npneutral
                    _ga = _npneutral.asarray(guide).copy()
                    _ga[_ga.max(axis=2) < 32] = 128
                    guide = _ImgGuide.fromarray(_ga, "RGB")
                ref = (ref if ref.size == (w, h)
                       else _fit_image(ref, w, h)).convert("RGB")
                gm = gen_mask.convert("L").resize(
                    (w, h), resample=_ImgGuide.Resampling.NEAREST)
                # 白(生成)にPose、黒(固定)にframe0由来の元絵。
                composed.append(_ImgGuide.composite(guide, ref, gm))
            frames = composed

        mode = str(mask_mode or "known").strip().lower()
        sparse = (mode in ("official", "official_sparse", "sparse")
                  or mode.startswith("sparse"))
        sparse_stride = 8
        if sparse and mode.startswith("sparse"):
            try:
                sparse_stride = max(1, int(mode[len("sparse"):]))
            except ValueError:
                sparse_stride = 8
        selected = list(range(0, num_frames, sparse_stride))
        if selected[-1] != num_frames - 1:
            selected.append(num_frames - 1)

        if sparse:
            # 公式 V3 image2video_any.py / V3.2 image2video.py と同型。
            # 未指定時刻は正規化後の0(中間灰)のままで、指定時刻
            # だけ画像を書き込む。連続Pose動画をVAE encodeすると
            # 「出力すべき画素」と解釈され、骨がそのまま再生される。
            video = torch.zeros(
                (1, 3, num_frames, h, w), device=condition.device,
                dtype=pipe.vae.dtype)
            for frame_id in selected:
                image = frames[frame_id]
                image = (image if image.size == (w, h)
                         else _fit_image(image, w, h))
                item = pipe.video_processor.preprocess(
                    image, height=h, width=w).unsqueeze(2)
                video[:, :, frame_id:frame_id + 1] = item.to(
                    device=condition.device, dtype=pipe.vae.dtype)
        else:
            chunks = []
            for image in frames:
                image = (image if image.size == (w, h)
                         else _fit_image(image, w, h))
                chunks.append(pipe.video_processor.preprocess(
                    image, height=h, width=w).unsqueeze(2))
            video = torch.cat(chunks, dim=2).to(
                device=condition.device, dtype=pipe.vae.dtype)
        enc = pipe.vae.encode(video)
        lat = (enc.latent_dist.mode() if hasattr(enc, "latent_dist")
               else enc.latents)
        lat = lat.to(device=condition.device, dtype=condition.dtype)
        lm_mean = torch.tensor(pipe.vae.config.latents_mean).view(
            1, -1, 1, 1, 1).to(condition.device, condition.dtype)
        lm_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
            1, -1, 1, 1, 1).to(condition.device, condition.dtype)
        lat = (lat - lm_mean) * lm_std

        mask_ch = condition.shape[1] - int(pipe.vae.config.z_dim)
        if tuple(lat.shape[2:]) != tuple(condition.shape[2:]):
            raise RuntimeError(
                "AniSora guidance latent形状が一致しません: "
                f"guidance={tuple(lat.shape)} / condition={tuple(condition.shape)}")
        condition[:, mask_ch:] = lat
        if spatial:
            # 画素時間軸の既知maskを、Wan公式と同じ
            # first-frame x4 + 4フレーム束の4ch形式へ畳む。
            # frame0の全黒生成maskは、ここで全画素既知になる。
            import numpy as _npguide
            known = []
            lh, lw = condition.shape[3], condition.shape[4]
            for gen_mask in spatial_masks:
                gm = gen_mask.convert("L").resize(
                    (lw, lh), resample=_ImgGuide.Resampling.NEAREST)
                known.append(torch.from_numpy(
                    (_npguide.asarray(gm) < 128).astype("float32")))
            px_mask = torch.stack(known, dim=0).unsqueeze(0).to(
                device=condition.device, dtype=condition.dtype)
            px_mask = torch.cat((
                px_mask[:, 0:1].repeat_interleave(4, dim=1),
                px_mask[:, 1:]), dim=1)
            px_mask = px_mask.view(
                1, px_mask.shape[1] // 4, 4, lh, lw).transpose(1, 2)
            if tuple(px_mask.shape) != tuple(condition[:, :mask_ch].shape):
                raise RuntimeError(
                    "AniSora空間mask形状が一致しません: "
                    f"mask={tuple(px_mask.shape)} / "
                    f"condition={tuple(condition[:, :mask_ch].shape)}")
            condition[:, :mask_ch] = px_mask
            mode = "spatial_known"
        elif sparse:
            # 公式は画素時間軸でmaskを作り、先頭を4回反復して
            # VAEの4フレーム時間圧縮に合わせる。この形はlatent slot
            # だけを1にする近似ではなく、公式テンソルそのもの。
            px_mask = torch.zeros(
                (1, num_frames, condition.shape[3], condition.shape[4]),
                device=condition.device, dtype=condition.dtype)
            for frame_id in selected:
                px_mask[:, frame_id:frame_id + 1] = 1.0
            px_mask = torch.cat((
                px_mask[:, 0:1].repeat_interleave(4, dim=1),
                px_mask[:, 1:]), dim=1)
            px_mask = px_mask.view(
                1, px_mask.shape[1] // 4, 4,
                condition.shape[3], condition.shape[4]).transpose(1, 2)
            if tuple(px_mask.shape) != tuple(condition[:, :mask_ch].shape):
                raise RuntimeError(
                    "AniSora公式mask形状が一致しません: "
                    f"mask={tuple(px_mask.shape)} / "
                    f"condition={tuple(condition[:, :mask_ch].shape)}")
            condition[:, :mask_ch] = px_mask
            mode = f"official_sparse{sparse_stride}"
        elif mode in ("first", "first_frame", "pose"):
            # READMEのPoseデモ入力: frame0だけ元絵、以後は骨動画。
            # 骨latentを消さず、通常画素としての既知指定だけを外す。
            condition[:, :mask_ch] = 0.0
            condition[:, :mask_ch, 0] = 1.0
            mode = "first"
        elif mode in ("unknown", "control", "zero", "0"):
            condition[:, :mask_ch] = 0.0
            mode = "unknown"
        else:
            condition[:, :mask_ch] = 1.0
            mode = "known"
        log("AniSora multimodal probe: guidanceを20ch I2V条件へ注入 "
            f"({num_frames}f -> latent {condition.shape[2]}f, mask={mode}"
            f"{', anchors=' + str(selected) if sparse else ''})")

    def generate(self, req: GenRequest, workdir: Path, log, progress) -> Path:
        import torch
        w = _snap(req.width, 16, 240)
        h = _snap(req.height, 16, 240)
        n = max(5, ((int(req.num_frames) - 1) // 4) * 4 + 1)     # 4k+1
        steps = int(req.steps)
        self._admit(w, h, n, log)          # 生成直前VRAMゲート (P0-3)
        img = _fit_image(req.images[0], w, h)
        kw = dict(image=img,
                  width=w, height=h, num_frames=n,
                  num_inference_steps=steps,
                  guidance_scale=float(req.guidance),
                  generator=torch.Generator("cpu").manual_seed(req.seed),
                  output_type="np", return_dict=False,
                  callback_on_step_end=_step_callback(progress, steps))
        kw.update(self._prompt_kwargs(self._build_prompt(req, log),
                                      req.negative or None,
                                      req.guidance, log))
        # 終端画像アンカー (2026-07-12): 2枚目の画像があれば last_image に。
        # 「imageとlast_imageの間を補間する」diffusers公式仕様。開始と同じ
        # 立ち絵を渡すと始点=終点のループ拘束になり、後ろ向きの回転を
        # 構造的に防ぐ (Veoの loop_anchor と同じ発想)。AniSoraは首尾
        # フレーム誘導で学習されており相性が良い見込み。
        if len(req.images) >= 2:
            kw["last_image"] = _fit_image(req.images[-1], w, h)
            log("終端画像アンカー: 始点と同じポーズで終わるよう拘束")
        # 公式READMEのMultimodal Guidance検証: 通常UI/工房からは送られない
        # 実験キー。骨/深度/線画動画を通常I2Vの条件latentへ一括注入する。
        guide_raw = list(req.extra.get("anisora_guidance_frames_b64") or [])
        guide_frames = load_images_b64(guide_raw) if guide_raw else []
        # 中間キーフレーム (images[1:-1]): prepare_latents をフックして
        # 条件テンソルに直接注入する。失敗しても先頭/終端拘束で続行。
        mids = [_fit_image(im, w, h) for im in req.images[1:-1]]
        mid_pos = (list(req.key_positions[1:-1])
                   if len(req.key_positions) == len(req.images)
                   else [(i + 1) / (len(mids) + 1)
                         for i in range(len(mids))])
        orig_prep = None
        base_condition = {}
        if mids or guide_frames:
            orig_prep = self.pipe.prepare_latents

            def _patched(*a, **k):
                latents, condition = orig_prep(*a, **k)
                try:
                    if guide_frames:
                        # High=Pose / Low=通常I2Vを切り替える際の復帰元。
                        # _inject_guidance_videoはconditionをin-place更新するため
                        # 先に独立copyを保持する。
                        base_condition["value"] = condition.detach().clone()
                        self._inject_guidance_video(
                            condition, guide_frames, n, w, h, log,
                            req.extra.get("anisora_guidance_mask", "known"))
                        # ControlNet風の最小近似: 主I2Vの原画条件を捨てず、
                        # Pose条件latentを同じ20ch内へ線形混合する。一回
                        # forwardなので、別予測合成より安い。学習済み制御枝
                        # ではないため、骨の画素模写が残るかは実走で判定。
                        if "anisora_condition_image_weight" in req.extra:
                            iw = max(0.0, min(1.0, float(
                                req.extra["anisora_condition_image_weight"])))
                            condition.mul_(1.0 - iw).add_(
                                base_condition["value"].to(
                                    device=condition.device,
                                    dtype=condition.dtype), alpha=iw)
                            log("AniSora conditioning混合: "
                                f"Pose {1-iw:.0%} / 原画 {iw:.0%} "
                                "(一回forward)")
                    else:
                        self._inject_mid_keyframes(condition, mids, mid_pos,
                                                   n, w, h, log)
                except Exception as e:   # noqa: BLE001
                    if guide_frames:
                        raise RuntimeError(
                            "AniSora multimodal guidance注入に失敗: "
                            f"{str(e)[:300]}") from e
                    log(f"中間キーフレーム注入に失敗 (先頭/終端のみで"
                        f"続行): {str(e)[:200]}")
                return latents, condition
            self.pipe.prepare_latents = _patched

        # Low単独でも、Pose画像と通常I2V画像を同じ36chへ混在させず、
        # 同一step・同一noisy latentを二度予測してノイズだけを合成する。
        # モデル実体は一体のまま、Pose=動き・原画=外見へ権威を分離する
        # 隔離実験。High+Low通常構成の二条件処理はrefine側の本線を使う。
        dual_model = getattr(self.pipe, "transformer", None)
        low_model = getattr(self.pipe, "transformer_2", None)
        low_alias = dual_model is not None and dual_model is low_model
        orig_dual_forward = None
        if (guide_frames and low_alias
                and req.extra.get("anisora_dual_condition")):
            orig_dual_forward = dual_model.forward
            zdim_dual = int(self.pipe.vae.config.z_dim)
            image_weight = max(0.0, min(0.95, float(
                req.extra.get("anisora_dual_condition_image_weight", 0.75))))
            dual_logged = [False]

            def _dual_low_forward(*a, **k):
                pose_out = orig_dual_forward(*a, **k)
                plain = base_condition.get("value")
                hidden = k.get("hidden_states")
                from_args = hidden is None and bool(a)
                if from_args:
                    hidden = a[0]
                if (plain is None or hidden is None
                        or hidden.shape[1] < zdim_dual + plain.shape[1]):
                    return pose_out
                image_hidden = hidden.clone()
                image_hidden[:, zdim_dual:zdim_dual + plain.shape[1]] = (
                    plain.to(device=hidden.device, dtype=hidden.dtype))
                if from_args:
                    image_args = (image_hidden,) + tuple(a[1:])
                    image_kwargs = dict(k)
                else:
                    image_args = a
                    image_kwargs = dict(k)
                    image_kwargs["hidden_states"] = image_hidden
                image_out = orig_dual_forward(*image_args, **image_kwargs)
                mixed = ((1.0 - image_weight) * pose_out[0]
                         + image_weight * image_out[0])
                if not dual_logged[0]:
                    log("AniSora Low単独二条件: 同じLowをPose/画像で別forward "
                        f"(Pose {1-image_weight:.0%} / 画像 {image_weight:.0%})")
                    dual_logged[0] = True
                return (mixed,) + tuple(pose_out[1:])

            dual_model.forward = _dual_low_forward

        # V3.2の二段デノイズを利用し、High-noise expertだけに
        # Pose条件を見せる。Low-noise expertの入力36chは
        # [noise16 + condition20]なので、後半20chをprepare_latents時に
        # 保存した通常の先頭画像条件へ戻す。これによりHighの
        # 大形を残しつつ、Lowが骨のRGB線を模写するのを防ぐ。
        release_low = bool(
            guide_frames and req.extra.get("anisora_guidance_release_low"))
        if release_low and low_alias:
            release_low = False
            log("Low単独構成にはHigh→Low境界が無いため、Pose解放は行わず"
                "二条件比を全stepへ適用")
        orig_low_forward = None
        if release_low and low_model is not None:
            orig_low_forward = low_model.forward
            zdim = int(self.pipe.vae.config.z_dim)
            logged_release = [False]

            def _low_forward(*a, **k):
                base = base_condition.get("value")
                hidden = k.get("hidden_states")
                from_args = hidden is None and bool(a)
                if from_args:
                    hidden = a[0]
                if (base is not None and hidden is not None
                        and hidden.shape[1] >= zdim + base.shape[1]):
                    replaced = hidden.clone()
                    replaced[:, zdim:zdim + base.shape[1]] = base.to(
                        device=hidden.device, dtype=hidden.dtype)
                    if from_args:
                        a = (replaced,) + tuple(a[1:])
                    else:
                        k["hidden_states"] = replaced
                    if not logged_release[0]:
                        log("AniSora Pose条件をHighのみに限定: "
                            "Lowは通常の先頭絵I2V条件へ解放")
                        logged_release[0] = True
                return orig_low_forward(*a, **k)

            low_model.forward = _low_forward
        elif release_low:
            log("PoseのLow解放を指定しましたが、二段モデルでは"
                "ないため通常条件のまま続行")
        log(f"生成開始: {w}x{h} {n}f steps={steps} cfg={req.guidance}")
        try:
            out = _call_with_optional_kwargs(
                self.pipe, kw, ["last_image", "callback_on_step_end"], log)
        finally:
            if orig_prep is not None:
                self.pipe.prepare_latents = orig_prep
            if orig_dual_forward is not None:
                dual_model.forward = orig_dual_forward
            if orig_low_forward is not None:
                low_model.forward = orig_low_forward
        progress(0.92)
        return _frames_to_mp4(list(out[0][0]), req.fps, workdir, log)


@register
class IllustriousAdapter(VideoAdapter):
    """Illustrious-XL (SDXL・アニメ特化) + ControlNet OpenPose の静止画t2i。

    2026-07-14ユーザー発案「ローカル完結の画像生成」: 8方向の直立OpenPose
    骨格グリッド(4x2)を条件に、キャラの8方向コンタクトシートを1枚で生成。
    1枚の中に閉じ込めることでキャラ一貫性を確保し、骨格が向きと左右を
    構造的に指定する (Codex比: 無料・左右取り違えに強い設計)。
    契約: mode="t2i" / prompt=danbooruタグ寄り /
    extra.pose_image_b64=骨格グリッドPNG(必須) /
    extra.controlnet_scale(既定0.9)。結果はPNG (resultはpngで配信)。"""
    id = "illustrious"
    label = "Illustrious-XL (画像・SDXL+OpenPose)"
    desc = ("アニメ特化SDXLで8方向シートを1枚生成 (ControlNet OpenPoseで"
            "向き・左右を骨格指定)。extra.pose_image_b64 必須、"
            "extra例: {\"controlnet_scale\": 0.9}")
    requires = "Colab T4/L4 / ローカル8GB+ (fp16・DL約10GB)"
    modes = ("t2i", "i2i")
    # 既定はv2.0 (2026-07-14ユーザー指定「高い安定してるやつ」。公開・
    # ゲートなし・ε-pred。v0.1はextra.ill_repo/ill_fileで戻せる)
    ckpt_repo = os.environ.get(
        "VIDEOLAB_ILLUSTRIOUS_REPO",
        "OnomaAIResearch/Illustrious-XL-v2.0")
    ckpt_file = os.environ.get("VIDEOLAB_ILLUSTRIOUS_FILE",
                               "Illustrious-XL-v2.0.safetensors")
    cn_repo = os.environ.get("VIDEOLAB_ILLUSTRIOUS_CN",
                             "xinsir/controlnet-openpose-sdxl-1.0")
    vae_repo = "madebyollin/sdxl-vae-fp16-fix"
    cache_repos = (ckpt_repo, cn_repo, vae_repo)
    disk_gb = 12
    defaults = {"width": 1152, "height": 896, "num_frames": 1, "fps": 1,
                "steps": 28, "guidance": 6.0}
    NEG = ("worst quality, low quality, bad anatomy, bad hands, "
           "extra digits, fewer digits, watermark, signature, text, "
           "jpeg artifacts, blurry, lowres")

    def __init__(self):
        super().__init__()
        self.pipe = None

    def unload(self, log):
        self.pipe = None
        self._i2i_pipe = None
        self.loaded = False

    def _want_ckpt(self, extra: dict) -> tuple:
        e = extra or {}
        return (str(e.get("ill_repo") or self.ckpt_repo),
                str(e.get("ill_file") or self.ckpt_file))

    def ensure_loaded(self, log):
        _require_deps(log)
        import torch
        from diffusers import (AutoencoderKL, ControlNetModel,
                               EulerAncestralDiscreteScheduler,
                               StableDiffusionXLControlNetPipeline)
        repo, fname = self._want_ckpt(getattr(self, "_next_extra", {}))
        ck = _hf_download(repo, fname, log)
        self._loaded_ckpt = (repo, fname)
        cn_dir = _snapshot_local(self.cn_repo, log)
        vae_dir = _snapshot_local(self.vae_repo, log)
        log("Illustrious-XL 読み込み (fp16) + OpenPose ControlNet")
        cn = ControlNetModel.from_pretrained(
            cn_dir, torch_dtype=torch.float16)
        vae = AutoencoderKL.from_pretrained(
            vae_dir, torch_dtype=torch.float16)
        pipe = StableDiffusionXLControlNetPipeline.from_single_file(
            ck, controlnet=cn, vae=vae, torch_dtype=torch.float16)
        # Illustrious標準レシピ: Euler a / cfg 5-7 / 24-32step
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config)
        pipe.to("cuda")
        # VAEタイリング: 1536級のencode(i2i下地)/decodeがL4 22GBを超える
        # 実測 (2026-07-14: i2i 1536でinit encodeがOOM、1152でも断片化と
        # 重なりdecodeで落ちた)。タイル境界の縫い目はflat color画風では
        # 実害なし
        try:
            pipe.vae.enable_tiling()
        except Exception:
            pass
        self.pipe = pipe
        self.loaded = True

    def generate(self, req: GenRequest, workdir: Path, log,
                 progress) -> Path:
        import torch
        from PIL import Image
        want = self._want_ckpt(req.extra)
        if self.loaded and want != getattr(self, "_loaded_ckpt", want):
            log(f"チェックポイント変更 -> {want[1]}: 積み替えます")
            self.unload(log)
            _free_cuda(log)
        if not self.loaded:
            self._next_extra = dict(req.extra or {})
            self.ensure_loaded(log)
        w = _snap(req.width, 8, 512)
        h = _snap(req.height, 8, 512)
        pb = req.extra.get("pose_image_b64")
        if not pb:
            raise RuntimeError("illustrious には extra.pose_image_b64 "
                               "(OpenPose骨格グリッドPNG) が必要です")
        pose = load_images_b64([pb])[0].convert("RGB")
        if pose.size != (w, h):
            pose = pose.resize((w, h), Image.LANCZOS)
        steps = max(8, int(req.steps or 28))
        scale = float(req.extra.get("controlnet_scale", 0.9))
        g = torch.Generator("cpu").manual_seed(req.seed)
        kw = dict(prompt=req.prompt,
                  negative_prompt=req.negative or self.NEG,
                  controlnet_conditioning_scale=scale,
                  guidance_scale=float(req.guidance or 6.0),
                  generator=g)
        if req.mode == "i2i" and req.images:
            # マネキンコンパス下地のi2i (v0.9.6): マゼンタ背景とセル配置・
            # 向きを下地から引き継ぐ (t2iは背景色指定を無視し、斜め45°を
            # 正面/背面へ丸める実測への対策)。骨格CNはt2iと同じく併用
            from diffusers import StableDiffusionXLControlNetImg2ImgPipeline
            init = req.images[0].convert("RGB")
            if init.size != (w, h):
                init = init.resize((w, h), Image.LANCZOS)
            den = min(1.0, max(0.05, float(req.extra.get("denoise", 0.7))))
            pipe = getattr(self, "_i2i_pipe", None)
            if pipe is None:
                # 部品共有の別ファサード (VRAM追加なし)。schedulerも共有
                # だが、ジョブはworker_loop単線でset_timestepsが毎回
                # リセットするため安全 — 並列化するときは要分離
                pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pipe(
                    self.pipe)
                self._i2i_pipe = pipe
            # diffusersのget_timestepsはfloor — roundだと+1件多く数えて
            # ゲージが張り付く
            eff = max(1, min(steps, int(steps * den)))
            log(f"i2i生成: {w}x{h} steps={steps} denoise={den} "
                f"cfg={req.guidance} CN openpose={scale}")
            kw.update(image=init, control_image=pose, strength=den,
                      num_inference_steps=steps,
                      callback_on_step_end=_step_callback(progress, eff))
        else:
            pipe = self.pipe
            log(f"t2i生成: {w}x{h} steps={steps} cfg={req.guidance} "
                f"CN openpose={scale}")
            kw.update(image=pose, num_inference_steps=steps,
                      width=w, height=h,
                      callback_on_step_end=_step_callback(progress, steps))
        out = _call_with_optional_kwargs(pipe, kw,
                                         ["callback_on_step_end"], log)
        img = out.images[0] if hasattr(out, "images") else out[0][0]
        dst = workdir / "out.png"
        img.save(dst)
        # ジョブ末にVRAMを掃除 — 断片化の持ち越しで2回目以降のジョブが
        # decodeで落ちる実測 (2026-07-14: 0.7と0.9は通ったのに0.8が
        # 「必要1.27GB/空き1.21GB」で失敗)
        del out
        torch.cuda.empty_cache()
        progress(0.98)
        return dst


@register
class AniSoraAdapter(_WanA14BBase):
    """Bilibili Index-AniSora V3.2 (アニメ特化・8ステップ蒸留) — GGUF Q8 差し込み。

    Wan2.2 I2V-A14B の VAE/テキストエンコーダ/スケジューラに、AniSora V3.2 の
    GGUF量子化transformer (High/Low 各15.9GB) を差し込む。Q8×2で計約32GBは
    A100 80GB の VRAM に収まり、ベース側のDLは約12GBで済む。
    注意: Wan2.2系GGUFの from_single_file は config 未指定だと
    「Cannot copy out of meta tensor」で失敗する既知問題があるため、
    config=ベースリポジトリを必ず明示する (diffusers issue #12009)。
    """
    id = "anisora"
    label = "AniSora V3.2 (アニメ特化・GGUF・8step)"
    desc = ("Bilibili のアニメ1000万クリップ特化モデル(Wan2.2ベース)。"
            "『motion score』で動きの強さを直接指定できる。i2vのみ。"
            "extra例: {\"motion_score\": 3.5}。量子化は環境変数 "
            "VIDEOLAB_ANISORA_QUANT=Q4_0(既定・計18GB)/Q8_0(計32GB・明示時のみ)。"
            "extra refine_frames_b64+refine_strength でSDEdit式リファイン"
            "(既存動画の質感上塗り。stepsはスケジュール解像度、実行は尻尾のみ)")
    requires = ("Colab A100/L4 / ローカル24GB+ (Q4_0) / 12GB級はQ4_0+"
                "逐次オフロード (低速・RAM32GB推奨)・DL 30〜45GB")
    gguf_repo = "QuantStack/Index-Anisora-V3.2-GGUF"
    cache_repos = ("QuantStack/Index-Anisora-V3.2-GGUF",)
    disk_gb = 45         # GGUF Q8 32GB + ベース部品12GB
    prompt_suffix = ("aesthetic score: 6.0. motion score: {motion:.1f}. "
                     "There is no text in the video.")
    defaults = {"width": 464, "height": 848, "num_frames": 81, "fps": 16,
                "steps": 8, "guidance": 1.0}

    def _resolve_want(self, extra: dict) -> tuple:
        """量子化とオフロード方式をリクエスト>環境変数>既定の順で解決。

        Colabサーバは遠隔で環境変数を触れないため、アプリからはジョブの
        extra {"quant": "Q4_0", "offload": "seq"} で指定できる(共通設定、
        2026-07-12要望: A100ばかり使えないのでL4はQ4_0で運用したい)。"""
        # 既定Q4_0 (2026-07-13方針: VRAM16GB未満級を上限に。Q8はGUI選択肢
        # からも撤廃済みで、明示指定されたときだけ受理する)。v0.9.12で
        # Q3_K_S等の下位quantも受理 (棚在庫はLow側のみ完備 — High側は
        # Q4_0/Q8_0だけなので _ensure_loaded_impl がHighを自動代替する)。
        # bf16はVACE専用の逃げ道設定が共通quantとしてstage2へ転送されて
        # くる正規ルートがある (エンジンは同じ値を両stageへ渡す) — GGUF
        # 専用の本アダプタでは従来どおりQ4_0で受ける (v0.9.13レビュー指摘:
        # ValueError化するとlatent_refine本線がキャンバス工程ごと失敗する)
        _qraw = ((extra or {}).get("quant")
                 or os.environ.get("VIDEOLAB_ANISORA_QUANT", "Q4_0"))
        if str(_qraw).strip().lower() == "bf16":
            _qraw = "Q4_0"
        q = _norm_quant(_qraw)
        # 既定は "" = auto (VRAMを見てGPU常駐 or model_cpu_offloadを選ぶ)。
        # "seq" のみ明示指定 (12GB級の省メモリモード)。
        off = str((extra or {}).get("offload")
                  or os.environ.get("VIDEOLAB_OFFLOAD", "")).lower()
        # "model"(=model_cpu_offload) も受理する: エンジンは大判の
        # グリッド生成でVRAM<60GBのとき offload=model を明示するが、
        # 従来は"seq"以外を握りつぶして常駐へ落とし、720x1296の
        # 活性化~21GBでOOM->方向別フォールバックしていた
        # (2026-07-13実障害)
        if off in ("seq", "sequential"):
            off = "seq"
        elif off in ("model", "offload"):
            off = "model"
        elif off in ("block", "group"):
            off = "block"    # v0.8.2: 重みCPU常駐+ブロック転送 (GGUF向け)
        else:
            off = ""
        return q, off

    @staticmethod
    def _lite_wanted(extra: dict) -> bool:
        """リファイン専用ジョブ (latent_from / refine_frames_b64) はLow体
        だけで完結する: σ<boundary(0.9)の尻尾は全stepがtransformer_2(Low)
        へルーティングされ、Highは一度も呼ばれない (2026-07-16調査で
        コード裏取り済み — 従来はロードだけして丸ごとCPU退避していた)。
        Highを積まないことで RAM/DL 約9GB (Q4) を節約し、Low側にしか
        棚在庫が無いQ3_K_S等も完全に使える。VIDEOLAB_ANISORA_LITE=off
        で従来動作 (常にHigh込み) へ戻せる。"""
        if os.environ.get("VIDEOLAB_ANISORA_LITE", "on").strip().lower() in (
                "off", "0", "false", "no"):
            return False
        e = extra or {}
        # ComfyUIで公開されているV3.2 Low単独レシピの隔離検証口。
        # 通常I2V・純ノイズ空間生成でも、transformer/transformer_2を
        # Low一体の別名にして全ノイズ域へ通す。工房本線はこのキーを
        # 送らないため、既定のHigh+Low構成には影響しない。
        if e.get("anisora_low_only") or e.get("anisora_high_only"):
            return True
        # 空間インペイントはσ1.0から始め、boundary以上の
        # High側ステップも必ず通る。Low単体ロードは不可。
        if e.get("latent_spatial_empty"):
            return False
        return bool(e.get("latent_from") or e.get("refine_frames_b64"))

    def _ensure_loaded_impl(self, log):
        _require_deps(log)
        import torch
        from diffusers import (GGUFQuantizationConfig, WanImageToVideoPipeline,
                               WanTransformer3DModel)
        from huggingface_hub import hf_hub_download
        quant, offload = self._resolve_want(getattr(self, "_next_extra", {}))
        next_extra = getattr(self, "_next_extra", {}) or {}
        lite = self._lite_wanted(next_extra)
        single_expert = ("high" if next_extra.get("anisora_high_only") else
                         "low" if next_extra.get("anisora_low_only") else "")
        # High側の棚在庫はQ4_0/Q8_0のみ — 静的規則で代替 (混載ペア。Highは
        # 序盤数stepの構図担当なので品質影響は小さい)。動的なfile_exists
        # 判定はオフライン/Drive固定で不発になり袋小路DLに落ちる
        # (v0.9.13レビュー指摘) — 詳細は _anisora_high_quant
        hq = _anisora_high_quant(quant)
        gguf_high = f"High/Index-Anisora-V3.2-High-{hq}.gguf"
        if not lite and hq != quant:
            log(f"High側は{quant}の棚在庫が無いため{hq}で代替します "
                "(Low側は指定どおり)")
        gguf_low = f"Low/Index-Anisora-V3.2-Low-{quant}.gguf"
        qc = GGUFQuantizationConfig(compute_dtype=_pick_dtype())
        if single_expert == "high":
            log(f"隔離実験ロード: High単体 {hq} (Lowを読まず全ノイズ域へ適用)")
            hi = _hf_download(self.gguf_repo, gguf_high, log)
            lo = None
        elif lite:
            log(f"リファイン専用ロード: Low単体 {quant} (High省略で"
                "RAM/DL約9GB節約。VIDEOLAB_ANISORA_LITE=offで従来動作)")
            hi = None
            lo = _hf_download(self.gguf_repo, gguf_low, log)
        else:
            log(f"GGUF DL: {self.gguf_repo} {quant} "
                f"(High+Low 各{'15.9GB' if quant == 'Q8_0' else '9GB前後'})")
            hi = _hf_download(self.gguf_repo, gguf_high, log)
            lo = _hf_download(self.gguf_repo, gguf_low, log)
        # ベース部品はsnapshot経由 (v0.8.7: Driveキャッシュ+429対策の
        # 多段DLに乗せる。transformer重みは除外済みの約12GB)
        snap = _snapshot_local(self.base_repo, log)
        self._te_lazy = _low_ram_vm()
        # 低RAM VM×block退避では「1体ロードするごとに即ディスク退避」:
        # 2体(約18GB)をRAMへ積んでからまとめて退避すると、その瞬間の山が
        # T4(51GB)の残りRAMに刺さる (2026-07-16実障害: anisora切替段で
        # メモリ黄色→VM死。anisoraはLoRAを載せないため早期退避が安全 —
        # vaceはLightning適用が後segueするため従来順のまま)
        _early_off = (offload == "block" and _low_ram_vm(60.0))
        # ロード前RAMゲート (P0-2): GGUF実サイズからロードピークを見積もり、
        # VM killになる前に明示エラーで止める。早期ディスク退避時は同時
        # 常駐が1体ぶんに減る。サイズが取れない環境 (テスト/Drive未DL) は
        # ゲートしない
        gg = _gguf_gb(*(p for p in (hi, lo) if p))
        if gg >= 1.0:
            # ベース部品のRAM実装は VAE+tokenizer≈2.5GB (スナップショット
            # 12GBはUMT5込みのディスクサイズ — UMT5は別項で計上)
            _peak = ((max(_gguf_gb(hi), _gguf_gb(lo)) if _early_off else gg)
                     + 2.5 + (0 if self._te_lazy else 11) + 6)
            _ram_gate(log, _peak, f"AniSora {quant} 読み込み")
        t_hi = None
        if hi is not None:
            log("transformer(High) 読み込み — configはWan2.2ベースを明示")
            t_hi = WanTransformer3DModel.from_single_file(
                hi, quantization_config=qc, config=snap,
                subfolder="transformer", torch_dtype=_pick_dtype())
            if _early_off:
                self._try_group_offload(t_hi, log, "transformer(早期退避)")
        t_lo = None
        if lo is not None:
            log("transformer_2(Low) 読み込み")
            t_lo = WanTransformer3DModel.from_single_file(
                lo, quantization_config=qc, config=snap,
                subfolder="transformer_2", torch_dtype=_pick_dtype())
            if _early_off:
                self._try_group_offload(t_lo, log, "transformer_2(早期退避)")
        pipe_kwargs = {}
        if self._te_lazy:
            # v0.8.2: 低RAM VM (L4=53GB) ではUMT5(11GB)を持たずにロードし、
            # ジョブ内で一時ロード→即解放する (handoff v0.7.6と同じ対策。
            # 従来は素のanisora/vaceがUMT5常駐のままモデル切替するたび
            # RAMスパイクでランタイムがOOM killされた)
            pipe_kwargs["text_encoder"] = None
            log("低RAM VM: UMT5はロードせずジョブ内で一時ロードします")
        log(f"ベース部品 読み込み: {snap} (VAE+UMT5)")
        # 単体モードはHigh/Lowの両属性へ同じ実体を渡す (追加メモリゼロ)。
        # None渡しはWanImageToVideoPipeline.__call__内の属性参照で壊れる
        # 可能性があるため、実体共有=追加メモリゼロの別名が安全
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            snap, transformer=(t_hi if t_hi is not None else t_lo),
            transformer_2=(t_lo if t_lo is not None else t_hi),
            torch_dtype=_pick_dtype(), **pipe_kwargs)
        self._te_from = snap
        # GPU常駐可否の判定用フットプリント(GB): GGUF実サイズ +
        # bf16テキストエンコーダ(~6) + VAE(~0.5) + 生成アクティベーション(~4)
        # サイズが取れない環境では従来の固定表へフォールバック
        if gg >= 1.0:
            fp = gg + (0 if self._te_lazy else 6) + 4.5
            self._dit_gb = max(_gguf_gb(hi), _gguf_gb(lo))
        else:
            fp = {"Q4_0": 27, "Q8_0": 42}.get(quant, 42)
            if lite:
                fp -= 9
            if self._te_lazy:
                fp -= 6          # UMT5不在ぶん常駐判定を緩める
            self._dit_gb = {"Q8_0": 16.0}.get(quant, 9.3)
        self._finalize_pipe(log, offload=offload, footprint_gb=fp, gguf=True)
        self.loaded_quant, self.loaded_offload = quant, offload
        self.loaded_lite = lite
        self.loaded_single_expert = single_expert or ("low" if lite else "")
        self.loaded_flow_shift = self.flow_shift
        self.loaded = True

    def _configure_experiment_scheduler(self, req: GenRequest, log) -> None:
        """ジョブ単位のflow shiftを反映し、指定なしでは既定へ戻す。"""
        try:
            shift = float(req.extra.get("anisora_flow_shift", self.flow_shift))
        except (TypeError, ValueError) as e:
            raise RuntimeError("anisora_flow_shiftは数値が必要です") from e
        shift = max(0.1, min(20.0, shift))
        current = float(getattr(self, "loaded_flow_shift", self.flow_shift))
        if abs(current - shift) < 1e-9:
            return
        from diffusers import UniPCMultistepScheduler
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config, flow_shift=shift)
        self.loaded_flow_shift = shift
        tag = ("High単独実験" if req.extra.get("anisora_high_only") else
               "Low単独実験" if req.extra.get("anisora_low_only") else "")
        log(f"AniSora scheduler: UniPC flow_shift={shift:g}"
            + (f" ({tag})" if tag else ""))

    def generate(self, req: GenRequest, workdir: Path, log, progress) -> Path:
        # 量子化/オフロード指定がロード時と違う場合は積み替え (共通設定を
        # ジョブ単位で反映するため。切替は数分かかるので必要時のみ)。
        # Low単体(lite)ロード中に通常i2vジョブが来たらHigh込みへ積み替え
        # (逆=High込みでリファインはそのまま賄える)
        want = self._resolve_want(req.extra)
        lite = self._lite_wanted(req.extra)
        requested_single = ("high" if req.extra.get("anisora_high_only") else
                            "low" if req.extra.get("anisora_low_only") else "")
        if self.loaded and (
                want != (getattr(self, "loaded_quant", None),
                         getattr(self, "loaded_offload", None))
                or (getattr(self, "loaded_lite", False) and not lite)
                or (requested_single and requested_single !=
                    getattr(self, "loaded_single_expert", ""))):
            log(f"設定変更 {self.loaded_quant}/{self.loaded_offload}"
                f"{'/Low単体' if getattr(self, 'loaded_lite', False) else ''}"
                f" -> {want[0]}/{want[1]}{'' if lite else '/High込み'}: "
                "モデルを積み替えます")
            self.unload(log)
            _free_cuda(log)
        if not self.loaded:
            self._next_extra = dict(req.extra or {})
            self.ensure_loaded(log)
        self._configure_experiment_scheduler(req, log)
        if req.extra.get("refine_frames_b64") or req.extra.get("latent_from"):
            return self._generate_refine(req, workdir, log, progress)
        return super().generate(req, workdir, log, progress)

    def _generate_refine(self, req: GenRequest, workdir: Path, log,
                         progress) -> Path:
        """SDEdit式リファイン: 1段目(VACE骨格制御)の動画を途中ノイズまで
        戻し、スケジュール終盤だけAniSoraで再デノイズして質感を上塗りする。

        「Funでポーズ・AniSoraで仕上げ」の2段目 (2026-07-12設計)。素の
        VACE-Funはポーズ・向きの制御が完璧だが、速い動きの末端(振り足の
        靴など)が実写priorでブラー溶けする。動き・構図はlatentの低周波に
        残したまま(開始σ=refine_strength、既定0.45)、アニメpriorが表面を
        描き直す。追加コスト=terminalの2〜4step+VAEencode。

        機構 (diffusers 0.39実測に基づく):
        - WanImageToVideoPipeline.__call__ は latents= を素通しで使い、
          条件テンソル(立ち絵)は通常どおり組む
        - タイムステップはループ直前に scheduler.set_timesteps で決まる
          ため、set_timesteps をラップして σ<=strength の尻尾へ切詰める
          (timesteps/sigmas を同じオフセットでスライス+終端σ保持 —
          UniPCはσを _step_index の位置参照で読むため両方必須)
        - 初期latents = (1-σ0)·x0 + σ0·ε (use_flow_sigmasの順方向そのもの)
        - 尻尾のtは全てboundary(0.9)未満 → 全stepがtransformer_2(Low=
          ディテール側エキスパート)に自動ルーティングされる
        """
        import numpy as np                                       # noqa: F401
        import torch
        pipe = self.pipe
        w = _snap(req.width, 16, 240)
        h = _snap(req.height, 16, 240)
        lat_from = str(req.extra.get("latent_from") or "").strip()
        if lat_from:
            # latent再加工モード (2026-07-13ユーザー発案「VACEをフルで
            # 当てて、その潜在をVAEを通す前にanisoraで再加工」):
            # フレームは受け取らず、前段vaceジョブの最終latentを直接読む
            if not req.images:
                raise RuntimeError("latent_from には条件画像 (images[0]="
                                   "参照キャンバス) が必要です")
            n = max(5, ((int(req.num_frames) - 1) // 4) * 4 + 1)
            frames = []
        else:
            frames = load_images_b64(list(req.extra["refine_frames_b64"]))
            n = max(5, ((len(frames) - 1) // 4) * 4 + 1)         # 4k+1
            frames = frames[:n]
            frames = [f if f.size == (w, h) else _fit_image(f, w, h)
                      for f in frames]
        spatial_empty = bool(req.extra.get("latent_spatial_empty"))
        strength = float(req.extra.get("refine_strength", 0.45))
        # 体を「空の潜在」にする本線は中途σからのSDEditでは
        # ない。純ノイズの分布とスケジューラを整合させるため必ず
        # σ1.0から全スケジュールを走らせる。この経路ではrefineノブより
        # 「体に元潜在を混ぜない」契約を優先する。
        strength = 1.0 if spatial_empty else max(0.10, min(0.90, strength))
        # AniSora V3.2は蒸留モデル。管理画面の契約どおり4--12を通し、
        # 4--7を黙って8へ戻さない（レガシー経路は呼出側の既定24のまま）。
        steps = max(4, int(req.steps))
        dev = pipe._execution_device

        def _dev_of(m):
            try:
                return next(m.parameters()).device.type
            except (StopIteration, AttributeError, TypeError):
                return None

        # ---- VRAM退避 (2026-07-13 A100-40実OOM対策): リファインの尻尾は
        # 全ステップが boundary(0.9) 未満 = Low側(transformer_2)しか
        # 走らないため、High側は丸ごとCPUへ (約9GB解放)。さらに81f一括
        # encode中はUMT5も不要なのでCPUへ (約11GB解放。常駐29.5GB +
        # encodeスパイク9-12GBで39.5GBの天井に激突した) ----
        moved_back = []
        hi_t = getattr(pipe, "transformer", None)
        if (not spatial_empty
                and hi_t is not None
                and hi_t is not getattr(pipe, "transformer_2", None)
                # ↑Low単体(lite)ロードではtransformer=transformer_2の別名。
                #  退避するとLow本体まで消える (v0.9.12)
                and _dev_of(hi_t) == "cuda"
                and not self._group_offloaded(hi_t)):
            # block offload時は重みが既にCPU側 (.to()はdiffusersが拒否)
            log("リファイン: High側transformerをCPUへ退避 (尻尾では不使用)")
            hi_t.to("cpu")
            moved_back.append(hi_t)
        te = getattr(pipe, "text_encoder", None)
        te_offloaded = False
        if te is not None and _dev_of(te) == "cuda":
            te.to("cpu")
            te_offloaded = True
        _free_cuda(log)
        # ---- 1段目動画をlatentへ ----
        # ★ no_grad必須: diffusersのパイプラインは@torch.no_grad()装飾
        # だが、アダプタから直接呼ぶvae.encodeは勾配グラフを全チャンク分
        # 保持してVRAMを食い尽くす (2026-07-13実障害: A100-80でも78GB
        # 積み上げてOOM。WanVAEは1+4+4...フレームのチャンク処理なので
        # 推論だけなら81fでもスパイクは数GBで済む)
        had_tiling = getattr(pipe.vae, "use_tiling", False)
        try:
            pipe.vae.enable_tiling()
        except Exception:
            pass

        def _vram(tag):
            try:
                fr, tot = torch.cuda.mem_get_info()
                log(f"リファインVRAM[{tag}]: 空き{fr / 2**30:.1f}"
                    f"/{tot / 2**30:.0f}GB")
            except Exception:
                pass

        _vram("encode前")
        # 生成直前VRAMゲート (P0-3): High退避後の空きで判定し、不足なら
        # block offloadへ自動降格 -> それでも足りなければ実行前に明示エラー
        self._admit(w, h, n, log, tag="refine")
        if lat_from:
            # VAEを一切通さないlatent直読み: decode→encode往復の劣化と
            # 一括encodeのVRAMスパイクが両方消える
            lp = WORK_ROOT / lat_from / "latent.pt"
            if not lp.is_file():
                raise RuntimeError(
                    f"latent_from={lat_from}: latent.pt がありません "
                    "(extra.emit_latent付きのvaceジョブIDを指定)")
            with torch.no_grad():
                try:
                    x0 = torch.load(lp, map_location="cpu",
                                    weights_only=True)
                except TypeError:                    # 旧torch
                    x0 = torch.load(lp, map_location="cpu")
                x0 = x0.to(dev, torch.float32)
            t_expect = (n - 1) // 4 + 1
            if x0.shape[2] != t_expect:
                raise RuntimeError(
                    f"latentの時間次元 {x0.shape[2]} が num_frames={n} "
                    f"(期待{t_expect}) と一致しません")
            log(f"latent再加工: {lat_from}/latent.pt {tuple(x0.shape)} "
                "(VAE encodeスキップ)")
            enc = vid = None
        else:
            with torch.no_grad():
                vid = torch.cat(
                    [pipe.video_processor.preprocess(f, height=h, width=w)
                     .unsqueeze(2) for f in frames], dim=2)
                vid = vid.to(device=dev, dtype=pipe.vae.dtype)
                enc = pipe.vae.encode(vid)
                x0 = (enc.latent_dist.mode()
                      if hasattr(enc, "latent_dist") else enc.latents)
                lm = torch.tensor(pipe.vae.config.latents_mean).view(
                    1, -1, 1, 1, 1).to(x0.device, x0.dtype)
                ls = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
                    1, -1, 1, 1, 1).to(x0.device, x0.dtype)
                x0 = ((x0 - lm) * ls).to(torch.float32)
        del vid, enc
        try:
            torch.cuda.empty_cache()   # encodeの中間バッファを即返却
        except Exception:
            pass
        _vram("encode後")
        if te_offloaded:
            te.to(dev)                 # プロンプトembedで必要になる
            te_offloaded = False
        # ---- スケジュール切詰めラッパ ----
        sched = pipe.scheduler
        orig_set = sched.set_timesteps
        box = {}
        _start_at_high_edge = bool(
            req.extra.get("latent_spatial_start_at_high_edge"))
        orig_guide_prep = None
        orig_guide_high_forward = None
        orig_guide_low_forward = None
        guide_high_model = None
        guide_low_model = None

        def _set_ts(nsteps, device=None, **kw3):
            orig_set(nsteps, device=device, **kw3)
            sig = sched.sigmas                    # CPU float32, 末尾に終端σ
            if _start_at_high_edge:
                try:
                    _bt = (float(pipe.config.boundary_ratio)
                           * float(sched.config.num_train_timesteps))
                    _high = (sched.timesteps >= _bt).nonzero()
                    idx = (int(_high[-1]) if len(_high)
                           else 0)
                except (AttributeError, TypeError, ValueError):
                    idx = 0
            else:
                hit = (sig[:-1] <= strength).nonzero()
                idx = int(hit[0]) if len(hit) else int(len(sig) - 2)
            idx = max(0, min(idx, len(sig) - 2))
            sched.timesteps = sched.timesteps[idx:]
            sched.sigmas = sched.sigmas[idx:]
            sched.num_inference_steps = len(sched.timesteps)
            box["s0"] = float(sched.sigmas[0])
            box["tail"] = len(sched.timesteps)

        sched.set_timesteps = _set_ts
        try:
            _set_ts(steps, device=dev)            # σ0確定のため先行実行
            s0, tail = box["s0"], box["tail"]
            g = torch.Generator("cpu").manual_seed(req.seed)
            noise = torch.randn(x0.shape, generator=g).to(
                x0.device, torch.float32)
            _smb = req.extra.get("latent_spatial_mask_b64")
            _smb_low = req.extra.get("latent_spatial_low_mask_b64")
            _smask_release_last_steps = None
            if "latent_spatial_release_last_steps" in req.extra:
                try:
                    _smask_release_last_steps = max(0, min(
                        tail, int(round(float(req.extra[
                            "latent_spatial_release_last_steps"])))))
                except (TypeError, ValueError):
                    raise RuntimeError(
                        "latent_spatial_release_last_stepsは整数が必要です")
            _smask = None
            _smask_low = None
            _smask_image = None
            if _smb:
                # Lマスク (255=生成/0=固定) をlatent格子へ。
                # NEARESTを明示し、顔窓の境界に中間値を発生させない。
                from PIL import Image as _ImgSL
                import base64 as _b64sl
                import io as _iosl
                import numpy as _npsl
                _smbs = list(_smb) if isinstance(_smb, (list, tuple)) else [_smb]
                _smask_images = [_ImgSL.open(_iosl.BytesIO(
                    _b64sl.b64decode(v))).convert("L") for v in _smbs]
                if len(_smask_images) not in (1, n):
                    raise RuntimeError(
                        "時間可変latent_spatial_mask_b64は1枚または"
                        f"動画と同じ{n}枚が必要です")

                def _latent_mask_at(frame_index):
                    src = _smask_images[0 if len(_smask_images) == 1
                                        else frame_index]
                    mi = src.resize((x0.shape[-1], x0.shape[-2]),
                                    resample=_ImgSL.Resampling.NEAREST)
                    return torch.from_numpy(_npsl.asarray(mi) < 128)

                if len(_smask_images) == 1:
                    _smask = _latent_mask_at(0)
                    _smask_image = _smask_images[0]
                else:
                    # Wan VAEはslot0=frame0、以降は4枚単位。各slotの
                    # 中央寄りフレームの移動maskをlatent固定位置に使う。
                    _smask = torch.stack([
                        _latent_mask_at(0 if s == 0 else min(n - 1, 4 * s - 1))
                        for s in range(x0.shape[2])])
                    _smask_image = _smask_images
                log(f"空間latent固定: {int(_smask.sum())}/"
                    f"{_smask.numel()}セルを固定 (255=生成/0=固定、"
                    + ("時刻別)" if _smask.ndim == 3 else "全時刻共通)"))
            if _smb_low:
                # Low側の持続固定mask。現行工房では背景だけを0=固定、
                # 顔/頭部は255=開放にした1枚を送る。
                if isinstance(_smb_low, (list, tuple)):
                    if len(_smb_low) != 1:
                        raise RuntimeError(
                            "latent_spatial_low_mask_b64は1枚が必要です")
                    _smb_low = _smb_low[0]
                from PIL import Image as _ImgLow
                import base64 as _b64low
                import io as _iolow
                import numpy as _nplow
                _low_img = _ImgLow.open(_iolow.BytesIO(
                    _b64low.b64decode(_smb_low))).convert("L")
                _low_img = _low_img.resize(
                    (x0.shape[-1], x0.shape[-2]),
                    resample=_ImgLow.Resampling.NEAREST)
                _smask_low = torch.from_numpy(
                    _nplow.asarray(_low_img) < 128)
                log(f"Low段階固定: 背景{int(_smask_low.sum())}/"
                    f"{_smask_low.numel()}セルを継続固定、顔/頭部は開放")
            if spatial_empty and _smask is None:
                raise RuntimeError(
                    "latent_spatial_emptyにはlatent_spatial_mask_b64が必要です")
            _source_mix = max(0.0, min(0.5, float(
                req.extra.get("latent_spatial_source_mix") or 0.0)))
            x_t = _spatial_inpaint_latents(
                x0, noise, s0, _smask, empty_generate=spatial_empty,
                generate_source_mix=_source_mix)
            if spatial_empty:
                try:
                    _br = float(pipe.config.boundary_ratio)
                    _bt = _br * float(sched.config.num_train_timesteps)
                    _hi = sum(float(t) >= _bt for t in sched.timesteps)
                    if _smask_low is not None:
                        if _smask_release_last_steps is not None:
                            _fixed_n = tail - _smask_release_last_steps
                            log(f"時点指定インペイント: 最初の{_fixed_n}stepは"
                                "顔/頭部+背景固定 / "
                                f"最後の{_smask_release_last_steps}stepは"
                                "顔/頭部開放・背景のみ固定")
                        else:
                            log(f"段階別インペイント: High={_hi}stepは顔/頭部+"
                                f"背景固定 / Low={tail - _hi}stepは背景のみ固定")
                except (AttributeError, TypeError, ValueError):
                    pass
                log("空間インペイント: 生成領域="
                    + ("純ノイズ" if _source_mix <= 0 else
                       f"元latent {_source_mix:.0%} + ノイズ {1-_source_mix:.0%}")
                    + " / "
                    f"固定領域=時系列参照軌道 / σ0={s0:.2f} "
                    f"実行{tail}/{steps}step"
                    + (" (High最終1stepから開始)"
                       if _start_at_high_edge else ""))
            else:
                log(f"リファイン: {len(frames)}f σ0={s0:.2f} "
                    f"実行{tail}/{steps}step (全stepがLow=ディテール側)")
            pin_frames = req.extra.get("latent_pin_frames") or []
            pin_slots = _latent_pin_slots(pin_frames, x0.shape[2])
            pin_release = float(req.extra.get("latent_pin_release") or 0.0)
            if pin_slots:
                log(f"latent固定: フレーム{list(pin_frames)}"
                    f" -> 時間スロット{pin_slots}"
                    + (f" (σ<{pin_release:.2f}で解放=質感馴染ませ)"
                       if pin_release > 0
                       else " (終端まで固定=stage1へ厳密着地)"))
            # 条件画像は既定で「1段目動画の先頭フレーム」。ただし
            # refine_cond_still 指定時は原画の立ち絵を条件にする:
            # 1段目の劣化(量子化+蒸留の粘土)がノイズ済みlatentの中では
            # 「正常な構造」に見えてしまい、リファインが安定した粘土に
            # 確定する (2026-07-13ユーザー指摘)。原画をframe0に注入すると
            # 時系列アテンションが劣化前の質感を全フレームから参照できる。
            # 前提=骨格の直立プレフィックス (frame0が直立なので立ち絵と
            # 矛盾しない。旧動画=全フレーム歩行に使うと先頭が跳ねる)
            cond_img = (frames[0] if frames
                        else _fit_image(req.images[0], w, h))
            if req.extra.get("refine_cond_still") and req.images:
                cond_img = _fit_image(req.images[0], w, h)
                log("リファイン条件画像: 原画立ち絵 (粘土を正常と誤認"
                    "させないための参照注入)")
            cb = _step_callback(progress, tail)
            if pin_slots or _smask is not None:
                cb = _pin_step_callback(cb, sched, x0, noise,
                                        pin_slots, pin_release,
                                        smask=_smask,
                                        smask_low=_smask_low,
                                        smask_release_last_steps=(
                                            _smask_release_last_steps))
            kw = dict(image=cond_img,
                      width=w, height=h, num_frames=n,
                      num_inference_steps=steps,
                      guidance_scale=float(req.guidance),
                      latents=x_t,
                      generator=g,
                      output_type="np", return_dict=False,
                      callback_on_step_end=cb)
            if len(req.images) >= 2:
                kw["last_image"] = _fit_image(req.images[-1], w, h)
                log("リファイン終端画像アンカー: 隣接方向へ着地")
            kw.update(self._prompt_kwargs(self._build_prompt(req, log),
                                          req.negative or None,
                                          req.guidance, log))
            # 空間インペイントとPoseを同時使用する。x_tと
            # callbackは画像空間の固定/生成領域を担当し、この
            # hookはモデルへ渡す36ch条件だけを切り替える。
            _guide_raw = list(
                req.extra.get("anisora_guidance_frames_b64") or [])
            _guide_frames = (load_images_b64(_guide_raw)
                             if _guide_raw else [])
            _image_guide_raw = list(
                req.extra.get("anisora_image_guidance_frames_b64") or [])
            _image_guide_frames = (load_images_b64(_image_guide_raw)
                                   if _image_guide_raw else [])
            if _guide_frames:
                orig_guide_prep = pipe.prepare_latents
                _base_condition = {}
                _spatial_guide = bool(
                    req.extra.get("anisora_guidance_spatial_condition")
                    and _smask_image is not None)
                _dual_guide = bool(
                    req.extra.get("anisora_dual_condition"))
                _spatial_masks = (_smask_image
                                  if isinstance(_smask_image, list)
                                  else [_smask_image])
                _fixed_guide_frames = (frames if frames else [cond_img])

                def _guide_prep(*a4, **k4):
                    _lat, _cond = orig_guide_prep(*a4, **k4)
                    _plain = _cond.detach().clone()
                    if _image_guide_frames:
                        # Poseと同じ16chに画像を混ぜず、画像forward
                        # 専用の条件列を作る。回転プローブでは
                        # 8方向の立ち絵列を入れ、Pose列と別予測する。
                        self._inject_guidance_video(
                            _plain, _image_guide_frames, n, w, h, log,
                            req.extra.get(
                                "anisora_image_guidance_mask", "first"))
                    self._inject_guidance_video(
                        _cond, _guide_frames, n, w, h, log,
                        req.extra.get("anisora_guidance_mask", "known"),
                        generate_masks=(_spatial_masks
                                        if _spatial_guide else None),
                        fixed_frames=(_fixed_guide_frames
                                      if _spatial_guide else None),
                        neutralize_black=bool(req.extra.get(
                            "anisora_guidance_neutralize_black")))
                    if _dual_guide:
                        # 二条件forwardの画像側とLowは、通常の
                        # first-frame I2V条件をそのまま使う。Poseと同じ
                        # 16chに詰め合わせないことがこの経路の契約。
                        _base_condition["value"] = _plain
                    elif (_spatial_guide
                            and req.extra.get("anisora_guidance_release_low")):
                        # Lowでは骨を外すが、Highと同じ空間既知
                        # maskは残す。mask=0でも条件latentの像は
                        # 弱く参照されるため、体に元絵を繰り返さず、
                        # 画像正規化後の0にほぼ相当する中立灰を入れる。
                        _low = _plain.detach().clone()
                        from PIL import Image as _ImgLowGuide
                        _neutral = [_ImgLowGuide.new(
                            "RGB", (w, h), (128, 128, 128))]
                        self._inject_guidance_video(
                            _low, _neutral, n, w, h, log,
                            "known", generate_masks=_spatial_masks,
                            fixed_frames=_fixed_guide_frames)
                        _base_condition["value"] = _low
                    else:
                        _base_condition["value"] = _plain
                    return _lat, _cond

                pipe.prepare_latents = _guide_prep
                if _dual_guide:
                    guide_high_model = getattr(pipe, "transformer", None)
                    if guide_high_model is not None:
                        orig_guide_high_forward = guide_high_model.forward
                        _zdim_high = int(pipe.vae.config.z_dim)
                        _image_weight = max(0.0, min(0.75, float(
                            req.extra.get(
                                "anisora_dual_condition_image_weight", 0.25))))
                        _image_weights_raw = req.extra.get(
                            "anisora_dual_condition_image_weights")
                        _image_weights = None
                        if (isinstance(_image_weights_raw, list)
                                and _image_weights_raw):
                            try:
                                _image_weights = [max(0.0, min(0.75, float(x)))
                                                  for x in _image_weights_raw]
                            except (TypeError, ValueError):
                                _image_weights = None
                        _dual_logged = [False]

                        def _dual_high_forward(*a4, **k4):
                            # 1) Pose条件のHigh予測
                            _pose_out = orig_guide_high_forward(*a4, **k4)
                            _plain = _base_condition.get("value")
                            _hidden = k4.get("hidden_states")
                            _from_args = _hidden is None and bool(a4)
                            if _from_args:
                                _hidden = a4[0]
                            if (_plain is None or _hidden is None
                                    or _hidden.shape[1] < (
                                        _zdim_high + _plain.shape[1])):
                                return _pose_out

                            # 2) 同じnoisy latentに通常画像条件を与えた
                            #    別forward。Pose画像はこちらに入らない。
                            _image_hidden = _hidden.clone()
                            _image_hidden[
                                :, _zdim_high:_zdim_high + _plain.shape[1]] = (
                                    _plain.to(device=_hidden.device,
                                              dtype=_hidden.dtype))
                            if _from_args:
                                _ia = (_image_hidden,) + tuple(a4[1:])
                                _ik = dict(k4)
                            else:
                                _ia = a4
                                _ik = dict(k4)
                                _ik["hidden_states"] = _image_hidden
                            _image_out = orig_guide_high_forward(*_ia, **_ik)
                            _weight = _image_weight
                            if (_image_weights is not None
                                    and _pose_out[0].ndim >= 3):
                                # Wanのnoise predictionはB,C,T,H,W。入力の
                                # 画素フレーム列をlatent時刻へ最近傍対応し、
                                # 安定歩行と旋回でPose/画像比を変えられるようにする。
                                import torch as _torch_dual
                                _t = int(_pose_out[0].shape[2])
                                _src_n = len(_image_weights)
                                if _t <= 1:
                                    _idx = _torch_dual.zeros(
                                        1, dtype=_torch_dual.long,
                                        device=_pose_out[0].device)
                                else:
                                    _idx = _torch_dual.linspace(
                                        0, _src_n - 1, _t,
                                        device=_pose_out[0].device
                                    ).round().long()
                                _weight = _torch_dual.tensor(
                                    _image_weights,
                                    device=_pose_out[0].device,
                                    dtype=_pose_out[0].dtype
                                )[_idx].view(1, 1, _t, 1, 1)
                            _mixed = ((1.0 - _weight) * _pose_out[0]
                                      + _weight * _image_out[0])
                            if not _dual_logged[0]:
                                if _image_weights is None:
                                    log("AniSora二条件High: Poseと画像を別forward "
                                        f"(Pose {1-_image_weight:.0%} / "
                                        f"画像 {_image_weight:.0%})")
                                else:
                                    log("AniSora二条件High: Poseと画像を別forward "
                                        "(フレーム別配分: 画像 "
                                        f"{min(_image_weights):.0%}--"
                                        f"{max(_image_weights):.0%})")
                                _dual_logged[0] = True
                            return (_mixed,) + tuple(_pose_out[1:])

                        guide_high_model.forward = _dual_high_forward
                    else:
                        log("AniSora二条件HighはHighモデルが無いため"
                            "スキップ")
                if req.extra.get("anisora_guidance_release_low"):
                    guide_low_model = getattr(pipe, "transformer_2", None)
                    if guide_low_model is not None:
                        orig_guide_low_forward = guide_low_model.forward
                        _zdim = int(pipe.vae.config.z_dim)
                        _release_logged = [False]

                        def _guide_low_forward(*a4, **k4):
                            _base = _base_condition.get("value")
                            _hidden = k4.get("hidden_states")
                            _from_args = _hidden is None and bool(a4)
                            if _from_args:
                                _hidden = a4[0]
                            if (_base is not None and _hidden is not None
                                    and _hidden.shape[1] >= (
                                        _zdim + _base.shape[1])):
                                _replaced = _hidden.clone()
                                _replaced[:, _zdim:_zdim + _base.shape[1]] = (
                                    _base.to(device=_hidden.device,
                                             dtype=_hidden.dtype))
                                if _from_args:
                                    a4 = (_replaced,) + tuple(a4[1:])
                                else:
                                    k4["hidden_states"] = _replaced
                                if not _release_logged[0]:
                                    log("Pose空間インペイント: Highは"
                                        "歩行骨 / Lowは通常I2V条件")
                                    _release_logged[0] = True
                            return orig_guide_low_forward(*a4, **k4)

                        guide_low_model.forward = _guide_low_forward
                    else:
                        log("PoseのLow解放は二段モデル専用のため"
                            "このモデルではスキップ")
            # latents未対応の古いdiffusersなら明示エラーにする (黙って
            # 落とすと「1段目を無視した素の生成」が静かに走る偽PASS)。
            # latent固定が要求されているときは callback_on_step_end も
            # 必須扱い — 黙って外すと「固定なしの再加工」が静かに走る
            out = _call_with_optional_kwargs(
                self.pipe, kw,
                [] if (pin_slots or _smask is not None)
                else ["callback_on_step_end"], log)
        finally:
            sched.set_timesteps = orig_set
            if orig_guide_prep is not None:
                pipe.prepare_latents = orig_guide_prep
            if (orig_guide_high_forward is not None
                    and guide_high_model is not None):
                guide_high_model.forward = orig_guide_high_forward
            if (orig_guide_low_forward is not None
                    and guide_low_model is not None):
                guide_low_model.forward = orig_guide_low_forward
            if not had_tiling:
                try:
                    pipe.vae.disable_tiling()
                except Exception:
                    pass
            if te_offloaded:
                # encode中の例外でUMT5がCPUに取り残されると、以降の全ジョブ
                # がdevice不一致で死ぬ (2026-07-13実障害: OOM後の連鎖)
                try:
                    te.to(dev)
                except Exception:
                    self.unload(log)
            for m in moved_back:      # High側を戻す (通常のanisoraジョブ用)
                try:
                    m.to(dev)
                except Exception as e:
                    log(f"High側の復帰に失敗 (次ジョブで再ロード): {e}")
                    self.unload(log)
        progress(0.92)
        out_frames = list(out[0][0])
        if (req.extra.get("latent_spatial_pixel_lock")
                and _smask_image is not None):
            out_frames = _pixel_lock_spatial_frames(
                out_frames,
                frames if isinstance(_smask_image, list) else cond_img,
                _smask_image)
            log("空間画素固定: デコード後の顔/頭部帯/背景を"
                "各時刻の参照画素へ復元")
        return _frames_to_mp4(out_frames, req.fps, workdir, log)


_GGML_BF16 = 30      # gguf.GGMLQuantizationType.BF16 (ggml固定値)


def _gguf_storage_only(x) -> bool:
    """ブロック量子化でない(=素の数値保存)テンソルか。

    diffusers 0.39のGGUFロードは F32/F16 を素のテンソル(quant_type無し)、
    BF16 を生バイトのGGUFParameter(quant_type=30) で保持する。"""
    q = getattr(x, "quant_type", None)
    if q is None:
        return True
    try:
        return int(q) == _GGML_BF16
    except Exception:
        return False


def _gguf_as_plain(v):
    """非ブロック量子化テンソルを素のtorchテンソルに変換 (不可ならNone)。"""
    import torch
    q = getattr(v, "quant_type", None)
    if q is None:
        return v
    try:
        if int(q) == _GGML_BF16:
            # BF16は生バイト(uint8)格納 -> 論理形状のbf16テンソルへ
            return v.data.view(torch.bfloat16)
    except Exception:
        return None
    return None


def _transplant_base_weights(target, donor, log, tag: str = "",
                             patch_mode: str = "fun",
                             min_keys: int = 800) -> tuple:
    """AniSora(Wan2.2-I2V系finetune)のbase重みをVACE transformerへ移植する。

    コミュニティ定石「AniSoraベース+VACEモジュール」のdiffusers版
    (2026-07-12設計)。diffusers変換後のパラメタ名は WanTransformer3DModel
    と WanVACETransformer3DModel の共有部分(blocks.* / condition_embedder.*
    / proj_out / scale_shift_table)で完全一致し(diffusers 0.39.0ソース+
    GGUFヘッダ実測で確認)、vace_* はドナーに存在しないため自動的に
    VACE-Fun側が温存される。

    - patch_embedding: I2V=36ch / VACE=16ch で形状が違うため既定では移植
      せず VACE-Fun側を維持。patch_mode="same" はWan-Animate等の同じ
      36ch構造へそのまま移植する。patch_mode="slice" は先頭16ch
      (=ノイズlatent側。I2Vの条件mask4+latent16chは後方連結)を切り出して
      移植する実験モード。
    - GGUF量子化テンソル(GGUFParameter)は named_parameters のオブジェクトを
      そのまま _parameters へ差し替えて quant_type ごと引き継ぐ
      (load_state_dict(assign=True) は Parameter を再構築するため
      quant_type 属性が失われる恐れがある)。
    - 形状/quant_type の不一致はスキップ(破損防止)。移植キー数が min_keys
      未満なら「命名不一致で空振り=素のVACE-Funが静かに動く」偽PASSを
      防ぐため例外で停止する。

    戻り値 = (移植キー数, スキップキー数)。
    """
    import torch
    tgt = dict(target.named_parameters())
    dn = dict(donor.named_parameters())
    take, skipped, coerced = {}, [], []
    for k, v in dn.items():
        if k.startswith("patch_embedding.") and patch_mode != "same":
            skipped.append(k)              # slice指定時は後段で個別処理
            continue
        t = tgt.get(k)
        if t is None:
            skipped.append(k)
            continue
        if (tuple(t.shape) == tuple(v.shape)
                and getattr(t, "quant_type", None)
                == getattr(v, "quant_type", None)):
            take[k] = v
            continue
        # 保存形式だけの差は素のParameterに揃えて移植する。実測:
        # AniSora=F32(素のテンソル) / VACE-Fun=BF16(GGUF格納) が
        # proj_out・text_embedder・time_proj 等6テンソルで食い違う
        # (2026-07-12 両GGUFヘッダ実測。time_embedder系はfp32昇格で両側
        # 素のまま一致)。素通しでスキップすると出力ヘッドだけVACE-Funの
        # キメラになる。GGUFLinearは素のParameterも扱える (AniSora
        # アダプタのF32 headで実績)。
        pv = _gguf_as_plain(v)
        if (pv is not None and _gguf_storage_only(t)
                and tuple(pv.shape)
                == tuple(getattr(t, "quant_shape", t.shape))):
            take[k] = torch.nn.Parameter(pv.clone(), requires_grad=False)
            coerced.append(k)
            continue
        skipped.append(k)
    if len(take) < min_keys:   # 実物の期待値: 40層×約27 + 埋め込み ≈ 1100
        raise RuntimeError(
            f"AniSora移植[{tag}]が空振りです (一致{len(take)}キー / "
            f"スキップ{len(skipped)})。キー命名の不一致の疑い。素のVACE-Fun"
            "のまま静かに生成されてしまうため停止します")
    # patch_embedding以外のスキップは想定外 (=キメラ化)。少数なら警告、
    # 多数なら停止 (「静かに混ざったモデル」で品質FAILの原因調査を
    # 迷宮入りさせないため。2026-07-12レビュー指摘)
    unexpected = [k for k in skipped if not k.startswith("patch_embedding.")]
    if unexpected:
        det = ", ".join(
            f"{k}(t={getattr(tgt.get(k), 'quant_type', None)}/"
            f"d={getattr(dn.get(k), 'quant_type', None)})"
            for k in unexpected[:8])
        if len(unexpected) > max(4, int(0.02 * len(dn))):
            raise RuntimeError(
                f"AniSora移植[{tag}]: 想定外スキップが{len(unexpected)}キー "
                f"({det} ...)。量子化レイアウトの相違でキメラ化するため"
                "停止します")
        log(f"⚠ AniSora移植[{tag}]: 想定外スキップ{len(unexpected)}キー "
            f"({det}) — このテンソルはVACE-Fun側のまま残ります")
    if patch_mode == "slice":
        w, tw = dn.get("patch_embedding.weight"), tgt.get("patch_embedding.weight")
        b, tb = dn.get("patch_embedding.bias"), tgt.get("patch_embedding.bias")
        sliceable = (w is not None and tw is not None
                     and getattr(w, "quant_type", None) is None
                     and getattr(tw, "quant_type", None) is None
                     and w.dim() == 5 and tw.dim() == 5
                     and w.shape[0] == tw.shape[0]
                     and w.shape[1] >= tw.shape[1]
                     and w.shape[2:] == tw.shape[2:])
        if sliceable:
            sw = w[:, : tw.shape[1]].clone().to(dtype=tw.dtype)
            take["patch_embedding.weight"] = torch.nn.Parameter(
                sw, requires_grad=False)
            if b is not None and tb is not None and b.shape == tb.shape:
                take["patch_embedding.bias"] = torch.nn.Parameter(
                    b.clone().to(dtype=tb.dtype), requires_grad=False)
            log(f"AniSora移植[{tag}]: patch_embedding 先頭{tw.shape[1]}ch"
                "スライスを移植 (実験モード)")
        else:
            log(f"AniSora移植[{tag}]: patch_embeddingはスライス不可 "
                "(量子化/形状) -> VACE-Fun側を維持")
    skipped = [k for k in skipped if k not in take]   # slice成功分を除外
    with torch.no_grad():
        for k, v in take.items():
            mod_path, _, leaf = k.rpartition(".")
            mod = target.get_submodule(mod_path) if mod_path else target
            mod._parameters[leaf] = v
    kept_control = sum(
        1 for k in tgt
        if k.startswith(("vace_", "pose_patch_embedding.",
                         "motion_encoder.", "face_encoder.",
                         "face_adapter.")))
    log(f"AniSora移植[{tag}]: {len(take)}キー移植"
        f"{f' (うち保存形式変換{len(coerced)})' if coerced else ''} / "
        f"スキップ{len(skipped)} "
        f"({', '.join(skipped[:3])}{' ...' if len(skipped) > 3 else ''}) / "
        f"制御専用層温存 {kept_control}キー")
    return len(take), len(skipped)


@register
class VACEAdapter(_WanA14BBase):
    """VACE骨格制御 × AniSora V3.2ベース — OpenPose骨格によるポーズ駆動 i2v。

    AniSora i2vの斜め後ろ(back_left/back_right)は、プロンプトロック・
    motion減速・終端アンカー・中間キーフレーム5点拘束の全てを貫通して
    「拘束点の合間で一回転する」抜け道が塞げなかった(2026-07-12 ロップで
    実証)。VACEは全フレームのポーズを骨格で指定するため回転は定義上
    起こり得ない。

    v0.4.xの素のWan2.2 VACE-Fun(汎用ベース・非蒸留30step/cfg5)は
    「粘土のような崩れ+激遅」でAniSoraに遠く及ばず不採用(2026-07-12
    ユーザー裁定)。v0.5.0からはコミュニティ定石「AniSoraベース+VACE
    モジュール」をdiffusersで再現する: VACE-Fun GGUFのtransformerに
    AniSora V3.2 GGUFのbase重み(blocks/condition_embedder/proj_out、
    40層で命名・形状が完全一致)を移植し、vace_*ブロックと16ch
    patch_embeddingだけVACE-Fun由来を残す。アニメprior+8step蒸留のまま
    骨格制御が効く構成(kijaiのVACEモジュール抽出と同じ発想の逆向き)。

    リクエスト契約:
      images[0]                  = 参照キャラ立ち絵 (reference_images)
      extra["pose_frames_b64"]   = OpenPose骨格フレーム列 (base64 PNGリスト、
                                   engine/pose_video.py が生成)
      images[1:]                 = 上記の代替 (webUI手動テスト用)
      extra["conditioning_scale"]= 制御強度 (既定1.0。<1.0は骨が浮くので
                                   下げないこと — 2026-07-12実証)
      extra["vace_end"]          = 骨格制御を適用するステップ割合 (既定1.0。
                                   0.6=序盤60%のみ適用・終盤解放。骨転写の
                                   第一対策。VIDEOLAB_VACE_END)
    実験ノブ (extra > 環境変数 > 既定):
      vace_base    = anisora(既定)|fun    VIDEOLAB_VACE_BASE  移植の有無
      vace_experts = both(既定)|high|low  VIDEOLAB_VACE_EXPERTS  半移植
      vace_patch   = fun(既定)|slice      VIDEOLAB_VACE_PATCH
    骨転写/濁りの後退ラダー: vace_end 0.6→0.4 → vace_experts=low →
    vace_patch=slice → vace_base=fun(steps30/cfg5)。
    """
    id = "vace"
    label = "VACE骨格制御 (AniSoraベース移植・GGUF)"
    desc = ("AniSora V3.2のbase重みをWan2.2 VACE-Funのtransformerへ移植した"
            "骨格制御i2v — アニメprior+8step蒸留のままOpenPose骨格動画で全"
            "フレームのポーズ・向きを直接指定する(向き回転の根絶用)。"
            "images[0]=参照立ち絵、骨格は extra pose_frames_b64 か images "
            "2枚目以降で渡す。extra例: {\"conditioning_scale\": 1.0, "
            "\"vace_base\": \"fun\"(移植なしの素のVACE-Fun・steps30/cfg5"
            "推奨)}。量子化は共通quant (Q4_0既定/Q8_0は明示時のみ。AniSora "
            "GGUFはこの2種のみ)。bf16は移植なし=素のVACE-Fun")
    requires = ("Colab A100/L4 / ローカル24GB+ (Q4_0) — DL 75GB前後 "
                "(VACE-Fun 31GB + AniSora 32GB + ベース12GB)。"
                "移植中はRAMを一時+16GB(Q8)使用")
    modes = ("i2v",)
    repo = os.environ.get("VIDEOLAB_VACE_REPO",
                          "linoyts/Wan2.2-VACE-Fun-14B-diffusers")
    gguf_repo = "QuantStack/Wan2.2-VACE-Fun-A14B-GGUF"
    anisora_gguf_repo = "QuantStack/Index-Anisora-V3.2-GGUF"
    cache_repos = ("QuantStack/Wan2.2-VACE-Fun-A14B-GGUF",
                   "QuantStack/Index-Anisora-V3.2-GGUF",
                   "linoyts/Wan2.2-VACE-Fun-14B-diffusers")
    disk_gb = 75         # VACE GGUF ~31 + AniSora GGUF ~32 + ベース部品 ~12
    flow_shift = float(os.environ.get("VIDEOLAB_VACE_SHIFT", "5.0"))
    # 既定はAniSoraベース移植版の蒸留レシピ (8step/cfg1.0)。
    # vace_base=fun(素のVACE-Fun)で使うときは 30/5.0 を明示すること。
    defaults = {"width": 464, "height": 848, "num_frames": 81, "fps": 16,
                "steps": 8, "guidance": 1.0}
    # Wan公式の標準ネガティブから「静态・静止・静止不动的画面」の
    # アンチ静止語を除いたポーズ制御用。骨格が動きを保証するので静止
    # 防止は不要になり、cfg>1では逆に「動け」圧がマント・髪の二次運動に
    # 集中して強風に煽られたような映像になる (2026-07-13 ロップ14step実測
    # 「強風に煽られるような動画」)。cfg=1.0 (Lightning) では計算に乗らず
    # 無害
    WAN_NEGATIVE = ("色调艳丽,过曝,细节模糊不清,字幕,风格,作品,画作,画面,"
                    "整体发灰,最差质量,低质量,JPEG压缩残留,丑陋的,残缺的,"
                    "多余的手指,画得不好的手部,画得不好的脸部,畸形的,毁容的,"
                    "形态畸形的肢体,手指融合,杂乱的背景,三条腿,"
                    "背景人很多,倒着走")

    def _resolve_want(self, extra: dict) -> tuple:
        """量子化・オフロード・移植構成をリクエスト>環境変数>既定で解決。
        アプリの共通quant設定がそのまま効く(2026-07-12「Q4選んでいても
        おっきいほうが使われます」対応 — 従来vaceはbf16固定だった)。
        戻り値 = (quant, offload, base, experts, patch)。"""
        e = extra or {}
        # 既定Q4_0 (2026-07-13方針: 16GB未満級を上限に。Q8/bf16は明示時のみ)。
        # v0.9.12でQ3_K_S等の下位quantも受理 (VACE-FunはHigh/Low対称で
        # Q3_K_S〜Q8_0の棚在庫あり。Q2_Kは不在=プリフライトが即エラー)
        q = _norm_quant(e.get("quant")
                        or os.environ.get("VIDEOLAB_VACE_QUANT", "Q4_0"),
                        allow_bf16=True)
        off = str(e.get("offload")
                  or os.environ.get("VIDEOLAB_OFFLOAD", "")).lower()
        # "model"(=model_cpu_offload) も受理する: エンジンは大判の
        # グリッド生成でVRAM<60GBのとき offload=model を明示するが、
        # 従来は"seq"以外を握りつぶして常駐へ落とし、720x1296の
        # 活性化~21GBでOOM->方向別フォールバックしていた
        # (2026-07-13実障害)
        if off in ("seq", "sequential"):
            off = "seq"
        elif off in ("model", "offload"):
            off = "model"
        elif off in ("block", "group"):
            off = "block"    # v0.8.2: 重みCPU常駐+ブロック転送 (GGUF向け)
        else:
            off = ""
        base = str(e.get("vace_base")
                   or os.environ.get("VIDEOLAB_VACE_BASE", "anisora")).lower()
        if base not in ("anisora", "fun"):
            base = "anisora"
        if q == "bf16":
            # bf16はGGUFLinearを持たないため量子化ドナーを差し込めない
            # → 移植なし(素のVACE-Fun)に固定。品質A/B比較の逃げ道。
            base = "fun"
        experts = str(e.get("vace_experts")
                      or os.environ.get("VIDEOLAB_VACE_EXPERTS",
                                        "both")).lower()
        if experts not in ("both", "high", "low"):
            experts = "both"
        patch = str(e.get("vace_patch")
                    or os.environ.get("VIDEOLAB_VACE_PATCH", "fun")).lower()
        if patch not in ("fun", "slice"):
            patch = "fun"
        lora = str(e.get("vace_lora")
                   or os.environ.get("VIDEOLAB_VACE_LORA", "")).lower()
        if lora not in ("lightning", "anisora_high"):
            lora = ""
        if base != "anisora":
            # 移植なしでは experts/patch は不活性 — 正規化してノブ操作
            # だけの無駄な積み替え(数分)を防ぐ
            experts, patch = "both", "fun"
        else:
            # LoRAはいずれも素のVACE-Funへ載せる比較経路。AniSora本体を
            # 移植した上へ重ねる二重適用はしない。
            lora = ""
        return q, off, base, experts, patch, lora

    def _ensure_loaded_impl(self, log):
        _require_deps(log)
        import gc
        import torch
        try:
            from diffusers import WanVACEPipeline
        except ImportError:
            raise RuntimeError(
                "この diffusers には WanVACEPipeline がありません。"
                'pip install -U "diffusers>=0.39.0" を実行してから'
                "サーバを再起動してください")
        want = self._resolve_want(getattr(self, "_next_extra", {}))
        quant, offload, base, experts, patch, lora = want
        pipe_kwargs = {}
        self._te_lazy = _low_ram_vm()
        if self._te_lazy:
            # v0.8.2: 低RAM VMではUMT5を持たずにロード (AniSora側と同じ)
            pipe_kwargs["text_encoder"] = None
            log("低RAM VM: UMT5はロードせずジョブ内で一時ロードします")
        if quant == "bf16":
            # フル精度: 常駐70GB+81f attentionの活性化~9GBはA100-80にも
            # 入らない(2026-07-12実測: 残り324MiBでOOM)ため余白12GBで判定
            # -> 実質オフロード運用。素のVACE-Fun品質のA/B比較用の逃げ道
            # (AniSora移植はGGUF経路のみ。_resolve_wantがbase=funに固定)
            log(f"読み込み開始: {self.repo} (bf16・素のVACE-Fun・DL約70GB)")
            self.pipe = WanVACEPipeline.from_pretrained(
                self.repo, torch_dtype=_pick_dtype(), **pipe_kwargs)
            fp, margin = 79, 12
        else:
            from diffusers import GGUFQuantizationConfig
            from huggingface_hub import hf_hub_download
            try:
                from diffusers import WanVACETransformer3DModel
            except ImportError:
                raise RuntimeError(
                    "この diffusers には WanVACETransformer3DModel が"
                    'ありません。pip install -U "diffusers>=0.39.0" を'
                    "実行してからサーバを再起動してください")
            gguf_high = (f"HighNoise/Wan2.2-VACE-Fun-A14B-high-noise-"
                         f"{quant}.gguf")
            gguf_low = (f"LowNoise/Wan2.2-VACE-Fun-A14B-low-noise-"
                        f"{quant}.gguf")
            # 在庫プリフライト (P2): 棚に無いquant (Q2_K等) は多段DLの
            # 数分のリトライを燃やす前に即エラーで知らせる
            if quant not in ("Q4_0", "Q8_0") \
                    and _hf_file_missing(self.gguf_repo, gguf_high):
                raise RuntimeError(
                    f"{self.gguf_repo} に {quant} の在庫がありません "
                    "(棚在庫: Q3_K_S/Q3_K_M/Q4_0/Q4_K_S/Q4_K_M/Q5_K_S/"
                    "Q5_0/Q5_K_M/Q6_K/Q8_0)")
            qc = GGUFQuantizationConfig(compute_dtype=_pick_dtype())
            log(f"GGUF DL: {self.gguf_repo} {quant} "
                f"(High+Low 各{'15.5GB' if quant == 'Q8_0' else '8.5GB前後'})")
            hi = _hf_download(self.gguf_repo, gguf_high, log)
            lo = _hf_download(self.gguf_repo, gguf_low, log)
            # ベース部品はsnapshot経由 (v0.8.7: Driveキャッシュ+429対策)
            snap = _snapshot_local(self.repo, log)
            # ロード前RAMゲート (P0-2): VACE2体 + (移植時) ドナー1体分の
            # 同時常駐がロードピーク。サイズ不明 (テスト等) はゲートしない
            _vgg = _gguf_gb(hi, lo)
            if _vgg >= 1.0:
                _donor = 16.0 if quant == "Q8_0" else 9.3
                # ベース部品のRAM実装はVAE等≈2.5GB (UMT5は別項)
                _ram_gate(log, _vgg + (_donor if base == "anisora" else 0)
                          + 2.5 + (0 if self._te_lazy else 11) + 6,
                          f"VACE {quant} 読み込み")
            log("transformer(HighNoise) 読み込み — configはVACEベースを明示")
            t_hi = WanVACETransformer3DModel.from_single_file(
                hi, quantization_config=qc, config=snap,
                subfolder="transformer", torch_dtype=_pick_dtype())
            log("transformer_2(LowNoise) 読み込み")
            t_lo = WanVACETransformer3DModel.from_single_file(
                lo, quantization_config=qc, config=snap,
                subfolder="transformer_2", torch_dtype=_pick_dtype())
            if base == "anisora":
                # AniSora V3.2のbase重みを移植 (High->HighNoise、
                # Low->LowNoise のエキスパート対応)。ドナーはCPU上で
                # 読み込み→移植→即解放 (Q8で一時+16GB)
                from diffusers import WanTransformer3DModel
                # ドナーのconfigも実体dir経由 (v0.10.3、2026-07-19監査):
                # 以前は config=self.base_repo とリポIDを渡していたため、
                # diffusersがtransformer/config.jsonをHubから直に引きに
                # 行き、_snapshot_local/_drive_only()を丸ごと迂回していた
                # (walk_packはvace_base="fun"固定で踏まないが、webUIの
                # 手動vaceジョブとlatent_refine以外の経路は既定
                # VIDEOLAB_VACE_BASE="anisora" でここへ来る)
                snap_base = _snapshot_local(self.base_repo, log)
                for tgt, tag, sub in ((t_hi, "High", "transformer"),
                                      (t_lo, "Low", "transformer_2")):
                    if experts != "both" and experts != tag.lower():
                        log(f"AniSora移植[{tag}]: スキップ "
                            f"(vace_experts={experts})")
                        continue
                    # AniSora High側の棚在庫はQ4_0/Q8_0のみ — 静的規則で
                    # 代替 (_anisora_high_quant参照、v0.9.13)
                    dq = (_anisora_high_quant(quant) if tag == "High"
                          else quant)
                    fname = f"{tag}/Index-Anisora-V3.2-{tag}-{dq}.gguf"
                    if dq != quant:
                        log(f"AniSora移植[High]: {quant}の棚在庫が無いため"
                            f"{dq}ドナーで代替します")
                    log(f"AniSora GGUF DL: {self.anisora_gguf_repo} {fname}")
                    dp = _hf_download(self.anisora_gguf_repo, fname, log)
                    log(f"AniSora {tag} 読み込み (移植ドナー・config="
                        "Wan2.2-I2Vベースを明示)")
                    donor = WanTransformer3DModel.from_single_file(
                        dp, quantization_config=qc, config=snap_base,
                        subfolder=sub, torch_dtype=_pick_dtype())
                    _transplant_base_weights(tgt, donor, log, tag=tag,
                                             patch_mode=patch)
                    del donor
                    gc.collect()
            else:
                log("vace_base=fun: AniSora移植なし (素のVACE-Fun。"
                    "steps30/cfg5.0を明示推奨)")
            log(f"ベース部品 読み込み: {snap} (VAE+UMT5)")
            self.pipe = WanVACEPipeline.from_pretrained(
                snap, transformer=t_hi, transformer_2=t_lo,
                torch_dtype=_pick_dtype(), **pipe_kwargs)
            # GGUF常駐 + UMT5(~6.7GB) + VAE。サイズが取れない環境 (テスト
            # /Drive未DL) は従来の固定表へフォールバック (81f骨格制御の
            # 活性化~9GBはmarginが受け持つ)
            if _vgg >= 1.0:
                fp = _vgg + (0 if self._te_lazy else 6.7) + 1.0
                self._dit_gb = max(_gguf_gb(hi), _gguf_gb(lo))
            else:
                fp = {"Q4_0": 27, "Q8_0": 42}.get(quant, 42)
                if self._te_lazy:
                    fp -= 6      # UMT5不在ぶん常駐判定を緩める
                self._dit_gb = {"Q8_0": 15.5}.get(quant, 10.3)
            margin = 10
            self._te_from = snap
        if quant == "bf16":
            self._te_from = self.repo
            self._dit_gb = 29.0
            if self._te_lazy:
                fp -= 6          # UMT5不在ぶん常駐判定を緩める
        if lora == "lightning":
            # Lightning 4step蒸留LoRA (T2V-A14B用がVACE-Funに適合。
            # cfg=1で回すためCFGパスも消え、14step/cfg5比で約7倍速)。
            # High/Lowの2エキスパートへそれぞれのLoRAを装着する
            rep = os.environ.get("VIDEOLAB_VACE_LORA_REPO",
                                 "lightx2v/Wan2.2-Lightning")
            fold = os.environ.get(
                "VIDEOLAB_VACE_LORA_DIR",
                "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0")
            log(f"Lightning 4step LoRA 読み込み: {rep}/{fold} "
                "(High/Low 各1.2GB)")
            # Colab同梱の古いtorchao(0.10)対策: peftのLoRA適用は全レイヤー
            # にtorchao用dispatcherを試し、is_torchao_available()が
            # 「古い版が入っている」ときFalseでなくImportErrorをraiseする
            # (2026-07-13実障害: torchao 0.10 vs 要求>=0.16)。torchaoは
            # 未使用なのでdispatcherを無効化して回避する
            try:
                from peft.tuners.lora import torchao as _plt
                try:
                    _plt.is_torchao_available()
                except ImportError:
                    _plt.is_torchao_available = lambda: False
                    log("非互換torchaoを検出 -> peftのtorchao dispatcher"
                        "を無効化 (LoRA適用には影響なし)")
            except Exception:
                pass
            try:
                # LoRA本体も多段DL経路で先に取得 (v0.8.7: 429対策+Drive
                # キャッシュ。1.2GB×2なのでサイズ検査は1GiB閾値を通る)
                _hi_p = Path(_hf_download(
                    rep, f"{fold}/high_noise_model.safetensors", log))
                _lo_p = Path(_hf_download(
                    rep, f"{fold}/low_noise_model.safetensors", log))
                self.pipe.load_lora_weights(
                    str(_hi_p.parent), adapter_name="lightning",
                    weight_name=_hi_p.name)
                self.pipe.load_lora_weights(
                    str(_lo_p.parent), adapter_name="lightning_2",
                    weight_name=_lo_p.name,
                    load_into_transformer_2=True)
                self.pipe.set_adapters(["lightning", "lightning_2"],
                                       [1.0, 1.0])
                log("Lightning適用完了: steps=6/cfg=1.0 で運用すること")
            except Exception as e:
                raise RuntimeError(
                    "Lightning LoRAの適用に失敗しました (GGUF量子化との"
                    "組み合わせが原因の可能性)。videolab_pose_lora を外すか "
                    f"quant=bf16 で再試行してください: {str(e)[:300]}")
        elif lora == "anisora_high":
            # 本命実験: AniSora High本体の移植ではなく、Kijaiが配布する
            # High要素抽出LoRAだけをVACE Highへ適用する。transformer_2へは
            # load_into_transformer_2を指定しないため、Lowは素のVACEのまま。
            rep = "Kijai/WanVideo_comfy"
            lora_name = ("LoRAs/AniSora/"
                         "Wan2_2_I2V_AniSora_3_2_HIGH_rank_64_fp16.safetensors")
            log("AniSora High要素抽出LoRA 読み込み: "
                f"{rep}/{lora_name} (VACE Highのみ)")
            try:
                from peft.tuners.lora import torchao as _plt
                try:
                    _plt.is_torchao_available()
                except ImportError:
                    _plt.is_torchao_available = lambda: False
                    log("非互換torchaoを検出 -> peftのtorchao dispatcher"
                        "を無効化 (LoRA適用には影響なし)")
            except Exception:
                pass
            try:
                _hi_p = Path(_hf_download(rep, lora_name, log))
                self.pipe.load_lora_weights(
                    str(_hi_p.parent), adapter_name="anisora_high",
                    weight_name=_hi_p.name)
                self.pipe.set_adapters(["anisora_high"], [1.0])
                log("AniSora High要素LoRA適用完了: VACE Highのみ / "
                    "VACE Lowは無改造")
            except Exception as e:
                raise RuntimeError(
                    "VACE HighへAniSora High要素抽出LoRAを適用できません"
                    f"でした: {str(e)[:300]}")
        # VACEは81fの骨格条件動画を丸ごとVAE encodeするため、常駐でも
        # タイリング必須 (無いとA100-80でもencodeでOOM。2026-07-12実障害)
        self._finalize_pipe(log, offload=offload, footprint_gb=fp,
                            vae_tiling=True, resident_margin_gb=margin,
                            gguf=(quant != "bf16"))
        # AniSoraベース時はmotion scoreプロンプト接尾辞も効く(蒸留ベースの
        # 学習時プロンプト形式。_build_prompt が extra.motion_score を反映)
        self.prompt_suffix = (AniSoraAdapter.prompt_suffix
                              if base == "anisora" else "")
        self.loaded_quant, self.loaded_offload = quant, offload
        self._loaded_want = want
        self.loaded = True

    def generate(self, req: GenRequest, workdir: Path, log, progress) -> Path:
        import torch
        # 量子化/オフロード/移植構成がロード時と違えば積み替え (anisoraと
        # 同様。_loaded_want未設定=外部からpipe注入されたテスト等は温存)
        want = self._resolve_want(req.extra)
        if self.loaded and want != getattr(self, "_loaded_want", want):
            log(f"設定変更 {getattr(self, '_loaded_want', '?')} -> "
                f"{want}: モデルを積み替えます")
            self.unload(log)
            _free_cuda(log)
        if not self.loaded:
            self._next_extra = dict(req.extra or {})
            self.ensure_loaded(log)
        w = _snap(req.width, 16, 240)
        h = _snap(req.height, 16, 240)
        n = max(5, ((int(req.num_frames) - 1) // 4) * 4 + 1)     # 4k+1
        steps = int(req.steps)
        self._admit(w, h, n, log)          # 生成直前VRAMゲート (P0-3)
        if want[2] == "fun" and steps <= 12 and want[5] != "lightning":
            # 非蒸留の素のVACE-Funに蒸留レシピは「保証つきの泥」
            # (Lightning LoRA装着時は蒸留済みなので低stepが正解)
            log(f"⚠ vace_base=fun は非蒸留です: steps={steps}/"
                f"cfg={req.guidance} では崩れます。steps=30 cfg=5.0 を"
                "指定してください。vace_lora=anisora_high はHighだけを変え、"
                "VACE Lowを蒸留しないため、この注意は残ります")
        # vace_end: 骨格制御を序盤ステップに限定し終盤は解放する
        # (kijaiワークフローの end_percent 相当の定石)。ポーズは序盤の
        # 高ノイズ段で確定するため、終盤解放で「骨格がそのまま画面に
        # 転写される」問題を消しにいく (2026-07-12 移植ハイブリッドで
        # 骨転写が発生 — vace残差はVACE-Funベース向け較正のため
        # AniSoraブロック上では骨格消去が効き切らない)
        try:
            end_frac = float(req.extra.get("vace_end")
                             or os.environ.get("VIDEOLAB_VACE_END", "1.0"))
        except (TypeError, ValueError):
            end_frac = 1.0
        end_frac = max(0.05, min(1.0, end_frac))
        pf = list(req.extra.get("pose_frames_b64") or [])
        if pf:
            control = load_images_b64(pf)
        elif len(req.images) > 1:
            control = list(req.images[1:])   # webUI手動テスト用の代替経路
        else:
            raise RuntimeError(
                "vace には骨格制御フレームが必要です: extra.pose_frames_b64 "
                "(base64 PNGのリスト) か images の2枚目以降で渡してください。"
                "SpriteMill本体からは --videolab-pose-dirs back で自動生成")
        got = len(control)
        control = [im if im.size == (w, h) else _fit_image(im, w, h)
                   for im in control]
        if got != n:
            # フレーム数が合わないときは線形リサンプルで合わせる
            idx = [round(i * (got - 1) / max(1, n - 1)) for i in range(n)]
            control = [control[i] for i in idx]
            log(f"制御フレーム数を調整: {got} -> {n}")
        ref = _fit_image(req.images[0], w, h)
        # High段間引き (vace_high_steps): 構図が決まるHigh段は数stepで
        # 足りるという運用向けに、σ>=boundaryのステップを等間隔でN本へ
        # 間引く。UniPCはσ/tを位置参照するため両配列を同じ添字で
        # 組み替えれば任意の単調減少ラダーで動く (2026-07-13、
        # 「Highのステップ数を低くしたい」要望)
        try:
            hs = int(req.extra.get("vace_high_steps")
                     or os.environ.get("VIDEOLAB_VACE_HIGH_STEPS", "0") or 0)
        except (TypeError, ValueError):
            hs = 0
        sched = getattr(self.pipe, "scheduler", None)
        orig_set_ts = getattr(sched, "set_timesteps", None)
        if hs > 0 and sched is not None:
            bnd = float(getattr(getattr(self.pipe, "config", None),
                                "boundary_ratio", None) or 0.9)

            def _set_thin(nsteps, device=None, **kw3):
                orig_set_ts(nsteps, device=device, **kw3)
                sig, ts = sched.sigmas, sched.timesteps
                hi = [i for i in range(len(ts)) if float(sig[i]) >= bnd]
                lo_i = [i for i in range(len(ts)) if float(sig[i]) < bnd]
                if len(hi) <= hs:
                    return
                keep = sorted({int(round(x)) for x in
                               [k * (len(hi) - 1) / max(1, hs - 1)
                                for k in range(hs)]})
                idx = [hi[k] for k in keep] + lo_i
                sched.timesteps = ts[idx]
                sched.sigmas = torch.cat([sig[idx], sig[-1:]])
                sched.num_inference_steps = len(idx)
                log(f"High段間引き: {len(hi)}→{len(keep)}本 "
                    f"(計{len(idx)}step)")

            sched.set_timesteps = _set_thin
        # 序盤限定適用の実装: 完了ステップ数をコールバックで数え、閾値を
        # 越えたら transformer への control_hidden_states_scale をゼロに
        # 差し替える (diffusers 0.39 WanVACEPipeline はループ内で
        # current_model(..., control_hidden_states_scale=テンソル) を呼ぶ)
        step_box = {"i": 0}
        cut = max(1, int(round(steps * end_frac)))
        unwraps = []
        if end_frac < 1.0:
            log(f"骨格制御は序盤 {cut}/{steps} ステップのみ適用 "
                f"(vace_end={end_frac} — 終盤解放で骨転写を抑制)")

            def _wrap(model):
                orig = model.forward

                def fwd(*a, **kw2):
                    s = kw2.get("control_hidden_states_scale")
                    if step_box["i"] >= cut and s is not None:
                        kw2["control_hidden_states_scale"] = s * 0.0
                    return orig(*a, **kw2)
                model.forward = fwd
                unwraps.append((model, orig))
            for m in (getattr(self.pipe, "transformer", None),
                      getattr(self.pipe, "transformer_2", None)):
                if m is not None:
                    _wrap(m)
        base_cb = _step_callback(progress, steps)

        def cb(pipe, step_index, timestep, callback_kwargs):
            step_box["i"] = step_index + 1
            return base_cb(pipe, step_index, timestep, callback_kwargs)

        # latent直出しモード (2026-07-13ユーザー発案): VACEフル制御の
        # 最終latentをVAE未通過のまま保存し、後段のAniSora latent再加工
        # (anisoraジョブの extra.latent_from=このジョブID) へ渡す
        emit_latent = bool(req.extra.get("emit_latent"))
        kw = dict(video=control, reference_images=[ref],
                  width=w, height=h, num_frames=n,
                  num_inference_steps=steps,
                  guidance_scale=float(req.guidance),
                  conditioning_scale=float(
                      req.extra.get("conditioning_scale", 1.0)),
                  generator=torch.Generator("cpu").manual_seed(req.seed),
                  output_type="latent" if emit_latent else "np",
                  return_dict=False,
                  callback_on_step_end=cb)
        kw.update(self._prompt_kwargs(self._build_prompt(req, log),
                                      req.negative or self.WAN_NEGATIVE,
                                      req.guidance, log))
        # キーフレーム補間 (2026-07-20ユーザー発案「実画像アンカー+間は空の
        # 潜在で自然に繋ぐ」— VACE-FunのExtension/Loopと同型): 指定フレームを
        # マスク0=保持 (controlに実画像を置く)、他=255=生成。
        _keep = set(int(x) for x in (req.extra.get("vace_keep_frames") or []))
        if _keep:
            from PIL import Image as _ImgKM
            kw["mask"] = [
                _ImgKM.new("L", (w, h), 0 if i in _keep else 255)
                for i in range(n)]
            log(f"キーフレーム補間: 保持{sorted(_keep)[:8]}"
                f"{'…' if len(_keep) > 8 else ''} / 他{n - len(_keep)}fを生成")
        _smask = req.extra.get("vace_mask_b64")
        if _smask:
            # 空間マスク (2026-07-21ユーザー発案「動かしたい部分だけマスク
            # して空の潜在で埋める」— オランウータン例と同じ標準インペイント
            # 用法): 各フレーム同形のLマスク (0=videoの実画素を保持 /
            # 255=生成)。1枚だけ渡せば全フレームに適用
            from PIL import Image as _ImgSM
            import base64 as _b64s
            import io as _ios
            _mimgs = [
                _ImgSM.open(_ios.BytesIO(_b64s.b64decode(b)))
                .convert("L").resize((w, h))
                for b in (_smask if isinstance(_smask, list) else [_smask])]
            kw["mask"] = ([_mimgs[i % len(_mimgs)] for i in range(n)]
                          if len(_mimgs) > 1
                          else [_mimgs[0]] * n)
            log(f"空間マスク: {len(_mimgs)}枚を{n}fへ適用 "
                "(0=実画素保持/255=生成)")
        log(f"生成開始: {w}x{h} {n}f steps={steps} cfg={req.guidance} "
            f"骨格制御{len(control)}f cond={kw['conditioning_scale']}"
            + (" [latent直出し]" if emit_latent else ""))
        try:
            out = _call_with_optional_kwargs(
                self.pipe, kw,
                ["callback_on_step_end", "conditioning_scale", "mask"],
                log)
        finally:
            for m, orig in unwraps:
                m.forward = orig
            if orig_set_ts is not None:
                sched.set_timesteps = orig_set_ts
        progress(0.92)
        if emit_latent:
            lat = out[0]
            lat = lat if torch.is_tensor(lat) else torch.as_tensor(lat)
            t_expect = (n - 1) // 4 + 1
            if lat.shape[2] > t_expect:      # VACEの参照タイムスロットを除去
                lat = lat[:, :, lat.shape[2] - t_expect:]
            torch.save(lat.detach().to("cpu", torch.float32),
                       Path(workdir) / "latent.pt")
            log(f"最終latent保存: {tuple(lat.shape)} -> latent.pt "
                "(VAE未通過・AniSora latent再加工用)")
            if not req.extra.get("emit_latent_preview"):
                # 本番はプレビューdecodeを省略 (2026-07-14ユーザー指示
                # 「絶対重いし使わないものだから時間無駄」)。ジョブ契約
                # (結果=mp4) は参照画像1フレームのサムネで満たす
                import numpy as np
                thumb = np.asarray(_fit_image(req.images[0], w, h)
                                   .convert("RGB"), dtype=np.float32) / 255.0
                log("プレビューdecode省略 (emit_latent_preview未指定)")
                return _frames_to_mp4([thumb], req.fps, workdir, log)
            # プレビュー兼デバッグ用に一度だけdecode (成果物契約はmp4)。
            # ★offloadフック運用時は重みとlatentのデバイスが食い違う
            # (2026-07-13実障害: CPUBFloat16 vs CUDABFloat16) — CUDAへ
            # 明示的に揃えてからdecodeする
            vae = self.pipe.vae
            with torch.no_grad():
                try:
                    vae.enable_tiling()
                except Exception:
                    pass
                vdev = (torch.device("cuda")
                        if torch.cuda.is_available()
                        else next(vae.parameters()).device)
                try:
                    vae.to(vdev)
                except Exception:      # フックと衝突したら実デバイスに従う
                    vdev = next(vae.parameters()).device
                lm = torch.tensor(vae.config.latents_mean).view(
                    1, -1, 1, 1, 1).to(vdev, vae.dtype)
                ls = 1.0 / torch.tensor(vae.config.latents_std).view(
                    1, -1, 1, 1, 1).to(vdev, vae.dtype)
                dec = vae.decode(lat.to(vdev, vae.dtype) / ls + lm,
                                 return_dict=False)[0]
                video = self.pipe.video_processor.postprocess_video(
                    dec, output_type="np")
            return _frames_to_mp4(list(video[0]), req.fps, workdir, log)
        return _frames_to_mp4(list(out[0][0]), req.fps, workdir, log)


def _low_ram_vm(threshold_gb: float = 60.0) -> bool:
    """システムRAMが少ないVMか (Colab L4=53GB / A100=83GB)。

    latent直結は非稼働モデルをRAMへ退避させる設計のため、低RAM VMでは
    RAM側が先に詰まりランタイムごとOOM killされる。該当時はUMT5の
    ジョブ毎解放などの節約運転へ切り替える。"""
    try:
        import psutil
        return psutil.virtual_memory().total < threshold_gb * 2**30
    except Exception:
        try:
            with open("/proc/meminfo", encoding="ascii") as f:
                kb = int(f.readline().split()[1])
            return kb * 1024 < threshold_gb * 2**30
        except Exception:
            return False


def _slice_handoff_scheduler_state(scheduler, drop_frames: int,
                                   old_frames: int) -> None:
    """VACEの参照用time slotをUniPC履歴からも同時に外す。

    latent本体だけを `[:, :, R:]` にすると、order-2 UniPCが保持する
    model_outputs/last_sampleはT+Rのままで、handoff直後のcorrectorがshape
    不一致になる。timestep_listやstep_indexは時間軸テンソルではないので
    そのまま保持する。
    """
    if drop_frames <= 0:
        return

    def _slice(v):
        if (hasattr(v, "ndim") and v.ndim >= 3
                and int(v.shape[2]) == int(old_frames)):
            return v[:, :, drop_frames:]
        return v

    outputs = getattr(scheduler, "model_outputs", None)
    if isinstance(outputs, list):
        scheduler.model_outputs = [_slice(v) for v in outputs]
    elif isinstance(outputs, tuple):
        scheduler.model_outputs = tuple(_slice(v) for v in outputs)
    last = getattr(scheduler, "last_sample", None)
    if last is not None:
        scheduler.last_sample = _slice(last)


@register
class VACEAniSoraHandoffAdapter(VACEAdapter):
    """VACE High + AniSora要素LoRA -> native AniSora Lowを直結する。

    完成VACE動画をMP4/JPEG化してVAE再encode・再加ノイズする旧refineとは
    異なり、Wan2.2の通常のHigh/Low expert切替と同じUniPC scheduler上で
    noise latentを一度も終了させない。VACEは16ch、AniSora I2Vは
    latent16+参照条件20=36ch入力だが、両者のnoise predictionは16chなので
    scheduler stateをそのまま継続できる。
    """
    id = "vace_anisora_handoff"
    label = "VACE + AniSora要素LoRA → VACE Low → AniSora Low tail"
    desc = ("OpenPose/VACEで構図と歩行を固定し、VACE HighへKijai配布の"
            "AniSora V3.2 High要素抽出LoRAを適用する。同じノイズlatentを"
            "最終1--2stepだけnative AniSora Lowへ直接渡す。AniSora High本体"
            "の移植、Lightning LoRA、中間動画、VAE再encode、再ノイズ化は"
            "行わない。extra hybrid_low_steps=1|2 (既定2)。"
            "hybrid_lightning4=true, hybrid_vace_low_steps=1なら、"
            "4stepをHigh 2→VACE Low 1→AniSora Low 1で処理する。")
    requires = ("Colab L4で十分 (Q4+動的キャンバス設計でほぼフル品質・"
                "料金はA100の約1/5。A100は最速だが贅沢品)。Q4でVACE High"
                "約8.5GB + AniSora Low約9GBを区間ごとにGPUへ載せ替え。"
                "High側へAniSora要素抽出LoRAを適用。DL約46GB。"
                "16GB未満のご家庭GPUはextra offloadでblock offload(実験的)")
    cache_repos = VACEAdapter.cache_repos + ("Kijai/WanVideo_comfy",)
    disk_gb = 46
    defaults = {"width": 464, "height": 848, "num_frames": 49, "fps": 16,
                "steps": 8, "guidance": 1.0}

    def __init__(self):
        super().__init__()
        self.ani_pipe = None

    def unload(self, log):
        self.ani_pipe = None
        self.pipe = None
        self.loaded = False

    @staticmethod
    def _hybrid_want(extra: dict) -> tuple:
        e = extra or {}
        _qraw = (e.get("quant")
                 or os.environ.get("VIDEOLAB_VACE_QUANT", "Q4_0"))
        if str(_qraw).strip().lower() == "bf16":
            _qraw = "Q4_0"     # bf16はGGUF経路に無い — 従来どおりQ4_0で受ける
        q = _norm_quant(_qraw)
        try:
            boundary = float(e.get("hybrid_boundary")
                             or os.environ.get("VIDEOLAB_HYBRID_BOUNDARY",
                                               "0.90"))
        except (TypeError, ValueError):
            boundary = 0.90
        return q, max(0.80, min(0.95, boundary))

    @staticmethod
    def _hybrid_lora(extra: dict) -> str:
        """VACE Highに使うAniSora要素抽出LoRA。offは比較用。"""
        e = extra or {}
        value = str(e.get("vace_lora")
                    or os.environ.get("VIDEOLAB_VACE_LORA", "anisora_high"))
        value = value.strip().lower()
        return "off" if value in ("off", "none", "0", "false") else "anisora_high"

    @staticmethod
    def _hybrid_low_steps(extra: dict, total_steps: int) -> int:
        """回転priorの強いAniSora Lowを終端1--2stepだけに制限する。"""
        e = extra or {}
        try:
            value = int(e.get("hybrid_low_steps")
                        or os.environ.get("VIDEOLAB_HYBRID_LOW_STEPS", "2"))
        except (TypeError, ValueError):
            value = 2
        return max(1, min(2, max(1, int(total_steps) - 1), value))

    @staticmethod
    def _hybrid_vace_low_steps(extra: dict, total_steps: int) -> int:
        """AniSora tailの直前に通すVACE Low step数。通常handoffは0。"""
        e = extra or {}
        try:
            value = int(e.get("hybrid_vace_low_steps", 0) or 0)
        except (TypeError, ValueError):
            value = 0
        return max(0, min(max(0, int(total_steps) - 2), value))

    @staticmethod
    def _hybrid_lightning4(extra: dict) -> bool:
        value = (extra or {}).get("hybrid_lightning4", False)
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    def _ensure_loaded_impl(self, log):
        """必要なexpertだけロードする (VACE High + native AniSora Low)。"""
        _require_deps(log)
        import gc
        import torch
        from diffusers import (GGUFQuantizationConfig,
                               UniPCMultistepScheduler,
                               WanImageToVideoPipeline,
                               WanTransformer3DModel,
                               WanVACEPipeline,
                               WanVACETransformer3DModel)
        from huggingface_hub import hf_hub_download

        next_extra = getattr(self, "_next_extra", {})
        quant, _boundary = self._hybrid_want(next_extra)
        lora = self._hybrid_lora(next_extra)
        lightning4 = self._hybrid_lightning4(next_extra)
        want_vace_low = self._hybrid_vace_low_steps(next_extra, 4) > 0
        qc = GGUFQuantizationConfig(compute_dtype=_pick_dtype())
        vname = (f"HighNoise/Wan2.2-VACE-Fun-A14B-high-noise-"
                 f"{quant}.gguf")
        log(f"latent直結ロード: VACE High {quant}")
        # ベース部品はsnapshot経由 (v0.8.8: Drive固定運転でもhandoffが動く
        # ように。configもここから読む)
        snap_vace = _snapshot_local(self.repo, log)
        snap_base = _snapshot_local(self.base_repo, log)
        vp = _hf_download(self.gguf_repo, vname, log)
        # ロード前RAMゲート (P0-2): vhighロード「前」に判定する。v0.11.41
        # 以降はAniSora High本体の移植ドナーを読まず、VACE High + AniSora
        # Lowの2体と631MB LoRAだけがピークになる。
        _vgg1 = _gguf_gb(vp)
        if _vgg1 >= 1.0:
            _dn = 16.0 if quant == "Q8_0" else 9.3
            # ベース部品のRAM実装はVAE等≈2.5GB (UMT5は別項)
            _ram_gate(log, _vgg1 + _dn
                      + (9.3 if want_vace_low else 0) + 2.5
                      + (0 if _low_ram_vm() else 11) + 6,
                      f"latent直結 {quant} 読み込み")
        vhigh = WanVACETransformer3DModel.from_single_file(
            vp, quantization_config=qc, config=snap_vace,
            subfolder="transformer", torch_dtype=_pick_dtype())
        log("latent直結: AniSora High本体は移植せず、VACE Highを維持")

        vlow = None
        if want_vace_low:
            vl_name = (f"LowNoise/Wan2.2-VACE-Fun-A14B-low-noise-"
                       f"{quant}.gguf")
            log(f"三段handoffロード: VACE Low {quant}")
            vlp = _hf_download(self.gguf_repo, vl_name, log)
            vlow = WanVACETransformer3DModel.from_single_file(
                vlp, quantization_config=qc, config=snap_vace,
                subfolder="transformer_2", torch_dtype=_pick_dtype())

        al_name = f"Low/Index-Anisora-V3.2-Low-{quant}.gguf"
        log(f"latent直結ロード: native AniSora Low {quant}")
        alp = _hf_download(self.anisora_gguf_repo, al_name, log)
        alow = WanTransformer3DModel.from_single_file(
            alp, quantization_config=qc, config=snap_base,
            subfolder="transformer_2", torch_dtype=_pick_dtype())
        self._dit_gb = max(_gguf_gb(vp), _gguf_gb(alp)) or 9.3

        # 低RAM VM (L4=53GB) ではUMT5(約11GB)をロード段階から持たない:
        # ロードピーク=モデル一式+移植ドナー+UMT5≈48GBが天井を叩き、
        # ジョブ2回目の再ロードでランタイムごとRAM OOM killされる実障害
        # (2026-07-13 L4実走で「RAMを増やす」画面を確認)。UMT5はジョブ毎に
        # 遅延ロード→encode後に解放 (v0.7.5の解放/再ロード機構を流用)
        _lowram = _low_ram_vm()
        pipe_kwargs = dict(transformer=vhigh, transformer_2=vlow,
                           torch_dtype=_pick_dtype())
        if _lowram:
            pipe_kwargs["text_encoder"] = None
            log(f"共通VAE読み込み (UMT5はジョブ毎に遅延ロード=低RAM運転): "
                f"{self.repo}")
        else:
            log(f"共通VAE/UMT5読み込み: {self.repo}")
        self.pipe = WanVACEPipeline.from_pretrained(
            snap_vace, **pipe_kwargs)
        self._te_from = snap_vace
        sched = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config, flow_shift=self.flow_shift)
        self.pipe.scheduler = sched
        adapter_names = []
        adapter_weights = []
        if lora == "anisora_high":
            # AniSora High本体ではなく、KijaiがWan2.2 I2V向けに抽出した
            # High要素LoRAをVACE Highの共有blockへだけ装着する。
            rep = "Kijai/WanVideo_comfy"
            lora_name = ("LoRAs/AniSora/"
                         "Wan2_2_I2V_AniSora_3_2_HIGH_rank_64_fp16.safetensors")
            log("latent直結: VACE HighへAniSora High要素抽出LoRA適用 "
                f"({rep}/{lora_name})")
            try:
                from peft.tuners.lora import torchao as _plt
                try:
                    _plt.is_torchao_available()
                except ImportError:
                    _plt.is_torchao_available = lambda: False
                    log("非互換torchaoを検出 -> peftのtorchao dispatcher"
                        "を無効化")
            except Exception:
                pass
            try:
                _hi_p = Path(_hf_download(
                    rep, lora_name, log))
                self.pipe.load_lora_weights(
                    str(_hi_p.parent), adapter_name="anisora_high",
                    weight_name=_hi_p.name)
                adapter_names.append("anisora_high")
                adapter_weights.append(1.0)
            except Exception as e:
                raise RuntimeError(
                    "latent直結のVACE HighへAniSora High要素抽出LoRAを"
                    "適用できません"
                    f"でした: {str(e)[:300]}")
            log("AniSora High要素LoRA適用完了: VACE Highのみ "
                "(native AniSora Lowには非適用)")
        else:
            log("latent直結: AniSora High要素LoRAは比較用に無効")
        if lightning4:
            if vlow is None:
                raise RuntimeError(
                    "hybrid_lightning4にはhybrid_vace_low_steps>=1が必要です")
            rep = os.environ.get("VIDEOLAB_VACE_LORA_REPO",
                                 "lightx2v/Wan2.2-Lightning")
            fold = os.environ.get(
                "VIDEOLAB_VACE_LORA_DIR",
                "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0")
            log(f"三段handoff: Lightning 4step LoRA読込 {rep}/{fold} "
                "(VACE High/Lowのみ)")
            try:
                _lh = Path(_hf_download(
                    rep, f"{fold}/high_noise_model.safetensors", log))
                _ll = Path(_hf_download(
                    rep, f"{fold}/low_noise_model.safetensors", log))
                self.pipe.load_lora_weights(
                    str(_lh.parent), adapter_name="lightning",
                    weight_name=_lh.name)
                self.pipe.load_lora_weights(
                    str(_ll.parent), adapter_name="lightning_2",
                    weight_name=_ll.name, load_into_transformer_2=True)
                adapter_names[:0] = ["lightning", "lightning_2"]
                adapter_weights[:0] = [1.0, 1.0]
            except Exception as e:
                raise RuntimeError(
                    "三段handoffのLightning 4step LoRAを適用できません"
                    f"でした: {str(e)[:300]}")
        if adapter_names:
            self.pipe.set_adapters(adapter_names, adapter_weights)
            log("三段handoff adapter有効: " + ", ".join(adapter_names))
        # VAE/UMT5/tokenizer/schedulerは同一オブジェクトを共有。両repoのVAE
        # safetensorsは同一hashなのでlatent scaleの変換も不要。
        self.ani_pipe = WanImageToVideoPipeline.from_pretrained(
            snap_base, transformer=None, transformer_2=alow,
            vae=self.pipe.vae, text_encoder=self.pipe.text_encoder,
            tokenizer=self.pipe.tokenizer, scheduler=sched,
            torch_dtype=_pick_dtype())
        # load_lora_weights/from_pretrainedが内部で置いたdeviceに依存しない。
        # condition encode前はDiT 2体を必ずCPUへ戻し、VAE/UMT5用の空きを作る。
        for module in (vhigh, vlow, alow, self.pipe.vae,
                       self.pipe.text_encoder):
            try:
                module.to("cpu")
            except Exception:
                pass
        _free_cuda(log)
        _log_cuda_state(log, "ロード後CPU待機")
        self.prompt_suffix = AniSoraAdapter.prompt_suffix
        self.loaded_quant = quant
        self.loaded_lora = lora
        self.loaded_lightning4 = lightning4
        self.loaded_vace_low = want_vace_low
        self.loaded = True
        log("latent直結モデル準備完了: VACE High + AniSora High要素LoRA + "
            "native AniSora Low "
            "(VAE/UMT5共有、通常時はCPU待機)")

    @staticmethod
    def _move(module, device, log, label):
        if module is None:
            return
        module.to(device)
        log(f"{label} -> {device}")

    # _try_group_offload は _WanA14BBase へ移設 (v0.8.2: 素のvace/anisora
    # でも使うため。挙動は同一)

    def generate(self, req: GenRequest, workdir: Path, log, progress) -> Path:
        import torch

        quant, boundary = self._hybrid_want(req.extra)
        lora = self._hybrid_lora(req.extra)
        lightning4 = self._hybrid_lightning4(req.extra)
        want_vace_low = self._hybrid_vace_low_steps(req.extra, 4) > 0
        if self.loaded and (quant, lora, lightning4, want_vace_low) != (
                getattr(self, "loaded_quant", None),
                getattr(self, "loaded_lora", None),
                getattr(self, "loaded_lightning4", False),
                getattr(self, "loaded_vace_low", False)):
            log(f"設定変更 {getattr(self, 'loaded_quant', '?')}/"
                f"{getattr(self, 'loaded_lora', '?')} -> {quant}/{lora}: "
                "latent直結モデルを積み替えます")
            self.unload(log)
            _free_cuda(log)
        if not self.loaded:
            self._next_extra = dict(req.extra or {})
            self.ensure_loaded(log)
        if not torch.cuda.is_available():
            raise RuntimeError("VACE→AniSora latent直結にはCUDA GPUが必要です")

        # diffusers Pipeline.__call__は@torch.no_gradだが、ここはVAE/modelを
        # 直接呼ぶ独自ループ。推論モードなしではVAE encodeの全チャンクに
        # autograd graphが残り、464x832でもA100-80GBを78GBまで食い尽くす
        # (v0.7.0実機障害)。モデルロード/LoRA装着は外で済ませ、推論だけを
        # inference_modeへ入れる。
        with torch.inference_mode():
            return self._generate_inference(
                req, workdir, log, progress, boundary)

    def _generate_inference(self, req: GenRequest, workdir: Path, log,
                            progress, boundary: float) -> Path:
        import torch

        pipe, apipe = self.pipe, self.ani_pipe
        vmodel, vlow, amodel = (pipe.transformer, pipe.transformer_2,
                                apipe.transformer_2)
        device = torch.device("cuda")
        w = _snap(req.width, 16, 240)
        h = _snap(req.height, 16, 240)
        n = max(9, ((int(req.num_frames) - 1) // 8) * 8 + 1)
        steps = max(4, int(req.steps))
        guidance = float(req.guidance)
        # 生成直前VRAMゲート (P0-3)。handoffは自前でDiTを1体ずつswapする
        # ため常駐余地=DiT1体。降格レバーは無い (extra.offload=blockが既存)
        _admit_vram(w, h, n, log,
                    resident_extra_gb=getattr(self, "_dit_gb", 0.0) or 9.3,
                    tag="handoff")

        pf = list(req.extra.get("pose_frames_b64") or [])
        if pf:
            control = load_images_b64(pf)
        elif len(req.images) > 1:
            control = list(req.images[1:])
        else:
            raise RuntimeError("latent直結には extra.pose_frames_b64 が必要です")
        got = len(control)
        if got != n:
            idx = [round(i * (got - 1) / max(1, n - 1)) for i in range(n)]
            control = [control[i] for i in idx]
            log(f"制御フレーム数を調整: {got} -> {n}")
        control = [im if im.size == (w, h) else _fit_image(im, w, h)
                   for im in control]
        ref = _fit_image(req.images[0], w, h)
        prompt = self._build_prompt(req, log)
        negative = req.negative or self.WAN_NEGATIVE
        generator = torch.Generator("cpu").manual_seed(req.seed)

        try:
            # 前ジョブやLoRA loaderのdevice状態にかかわらず、条件生成中に
            # DiTがVRAMへ残らないことを毎回保証する。
            vmodel.to("cpu")
            if vlow is not None:
                vlow.to("cpu")
            amodel.to("cpu")
            _free_cuda(log)
            _log_cuda_state(log, "condition前/DiT退避済み")
            # ---- promptを一度だけencode。UMT5は以後CPUへ ----
            if pipe.text_encoder is None:
                # 低RAM VMではジョブ末尾でUMT5をRAMから解放している
                from transformers import UMT5EncoderModel
                log("UMT5を再ロード (低RAM運転・約30秒)")
                pipe.text_encoder = UMT5EncoderModel.from_pretrained(
                    getattr(self, "_te_from", None) or self.repo,
                    subfolder="text_encoder", torch_dtype=_pick_dtype())
                self.ani_pipe.text_encoder = pipe.text_encoder
            self._move(pipe.text_encoder, device, log, "UMT5")
            prompt_embeds, negative_embeds = pipe.encode_prompt(
                prompt=prompt, negative_prompt=negative,
                do_classifier_free_guidance=guidance > 1.0,
                num_videos_per_prompt=1, max_sequence_length=512,
                device=device)
            pipe.text_encoder.to("cpu")
            if _low_ram_vm():
                # L4級VM (RAM53GB) はモデルのCPU退避場所が足りず、RAMの
                # 一時スパイクがランタイムごとOOM killされる (2026-07-13
                # 実障害疑い: システムRAM 44.2/53GBで運転中にセッション消滅)。
                # UMT5 (約11GB) をジョブ毎に解放して常時ヘッドルームを確保
                import gc
                log("低RAM VM: UMT5をRAMから解放 (44→33GB級。次ジョブ冒頭で"
                    "再ロード)")
                pipe.text_encoder = None
                self.ani_pipe.text_encoder = None
                gc.collect()
            _free_cuda(log)
            _log_cuda_state(log, "UMT5 encode後")

            # ---- VACE制御条件とnative I2V条件を同じVAEで一度だけencode ----
            self._move(pipe.vae, device, log, "共有VAE(condition encode)")
            try:
                pipe.vae.enable_tiling()
            except Exception:
                pass
            video, mask, refs = pipe.preprocess_conditions(
                control, None, [ref], 1, h, w, n,
                torch.float32, device)
            num_refs = len(refs[0])
            control_cond = pipe.prepare_video_latents(
                video, mask, refs, generator, device)
            mask_lat = pipe.prepare_masks(mask, refs, generator)
            control_cond = torch.cat([control_cond, mask_lat], dim=1)

            # native AniSoraの20ch条件(mask4+reference latent16)。dummyは
            # condition生成のためだけで、noise stateには使用しない。
            ref_tensor = apipe.video_processor.preprocess(
                ref, height=h, width=w).to(device, dtype=torch.float32)
            t_video = (n - 1) // apipe.vae_scale_factor_temporal + 1
            dummy = torch.zeros(
                (1, pipe.vae.config.z_dim, t_video,
                 h // apipe.vae_scale_factor_spatial,
                 w // apipe.vae_scale_factor_spatial),
                device=device, dtype=torch.float32)
            _dummy, i2v_cond = apipe.prepare_latents(
                ref_tensor, 1, pipe.vae.config.z_dim, h, w, n,
                torch.float32, device, generator, dummy, None)
            del _dummy, dummy, ref_tensor, video, mask, mask_lat

            # VACEは参照画像を時間軸先頭へ1slot追加するため、初期noiseも
            # T+Rで作る。handoff時にstateとUniPC履歴から同時に外す。
            latents = pipe.prepare_latents(
                1, vmodel.config.in_channels, h, w,
                n + num_refs * pipe.vae_scale_factor_temporal,
                torch.float32, device, generator, None)
            pipe.vae.to("cpu")
            _free_cuda(log)
            _log_cuda_state(log, "condition encode後")

            sched = pipe.scheduler
            sched.set_timesteps(steps, device=device)
            timesteps = sched.timesteps
            ani_low_count = self._hybrid_low_steps(
                req.extra, len(timesteps))
            vace_low_count = self._hybrid_vace_low_steps(
                req.extra, len(timesteps))
            high_count = (len(timesteps) - ani_low_count
                          - vace_low_count)
            if vace_low_count and vlow is None:
                raise RuntimeError(
                    "hybrid_vace_low_stepsを指定しましたがVACE Lowが"
                    "ロードされていません")
            if vace_low_count:
                log(f"三段latent直結生成: {w}x{h} {n}f / {steps}step — "
                    f"VACE High + Lightning + AniSora要素LoRA "
                    f"{high_count}step → VACE Low + Lightning "
                    f"{vace_low_count}step → native AniSora Low "
                    f"{ani_low_count}step")
            else:
                log(f"latent直結生成: {w}x{h} {n}f / {steps}step — "
                    f"VACE High + AniSora要素LoRA {high_count}step → "
                    f"native AniSora Low {ani_low_count}step (終端限定)")

            scale = float(req.extra.get("conditioning_scale", 1.0))
            scale = torch.tensor(
                [scale] * len(vmodel.config.vace_layers),
                device=device, dtype=vmodel.dtype)
            prompt_embeds = prompt_embeds.to(device=device,
                                              dtype=vmodel.dtype)
            if negative_embeds is not None:
                negative_embeds = negative_embeds.to(
                    device=device, dtype=vmodel.dtype)
            control_cond = control_cond.to(device=device, dtype=vmodel.dtype)
            i2v_cond = i2v_cond.to(device=device, dtype=amodel.dtype)

            want_off = str((req.extra or {}).get("offload")
                           or "").lower() in ("seq", "model")
            if want_off and vace_low_count:
                log("三段handoffではexpertを明示swapするためoffload指定を"
                    "無効化")
                want_off = False
            v_hooked = (want_off
                        and self._try_group_offload(vmodel, log, "VACE High"))
            a_hooked = False
            if not v_hooked:
                self._move(vmodel, device, log, "VACE High")
            # 49f×720x1296級はA100-40で残余0〜2GBの見積り (2026-07-13机上
            # 外挿)。実測peakをログへ残し、次回のフレーム数/セル寸法の
            # 可否判定を外挿でなく実測で行えるようにする
            torch.cuda.reset_peak_memory_stats()
            switched = False
            switched_vace_low = False
            active_vace_model = vmodel
            for si, t in enumerate(timesteps):
                use_vace_high = si < high_count
                use_vace_low = (high_count <= si
                                < high_count + vace_low_count)
                timestep = t.expand(latents.shape[0])
                if use_vace_high or use_vace_low:
                    stage_model = vmodel
                    if use_vace_low:
                        stage_model = vlow
                        if not switched_vace_low:
                            self._move(vlow, device, log,
                                       "VACE Low + Lightning")
                            if not v_hooked:
                                vmodel.to("cpu")
                                _free_cuda(log)
                            active_vace_model = vlow
                            switched_vace_low = True
                            log("handoff 1/2完了: VACE High → VACE Low "
                                "(latent/UniPC履歴を維持)")
                    latent_input = latents.to(stage_model.dtype)
                    with stage_model.cache_context("cond"):
                        pred = stage_model(
                            hidden_states=latent_input, timestep=timestep,
                            encoder_hidden_states=prompt_embeds,
                            control_hidden_states=control_cond,
                            control_hidden_states_scale=scale,
                            return_dict=False)[0]
                    if guidance > 1.0:
                        with stage_model.cache_context("uncond"):
                            uncond = stage_model(
                                hidden_states=latent_input,
                                timestep=timestep,
                                encoder_hidden_states=negative_embeds,
                                control_hidden_states=control_cond,
                                control_hidden_states_scale=scale,
                                return_dict=False)[0]
                        pred = uncond + guidance * (pred - uncond)
                else:
                    if not switched:
                        old_t = int(latents.shape[2])
                        latents = latents[:, :, num_refs:]
                        _slice_handoff_scheduler_state(
                            sched, num_refs, old_t)
                        if int(i2v_cond.shape[2]) != int(latents.shape[2]):
                            raise RuntimeError(
                                "handoff temporal shape不一致: "
                                f"latent={tuple(latents.shape)} / "
                                f"condition={tuple(i2v_cond.shape)}")
                        del control_cond, scale
                        _free_cuda(log)
                        a_hooked = (want_off and self._try_group_offload(
                            amodel, log, "AniSora Low"))
                        # 低RAM VM対策の載せ替え順: 空きVRAMが足りるなら
                        # 先にLowをGPUへ (CPU側が9GB空く) → HighをCPUへ。
                        # 逆順だとCPU一時+8.5GBのスパイクがL4級VM (53GB)
                        # のOOM killを招く。VRAMが狭いGPUは従来順のまま
                        swap_first = False
                        if not a_hooked:
                            try:
                                swap_first = (torch.cuda.mem_get_info()[0]
                                              > 10.5 * 2**30)
                            except Exception:
                                swap_first = False
                            if swap_first:
                                self._move(amodel, device, log,
                                           "native AniSora Low (RAMスパイク"
                                           "回避の先載せ)")
                        if not v_hooked:
                            active_vace_model.to("cpu")
                            _free_cuda(log)
                        if not a_hooked and not swap_first:
                            self._move(amodel, device, log,
                                       "native AniSora Low")
                        switched = True
                        log("handoff最終完了: pixel化せずlatent16chとUniPC"
                            "履歴をnative AniSora Low 36ch入力へ接続")
                    latent_input = torch.cat(
                        [latents, i2v_cond], dim=1).to(amodel.dtype)
                    with amodel.cache_context("cond"):
                        pred = amodel(
                            hidden_states=latent_input, timestep=timestep,
                            encoder_hidden_states=prompt_embeds,
                            encoder_hidden_states_image=None,
                            return_dict=False)[0]
                    if guidance > 1.0:
                        with amodel.cache_context("uncond"):
                            uncond = amodel(
                                hidden_states=latent_input,
                                timestep=timestep,
                                encoder_hidden_states=negative_embeds,
                                encoder_hidden_states_image=None,
                                return_dict=False)[0]
                        pred = uncond + guidance * (pred - uncond)
                latents = sched.step(pred, t, latents, return_dict=False)[0]
                progress(0.10 + 0.76 * (si + 1) / len(timesteps))

            if not switched:
                raise RuntimeError("native AniSora Lowへhandoffされませんでした")
            log(f"denoise区間peak VRAM "
                f"{torch.cuda.max_memory_allocated() / 2**30:.1f}GB "
                f"({w}x{h} {n}f — フレーム数/セル寸法増の可否判定用)")
            if not a_hooked:
                amodel.to("cpu")
            del i2v_cond
            _free_cuda(log)
            if v_hooked or a_hooked:
                # group offloadフックはモデルへ残留し次ジョブの常駐運転を
                # 汚すため、このジョブ限りでモデルを解放する (弱GPU運用の
                # 割り切り: 次ジョブは再ロードから)
                self.unload(log)
                log("offload構成のためモデルを解放 (次ジョブで再ロード)")

            # ---- decodeはここで一度だけ ----
            self._move(pipe.vae, device, log, "共有VAE(final decode)")
            latents = latents.to(pipe.vae.dtype)
            lm = torch.tensor(pipe.vae.config.latents_mean).view(
                1, -1, 1, 1, 1).to(latents.device, latents.dtype)
            ls = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
                1, -1, 1, 1, 1).to(latents.device, latents.dtype)
            decoded = pipe.vae.decode(
                latents / ls + lm, return_dict=False)[0]
            video = apipe.video_processor.postprocess_video(
                decoded, output_type="np")
            progress(0.94)
            return _frames_to_mp4(list(video[0]), req.fps, workdir, log)
        finally:
            for module in (getattr(pipe, "text_encoder", None),
                           getattr(pipe, "vae", None), vmodel, vlow,
                           amodel):
                try:
                    module.to("cpu")
                except Exception:
                    pass
            _free_cuda(log)


@register
class Wan22A14BAdapter(_WanA14BBase):
    """Wan2.2 I2V-A14B フル + Lightning 4step LoRA + スプライト歩行LoRA。

    extra:
      {"lightning": true}            … lightx2v 4step LoRA (既定ON, steps=4/cfg=1)
      {"walk_lora": "pixel_walk"}    … pix3lwalk 歩行LoRA (トリガー語を自動付与)
      {"walk_lora": "styly"}         … styly-agents スプライトアニメLoRA
      {"walk_weight": 0.8}           … 歩行LoRAの重み (0.7-0.9推奨)
    """
    id = "wan22a14b"
    label = "Wan 2.2 I2V-A14B + 歩行LoRA (品質本命・DL126GB)"
    desc = ("最高評判のオープンi2v。Lightning 4stepで高速化し、スプライト歩行"
            "専用LoRA(pix3lwalk)を併用可能。i2vのみ。"
            "extra例: {\"walk_lora\": \"pixel_walk\", \"walk_weight\": 0.8}")
    requires = "Colab A100 (DL約126GB — ディスク要注意)"
    disk_gb = 126
    cache_repos = ("Wan-AI/Wan2.2-I2V-A14B-Diffusers", "lightx2v/Wan2.2-Lightning",
                   "theamusing/wan2.2_pixel_walk_lora", "styly-agents/Wan2-2-pixel-animate")
    WALK_LORAS = {
        "pixel_walk": dict(repo="theamusing/wan2.2_pixel_walk_lora",
                           high="pixel_walk_lora_v1_high_noise.safetensors",
                           low="pixel_walk_lora_v1_low_noise.safetensors",
                           trigger="pix3lwalk, side view character walk animation, "),
    }
    defaults = {"width": 464, "height": 848, "num_frames": 81, "fps": 16,
                "steps": 4, "guidance": 1.0}

    def __init__(self):
        super().__init__()
        self._lora_state = None      # 直近に適用した (lightning, walk, weight)

    def _ensure_loaded_impl(self, log):
        _require_deps(log)
        import torch
        from diffusers import WanImageToVideoPipeline
        log(f"読み込み開始: {self.base_repo} (DL約126GB・初回は20分前後)")
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            self.base_repo, torch_dtype=_pick_dtype())
        self._finalize_pipe(log)
        self.loaded = True

    def _apply_loras(self, req: GenRequest, log):
        lightning = bool(req.extra.get("lightning", True))
        walk = req.extra.get("walk_lora") or None
        weight = float(req.extra.get("walk_weight", 0.8))
        state = (lightning, walk, weight)
        if state == self._lora_state:
            return
        names, weights = [], []
        if lightning:
            if "lightning" not in getattr(self.pipe, "_vl_loaded", set()):
                log("Lightning 4step LoRA 読み込み (lightx2v Seko-V1)")
                self.pipe.load_lora_weights(
                    "lightx2v/Wan2.2-Lightning", adapter_name="lightning",
                    weight_name="Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors")
                self.pipe.load_lora_weights(
                    "lightx2v/Wan2.2-Lightning", adapter_name="lightning_2",
                    weight_name="Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors",
                    load_into_transformer_2=True)
                self.pipe._vl_loaded = getattr(self.pipe, "_vl_loaded", set()) | {"lightning"}
            names += ["lightning", "lightning_2"]
            weights += [1.0, 1.0]
        if walk in self.WALK_LORAS:
            cfgw = self.WALK_LORAS[walk]
            if walk not in getattr(self.pipe, "_vl_loaded", set()):
                log(f"歩行LoRA 読み込み: {cfgw['repo']}")
                self.pipe.load_lora_weights(cfgw["repo"], adapter_name="walk",
                                            weight_name=cfgw["high"])
                self.pipe.load_lora_weights(cfgw["repo"], adapter_name="walk_2",
                                            weight_name=cfgw["low"],
                                            load_into_transformer_2=True)
                self.pipe._vl_loaded = getattr(self.pipe, "_vl_loaded", set()) | {walk}
            names += ["walk", "walk_2"]
            weights += [weight, weight]
        if names:
            self.pipe.set_adapters(names, adapter_weights=weights)
            log(f"LoRA適用: {list(zip(names, weights))}")
        self._lora_state = state

    def generate(self, req: GenRequest, workdir: Path, log, progress) -> Path:
        self._apply_loras(req, log)
        walk = req.extra.get("walk_lora") or None
        if walk in self.WALK_LORAS:
            trig = self.WALK_LORAS[walk]["trigger"]
            if trig.split(",")[0] not in req.prompt:
                req.prompt = trig + req.prompt
                log(f"LoRAトリガー語を自動付与: {trig}")
        return super().generate(req, workdir, log, progress)


@register
class AniSoraV3MultimodalPilotAdapter(_WanA14BBase):
    """公式READMEのPose guidanceを、公開V3重みそのもので検証する。

    V3とV3.2の公式ソースは、指定時刻だけ画像を置いたゼロ動画と
    同時刻マスクを同じ方式で作る。Wan2.1ベースのV3公開重みで
    公式Poseデモの条件構成を比較する隔離パイロット。
    工房本線や通常モデル一覧から自動選択されない隔離パイロット。
    """
    id = "anisora_v3_multimodal_pilot"
    label = "AniSora V3 Multimodal Guidance (公式Pose検証)"
    desc = ("公式READMEのPose/Depth/Line guidanceを公開V3重みで検証する"
            "隔離経路。extra.anisora_guidance_frames_b64必須。工房本線では"
            "未使用。")
    requires = "GPU 48GB+推奨・公式V3一式の初回DL約45GB"
    repo = "IndexTeam/Index-anisora"
    model_file = "V3/diffusion_pytorch_model.safetensors"
    base_repo = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
    cache_repos = (repo, base_repo)
    disk_gb = 55
    prompt_suffix = ("aesthetic score: 6.0. motion score: {motion:.1f}. "
                     "There is no text in the video.")
    defaults = {"width": 480, "height": 864, "num_frames": 81, "fps": 16,
                "steps": 8, "guidance": 1.0}

    def _ensure_loaded_impl(self, log):
        _require_deps(log)
        from diffusers import WanImageToVideoPipeline, WanTransformer3DModel

        extra = getattr(self, "_next_extra", {}) or {}
        log("AniSora V3公式重みを取得 (README Multimodal Guidance検証)")
        weight = _hf_download(self.repo, self.model_file, log)
        _hf_download(self.repo, "V3/config.json", log)  # 公式配置の健全性確認
        base = _snapshot_local(self.base_repo, log)
        model_gb = _gguf_gb(weight)
        if model_gb >= 1.0:
            _ram_gate(log, model_gb + 12 + 8,
                      "AniSora V3 multimodal pilot読み込み")
        log("AniSora V3 DiT読み込み (Wan2.1・単一expert)")
        transformer = WanTransformer3DModel.from_single_file(
            weight, config=base, subfolder="transformer",
            torch_dtype=_pick_dtype())
        log(f"Wan2.1 I2V共通部品を読み込み: {base}")
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            base, transformer=transformer, torch_dtype=_pick_dtype())
        self._te_from = base
        self._dit_gb = model_gb or 28.0
        offload = str(extra.get("offload") or "").lower()
        self._finalize_pipe(
            log, offload=offload, footprint_gb=self._dit_gb + 17,
            vae_tiling=True)
        self.loaded = True
        log("AniSora V3 multimodal pilot準備完了")

    def generate(self, req: GenRequest, workdir: Path, log, progress) -> Path:
        if not req.extra.get("anisora_guidance_frames_b64"):
            raise RuntimeError(
                "V3 multimodal pilotには "
                "extra.anisora_guidance_frames_b64 が必要です")
        if not self.loaded:
            self._next_extra = dict(req.extra or {})
            self.ensure_loaded(log)
        return _WanA14BBase.generate(self, req, workdir, log, progress)


def _wan_animate_face_crop(image, pose_module):
    """静止表情用に参照立ち絵の頭部を正方形クロップする。

    Wan-Animate公式前処理のface動画は駆動動画の顔クロップ列である。工房の
    歩行では表情を動かさないので、参照の頭部を1枚だけ切り出して全時刻へ
    反復する。前景検出はpose_videoと共有し、失敗時だけ四隅色差へ退避する。
    """
    import numpy as np
    from PIL import Image, ImageOps

    im = image.convert("RGB")
    arr = np.asarray(im)
    try:
        fg = pose_module._fg_mask(im)
    except Exception:
        border = np.concatenate((arr[0], arr[-1], arr[:, 0], arr[:, -1]))
        bgv = np.median(border.astype(np.float32), axis=0)
        fg = np.linalg.norm(arr.astype(np.float32) - bgv, axis=2) > 32.0
    ys, xs = np.where(fg)
    if len(xs) < 40:
        return ImageOps.fit(im, (512, 512), method=Image.Resampling.LANCZOS)

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    char_h = y1 - y0 + 1
    char_w = x1 - x0 + 1
    # 巨頭チビを含め、頭髪・耳・顎まで入る上部約48%。肩より下は除く。
    side = max(32, round(max(char_w * 1.08, char_h * 0.48)))
    side = min(side, im.width)
    cx = (x0 + x1) / 2.0
    left = max(0, min(im.width - side, round(cx - side / 2)))
    top = max(0, min(im.height - side, round(y0 - char_h * 0.025)))
    crop = im.crop((left, top, left + side, top + side))
    bg = tuple(int(x) for x in np.median(
        np.concatenate((arr[0], arr[-1], arr[:, 0], arr[:, -1])),
        axis=0))
    return ImageOps.pad(crop, (512, 512), method=Image.Resampling.LANCZOS,
                        color=bg, centering=(0.5, 0.5))


@register
class Wan22AnimatePilotAdapter(VideoAdapter):
    """Wan2.2-Animateの歩容追従を判定する隔離実験アダプタ。

    現行AniSora経路や工房の4択からは呼ばない。直接 /api/generate へ
    model=wan22animate_pilot を指定した検証だけが到達する。
    """
    id = "wan22animate_pilot"
    label = "実験: Wan2.2-Animate-14B (姿勢動画で歩行)"
    desc = ("工房の連続OpenPose歩容と静止顔動画をWan-Animateへ直接入力する"
            "隔離試験。現行AniSora生成には影響しない。")
    requires = "96GB GPU推奨 / DL約77GB / 公式標準20step"
    modes = ("i2v",)
    repo = "Wan-AI/Wan2.2-Animate-14B-Diffusers"
    gguf_repo = "QuantStack/Wan2.2-Animate-14B-GGUF"
    anisora_gguf_repo = "QuantStack/Index-Anisora-V3.2-GGUF"
    anisora_base_repo = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    cache_repos = (repo, gguf_repo, anisora_gguf_repo, anisora_base_repo)
    disk_gb = 24
    defaults = {"width": 480, "height": 864, "num_frames": 77,
                "fps": 16, "steps": 20, "guidance": 1.0}

    def __init__(self):
        super().__init__()
        self.pipe = None
        self._anisora_high_lora_loaded = False
        self._animate_high = None
        self._animate_low = None

    def ensure_loaded(self, log):
        _require_deps(log)
        import gc
        from diffusers import WanAnimatePipeline
        extra = getattr(self, "_next_extra", {}) or {}
        dual_transplant = bool(extra.get("anisora_dual_transplant"))
        transplant = bool(extra.get("anisora_low_transplant")
                          or extra.get("anisora_fp8_transplant")
                          or dual_transplant)
        quantized_base = bool(extra.get("wan_animate_quantized"))
        if quantized_base and not transplant:
            from diffusers import (GGUFQuantizationConfig,
                                   WanAnimateTransformer3DModel)
            quant = _norm_quant(extra.get("quant") or "Q4_0")
            if quant not in ("Q4_0", "Q8_0"):
                log(f"Wan-Animate量子化実験は{quant}未検証のためQ4_0へ変更")
                quant = "Q4_0"
            qc = GGUFQuantizationConfig(compute_dtype=_pick_dtype())
            target_name = f"Wan2.2-Animate-14B-{quant}.gguf"
            target_path = _hf_download(self.gguf_repo, target_name, log)
            snap = _snapshot_local(self.repo, log)
            log(f"量子化Wan-Animate本体を読み込み: {quant} "
                "(AniSora Low移植なし)")
            target = WanAnimateTransformer3DModel.from_single_file(
                target_path, quantization_config=qc, config=snap,
                subfolder="transformer", torch_dtype=_pick_dtype())
            self.pipe = WanAnimatePipeline.from_pretrained(
                snap, transformer=target, torch_dtype=_pick_dtype(),
                low_cpu_mem_usage=True)
            self.loaded_variant = "quantized_base"
        elif transplant:
            from diffusers import (GGUFQuantizationConfig,
                                   WanAnimateTransformer3DModel,
                                   WanTransformer3DModel)
            quant = _norm_quant(extra.get("quant") or "Q4_0")
            if quant not in ("Q4_0", "Q8_0"):
                log(f"Wan-Animate移植実験は{quant}未検証のためQ4_0へ変更")
                quant = "Q4_0"
            qc = GGUFQuantizationConfig(compute_dtype=_pick_dtype())
            target_name = f"Wan2.2-Animate-14B-{quant}.gguf"
            donor_name = f"Low/Index-Anisora-V3.2-Low-{quant}.gguf"
            target_path = _hf_download(self.gguf_repo, target_name, log)
            donor_path = _hf_download(
                self.anisora_gguf_repo, donor_name, log)
            high_path = None
            if dual_transplant:
                high_quant = _anisora_high_quant(quant)
                high_name = (f"High/Index-Anisora-V3.2-High-"
                             f"{high_quant}.gguf")
                high_path = _hf_download(
                    self.anisora_gguf_repo, high_name, log)
            snap = _snapshot_local(self.repo, log)
            donor_snap = _snapshot_local(self.anisora_base_repo, log)
            model_gb = _gguf_gb(target_path, donor_path, high_path)
            if model_gb >= 1.0:
                _ram_gate(log, model_gb + (24 if dual_transplant else 14),
                          "Wan-Animate + AniSora High/Low移植")
            log(f"Wan-Animate Low側Body/Face Adapter本体を読み込み: {quant}")
            target_low = WanAnimateTransformer3DModel.from_single_file(
                target_path, quantization_config=qc, config=snap,
                subfolder="transformer", torch_dtype=_pick_dtype())
            log(f"AniSora V3.2 Low移植ドナーを読み込み: {quant}")
            donor = WanTransformer3DModel.from_single_file(
                donor_path, quantization_config=qc, config=donor_snap,
                subfolder="transformer_2", torch_dtype=_pick_dtype())
            _transplant_base_weights(
                target_low, donor, log, tag="WanAnimate+AniSoraLow",
                patch_mode="same")
            del donor
            gc.collect()
            target_high = None
            if dual_transplant:
                log(f"Wan-Animate High側Body/Face Adapter本体を読み込み: {quant}")
                target_high = WanAnimateTransformer3DModel.from_single_file(
                    target_path, quantization_config=qc, config=snap,
                    subfolder="transformer", torch_dtype=_pick_dtype())
                log("AniSora V3.2 High移植ドナーを読み込み")
                donor_high = WanTransformer3DModel.from_single_file(
                    high_path, quantization_config=qc, config=donor_snap,
                    subfolder="transformer", torch_dtype=_pick_dtype())
                _transplant_base_weights(
                    target_high, donor_high, log,
                    tag="WanAnimate+AniSoraHigh", patch_mode="same")
                del donor_high
                gc.collect()
                log("共通40層=完整AniSora High/Low二体 / "
                    "各骨・顔制御層=Wan-Animateで構築")
            else:
                log("共通40層=完整AniSora Low / 骨・顔制御層=Wan-Animateで構築")
            self.pipe = WanAnimatePipeline.from_pretrained(
                snap, transformer=target_low, torch_dtype=_pick_dtype(),
                low_cpu_mem_usage=True)
            self._animate_high = target_high
            self._animate_low = target_low
            self.loaded_variant = ("anisora_dual_transplant" if dual_transplant
                                   else "anisora_low_transplant")
        else:
            log(f"読み込み開始: {self.repo} (bf16・初回DL約77GB)")
            self.pipe = WanAnimatePipeline.from_pretrained(
                self.repo, torch_dtype=_pick_dtype(), low_cpu_mem_usage=True)
            self.loaded_variant = "official_bf16"
        if dual_transplant:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.pipe.to(device)
            self._animate_high.to(device)
            try:
                self.pipe.vae.enable_tiling()
            except Exception:
                pass
            log("AniSora High/Low＋制御枝をGPU常駐 (step境界で本体切替)")
        else:
            _apply_offload(self.pipe, log)
        self.loaded = True

    def unload(self, log):
        self.pipe = None
        self._animate_high = None
        self._animate_low = None
        self._anisora_high_lora_loaded = False
        self.loaded_variant = None
        self.loaded = False

    @staticmethod
    def _save_control_video(frames, fps: int, workdir: Path, log):
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            return
        pdir = workdir / "control_pose_frames"
        pdir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            frame.save(pdir / f"{i:05d}.png")
        encode_mp4(ffmpeg, pdir, fps, workdir / "control_pose.mp4")
        log("検証用の姿勢制御動画を保存: control_pose.mp4")

    def generate(self, req: GenRequest, workdir: Path, log, progress) -> Path:
        import torch

        if not req.images:
            raise RuntimeError("Wan-Animateには参照キャラ画像が必要です")
        requested_w = _snap(req.width, 16, 256)
        requested_h = _snap(req.height, 16, 256)
        w, h = _snap_min_short_edge(requested_w, requested_h, 480, 16)
        if (w, h) != (requested_w, requested_h):
            log("Wan-Animate個別生成の短辺480px下限を適用: "
                f"{requested_w}x{requested_h} -> {w}x{h}")
        steps = max(4, min(30, int(req.steps or 20)))
        if steps < 20:
            log(f"注意: Wan-Animateは非蒸留。公式20stepより少ない{steps}stepで実行")
        direction = str(req.extra.get("direction") or "front")
        raw_sequence = req.extra.get("direction_sequence")
        if isinstance(raw_sequence, str):
            directions = [x.strip() for x in raw_sequence.split(",")
                          if x.strip()]
        elif isinstance(raw_sequence, (list, tuple)):
            directions = [str(x).strip() for x in raw_sequence if str(x).strip()]
        else:
            directions = []

        eng = _engine()
        pv = eng["pose_video"]
        check_dirs = directions or [direction]
        bad_dirs = [d for d in check_dirs if d not in pv.DIR_YAW]
        if bad_dirs:
            raise RuntimeError(f"未対応の方向です: {', '.join(bad_dirs)}")
        ref = req.images[0].convert("RGB")
        pose_kwargs = {
            "ref_image": ref,
            "arm_swing": float(req.extra.get("arm_swing", 1.0)),
            "leg_swing": float(req.extra.get("leg_swing", 1.0)),
            "bob": float(req.extra.get("bob", 1.0)),
            "leg_cross": float(req.extra.get("leg_cross", 1.0)),
            "face68": False,
            # この実験は正面参照1枚を全方向へ共用する。既存のヨー追従は
            # 斜め方向の「自方向立ち絵」を測る前提なので、正面絵を渡すと
            # 顔正面化(auto)が発火して頭だけ0度になる。Wan-Animateでは
            # 操作骨を真の8方位にするため、頭も胴も公称DIR_YAWへ固定する。
            "yaw_adapt": False,
        }
        if directions:
            segment_frames = max(9, int(req.extra.get("segment_frames", 49)))
            if segment_frames % 4 != 1:
                old = segment_frames
                segment_frames = ((segment_frames - 1) // 4) * 4 + 1
                log(f"区間フレーム制約4k+1へ丸め: {old} -> {segment_frames}")
            pose_frames = []
            for i, d in enumerate(directions):
                part = pv.build_walk_pose_frames(
                    d, segment_frames, w, h, **pose_kwargs)
                # Wan-Animateの次区間は前区間末尾1fを条件として共有する。
                pose_frames.extend(part if i == 0 else part[1:])
            n = len(pose_frames)
            direction = "sequence"
            log(f"8方向連結姿勢動画: {' → '.join(directions)} / "
                f"{segment_frames}f区間(1f重複) / 合計{n}f")
        else:
            n = max(5, int(req.num_frames))
            if n % 4 != 1:
                old = n
                n = ((n - 1) // 4) * 4 + 1
                log(f"フレーム制約4k+1へ丸め: {old} -> {n}")
            segment_frames = n
            log(f"姿勢動画生成: {direction} / {w}x{h} / {n}f")
            pose_frames = pv.build_walk_pose_frames(
                direction, n, w, h, **pose_kwargs)
        default_face = "blank" if directions else "reference"
        face_mode = str(req.extra.get("face_mode") or default_face).lower()
        if face_mode == "blank":
            from PIL import Image
            face = Image.new("RGB", (512, 512), (0, 0, 0))
            log("顔モーション条件: blank (背面化と正面顔の競合を除外)")
        else:
            face_mode = "reference"
            face = _wan_animate_face_crop(ref, pv)
            log("顔モーション条件: 静止参照クロップ")
        face.save(workdir / "control_face.png")
        face_frames = [face.copy() for _ in range(n)]
        self._save_control_video(pose_frames, req.fps, workdir, log)
        progress(0.04)

        prompt = req.prompt.strip() or (
            "Full-body chibi anime game character, exactly the same character, "
            "costume, colors and proportions as the reference image. The character "
            "walks in place facing directly forward with clear alternating left and "
            "right steps, natural opposite arm swings and visible foot contact. The "
            "torso and head remain facing forward. Fixed camera, fixed framing, flat "
            "solid magenta background, no added objects, smooth cyclic motion.")
        if abs(float(req.guidance) - 1.0) > 1e-6:
            log("Wan-Animate公式設定に合わせguidanceを1.0へ固定")
        gen_device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=gen_device).manual_seed(int(req.seed))
        segment_count = len(directions) if directions else 1
        callback_count = [0]

        def _progress_cb(pipe, step_index, timestep, callback_kwargs):
            callback_count[0] += 1
            if (use_anisora_lora and anisora_high_lora_steps > 0 and
                    not lora_switched_off[0] and
                    int(step_index) + 1 >= anisora_high_lora_steps):
                self.pipe.disable_lora()
                lora_switched_off[0] = True
                log("AniSora High LoRAを終了: "
                    f"{anisora_high_lora_steps}/{steps}step、以降は素のWan-Animate")
            total = max(1, steps * segment_count)
            progress(0.05 + 0.85 * min(1.0, callback_count[0] / total))
            return callback_kwargs

        log(f"Wan-Animate生成開始: {steps}step x {segment_count}区間 / "
            f"CFG 1.0 / seed {req.seed}")
        use_anisora_lora = bool(req.extra.get("anisora_high_lora", False))
        anisora_lora_scale = max(0.0, min(2.0, float(
            req.extra.get("anisora_high_lora_scale", 1.0))))
        anisora_high_lora_steps = max(0, min(steps, int(
            req.extra.get("anisora_high_lora_steps", 0))))
        lora_switched_off = [False]
        if use_anisora_lora:
            if not self._anisora_high_lora_loaded:
                log("AniSora V3.2 High rank64 LoRAを読み込み: Kijai/WanVideo_comfy")
                self.pipe.load_lora_weights(
                    "Kijai/WanVideo_comfy", subfolder="LoRAs/AniSora",
                    weight_name=(
                        "Wan2_2_I2V_AniSora_3_2_HIGH_rank_64_fp16.safetensors"),
                    adapter_name="anisora_high")
                self._anisora_high_lora_loaded = True
            self.pipe.set_adapters(
                ["anisora_high"], adapter_weights=[anisora_lora_scale])
            log(f"AniSora High LoRAを適用: scale={anisora_lora_scale:.2f}")
            if anisora_high_lora_steps:
                log("High限定適用: "
                    f"最初の{anisora_high_lora_steps}/{steps}step")
        elif self._anisora_high_lora_loaded:
            self.pipe.disable_lora()
            log("AniSora High LoRAを無効化 (素のWan-Animate)")
        dual_active = (self._animate_high is not None and
                       self._animate_low is not None)
        low_forward = None
        dual_calls = [0]
        high_steps = max(1, min(steps - 1, int(
            req.extra.get("anisora_high_steps", max(1, steps // 2)))))
        if dual_active:
            low_forward = self._animate_low.forward
            high_forward = self._animate_high.forward

            def _dual_forward(*args, **kwargs):
                step_in_segment = dual_calls[0] % steps
                dual_calls[0] += 1
                fn = high_forward if step_in_segment < high_steps else low_forward
                return fn(*args, **kwargs)

            self._animate_low.forward = _dual_forward
            log(f"AniSora二段切替: High {high_steps}step → "
                f"Low {steps - high_steps}step (骨・顔制御枝は両段に維持)")
        try:
            result = self.pipe(
                image=ref, pose_video=pose_frames, face_video=face_frames,
                prompt=prompt, negative_prompt=req.negative or None,
                height=h, width=w, segment_frame_length=segment_frames,
                prev_segment_conditioning_frames=1,
                num_inference_steps=steps, mode="animate", guidance_scale=1.0,
                motion_encode_batch_size=max(1, min(16, int(
                    req.extra.get("motion_encode_batch_size", 8)))),
                generator=generator, output_type="np",
                callback_on_step_end=_progress_cb,
                callback_on_step_end_tensor_inputs=["latents"])
        finally:
            if low_forward is not None:
                self._animate_low.forward = low_forward
        frames = result.frames[0]
        progress(0.94)
        meta = {"model": self.repo, "direction": direction,
                "direction_sequence": directions or None,
                "segment_frames": segment_frames, "width": w,
                "height": h, "frames": n, "fps": req.fps, "steps": steps,
                "seed": req.seed, "guidance": 1.0,
                "anisora_high_lora": use_anisora_lora,
                "anisora_high_lora_scale": (anisora_lora_scale
                                             if use_anisora_lora else None),
                "anisora_high_lora_steps": (anisora_high_lora_steps
                                             if use_anisora_lora else None),
                "face_motion": ("blank" if face_mode == "blank" else
                                "static_reference_head_crop")}
        (workdir / "pilot_settings.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        out = _frames_to_mp4(list(frames), req.fps, workdir, log)
        _free_cuda(log)
        return out


# ---------------------------------------------------------------- ジョブ管理
JOBS: dict[str, dict] = {}
JOB_ORDER: list[str] = []
WORK_Q: "queue.Queue[str]" = queue.Queue()
CURRENT_MODEL: str | None = None
_LOCK = threading.Lock()


# 孤児ジョブ検知: cancel_if_unpolled 指定ジョブは、クライアントの
# /status ポーリングがこの秒数途絶えたら自動中止する (アプリ強制終了・
# クラッシュでColab GPUが無駄に回り続けるのを防ぐ 2026-07-12要望)
ORPHAN_SEC = int(os.environ.get("VIDEOLAB_ORPHAN_SEC", "90"))


def submit_job(model: str, req: GenRequest, watch_poll: bool = False) -> str:
    jid = uuid.uuid4().hex[:12]
    with _LOCK:
        JOBS[jid] = {
            "id": jid, "status": "queued", "model": model, "mode": req.mode,
            "prompt": req.prompt[:200], "progress": 0.0, "detail": "",
            "created": time.time(), "started": None, "finished": None,
            "path": None, "log": [],
            "params": {"width": req.width, "height": req.height,
                       "num_frames": req.num_frames, "fps": req.fps,
                       "steps": req.steps, "seed": req.seed,
                       "guidance": req.guidance, "images": len(req.images)},
            "_req": req, "_cancel": False,
            "_watch_poll": watch_poll, "_last_poll": time.time(),
        }
        JOB_ORDER.append(jid)
    WORK_Q.put(jid)
    return jid


def _job_public(j: dict) -> dict:
    return {k: v for k, v in j.items() if not k.startswith("_")}


class JobCancelled(Exception):
    pass


def worker_loop():
    global CURRENT_MODEL, _WORKER_STALLED
    while True:
        jid = WORK_Q.get()
        _WORKER_STALLED = False        # popできた=workerは生きている (P1)
        j = JOBS.get(jid)
        if not j or j.get("_cancel"):
            if j:
                j["status"] = "cancelled"
                j["finished"] = j.get("finished") or time.time()
                j.pop("_req", None)    # 画像/骨格b64のRAM残留を防ぐ
            continue
        # 孤児検知: キュー待ちの間にクライアントが消えたジョブは始めない
        if (j.get("_watch_poll")
                and time.time() - j.get("_last_poll", 0) > ORPHAN_SEC):
            j["status"] = "cancelled"
            j["detail"] = "クライアント切断により自動中止 (ポーリング途絶)"
            j["finished"] = j.get("finished") or time.time()
            j.pop("_req", None)
            print(f"[{jid}] 孤児ジョブを自動中止", flush=True)
            continue

        def log(msg, _j=j):
            line = f"[{time.strftime('%H:%M:%S')}] {msg}"
            _j["log"].append(line)
            _j["_beat"] = time.time()      # watchdog用の鼓動 (P1)
            print(f"[{_j['id']}] {msg}", flush=True)
            # ロード鼓動 (v0.8.3): モデルDL/読み込み中はログ1行ごとに
            # 進捗を+1%刻む (上限10%)。「ゲージが全く動かない」不安の解消
            # — denoise開始で_step_callbackが5%+から上書きし単調性は
            # クライアント側(_bump_progress)が保証する
            if _j.get("status") == "loading":
                _j["progress"] = min(0.10,
                                     round(float(_j.get("progress") or 0)
                                           + 0.01, 4))

        def progress(p, _j=j):
            _j["progress"] = round(float(p), 4)
            _j["_beat"] = time.time()      # watchdog用の鼓動 (P1)
            if _j.get("_cancel"):
                raise JobCancelled()
            if (_j.get("_watch_poll") and time.time() - _j.get("_last_poll", 0)
                    > max(120, ORPHAN_SEC)):
                _j["detail"] = "クライアント切断により自動中止 (ポーリング途絶)"
                raise JobCancelled()

        try:
            adapter = ADAPTERS[j["model"]]
            # ロード時にジョブのextra(量子化指定など)を参照できるように渡す
            adapter._next_extra = dict(getattr(j["_req"], "extra", {}) or {})
            # HFトークン (v0.8.5): アプリのconfig hf_tokenがextraで届く。
            # 認証つきDLはIP帯域制限の対象外になりやすい (値はログに出さない)
            _hft = str(adapter._next_extra.pop("hf_token", "") or "").strip()
            if _hft:
                os.environ["HF_TOKEN"] = _hft
                os.environ["HUGGING_FACE_HUB_TOKEN"] = _hft
                log("HFトークンを受領 (認証つきDL — IP帯域制限を回避)")
            # モデル切替: 前のモデルをアンロードしてVRAMを空ける
            if CURRENT_MODEL not in (None, j["model"]):
                prev = ADAPTERS.get(CURRENT_MODEL)
                if prev and prev.loaded:
                    j["status"] = "loading"
                    j["detail"] = f"{prev.label} をアンロード中"
                    log(f"モデル切替: {CURRENT_MODEL} -> {j['model']}")
                    prev.unload(log)
                    _free_cuda(log)
                    if os.environ.get("VIDEOLAB_PURGE_ON_SWITCH") == "1":
                        if _can_keep_cache(adapter, log):
                            log("空きが十分なため前モデルのキャッシュを温存 "
                                "(戻したときの再DLを回避)")
                        else:
                            repos = (getattr(prev, "cache_repos", None)
                                     or [getattr(prev, "repo", "")])
                            # 次モデルも使うリポは消さない (anisora⇔vaceは
                            # AniSora GGUFを共有: 消すと即再DLで純損)
                            keep = set(getattr(adapter, "cache_repos", None)
                                       or [getattr(adapter, "repo", "")])
                            for r in repos:
                                if r in keep:
                                    log(f"共有キャッシュを温存: {r}")
                                else:
                                    _purge_model_cache(r, log)
            if not adapter.loaded:
                j["status"] = "loading"
                j["detail"] = f"{adapter.label} を読み込み中(初回はDLに数分〜数十分)"
                t0 = time.time()
                # DL進捗ウォッチャ (v0.8.4): HFキャッシュ実サイズを20秒毎に
                # 見て進行をログへ (ログ行はロード鼓動=ゲージ+1%も兼ねる)。
                # 成長ゼロが続くときは停滞を明示 — 「固まった?」の判別材料
                _done = threading.Event()

                def _dl_watch(_a=adapter, _j=j):
                    # v0.8.6: リポキャッシュdirでなくディスクusedを見る
                    # (Xet経路はチャンクを別キャッシュへ書くため)
                    last = -1.0
                    still = 0
                    while not _done.wait(20):
                        cur = _disk_used_gb()
                        if cur < 0:
                            continue
                        if last < 0:
                            last = cur
                            continue
                        if cur - last > 0.005:
                            # 低速でも進んでいる限り鼓動を打つ (v0.9.13:
                            # 1〜2.5MB/s帯の健全DLがログ閾値0.05GB/20sに
                            # 届かず、watchdogが健全ロードを打ち切る回廊が
                            # あった — レビュー指摘)
                            _j["_beat"] = time.time()
                        if cur - last > 0.05:
                            log(f"DL/読込 進行中: +{cur - last:.1f}GB/20s")
                            still = 0
                        else:
                            still += 1
                            if still in (3, 9):
                                log(f"⚠ DLが{still * 20}秒進んでいません "
                                    "(停滞検知が別経路への切替を行います)")
                        last = cur
                threading.Thread(target=_dl_watch, daemon=True).start()
                try:
                    adapter.ensure_loaded(log)
                finally:
                    _done.set()
                log(f"モデル準備完了 ({time.time() - t0:.0f}s)")
            CURRENT_MODEL = j["model"]

            # モデル読み込み中はキャンセルを検知できないため、ここで再判定
            # (読み込みブロック中に中止されたジョブの生成を始めない)。
            # watchdogがerror化済みのジョブも同様 — 黙ってrunning/doneへ
            # 上書きするとクライアントの再投入と二重生成になる (v0.9.13)
            if j.get("_cancel") or j.get("status") == "error":
                raise JobCancelled()

            j["status"] = "running"
            j["detail"] = ""
            j["started"] = time.time()
            workdir = WORK_ROOT / jid
            workdir.mkdir(parents=True, exist_ok=True)
            out = adapter.generate(j["_req"], workdir, log, progress)
            if j.get("status") == "error":
                # 生成中にwatchdogが打ち切り裁定済み — 結果は保存するが
                # 裁定は覆さない (クライアントは既にerrorを見ている)
                j["path"] = str(out)
                log(f"watchdog打ち切り済みジョブが完走: 成果物={out} "
                    "(statusはerrorのまま)")
            else:
                j["path"] = str(out)
                j["status"] = "done"
                j["progress"] = 1.0
                log(f"完了: {out}")
        except JobCancelled:
            if j.get("status") != "error":     # watchdog裁定は上書きしない
                j["status"] = "cancelled"
                j["detail"] = "ユーザーによりキャンセル"
            # 中断で放置された中間テンソルが次のジョブをOOMさせないように
            _free_cuda(log)
        except Exception as e:
            # OOM専用ハンドラ (P2-6): 断片回収+降格提案つきの定型detail。
            # extra.retry_on_oom 指定時は block offload で1回だけ自動再試行
            _oom = False
            try:
                import torch
                _oom = isinstance(e, torch.cuda.OutOfMemoryError)
            except Exception:
                pass
            if _oom:
                _p = j.get("params") or {}
                _adv = (f"VRAM不足 (OOM): {_p.get('width')}x"
                        f"{_p.get('height')}/{_p.get('num_frames')}f。"
                        "対処: 解像度/フレームを下げる・extra.offload=block"
                        "・quantを下げる (Q3_K_S等)")
                _req0 = j.get("_req")
                if (_req0 is not None
                        and (_req0.extra or {}).get("retry_on_oom")
                        and not j.get("_oom_retried")
                        and not j.get("_cancel")   # キャンセル済みは再試行
                        #                            しない (競合での_req
                        #                            永久残留防止 v0.9.13)
                        and str((_req0.extra or {}).get("offload") or "")
                        != "block"):
                    j["_oom_retried"] = True
                    _req0.extra["offload"] = "block"
                    try:
                        adapter.unload(log)
                    except Exception:
                        pass
                    _free_cuda(log)
                    j["status"] = "queued"
                    j["progress"] = 0.0
                    j["finished"] = None   # watchdog等の終了刻印をクリア
                    j["detail"] = "OOM -> block offloadで自動再試行"
                    j["_requeue"] = True
                    print(f"[{jid}] OOM -> block offloadで自動再試行",
                          flush=True)
                else:
                    j["status"] = "error"
                    j["detail"] = _adv + f" ({str(e)[:200]})"
                    j["log"].append(traceback.format_exc()[-1500:])
                    print(f"[{jid}] ERROR(OOM): {e}", flush=True)
                    _free_cuda(log)
            else:
                j["status"] = "error"
                j["detail"] = str(e)[:600]
                j["log"].append(traceback.format_exc()[-1500:])
                print(f"[{jid}] ERROR: {e}", flush=True)
                # OOM等の失敗断片を回収してから次のジョブへ
                _free_cuda(log)
        finally:
            if j.pop("_requeue", False):
                WORK_Q.put(jid)            # 再試行: _reqを温存したまま再投入
            else:
                j["finished"] = time.time()
                j.pop("_req", None)


_WORKER_STALLED = False    # watchdogがrunning/loadingを打ち切った=worker
#                            スレッドがハングしている疑い (popで解除)


def _watchdog_scan(now: float | None = None) -> list:
    """生成watchdog (P1) の1スキャン: running/loading ジョブの鼓動が
    VIDEOLAB_STALL_MIN分 (既定15) 途絶えていたらstalled扱いにする。
    処理したジョブidのリストを返す (テスト用)。

    ハングしたworkerスレッドはPythonからkillできない。ジョブをerror化して
    クライアントの永久待ちを解き、以後のスキャンでは「workerが死んでいる
    疑い」フラグを立てて後続のqueuedジョブも即error化する (キューの
    消費者が消えているのに永久にqueuedのまま=飢餓の再発防止、v0.9.13
    レビュー指摘)。VIDEOLAB_WATCHDOG_EXIT=1 のときは runtime.unassign で
    VMごと解放して自沈する (Colabでは課金も止まる — 素のos._exitは
    カーネルだけ死んで課金が続く、レビュー指摘で修正)。"""
    global _WORKER_STALLED
    try:
        stall_min = float(os.environ.get("VIDEOLAB_STALL_MIN", "15") or 15)
    except ValueError:
        stall_min = 15.0
    if stall_min <= 0:
        return []
    now = now or time.time()
    hit = []
    for j in list(JOBS.values()):
        if j.get("status") not in ("running", "loading"):
            continue
        beat = (j.get("_beat") or j.get("started")
                or j.get("created") or now)
        age = now - float(beat)
        if age < stall_min * 60:
            continue
        j["status"] = "error"
        j["detail"] = (f"watchdog: {int(age // 60)}分間無応答のため中止 "
                       "(生成ハング疑い。Colabランタイムの張り直しを推奨)")
        j["finished"] = now
        hit.append(j["id"])
        _WORKER_STALLED = True
        print(f"[{j['id']}] WATCHDOG: {int(age // 60)}分無応答 -> stalled",
              flush=True)
    if _WORKER_STALLED:
        # workerスレッドがハング中の疑い: queuedのジョブは誰にも実行され
        # ない — 永久待ちにせず明示エラーで返す (workerが復活してpopしたら
        # フラグは解除され、新規ジョブは通常どおり動く)
        for j in list(JOBS.values()):
            if j.get("status") != "queued":
                continue
            j["status"] = "error"
            j["detail"] = ("watchdog: workerが停止している疑いのため実行"
                           "できません (Colabランタイムを張り直して再投入"
                           "してください)")
            j["finished"] = now
            j.pop("_req", None)
            hit.append(j["id"])
            print(f"[{j['id']}] WATCHDOG: worker停止疑いのため実行不能",
                  flush=True)
    if hit and os.environ.get("VIDEOLAB_WATCHDOG_EXIT") == "1":
        print("[videolab] VIDEOLAB_WATCHDOG_EXIT=1: ハング検知のため"
              "ランタイムを解放して自沈します (Colab=課金停止)", flush=True)
        _shutdown_runtime(delay=0.5)
    return hit


def _watchdog_loop():
    while True:
        time.sleep(60)
        try:
            _watchdog_scan()
        except Exception:
            pass


def _shutdown_runtime(delay: float = 2.0) -> None:
    """ランタイムを解放して自沈する (アプリ終了時の自動片付け 2026-07-13要望
    「ランタイム削除が毎回地味に手間」)。

    Colab上なら google.colab.runtime.unassign() でVMごと解放 (課金停止・
    「ランタイムを接続解除して削除」と同じ)。非Colab (ローカルサーバ) は
    プロセス終了のみ。HTTP応答を返してから実行するため遅延スレッドで
    呼ぶこと。"""
    time.sleep(delay)
    try:
        from google.colab import runtime      # type: ignore
        print("[videolab] runtime.unassign() でColabランタイムを解放します",
              flush=True)
        runtime.unassign()
    except Exception:
        os._exit(0)


# ------------------------------------------------- 工房モード (walk_pack)
# お友だち(非技術者)がブラウザだけで キャラパック選択 → 歩行生成 →
# プレビュー → ドット絵化 → まとめてDL まで完結するための機能群 (v0.10.0)。
#
# ・キャラパック: 母艦アプリが /api/packs/upload でzipを置く。展開先は
#   packs/<pack_id>/01_generation/split_centered/*.png の形にする —
#   pose_video の landmarks 探索が「ref画像パスの親の親/landmarks.json」
#   なので、ラウンドディレクトリ互換のこの構造に置くだけで landmarks が
#   自動で効く。
# ・walk_pack: compass_vace._run_layout の latent_refine 分岐のサーバ内部
#   移植。pipeline.py には依存しない (エンジンは engine_pack/ から import)。
# ・v1の省略事項 (実装コスト裁定): build_T_sheet_from_mp4 のフルQC
#   (向き検査つきコマ探索・頭サイズ整合・idle中心整列・ゲート群) は使わず、
#   walk_layout の決定的なフレーム配分から簡易シート ({char}T/LT.png) を
#   組む。inspect_walk_mp4 / make_walk_preview 本体も呼ばない
#   (preview.webp は make_walk_preview 相当の 2x4 グリッドを内製)。
#   T規格の厳密な品質が要る場合は out/ の方向別mp4を母艦側の正規
#   ビルダーへ通すこと。

# engine_pack: サーバ隣接のエンジン同梱ディレクトリ (Colabへはサーバと
# 一緒に配布)。無ければ walk_pack API は 503 を返す (パックの置き場と
# 一覧・サムネは PIL だけで動くので生かす)
ENGINE_PACK_DIR = Path(os.environ.get("VIDEOLAB_ENGINE_PACK", "")
                       or (HERE / "engine_pack"))
_ENGINE_MODS: dict | None = None
_ENGINE_ERR: str | None = None


def _engine() -> dict:
    """engine_pack のモジュール束をロードして返す (失敗は RuntimeError)。"""
    global _ENGINE_MODS, _ENGINE_ERR
    if _ENGINE_MODS is not None:
        return _ENGINE_MODS
    if not ENGINE_PACK_DIR.is_dir():
        # 未配備はキャッシュしない (後から配備すれば再起動なしで有効化)
        raise RuntimeError(f"engine_pack未配備 ({ENGINE_PACK_DIR})")
    if _ENGINE_ERR is not None:
        raise RuntimeError(_ENGINE_ERR)
    if str(ENGINE_PACK_DIR) not in sys.path:
        sys.path.insert(0, str(ENGINE_PACK_DIR))
    try:
        import importlib
        # inspect_walk_mp4 は顔消失ゲートの指標 (head_band/head_diversity)
        # を借りるために読む。標準ライブラリのみ・__main__ガード付きなので
        # import副作用は無い (実測0.09s)。
        mods = {name: importlib.import_module(name)
                for name in ("pose_video", "canvas_walk", "compass_vace",
                             "color_anchor", "pixelize_sheet",
                             "inspect_walk_mp4")}
    except Exception as e:              # noqa: BLE001
        _ENGINE_ERR = f"engine_packのimportに失敗: {e}"
        raise RuntimeError(_ENGINE_ERR) from e
    _ENGINE_MODS = mods
    return mods


def packs_root() -> Path:
    """キャラパック置き場 (PACKS_DIR 相当)。

    優先: VIDEOLAB_PACKS 明示 > Driveマウント時はモデルキャッシュ
    (MyDrive/SpriteMill_models) の隣 = MyDrive/SpriteMill_packs
    (セッションが死んでもパックが消えない) > ローカルは WORK_ROOT の隣。"""
    p = os.environ.get("VIDEOLAB_PACKS", "").strip()
    if p:
        return Path(p)
    dc = _drive_cache_dir()
    if dc is not None:
        return dc.parent / "SpriteMill_packs"
    return WORK_ROOT.parent / "videolab_packs"


# ---- 依頼リレー (v0.10.1): お友だちの生成依頼を母艦がポーリングで拾う ----
_REQ_RID_RE = re.compile(r"^[0-9a-f]{12}$")
_REQ_CLAIM_TIMEOUT = 600.0    # claimedのまま10分無応答なら再claim可 (母艦クラッシュ対応)
_REQ_LOCK = threading.Lock()  # request.json の read-modify-write を直列化


def requests_root() -> Path:
    """依頼置き場。packs_root() の隣 (Drive運転ならセッションを跨いで永続)。"""
    p = os.environ.get("VIDEOLAB_REQUESTS", "").strip()
    if p:
        return Path(p)
    return packs_root().parent / "requests"


def _req_load(rid: str) -> dict | None:
    try:
        d = json.loads((requests_root() / rid / "request.json")
                       .read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else None
    except Exception:                     # noqa: BLE001
        return None


def _req_save(rid: str, data: dict) -> None:
    """tmp→os.replace の原子的書き込み (一覧ポーリングとの torn read 防止)。"""
    rd = requests_root() / rid
    rd.mkdir(parents=True, exist_ok=True)
    tmp = rd / "request.json.tmp"
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    os.replace(tmp, rd / "request.json")


# walk_pack の生成規格 (compass_vace の 4x2 半球キャンバスと同一)
WALKPACK_NF = 57                 # 直立6+歩行2周期+末尾静止8 (walk_layout)
WALKPACK_W, WALKPACK_H = 480, 864   # 2x2 x セル240x432
WALKPACK_FPS = 16

# 生成の既定詳細設定 (2026-07-19 ユーザー指定)。母艦アプリの config.json と
# 同じ値をクラウド側 (walkpack) の既定にも入れ、どちらの経路でも絵が揃う。
# 環境変数が入っていればそちらが優先 (実験用の逃げ道)。
WP_DEFAULTS = {
    "SM_POSE_ARM_SWING": "0.8",
    "SM_POSE_LEG_SWING": "0.8",
    "SM_POSE_LEG_CROSS": "0.3",
    "SM_POSE_BOB": "0.9",
    # ★顔正面化は工房経路では切る (2026-07-19、ロップ実測)。
    # 立ち絵の顔が正面寄り (実測5°/8°) だと auto が発動して「体は45°・顔は
    # 0°」を宣言し、その無理な姿勢をモデルが髪で埋めて片方の斜め前だけ顔が
    # 髪に覆われた。offにすると顔も実測どおり左右対称に傾き (∓10°)、症状が
    # 消えることを実機で確認 (頭部色多様性 20.2→27.2、正面29.4と同水準)。
    # ★全体の既定は auto のまま (ここはwalkpackにだけ効く)。offは2026-07-18
    # に「歩行開始と同時に後頭部へ吸われる」で8+2試行全滅した旧構成でもある
    # ため。全滅したのは前後半球レイアウト導入前=rear語彙が同居していた頃で、
    # 今の F4/B4 + 斜めの左右明記なら安全と判断したが、GUI経路 (compass等)
    # まで巻き戻す根拠は無い。明示的に環境変数を設定すればこの既定は上書きできる。
    "SM_POSE_FACE_FRONT": "off",
}
WP_LR_REFINE = 0.55          # AniSora latent再加工のσ (旧既定0.45)
WP_LR_PIN_RELEASE = 0.2      # σ<この値でlatent固定を解除して質感を馴染ませる


def _wp_apply_pose_defaults() -> None:
    """姿勢の既定値を環境変数へ (未設定のときだけ)。pose_video /
    compass_vace は SM_POSE_* を読むので、ここで入れれば骨格生成に効く。"""
    for k, v in WP_DEFAULTS.items():
        if not os.environ.get(k, "").strip():
            os.environ[k] = v


# 管理ノブ (2026-07-20要望「私側から設定を変えられる管理GUI」): 受付台の
# /admin が GCS config/walkpack_knobs.json に保存し、walkpackが依頼ごとに
# 読む=サーバ再起動不要で次の生成から反映 (アプリの詳細設定モーダルと同じ
# 思想)。優先順位: 実験ノブ(_exp明示) > 管理ノブ > 環境変数 > 既定値。
# flyingの姿勢envだけは体格の生命線なので管理ノブより後に上書きする。
_WP_KNOB_ENV = {"nat_control": "SM_WP_NAT_CONTROL",
                "arm_swing": "SM_POSE_ARM_SWING",
                "leg_swing": "SM_POSE_LEG_SWING",
                "leg_cross": "SM_POSE_LEG_CROSS",
                "bob": "SM_POSE_BOB",
                "refine": "SM_VACE_LR_REFINE",
                "pin_release": "SM_VACE_LR_PIN_RELEASE"}


def _wp_admin_knobs() -> dict:
    """GCSの管理ノブを読む (無い/壊れている/非GCS環境は空={既定運転})。"""
    if not _gcs_active():
        return {}
    try:
        raw = _gcs_read("config/walkpack_knobs.json")
        if raw is None:
            return {}
        d = json.loads(raw.decode("utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:                             # noqa: BLE001
        return {}


def _wp_knob_int(kn: dict, key: str, dflt: int, lo: int, hi: int) -> int:
    """管理ノブの整数値をクランプして読む (壊れた値でジョブを落とさない)。"""
    try:
        raw = kn.get(key)
        return max(lo, min(hi, int(float(dflt if raw is None else raw))))
    except (TypeError, ValueError):
        return dflt


def _wp_knob_float(kn: dict, key: str, dflt: float, lo: float,
                   hi: float) -> float:
    """管理ノブの実数値をクランプして読む (壊れた値でジョブを落とさない)。"""
    try:
        raw = kn.get(key)
        return max(lo, min(hi, float(dflt if raw is None else raw)))
    except (TypeError, ValueError):
        return dflt


def _wp_knob_env_set(knobs: dict):
    """管理ノブを環境変数へ (復元用の退避を返す)。_WALKPACK_LOCK 直列前提。"""
    saved = {}
    for key, env in _WP_KNOB_ENV.items():
        if key not in knobs or knobs[key] is None:
            continue
        saved[env] = os.environ.get(env)
        os.environ[env] = str(knobs[key])
    return saved


# 飛行 (ホバリング) の骨格ノブ (2026-07-19ユーザー報告「浮遊ついてるのに
# 飛んでない」): walkpackは従来biped歩行の骨格+文面しか持たず、body_plan=
# flyingでも脚を振って歩いていた。専用骨格を作らず既存クランプ内で寄せる:
# 脚振りは下限0.3 (pose_videoのクランプ)・交差なし・腕は下限0.5・上下動を
# 強めてホバーの浮き沈みに流用する。文面は _WP_HOVER_PROMPT が担う。
_WP_FLYING_ENV = {"SM_POSE_LEG_SWING": "0.3", "SM_POSE_LEG_CROSS": "0.0",
                  "SM_POSE_ARM_SWING": "0.5", "SM_POSE_BOB": "1.6"}


def _wp_plan_env_set(meta: dict):
    """body_plan=flyingならノブを上書きし、復元用の退避を返す。
    _WALKPACK_LOCK で直列化されている前提 (env競合なし)。"""
    if str((meta or {}).get("body_plan") or "").strip() != "flying":
        return None
    saved = {}
    for k, v in _WP_FLYING_ENV.items():
        saved[k] = os.environ.get(k)
        os.environ[k] = v
    return saved


def _wp_plan_env_restore(saved) -> None:
    """flying上書きの復元。次のジョブ (biped等) に持ち越さない。"""
    for k, old in (saved or {}).items():
        if old is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = old


def _wp_edge_ref(im):
    """立ち絵1枚 → キャラ限定の線画 (a-2実験と同じ: 内部輝度エッジ+
    シルエット輪郭の白線、黒地RGB)。"""
    import numpy as np
    from PIL import Image, ImageFilter
    keyed = _wp_key(im)                           # RGBA・キャラのみ不透過
    arr = np.asarray(keyed)
    mask = arr[..., 3] > 128
    g = np.asarray(keyed.convert("L"), dtype=np.float32)
    gx = np.abs(np.diff(g, axis=1, prepend=g[:, :1]))
    gy = np.abs(np.diff(g, axis=0, prepend=g[:1]))
    inner = ((gx + gy) > 40) & mask               # キャラ内部の輝度エッジ
    m = mask.astype(np.uint8)
    sil = np.zeros_like(m)                        # シルエット輪郭 (境界画素)
    sil[1:-1, 1:-1] = m[1:-1, 1:-1] & ~(
        m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:])
    e = np.where(inner | (sil > 0), 255, 0).astype(np.uint8)
    return (Image.fromarray(e, "L").filter(ImageFilter.MaxFilter(3))
            .convert("RGB"))


def _wp_edge_canvas(cv_mod, refs: dict, w: int, h: int, layout):
    """立ち絵から「キャラ限定」エッジ制御キャンバスを作る (実験a-2)。

    v1 (参照キャンバスへFIND_EDGES直がけ) は、セル境界の四角枠や背景の
    色境界まで白線化して制御を汚し、顔消失・落ち影などの崩壊を招いた
    (2026-07-19実走)。v2はマゼンタキーでキャラを抜き、キャラ内部の輝度
    エッジ+シルエット輪郭だけを白線化。配置は本番と同じ compose_reference
    写像 (骨格とのキャラ位置一致が前提条件) に任せ、最後に白黒正規化して
    レターボックス余白・セル間隙のマゼンタを完全に消す。"""
    import tempfile
    import numpy as np
    from PIL import Image, ImageFilter
    tmp = Path(tempfile.mkdtemp(prefix="edge_refs_"))
    erefs = {}
    for d, rp in refs.items():
        eim = _wp_edge_ref(Image.open(rp))
        ep = tmp / f"{d}.png"
        eim.save(ep)
        erefs[d] = ep
    canvas = cv_mod.compose_reference(erefs, w, h, layout)
    ca = np.asarray(canvas.convert("RGB"), dtype=np.uint8)
    white = (ca.min(axis=2) > 180)                # ほぼ白の線だけ残す
    return Image.fromarray(
        np.where(white, 255, 0).astype(np.uint8), "L").convert("RGB")


def _wp_edge_head_canvas(cv_mod, refs: dict, w: int, h: int, layout,
                         head_frac: float = 0.45):
    """頭部限定のエッジ制御キャンバス (実験a-4、2026-07-19ユーザー発案
    「キャニーで頭部=最も崩れやすくて崩れたら困る部位を固定」)。

    頭部は歩行/ホバー中もほぼ剛体 (ボブで平行移動するだけ) なので、
    四肢と違い動きの窓に置いても矛盾しない。キャラのbbox上部 head_frac
    をエッジ許可ゾーンとし、それ以外は黒=自由領域。"""
    import tempfile
    import numpy as np
    from PIL import Image, ImageFilter
    tmp = Path(tempfile.mkdtemp(prefix="edge_head_"))
    erefs = {}
    for d, rp in refs.items():
        keyed = _wp_key(Image.open(rp))
        arr = np.asarray(keyed)
        mask = arr[..., 3] > 128
        rows = np.where(mask.any(axis=1))[0]
        g = np.asarray(keyed.convert("L"), dtype=np.float32)
        gx = np.abs(np.diff(g, axis=1, prepend=g[:, :1]))
        gy = np.abs(np.diff(g, axis=0, prepend=g[:1]))
        inner = ((gx + gy) > 40) & mask
        m = mask.astype(np.uint8)
        sil = np.zeros_like(m)
        sil[1:-1, 1:-1] = m[1:-1, 1:-1] & ~(
            m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:])
        e = (inner | (sil > 0))
        if len(rows):                       # 頭部ゾーン以外を消す
            top = rows[0]
            cut = top + int(round((rows[-1] - top + 1) * head_frac))
            e[cut:, :] = False
        eimg = np.where(e, 255, 0).astype(np.uint8)
        eim = (Image.fromarray(eimg, "L").filter(ImageFilter.MaxFilter(3))
               .convert("RGB"))
        ep = tmp / f"{d}.png"
        eim.save(ep)
        erefs[d] = ep
    canvas = cv_mod.compose_reference(erefs, w, h, layout)
    ca = np.asarray(canvas.convert("RGB"), dtype=np.uint8)
    white = (ca.min(axis=2) > 180)
    return Image.fromarray(
        np.where(white, 255, 0).astype(np.uint8), "L").convert("RGB")


def _wp_depth_ref(im):
    """立ち絵から疑似深度マップを作る (2026-07-20ユーザー発案「Depthを計測で
    動かせばいろんな形状に対応できるのでは」の第1弾)。

    単眼深度モデルは持ち込まず、シルエットの侵食距離でドーム状の起伏を
    作る近似 (輪郭=遠・中心=近)。VACEに「この形の立体が居る」と言うのが
    目的で、正確な深度である必要はない。戻り値はRGB化したグレースケール。"""
    import numpy as np
    from PIL import Image, ImageFilter
    keyed = _wp_key(im)
    a = np.asarray(keyed)[:, :, 3] > 128
    mask = Image.fromarray(np.where(a, 255, 0).astype(np.uint8), "L")
    # 繰り返しMinFilterで侵食し「何回で消えたか」= 輪郭からの深さ
    depth = np.zeros(a.shape, dtype=np.float32)
    cur = mask
    for i in range(24):
        arr = np.asarray(cur) > 128
        if not arr.any():
            break
        depth[arr] = i + 1
        cur = cur.filter(ImageFilter.MinFilter(3))
    if depth.max() > 0:
        depth /= depth.max()
    g = np.where(a, (70 + 185 * depth), 0).astype(np.uint8)
    return Image.fromarray(g, "L").convert("RGB")


# 体格別の手続き運動 (深度/線画リファレンスを1フレームぶん動かす)。
# 位相 ph=0..1 (歩行窓内)。戻り値 = (dx, dy, 回転deg, sx, sy)
def _wp_move_params(plan: str, ph: float, cw: int, ch: int):
    import math
    s = math.sin(2 * math.pi * 2.0 * ph)          # 2往復/周期
    if plan == "flying":
        return (0, round(-abs(ch * 0.012) * s), 0.0, 1.0, 1.0)
    if plan == "quadruped":
        # 這行のロッキング: 前後に小さく揺れ+わずかな上下
        return (round(cw * 0.008 * s), round(-ch * 0.006 * abs(s)),
                1.8 * s, 1.0, 1.0)
    if plan == "serpentine":
        return (round(cw * 0.02 * s), 0, 0.0, 1.0, 1.0)
    if plan in ("amorphous", "stretch_v"):
        # 上下伸縮 (動きの型4択のstretch_v=amorphousと同式のスカッシュ)
        return (0, 0, 0.0, 1.0 + 0.04 * s, 1.0 - 0.06 * s)
    if plan == "stretch_h":
        # 左右伸縮 (動きの型4択 2026-07-21): x主体の伸縮+yで体積補償
        return (0, 0, 0.0, 1.0 + 0.06 * s, 1.0 - 0.04 * s)
    if plan == "move_v":
        # 上下移動 (動きの型4択): flyingと同式の上下ボブ
        return (0, round(-abs(ch * 0.012) * s), 0.0, 1.0, 1.0)
    return (0, round(-ch * 0.008 * abs(s)), 0.0, 1.0, 1.0)   # other


def _wp_bottom_mask(canvas_rgb, layout, w: int, h: int, frac: float,
                    face_boxes: dict | None = None,
                    refs: dict | None = None):
    """セルごとにキャラbbox下部fracを255=生成、他を0=固定にしたLマスク。

    実験r (VACE空間マスク) と実験r2 (AniSora空間latent固定) の共通部品。
    face_boxes+refs があれば切断線を実測顔ボックスの下端+4%より下に
    クランプする (2026-07-21 赤さん瞬き実障害: 巨頭チビはbbox比率だけの
    切断線が顔の真ん中を横切り、目と口が生成領域に入って瞬き・口パクが
    復活した)。戻り値 = (ndarray uint8 [h, w], base64 PNG)。"""
    import base64 as _b64
    import io as _io
    import numpy as np
    from PIL import Image
    arr = np.array(canvas_rgb.convert("RGB"))
    mg = ((arr[..., 0] > 200) & (arr[..., 2] > 200) & (arr[..., 1] < 120))
    msk = np.zeros(arr.shape[:2], dtype=np.uint8)
    cols_n, rows_n, dirs_n = layout
    cwc, chc = w // cols_n, h // rows_n
    for ci, dn in enumerate(dirs_n):
        if dn is None:
            continue
        x0, y0 = (ci % cols_n) * cwc, (ci // cols_n) * chc
        sub = ~mg[y0:y0 + chc, x0:x0 + cwc]
        ys, xs = np.nonzero(sub)
        if ys.size < 50:
            continue
        cut = int(ys.max() - (ys.max() - ys.min() + 1) * frac)
        if face_boxes and refs and dn in face_boxes and dn in refs:
            try:
                iw, ih = Image.open(refs[dn]).size
                s = min(cwc / iw, chc / ih)     # compose_referenceの
                off_y = (chc - ih * s) / 2.0    # レターボックス写像
                fb_bottom = off_y + face_boxes[dn][3] * ih * s
                cut = max(cut, int(fb_bottom + chc * 0.04))
            except Exception:                 # noqa: BLE001
                pass
        if cut >= ys.max():
            continue                          # 顔が低すぎるセルは全凍結
        msk[y0 + cut:y0 + ys.max() + 1,
            x0 + xs.min():x0 + xs.max() + 1] = 255
    buf = _io.BytesIO()
    Image.fromarray(msk, "L").save(buf, format="PNG")
    return msk, _b64.b64encode(buf.getvalue()).decode("ascii")


def _wp_face_keep_mask(canvas_rgb, layout, w: int, h: int,
                       face_boxes: dict | None, refs: dict | None,
                       pad: float = 0.06, return_head_keep: bool = False):
    """A型インペイントマスク (2026-07-21ユーザー図解「Aにしないと体は
    動かない」): 凍結=顔ボックスの窓だけ+キャラbbox外の背景、
    生成=キャラの体全体。B型 (下部帯だけ生成) は顔より上の胴まで凍結して
    しまい、AIの芝居が下のひと帯に限られていた。
    戻り値 = (ndarray uint8 [h, w] 255=生成, base64 PNG)。
    return_head_keep=Trueでは第3値に、歩行ボブへ追従させる顔/頭部の
    固定島 (255=追従固定) も返す。"""
    import base64 as _b64
    import io as _io
    import numpy as np
    from PIL import Image
    arr = np.array(canvas_rgb.convert("RGB"))
    mg = ((arr[..., 0] > 200) & (arr[..., 2] > 200) & (arr[..., 1] < 120))
    msk = np.zeros(arr.shape[:2], dtype=np.uint8)
    head_keep = np.zeros(arr.shape[:2], dtype=np.uint8)
    cols_n, rows_n, dirs_n = layout
    cwc, chc = w // cols_n, h // rows_n
    # 背面の頭部固定 (2026-07-21ユーザー指摘「後ろ向きは頭部が無防備で
    # こっち向く危険」): 顔ボックスの無い方向 (背面系) は従来bbox全体が
    # 生成領域=頭部ごとAIに渡り、振り向き顔の発明を許していた。実測済み
    # 方向の顔ボックス高さ帯の中央値から「顔が出うる帯」を推定して凍結
    # する (立ち絵8方向は同一フォーマットなので相対高さは方向間で移送
    # 可能)。全方向未実測なら推定不能=従来どおり
    band = None
    fb_vals = list((face_boxes or {}).values())
    if fb_vals:
        _ys0 = sorted(v[1] for v in fb_vals)
        _ys1 = sorted(v[3] for v in fb_vals)
        band = (_ys0[len(_ys0) // 2], _ys1[len(_ys1) // 2])
    for ci, dn in enumerate(dirs_n):
        if dn is None:
            continue
        x0, y0 = (ci % cols_n) * cwc, (ci // cols_n) * chc
        sub = ~mg[y0:y0 + chc, x0:x0 + cwc]
        ys, xs = np.nonzero(sub)
        if ys.size < 50:
            continue
        # 生成=キャラbbox (少し外側へ余裕) — 背景の大半は凍結のまま
        m = max(2, int(chc * 0.02))
        by0, by1 = max(0, ys.min() - m), min(chc, ys.max() + 1 + m)
        bx0, bx1 = max(0, xs.min() - m), min(cwc, xs.max() + 1 + m)
        msk[y0 + by0:y0 + by1, x0 + bx0:x0 + bx1] = 255
        # 凍結=顔ボックスの窓 (パディング付き)
        if face_boxes and refs and dn in face_boxes and dn in refs:
            try:
                iw, ih = Image.open(refs[dn]).size
                s = min(cwc / iw, chc / ih)
                offx = (cwc - iw * s) / 2.0
                offy = (chc - ih * s) / 2.0
                fx0, fy0, fx1, fy1 = face_boxes[dn]
                px = (fx1 - fx0) * iw * s * pad
                py = (fy1 - fy0) * ih * s * pad
                gx0 = int(offx + fx0 * iw * s - px)
                gy0 = int(offy + fy0 * ih * s - py)
                gx1 = int(offx + fx1 * iw * s + px) + 1
                gy1 = int(offy + fy1 * ih * s + py) + 1
                _fy0, _fy1 = y0 + max(0, gy0), y0 + min(chc, gy1)
                _fx0, _fx1 = x0 + max(0, gx0), x0 + min(cwc, gx1)
                msk[_fy0:_fy1, _fx0:_fx1] = 0
                head_keep[_fy0:_fy1, _fx0:_fx1] = 255
            except Exception:                 # noqa: BLE001
                pass
        elif band is not None and refs and dn in refs:
            # 顔なしセル (背面系) の頭部帯凍結: 高さ=推定帯を上へ30%延長
            # (額・生え際まで)、横幅=帯内のシルエット実測 (髪ごと)。
            # 四足のおしり等、帯の外の芝居は生成領域のまま残る
            try:
                iw, ih = Image.open(refs[dn]).size
                s = min(cwc / iw, chc / ih)
                offy = (chc - ih * s) / 2.0
                bh = (band[1] - band[0]) * ih * s
                gy0 = max(0, int(offy + band[0] * ih * s - bh * 0.30))
                gy1 = min(chc, int(offy + band[1] * ih * s
                                   + bh * pad) + 1)
                rows = sub[gy0:gy1, :]
                xs_b = np.nonzero(rows.any(axis=0))[0]
                if xs_b.size:
                    mx = max(2, int(cwc * 0.02))
                    hx0 = max(0, int(xs_b.min()) - mx)
                    hx1 = min(cwc, int(xs_b.max()) + 1 + mx)
                    _hy0, _hy1 = y0 + gy0, y0 + gy1
                    _hx0, _hx1 = x0 + hx0, x0 + hx1
                    msk[_hy0:_hy1, _hx0:_hx1] = 0
                    head_keep[_hy0:_hy1, _hx0:_hx1] = 255
            except Exception:                 # noqa: BLE001
                pass
    buf = _io.BytesIO()
    Image.fromarray(msk, "L").save(buf, format="PNG")
    result = (msk, _b64.b64encode(buf.getvalue()).decode("ascii"))
    return result + (head_keep,) if return_head_keep else result


def _wp_bobbing_fixed_refs(canvas_rgb, generate_mask, head_keep, layout,
                            nf: int, idle_n: int, gait_end: int,
                            bob_scale: float = 1.0):
    """顔/背面頭部の原画と固定島を、歩行の上下ボブへ一緒に追従させる。

    bbox外背景は静止固定のまま。固定島だけを各セル内で上へ平行移動し、
    元位置は生成領域へ返す。歩行窓では既存の二足ボブ式と同位相
    (2周期中に4回、セル高の0.8%上昇)、先頭/末尾は元位置へ戻す。
    bob_scaleは管理ノブhead_bob (0.0--2.0、既定1.0)。
    戻り値=(時刻別RGB参照, 時刻別L生成mask, maskのbase64 PNG列)。"""
    import base64 as _b64
    import io as _io
    import numpy as np
    from PIL import Image
    base = np.asarray(canvas_rgb.convert("RGB"), dtype=np.uint8)
    gm = np.asarray(generate_mask, dtype=np.uint8)
    hk = np.asarray(head_keep, dtype=np.uint8) > 0
    if gm.shape != base.shape[:2] or hk.shape != base.shape[:2]:
        raise ValueError("歩行ボブmaskの寸法がcanvasと不一致")
    static_fixed = (gm < 128) & ~hk
    cols_n, rows_n, _dirs_n = layout
    cwc, chc = base.shape[1] // cols_n, base.shape[0] // rows_n
    win = max(1, gait_end - idle_n + 1)
    bob_scale = max(0.0, min(2.0, float(bob_scale)))
    refs_out, masks_out, masks_b64 = [], [], []
    for k in range(int(nf)):
        if k < idle_n or k > gait_end:
            dy = 0
        else:
            # other/bipedの既存式と一致: 4回の接地で頭が上がり、元へ戻る。
            dy = int(round(_wp_move_params(
                "other", (k - idle_n) / win, cwc, chc)[1] * bob_scale))
        shifted_head = np.zeros_like(hk)
        shifted_pixels = np.zeros_like(base)
        for ci in range(cols_n * rows_n):
            x0, y0 = (ci % cols_n) * cwc, (ci // cols_n) * chc
            x1, y1 = x0 + cwc, y0 + chc
            cell_h = hk[y0:y1, x0:x1]
            cell_p = base[y0:y1, x0:x1]
            if dy <= 0:
                take = chc + dy
                if take > 0:
                    shifted_head[y0:y0 + take, x0:x1] = cell_h[-dy:]
                    shifted_pixels[y0:y0 + take, x0:x1] = cell_p[-dy:]
            else:
                take = chc - dy
                if take > 0:
                    shifted_head[y0 + dy:y1, x0:x1] = cell_h[:take]
                    shifted_pixels[y0 + dy:y1, x0:x1] = cell_p[:take]
        ref_arr = base.copy()
        ref_arr[shifted_head] = shifted_pixels[shifted_head]
        mask_arr = np.full(gm.shape, 255, dtype=np.uint8)
        mask_arr[static_fixed] = 0
        mask_arr[shifted_head] = 0
        refs_out.append(Image.fromarray(ref_arr, "RGB"))
        masks_out.append(Image.fromarray(mask_arr, "L"))
        buf = _io.BytesIO()
        masks_out[-1].save(buf, format="PNG")
        masks_b64.append(_b64.b64encode(buf.getvalue()).decode("ascii"))
    return refs_out, masks_out, masks_b64


def _wp_dynamic_pose_masks(canvas_rgb, allowed_generate_mask, pose_frames,
                           nf: int, source_radius: int = 4,
                           pose_radius: int = 24):
    """frame0は全面固定、以後は実シルエット+骨周囲だけ生成。

    旧A型maskはキャラbboxの長方形を開け、腕の高さの背景まで
    AIに描き直させていた。その余白が暗い帯になるため、
    元絵のマゼンタキー実シルエットと、各時刻のPose線を
    太らせた移動先のみを開ける。allowed_generate_maskの外(顔/
    背景)は必ず固定。戻り値=(PIL L列, base64 PNG列)。"""
    import base64 as _b64
    import io as _io
    import numpy as np
    from PIL import Image

    base_rgb = np.asarray(canvas_rgb.convert("RGB"), dtype=np.int16)
    allowed = np.asarray(allowed_generate_mask, dtype=np.uint8) >= 128
    if allowed.shape != base_rgb.shape[:2]:
        raise ValueError("動的Pose maskの寸法がcanvasと不一致")

    def _box_dilate(mask, radius):
        """SciPy依存なしのO(HW)矩形膨張 (積分画像)。"""
        r = max(0, int(radius))
        if r <= 0:
            return mask.astype(bool, copy=True)
        src = np.pad(mask.astype(np.uint8), ((r, r), (r, r)))
        ii = np.pad(src, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        k = 2 * r + 1
        win = ii[k:, k:] - ii[:-k, k:] - ii[k:, :-k] + ii[:-k, :-k]
        return win > 0

    # 工房キーと同じ min(R,B)-G>=70 を背景と見なす。
    fg = ((np.minimum(base_rgb[..., 0], base_rgb[..., 2])
           - base_rgb[..., 1]) < 70)
    source_body = _box_dilate(fg & allowed, source_radius) & allowed
    poses = list(pose_frames or [])
    if not poses:
        raise ValueError("動的Pose maskにPoseフレームがありません")
    if len(poses) != int(nf):
        ids = [round(i * (len(poses) - 1) / max(1, int(nf) - 1))
               for i in range(int(nf))]
        poses = [poses[i] for i in ids]

    images, encoded = [], []
    for k, pose in enumerate(poses):
        if k == 0:
            dyn = np.zeros(allowed.shape, dtype=bool)
        else:
            pa = np.asarray((pose if pose.size == canvas_rgb.size
                             else pose.resize(canvas_rgb.size)).convert("RGB"))
            ink = pa.max(axis=2) >= 40
            pose_env = _box_dilate(ink & allowed, pose_radius) & allowed
            dyn = source_body | pose_env
        image = Image.fromarray((dyn.astype(np.uint8) * 255), "L")
        buf = _io.BytesIO()
        image.save(buf, format="PNG")
        images.append(image)
        encoded.append(_b64.b64encode(buf.getvalue()).decode("ascii"))
    return images, encoded


def _wp_motion_prompt(pack: Path) -> str:
    """パックのキャラ別モーション文 (母艦Codex作) を読む。

    2026-07-21ユーザー要望「母艦のCodexに毎回動きを英語でプロンプト化
    させ、GPUの動画プロンプトへ追加」。無し/壊れは空文字=注入なし。
    英語以外・過長は破棄 (プロンプト汚染防止)。"""
    p = pack / "01_generation" / "motion_prompt.txt"
    if not p.is_file():
        return ""
    try:
        t = " ".join(p.read_text(encoding="utf-8",
                                 errors="replace").split())[:600]
    except Exception:                             # noqa: BLE001
        return ""
    if len(t) < 20 or sum(1 for c in t if ord(c) < 128) < len(t) * 0.9:
        return ""
    return t


_WP_MOTION_CONTROLS = (
    "biped", "quadruped", "flying", "serpentine", "amorphous", "other")


def _wp_infer_motion_control(text: str, body_plan: str = "ai") -> str:
    """旧パックの動作文から内部制御方式を保守的に推定する。"""
    legacy = {
        "biped": "biped", "biped_legs": "biped",
        "quadruped": "quadruped", "quadruped_ai": "quadruped",
        "quadruped_bone": "quadruped", "flying": "flying",
        "flying_ai": "flying", "serpentine": "serpentine",
        "serpentine_ai": "serpentine", "amorphous": "amorphous",
        "amorphous_ai": "amorphous",
    }
    if body_plan in legacy:
        return legacy[body_plan]
    src = (text or "").lower()
    rules = (
        ("flying", ("flies in place", "flying in place", "hovers",
                    "hovering", "wingbeat", "wing beat", "flaps its wings")),
        ("quadruped", ("on all fours", "four-legged", "four legged",
                       "quadruped", "forelegs", "forelimbs", "hind legs",
                       "hindlimbs", "front paws", "crawls", "crawling")),
        ("serpentine", ("slithers", "slithering", "serpentine", "snake-like",
                        "snakelike", "legless body")),
        ("amorphous", ("amorphous", "slime", "gelatinous", "oozes",
                       "squashes and stretches", "squash and stretch", "blob")),
        ("biped", ("two legs", "left leg", "right leg", "left foot",
                   "right foot", "biped", "upright run", "upright walk")),
    )
    for kind, needles in rules:
        if any(n in src for n in needles):
            return kind
    # 0.11.34以前のaiパックは大半が二足。明記のない既存依頼だけは挙動を
    # 変えず、曖昧判定は新規profile側でotherとして保存する。
    return "biped" if body_plan == "ai" else "other"


def _wp_motion_control(pack: Path, body_plan: str = "ai") -> str:
    """母艦が同梱した内部制御種別を読む。表示上の4択とは独立。"""
    p = pack / "01_generation" / "motion_profile.json"
    if p.is_file():
        try:
            kind = str(json.loads(p.read_text(encoding="utf-8"))
                       .get("control_kind") or "").strip().lower()
            if kind in _WP_MOTION_CONTROLS:
                return kind
        except Exception:                         # noqa: BLE001
            pass
    return _wp_infer_motion_control(_wp_motion_prompt(pack), body_plan)


def _wp_motion_profile(pack: Path) -> dict:
    """母艦Codexの内部モーション判定を安全な値だけに正規化して読む。"""
    p = pack / "01_generation" / "motion_profile.json"
    if not p.is_file():
        return {}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:                             # noqa: BLE001
        return {}


def _wp_motion_gait(pack: Path) -> str:
    v = str(_wp_motion_profile(pack).get("gait") or "walk").strip().lower()
    return v if v in ("walk", "run", "crawl", "fly", "slither",
                      "pulse", "custom") else "walk"


def _wp_limb_mode(pack: Path) -> str:
    v = str(_wp_motion_profile(pack).get("limb_mode") or "full").strip().lower()
    return v if v in ("full", "legs", "arms", "none") else "full"


def _wp_face_boxes(pack: Path) -> dict:
    """pack/01_generation/face_boxes.json を読む (実験g3の顔限定線画用)。

    形式: {方向: [x0, y0, x1, y1]} — 立ち絵画像に対する相対座標 (0..1)。
    顔が見えない方向 (背面) は載せない。壊れた値は方向ごとに黙って捨てる
    (顔なし扱い=そのセルは黒ギャップ)。"""
    p = pack / "01_generation" / "face_boxes.json"
    if not p.is_file():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:                             # noqa: BLE001
        return {}
    out = {}
    for d, v in (raw or {}).items():
        try:
            x0, y0, x1, y1 = (float(v[0]), float(v[1]),
                              float(v[2]), float(v[3]))
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1:
            out[str(d)] = (x0, y0, x1, y1)
    return out


def _wp_face_mask_ref(im, box):
    """線画/深度リファレンスを顔ボックス (相対座標) だけ残して黒にする。
    手続き運動の前に適用するので、顔線は全身と同じ運動に追従する。"""
    import numpy as np
    from PIL import Image
    arr = np.array(im.convert("RGB"))
    H0, W0 = arr.shape[:2]
    x0 = max(0, int(round(box[0] * W0)))
    y0 = max(0, int(round(box[1] * H0)))
    x1 = min(W0, int(round(box[2] * W0)))
    y1 = min(H0, int(round(box[3] * H0)))
    keep = np.zeros((H0, W0), dtype=bool)
    keep[y0:y1, x0:x1] = True
    arr[~keep] = 0
    return Image.fromarray(arr, "RGB")


def _wp_moving_frames(cv_mod, refs: dict, plan: str, nf: int, idle_n: int,
                      gait_end: int, w: int, h: int, layout,
                      mode: str = "depth", face_boxes: dict | None = None):
    """深度/線画リファレンスを体格別の手続き運動で動かした制御フレーム列。

    骨格語彙に依存しないので任意形状に対応する: 形の権威=立ち絵実測の
    深度(または線画)、動き=体格ごとの剛体/伸縮変換 (ボブ・ロッキング・
    蛇行・拍動)。idle/末尾静止窓は無変換の同じ絵=静止アンカー。"""
    from PIL import Image
    import tempfile as _tf
    base_refs = {}
    _fill = (255, 0, 255) if mode == "art" else 0
    for d, rp in refs.items():
        im = Image.open(rp)
        if mode == "art":
            # 実験p (手続きアニメ): 参照絵そのものを動かす。背景=マゼンタ
            keyed = _wp_key(im)
            ref = Image.new("RGB", keyed.size, (255, 0, 255))
            ref.paste(keyed, (0, 0), keyed)
        else:
            ref = (_wp_depth_ref(im) if mode == "depth"
                   else _wp_edge_ref(im))    # line=a-2のキャラ限定線画
        if face_boxes is not None:
            # 実験g3: 顔ボックスだけ残す (無い方向=背面等は全黒=制御なし)
            ref = (_wp_face_mask_ref(ref, face_boxes[d])
                   if d in face_boxes
                   else Image.new("RGB", ref.size, 0))
        base_refs[d] = ref
    win = max(1, gait_end - idle_n + 1)
    cols_n, rows_n, _dirs_n = layout
    cw_cell, ch_cell = w // max(1, cols_n), h // max(1, rows_n)
    tmp = Path(_tf.mkdtemp(prefix="depth_refs_"))
    frames = []
    cache = {}
    for k in range(nf):
        if k < idle_n or k > gait_end:
            key = None
        else:
            key = _wp_move_params(plan, (k - idle_n) / win, cw_cell, ch_cell)
        if key in cache:
            frames.append(cache[key])
            continue
        drefs = {}
        for d, dim in base_refs.items():
            out = dim
            if key is not None:
                dx, dy, rot, sx, sy = key
                out = dim
                if rot or sx != 1.0 or sy != 1.0:
                    ow, oh = out.size
                    out = out.rotate(rot, resample=Image.BILINEAR,
                                     center=(ow / 2, oh * 0.9),
                                     fillcolor=_fill)
                    if sx != 1.0 or sy != 1.0:
                        nw, nh = max(1, round(ow * sx)), max(1, round(oh * sy))
                        scaled = out.resize((nw, nh), Image.BILINEAR)
                        out = Image.new(dim.mode, (ow, oh), _fill)
                        # 足元 (下端中央) を基準に貼る=接地維持
                        out.paste(scaled, ((ow - nw) // 2, oh - nh))
                if dx or dy:
                    sh = Image.new(out.mode, out.size, _fill)
                    sh.paste(out, (dx, dy))
                    out = sh
            p = tmp / f"{d}_{hash(key) & 0xffffff:x}.png"
            out.save(p)
            drefs[d] = p
        fr = cv_mod.compose_reference(drefs, w, h, layout).convert("RGB")
        if mode != "art":
            # ★セル間隙・レターボックスのマゼンタを黒へ正規化 (a-2と同じ
            # 教訓: 骨格キャンバスは全面黒地=マゼンタ混じりの制御は分布外)。
            # artモード (手続きアニメ) は成果物そのものなのでマゼンタ維持
            import numpy as _npM
            fa = _npM.asarray(fr).astype(_npM.int16)
            mg = (_npM.minimum(fa[..., 0], fa[..., 2]) - fa[..., 1]) >= 70
            fa[mg] = 0
            fr = Image.fromarray(fa.astype("uint8"), "RGB")
        cache[key] = fr
        frames.append(fr)
    return frames


# パペット (2026-07-20ユーザー発案「骨に線画を貼り付けて動かす感じにすれば
# アニメーションになるのかも」): 骨格キーポイントで線画をパーツ分割し、
# ボーンごとの剛体ワープで「キャラ自身の線が歩く」制御ビデオを作る。
# 骨の動きの正確さ×線画の同一性固定の合流。パーツ= (BODY_18の端点対)。
_PUPPET_PARTS = ((2, 3), (3, 4), (5, 6), (6, 7),      # 腕 (上腕/前腕 左右)
                 (8, 9), (9, 10), (11, 12), (12, 13),  # 脚 (腿/脛 左右)
                 (1, 0))                               # 頭 (首→鼻)
# 胴 = 首(1)と腰中点(8,11)の仮想セグメント (特別扱い)


def _sim_affine(p0, p1, q0, q1):
    """相似変換 q=T(p): p0->q0, p1->q1 の PIL逆係数 (dst->src)。"""
    import numpy as np
    vp, vq = p1 - p0, q1 - q0
    lp = float(np.hypot(*vp)) or 1.0
    lq = float(np.hypot(*vq)) or 1.0
    sc = lq / lp
    ang = np.arctan2(vq[1], vq[0]) - np.arctan2(vp[1], vp[0])
    ca, sa = np.cos(ang), np.sin(ang)
    # 順方向: q = sc*R*(p-p0)+q0 → 逆: p = R^-1*(q-q0)/sc + p0
    inv_s = 1.0 / sc
    a, b = ca * inv_s, sa * inv_s
    c = p0[0] - a * q0[0] - b * q0[1]
    d2, e2 = -sa * inv_s, ca * inv_s
    f2 = p0[1] + sa * inv_s * q0[0] - ca * inv_s * q0[1]
    return (a, b, c, d2, e2, f2)


def _wp_puppet_frames(line_canvas, kps_frames, layout, w: int, h: int):
    """線画キャンバス+フレーム別キーポイント → パペット制御フレーム列。

    各セルの線画画素を「idle姿勢のどのボーンに最も近いか」で分割し、
    フレームごとに idleボーン→現フレームボーン の相似変換でワープして
    再合成する。関節の割れ・パーツ重なりは制御用途では許容 (VACEが均す)。"""
    import numpy as np
    from PIL import Image
    cols, rows, dirs = layout
    cw, ch = w // cols, h // rows
    la = np.asarray(line_canvas.convert("L"))
    idle = kps_frames[0]
    n_cells = len(idle) if idle else 0

    def _seg_pts(kps):
        segs = []
        for a, b in _PUPPET_PARTS:
            if kps[a] is not None and kps[b] is not None:
                segs.append((np.array(kps[a]), np.array(kps[b])))
            else:
                segs.append(None)
        # 胴: 首→腰中点
        if kps[1] is not None and kps[8] is not None and kps[11] is not None:
            hip = (np.array(kps[8]) + np.array(kps[11])) / 2.0
            segs.append((np.array(kps[1]), hip))
        else:
            segs.append(None)
        return segs

    # セル矩形をdirs順に (Noneセルはキーポイント記録に現れない)
    cell_rects = []
    ci = 0
    for i, d in enumerate(dirs):
        if d is None:
            continue
        cell_rects.append(((i % cols) * cw, (i // cols) * ch))
        ci += 1
    cell_rects = cell_rects[:n_cells]

    # パーツ割り当て (セルごとに1回)
    cell_parts = []          # [(part_layers, idle_segs, (ox,oy))]
    for ci, (ox, oy) in enumerate(cell_rects):
        segs0 = _seg_pts(idle[ci])
        sub = la[oy:oy + ch, ox:ox + cw]
        ys, xs = np.nonzero(sub > 40)
        if not len(ys):
            cell_parts.append(None)
            continue
        pts = np.stack([xs + ox, ys + oy], axis=1).astype(np.float64)
        dists = np.full((len(pts), len(segs0)), 1e9)
        for si, seg in enumerate(segs0):
            if seg is None:
                continue
            p0, p1 = seg
            v = p1 - p0
            vv = float(v @ v) or 1.0
            t = np.clip(((pts - p0) @ v) / vv, 0.0, 1.0)
            proj = p0[None, :] + t[:, None] * v[None, :]
            dists[:, si] = np.linalg.norm(pts - proj, axis=1)
        # 頭の所有権を優先 (2026-07-20ユーザー指摘「これ後頭部割れますよw」):
        # 最近傍だけだと後頭部の線が胴パーツに取られ、頭がボブした瞬間に
        # 頭蓋の線が泣き別れる。①頭セグメント距離に0.5バイアス (髭など
        # アゴ下の頭部要素も頭へ)、②首より上の画素は無条件で頭
        _head_si = len(_PUPPET_PARTS) - 1          # (1,0)=首→鼻
        dists[:, _head_si] *= 0.5
        owner = np.argmin(dists, axis=1)
        _neck = idle[ci][1]
        if _neck is not None and segs0[_head_si] is not None:
            owner[pts[:, 1] < float(_neck[1])] = _head_si
        layers = []
        for si in range(len(segs0)):
            sel = owner == si
            if not sel.any() or segs0[si] is None:
                layers.append(None)
                continue
            lay = np.zeros((ch, cw), dtype=np.uint8)
            lay[ys[sel], xs[sel]] = sub[ys[sel], xs[sel]]
            layers.append(Image.fromarray(lay, "L"))
        cell_parts.append((layers, segs0, (ox, oy)))

    frames = []
    cache = {}
    for kf in kps_frames:
        key = tuple(
            tuple((round(p[0], 1), round(p[1], 1)) if p else None
                  for p in kps)
            for kps in kf)
        if key in cache:
            frames.append(cache[key])
            continue
        canvas = np.zeros((h, w), dtype=np.uint8)
        for ci, cp in enumerate(cell_parts):
            if cp is None or ci >= len(kf):
                continue
            layers, segs0, (ox, oy) = cp
            segsF = _seg_pts(kf[ci])
            for si, lay in enumerate(layers):
                if lay is None or segsF[si] is None or segs0[si] is None:
                    continue
                # 頭も他パーツ同様にボーン相似変換で動かす (上下ボブ追従)。
                # 完全固定案 (0.10.35) はユーザー裁定で撤回 —
                # 「上下移動はしてほしい」(2026-07-20)
                p0, p1 = segs0[si]
                q0, q1 = segsF[si]
                # セルローカル座標へ
                co = _sim_affine(p0 - (ox, oy), p1 - (ox, oy),
                                 q0 - (ox, oy), q1 - (ox, oy))
                warped = lay.transform((cw, ch), Image.AFFINE, co,
                                       resample=Image.BILINEAR)
                wa = np.asarray(warped)
                region = canvas[oy:oy + ch, ox:ox + cw]
                np.maximum(region, wa, out=region)
        fr = Image.fromarray(canvas, "L").convert("RGB")
        cache[key] = fr
        frames.append(fr)
    return frames


def _wp_puppet_art_frames(art_canvas, kps_frames, layout, w: int, h: int,
                          freeze_head: bool = True):
    """原画の各画素を最寄りボーンへ割り当て、色付きパペット下地を作る。

    OpenPoseを画像条件として混ぜてもAniSoraは意味解釈せず、左右向きでは
    緑の骨が漏れるだけだった。そこで側面だけは、原画の脚画素そのものを
    交互位相へ仮移動する。粗い関節境界は最終画ではなくHigh予測の下地で、
    AniSoraが修復する。顔は空間maskの固定位置と衝突しないよう原位置固定。
    """
    import numpy as np
    from PIL import Image
    cols, rows, dirs = layout
    cw, ch = w // cols, h // rows
    rgb = np.asarray(art_canvas.convert("RGB"))
    # 工房のマゼンタキー。背景画素をボーンへ所有させない。
    fg = ((np.minimum(rgb[..., 0], rgb[..., 2]).astype(np.int16)
           - rgb[..., 1].astype(np.int16)) < 70)
    idle = kps_frames[0]

    def _seg_pts(kps):
        segs = []
        for a, b in _PUPPET_PARTS:
            if kps[a] is not None and kps[b] is not None:
                segs.append((np.array(kps[a]), np.array(kps[b])))
            else:
                segs.append(None)
        if kps[1] is not None and kps[8] is not None and kps[11] is not None:
            hip = (np.array(kps[8]) + np.array(kps[11])) / 2.0
            segs.append((np.array(kps[1]), hip))
        else:
            segs.append(None)
        return segs

    cell_rects = [((i % cols) * cw, (i // cols) * ch)
                  for i, d in enumerate(dirs) if d is not None]
    cell_rects = cell_rects[:len(idle) if idle else 0]
    cell_parts = []
    _head_si = len(_PUPPET_PARTS) - 1
    for ci, (ox, oy) in enumerate(cell_rects):
        segs0 = _seg_pts(idle[ci])
        sub_fg = fg[oy:oy + ch, ox:ox + cw]
        ys, xs = np.nonzero(sub_fg)
        if not len(ys):
            cell_parts.append(None)
            continue
        pts = np.stack([xs + ox, ys + oy], axis=1).astype(np.float64)
        dists = np.full((len(pts), len(segs0)), 1e9)
        for si, seg in enumerate(segs0):
            if seg is None:
                continue
            p0, p1 = seg
            v = p1 - p0
            vv = float(v @ v) or 1.0
            t = np.clip(((pts - p0) @ v) / vv, 0.0, 1.0)
            proj = p0[None, :] + t[:, None] * v[None, :]
            dists[:, si] = np.linalg.norm(pts - proj, axis=1)
        dists[:, _head_si] *= 0.5
        owner = np.argmin(dists, axis=1)
        neck = idle[ci][1]
        if neck is not None and segs0[_head_si] is not None:
            owner[pts[:, 1] < float(neck[1])] = _head_si
        layers = []
        for si in range(len(segs0)):
            sel = owner == si
            if not sel.any() or segs0[si] is None:
                layers.append(None)
                continue
            la = np.zeros((ch, cw, 4), dtype=np.uint8)
            gy, gx = ys[sel] + oy, xs[sel] + ox
            la[ys[sel], xs[sel], :3] = rgb[gy, gx]
            la[ys[sel], xs[sel], 3] = 255
            layers.append(Image.fromarray(la, "RGBA"))
        cell_parts.append((layers, segs0, (ox, oy)))

    # 背景は元キャンバスからキャラだけをキー色へ戻す。セル間も保持。
    bg = rgb.copy()
    bg[fg] = (255, 0, 255)
    frames, cache = [], {}
    for kf in kps_frames:
        key = tuple(tuple((round(p[0], 1), round(p[1], 1)) if p else None
                          for p in kps) for kps in kf)
        if key in cache:
            frames.append(cache[key])
            continue
        canvas = Image.fromarray(bg, "RGB").convert("RGBA")
        for ci, cp in enumerate(cell_parts):
            if cp is None or ci >= len(kf):
                continue
            layers, segs0, (ox, oy) = cp
            segsF = _seg_pts(kf[ci])
            # 胴→四肢→頭。顔は最後に原位置で載せる。
            order = [len(segs0) - 1] + [
                i for i in range(len(segs0) - 1) if i != _head_si] + [_head_si]
            for si in order:
                lay = layers[si]
                if lay is None or segsF[si] is None or segs0[si] is None:
                    continue
                if freeze_head and si == _head_si:
                    warped = lay
                else:
                    p0, p1 = segs0[si]
                    q0, q1 = segsF[si]
                    co = _sim_affine(p0 - (ox, oy), p1 - (ox, oy),
                                     q0 - (ox, oy), q1 - (ox, oy))
                    warped = lay.transform((cw, ch), Image.AFFINE, co,
                                           resample=Image.BILINEAR)
                canvas.alpha_composite(warped, (ox, oy))
        fr = canvas.convert("RGB")
        cache[key] = fr
        frames.append(fr)
    return frames


def _wp_ai_control_frames(cv_mod, pv_mod, refs: dict, motion_kind: str,
                          nf: int, idle_n: int, gait_end: int,
                          w: int, h: int, layout, gait: str = "walk",
                          limb_mode: str = "full"):
    """ai本線の内部制御列。側面二足だけは色付き原画パペットを使う。"""
    dirs = list(layout[2]) if len(layout) >= 3 else []
    side = (len(dirs) == 1 and dirs[0] in ("left", "right"))
    if motion_kind == "biped":
        kps = []
        old_gait = os.environ.get("SM_POSE_GAIT")
        old_parts = os.environ.get("SM_POSE_BODY_PARTS")
        os.environ["SM_POSE_GAIT"] = "run" if gait == "run" else "walk"
        os.environ["SM_POSE_BODY_PARTS"] = limb_mode
        try:
            pose = pv_mod.build_canvas_pose_frames(
                refs, nf, w, h, layout, kps_out=kps)
        finally:
            if old_gait is None:
                os.environ.pop("SM_POSE_GAIT", None)
            else:
                os.environ["SM_POSE_GAIT"] = old_gait
            if old_parts is None:
                os.environ.pop("SM_POSE_BODY_PARTS", None)
            else:
                os.environ["SM_POSE_BODY_PARTS"] = old_parts
        if side and limb_mode == "full":
            art = cv_mod.compose_reference(refs, w, h, layout).convert("RGB")
            return (_wp_puppet_art_frames(art, kps, layout, w, h),
                    "side_art_puppet")
        return pose, f"openpose_{limb_mode}_{gait}"
    return (_wp_moving_frames(cv_mod, refs, motion_kind, nf, idle_n,
                              gait_end, w, h, layout, mode="art"),
            f"{motion_kind}_art_motion")


# スクリブル混成 (2026-07-20ユーザー発案「首から上=立ち絵そのものの線画、
# 下=ボーンの線画、をスクリブルとして入力したら?」): 頭はキャラ自身の
# 線 (同一性の権威・ボブ追従)、体は白ストロークの棒人間 (服のヒダを
# 変形させない純粋なポーズ指示)。ローブ等でパペットの脚が溶ける問題への
# 回答 — 体には最初から服を描かない。
_SCRIBBLE_LIMBS = ((1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
                   (1, 8), (1, 11), (8, 11),
                   (8, 9), (9, 10), (11, 12), (12, 13))


def _wp_scribble_frames(line_canvas, kps_frames, layout, w: int, h: int):
    """頭=線画パペット (ボブ追従) + 体=白ストローク棒人間の制御フレーム列。"""
    import numpy as np
    from PIL import Image, ImageDraw
    cols, rows, dirs = layout
    cw, ch = w // cols, h // rows
    la = np.asarray(line_canvas.convert("L"))
    idle = kps_frames[0]
    _head_si = len(_PUPPET_PARTS) - 1              # (1,0)=首→鼻

    def _head_seg(kps):
        if kps[1] is not None and kps[0] is not None:
            return (np.array(kps[1], dtype=float),
                    np.array(kps[0], dtype=float))
        return None

    cell_rects = []
    for i, d in enumerate(dirs):
        if d is None:
            continue
        cell_rects.append(((i % cols) * cw, (i // cols) * ch))
    cell_rects = cell_rects[:len(idle) if idle else 0]

    # 頭レイヤー抽出 (パペットと同じ所有権: 首より上+アゴ下バイアス)
    heads = []
    for ci, (ox, oy) in enumerate(cell_rects):
        seg0 = _head_seg(idle[ci])
        neck = idle[ci][1]
        sub = la[oy:oy + ch, ox:ox + cw]
        if seg0 is None or neck is None:
            heads.append(None)
            continue
        ys, xs = np.nonzero(sub > 40)
        if not len(ys):
            heads.append(None)
            continue
        pts = np.stack([xs + ox, ys + oy], axis=1).astype(np.float64)
        # 頭所有: 首点 (BODY_18の1=肩中央) より上だけ。髭・アゴも首点より
        # 上にあるのでこれで足りる。距離球バイアスは肩の線まで頭に
        # 持って行く過剰さがあったため撤去 (2026-07-20実測)
        own = pts[:, 1] < float(neck[1])
        lay = np.zeros((ch, cw), dtype=np.uint8)
        lay[ys[own], xs[own]] = sub[ys[own], xs[own]]
        heads.append((Image.fromarray(lay, "L"), seg0, (ox, oy)))

    sw = max(3, round(4 * min(cw, ch) / 512.0) + 2)   # 線画と同程度の白線
    frames = []
    cache = {}
    for kf in kps_frames:
        key = tuple(tuple((round(p[0], 1), round(p[1], 1)) if p else None
                          for p in kps) for kps in kf)
        if key in cache:
            frames.append(cache[key])
            continue
        limb_img = Image.new("L", (w, h), 0)
        dr = ImageDraw.Draw(limb_img)
        for ci in range(min(len(heads), len(kf))):
            kps = kf[ci]
            # 体: 白ストロークの棒人間
            for a, b in _SCRIBBLE_LIMBS:
                if kps[a] is None or kps[b] is None:
                    continue
                dr.line([tuple(kps[a]), tuple(kps[b])],
                        fill=255, width=sw, joint="curve")
        canvas = np.array(limb_img)          # 書込可能コピー
        for ci, hd in enumerate(heads):
            if hd is None or ci >= len(kf):
                continue
            # 頭: 線画パペット (ボブ追従の相似変換)
            lay, seg0, (ox, oy) = hd
            segF = _head_seg(kf[ci])
            if segF is None:
                continue
            p0, p1 = seg0
            q0, q1 = segF
            co = _sim_affine(p0 - (ox, oy), p1 - (ox, oy),
                             q0 - (ox, oy), q1 - (ox, oy))
            warped = lay.transform((cw, ch), Image.AFFINE, co,
                                   resample=Image.BILINEAR)
            region = canvas[oy:oy + ch, ox:ox + cw]
            np.maximum(region, np.asarray(warped), out=region)
        fr = Image.fromarray(canvas, "L").convert("RGB")
        cache[key] = fr
        frames.append(fr)
    return frames


def _wp_flying_frames(frames, idle_n: int, gait_end: int, canvas_h: int):
    """飛行用poseフレーム列: 先頭の直立骨格を全フレームに使い、歩行窓だけ
    上下ボブ (sin・2往復) を与える。ノブで脚振りを最小化しても、歩行骨格を
    条件に渡す限りVACEはステップを描く (2026-07-19ユーザー報告「ポーズ画像に
    引っ張られて浮遊しない」— 条件付けはプロンプトより強い)。だから条件
    そのものをホバーにする: 脚は全フレーム直立のまま、全身が上下に浮き沈み
    し、羽ばたき等のディテールは文面とAniSora再加工に任せる。末尾静止窓は
    ボブも止めて直立コマ選出のアンカーを保つ。"""
    import math
    from PIL import Image
    base = frames[0]
    amp = max(2, round(canvas_h * 0.012))
    win = max(1, gait_end - idle_n + 1)
    out = []
    for k in range(len(frames)):
        if k < idle_n or k > gait_end:
            out.append(base)
            continue
        dy = round(amp * math.sin(2 * math.pi * 2.0 * (k - idle_n) / win))
        if not dy:
            out.append(base)
            continue
        im = Image.new(base.mode, base.size, 0)
        im.paste(base, (0, dy))
        out.append(im)
    return out
_WALKPACK_LOCK = threading.Lock()   # 直列化 (SM_LEG_SCALE等のenvを守る)
_WP_DIRS = ("front", "left", "right", "back",
            "front_left", "front_right", "back_left", "back_right")
_WP_TURN_ORDER = ("front", "front_left", "left", "back_left",
                  "back", "back_right", "right", "front_right")
_WP_DIRS_LONG = sorted(_WP_DIRS, key=len, reverse=True)
_WP_PID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
# ファイル名は Unicode 語文字を許可 (本番コーパスは日本語キャラ名:
# 真ロップ_01_front_centered.png 等。ASCII限定だと8方向PNGが黙って
# スキップされ全アップロードが400になる 2026-07-18スモーク実測)。
# パス安全性は basename化+resolve封じ込めが担うため、ここでは
# 先頭ドット・パス区切り・制御文字だけ弾けばよい
_WP_FNAME_RE = re.compile(r"^[\w][\w.@ \-]{0,120}$", re.UNICODE)
# make_walk_preview.py と同じ 2x4 プレビュー格子 (direction -> (col, row))
_WP_PREVIEW_GRID = {"front": (0, 0), "left": (0, 1), "right": (0, 2),
                    "back": (0, 3), "front_left": (1, 0),
                    "front_right": (1, 1), "back_left": (1, 2),
                    "back_right": (1, 3)}
# build_T_sheet_from_mp4.py と同じ T配置 (direction -> (row, block先頭col))
_WP_SHEET_PLACE = {"front": (0, 0), "left": (1, 0), "right": (2, 0),
                   "back": (3, 0), "front_left": (0, 6),
                   "front_right": (1, 6), "back_left": (2, 6),
                   "back_right": (3, 6)}
_WP_MP4_RE = re.compile(
    r"^(?P<char>.+)_(?P<idx>\d{2})_(?P<dir>[a-z_]+)_walkT\.mp4$")


def _wp_print(msg: str) -> None:
    """コンソールのコードページ非依存print (ログ出力でジョブを殺さない)。"""
    try:
        print(msg, flush=True)
    except Exception:                     # noqa: BLE001
        try:
            print(msg.encode("ascii", "replace").decode(), flush=True)
        except Exception:                 # noqa: BLE001
            pass


def _pack_meta(pack: Path) -> dict:
    try:
        m = json.loads((pack / "meta.json").read_text(encoding="utf-8"))
        return m if isinstance(m, dict) else {}
    except Exception:                     # noqa: BLE001
        return {}


def _pack_refs_dir(pack: Path) -> tuple:
    """パック内の split_centered から (char名, 方向->立ち絵Path) を取る。"""
    src = pack / "01_generation" / "split_centered"
    refs: dict = {}
    char = None
    if src.is_dir():
        for p in sorted(src.glob("*_centered.png")):
            stem = p.stem[: -len("_centered")]
            for d in _WP_DIRS_LONG:
                if stem.endswith("_" + d):
                    refs[d] = p
                    char = char or stem[: -(len(d) + 1)]
                    break
    return (char or "char"), refs


def _pack_extract(pid: str, raw: bytes, pack_kind: str = "walkpack") -> int:
    """zipバイト列をパック構造へ展開する (既存同名は置換)。

    パストラバーサル対策は二重: ①メンバー名はbasenameだけを使い、置き先は
    種別ごとの固定ディレクトリに限定 ②書き込み直前に resolve() でパック
    ディレクトリ内であることを検証。

    walkpackは従来どおり8方向PNGを必須にする。annexは一枚絵I2Vなので
    annex_source.png + annex_prompt_en.txtだけをルートへ展開し、8方向検査を
    絶対に通さない。"""
    if pack_kind not in ("walkpack", "annex"):
        raise ValueError(f"不正なパック種別です: {pack_kind}")
    annex = pack_kind == "annex"
    pack = (packs_root() / pid)
    root = packs_root()
    root.mkdir(parents=True, exist_ok=True)
    if pack.exists():
        shutil.rmtree(pack)
    sc = pack / "01_generation" / "split_centered"
    if annex:
        pack.mkdir(parents=True)
    else:
        sc.mkdir(parents=True)
    count = 0
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        total = sum(i.file_size for i in zf.infolist())
        if total > 500 * 2**20:
            raise ValueError("zipの展開後サイズが大きすぎます (>500MB)")
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = PurePosixPath(info.filename.replace("\\", "/")).name
            if not name or not _WP_FNAME_RE.match(name):
                continue
            if name == "meta.json":
                dest = pack / "meta.json"
            elif annex and name in ("annex_source.png",
                                    "annex_prompt_en.txt"):
                dest = pack / name
            elif annex:
                continue        # 分室は上の固定3ファイル以外を展開しない
            elif name == "landmarks.json":
                dest = pack / "01_generation" / "landmarks.json"
            elif name == "face_boxes.json":
                # 実験g3: VLM実測の顔ボックス (方向→相対座標)
                dest = pack / "01_generation" / "face_boxes.json"
            elif name == "motion_prompt.txt":
                # キャラ別モーション文 (母艦Codex作、2026-07-21)
                dest = pack / "01_generation" / "motion_prompt.txt"
            elif name == "motion_profile.json":
                # 4択UIを増やさず、キャラの形に合う内部制御だけを自動選択。
                dest = pack / "01_generation" / "motion_profile.json"
            elif name == "template.json":
                # 依頼のシート形式 (母艦が templates/<name>.json を同梱)。
                # ★これを落とすと _wp_sheet_layout が見つけられず、無言で
                # T規格へフォールバックする。2026-07-19: ウディタ8方向で
                # 依頼したのに 768x512 のT規格シートが出てきた実害。
                # meta.json の "template" は正しいのにシートだけ違う、と
                # いう分かりにくい壊れ方をするので必ず通すこと。
                dest = pack / "template.json"
            elif name.lower().endswith(".png"):
                dest = sc / name
            else:
                continue        # 想定外の拡張子は置かない
            rd = dest.resolve()
            if not str(rd).startswith(str(pack.resolve()) + os.sep):
                raise ValueError(f"不正な展開先: {info.filename}")
            with zf.open(info) as f:
                data = f.read(120 * 2**20 + 1)
            if len(data) > 120 * 2**20:
                raise ValueError(f"ファイルが大きすぎます: {name}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            count += 1
    if annex:
        missing = [name for name in ("annex_source.png",
                                     "annex_prompt_en.txt")
                   if not (pack / name).is_file()]
        if missing:
            shutil.rmtree(pack, ignore_errors=True)
            raise ValueError(f"分室パックのファイルが不足しています: {missing}")
        meta_p = pack / "meta.json"
        meta = _pack_meta(pack)
        meta.setdefault("type", "annex_i2v")
        meta.setdefault("name", pid)
        meta["created"] = time.time()
        meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                          encoding="utf-8")
        return count
    char, refs = _pack_refs_dir(pack)
    missing = [d for d in _WP_DIRS if d not in refs]
    if missing:
        shutil.rmtree(pack, ignore_errors=True)
        raise ValueError(f"8方向PNGが不足しています: {missing} "
                         "(*_<方向>_centered.png を8枚入れてください)")
    meta_p = pack / "meta.json"
    meta = _pack_meta(pack)
    meta.setdefault("name", pid)
    meta.setdefault("char_id", char if char != "char" else pid)
    meta.setdefault("leg_scale", 1.0)
    meta.setdefault("cell_w", 64)
    meta.setdefault("cell_h", 128)
    meta["created"] = time.time()
    meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                      encoding="utf-8")
    return count


def _wp_offload() -> str:
    """L4級 (VRAM<30GB) は両ステージ block offload — compass_vace の
    latent_refine 分岐と同じ裁定 (素GGUFはseq不可・常駐は22.5GB超)。"""
    try:
        import torch
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / 2**30
            if 0 < total < 30:
                return "block"
    except Exception:                     # noqa: BLE001
        pass
    return ""


# body_plan=flying 用のキャンバス文面。canvas_walk.CANVAS_PROMPT と同じ
# 拘束 (セル中央固定・向き固定・カメラ固定・マゼンタ背景) を保ち、
# 歩行の記述だけをホバリングに置き換える (2026-07-19)
_WP_HOVER_PROMPT = (
    "A game character sprite sheet: the same chibi flying character shown "
    "in several fixed compass cells. Each figure hovers in place at a fixed "
    "height with steady rhythmic wingbeats and a gentle vertical bob -- the "
    "feet never touch the ground and never take a step, the legs hang "
    "relaxed and still. Every figure stays centered in its own cell and "
    "keeps facing its own fixed direction the entire time; the body and "
    "head never turn, rotate, spin, or change direction. Static locked-off "
    "tripod camera, fixed frame, no camera movement, no pan, no zoom, no "
    "orbit. Plain flat magenta background, smooth seamless looping hover "
    "animation."
)

# 新本線 ai の専用文面。体領域は純ノイズからAniSoraが
# 作るため、「subtle」「exact posture/silhouette」のような元姿勢の
# 再建を促す語は入れない。キャラ別の具体的な動きは母艦Codexの
# motion_promptが後置で完成させる。
_WP_AI_PROMPT = (
    "A game character animation. The character performs continuous, clear, "
    "readable in-place locomotion for the entire video. Complete at least "
    "two full locomotion cycles without pausing: one side of the body leads, "
    "then the opposite side leads, with unmistakably different poses at the "
    "quarter points of each cycle. Limbs or equivalent body parts articulate, "
    "weight visibly shifts, and the torso responds naturally. This is a real "
    "animation cycle, never a still image, pose morph, or barely moving idle. "
    "The motion continues rhythmically from start to finish. Keep "
    "the character's identity, outfit, colors, proportions, and number of "
    "limbs consistent throughout the cycle."
)

_WP_AI_CELL_LOCK = (
    " Every figure stays centered in its own cell and keeps facing its own "
    "fixed direction for the entire animation; neither the body nor the "
    "head turns toward another direction. Static locked-off tripod camera, "
    "fixed frame, no camera movement, no pan, no zoom, no orbit. Plain flat "
    "magenta background, smooth seamless looping animation."
)

# 共通拘束 (セル固定・向き固定・カメラ固定・マゼンタ背景)。体格別文面の
# 語尾に連結する (2026-07-20 赤さん実障害: quadrupedにも歩行文面が出て
# ハイハイが直立歩行に化けた)
_WP_CELL_LOCK = (
    " Every figure stays centered in its own cell and keeps facing its own "
    "fixed direction the entire time; the body and head never turn, rotate, "
    "spin, or change direction. The character never stands up on two legs "
    "and never changes its overall posture from the reference. Static "
    "locked-off tripod camera, fixed frame, no camera movement, no pan, no "
    "zoom, no orbit. Plain flat magenta background, smooth seamless looping "
    "animation."
)

# 非二足の「自然移動」体格 -> 動きの文面。骨格は出さず (二足マネキンしか
# 無いため)、キー錨+立ち絵と文面だけで動きを誘導する
_WP_PLAN_PROMPTS = {
    "quadruped": (
        "A game character sprite sheet: the same character shown in several "
        "fixed compass cells. Each figure moves on all fours in place with "
        "a steady four-legged gait -- hands/front limbs and knees/hind "
        "limbs alternate in a natural crawling rhythm while the body stays "
        "low to the ground the entire time."
    ),
    "serpentine": (
        "A game character sprite sheet: the same character shown in several "
        "fixed compass cells. Each figure slithers in place with a smooth "
        "serpentine undulation -- the body waves side to side in a steady "
        "rhythm, staying low to the ground, with no legs and no steps."
    ),
    "amorphous": (
        "A game character sprite sheet: the same character shown in several "
        "fixed compass cells. Each figure bounces and squishes in place "
        "with a soft rhythmic wobble -- the whole body gently squashes and "
        "stretches like a blob, keeping its overall silhouette."
    ),
    "other": (
        "A game character sprite sheet: the same character shown in several "
        "fixed compass cells. Each figure moves in place with the natural "
        "locomotion that fits its body -- a subtle steady movement rhythm "
        "that keeps the exact posture and silhouette of the reference."
    ),
}
# 上記の体格 (walkpackで二足歩行骨格を当ててはいけないもの)
_WP_NAT_PLANS = tuple(_WP_PLAN_PROMPTS)

# 発明抑制 (2026-07-20ユーザー観察「キャラを動かさずに何もない場所に新しく
# 何かを描く」への第1弾): stage2は guidance=1.0 でネガティブプロンプトが
# 効かないため、「空白は空白のまま」を正のプロンプトで宣言する。骨格なし
# 経路はσ0.9の自由スロットが約8個あり、ノイズの解決先から「新規物体」を
# 言語で奪う。bipedは骨格latentが動きを強制するので対象外 (実績運転不変)
_WP_NO_PROPS = (
    " Nothing new ever appears anywhere in the frame: no props, no objects, "
    "no effects, no particles, no extra creatures; the flat magenta "
    "background stays completely empty at all times.")


def _wp_prompt(eng: dict, refs: dict, layout, nf: int,
               plan: str = "biped", gait_run: bool = False,
               keep_posture: bool = False,
               crawl_bone: bool = False) -> str:
    """CANVAS_PROMPT + NO_WIND + (スカート/末尾静止節) + 方向明文。

    顔正面化/体ヨー追従の発動判定は compass_vace._run_layout と同じく
    pose_video._adapted_yaw / _adapted_body_yaw を quiet=True の探針で
    呼んで文面を骨格の宣言と一致させる (体ヨー追従はエンジン既定=off)。
    plan="flying" は歩行文面をホバリングへ差し替える (骨格ノブは
    _wp_plan_env_set が対で担当)。"""
    from PIL import Image
    pv = eng["pose_video"]
    cw = eng["canvas_walk"]
    cv = eng["compass_vace"]
    if plan == "ai":
        prompt = _WP_AI_PROMPT + _WP_AI_CELL_LOCK
    elif plan == "flying":
        prompt = _WP_HOVER_PROMPT
    elif plan in _WP_NAT_PLANS:
        prompt = _WP_PLAN_PROMPTS[plan] + _WP_CELL_LOCK
    else:
        prompt = cw.CANVAS_PROMPT
    if plan == "biped":
        prompt += cv.NO_WIND
        if keep_posture:
            # biped_legs (体格メニュー「二足歩行(脚のみ固定)」): 上半身の
            # 姿勢権威は参照立ち絵。歩様は参照ポーズが示すリズムに委ねる。
            # ★四肢の数を正宣言 (2026-07-21実走: 参照の「上げた脚」が
            # 第3の脚として残存し骨格の2本と共存した — guidance=1.0で
            # ネガ無効のため「参照の脚=動いている2本のうちの1本」と
            # 正面から同一化する)
            prompt += (" The torso, head and arms of every figure keep "
                       "exactly the posture shown in the reference art "
                       "through the whole cycle -- only the legs move, "
                       "cycling in the rhythm the reference pose implies "
                       "(a walk, or a fast run if the pose is a running "
                       "stance). Every figure has exactly TWO legs and "
                       "TWO arms at every moment: the lifted or bent leg "
                       "seen in the reference art is simply one of these "
                       "two legs in mid-stride, so it keeps cycling as a "
                       "leg -- it never remains behind as a separate "
                       "dark shape, and no limb ever splits or "
                       "duplicates.")
        if gait_run:
            # 実験h (2026-07-21「走る忍者」): 依頼が走りの動き。guidance=1.0
            # でネガ無効のため正宣言のみ。上半身は参照姿勢の維持を明言する
            # (legs_onlyで骨格の上半身権威を外した分の言い分)
            prompt += (" LOCOMOTION: every figure is RUNNING in place, "
                       "not walking -- a fast sprint cycle: knees driving "
                       "high, rear foot kicking up, and a readable forward "
                       "lean. ")
            if keep_posture:
                prompt += ("The torso, head and arms keep exactly the "
                           "posture shown in the reference art through the "
                           "whole cycle; only the two legs cycle rapidly.")
            else:
                prompt += ("The two bent arms pump in opposition to the "
                           "two legs, and the whole body clearly follows "
                           "the running rhythm.")
    else:
        # 歩かない体格では「歩行由来の揺れ」文言が空転する (文意の整合)。
        # あわせて発明抑制節 (_WP_NO_PROPS) を宣言する
        prompt += cv.NO_WIND.replace("the walking motion itself",
                                     "the character's own movement")
        prompt += _WP_NO_PROPS
        if crawl_bone:
            # quadruped_bone (四つん這い骨格): 手の二重化対策の正宣言
            # (2026-07-21実走: 骨格の手と参照の接地した手の位置ズレで
            # 前に出した手が二重になった)
            prompt += (" The creature has exactly TWO hands and TWO "
                       "knees on the ground: the planted hands in the "
                       "reference art are the same two hands seen "
                       "mid-crawl, reaching forward one at a time -- "
                       "no hand or leg ever splits into two.")
    idle_n, cyc, period, tail = pv.walk_layout(nf)
    try:
        fr = refs.get("front")
        if fr is not None and pv.skirt_hem_y(Image.open(fr),
                                             240, 432) is not None:
            prompt += (" The character wears a long skirt that stays "
                       "completely opaque: the legs never show through the "
                       "fabric, no slit ever opens in it, and only the feet "
                       "appear below the hem while walking.")
    except Exception:                     # noqa: BLE001
        pass
    if tail and plan != "ai":
        if plan == "flying":
            prompt += (" At the very end of the video every figure stops "
                       "bobbing and hovers perfectly still at the same "
                       "fixed height and pose as the first frames.")
        elif plan in _WP_NAT_PLANS:
            prompt += (" At the very end of the video every figure stops "
                       "moving and rests perfectly still in exactly the "
                       "same pose as the first frames.")
        else:
            prompt += (" At the very end of the video every figure stops "
                       "walking and stands perfectly still in the same "
                       "relaxed upright standing pose as the first frames, "
                       "arms hanging naturally at the sides.")
    ff_diag = db_diag = False
    try:
        fig = pv.Figure(head_frac=pv.head_frac_for_leg_scale(
            pv._leg_scale_env()))
        fim = None
        if refs.get("front") is not None:
            fim = Image.open(refs["front"])
            fig = pv._fit_figure_to_char(fig, fim)
        present = [x for x in layout[2] if x]
        for d in ("front_left", "front_right"):
            if d not in refs or d not in present:
                continue
            rim = Image.open(refs[d])
            # ★骨格と同じ判定を読むこと (2026-07-19)。ここが "auto" 固定
            # だと、SM_POSE_FACE_FRONT=off にしたとき骨格は「顔を実測どおり
            # 傾ける」に変わるのに文面だけ「顔はカメラを向く」と言い続け、
            # 綱引きがかえって悪化する。骨格側は
            # build_canvas_pose_frames が引数無し=環境変数読みなので、
            # 探針も _face_front_mode() を通す。
            if pv._adapted_yaw(d, rim, fig, front_ref=fim,
                               face_front=pv._face_front_mode(),
                               quiet=True) == 0.0:
                ff_diag = True
            if pv._adapted_body_yaw(d, rim, fig, front_ref=fim,
                                    mode="off", quiet=True) \
                    != float(pv.DIR_YAW[d]):
                db_diag = True
    except Exception:                     # noqa: BLE001
        ff_diag = db_diag = False        # 探針失敗時は従来文面
    return prompt + cv._direction_text(layout, face_front_diag=ff_diag,
                                       body_follow_diag=db_diag)


def _wp_wait(j: dict, sub: str, lo: float, hi: float) -> dict:
    """内部ジョブの完了待ち: ログ転写・進捗写像・キャンセル伝播。"""
    seen = 0
    while True:
        sj = JOBS.get(sub)
        if sj is None:
            raise RuntimeError(f"内部ジョブが見つかりません: {sub}")
        for line in (sj.get("log") or [])[seen:]:
            j["log"].append(f"  ┃ {line}")
        seen = len(sj.get("log") or [])
        try:
            p = float(sj.get("progress") or 0.0)
        except (TypeError, ValueError):
            p = 0.0
        j["progress"] = round(lo + (hi - lo) * max(0.0, min(1.0, p)), 4)
        j["_beat"] = time.time()
        st = sj.get("status")
        if st == "done":
            return sj
        if st in ("error", "cancelled"):
            raise RuntimeError(
                f"内部ジョブ({sj.get('model')}){st}: "
                f"{str(sj.get('detail') or '')[:300]}")
        if j.get("_cancel"):
            sj["_cancel"] = True
            raise JobCancelled()
        time.sleep(2)


def _wp_split(eng: dict, ffmpeg: str, cvid: Path, layout, refs: dict,
              char: str, out: Path, idle_n: int, gait_end, log,
              canvas_w: int = None, canvas_h: int = None) -> None:
    """キャンバスmp4を方向別セルへ。color_anchor があれば
    分割+カラーアンカー+拡大の単一パス、失敗時は素のcrop分割
    (canvas_walk.split_canvas_video の簡易移植) へ後退。

    ★セル寸法は必ず実キャンバス寸法から割ること (2026-07-20実障害):
    旧実装はグローバル定数 WALKPACK_W/H (480x864=半球前提) を直書きして
    おり、コンパス3x3 (720x1296) では160x288の窓で左上領域だけを切る
    壊れた分割になった。dir mp4は「存在するが中身が誤領域」で、キーイング
    空→シート組立が無言スキップ→空manifest→「DLできない」(バイクr4/r5・
    ファッティーナの実障害)。"""
    cv = eng["compass_vace"]
    idx_of = eng["canvas_walk"].IDX
    cols, rows = layout[0], layout[1]
    cw = (canvas_w or WALKPACK_W) // cols
    ch = (canvas_h or WALKPACK_H) // rows
    jobs = []
    for i, d in enumerate(layout[2]):
        if d is None:
            continue
        jobs.append({"crop": ((i % cols) * cw, (i // cols) * ch, cw, ch),
                     "ref": refs[d],
                     "dest": out / f"{char}_{idx_of[d]:02d}_{d}_walkT.mp4",
                     "size": cv._auto_size(refs[d])})
    try:
        eng["color_anchor"].split_anchor_scale(
            ffmpeg, cvid, jobs, fps=WALKPACK_FPS,
            idle_n=idle_n, gait_end=gait_end)
        log(f"  分割+カラーアンカー+拡大を単一パスで実行 ({len(jobs)}セル)")
        return
    except Exception as e:                # noqa: BLE001
        log(f"  ⚠ カラーアンカー分割に失敗 — 素の分割へ後退: {str(e)[:200]}")
    for job in jobs:
        x, y, w, h = job["crop"]
        r = subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(cvid),
             "-filter:v", f"crop={w}:{h}:{x}:{y}", "-an",
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             str(job["dest"])],
            capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg分割失敗: {r.stderr[-300:]}")


def _wp_drop_stray(im, min_keep_ratio: float = 0.05):
    """セル境界からはみ込んだ隣セルの欠片 (孤立成分) を透明化する。

    コンパス/半球キャンバスはセルが隣接しており、歩行で腕や足がセル境界を
    またぐ位相では、隣セルの切り出しに手先の欠片が写り込む (2026-07-20
    神爺さん front歩行1コマ目の実障害、ユーザー指摘「隣の見切れてるの
    気になります」)。キーイング後の不透過画素を連結成分に分け、
    「最大成分 (=キャラ本体) に非接続 かつ 縁から1割以内の帯にある かつ
    面積が本体の5%以下」の成分だけを消す。判定を接触でなく帯にするのは、
    組立の下端中央アンカーが内容を数px水平シフトさせ、境界で切れた欠片が
    縁から浮くため (実測: 神爺さんの欠片はx=2..7で縁非接触)。本体は縁に
    近くても最大成分なので消えない。帯の外の孤立成分 (装飾等) は温存。"""
    import numpy as np
    from PIL import Image
    arr = np.array(im)
    if arr.ndim != 3 or arr.shape[2] != 4:
        return im
    mask = arr[:, :, 3] > 16
    if not mask.any():
        return im
    H, W = mask.shape
    comps = []                      # (面積, 縁接触, 成分マスク)
    work = mask.copy()
    while work.any():
        ys, xs = np.nonzero(work)
        comp = np.zeros((H, W), dtype=bool)
        comp[ys[0], xs[0]] = True
        n = 1
        while True:                 # 4近傍の膨張をマスク内で飽和するまで
            grow = comp.copy()
            grow[1:, :] |= comp[:-1, :]
            grow[:-1, :] |= comp[1:, :]
            grow[:, 1:] |= comp[:, :-1]
            grow[:, :-1] |= comp[:, 1:]
            grow &= mask
            m = int(grow.sum())
            if m == n:
                break
            comp, n = grow, m
        cys, cxs = np.nonzero(comp)
        mx, my = max(2, W // 10), max(2, H // 10)
        near = bool(cxs.min() < mx or cxs.max() >= W - mx
                    or cys.min() < my or cys.max() >= H - my)
        comps.append((n, near, comp))
        work &= ~comp
    if len(comps) <= 1:
        return im
    main_area = max(c[0] for c in comps)
    thr = max(48.0, min_keep_ratio * main_area)
    dropped = False
    for area, near, comp in comps:
        if area != main_area and near and area <= thr:
            arr[:, :, 3][comp] = 0
            dropped = True
    return Image.fromarray(arr) if dropped else im


def _wp_key(im, thr: int = 70):
    """マゼンタ背景 -> 透過 (build_T_sheet の素朴キー版: min(R,B)-G>=thr)。
    連結成分キーイング (bg_magenta_mask) は使わない簡易版。
    キー後に隣セルの見切れ欠片を落とす (_wp_drop_stray)。"""
    import numpy as np
    from PIL import Image
    a = np.asarray(im.convert("RGB"), dtype=np.int16)
    bg = (np.minimum(a[..., 0], a[..., 2]) - a[..., 1]) >= thr
    alpha = np.where(bg, 0, 255).astype(np.uint8)
    out = im.convert("RGBA")
    out.putalpha(Image.fromarray(alpha, "L"))
    return _wp_drop_stray(out)


def _wp_dir_mp4s(out: Path) -> tuple:
    """out/ の方向別mp4を発見して (char名, 方向->Path) を返す。"""
    char = None
    mp4s: dict = {}
    for p in sorted(out.glob("*_walkT.mp4")):
        m = _WP_MP4_RE.match(p.name)
        if not m or m.group("dir") not in _WP_DIRS:
            continue
        mp4s[m.group("dir")] = p
        char = char or m.group("char")
    return (char or "char"), mp4s


# 顔が写る方向 (横顔は顔面積が小さく、後ろ系は顔が無いのが正しいので対象外。
# inspect_walk_mp4.FACE_DIRS と同じ範囲)
_WP_FACE_DIRS = ("front", "front_left", "front_right")
# 前向き系のうち最良に対してこの比を下回る方向は「顔が隠れている」と見なす。
# 実測: 正常な回は 0.93、髪が顔を覆った回は 0.57 (どちらもロップ)。
_WP_FACE_MIN_RATIO = 0.75
# 作り直しのシード。1レイアウト=1回の生成=1個のノイズで、どのセルが崩れるかは
# そのノイズ実現で決まる。既定42と別の実現を引き直すのが目的なので値自体に
# 意味は無い (再現性のため固定値にしてある)。
WP_FACE_RETRY_SEED = 43


def _wp_face_retry_on() -> bool:
    """顔ゲートNG時の作り直しをするか (SM_WP_FACE_RETRY、既定on)。

    offにすると検査とログだけ残して作り直さない (GPU時間を使いたくない
    ときや、ゲート自体の挙動を見たいときの逃げ道)。"""
    return os.environ.get("SM_WP_FACE_RETRY", "on").strip().lower() \
        not in ("off", "0", "false", "no")


def _wp_face_score(eng: dict, im) -> float | None:
    """キー済みRGBAセルの顔スコア (頭部帯の色多様性)。

    顔が見えている頭 = 髪+肌+目で色が多様、髪に覆われた頭/後頭部は
    髪一色に寄って低い。指標は inspect_walk_mp4 の実装をそのまま借りる
    (二重実装するとゲートと本番で判定がずれるため)。"""
    try:
        iw = eng["inspect_walk_mp4"]
        return iw.head_diversity(iw.head_band(im))
    except Exception:                     # noqa: BLE001
        return None


def _wp_collect_cells(eng: dict, ffmpeg: str, out: Path,
                      nf: int = WALKPACK_NF) -> tuple:
    """方向別mp4から idle+walk1..5 のキー済みRGBAセルを取る。

    コマ位置は walk_layout の決定的な配分から選ぶ (build_T_sheet の
    コマ探索QCの簡易代替): idle=末尾静止 (nf-2)、walk1..5=最初の歩行
    1周期を等分。戻り値 (char, {dir: [idle, w1..w5] (フル解像度RGBA)})。"""
    from PIL import Image
    pv = eng["pose_video"]
    idle_n, cyc, period, tail = pv.walk_layout(nf)
    idle_idx = (nf - 2) if tail else 0
    walk_idx = [idle_n + int(round(period * k / 5.0)) for k in range(5)]
    order = [idle_idx] + walk_idx
    char, mp4s = _wp_dir_mp4s(out)
    missing = [d for d in _WP_DIRS if d not in mp4s]
    if missing:
        raise RuntimeError(f"方向別mp4が不足しています: {missing}")
    cells: dict = {}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for d in _WP_DIRS:
            sub = tmp / d
            sub.mkdir()
            r = subprocess.run(
                [ffmpeg, "-y", "-loglevel", "error", "-i", str(mp4s[d]),
                 str(sub / "f_%05d.png")],
                capture_output=True, text=True, timeout=600)
            if r.returncode != 0:
                raise RuntimeError(f"フレーム抽出失敗[{d}]: "
                                   f"{r.stderr[-200:]}")
            avail = sorted(sub.glob("f_*.png"))
            if not avail:
                raise RuntimeError(f"フレームが空です: {mp4s[d].name}")
            got = []
            for i in order:
                p = sub / f"f_{i + 1:05d}.png"
                im = Image.open(p if p.is_file() else avail[-1])
                got.append(_wp_key(im.convert("RGB")))
            cells[d] = got
    return char, cells

# ★コマ探索QC (まばたきコマを避けて隣を拾う) は入れていない。2026-07-19に
# 実装しかけて実測で取り下げた:
#   ・顔ゲートの指標 (head_diversity=頭部帯の色多様性) は、髪が顔を覆う
#     ような大きな変化は捉えるが、目の開閉は拾えない。実測でまばたきコマ
#     26.7 > 次コマ 26.5 と逆転していて、この指標での探索は無意味。
#   ・常に窓内最良を取る素朴な実装は、正常な回でも全スロットを一律にずらす
#     (実測: 5スロット全部が+2コマ移動) ので歩容が変わる副作用がある。
#   ・目の暗画素比なら まばたきを部分的に捉えるが、front_right 32.8 に対し
#     front_left 37.6 (通常40前後) と分離が弱く、確実に拾えない。
# まばたき対策をやるなら目専用の判定 (_detect_eye_pair の開閉版) が要る。
# 中途半端な指標で歩容をずらすのは、直す量より壊す量が大きい。


def _wp_face_gate(eng: dict, ffmpeg: str, out: Path, log) -> list:
    """前向き系の方向で「顔が髪に覆われている/後頭部化している」を検出する。

    絶対値のしきい値はキャラの画風 (髪色の単調さ) に強く依存するので使わず、
    同じキャラの前向き系どうしを比べる。顔が出ている方向は互いに近い値に
    なり、崩れた方向だけが後頭部並みまで落ちる — 実測 (ロップ):
      正常な回   front 29.4 / front_left 28.9 / front_right 27.2 → 比0.93
      崩れた回   front 29.0 / front_left 26.5 / front_right 16.4 → 比0.57
                 (このとき back系が 14.7〜16.3 = front_right は後頭部並み)
    戻り値: (崩れていると判定した方向のリスト, 方向->スコア)。

    ★半球ごとに呼べるよう、_wp_collect_cells は使わず対象方向のmp4だけを
    自前で読む (F4の直後はB4のmp4がまだ無く、8方向前提の収集は落ちる)。"""
    from PIL import Image
    _, mp4s = _wp_dir_mp4s(out)
    targets = [d for d in _WP_FACE_DIRS if d in mp4s]
    scores: dict = {}
    if len(targets) < 2:
        return [], scores
    idle_n, cyc, period, tail = eng["pose_video"].walk_layout(WALKPACK_NF)
    want = [idle_n + int(round(period * k / 5.0)) for k in range(5)]
    with tempfile.TemporaryDirectory() as td:
        for d in targets:
            sub = Path(td) / d
            sub.mkdir()
            r = subprocess.run(
                [ffmpeg, "-y", "-loglevel", "error", "-i", str(mp4s[d]),
                 str(sub / "f_%05d.png")],
                capture_output=True, text=True, timeout=600)
            if r.returncode != 0:
                continue
            vals = []
            for i in want:
                p = sub / f"f_{i + 1:05d}.png"
                if not p.is_file():
                    continue
                try:
                    s = _wp_face_score(
                        eng, _wp_key(Image.open(p).convert("RGB")))
                except Exception:         # noqa: BLE001
                    continue
                if s is not None:
                    vals.append(s)
            if vals:
                scores[d] = sum(vals) / len(vals)
    if len(scores) < 2:
        return [], scores
    best = max(scores.values())
    bad = [d for d, v in scores.items() if v < best * _WP_FACE_MIN_RATIO]
    detail = " / ".join(f"{d}={v:.1f}" for d, v in sorted(scores.items()))
    if bad:
        log(f"  ★顔ゲート: 顔が隠れている方向 {bad} ({detail}、"
            f"最良比 {min(scores.values()) / best:.2f} "
            f"< {_WP_FACE_MIN_RATIO})")
    else:
        log(f"  顔ゲート OK ({detail})")
    return bad, scores


def _wp_write_preview(out: Path, tcells: dict, cell_w: int, cell_h: int,
                      scale: int = 2) -> None:
    """make_walk_preview 相当の 2x4 全方向ループwebp (walk1..5)。"""
    from PIL import Image
    cw, ch = cell_w * scale, cell_h * scale
    frames = []
    for k in range(1, 6):
        cv = Image.new("RGBA", (cw * 2, ch * 4), (0, 0, 0, 0))
        for d, (col, row) in _WP_PREVIEW_GRID.items():
            cl = tcells.get(d)
            if not cl or len(cl) <= k:
                continue
            cv.alpha_composite(cl[k].resize((cw, ch), Image.NEAREST),
                               (col * cw, row * ch))
        frames.append(cv)
    frames[0].save(out / "preview.webp", save_all=True,
                   append_images=frames[1:], duration=140, loop=0,
                   disposal=2)


# シートのコマ名 -> walkpackが持つ方向別コマの添字 (idle, walk1..walk5)。
# walkA/walkB = 歩幅が反対の2コマ (母艦の walkAB.json と同じ walk1/walk3)。
_WP_FRAME_IDX = {"idle": 0, "walk1": 1, "walk2": 2, "walk3": 3,
                 "walk4": 4, "walk5": 5, "walkA": 1, "walkB": 3}


def _wp_sheet_layout(pack: Path) -> tuple:
    """(列数, 行数, [(col,row,dir,コマ添字)]) を返す。

    パックに template.json (母艦が依頼のシート形式を同梱) があればそれに
    従い、無ければ従来のT規格 (_WP_SHEET_PLACE) で組む。"""
    tp = pack / "template.json"
    if tp.is_file():
        try:
            cells = json.loads(tp.read_text(encoding="utf-8")).get("cells")
            places = []
            for c in (cells or []):
                d, fr = str(c.get("dir")), str(c.get("frame"))
                if d in _WP_DIRS and fr in _WP_FRAME_IDX:
                    places.append((int(c["col"]), int(c["row"]), d,
                                   _WP_FRAME_IDX[fr]))
            if places:
                cols = max(p[0] for p in places) + 1
                rows = max(p[1] for p in places) + 1
                return cols, rows, places
        except Exception:                     # noqa: BLE001
            pass                              # 壊れた定義はT規格へ退避
    places = [(block + k, row, d, k)
              for d, (row, block) in _WP_SHEET_PLACE.items()
              for k in range(6)]
    return 12, 4, places


def _wp_assemble(eng: dict, ffmpeg: str, out: Path, meta: dict, log) -> list:
    """簡易シート ({char}T/LT.png) + preview.webp を out/ に組む。

    ★v1簡易版 (省略事項): build_T_sheet_from_mp4 のフルビルダー
    (向き検査つきコマ探索・方向別スケール整合・idle中心整列・各種ゲート)
    は移植していない。スケールは全コマ共通のグローバル縮尺 (セルに収まる
    最大)、アンカーは下端中央固定、キーイングは素朴なマゼンタしきい値。"""
    from PIL import Image
    char, cells = _wp_collect_cells(eng, ffmpeg, out)
    cell_w = max(16, int(meta.get("cell_w") or 64))
    cell_h = max(16, int(meta.get("cell_h") or 128))
    ltf = 5                                  # LT = T の5倍 (T規格と同じ比)
    ltw, lth = cell_w * ltf, cell_h * ltf
    boxes = []
    for d in _WP_DIRS:
        for im in cells[d]:
            bb = im.getchannel("A").getbbox()
            if bb is None:
                raise RuntimeError(
                    f"キーイング結果が空です ({d}) — マゼンタ背景でない?")
            boxes.append(bb)
    scale = min(min((ltw * 0.94) / max(1, bb[2] - bb[0]),
                    (lth * 0.96) / max(1, bb[3] - bb[1])) for bb in boxes)
    # 方向別コマを正規化 (LTサイズ=シート合成用 / セルサイズ=プレビュー用)
    ltcells: dict = {}
    tcells: dict = {}
    for d in _WP_DIRS:
        ll, tl = [], []
        for im in cells[d]:
            bb = im.getchannel("A").getbbox()
            crop = im.crop(bb)
            nw = max(1, round(crop.width * scale))
            nh = max(1, round(crop.height * scale))
            crop = crop.resize((nw, nh), Image.LANCZOS)
            cell = Image.new("RGBA", (ltw, lth), (0, 0, 0, 0))
            cell.alpha_composite(
                crop, ((ltw - nw) // 2,
                       max(0, lth - round(lth * 0.02) - nh)))
            ll.append(cell)
            tl.append(cell.resize((cell_w, cell_h), Image.LANCZOS))
        ltcells[d] = ll
        tcells[d] = tl
    # 依頼のシート形式で合成 (T規格/ツクール/ウディタ)。定義はパック同梱の
    # template.json = 母艦の templates/ が唯一の正
    scols, srows, places = _wp_sheet_layout(out.parent)
    sheet = Image.new("RGBA", (ltw * scols, lth * srows), (0, 0, 0, 0))
    for col, row, d, fi in places:
        cl = ltcells.get(d) or []
        if fi < len(cl):
            sheet.alpha_composite(cl[fi], (col * ltw, row * lth))
    sheet.save(out / f"{char}LT.png")
    sheet.resize((cell_w * scols, cell_h * srows),
                 Image.LANCZOS).save(out / f"{char}T.png")
    log(f"シート形式: {scols}列x{srows}行 ({len(places)}コマ)")
    _wp_write_preview(out, tcells, cell_w, cell_h)
    # 方向別の「背景抜き済み実スプライト」歩行webp + 静止ポスターを出力。
    # ギャラリーが本家GUIと同じく透明スプライトでターンテーブルを回せる
    # (mp4は背景付きの生映像なので見た目が違う、というユーザー指摘の根治)。
    ps = 2                                    # 見やすさ用に2倍 (透明のまま)
    for d in _WP_DIRS:
        cl = tcells.get(d) or []
        walk = [c.resize((cell_w * ps, cell_h * ps), Image.NEAREST)
                for c in cl[1:6]]            # walk1..5 = 歩行コマ
        if walk:
            walk[0].save(out / f"{char}_{d}_walk.webp", save_all=True,
                         append_images=walk[1:], duration=140, loop=0,
                         disposal=2)
    fr = tcells.get("front") or []
    if fr:                                     # front idle = 透明ポスター
        fr[0].resize((cell_w * ps, cell_h * ps),
                     Image.NEAREST).save(out / f"{char}_poster.png")
    log(f"シート/プレビュー: {char}T.png / {char}LT.png / preview.webp "
        "+ 方向別歩行webp8 + poster (簡易版 — T規格フルQCなし)")
    return [f"{char}T.png", f"{char}LT.png", "preview.webp"]


def _wp_regen_preview(pid: str) -> list:
    """preview.webp (と簡易シート) を out/ の方向別mp4から再生成 (同期)。"""
    eng = _engine()
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpegが見つかりません")
    pack = packs_root() / pid
    out = pack / "out"
    if not out.is_dir():
        raise RuntimeError("生成結果がまだありません — 先に歩行生成を"
                           "実行してください")
    return _wp_assemble(eng, ffmpeg, out, _pack_meta(pack),
                        lambda m: _wp_print(f"[walkpack:{pid}] {m}"))


def _wp_pixelize(pid: str, colors: int = 24, dither: float = 1.0) -> list:
    """簡易シート ({char}LT.png) をドット絵化 (pixelize_sheet の
    dither経路: Floyd-Steinberg減色+黒系1ドット縁取り)。同期・数秒。

    dither = 誤差拡散のスケール (GUIの「ディザの強さ」と同じ):
    1.0=フル(市松のザラつき) / 0.6=中 / 0.3=弱 / 0=なし(ベタ塗り)。"""
    from PIL import Image
    eng = _engine()
    pz = eng["pixelize_sheet"]
    pack = packs_root() / pid
    out = pack / "out"
    lt = next(iter(sorted(out.glob("*LT.png"))), None) \
        if out.is_dir() else None
    if lt is None:
        raise RuntimeError("シート(*LT.png)がまだありません — 先に歩行生成"
                           "を実行してください")
    char = lt.name[: -len("LT.png")]
    cell_h = max(16, int(_pack_meta(pack).get("cell_h") or 128))
    src = Image.open(lt)
    factor = max(1, src.height // (cell_h * 4))
    colors = max(8, min(64, int(colors)))
    dither = max(0.0, min(1.0, float(dither)))
    idx, mask, pal = pz.pixelize_dither(src, colors, factor, dither=dither)
    pz.outline_pass(idx, mask, pal)
    img = pz.to_image(idx, mask, pal)
    img.save(out / f"{char}T_pixel.png")
    img.resize((img.width * 2, img.height * 2),
               Image.NEAREST).save(out / f"{char}T_pixel@2x.png")
    return [f"{char}T_pixel.png", f"{char}T_pixel@2x.png"]


def _wp_kind(name: str) -> str:
    if name.endswith("_poster.png"):     # 静止ポスター (透明front idle)
        return "poster"
    if name.endswith("_walk.webp"):      # 方向別 透明歩行アニメ
        return "dirwalk"
    if name == "preview.webp":
        return "preview"
    if "_pixel" in name:
        return "pixel"
    if name.endswith("T.png"):           # {char}T.png / {char}LT.png
        return "sheet"
    if name.endswith(".mp4"):
        return "mp4"
    return "other"


def _walkpack_run(j: dict, pid: str, meta: dict, log) -> None:
    """walk_pack 本体: 半球2キャンバス x (VACE 4step -> AniSora latent再加工)
    -> セル分割 -> 簡易シート/プレビュー。compass_vace._run_layout の
    latent_refine 分岐のサーバ内部版 (submit_job で自サーバの実ジョブを
    投入し、このオーケストレーションスレッドが完了をポーリングする)。"""
    _wp_apply_pose_defaults()     # 姿勢の既定値 (腕/脚の振り・上下動・交差)
    plan_raw = str((meta or {}).get("body_plan") or "biped").strip() or "biped"
    # 体格メニュー8種 (2026-07-21ユーザー要望): 亜種はベース体格へ正規化し、
    # 差分は文面/骨格モードで表現する。
    #   biped_legs     = 二足歩行(脚のみ固定): 骨格=脚だけ+上半身は参照姿勢
    #                    (走る忍者実障害で実証した実験hの正式化)
    #   quadruped_bone = 四足歩行(骨格固定): 人型骨格を四つん這いに変形
    plan = {"biped_legs": "biped",
            "quadruped_bone": "quadruped",
            "quadruped_ai": "quadruped", "flying_ai": "flying",
            "amorphous_ai": "amorphous",
            "serpentine_ai": "serpentine",
            # 動きの型4択 (0.11.0、2026-07-21ユーザー裁定「体型という
            # 名前は撤退・AI経路はAniSora頭部固定インペイントに一本化」):
            # ai=AI生成は専用のAniSora空間インペイントとして
            # 生値を保つ。otherに落とすと「姿勢とシルエットを変えるな」
            # という旧制約がCodexの歩行文と衝突する。stretch_v/stretch_h/
            # move_vは素通し=手続き式で分岐 (通常は母艦完結でここに来ない)
            }.get(plan_raw, plan_raw)
    if plan_raw == "ai":
        _wp_print("[walkpack] 動きの型=AI生成: AniSora頭部固定インペイント"
                  " (VACE不使用)")
    elif plan_raw == "biped_legs":
        _wp_print("[walkpack] body_plan=biped_legs: 骨格=脚のみ "
                  "(上半身の姿勢は参照立ち絵に委ねる)")
    elif plan_raw == "quadruped_bone":
        _wp_print("[walkpack] body_plan=quadruped_bone: 四つん這い骨格")
    if plan == "flying":
        _wp_print("[walkpack] body_plan=flying: ホバリング文面+骨格ノブで生成")
    elif plan in _WP_NAT_PLANS:
        # (2026-07-20 赤さん実障害「ハイハイを無理やり二足歩行に」): 非二足は
        # 二足マネキン骨格が姿勢を直立へ引っ張るため骨格制御を出さず、
        # キー錨 (立ち絵由来のstage1固定) + 体格別文面だけで動きを誘導する
        _wp_print(f"[walkpack] body_plan={plan}: 自然移動ルート "
                  "(二足骨格なし+キー錨+体格別文面)")
    eng = _engine()
    pv = eng["pose_video"]
    cw = eng["canvas_walk"]
    cv = eng["compass_vace"]
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpegが見つかりません")
    pack = packs_root() / pid
    char, refs = _pack_refs_dir(pack)
    missing = [d for d in _WP_DIRS if d not in refs]
    if missing:
        raise RuntimeError(f"パックに8方向PNGが不足: {missing}")
    _motion_kind = (_wp_motion_control(pack, plan_raw)
                    if plan_raw == "ai" else
                    _wp_infer_motion_control("", plan_raw))
    _motion_gait = _wp_motion_gait(pack) if plan_raw == "ai" else "walk"
    _limb_mode = _wp_limb_mode(pack) if plan_raw == "ai" else "full"
    if plan_raw == "ai":
        _wp_print(f"[walkpack] AI内部制御={_motion_kind} "
                  f"gait={_motion_gait} limbs={_limb_mode} "
                  f"({'人型OpenPose' if _motion_kind == 'biped' else '原画変形列'})")
    char = str(meta.get("char_id") or char)
    out = pack / "out"
    if out.is_dir():
        shutil.rmtree(out)               # 再生成は一式作り直し
    out.mkdir(parents=True)
    nf, w, h = WALKPACK_NF, WALKPACK_W, WALKPACK_H
    idle_n, cyc, period, tail = pv.walk_layout(nf)
    gait_end = idle_n + int(round(cyc * period))
    _exp = j.get("_wp_exp") or {}
    if plan_raw.endswith("_ai") or plan_raw == "ai":
        # AI経路の一本化 (0.11.0ユーザー裁定「動画AI処理は共通して
        # AniSoraのみの頭部固定インペイント方式」): 動きの型4択の
        # ai=AI生成 (既定) と、旧12択の *_ai=AI通過 (レガシー) の両方が
        # ここへ来る。インペイント式=手続き土台+空間latent固定 (実験r2の
        # 正式化)。fracは旧体格別値を温存 (face既定では未使用)
        _exp.setdefault("anisora_inpaint", {
            "quadruped_ai": 0.6, "amorphous_ai": 0.55,
            "serpentine_ai": 0.6, "flying_ai": 0.5}.get(plan_raw, 0.55))
    if plan_raw in ("stretch_v", "stretch_h", "move_v"):
        # 動きの型4択の加工系: 通常は母艦の手続きアニメで完結しGPUには
        # 来ないが、pack_ready戻し (歩行だけ再生成する管理手順) で来た
        # ときも同じ画素完全の手続きで作る (VACE/骨格には流さない)
        _exp.setdefault("procedural", True)
    if plan_raw == "biped_legs":
        # 体格メニューの正式経路: 骨格=脚のみ+姿勢維持文面+legs_mask
        # (実験h→r3の昇格 2026-07-21: 文面の正宣言だけでは参照の上げた脚が
        # 「第3の脚」として残存した — VACEマスク同居で上半身=実画素凍結・
        # 脚領域=骨格入り生成にすると構造的に排除される。忍者実走で確定)
        _exp.setdefault("legs_only", True)
        _exp.setdefault("keep_posture", True)
        _exp.setdefault("legs_mask", 0.55)
    if (plan in (("flying",) + _WP_NAT_PLANS) and not _exp
            and plan_raw != "quadruped_bone"):
        # ★飛行の既定=キー錨方式 (2026-07-19ユーザー発案→イーリス/ドラゴン
        # 2体のA/Bで昇格確定): 均等5キー+末尾静止をlatent固定し、中間は
        # σ0.9でAniSoraに生成させる。σ0.45の全体浅がけより羽ばたきが
        # 生き生きし、骨格に無い部位 (尻尾・翼) の「無条件区間の発明」
        # (浮いた尻尾の塊等) も固定間隔が短くなることで実測で消えた。
        # 自然移動体格 (quadruped等) も同方式: 骨格が無い分、立ち絵由来の
        # stage1キーが姿勢の錨になる (2026-07-20)。
        # APIの pin_conf/refine 明示指定はこの既定より優先される
        _keys = sorted({0, gait_end // 4, gait_end // 2,
                        gait_end * 3 // 4, gait_end}
                       | set(range(gait_end + 1, nf)))
        # σは環境変数 (=管理ノブ経由) を尊重する。従来の0.9ハードコードは
        # 管理ノブ"refine"がキー錨経路に一切届かない死にノブだった
        # (2026-07-20監査案F)。未設定時は0.9で完全同値
        _exp = {"pin_conf": ",".join(map(str, _keys)),
                "refine": float(os.environ.get(
                    "SM_VACE_LR_REFINE", "").strip() or 0.9)}
        _wp_print(f"[walkpack] {plan}: キー錨既定 "
                  f"(keys={_keys[:5]}+末尾静止, σ0.9)")
    _kn = j.get("_wp_knobs") or {}       # 管理GUIのノブ (受付台/adminで設定)
    if _kn:
        _wp_print(f"[walkpack] 管理ノブ適用: {_kn}")
    # 作品詳細へ残す「実際に使った値」。GCS workerが完成manifestへ写す。
    # 生の管理JSONではなく既定補完+クランプ後を記録し、後日の設定変更で
    # 過去作品の表示が変わらないようにする。
    if _exp.get("anisora_inpaint"):
        _settings_steps = _wp_knob_int(
            _kn, "refine_total", 8, 4, 12)
        _settings_release = min(_settings_steps, _wp_knob_int(
            _kn, "head_release_steps", 0, 0, 12))
        _settings_pose = _wp_knob_float(
            _kn, "pose_weight", 0.25, 0.05, 0.50)
        j["_generation_settings"] = {
            "engine": "anisora",
            "directions": len(_WP_DIRS),
            "steps": _settings_steps,
            "motion": _wp_knob_float(_kn, "motion", 3.0, 2.0, 4.0),
            "pose_weight": _settings_pose,
            "image_weight": 1.0 - _settings_pose,
            "motion_control": _motion_kind,
            "gait": _motion_gait,
            "limb_mode": _limb_mode,
            "side_control": ("art_puppet" if _motion_kind == "biped"
                             else "morphology_art_motion"),
            "side_control_weight": (max(0.50, _settings_pose)
                                    if _motion_kind == "biped"
                                    else _settings_pose),
            "head_release_steps": _settings_release,
            "inpaint": "frame0_fixed_dual_condition",
            "direction_chain": "adjacent_end_anchor",
            "transition_frames": 4,
            "pixel_paste": False,
            "server_version": __version__,
        }
    elif _exp.get("procedural"):
        j["_generation_settings"] = {
            "engine": "procedural", "directions": len(_WP_DIRS),
            "server_version": __version__}
    else:
        j["_generation_settings"] = {
            "engine": "legacy", "directions": len(_WP_DIRS),
            "server_version": __version__}
    pins = cv._lr_pin_frames(nf, str(
        _exp.get("pin_conf")
        or ("off" if _kn.get("latent_pin") is False else "on")))
    if _exp:
        _wp_print(f"[walkpack] 実験ノブ: {_exp} -> pins={pins}")
    offload = _wp_offload()
    if offload:
        log(f"VRAM<30GBのため両ステージを{offload} offload運転にします")
    # 進捗スパン: F4 (0.02-0.45) -> B4 (0.50-0.88) -> 組み立て (0.90-)
    spans = {"F4": (0.02, 0.24, 0.24, 0.45), "B4": (0.50, 0.70, 0.70, 0.88),
             "C9": (0.02, 0.45, 0.45, 0.88)}
    # 新本線aiは1方向=1キャンバス。8回の進捗を均等に割り当てる。
    for _di, _dd in enumerate(_WP_DIRS):
        _lo = 0.02 + 0.86 * _di / len(_WP_DIRS)
        _hi = 0.02 + 0.86 * (_di + 1) / len(_WP_DIRS)
        spans[f"D{_di + 1:02d}_{_dd}"] = (_lo, _lo, _lo, _hi)
    def _gen_hemisphere(tag: str, layout, seed: int = 42) -> None:
        """半球1つ分 (骨格グリッド→VACE→AniSora再加工→セル分割)。

        顔ゲートが落ちたときにシードだけ変えて作り直せるよう関数にしてある。
        書き出し先は out/ 固定なので、呼び直すと同じ名前を上書きする
        (退避は呼び出し側の責任)。"""
        if j.get("_cancel"):
            raise JobCancelled()
        s1lo, s1hi, s2lo, s2hi = spans[tag]
        _face_gap = None       # 実験g3: 間引きギャップ用の顔限定制御列
        j["detail"] = f"[{tag}] 骨格グリッド生成"
        log(f"[{tag}] 骨格グリッド生成 ({nf}f {w}x{h}, "
            f"直立{idle_n}+歩行{gait_end - idle_n + 1}+静止{tail})")
        # 顔エッジ固定 (2026-07-20昇格、ユーザー報告「OpenPoseが目の位置を
        # 外して顔を壊す事例が数件」): 顔の権威を骨格の顔点から頭部エッジ
        # (立ち絵実測キャニー=a-4実験部品) へ移す。二足=体のみ骨格+歩行窓に
        # 顔エッジ、flying/非二足=骨格なし+全域顔エッジ (flyingのボブは
        # エッジをsin同期シフト)。SM_WP_EDGE_FACE=off で旧動作。実験ノブ
        # (edge_idle等) 指定時は実験側に譲る
        # ★2026-07-20実走で既定offへ後退: 顔は直るが (神爺さん/ドラゴン
        # 両方で顔無傷・線写り込みなし)、①二足=顔点を消すと頭の位置権威が
        # 消えて猫背が復活 ②flying=ボブ骨格まで外すと動きの源が尽きて静止
        # (motion score 3.0問題と複合)。動き量の適正化と頭アンカーの再設計
        # とセットで再昇格する。SM_WP_EDGE_FACE=on で実験再開できる
        _edge_face = ((_exp.get("edge_face")
                       or os.environ.get("SM_WP_EDGE_FACE", "off")
                       .strip().lower() not in ("off", "0", "false", "no"))
                      and not (_exp.get("edge_idle") or _exp.get("edge_head")
                               or _exp.get("no_pose")))
        if _exp.get("region_mask"):
            # 実験r (2026-07-21ユーザー発案「動かしたい部分だけマスクして
            # 空の潜在で埋める」— 標準インペイント用法): video=参照キャンバス
            # の実画素 (完全な同一性)、キャラbbox下部 frac をセルごとに
            # 灰色=空の潜在にし、そこだけ生成させる。骨格・線画は出さない
            # (動きは文面とインペイント文脈に委ねる)
            import numpy as _npRM
            from PIL import Image as _ImgRM
            _frac = max(0.2, min(0.8, float(_exp["region_mask"])))
            _cv0 = cv.compose_reference(refs, w, h, layout).convert("RGB")
            _msk, _exp["_region_mask_b64"] = _wp_bottom_mask(
                _cv0, layout, w, h, _frac,
                face_boxes=_wp_face_boxes(pack), refs=refs)
            _arr2 = _npRM.array(_cv0)
            _arr2[_msk > 0] = 127
            _vf = _ImgRM.fromarray(_arr2)
            frames = [_vf] * nf
            try:
                _vf.save(out / f"control_{tag}_regionmask.png")
            except Exception:                 # noqa: BLE001
                pass
            log(f"[{tag}] 実験region_mask={_frac}: キャラ下部を空の潜在で"
                "生成 (上部=実画素保持)")
        elif plan == "biped" and _exp.get("key_interp"):
            # 実験e (2026-07-20ユーザー発案・VACE-Fun Extension/Loop同型):
            # 実画像 (参照キャンバス=立ち絵コラージュ) をアンカーフレーム
            # (直立窓・周期境界・末尾静止) に置き、間は灰色=空の潜在。
            # 見た目の権威を線でなく実画像そのもので与え、間の動きは
            # VACEの補間学習に任せる。マスクはアダプタ側で構築
            from PIL import Image as _ImgKI
            _cv0 = cv.compose_reference(refs, w, h, layout).convert("RGB")
            _gray = _ImgKI.new("RGB", (w, h), (127, 127, 127))
            _pe = int(_exp.get("key_interp_pose") or 0)
            # アンカー: 直立窓+末尾静止。歩行窓の中間実画像アンカーは
            # 骨格道しるべ無し (純補間) のときだけ (実走e1の教訓: 中間に
            # 直立を錨止めすると「ずっと直立」に補間されて歩かない —
            # 道しるべがあるなら中間は骨格に任せる)
            _anchors = set(range(idle_n)) | set(range(gait_end + 1, nf))
            if not _pe:
                _anchors.add(idle_n + int(round((gait_end - idle_n + 1) / 2)))
            _sk = (pv.build_canvas_pose_frames(refs, nf, w, h, layout)
                   if _pe > 0 else None)
            frames = []
            for k in range(nf):
                if k in _anchors:
                    frames.append(_cv0)
                elif (_sk is not None and idle_n <= k <= gait_end
                      and (k - idle_n) % _pe == 0):
                    # 疎な骨格の道しるべ (2026-07-20ユーザー発案
                    # 「10フレームに1回だけ実績のあるオープンポーズ」)
                    frames.append(_sk[k])
                else:
                    frames.append(_gray)
            _exp["_keep_frames"] = sorted(_anchors)
            try:
                frames[idle_n].save(out / f"control_{tag}_idle.png")
                frames[idle_n + max(1, (gait_end - idle_n) // 4)].save(
                    out / f"control_{tag}_move.png")
            except Exception:                 # noqa: BLE001
                pass
            _npose = (len([k for k in range(idle_n, gait_end + 1)
                           if _pe and (k - idle_n) % _pe == 0
                           and k not in _anchors]) if _pe else 0)
            log(f"[{tag}] 実験key_interp: 実画像アンカー{len(_anchors)}f"
                f"+骨格道しるべ{_npose}f+空{nf - len(_anchors) - _npose}f")
        elif plan == "biped" and _exp.get("scribble_mix"):
            # 実験d (2026-07-20ユーザー発案): 頭=立ち絵の線画 (ボブ追従)、
            # 体=白ストロークの棒人間。服のヒダを変形させないポーズ指示
            _krec = []
            pv.build_canvas_pose_frames(refs, nf, w, h, layout,
                                        kps_out=_krec)
            _lcv = _wp_edge_canvas(cv, refs, w, h, layout)
            if _lcv.size != (w, h):
                _lcv = _lcv.resize((w, h))
            frames = _wp_scribble_frames(_lcv, _krec, layout, w, h)
            try:
                frames[idle_n].save(out / f"control_{tag}_idle.png")
                frames[idle_n + max(1, (gait_end - idle_n) // 4)].save(
                    out / f"control_{tag}_move.png")
            except Exception:                 # noqa: BLE001
                pass
            log(f"[{tag}] 実験scribble_mix: 頭=線画+体=棒人間スクリブル")
        elif plan == "biped" and _exp.get("line_puppet"):
            # 実験c (2026-07-20ユーザー発案「骨に線画を貼り付けて動かす」):
            # 骨格キーポイントで線画をパーツ分割し、ボーン相似変換で
            # 「キャラ自身の線が歩く」制御ビデオを作る。骨の動きの正確さ×
            # 線画の同一性固定の合流。idle窓もパペット線画を出す (idle変換=
            # 恒等=立ち絵自身の線なのでfree_idleの矛盾問題は起きない)
            _krec = []
            pv.build_canvas_pose_frames(refs, nf, w, h, layout,
                                        kps_out=_krec)
            _lcv = _wp_edge_canvas(cv, refs, w, h, layout)
            if _lcv.size != (w, h):
                _lcv = _lcv.resize((w, h))
            frames = _wp_puppet_frames(_lcv, _krec, layout, w, h)
            try:
                frames[idle_n].save(out / f"control_{tag}_idle.png")
                frames[idle_n + max(1, (gait_end - idle_n) // 4)].save(
                    out / f"control_{tag}_move.png")
            except Exception:                 # noqa: BLE001
                pass
            log(f"[{tag}] 実験line_puppet: 線画パペット制御 (骨で線を駆動)")
        elif plan_raw == "ai":
            frames, _ai_ctl = _wp_ai_control_frames(
                cv, pv, refs, _motion_kind, nf, idle_n, gait_end,
                w, h, layout, gait=_motion_gait,
                limb_mode=_limb_mode)
            log(f"[{tag}] AI生成: 内部制御={_ai_ctl} (VACE不使用)")
        elif plan == "biped":
            _fp_save = os.environ.get("SM_POSE_FACE_POINTS")
            _bp_save = os.environ.get("SM_POSE_BODY_PARTS")
            if _edge_face:
                # noeyes=目とface68だけ消す (2026-07-20実走の教訓: 全消しは
                # 頭の位置権威まで消えて猫背復活。鼻・耳=頭アンカーは残し、
                # 壊し屋の目だけ排除して顔の細部はエッジに委ねる)
                os.environ["SM_POSE_FACE_POINTS"] = "noeyes"
            if _exp.get("legs_only"):
                # 実験h (2026-07-21ユーザー発案「ポーズ制御を脚だけに」):
                # 参照が走り姿勢の依頼で、直立歩行骨格の上半身が参照と
                # 全面矛盾して二重人格化した対策。脚(腰8-13)だけ誘導し、
                # 上半身の姿勢権威を参照立ち絵へ返す
                os.environ["SM_POSE_BODY_PARTS"] = "legs"
            try:
                frames = pv.build_canvas_pose_frames(refs, nf, w, h, layout)
            finally:
                if _edge_face:
                    if _fp_save is None:
                        os.environ.pop("SM_POSE_FACE_POINTS", None)
                    else:
                        os.environ["SM_POSE_FACE_POINTS"] = _fp_save
                if _exp.get("legs_only"):
                    if _bp_save is None:
                        os.environ.pop("SM_POSE_BODY_PARTS", None)
                    else:
                        os.environ["SM_POSE_BODY_PARTS"] = _bp_save
            if _edge_face:
                log(f"[{tag}] 骨格=目なし (鼻耳=頭アンカー維持・顔は頭部エッジで固定)")
            if _exp.get("legs_only"):
                log(f"[{tag}] 実験legs_only: 骨格=脚のみ "
                    "(上半身の姿勢は参照立ち絵に委ねる)")
        elif plan_raw == "quadruped_bone":
            # 体格メニュー「四足歩行(骨格固定)」(2026-07-21ユーザー裁定
            # 「人型骨格を四つん這いにして四足に対応」): 人型骨格をクロール
            # 姿勢 (胴前傾・手=前脚として接地・対角肢交互) で出す。
            # ハイハイ赤ちゃん等「人型が四つん這いになった」体格向け
            _gait_save = os.environ.get("SM_POSE_GAIT")
            os.environ["SM_POSE_GAIT"] = "crawl"
            try:
                frames = pv.build_canvas_pose_frames(refs, nf, w, h, layout)
            finally:
                if _gait_save is None:
                    os.environ.pop("SM_POSE_GAIT", None)
                else:
                    os.environ["SM_POSE_GAIT"] = _gait_save
            log(f"[{tag}] 四つん這い骨格 (crawl gait)")
        elif plan == "flying":
            # v0.10.6実証・キー錨と併用の実績経路: 直立+上下ボブ骨格。
            # ★エッジ実験on時もこの骨格は維持する (2026-07-20実走: 骨格まで
            # 外すと動きの源が尽きて静止化)。エッジは同じsin式でボブに同期
            # して上に重なるだけ
            frames = pv.build_canvas_pose_frames(refs, nf, w, h, layout)
            frames = _wp_flying_frames(frames, idle_n, gait_end, h)
            log(f"[{tag}] 飛行: 骨格を直立+上下ボブ列に差し替え")
        elif _exp.get("depth_move") or _exp.get("line_move"):
            # 実験b (2026-07-20ユーザー発案「Depthを計測で動かせば任意形状に
            # 対応できるのでは」+「ラインアートも」): 立ち絵実測の深度/線画を
            # 体格別の手続き運動 (ボブ・ロッキング・蛇行・拍動) で動かして
            # 制御に流す。形の権威と動きの源を同時に供給=発明抑制+静止化
            # 対策の両取りを狙う。VACEはマルチモーダル制御学習なので語彙内
            _mode = "depth" if _exp.get("depth_move") else "line"
            frames = _wp_moving_frames(cv, refs, plan, nf, idle_n,
                                       gait_end, w, h, layout, mode=_mode)
            try:                     # 何で誘導したか後から目視できるように
                frames[idle_n].save(out / f"control_{tag}_idle.png")
                frames[idle_n + max(1, (gait_end - idle_n) // 4)].save(
                    out / f"control_{tag}_move.png")
            except Exception:                 # noqa: BLE001
                pass
            log(f"[{tag}] 実験{_mode}_move: {plan}の{_mode}制御 "
                "(実測マップ+手続き運動)")
        else:
            # ★線画制御が既定 (2026-07-20夜、ユーザー目視判定で深度から反転:
            # 「深度だと細かいディテールが崩れるが、ラインアートだと完璧に
            # 維持」— スライム娘の前髪の房で実証)。深度ドームは体積しか
            # 語らず内部線は毎スパン再創作になるが、線画は内部エッジを毎
            # フレーム明示するので同一性が固定される。動きの振幅は深度が
            # 僅差で上だが、スプライトは同一性が王様。
            # SM_WP_NAT_CONTROL=depth/none、管理ノブnat_controlで切替可。
            _nat_ctl = os.environ.get(
                "SM_WP_NAT_CONTROL", "line").strip().lower()
            if plan in ("other", "ai"):
                # 2026-07-21ユーザー裁定 (体格メニュー8種): その他=
                # 「動画AIに動きを委ねる」— 手続き運動を出さず、
                # キー錨+文面のみで誘導する
                _nat_ctl = "none"
            if _nat_ctl in ("depth", "line"):
                frames = _wp_moving_frames(cv, refs, plan, nf, idle_n,
                                           gait_end, w, h, layout,
                                           mode=_nat_ctl)
                if _exp.get("face_line"):
                    # 実験g3: 間引きギャップ用の顔限定制御列 (同じ手続き
                    # 運動に顔線が追従する)。顔ボックス未計測なら従来の
                    # 黒ギャップへフォールバック
                    _fbx = _wp_face_boxes(pack)
                    if _fbx:
                        _face_gap = _wp_moving_frames(
                            cv, refs, plan, nf, idle_n, gait_end, w, h,
                            layout, mode=_nat_ctl, face_boxes=_fbx)
                        try:
                            _face_gap[idle_n + max(
                                1, (gait_end - idle_n) // 4)].save(
                                out / f"control_{tag}_facegap.png")
                        except Exception:     # noqa: BLE001
                            pass
                        log(f"[{tag}] 実験face_line: 顔限定線画を"
                            f"間引きギャップへ ({len(_fbx)}方向に顔)")
                    else:
                        log(f"[{tag}] 実験face_line: face_boxes.json"
                            "が無いためギャップは黒 (従来)")
                try:
                    frames[idle_n].save(out / f"control_{tag}_idle.png")
                    frames[idle_n + max(1, (gait_end - idle_n) // 4)].save(
                        out / f"control_{tag}_move.png")
                except Exception:             # noqa: BLE001
                    pass
                log(f"[{tag}] {plan}: {_nat_ctl}制御既定 "
                    "(実測マップ+手続き運動+キー錨)")
            else:
                # 旧動作: 二足マネキンは非二足の姿勢を表現できないため
                # 全フレーム黒。姿勢はキー錨 (立ち絵由来) と文面が担う
                from PIL import Image as _ImgN
                frames = [_ImgN.new("RGB", (w, h), 0) for _ in range(nf)]
                log(f"[{tag}] {plan}: 骨格制御なし (キー錨+文面で誘導)")
        if _edge_face:
            import numpy as _npE
            _hcv = _wp_edge_head_canvas(cv, refs, w, h, layout)
            if _hcv.size != frames[0].size:
                _hcv = _hcv.resize(frames[0].size)
            try:
                _hcv.save(out / f"edgehead_{tag}.png")   # 検証用 (backyard)
            except Exception:                 # noqa: BLE001
                pass
            _ha = _npE.asarray(_hcv.convert("RGB"))
            _ampF = max(2, round(h * 0.012))
            _winF = max(1, gait_end - idle_n + 1)
            def _face_fix(fr, k):
                if plan == "biped" and not (idle_n <= k <= gait_end):
                    return fr                 # idle窓はfree_idleの領分
                dy = 0
                if plan == "flying" and idle_n <= k <= gait_end:
                    dy = round(_ampF * math.sin(
                        2 * math.pi * 2.0 * (k - idle_n) / _winF))
                sh = _npE.roll(_ha, dy, axis=0) if dy else _ha
                base = _npE.asarray(fr.convert("RGB"))
                from PIL import Image as _ImgE
                return _ImgE.fromarray(_npE.maximum(base, sh))
            frames = [_face_fix(fr, k) for k, fr in enumerate(frames)]
            log(f"[{tag}] 顔エッジ固定: 頭部キャニーを制御へ合成 "
                f"({'歩行窓のみ' if plan == 'biped' else '全域'})")
        canvas = cv.compose_reference(refs, w, h, layout)
        # AI単方向生成を、隣接方向と終端を共有する鎖へする。従来57fの
        # 採用外tail 8fのうち末尾4fだけを旋回に使うので、歩行2周期の
        # 抽出位置・枚数・生成回数は変わらない。各推論が見る人物画像は
        # 現在/次の2方向だけで、193fへ8方向を同居させた時の遠距離残像を
        # 原理的に避ける。
        _chain_next_canvas = None
        _chain_turn_start = nf - 4
        _chain_current = None
        _chain_next = None
        if (plan_raw == "ai" and isinstance(layout, (tuple, list))
                and len(layout) >= 3 and tuple(layout[:2]) == (1, 1)
                and len(layout[2]) == 1):
            _chain_current = str(layout[2][0])
            if _chain_current in _WP_TURN_ORDER:
                _ci = _WP_TURN_ORDER.index(_chain_current)
                _chain_next = _WP_TURN_ORDER[(_ci + 1) % len(_WP_TURN_ORDER)]
                _next_layout = (1, 1, [_chain_next])
                _next_pose, _next_ctl = _wp_ai_control_frames(
                    cv, pv, refs, _motion_kind, nf, idle_n, gait_end,
                    w, h, _next_layout, gait=_motion_gait,
                    limb_mode=_limb_mode)
                _chain_next_canvas = cv.compose_reference(
                    refs, w, h, _next_layout).convert("RGB")
                from PIL import Image as _ImgChain
                for _k in range(_chain_turn_start, nf):
                    _t = (_k - _chain_turn_start + 1) / 4.0
                    frames[_k] = _ImgChain.blend(
                        frames[_k].convert("RGB"),
                        _next_pose[_k].convert("RGB"), _t)
                log(f"[{tag}] 方向連鎖: {_chain_current} -> {_chain_next} "
                    "(末尾4fは採用外の旋回・共有終端錨)")
        # 骨格駆動の体格 (biped / 四つん這い骨格): free_idle・pose_every の
        # 「骨格=抽象記号だから間引ける」系の裁定を共有する
        _bone = (plan == "biped" or plan_raw == "quadruped_bone")
        _free = _exp.get("free_idle")
        if (_exp.get("region_mask") or _exp.get("procedural")
                or _exp.get("anisora_inpaint")):
            _free = False        # 実画素/手続き経路に骨格系の後処理は無縁
        elif _free is None:
            # ★既定ON (2026-07-20昇格): idle/末尾静止窓の骨格制御を出さず、
            # 立ち絵自身に姿勢を委ねる。静止フレームに「参照の姿勢」と
            # 「骨格の標準姿勢」が同時に権利主張する重ね合わせが四肢増産の
            # 正体だった (足開きローブの神爺さんで脚が増える実障害。
            # ユーザー診断「頭部問題の四肢版」)。姿勢権限を各フレーム1つに
            # すると増産が消え (神爺さん実証)、標準ポーズには無害 (ロップ
            # 実証)、骨格開始境界 f6 前後も全数検査でクリーン。flyingは
            # キー錨既定で充分な実績があり未検証のため見送り。自然移動
            # 体格は全フレーム黒なので対象外 (再ブランクは無意味)
            _free = (_bone
                     and not _exp.get("line_puppet")
                     and not _exp.get("scribble_mix")
                     and not _exp.get("key_interp"))
        if _free:
            from PIL import Image as _ImgF
            _blkF = _ImgF.new(frames[0].mode, frames[0].size, 0)
            frames = [_blkF if (k < idle_n or k > gait_end) else fr
                      for k, fr in enumerate(frames)]
            log(f"[{tag}] free_idle: idle/静止の骨格制御を撤去 "
                "(立ち絵の姿勢を尊重)")
        _pe_exp = int(_exp.get("pose_every") or 0)
        _pev = _pe_exp
        if not _pev:
            # 管理ノブ/環境変数の層 (実験ノブ明示 > 管理ノブ > env > 既定3)。
            # 1=毎フレーム制御 (間引きなし=従来のフル制御)。0は既存ヘルパが
            # 「未設定」と同一視するためオフ値には使えない — 1が明示オフ
            _pev = _wp_knob_int(_kn, "pose_every", int(os.environ.get(
                "SM_WP_POSE_EVERY", "").strip() or 3), 1, 12)
        # 実験g2 (2026-07-21): 非人型 (線画制御) はAPI明示のときだけ間引き
        # 可 — 既定は毎フレーム線画のまま (間引き昇格はA/B判定後)。
        # flyingは対象外 (ボブ骨格が動きの源で、間引くと静止化リスク)
        _pe_on = ((_bone
                   or (_pe_exp > 1 and plan in _WP_NAT_PLANS))
                  and not _exp.get("region_mask")
                  and not _exp.get("procedural")
                  and not _exp.get("anisora_inpaint"))
        if _pev > 1 and _pe_on:
            # 実験f (2026-07-20ユーザー発案「3フレームごとくらいに制御して
            # 間を中割させる」→同日、神爺さん/裏ファール/岡田の3体A/Bで
            # 既定昇格): 歩行窓の骨格をNフレームごとに間引き、
            # 間は黒=制御なし。モダリティは全編骨格制御のまま (e2の
            # マスク文脈ゴーストが原理的に出ない)。free_idleの歩行窓拡張
            from PIL import Image as _ImgPE
            _blkP = _ImgPE.new(frames[0].mode, frames[0].size, 0)
            frames = [fr if (k < idle_n or k > gait_end
                             or (k - idle_n) % _pev == 0)
                      else (_face_gap[k] if _face_gap else _blkP)
                      for k, fr in enumerate(frames)]
            log(f"[{tag}] pose_every={_pev}: 歩行窓の"
                f"{'骨格' if _bone else '線画制御'}を間引き "
                f"(ギャップ={'顔限定線画' if _face_gap else '黒'}、"
                "中割はVACEに委ねる)")
        if _exp.get("edge_idle"):
            _ecv = _wp_edge_canvas(cv, refs, w, h, layout)
            if _ecv.size != frames[0].size:
                _ecv = _ecv.resize(frames[0].size)
            from PIL import Image as _Img
            _blk = _Img.new(frames[0].mode, frames[0].size, 0)
            _mid = _blk if _exp.get("no_pose") else None
            _heads = None
            if _exp.get("edge_head"):
                # 頭部限定エッジを動きの窓に敷く。flyingのボブと同じsinで
                # 平行移動させ、拘束と動きの矛盾を消す (a-4)
                import math
                _hcv = _wp_edge_head_canvas(cv, refs, w, h, layout)
                if _hcv.size != frames[0].size:
                    _hcv = _hcv.resize(frames[0].size)
                _amp = max(2, round(h * 0.012))
                _win = max(1, gait_end - idle_n + 1)
                _heads = {}
                for k in range(idle_n, gait_end + 1):
                    dy = round(_amp * math.sin(
                        2 * math.pi * 2.0 * (k - idle_n) / _win))                         if plan == "flying" else 0
                    if dy not in _heads:
                        im = _Img.new(_hcv.mode, _hcv.size, 0)
                        im.paste(_hcv, (0, dy))
                        _heads[dy] = im
                try:
                    _hcv.save(out / f"edgehead_{tag}.png")
                except Exception:                 # noqa: BLE001
                    pass

                def _mid_frame(k):
                    import math as _m
                    dy = round(_amp * _m.sin(
                        2 * _m.pi * 2.0 * (k - idle_n) / _win))                         if plan == "flying" else 0
                    return _heads[dy]
            frames = [_ecv if (k < idle_n or k > gait_end)
                      else (_mid_frame(k) if _heads is not None
                            else (_mid or fr))
                      for k, fr in enumerate(frames)]
            try:                     # 何を食わせたか後から目視できるように
                _ecv.save(out / f"edgecanvas_{tag}.png")
            except Exception:                     # noqa: BLE001
                pass
            log(f"[{tag}] 実験edge_idle v2: キャラ限定エッジ (セル枠なし)")
        prompt = _wp_prompt(eng, refs, layout, nf, plan=plan,
                            gait_run=bool(_exp.get("gait_run")
                                          or _motion_gait == "run"),
                            keep_posture=bool(_exp.get("keep_posture")),
                            crawl_bone=(plan_raw == "quadruped_bone"))
        _mtext = _wp_motion_prompt(pack)
        if _mtext:
            # キャラ別モーション文の注入: 依頼の日本語コンセプトが初めて
            # 動画モデルへ届く経路 (英語ハードコード文の後置なので、
            # 既存の実績文面の座標は動かさない)
            prompt += (" Motion notes for this specific character: "
                       + _mtext)
            log(f"[{tag}] キャラ別モーション文を注入 ({len(_mtext)}字)")
        if _exp.get("legs_mask"):
            # 実験r3 (2026-07-21、忍者の黒い塊の決定打): VACEマスクの正規
            # 用法で「上半身=実画素で凍結 (mask=0・video=参照キャンバス) /
            # 脚領域=骨格入りで生成 (mask=255・video=脚骨格)」を1本に同居。
            # 参照の上げた脚は凍結領域から物理的に排除され、脚領域には
            # 骨格という動きの源が残る (実験rの静止化も回避)。
            # idle/末尾静止はマスク全0=全身実画素 (完璧な静止コマ)
            import numpy as _npLM
            from PIL import Image as _ImgLM
            _fracL = max(0.2, min(0.8, float(_exp["legs_mask"])))
            _mskL, _mbL = _wp_bottom_mask(canvas, layout, w, h, _fracL,
                                          face_boxes=_wp_face_boxes(pack),
                                          refs=refs)
            _cvA = _npLM.array(canvas.convert("RGB"))
            _mB = _mskL > 0
            _blank = _ImgLM.new("L", (w, h), 0)
            import io as _ioLM
            import base64 as _b64LM
            _bufB = _ioLM.BytesIO()
            _blank.save(_bufB, format="PNG")
            _mb_keep = _b64LM.b64encode(_bufB.getvalue()).decode("ascii")
            _masks = []
            _vid = []
            for k, fr in enumerate(frames):
                if k < idle_n or k > gait_end:
                    _vid.append(canvas.convert("RGB"))
                    _masks.append(_mb_keep)
                else:
                    _fa = _npLM.array(fr.convert("RGB"))
                    _cmp = _npLM.where(_mB[..., None], _fa, _cvA)
                    _vid.append(_ImgLM.fromarray(_cmp.astype("uint8")))
                    _masks.append(_mbL)
            frames = _vid
            _exp["_legs_masks_b64"] = _masks
            try:
                frames[idle_n + max(1, (gait_end - idle_n) // 4)].save(
                    out / f"control_{tag}_legsmask.png")
            except Exception:                 # noqa: BLE001
                pass
            log(f"[{tag}] 実験legs_mask={_fracL}: 上=実画素凍結 / "
                "下=脚骨格入り生成 (idle/静止=全身実画素)")
        if _exp.get("anisora_inpaint"):
            # 本線AI (0.11.1): 手続き動画をSDEditの土台にしない。
            # 固定側の頭部原画+maskは歩行ボブへ追従し、bbox外背景だけ静止。
            # 体マスク内はσ1.0の純ノイズからAniSoraに歩行を埋めさせる。
            # VACE不使用。
            _frac2 = max(0.2, min(0.8, float(_exp["anisora_inpaint"])))
            _im_mode = str(_exp.get("inpaint_mode") or "face").lower()
            if _im_mode == "face":
                # A型 (2026-07-21ユーザー図解): 凍結=顔窓+背景、体は全部
                # AIに渡す — B型 (下部帯のみ生成) は顔より上の胴まで凍結
                # してしまい、芝居が下のひと帯に限られていた
                _fbx2 = _wp_face_boxes(pack)
                if not _fbx2:
                    raise RuntimeError(
                        f"[{tag}] AI生成に必要なface_boxes.jsonがありません "
                        "(顔/背面頭部を凍結できないため生成を中止)")
                _msk2, _mb64, _head2 = _wp_face_keep_mask(
                    canvas, layout, w, h,
                    face_boxes=_fbx2, refs=refs, return_head_keep=True)
                # 今回確定した契約: 固定側は常にframe0由来。
                # 顔参照やmaskをボブ移動させず、体の動きは
                # HighのPose条件だけが担う。
                _art2 = [canvas.convert("RGB").copy() for _ in range(nf)]
                _masks2 = None
                _mb64s = [_mb64] * nf
                # Lowでは顔/頭部だけ開放し、bbox外背景のlatent固定は継続。
                # 最終画素貼り戻しはしないので、境界をLowが自然に描ける。
                import base64 as _b64LOW
                import io as _ioLOW
                import numpy as _npLOW
                from PIL import Image as _ImgLOW
                _low_m = _npLOW.full(_msk2.shape, 255, dtype=_npLOW.uint8)
                _low_m[(_msk2 < 128) & ~(_head2 > 0)] = 0
                _low_buf = _ioLOW.BytesIO()
                _ImgLOW.fromarray(_low_m, "L").save(_low_buf, format="PNG")
                _low_mb64 = _b64LOW.b64encode(
                    _low_buf.getvalue()).decode("ascii")
            else:
                _msk2, _mb64 = _wp_bottom_mask(
                    canvas, layout, w, h, _frac2,
                    face_boxes=_wp_face_boxes(pack), refs=refs)
                _art2 = [canvas.convert("RGB").copy() for _ in range(nf)]
                _masks2 = None
                _mb64s = [_mb64] * nf
                _low_mb64 = None
            # frame0=全面固定、以後=元絵の実シルエット+
            # その時刻のPose周囲だけ生成。bbox長方形の余白を
            # AIに背景再描画させない。
            _masks2, _mb64s = _wp_dynamic_pose_masks(
                canvas, _msk2, frames, nf)
            if _chain_next_canvas is not None:
                from PIL import Image as _ImgChainArt
                for _k in range(_chain_turn_start, nf):
                    _t = (_k - _chain_turn_start + 1) / 4.0
                    _art2[_k] = _ImgChainArt.blend(
                        canvas.convert("RGB"), _chain_next_canvas, _t)
                # Wanの最終latent slotは画素frame(nf-2)のmaskを代表に採る。
                # そこを全面既知にし、4f旋回+最終次方向絵のrefine潜在へ固定。
                import base64 as _b64Chain
                import io as _ioChain
                _anchor_mask = _ImgChainArt.new("L", (w, h), 0)
                _anchor_buf = _ioChain.BytesIO()
                _anchor_mask.save(_anchor_buf, format="PNG")
                _anchor_b64 = _b64Chain.b64encode(
                    _anchor_buf.getvalue()).decode("ascii")
                _masks2[nf - 2] = _anchor_mask
                _mb64s[nf - 2] = _anchor_b64
            try:
                _art2[0].save(out / f"control_{tag}_staticbase.png")
                from PIL import Image as _ImgAI
                _ImgAI.fromarray(_msk2, "L").save(
                    out / f"control_{tag}_spatialmask.png")
                _peak2 = idle_n + max(1, (gait_end - idle_n) // 4)
                _masks2[min(nf - 1, _peak2)].save(
                    out / f"control_{tag}_dynamicmask.png")
            except Exception:                 # noqa: BLE001
                pass
            j["detail"] = f"[{tag}] AniSora単体インペイント"
            # 空潜在の契約を守るため開始σは1.0固定。中途σに下げると
            # 「純ノイズを途中のノイズレベルへ投入」となり分布が矛盾する。
            _sg2 = 1.0
            # 1方向ずつ8回へ分離したため各方向は8stepを既定にする。
            # 半球2枚×24stepに近い総予算のまま、注意を1体へ集中できる。
            _st_ai = _wp_knob_int(_kn, "refine_total", 8, 4, 12)
            _release_ai = min(_st_ai, _wp_knob_int(
                _kn, "head_release_steps", 0, 0, 12))
            _pose_weight_ai = _wp_knob_float(
                _kn, "pose_weight", 0.25, 0.05, 0.50)
            if (_motion_kind == "biped"
                    and _chain_current in ("left", "right")):
                # OpenPoseを強めると緑骨が漏れるだけだったため、側面は
                # 色付き原画パペットへ差し替えたうえで50%まで効かせる。
                _pose_weight_ai = max(0.50, _pose_weight_ai)
            _guide2 = [canvas.convert("RGB").copy()] + list(frames[1:])
            if _chain_next_canvas is not None:
                prompt += (
                    f" During only the final four transition frames, turn "
                    f"smoothly clockwise from the {_chain_current} view to "
                    f"the {_chain_next} view without stopping the gait.")
            log(f"[{tag}] anisora_inpaint mode={_im_mode}: frame0全固定+"
                "体領域純ノイズ (AniSora単体・VACE不使用、"
                f"開始σ={_sg2} steps={_st_ai} "
                f"Pose={_pose_weight_ai:.0%}/画像={1-_pose_weight_ai:.0%} "
                f"頭部固定={_st_ai - _release_ai}step/"
                f"終盤開放={_release_ai}step)")
            extra2 = {"refine_frames_b64": pv.encode_frames_b64(_art2),
                      "refine_strength": _sg2,
                      "refine_cond_still": True,
                      "latent_spatial_mask_b64": _mb64s,
                      "latent_spatial_empty": True,
                      "latent_spatial_source_mix": 0.0,
                      "latent_spatial_release_last_steps": _release_ai,
                      "anisora_guidance_frames_b64":
                          pv.encode_frames_b64(_guide2),
                      "anisora_guidance_mask": "first",
                      "anisora_guidance_release_low": True,
                      "anisora_guidance_spatial_condition": True,
                      "anisora_guidance_neutralize_black": True,
                      "anisora_dual_condition": True,
                      "anisora_dual_condition_image_weight":
                          1.0 - _pose_weight_ai,
                      "motion_score": _wp_knob_float(
                          _kn, "motion", 3.0, 2.0, 4.0)}
            if _chain_next_canvas is not None:
                extra2["anisora_dual_condition_image_weights"] = (
                    [1.0 - _pose_weight_ai] * _chain_turn_start
                    + [0.25] * 4)
            if _low_mb64:
                extra2["latent_spatial_low_mask_b64"] = _low_mb64
            if offload:
                extra2["offload"] = offload
            _wmAI = "mock" if j.get("_wp_mock") else "anisora"
            jid2 = submit_job(_wmAI, GenRequest(
                mode="i2v", prompt=prompt,
                images=([canvas, _chain_next_canvas]
                        if _chain_next_canvas is not None else [canvas]),
                width=w, height=h, num_frames=nf, fps=WALKPACK_FPS,
                steps=_st_ai, seed=seed, guidance=1.0, extra=extra2))
            sj2 = _wp_wait(j, jid2, s1lo, s2hi)
            cvid = out / f"canvas_{tag}.mp4"
            shutil.copy2(sj2["path"], cvid)
            try:
                (out / f"prompt_{tag}.txt").write_text(prompt,
                                                       encoding="utf-8")
                canvas.save(out / f"refcanvas_{tag}.png")
            except Exception:                 # noqa: BLE001
                pass
            j["detail"] = f"[{tag}] セル分割"
            _wp_split(eng, ffmpeg, cvid, layout, refs, char, out,
                      idle_n, gait_end if tail else None, log,
                      canvas_w=w, canvas_h=h)
            return
        if _exp.get("procedural"):
            # 実験p (2026-07-21ユーザー発案「ゆらゆら・上下・変形だけなら
            # AIを使う必要すらない」): 参照絵そのものへ体格別の手続き運動を
            # かけ、生成なしでcanvas動画を組む。同一性=画素完全・生成コスト
            # ゼロ・数秒で完成
            j["detail"] = f"[{tag}] 手続きアニメ (生成なし)"
            _art = _wp_moving_frames(cv, refs, plan, nf, idle_n, gait_end,
                                     w, h, layout, mode="art")
            import tempfile as _tfP
            _fdir = Path(_tfP.mkdtemp(prefix="proc_frames_"))
            for _i, _fr in enumerate(_art):
                _fr.save(_fdir / f"{_i:05d}.png")
            cvid = out / f"canvas_{tag}.mp4"
            encode_mp4(ffmpeg, _fdir, WALKPACK_FPS, cvid)
            log(f"[{tag}] 手続きアニメ: {plan}の変形を参照絵へ直接適用 "
                f"({nf}f、生成なし)")
            try:
                (out / f"prompt_{tag}.txt").write_text("(procedural)",
                                                       encoding="utf-8")
                canvas.save(out / f"refcanvas_{tag}.png")
            except Exception:                 # noqa: BLE001
                pass
            j["detail"] = f"[{tag}] セル分割"
            _wp_split(eng, ffmpeg, cvid, layout, refs, char, out,
                      idle_n, gait_end if tail else None, log,
                      canvas_w=w, canvas_h=h)
            return
        extra1 = {"pose_frames_b64": pv.encode_frames_b64(frames),
                  "conditioning_scale": 1.0, "motion_score": 3.0,
                  "vace_base": "fun", "vace_lora": "lightning",
                  "emit_latent": 1}
        if _exp.get("_keep_frames"):
            extra1["vace_keep_frames"] = _exp["_keep_frames"]
        if _exp.get("_region_mask_b64"):
            extra1["vace_mask_b64"] = [_exp["_region_mask_b64"]]
        if _exp.get("_legs_masks_b64"):
            extra1["vace_mask_b64"] = _exp["_legs_masks_b64"]
        if offload:
            extra1["offload"] = offload
        j["detail"] = f"[{tag}] 生成1/2: VACEフル骨格"
        log(f"[{tag}] 生成1/2: VACEフル骨格制御 "
            f"steps={_wp_knob_int(_kn, 'vace_steps', 4, 1, 8)} cfg=1.0 "
            "(latent直出し)")
        if _exp.get("skip_vace"):
            # ★実験: VACE抜き。骨格フレーム列を refine_frames_b64 で直接
            # latent化し、σ再加工の出発点+latent固定キーにする。つまり
            # 「AniSoraが参照キャンバス(i2v条件)を見ながら、骨格画像の
            # 錨の間を自力で描く」構図。σはアダプタ側クランプで最大0.90
            j["detail"] = f"[{tag}] 実験: VACE抜き (骨格を直接latent化)"
            log(f"[{tag}] 実験skip_vace: 骨格{len(frames)}fを直接latent源に")
            extra2 = {"refine_frames_b64": pv.encode_frames_b64(frames),
                      "refine_strength": float(_exp.get("refine") or 0.9),
                      "refine_cond_still": True,
                      "motion_score": _wp_knob_float(
                          _kn, "motion", 3.0, 2.0, 4.0)}
            _pin_rel0 = float(os.environ.get(
                "SM_VACE_LR_PIN_RELEASE", "").strip() or WP_LR_PIN_RELEASE)
            if _pin_rel0 > 0:
                extra2["latent_pin_release"] = _pin_rel0
            if pins:
                extra2["latent_pin_frames"] = pins
            if offload:
                extra2["offload"] = offload
            _wm2x = "mock" if j.get("_wp_mock") else "anisora"
            jid2 = submit_job(_wm2x, GenRequest(
                mode="i2v", prompt=prompt, images=[canvas],
                width=w, height=h, num_frames=nf, fps=WALKPACK_FPS,
                steps=24, seed=seed, guidance=1.0, extra=extra2))
            sj2 = _wp_wait(j, jid2, s1lo, s2hi)
            cvid = out / f"canvas_{tag}.mp4"
            shutil.copy2(sj2["path"], cvid)
            try:
                (out / f"prompt_{tag}.txt").write_text(prompt,
                                                       encoding="utf-8")
                canvas.save(out / f"refcanvas_{tag}.png")
            except Exception:                 # noqa: BLE001
                pass
            j["detail"] = f"[{tag}] セル分割"
            _wp_split(eng, ffmpeg, cvid, layout, refs, char, out,
                      idle_n, gait_end if tail else None, log,
                      canvas_w=w, canvas_h=h)
            return
        _st1 = _wp_knob_int(_kn, "vace_steps", 4, 1, 8)
        _wm1 = "mock" if j.get("_wp_mock") else "vace"
        jid1 = submit_job(_wm1, GenRequest(
            mode="i2v", prompt=prompt, images=[canvas],
            width=w, height=h, num_frames=nf, fps=WALKPACK_FPS,
            steps=_st1, seed=seed, guidance=1.0, extra=extra1))
        sj1 = _wp_wait(j, jid1, s1lo, s1hi)
        try:
            # ★検証用の中間保全 (2026-07-19ユーザー提案「結果のみだと
            # どの段階で劣化したか特定が難しい」): stage1(VACE)動画・
            # プロンプト・参照キャンバスを out/ へ残す。stage2の
            # canvas_{tag}.mp4 と対にすると、劣化がVACE段かAniSora再加工段
            # かを後から画で切り分けられる。完了時に debug/ へ写す
            _p1 = (sj1 or {}).get("path")
            if _p1 and Path(_p1).is_file():
                shutil.copy2(_p1, out / f"canvas_{tag}_stage1.mp4")
            (out / f"prompt_{tag}.txt").write_text(prompt, encoding="utf-8")
            canvas.save(out / f"refcanvas_{tag}.png")
        except Exception as e:                # noqa: BLE001
            log(f"[{tag}] 中間保全に失敗 (無視して続行): {str(e)[:80]}")
        extra2 = {"latent_from": jid1,
                  "refine_strength": float(
                      _exp.get("refine")
                      or os.environ.get("SM_VACE_LR_REFINE", "").strip()
                      or WP_LR_REFINE),
                  "refine_cond_still": True,
                  # stage2の動き量は管理ノブ"motion"で調整可 (公式レンジ
                  # 2.0-4.0、既定3.0=V3.2公式例の標準値)。stage1は3.0固定
                  # (キー錨の錨自体を動かす変更は別実験に分離)
                  "motion_score": _wp_knob_float(
                      _kn, "motion", 3.0, 2.0, 4.0)}
        _pin_rel = float(os.environ.get("SM_VACE_LR_PIN_RELEASE", "").strip()
                         or WP_LR_PIN_RELEASE)
        if _pin_rel > 0:
            extra2["latent_pin_release"] = _pin_rel
        if pins:
            extra2["latent_pin_frames"] = pins
        if offload:
            extra2["offload"] = offload
        _st2 = _wp_knob_int(_kn, "refine_total", 24, 4, 40)
        j["detail"] = f"[{tag}] 生成2/2: AniSora latent再加工"
        log(f"[{tag}] 生成2/2: AniSora latent再加工 "
            f"σ={extra2['refine_strength']} steps={_st2} "
            f"(latent固定 {pins})")
        _wm2 = "mock" if j.get("_wp_mock") else "anisora"
        jid2 = submit_job(_wm2, GenRequest(
            mode="i2v", prompt=prompt, images=[canvas],
            width=w, height=h, num_frames=nf, fps=WALKPACK_FPS,
            steps=_st2, seed=seed, guidance=1.0, extra=extra2))
        sj2 = _wp_wait(j, jid2, s2lo, s2hi)
        cvid = out / f"canvas_{tag}.mp4"
        shutil.copy2(sj2["path"], cvid)
        j["detail"] = f"[{tag}] セル分割"
        _wp_split(eng, ffmpeg, cvid, layout, refs, char, out,
                  idle_n, gait_end if tail else None, log,
                  canvas_w=w, canvas_h=h)

    _lay_req = str(_exp.get("layout") or "").strip()
    _use_single = (plan_raw == "ai" and not _lay_req) or _lay_req in (
        "single", "individual", "8dir")
    if _lay_req == "compass":
        _use_c9 = True
    elif _lay_req in ("hemi", "4x2"):
        _use_c9 = False
    else:
        # ★既定=コンパス昇格 (2026-07-20ユーザー仮説→3体実証):
        # 半球2枚 (F4/B4) は backの隣にright という角度不連続の同居で
        # 隣接セル汚染が起きる (backに余計な尻尾がrightから染み、rightの
        # 翼がbackへ吸われて板化)。3x3コンパスは隣=隣接角度なので染みても
        # 無害 — ドラゴン(尻尾/板翼消滅)・イーリス(right品質回復)・
        # ロップ(biped歩行健全) で確認。キャンバス2.25倍のVRAMが要るため
        # 40GB未満の機体 (L4/T4/A100-40) は従来の半球へフォールバック
        # (半球時代の絵はそのまま再現可能 = layout="hemi" 明示でも可)
        _use_c9 = False
        try:
            import torch
            if torch.cuda.is_available():
                _tot = torch.cuda.get_device_properties(0).total_memory
                _use_c9 = _tot >= 40 * (1 << 30)
        except Exception:                     # noqa: BLE001
            pass
    if _use_single:
        _use_c9 = False
        _wp_print("[walkpack] AIレイアウト: 1方向ずつ8回 "
                  "(各8step・注意集中)")
    elif _use_c9 and _lay_req != "compass":
        _wp_print("[walkpack] レイアウト: コンパス3x3 (VRAM充足・既定)")
    if _use_c9:
        w, h = (WALKPACK_W // 2) * 3, (WALKPACK_H // 2) * 3
    if _use_single:
        _layouts = tuple(
            (f"D{i + 1:02d}_{d}", (1, 1, [d]))
            for i, d in enumerate(_WP_DIRS))
    else:
        _layouts = ((("C9", cw.LAYOUT_COMPASS),) if _use_c9
                    else (("F4", cw.LAYOUT_F4), ("B4", cw.LAYOUT_B4)))
    for tag, layout in _layouts:
        _gen_hemisphere(tag, layout)
        # 顔が写るのは前半球だけ (B4は後ろ姿3方向+横顔) なので、ゲートは
        # F4の直後に1回だけ回す。1レイアウト=1回の生成=1個のノイズなので、
        # 崩れた方向だけを個別に作り直すことはできない — 半球ごと引き直す。
        if tag != "F4" or not _wp_face_retry_on():
            continue
        bad, sc = _wp_face_gate(eng, ffmpeg, out, log)
        if not bad:
            continue
        keep = out / "_face_retry"
        shutil.rmtree(keep, ignore_errors=True)
        keep.mkdir(parents=True)
        names = [p.name for p in out.glob("*_walkT.mp4")]
        names.append(f"canvas_{tag}.mp4")
        for n in names:                   # 1回目を退避 (悪化したら戻す)
            if (out / n).is_file():
                shutil.copy2(out / n, keep / n)
        log(f"  顔ゲートNG → シードを変えて{tag}を作り直します "
            f"(1回目を{len(names)}件退避)")
        j["detail"] = f"[{tag}] 顔ゲートNG — 作り直し中"
        _gen_hemisphere(tag, layout, seed=WP_FACE_RETRY_SEED)
        _bad2, sc2 = _wp_face_gate(eng, ffmpeg, out, log)

        def _ratio(s: dict) -> float:
            """前向き系の最小/最大。1に近いほど全方向で顔が出ている。"""
            return (min(s.values()) / max(s.values())) if s else 0.0
        if _ratio(sc2) <= _ratio(sc):
            log(f"  作り直しは改善せず ({_ratio(sc):.2f} → "
                f"{_ratio(sc2):.2f}) — 1回目を採用します")
            for n in names:
                if (keep / n).is_file():
                    shutil.copy2(keep / n, out / n)
        else:
            log(f"  作り直しで改善 ({_ratio(sc):.2f} → {_ratio(sc2):.2f})")
        shutil.rmtree(keep, ignore_errors=True)
    if plan_raw == "ai" and _use_single:
        # 共有終端=次区間の共有始端なので、2区間目以降のframe0だけ
        # 落として時計回りの一本へ連結する。シート/QCは従来どおり各方向の
        # 歩行窓だけを使い、この動画は連続性の証跡と将来の時系列抽出用。
        _dir_index = {d: i + 1 for i, d in enumerate(_WP_DIRS)}
        _chain_inputs = [
            out / f"canvas_D{_dir_index[d]:02d}_{d}.mp4"
            for d in _WP_TURN_ORDER]
        if all(p.is_file() for p in _chain_inputs):
            try:
                _cmd = [str(ffmpeg), "-y", "-loglevel", "error"]
                for _p in _chain_inputs:
                    _cmd += ["-i", str(_p)]
                _parts = []
                for _i in range(len(_chain_inputs)):
                    _start = 0 if _i == 0 else 1
                    _parts.append(
                        f"[{_i}:v]trim=start_frame={_start}:end_frame={nf},"
                        f"setpts=PTS-STARTPTS[c{_i}]")
                _links = "".join(
                    f"[c{_i}]" for _i in range(len(_chain_inputs)))
                _filter = (";".join(_parts) + ";" + _links
                           + f"concat=n={len(_chain_inputs)}:v=1:a=0[outv]")
                _cmd += ["-filter_complex", _filter, "-map", "[outv]",
                         "-r", str(WALKPACK_FPS), "-c:v", "libx264",
                         "-crf", "18", "-pix_fmt", "yuv420p",
                         str(out / "canvas_AI_chain.mp4")]
                subprocess.run(_cmd, check=True, capture_output=True)
                log("AI方向連鎖: 時計回り8区間をcanvas_AI_chain.mp4へ連結")
            except Exception as e:            # noqa: BLE001
                log(f"⚠ AI方向連鎖動画の連結に失敗 (方向別成果は正常): "
                    f"{str(e)[:200]}")
    j["progress"] = 0.9
    j["_beat"] = time.time()
    j["detail"] = "シート/プレビュー組み立て"
    try:
        _wp_assemble(eng, ffmpeg, out, meta, log)
    except Exception as e:                # noqa: BLE001
        # 必須成果物 (方向別mp4 8本 + canvas mp4) は揃っているので続行
        log(f"⚠ シート/プレビュー組み立てに失敗 (方向別mp4は利用可能): "
            f"{str(e)[:300]}")
    j["path"] = str(out)
    j["detail"] = ""


def _walkpack_thread(jid: str, pid: str) -> None:
    j = JOBS.get(jid)
    if j is None:
        return

    def log(msg):
        j["log"].append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        j["_beat"] = time.time()
        _wp_print(f"[{jid}] {msg}")

    prev_ls = os.environ.get("SM_LEG_SCALE")
    try:
        with _WALKPACK_LOCK:
            if j.get("_cancel"):
                raise JobCancelled()
            j["status"] = "running"
            j["started"] = time.time()
            meta = _pack_meta(packs_root() / pid)
            try:
                ls = float(meta.get("leg_scale") or 1.0)
            except (TypeError, ValueError):
                ls = 1.0
            os.environ["SM_LEG_SCALE"] = str(ls)
            _knobs = _wp_admin_knobs()           # 管理GUI (受付台/admin) の値
            j["_wp_knobs"] = _knobs
            _knob_env = _wp_knob_env_set(_knobs)
            _plan_env = _wp_plan_env_set(meta)   # flyingならノブ上書き
            try:
                _walkpack_run(j, pid, meta, log)
            finally:
                _wp_plan_env_restore(_plan_env)
                _wp_plan_env_restore(_knob_env)
            j["status"] = "done"
            j["progress"] = 1.0
            log("walk_pack 完了")
    except JobCancelled:
        j["status"] = "cancelled"
        j["detail"] = "ユーザーによりキャンセル"
    except Exception as e:                # noqa: BLE001
        j["status"] = "error"
        j["detail"] = str(e)[:600]
        j["log"].append(traceback.format_exc()[-1500:])
        _wp_print(f"[{jid}] ERROR: {e}")
    finally:
        if prev_ls is None:
            os.environ.pop("SM_LEG_SCALE", None)
        else:
            os.environ["SM_LEG_SCALE"] = prev_ls
        j["finished"] = time.time()


def submit_walkpack(pid: str, mock: bool = False,
                    exp: dict | None = None) -> str:
    """疑似ジョブ (model=walkpack) を登録してオーケストレーション
    スレッドを起動する。既存workerキューには入れない (実生成は内部の
    submit_job が通常経路で流れる)。同一パックの進行中ジョブがあれば
    それを返す (二重投入防止)。"""
    with _LOCK:
        for jid0 in reversed(JOB_ORDER):
            j0 = JOBS.get(jid0)
            if (j0 and j0.get("model") == "walkpack"
                    and j0.get("pack") == pid
                    and j0.get("status") in ("queued", "running")):
                return jid0
        jid = uuid.uuid4().hex[:12]
        JOBS[jid] = {
            "id": jid, "status": "queued", "model": "walkpack",
            "_wp_exp": dict(exp or {}),
            "mode": "walkpack", "pack": pid,
            "prompt": f"walk_pack: {pid}", "progress": 0.0,
            "detail": "工房キュー待ち", "created": time.time(),
            "started": None, "finished": None, "path": None, "log": [],
            "_wp_mock": bool(mock),
            "params": {"width": WALKPACK_W, "height": WALKPACK_H,
                       "num_frames": WALKPACK_NF, "fps": WALKPACK_FPS,
                       "steps": 4, "seed": 42, "guidance": 1.0,
                       "images": 8},
            "_cancel": False,
        }
        JOB_ORDER.append(jid)
    threading.Thread(target=_walkpack_thread, args=(jid, pid),
                     daemon=True).start()
    return jid


# ============================================================ GCE作業員 (v0.10.2)
# 「呼ばれたら起きて仕事し、暇になったら自分で電源を切る」GPU作業員モード。
# 依頼の真実の置き場はGCSバケット (Cloud Runの受付台 kobo_front が管理)。
# VMはGCSから status=pack_ready の依頼を拾い、歩行生成して outputs/ へ書き戻す。
# 本番はVMが公開ポートを持たずGCSだけと対話する構成でも成立する。
#
# 全機能は環境変数ゲート。無設定なら Colab/ローカル経路は一切変わらない:
#   VIDEOLAB_GCS_BUCKET     … 立つとGCSワーカーが回り出す (依頼の取り込み〜書き戻し)
#   VIDEOLAB_IDLE_STOP_MIN  … >0 でアイドルN分の自己停止を有効化 (課金の守り)
#   VIDEOLAB_GCE=1          … 停止手段を poweroff にする (未設定は _shutdown_runtime)
#   VIDEOLAB_GCS_FAKE=<dir> … GCSをローカルdirで模擬 (テスト用・バケット名無視)
import urllib.request as _url_req
import urllib.parse as _url_parse
import urllib.error as _url_err

GCS_BUCKET = os.environ.get("VIDEOLAB_GCS_BUCKET", "").strip()
_GCS_FAKE = os.environ.get("VIDEOLAB_GCS_FAKE", "").strip()
try:
    _GCS_POLL_SEC = max(5, int(os.environ.get("VIDEOLAB_GCS_POLL", "30") or "30"))
except ValueError:
    _GCS_POLL_SEC = 30
_LAST_HTTP = [time.time()]       # HTTPミドルウェアが更新 (可変参照でクロージャ共有)
_LAST_GCS_WORK = [time.time()]   # GCSワーカーが最後に仕事した時刻
_GCE_THREADS_UP = [False]
# 請負中の依頼 (rid, pack_id)。SIGTERM/自己停止時に generating を pack_ready
# へ戻すための目印 (2026-07-20 実障害4eb7f4ce: 外部stopが請負91秒後に直撃し
# generating が永久残留)。generating 書込より前に立てること — 書込前に死ねば
# 解放は no-op で pack_ready のまま=受付台の起こし直しが効く。
_GCS_INFLIGHT: list = [None]
# 停止シーケンス開始後は新規の請負を止める (停止決定〜電源断の窓で
# pack_ready を generating に引き込んで死ぬTOCTOUの防止)
_GCS_STOPPING = [False]

_META = "http://metadata.google.internal/computeMetadata/v1/"
_GCS_TOK = {"tok": "", "exp": 0.0}
_GCS_TOK_LOCK = threading.Lock()


def _gcs_active() -> bool:
    return bool(GCS_BUCKET or _GCS_FAKE)


def _idle_min() -> int:
    try:
        return int(os.environ.get("VIDEOLAB_IDLE_STOP_MIN", "0") or "0")
    except ValueError:
        return 0


def _meta_get(path: str, timeout: int = 8) -> str:
    r = _url_req.Request(_META + path, headers={"Metadata-Flavor": "Google"})
    with _url_req.urlopen(r, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def _gcs_token() -> str:
    with _GCS_TOK_LOCK:
        if time.time() < _GCS_TOK["exp"]:
            return _GCS_TOK["tok"]
        d = json.loads(_meta_get("instance/service-accounts/default/token"))
        _GCS_TOK["tok"] = str(d["access_token"])
        _GCS_TOK["exp"] = time.time() + 60.0
        return _GCS_TOK["tok"]


def _gcs_read(name: str):
    if _GCS_FAKE:
        p = Path(_GCS_FAKE) / name
        return p.read_bytes() if p.is_file() else None
    url = ("https://storage.googleapis.com/storage/v1/b/"
           + _url_parse.quote(GCS_BUCKET, safe="") + "/o/"
           + _url_parse.quote(name, safe="") + "?alt=media")
    r = _url_req.Request(url,
                         headers={"Authorization": "Bearer " + _gcs_token()})
    try:
        with _url_req.urlopen(r, timeout=300) as resp:
            return resp.read()
    except _url_err.HTTPError as e:
        if e.code == 404:
            return None
        raise


def _gcs_write(name: str, data: bytes,
               content_type: str = "application/octet-stream") -> None:
    if _GCS_FAKE:
        p = Path(_GCS_FAKE) / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return
    url = ("https://storage.googleapis.com/upload/storage/v1/b/"
           + _url_parse.quote(GCS_BUCKET, safe="")
           + "/o?uploadType=media&name=" + _url_parse.quote(name, safe=""))
    r = _url_req.Request(url, data=data, method="POST",
                         headers={"Authorization": "Bearer " + _gcs_token(),
                                  "Content-Type": content_type})
    with _url_req.urlopen(r, timeout=600) as resp:
        resp.read()


def _gcs_list(prefix: str) -> list:
    if _GCS_FAKE:
        root = Path(_GCS_FAKE)
        out = []
        for p in root.rglob("*"):
            if p.is_file():
                nm = p.relative_to(root).as_posix()
                if nm.startswith(prefix):
                    out.append(nm)
        return sorted(out)
    names, page = [], ""
    while True:
        url = ("https://storage.googleapis.com/storage/v1/b/"
               + _url_parse.quote(GCS_BUCKET, safe="") + "/o?prefix="
               + _url_parse.quote(prefix, safe="")
               + "&fields=items(name),nextPageToken")
        if page:
            url += "&pageToken=" + _url_parse.quote(page)
        r = _url_req.Request(
            url, headers={"Authorization": "Bearer " + _gcs_token()})
        with _url_req.urlopen(r, timeout=60) as resp:
            d = json.loads(resp.read())
        names += [str(it["name"]) for it in d.get("items", [])]
        page = str(d.get("nextPageToken") or "")
        if not page:
            return names


def _gcs_delete(name: str) -> None:
    if _GCS_FAKE:
        p = Path(_GCS_FAKE) / name
        if p.is_file():
            p.unlink()
        return
    url = ("https://storage.googleapis.com/storage/v1/b/"
           + _url_parse.quote(GCS_BUCKET, safe="") + "/o/"
           + _url_parse.quote(name, safe=""))
    r = _url_req.Request(url, method="DELETE",
                         headers={"Authorization": "Bearer " + _gcs_token()})
    try:
        with _url_req.urlopen(r, timeout=60) as resp:
            resp.read()
    except _url_err.HTTPError as e:
        if e.code != 404:
            raise


GCS_KEEP_OUTPUTS = 100   # 生成物の保持件数 (容量圧迫回避)。超過分は古い順に削除


def _gcs_debug_keep() -> int:
    """検証用中間成果物 (debug/) の保持件数。0で無効。"""
    try:
        return int(os.environ.get("VIDEOLAB_DEBUG_KEEP", "12"))
    except ValueError:
        return 12


def _gcs_debug_upload(rid: str, out) -> None:
    """中間成果物 (stage1/2キャンバス動画・プロンプト・参照キャンバス) を
    gs://.../debug/<epoch>_<rid>/ へ写す (2026-07-19ユーザー提案)。配布物
    ではないのでギャラリー/manifestには載せない。保持は新しい順に
    VIDEOLAB_DEBUG_KEEP件 (既定12)。失敗しても本流は止めない。"""
    keep = _gcs_debug_keep()
    if keep <= 0 or not _gcs_active():
        return
    try:
        pre = f"debug/{int(time.time())}_{rid}/"
        n = 0
        for p in sorted(Path(out).iterdir()):
            if not p.is_file():
                continue
            if not (p.name.startswith(("canvas_", "prompt_", "refcanvas_"))):
                continue
            _gcs_write(pre + p.name, p.read_bytes(),
                       _GCS_CT.get(p.suffix.lower(),
                                   "application/octet-stream"))
            n += 1
        _wp_print(f"[GCS] 検証用中間 {n}件を {pre} へ保全")
        # 保持を超えた古いdebugセットを削除 (プレフィクスはepoch付きで
        # 名前順=時系列)
        seen = sorted({name.split("/", 2)[1]
                       for name in _gcs_list("debug/")
                       if name.count("/") >= 2})
        for stale in seen[:-keep] if len(seen) > keep else []:
            for name in _gcs_list(f"debug/{stale}/"):
                _gcs_delete(name)
    except Exception as e:                    # noqa: BLE001
        _wp_print(f"[GCS] 中間保全アップロード失敗 (無視): {str(e)[:120]}")


def _gcs_prune_outputs(keep: int = GCS_KEEP_OUTPUTS) -> None:
    """終端状態の生成物を最新keep件だけ残し、古いものを消す。

    削除対象は outputs/<rid>/ 一式 + requests/<rid>.json(+ref) + packs/<pid>.zip。
    (2026-07-19) 以前は done だけが対象で、failed/cancelled の依頼が抱える
    ref.png (最大20MB) / pack zip (最大300MB) / manifest無しの途中outputs が
    永遠に残り課金され続けた。done と別窓の同じkeep件で刈る (失敗が積もっても
    完成品の保持数を圧迫しない)。
    ベストエフォート — 失敗しても本処理は継続 (呼び出し側でtry)。"""
    dones, fails = [], []
    for name in _gcs_list("requests/"):
        if not name.endswith(".json"):
            continue
        rid = name[len("requests/"):-len(".json")]
        req = _gcs_req_load(rid)
        if not req:
            continue
        row = (float(req.get("updated") or 0), rid,
               str(req.get("pack_id") or ""))
        if req.get("status") == "done":
            dones.append(row)
        elif req.get("status") in ("failed", "cancelled"):
            fails.append(row)
    doomed = []
    for rows in (dones, fails):
        if len(rows) <= keep:
            continue
        rows.sort(reverse=True)               # 新しい順
        doomed += rows[keep:]                 # keep番目以降=古い
    if not doomed:
        return
    for _ts, rid, pid in doomed:
        for on in _gcs_list(f"outputs/{rid}/"):
            _gcs_delete(on)
        _gcs_delete(f"requests/{rid}.json")
        _gcs_delete(f"requests/{rid}.ref.png")
        if pid:
            _gcs_delete(f"packs/{pid}.zip")
    _wp_print(f"[GCS] 保持{keep}件を超える{len(doomed)}件の古い生成物を削除")


def _gcs_req_load(rid: str):
    b = _gcs_read(f"requests/{rid}.json")
    if b is None:
        return None
    try:
        d = json.loads(b.decode("utf-8"))
        return d if isinstance(d, dict) else None
    except (ValueError, UnicodeDecodeError):
        return None      # 書き込み途中/壊れた依頼


def _gcs_req_save(req: dict) -> None:
    req["updated"] = time.time()
    _gcs_write(f"requests/{req['request_id']}.json",
               json.dumps(req, ensure_ascii=False).encode("utf-8"),
               "application/json")


def _gcs_req_finish(req: dict, status: str, error: str = "") -> None:
    """生成の終了状態 (done/failed) をGCSへ書き戻す。

    (2026-07-19) 数十分の生成中に受付 (kobo_front) が同じ依頼を書き換える
    ことがある (作り直し=waiting化+pack_id剥がし+outputs退避)。手元の古い
    スナップショットを丸ごと保存すると作り直しが黙って巻き戻り、redo_count
    喪失で次のarchiveが前のarchiveを上書きする実害まで出る。そこで書く直前
    にGCSから読み直し、状態が今もこの作業員の所有 — generating、または
    「同じpack_idのままのpack_ready」(=取り込み前に失敗した自分の依頼) —
    のときだけ status/error を差し込んで保存する。所有が外れていたら保存を
    放棄 (受付側の新しい内容が勝ち)。依頼自体が消えていた場合も復活させ
    ない。

    (2026-07-20) 読み直し失敗時に手元の古いスナップショットを無条件保存する
    従来動作を廃止 — 所有判定を素通りして受付側のredo (waiting化) を黙って
    doneへ巻き戻す実害の方が大きい。読み直しも保存も有界リトライにし、
    それでも駄目なら generating のまま残す (ハートビート途絶→stale拾い直し
    が最大~16分で自動回収するので、終了状態の喪失はもう永久ではない)。
    リトライ中は _LAST_GCS_WORK を進めてアイドル自己停止に断ち切られない
    ようにする。"""
    rid = req["request_id"]
    req["status"] = status                    # 呼び出し側から見える手元も更新
    req["error"] = error
    cur = None
    for i in range(4):                        # 読み直し: 計~45秒粘る
        try:
            cur = _gcs_req_load(rid)
            break
        except Exception:                     # noqa: BLE001
            if i == 3:
                _wp_print(f"[GCS] 依頼 {rid} の読み直し不能 — 書き戻しを"
                          "見送り (stale拾い直しに委ねる)")
                return
            _LAST_GCS_WORK[0] = time.time()
            time.sleep(15)
    if cur is None:                           # 依頼が削除済み → 復活させない
        _wp_print(f"[GCS] 依頼 {rid} は生成中に削除された — 書き戻しを放棄")
        return
    owned = (cur.get("status") == "generating"
             or (cur.get("status") == "pack_ready"
                 and str(cur.get("pack_id") or "") == str(req.get("pack_id") or "")))
    if not owned:
        _wp_print(f"[GCS] 依頼 {rid} は生成中に {cur.get('status')} へ"
                  "変更された — 書き戻しを放棄 (受付側を優先)")
        return
    cur["status"] = status
    cur["error"] = error
    for i in range(6):                        # 保存: 計~2.5分粘る (一発勝負廃止)
        try:
            _gcs_req_save(cur)
            return
        except Exception:                     # noqa: BLE001
            if i == 5:
                raise
            _LAST_GCS_WORK[0] = time.time()
            time.sleep(30)


def _gcs_req_heartbeat(rid: str, pid: str) -> bool:
    """生成中の依頼の updated を進める (生存証明 2026-07-20)。

    これで「updated が止まった generating = 死んだ請負」が成立し、
    _GCS_STALE_GEN_SEC を数分オーダーへ短縮できる (従来7800sは
    walkpackタイムアウト基準の壁時計で、生存性を証明しなかった)。
    読み直した cur を保存するのが要点 — 手元の req を書くと受付側の
    フィールド (pw_salt等) を潰す。所有が外れていたら False (以後打たない)。"""
    cur = _gcs_req_load(rid)
    if (not cur or cur.get("status") != "generating"
            or str(cur.get("pack_id") or "") != pid):
        return False
    _gcs_req_save(cur)
    return True


_GCS_CT = {".mp4": "video/mp4", ".webp": "image/webp", ".png": "image/png",
           ".gif": "image/gif", ".json": "application/json"}


def _submit_annex_i2v(pid: str, mock: bool = False) -> str:
    """分室パックの原画+英訳文を通常AniSora I2Vジョブへ投入する。"""
    import math
    from PIL import Image
    pack = packs_root() / pid
    image_path = pack / "annex_source.png"
    prompt_path = pack / "annex_prompt_en.txt"
    if not image_path.is_file() or not prompt_path.is_file():
        raise RuntimeError("分室パックにannex_source.png/英訳文がありません")
    prompt = " ".join(prompt_path.read_text(
        encoding="utf-8", errors="replace").split())[:2000]
    if not prompt:
        raise RuntimeError("分室の英訳プロンプトが空です")
    image = Image.open(image_path).convert("RGB")
    iw, ih = image.size
    # walkpackと近い総画素数で縦横比を保ち、Wanの16倍数制約へ揃える。
    scale = math.sqrt((832 * 480) / max(1.0, float(iw * ih)))
    w = max(224, int(round(iw * scale / 16)) * 16)
    h = max(224, int(round(ih * scale / 16)) * 16)
    req = GenRequest(mode="i2v", prompt=prompt, images=[image],
                     width=w, height=h, num_frames=81, fps=16,
                     steps=8, seed=int(time.time()) % (2 ** 31),
                     guidance=1.0, extra={"motion_score": 3.0})
    return submit_job("mock" if mock else "anisora", req)


def _gcs_process_one(req: dict, wait: bool = True):
    """pack_ready の依頼を1件処理: パック取り込み→walkpack→outputs書き戻し。

    wait=False はテスト用 (walkpack投入=generating遷移までで戻る)。戻り値=jid。
    例外は呼び出し側 (_gcs_worker_loop) が failed 化する。"""
    rid = req["request_id"]
    pid = str(req.get("pack_id") or "")
    if not pid:
        raise RuntimeError("pack_idがありません")
    blob = _gcs_read(f"packs/{pid}.zip")
    if blob is None:
        raise RuntimeError(f"パックが見つかりません: {pid}")
    # 分室は8方向PNGを持たない。一枚絵パックを通常extractへ入れた時点で
    # 「8方向PNG不足」になるため、展開より前に種別を決める。requestの種別
    # フィールドが古い中継で欠けても annex_ pack_id なら救済する。
    annex = (req.get("room") == "annex"
             or req.get("request_type") == "annex_i2v"
             or pid.startswith("annex_"))
    _pack_extract(pid, blob, "annex" if annex else "walkpack")
    req["status"] = "generating"
    req["error"] = ""
    _gcs_req_save(req)
    _LAST_GCS_WORK[0] = time.time()
    jid = (_submit_annex_i2v(pid, mock=bool(req.get("_mock")))
           if annex else submit_walkpack(pid, mock=bool(req.get("_mock"))))
    if not wait:
        return jid
    beat = time.time()                           # ハートビート (120s毎)
    own = True
    while True:                                  # ジョブ完了までポーリング
        time.sleep(5)
        j = JOBS.get(jid)
        _LAST_GCS_WORK[0] = time.time()
        if own and time.time() - beat >= 120:
            beat = time.time()
            try:
                own = _gcs_req_heartbeat(rid, pid)
            except Exception:                     # noqa: BLE001
                pass                              # 欠け打ち許容 (900sは7回分超)
        if j is None:
            raise RuntimeError("ジョブが消えました")
        if j.get("status") in ("done", "error", "cancelled"):
            break
    if j.get("status") != "done":
        raise RuntimeError(str(j.get("detail") or "生成失敗")[:400])
    if annex:
        result = Path(str(j.get("path") or ""))
        if not result.is_file():
            raise RuntimeError("分室I2VのMP4が見つかりません")
        mp4_name = f"annex_{rid}.mp4"
        source = packs_root() / pid / "annex_source.png"
        _gcs_write(f"outputs/{rid}/{mp4_name}", result.read_bytes(),
                   "video/mp4")
        files = [{"name": mp4_name, "size": result.stat().st_size,
                  "kind": "mp4"}]
        if source.is_file():
            _gcs_write(f"outputs/{rid}/source.png", source.read_bytes(),
                       "image/png")
            files.append({"name": "source.png", "size": source.stat().st_size,
                          "kind": "poster"})
        manifest = {"pack_id": pid, "files": files,
                    "finished": time.time(),
                    "generation": {"engine": "anisora_i2v", "frames": 81,
                                   "fps": 16, "steps": 8,
                                   "server_version": __version__}}
        _gcs_write(f"outputs/{rid}/manifest.json",
                   json.dumps(manifest, ensure_ascii=False).encode("utf-8"),
                   "application/json")
        _gcs_req_finish(req, "done")
        _LAST_GCS_WORK[0] = time.time()
        shutil.rmtree(packs_root() / pid, ignore_errors=True)
        try:
            _gcs_prune_outputs()
        except Exception as e:                    # noqa: BLE001
            _wp_print(f"[GCS] 分室完了後の刈り込み失敗 (継続): {str(e)[:160]}")
        return jid
    # 依頼がドット絵化を求めていれば歩行後に実行 (out/ にpixelシートを追加)。
    # ベストエフォート — 失敗しても歩行成果物は返す。
    if req.get("pixelize"):
        try:
            _wp_pixelize(pid, colors=int(req.get("pixel_colors") or 24),
                         dither=float(req.get("pixel_dither", 1.0)))
        except Exception as e:                    # noqa: BLE001
            _wp_print(f"[GCS] ドット絵化に失敗 (継続): {str(e)[:160]}")
    out = packs_root() / pid / "out"             # 成果物を outputs/<rid>/ へ
    files = []
    if out.is_dir():
        for p in sorted(out.iterdir()):
            if not p.is_file():
                continue
            k = _wp_kind(p.name)
            # 生mp4 (方向別/canvasの背景付き動画) は途中成果物なのでGCSへ
            # 上げない。配布物 = シート/透明歩行webp/ポスター/ピクセル/preview。
            if k in ("other", "mp4"):
                continue
            _gcs_write(f"outputs/{rid}/{p.name}", p.read_bytes(),
                       _GCS_CT.get(p.suffix.lower(),
                                   "application/octet-stream"))
            files.append({"name": p.name, "size": p.stat().st_size, "kind": k})
    order = {"preview": 0, "mp4": 1, "sheet": 2, "pixel": 3}
    files.sort(key=lambda f: (order.get(f["kind"], 9), f["name"]))
    manifest = {"pack_id": pid, "files": files, "finished": time.time(),
                "generation": j.get("_generation_settings") or {}}
    _gcs_write(f"outputs/{rid}/manifest.json",
               json.dumps(manifest, ensure_ascii=False).encode("utf-8"),
               "application/json")
    # (2026-07-19) 丸ごと上書きせず、生成中に受付側で変更されていないか
    # 読み直してから done を書く (作り直しの黙殺防止)
    _gcs_req_finish(req, "done")
    _LAST_GCS_WORK[0] = time.time()
    try:                                      # 検証用中間の保全 (rmtree前)
        _gcs_debug_upload(rid, packs_root() / pid / "out")
    except Exception:                         # noqa: BLE001
        pass
    # 完成したら途中成果物(生mp4を含むローカルパック一式)を削除して
    # VMディスクを解放する (容量圧迫の防止)。配布物はGCSにあるので安全。
    try:
        shutil.rmtree(packs_root() / pid, ignore_errors=True)
    except Exception:                         # noqa: BLE001
        pass
    try:                                      # 保持100件超の古い生成物を削除
        _gcs_prune_outputs()
    except Exception as e:                    # noqa: BLE001
        _wp_print(f"[GCS] 刈り込みに失敗 (継続): {str(e)[:160]}")
    return jid


# (2026-07-19) generating のまま宙に浮いた依頼の再取り込み猶予。VMがジョブ中に
# 死ぬと (アイドル自己停止/プリエンプト/OOM/startup.shのpkill) 依頼は
# generating のまま誰も再試行せず、受付側の complete/fail も409で拒むため
# 永久に「生成中」で固まっていた。
# (2026-07-20) 基準を「walkpackタイムアウト(7800s)の壁時計」から「ハート
# ビート途絶」へ変更 — 生成中は _gcs_req_heartbeat が120s毎に updated を
# 進めるので、900s (7回分超の余裕) 止まっていれば死産と断定できる。
# ★受付台側の起こし直し閾値 _STALE_GEN_WAKE (kobo_front/main.py) は必ず
# この値より大きく保つこと — 逆だと起きたVMが拾えず起床スラッシングになる。
_GCS_STALE_GEN_SEC = 900.0


def _gcs_pick_pack_ready():
    """GCSの依頼から status=pack_ready を1件返す (無ければNone)。

    (2026-07-19) VM急停止で generating のまま放置された依頼も、updated が
    _GCS_STALE_GEN_SEC を超えて古ければ拾い直す (処理先頭で generating を
    保存し直すので updated が進み、二重取り込みにはならない)。"""
    for name in _gcs_list("requests/"):
        if not name.endswith(".json"):
            continue
        rid = name[len("requests/"):-len(".json")]
        req = _gcs_req_load(rid)
        if not req:
            continue
        if req.get("status") == "pack_ready":
            return req
        if (req.get("status") == "generating"
                and time.time() - float(req.get("updated") or 0)
                > _GCS_STALE_GEN_SEC):
            _wp_print(f"[GCS] 宙に浮いたgenerating依頼を再取り込み: {rid} "
                      f"(updated停止から{_GCS_STALE_GEN_SEC:.0f}s超)")
            return req
    return None


def _gcs_worker_loop() -> None:
    _wp_print(f"[GCS] ワーカー開始 (bucket={GCS_BUCKET or _GCS_FAKE}, "
              f"poll={_GCS_POLL_SEC}s)")
    while True:
        try:
            if _GCS_STOPPING[0]:                  # 停止決定後は新規請負なし
                return
            req = _gcs_pick_pack_ready()
            if req:
                rid = req["request_id"]
                # 取り込み窓 (pack DL/展開〜generating書込) もアイドル時計を
                # 進める — 大きいzipのDL中に猶予満了→電源断の競合防止
                _LAST_GCS_WORK[0] = time.time()
                _GCS_INFLIGHT[0] = (rid, str(req.get("pack_id") or ""))
                _wp_print(f"[GCS] 取り込み {rid} pack={req.get('pack_id')}")
                try:
                    _gcs_process_one(req)
                    _wp_print(f"[GCS] 完了 {rid}")
                except Exception as e:            # noqa: BLE001
                    _wp_print(f"[GCS] 失敗 {rid}: {str(e)[:200]}")
                    try:
                        # (2026-07-19) done側と同じく読み直してから failed を
                        # 書く (生成中の作り直しを古いスナップショットで潰さない)
                        _gcs_req_finish(req, "failed", str(e)[:500])
                    except Exception:             # noqa: BLE001
                        pass
                finally:
                    _GCS_INFLIGHT[0] = None
                continue                          # 連続処理: すぐ次を探す
        except Exception as e:                    # noqa: BLE001
            _wp_print(f"[GCS] ポーリングエラー (継続): {str(e)[:160]}")
        time.sleep(_GCS_POLL_SEC)


def _gcs_release_inflight() -> None:
    """請負中の依頼を pack_ready へ戻してから死ぬ (2026-07-20)。

    uvicorn は SIGTERM 受信→graceful shutdown→FastAPIのshutdownイベント、
    の順で確実にここへ来る。gcloud stop (猶予~90s)・Spotプリエンプト (30s)・
    startup.sh の pkill・自己停止poweroff→systemdのTERM、の全経路が対象。
    守れないのは SIGKILL/OOM-kill だけで、そこはハートビート途絶→stale
    拾い直し (最大~16分) がバックストップする。

    二重解放は無害: 解放が pack_ready を書いた直後に走行中ワーカーが done を
    書く順序でも、_gcs_req_finish の所有判定が「同pack_idのpack_ready」を
    所有扱いするため done が勝つ。逆順は所有外で解放がskipされる。"""
    _GCS_STOPPING[0] = True                   # 先に新規請負を止める
    snap = _GCS_INFLIGHT[0]
    if not snap or not _gcs_active():
        return
    rid, pid = snap
    try:
        cur = _gcs_req_load(rid)
        if (cur and cur.get("status") == "generating"
                and str(cur.get("pack_id") or "") == pid):
            cur["status"] = "pack_ready"
            cur["error"] = ""
            _gcs_req_save(cur)
            _wp_print(f"[GCS] 停止前に請負を解放: {rid} → pack_ready")
    except Exception as e:                    # noqa: BLE001
        _wp_print(f"[GCS] 請負解放に失敗 (stale拾い直しへ委任): {str(e)[:120]}")


def _gce_is_idle() -> bool:
    """稼働中ジョブなし・キュー空・walkpack非実行なら暇。"""
    for j in list(JOBS.values()):
        if j.get("status") in ("queued", "running", "loading"):
            return False
    if WORK_Q.qsize() > 0:
        return False
    if _WALKPACK_LOCK.locked():
        return False
    return True


def _gce_self_stop() -> None:
    """vm/state.json を stopping にしてから電源断 (課金の守り)。

    VIDEOLAB_GCE=1 のときだけ実機を poweroff する。それ以外の環境
    (Colab/ローカル/テスト) では絶対に電源を切らず _shutdown_runtime へ。"""
    _GCS_STOPPING[0] = True   # 停止決定〜電源断の窓で新規請負しない (TOCTOU防止)
    try:
        if _gcs_active():
            _gcs_write("vm/state.json",
                       json.dumps({"status": "stopping",
                                   "updated": time.time()}).encode("utf-8"),
                       "application/json")
    except Exception:                             # noqa: BLE001
        pass
    if os.environ.get("VIDEOLAB_GCE", "").strip() == "1":
        _wp_print("[IDLE] アイドル継続 — 電源を切ります (poweroff)")
        try:
            subprocess.run(["poweroff"], timeout=30)
        except Exception:                         # noqa: BLE001
            try:
                subprocess.run(["shutdown", "-h", "now"], timeout=30)
            except Exception:                     # noqa: BLE001
                pass
    else:
        _wp_print("[IDLE] アイドル継続 — ランタイムを解放します")
        _shutdown_runtime(delay=0.5)


def _idle_stop_loop(minutes: int) -> None:
    grace = minutes * 60.0
    _wp_print(f"[IDLE] 自己停止監視 開始 (アイドル{minutes}分で停止)")
    while True:
        time.sleep(30)
        try:
            if not _gce_is_idle():
                # (2026-07-19) ジョブ実行中はアイドル時計も進める。以前は
                # 停止判定を先送りするだけで時計が止まったままだったため、
                # 猶予より長いジョブ (直HTTP運転のwalkpack等) が終わった
                # 30秒後に即電源断し、成果物DL前にVMが落ちていた。
                # 猶予はマシンが実際に暇になった時点から数え始める。
                _LAST_GCS_WORK[0] = time.time()
                continue
            idle_for = time.time() - max(_LAST_HTTP[0], _LAST_GCS_WORK[0])
            if idle_for >= grace:
                _gce_self_stop()
                return
        except Exception as e:                    # noqa: BLE001
            _wp_print(f"[IDLE] 監視エラー (継続): {str(e)[:120]}")


def _gce_external_ip() -> str:
    try:
        return _meta_get(
            "instance/network-interfaces/0/access-configs/0/external-ip").strip()
    except Exception:                             # noqa: BLE001
        return ""


def _gce_publish_state(url: str, token: str) -> None:
    if not _gcs_active():
        return
    try:
        _gcs_write("vm/state.json",
                   json.dumps({"status": "up", "url": url, "token": token,
                               "updated": time.time()},
                              ensure_ascii=False).encode("utf-8"),
                   "application/json")
    except Exception as e:                        # noqa: BLE001
        _wp_print(f"[GCS] state公開に失敗: {str(e)[:120]}")


def start_gce_workers(url: str = "", token: str = "",
                      port: int | None = None) -> bool:
    """GCSワーカー + アイドル自己停止スレッドを起動 (多重起動防止)。

    GCSもアイドル停止も無効なら何もしない (Colab/ローカルは素通り)。
    起動したら True。
    (2026-07-19) url未指定時の公開URLは port引数 (実際の待受ポート) で組む。
    以前はDEFAULT_PORT固定だったため、--port変更時に健全なサーバなのに
    到達不能URLを配ってしまい、ネットワーク障害にしか見えなかった。"""
    if _GCE_THREADS_UP[0]:
        return True
    if not (_gcs_active() or _idle_min() > 0):
        return False
    _GCE_THREADS_UP[0] = True
    now = time.time()
    _LAST_HTTP[0] = now
    _LAST_GCS_WORK[0] = now
    if not url:
        ip = _gce_external_ip()
        if ip:
            url = f"http://{ip}:{port or DEFAULT_PORT}"
    _gce_publish_state(url, token)
    if _gcs_active():
        threading.Thread(target=_gcs_worker_loop, daemon=True).start()
    if _idle_min() > 0:
        threading.Thread(target=_idle_stop_loop, args=(_idle_min(),),
                         daemon=True).start()
    return True


def run_in_gce(host: str = "0.0.0.0", port: int = None, token: str = None):
    """GCE VM上でサーバを起動しGCS作業員スレッドを回す ({url, token} を返す)。

    トークンは 引数 > /mnt/models/token.txt > VIDEOLAB_TOKEN の順。
    startup.sh からは `python -m videolab_server --host 0.0.0.0 ...` で
    main() 経由に入ってもよい (main() も start_gce_workers を呼ぶ)。"""
    port = port or DEFAULT_PORT
    if not token:
        tf = Path("/mnt/models/token.txt")
        if tf.is_file():
            token = tf.read_text(encoding="utf-8").strip()
        token = token or os.environ.get("VIDEOLAB_TOKEN") or None
    server = start_server(host, port, token)
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(60):
        try:
            with _url_req.urlopen(f"http://127.0.0.1:{port}/health",
                                  timeout=2) as r:
                if r.status == 200:
                    break
        except Exception:                         # noqa: BLE001
            time.sleep(1)
    ip = _gce_external_ip()
    url = f"http://{ip}:{port}" if ip else f"http://127.0.0.1:{port}"
    start_gce_workers(url, token or "")
    _wp_print(f"SpriteMill VideoLab (GCE) 起動: {url}")
    return url, token


# ---------------------------------------------------------------- FastAPI
def build_app(token: str | None):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, HTMLResponse, Response

    app = FastAPI(title="SpriteMill VideoLab", version=__version__)

    # SIGTERM→uvicorn graceful→ここ、の順で必ず呼ばれる (signal.signalは
    # uvicornが上書きするので使えない)。請負中依頼の pack_ready 戻し。
    # ★app.add_event_handler は VMの古いFastAPIに存在しない (AttributeError
    # でサーバ即死、2026-07-20実障害)。on_event デコレータは新旧両方にある。
    @app.on_event("shutdown")
    def _release_on_shutdown():
        _gcs_release_inflight()

    @app.middleware("http")
    async def _track_access(request: Request, call_next):
        # アイドル自己停止の「最後の人間アクセス」時刻。/health は監視の
        # 定期ノックなので数えない (寝坊防止より正しくアイドル判定するため)
        if request.url.path != "/health":
            _LAST_HTTP[0] = time.time()
        return await call_next(request)

    def _auth(request: Request):
        if not token:
            return
        got = (request.headers.get("authorization", "").removeprefix("Bearer ").strip()
               or request.headers.get("x-token", "")
               or request.query_params.get("token", ""))
        if got != token:
            raise HTTPException(401, "bad token")

    @app.get("/", response_class=HTMLResponse)
    def index():
        # v0.10.0: トップはお友だち用の工房ページ。旧UIは /advanced へ
        return KOBO_HTML

    @app.get("/advanced", response_class=HTMLResponse)
    def advanced():
        return INDEX_HTML

    @app.get("/health")
    def health():
        gpu = None
        try:
            import torch
            if torch.cuda.is_available():
                p = torch.cuda.get_device_properties(0)
                free, total = torch.cuda.mem_get_info()
                gpu = {"name": p.name,
                       "vram_gb": round(total / 2**30, 1),
                       "free_gb": round(free / 2**30, 1)}
        except Exception:
            pass
        disk = None
        try:
            u = shutil.disk_usage(Path.home())
            disk = {"total_gb": round(u.total / 2**30), "free_gb": round(u.free / 2**30)}
        except Exception:
            pass
        # ライブラリ版数 (遠隔デバッグ用。metadata参照なのでimport不要で軽い)
        libs = {}
        try:
            from importlib.metadata import version as _pkgver
            for pkg in ("torch", "diffusers", "transformers"):
                try:
                    libs[pkg] = _pkgver(pkg)
                except Exception:
                    libs[pkg] = None
        except Exception:
            pass
        # Drive固定運転の状態 (v0.9.0: ⚡自動運転がマウント承諾待ちを
        # ここで判定する — DOM文字列より確実)
        drv = {"only": _drive_only(),
               "mounted": _drive_cache_dir() is not None, "ready": False}
        if drv["mounted"]:
            try:
                drv["ready"] = bool(drive_cache_ready()[0])
            except Exception:
                pass
        # 実行中ジョブの鼓動 (P1): クライアントが「生成ハング」と「単に
        # 遅い」を区別できる。admission係数の公開はクライアントの
        # plan_canvasと見積もり式を単一ソース化するため (P0-3)
        job = None
        for _j in list(JOBS.values()):   # 並行submitとの競合防止 (v0.9.13)
            if _j.get("status") in ("running", "loading"):
                _b = _j.get("_beat") or _j.get("started") or _j.get("created")
                job = {"id": _j["id"], "status": _j["status"],
                       "beat_age_sec": round(time.time() - float(_b or
                                                                 time.time()))}
                break
        ram = None
        _avail = _avail_ram_gb()
        if _avail >= 0:
            ram = {"available_gb": round(_avail, 1)}
        return {"ok": True, "app": "SpriteMill VideoLab", "version": __version__,
                "auth": bool(token), "queued": WORK_Q.qsize(),
                "current_model": CURRENT_MODEL, "gpu": gpu, "disk": disk,
                "drive": drv, "libs": libs, "job": job, "ram": ram,
                "worker_stalled": _WORKER_STALLED,
                "admission": {"act_gb_per_lat_frame_720x1296": ACT_GB_PER_LAT,
                              "safety_gb": ACT_SAFETY_GB}}

    @app.get("/api/models")
    def models(request: Request):
        _auth(request)
        return {"models": [a.info() for a in ADAPTERS.values()],
                "current": CURRENT_MODEL}

    @app.post("/api/shutdown")
    def api_shutdown(request: Request):
        """ランタイム解放 (Colab=unassignでVM削除 / ローカル=プロセス終了)。

        SpriteMill本体がアプリ終了時・内蔵ブラウザが閉じられたときに叩く
        (2026-07-13要望)。認証必須。応答を返してから2秒後に実行。"""
        _auth(request)
        threading.Thread(target=_shutdown_runtime, daemon=True).start()
        return {"ok": True, "detail": "runtime shutdown scheduled"}

    @app.post("/api/generate")
    async def api_generate(request: Request):
        _auth(request)
        body = await request.json()
        model = body.get("model", "mock")
        if model not in ADAPTERS:
            raise HTTPException(400, f"unknown model: {model}")
        mode = body.get("mode", "i2v")
        images = load_images_b64(body.get("images_b64", []))
        if mode in ("i2v", "multikey", "i2i") and not images:
            # i2iも必須: 欠けたまま通すと黙ってt2i挙動になり、i2iの目的
            # (下地の背景・配置維持) が視認でしか気づけない形で消える
            raise HTTPException(400, "画像がありません (images_b64)")
        req = GenRequest(
            mode=mode, prompt=body.get("prompt", ""),
            negative=body.get("negative", ""), images=images,
            key_positions=[float(x) for x in body.get("key_positions", [])],
            width=int(body.get("width", 768)), height=int(body.get("height", 512)),
            num_frames=int(body.get("num_frames", 97)),
            fps=int(body.get("fps", 24)), steps=int(body.get("steps", 30)),
            seed=int(body.get("seed", 0)) or int(time.time()) % 2**31,
            guidance=float(body.get("guidance", 3.0)),
            extra=body.get("extra", {}) or {})
        return {"job": submit_job(
            model, req, watch_poll=bool(body.get("cancel_if_unpolled")))}

    # ---- SpriteMill canvas_walk.py 互換契約 (旧FramePackサーバと同一) ----
    @app.post("/generate_multikey")
    async def generate_multikey(request: Request):
        _auth(request)
        body = await request.json()
        images = load_images_b64(body.get("keys_b64", []))
        if not images:
            raise HTTPException(400, "keys_b64 required")
        # モデル未選択時: GPUがあれば既定モデル、なければ mock(疎通テスト)
        d = ADAPTERS.get(CURRENT_MODEL or "")
        if d is None:
            has_gpu = False
            try:
                import torch
                has_gpu = torch.cuda.is_available()
            except Exception:
                pass
            d = (next((a for a in ADAPTERS.values() if a.id != "mock"), None)
                 if has_gpu else None) or ADAPTERS["mock"]
        base = dict(d.defaults)
        w, h = images[0].size
        req = GenRequest(
            mode="multikey", prompt=body.get("prompt", ""), images=images,
            width=int(body.get("width", w)), height=int(body.get("height", h)),
            num_frames=int(body.get("num_frames", base.get("num_frames", 97))),
            fps=int(body.get("fps", base.get("fps", 24))),
            steps=int(body.get("steps", base.get("steps", 30))),
            seed=int(body.get("seed", 0)) or int(time.time()) % 2**31,
            guidance=float(body.get("guidance", base.get("guidance", 3.0))))
        return {"job": submit_job(d.id, req)}

    @app.get("/status/{jid}")
    def status(jid: str, request: Request):
        _auth(request)
        j = JOBS.get(jid)
        if not j:
            return {"status": "error", "detail": "unknown job"}
        j["_last_poll"] = time.time()   # 孤児検知のハートビート
        return _job_public(j)

    @app.get("/result/{jid}")
    def result(jid: str, request: Request):
        _auth(request)
        j = JOBS.get(jid)
        if not j or j.get("status") != "done" or not j.get("path"):
            raise HTTPException(404, "not ready")
        # 拡張子でメディアタイプを推定 (v0.9.5: illustriousはPNGを返す)
        _p = Path(j["path"])
        _mt = ("image/png" if _p.suffix.lower() == ".png"
               else "video/mp4")
        return FileResponse(str(_p), media_type=_mt,
                            filename=f"videolab_{jid}{_p.suffix}")

    @app.get("/api/jobs")
    def jobs(request: Request):
        _auth(request)
        return {"jobs": [_job_public(JOBS[i]) for i in reversed(JOB_ORDER)][:50]}

    @app.post("/api/cancel/{jid}")
    def cancel(jid: str, request: Request):
        _auth(request)
        j = JOBS.get(jid)
        if not j:
            raise HTTPException(404, "unknown job")
        j["_cancel"] = True
        if j["status"] == "queued":
            j["status"] = "cancelled"
        return {"ok": True}

    # ---- 工房モード: キャラパック + walk_pack (v0.10.0) ----
    def _require_pid(pid: str) -> Path:
        if not _WP_PID_RE.match(pid or ""):
            raise HTTPException(400, "pack_idが不正です (英数と._-のみ)")
        return packs_root() / pid

    @app.post("/api/packs/upload")
    async def packs_upload(request: Request):
        """母艦アプリがキャラパックzipを置く。既存同名は置換。"""
        _auth(request)
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, "JSONボディが必要です")
        pid = str(body.get("pack_id") or "").strip()
        _require_pid(pid)
        # 生成中のパックを置換するとrmtreeで実行中ジョブが壊れる
        for j0 in list(JOBS.values()):
            if (j0.get("model") == "walkpack" and j0.get("pack") == pid
                    and j0.get("status") in ("queued", "running")):
                raise HTTPException(
                    409, "このパックは歩行生成の実行中です — 完了/中止後に"
                         "アップロードしてください")
        try:
            raw = base64.b64decode(str(body.get("zip_b64") or ""))
        except Exception:
            raise HTTPException(400, "zip_b64をデコードできません")
        if not raw:
            raise HTTPException(400, "zip_b64が空です")
        if len(raw) > 300 * 2**20:
            raise HTTPException(400, "zipが大きすぎます (>300MB)")
        try:
            n = _pack_extract(pid, raw)
        except zipfile.BadZipFile:
            raise HTTPException(400, "zipとして読めません")
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {"ok": True, "pack_id": pid, "files": n}

    @app.get("/api/packs")
    def packs_list(request: Request):
        _auth(request)
        items = []
        root = packs_root()
        if root.is_dir():
            for pd in sorted(root.iterdir()):
                if not pd.is_dir() or not _WP_PID_RE.match(pd.name):
                    continue
                _, refs = _pack_refs_dir(pd)
                if not refs:
                    continue
                meta = _pack_meta(pd)
                try:
                    created = float(meta.get("created") or 0) \
                        or pd.stat().st_mtime
                except (TypeError, ValueError, OSError):
                    created = 0.0
                items.append({
                    "pack_id": pd.name,
                    "name": str(meta.get("name") or pd.name),
                    "has_landmarks": (pd / "01_generation"
                                      / "landmarks.json").is_file(),
                    "created": created})
        items.sort(key=lambda x: x["created"], reverse=True)
        return items

    @app.get("/api/packs/{pid}/thumb.png")
    def pack_thumb(pid: str, request: Request):
        _auth(request)
        pack = _require_pid(pid)
        _, refs = _pack_refs_dir(pack)
        fr = refs.get("front") or next(iter(refs.values()), None)
        if fr is None:
            raise HTTPException(404, "unknown pack")
        from PIL import Image
        im = Image.open(fr).convert("RGBA")
        th = 200
        tw = max(1, round(im.width * th / im.height))
        im = im.resize((tw, th), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")

    @app.post("/api/walkpack/{pid}")
    async def walkpack_start(pid: str, request: Request):
        """歩行生成ジョブ投入 (疑似ジョブ model=walkpack を返す)。

        body {"mock": true} でモックアダプタ経路 (テスト用)。GPUの無い
        サーバでは mock 以外を拒否する — 拒否しないと「ボタン1つで
        20GB級モデルのDLがCPUマシンに始まる」事故になる (2026-07-18
        ローカルスモークで実発生、部分DL10GBを掃除した実害)。"""
        _auth(request)
        pack = _require_pid(pid)
        try:
            _engine()
        except RuntimeError as e:
            raise HTTPException(503, str(e))
        if not pack.is_dir():
            raise HTTPException(404, "unknown pack")
        try:
            body = await request.json()
        except Exception:                 # noqa: BLE001
            body = {}
        mock = bool((body or {}).get("mock"))
        # 実験ノブ (2026-07-19ユーザー発案「VACEはキーフレームの錨・中間は
        # AniSoraに生成させる」): pin_conf="0,12,24,36,48"等の明示固定リスト
        # + refine=σ をジョブ単位で上書きできる。省略時は本線の既定のまま
        exp = {}
        _pc = str((body or {}).get("pin_conf") or "").strip()
        if _pc:
            exp["pin_conf"] = _pc[:200]
        try:
            if (body or {}).get("refine") is not None:
                exp["refine"] = max(0.05, min(1.0,
                                              float(body["refine"])))
        except (TypeError, ValueError):
            pass
        if (body or {}).get("edge_idle"):
            # 実験a (2026-07-19): idle/末尾静止の制御を立ち絵Cannyへ差し替え
            exp["edge_idle"] = True
        if str((body or {}).get("layout") or "").strip() in ("hemi", "4x2"):
            exp["layout"] = "hemi"     # 半球へ明示固定 (比較実験用)
        if str((body or {}).get("layout") or "").strip() == "compass":
            # 実験 (2026-07-20ユーザー仮説「半球配置の隣接汚染」— B4は
            # backの隣にrightという角度不連続な同居で、尻尾がrightへ染み
            # 翼がbackへ吸われる疑い): 3x3コンパス配置 (隣=隣接角度) で
            # 同条件生成して比較する
            exp["layout"] = "compass"
        if str((body or {}).get("layout") or "").strip() in (
                "single", "individual", "8dir"):
            exp["layout"] = "single"
        if "free_idle" in (body or {}):
            exp["free_idle"] = bool(body["free_idle"])
        if (body or {}).get("edge_face"):
            # 顔エッジv2の単発検証用 (2026-07-20): 既定offのままジョブ単位で
            # noeyes骨格+頭部キャニーを有効化できる
            exp["edge_face"] = True
        if (body or {}).get("depth_move"):
            exp["depth_move"] = True   # 実験b: 実測深度+手続き運動 (非二足)
        if (body or {}).get("line_move"):
            exp["line_move"] = True    # 実験b: 実測線画+手続き運動 (非二足)
        if (body or {}).get("line_puppet"):
            exp["line_puppet"] = True  # 実験c: 骨駆動の線画パペット (二足)
        if (body or {}).get("scribble_mix"):
            exp["scribble_mix"] = True  # 実験d: 頭=線画+体=棒人間スクリブル
        if (body or {}).get("key_interp"):
            exp["key_interp"] = True   # 実験e: 実画像アンカー+空潜在の補間
        try:
            if (body or {}).get("pose_every") is not None:
                # 実験f: 純制御モードのまま骨格をNフレームごとに間引き
                # (間=黒=制御なし)。free_idleの歩行窓拡張=中割をVACEに委ねる
                exp["pose_every"] = max(2, min(12, int(body["pose_every"])))
        except (TypeError, ValueError):
            pass
        if (body or {}).get("face_line"):
            # 実験g3 (2026-07-21ユーザー発案「顔だけ毎フレーム線画で出し、
            # 3フレームに1回全身線画」): 非人型の間引きギャップを黒でなく
            # 顔限定線画にする。顔=見た目の権威を毎フレーム守り、体の
            # 中割だけVACEに委ねる。要 pack/01_generation/face_boxes.json
            exp["face_line"] = True
        if (body or {}).get("legs_only"):
            # 実験h (2026-07-21「走る忍者」): 骨格=脚のみ。参照が走り姿勢
            # の依頼で直立歩行の上半身骨格が参照と全面矛盾する対策
            exp["legs_only"] = True
        if (body or {}).get("gait_run"):
            exp["gait_run"] = True     # 実験h: 文面を走りサイクル宣言に
        try:
            if (body or {}).get("region_mask") is not None:
                # 実験r: キャラ下部fracを空の潜在にして部分生成
                exp["region_mask"] = max(
                    0.2, min(0.8, float(body["region_mask"])))
        except (TypeError, ValueError):
            pass
        if (body or {}).get("procedural"):
            exp["procedural"] = True   # 実験p: 生成なしの手続きアニメ
        if (body or {}).get("inpaint_mode") in ("face", "bottom"):
            exp["inpaint_mode"] = body["inpaint_mode"]
        try:
            if (body or {}).get("legs_mask") is not None:
                # 実験r3: 上=実画素凍結+下=脚骨格入り生成 (VACEマスク同居)
                exp["legs_mask"] = max(
                    0.2, min(0.8, float(body["legs_mask"])))
        except (TypeError, ValueError):
            pass
        try:
            if (body or {}).get("anisora_inpaint") is not None:
                # 実験r2: 手続き土台+空間latent固定 (VACE抜き)
                exp["anisora_inpaint"] = max(
                    0.2, min(0.8, float(body["anisora_inpaint"])))
        except (TypeError, ValueError):
            pass
        try:
            if (body or {}).get("key_interp_pose") is not None:
                # 実験e2: N フレームに1回だけ骨格ドットの道しるべを置く
                exp["key_interp_pose"] = max(
                    2, min(30, int(body["key_interp_pose"])))
        except (TypeError, ValueError):
            pass
        # (旧: free_idle強制の elif False 実験ブロックは撤去 — free_idleは
        #  0.10.19で既定昇格済みのため死にコードだった)
        if (body or {}).get("edge_head"):
            # 実験a-4: 動きの窓に頭部限定エッジ (ボブ追従) を敷く
            exp["edge_head"] = True
        if (body or {}).get("no_pose"):
            # 実験a-3 (2026-07-19ユーザー仮説「骨格が人間じゃないから
            # オープンポーズが邪魔」): 歩行窓の制御を黒=無条件にして、
            # 人型骨格の強制もモダリティ混載も両方消す (edge_idleと併用)
            exp["no_pose"] = True
        if (body or {}).get("skip_vace"):
            # 実験 (2026-07-19ユーザー発案): VACEを通さず、骨格フレーム列
            # そのものをlatent源にしてAniSoraへ渡す (入力画像=参照キャンバス、
            # キー=骨格のlatent固定)。見え方の観察用
            exp["skip_vace"] = True
        if not mock:
            has_gpu = False
            try:
                import torch
                has_gpu = torch.cuda.is_available()
            except Exception:             # noqa: BLE001
                has_gpu = False
            if not has_gpu:
                raise HTTPException(
                    503, "GPUがありません — 歩行生成はGPUサーバ (Colab等) "
                         "で実行してください (テストは {\"mock\": true})")
        _, refs = _pack_refs_dir(pack)
        missing = [d for d in _WP_DIRS if d not in refs]
        if missing:
            raise HTTPException(400, f"8方向PNGが不足しています: {missing}")
        return {"job": submit_walkpack(pid, mock=mock, exp=exp)}

    @app.get("/api/walkpack/{pid}/files")
    def walkpack_files(pid: str, request: Request):
        _auth(request)
        pack = _require_pid(pid)
        out = pack / "out"
        files = []
        if out.is_dir():
            for p in sorted(out.iterdir()):
                if not p.is_file():
                    continue
                k = _wp_kind(p.name)
                if k != "other":
                    files.append({"name": p.name, "kind": k})
        order = {"preview": 0, "mp4": 1, "sheet": 2, "pixel": 3}
        files.sort(key=lambda f: (order.get(f["kind"], 9), f["name"]))
        return files

    @app.get("/api/walkpack/{pid}/file/{name}")
    def walkpack_file(pid: str, name: str, request: Request):
        _auth(request)
        pack = _require_pid(pid)
        if not _WP_FNAME_RE.match(name or "") or ".." in name:
            raise HTTPException(400, "不正なファイル名です")
        out = pack / "out"
        p = out / name
        # out/ 直下のみ (名前検証+resolveで親ディレクトリ一致を確認)
        if not p.is_file() or p.resolve().parent != out.resolve():
            raise HTTPException(404, "not found")
        mt = {".mp4": "video/mp4", ".webp": "image/webp",
              ".png": "image/png"}.get(p.suffix.lower(),
                                       "application/octet-stream")
        return FileResponse(str(p), media_type=mt, filename=name)

    @app.post("/api/walkpack/{pid}/pixelize")
    async def walkpack_pixelize(pid: str, request: Request):
        """シートのドット絵化 (同期・数秒)。body.target="preview" なら
        preview.webp (と簡易シート) の再生成。"""
        _auth(request)
        _require_pid(pid)
        try:
            _engine()
        except RuntimeError as e:
            raise HTTPException(503, str(e))
        try:
            body = await request.json()
        except Exception:
            body = {}
        body = body if isinstance(body, dict) else {}
        target = str(body.get("target") or "pixel")
        try:
            if target == "preview":
                files = _wp_regen_preview(pid)
            else:
                try:
                    colors = int(body.get("colors", 24))
                except (TypeError, ValueError):
                    colors = 24
                try:                      # ディザの強さ (0=ベタ塗り..1=フル)
                    dither = float(body.get("dither", 1.0))
                except (TypeError, ValueError):
                    dither = 1.0
                files = _wp_pixelize(pid, colors=colors, dither=dither)
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        return {"ok": True, "files": files}

    @app.get("/api/walkpack/{pid}/download")
    def walkpack_download(pid: str, request: Request):
        _auth(request)
        pack = _require_pid(pid)
        out = pack / "out"
        names = [p for p in sorted(out.iterdir())
                 if p.is_file()] if out.is_dir() else []
        if not names:
            raise HTTPException(404, "生成結果がまだありません")
        WORK_ROOT.mkdir(parents=True, exist_ok=True)
        dest = WORK_ROOT / f"walkpack_{pid}.zip"
        with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in names:
                zf.write(p, p.name)
        return FileResponse(str(dest), media_type="application/zip",
                            filename=f"{pid}_walkpack.zip")

    # ---- 依頼リレー: webUIの生成依頼 → 母艦がclaim → パック返送 (v0.10.1) ----
    def _require_req(rid: str) -> dict:
        if not _REQ_RID_RE.match(rid or ""):
            raise HTTPException(400, "request_idが不正です")
        req = _req_load(rid)
        if req is None:
            raise HTTPException(404, "unknown request")
        return req

    @app.post("/api/requests")
    async def request_create(request: Request):
        """お友だちの生成依頼 (キャラ名+コンセプト+参考画像)。

        依頼は requests/<rid>/request.json に保管され、母艦 (Codexのある
        ユーザーPC) の常駐リレーが claim → 立ち絵生成 → packs_upload →
        complete で返す。お友だち体験はアプリのキュー投入と同等。"""
        _auth(request)
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, "JSONボディが必要です")
        if not isinstance(body, dict):
            raise HTTPException(400, "JSONオブジェクトが必要です")
        name = str(body.get("name") or "").strip()
        if not name or len(name) > 40:
            raise HTTPException(400, "nameは1〜40文字で指定してください")
        concept = str(body.get("concept") or "").strip()
        if len(concept) > 500:
            raise HTTPException(400, "conceptは500文字までです")
        leg_scale = body.get("leg_scale")
        if leg_scale is not None:
            try:
                leg_scale = float(leg_scale)
            except (TypeError, ValueError):
                raise HTTPException(400, "leg_scaleは数値で指定してください")
            if not (0.6 <= leg_scale <= 4.0):
                raise HTTPException(400, "leg_scaleは0.6〜4.0の範囲です")
        ref_im = None
        b64 = body.get("image_b64")
        if b64:
            s = str(b64)
            if "," in s[:80] and s.lstrip().startswith("data:"):
                s = s.split(",", 1)[1]     # data URL 形式を許容
            try:
                raw = base64.b64decode(s)
            except Exception:
                raise HTTPException(400, "image_b64をデコードできません")
            if len(raw) > 20 * 2**20:
                raise HTTPException(400, "参考画像が大きすぎます (20MBまで)")
            from PIL import Image
            try:
                ref_im = Image.open(io.BytesIO(raw))
                if (ref_im.format or "").upper() not in ("PNG", "JPEG"):
                    raise ValueError(ref_im.format)
                ref_im.load()
            except Exception:
                raise HTTPException(400, "参考画像はPNGかJPEGにしてください")
            if ref_im.mode not in ("RGB", "RGBA"):
                ref_im = ref_im.convert("RGBA")   # CMYK等はPNGへ保存不可
        rid = uuid.uuid4().hex[:12]
        rd = requests_root() / rid
        try:
            rd.mkdir(parents=True, exist_ok=True)
            if ref_im is not None:
                ref_im.save(rd / "ref.png", format="PNG")
            _req_save(rid, {
                "request_id": rid, "name": name, "concept": concept,
                "leg_scale": leg_scale, "has_ref": ref_im is not None,
                "status": "waiting", "detail": "", "pack_id": None,
                "walk_job": None, "created": time.time(),
                "claimed_at": None})
        except Exception as e:            # noqa: BLE001
            shutil.rmtree(rd, ignore_errors=True)
            raise HTTPException(500, f"依頼を保存できません: {e}")
        return {"request_id": rid}

    @app.get("/api/requests")
    def requests_list(request: Request):
        _auth(request)
        items = []
        root = requests_root()
        if root.is_dir():
            for rdir in root.iterdir():
                if not rdir.is_dir() or not _REQ_RID_RE.match(rdir.name):
                    continue
                r = _req_load(rdir.name)
                if r is None:
                    continue          # 書き込み途中/壊れた依頼は一覧に出さない
                items.append({k: r.get(k) for k in (
                    "request_id", "name", "status", "detail", "pack_id",
                    "created", "claimed_at")})
        try:
            items.sort(key=lambda x: float(x.get("created") or 0),
                       reverse=True)
        except (TypeError, ValueError):
            pass
        return items

    @app.get("/api/requests/{rid}/ref.png")
    def request_ref(rid: str, request: Request):
        _auth(request)
        _require_req(rid)
        p = requests_root() / rid / "ref.png"
        if not p.is_file():
            raise HTTPException(404, "参考画像はありません")
        return FileResponse(str(p), media_type="image/png",
                            filename="ref.png")

    @app.post("/api/requests/{rid}/claim")
    def request_claim(rid: str, request: Request):
        """母艦が依頼を取得。waiting→claimed (claimed_at記録)。claimedでも
        10分無応答なら再claim可 (母艦クラッシュ対応)。応答は request.json
        の中身一式 (concept / leg_scale / has_ref を含む)。"""
        _auth(request)
        with _REQ_LOCK:
            req = _require_req(rid)
            st = req.get("status")
            stale = (st == "claimed"
                     and time.time() - float(req.get("claimed_at") or 0)
                     > _REQ_CLAIM_TIMEOUT)
            if st != "waiting" and not stale:
                raise HTTPException(409, f"claimできない状態です: {st}")
            req["status"] = "claimed"
            req["claimed_at"] = time.time()
            req["detail"] = ""
            _req_save(rid, req)
        return req

    @app.post("/api/requests/{rid}/complete")
    async def request_complete(rid: str, request: Request):
        """母艦がパック返送 (packs_upload) 後に叩く。パック実在確認 →
        pack_ready にし、GPUがあれば歩行生成 (walkpack) を自動投入して
        そのjidを記録する (GPU/エンジン無しでも pack_ready にはする)。"""
        _auth(request)
        try:
            body = await request.json()
        except Exception:
            body = {}
        pid = str((body or {}).get("pack_id") or "").strip()
        if not _WP_PID_RE.match(pid):
            raise HTTPException(400, "pack_idが不正です (英数と._-のみ)")
        pack = packs_root() / pid
        if not pack.is_dir():
            raise HTTPException(404, f"パックがありません: {pid}")
        with _REQ_LOCK:
            req = _require_req(rid)
            if (req.get("status") == "pack_ready"
                    and req.get("pack_id") == pid):
                # 冪等: 母艦のリトライで歩行ジョブを二重投入しない
                return {"ok": True, "pack_id": pid,
                        "walk_job": req.get("walk_job")}
            jid = None
            try:
                _engine()                 # エンジン未配備なら投入しない
                import torch
                if torch.cuda.is_available():
                    _, refs = _pack_refs_dir(pack)
                    if all(d in refs for d in _WP_DIRS):
                        jid = submit_walkpack(pid)
            except Exception:             # noqa: BLE001
                jid = None                # GPU無しでも pack_ready にはする
            req["status"] = "pack_ready"
            req["pack_id"] = pid
            req["walk_job"] = jid
            req["detail"] = ""
            _req_save(rid, req)
        return {"ok": True, "pack_id": pid, "walk_job": jid}

    @app.post("/api/requests/{rid}/fail")
    async def request_fail(rid: str, request: Request):
        """母艦が生成失敗を報告。status=failed, detail=reason。"""
        _auth(request)
        try:
            body = await request.json()
        except Exception:
            body = {}
        reason = str((body or {}).get("reason") or "生成に失敗しました")
        with _REQ_LOCK:
            req = _require_req(rid)
            if req.get("status") == "pack_ready":
                raise HTTPException(409, "既に完成した依頼はfailにできません")
            req["status"] = "failed"
            req["detail"] = reason[:500]
            _req_save(rid, req)
        return {"ok": True}

    @app.post("/api/requests/{rid}/delete")
    def request_delete(rid: str, request: Request):
        """DELETE相当。waiting/failed のみ削除可 (処理中は消せない)。"""
        _auth(request)
        with _REQ_LOCK:
            req = _require_req(rid)
            if req.get("status") not in ("waiting", "failed"):
                raise HTTPException(409, "waiting/failedの依頼のみ削除できます")
            shutil.rmtree(requests_root() / rid, ignore_errors=True)
        return {"ok": True}

    return app


# ---------------------------------------------------------------- webUI
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="ja"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SpriteMill VideoLab</title>
<style>
:root{--bg:#1c1c1c;--panel:#252526;--panel2:#2d2d30;--fg:#e8e8e8;--dim:#9a9a9a;
--accent:#57a6ff;--ok:#57d38c;--err:#ff6b6b;--warn:#ffc857;--border:#3c3c3c}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font-family:"Segoe UI","Yu Gothic UI",Meiryo,sans-serif;font-size:14px}
header{display:flex;align-items:center;gap:12px;padding:10px 16px;
background:var(--panel);border-bottom:1px solid var(--border)}
header h1{font-size:16px;margin:0}
header .badge{font-size:12px;color:var(--dim)}
#gpu{margin-left:auto;font-size:12px;color:var(--dim)}
main{display:grid;grid-template-columns:minmax(330px,430px) 1fr;gap:12px;
padding:12px;max-width:1500px;margin:0 auto}
@media(max-width:900px){main{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--border);border-radius:8px;
padding:14px;margin-bottom:12px}
.card h2{font-size:13px;margin:0 0 10px;color:var(--accent);
text-transform:uppercase;letter-spacing:.05em}
label{display:block;font-size:12px;color:var(--dim);margin:8px 0 3px}
input,select,textarea{width:100%;background:var(--panel2);color:var(--fg);
border:1px solid var(--border);border-radius:5px;padding:6px 8px;font-size:13px}
textarea{resize:vertical;min-height:64px;font-family:inherit}
input:focus,select:focus,textarea:focus{outline:1px solid var(--accent)}
.row{display:flex;gap:8px}.row>*{flex:1;min-width:0}
button{background:var(--accent);color:#0b1320;border:0;border-radius:6px;
padding:9px 14px;font-size:14px;font-weight:600;cursor:pointer}
button:hover{filter:brightness(1.1)}
button.sec{background:var(--panel2);color:var(--fg);border:1px solid var(--border);
font-weight:400;padding:5px 10px;font-size:12px}
button:disabled{opacity:.5;cursor:default}
.tabs{display:flex;gap:4px;margin-bottom:10px}
.tabs button{flex:1;background:var(--panel2);color:var(--dim);font-weight:400;
border:1px solid var(--border);padding:7px 4px;font-size:12.5px}
.tabs button.on{background:#123a63;color:var(--fg);border-color:var(--accent)}
#drop{border:2px dashed var(--border);border-radius:8px;padding:14px;
text-align:center;color:var(--dim);cursor:pointer;font-size:12.5px}
#drop.over{border-color:var(--accent);color:var(--accent)}
#thumbs{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
.thumb{position:relative;width:76px}
.thumb img{width:76px;height:76px;object-fit:contain;background:#111;
border:1px solid var(--border);border-radius:5px;display:block}
.thumb .x{position:absolute;top:-6px;right:-6px;width:18px;height:18px;
border-radius:50%;background:var(--err);color:#fff;border:0;font-size:11px;
line-height:18px;padding:0;text-align:center;cursor:pointer}
.thumb .pos{width:100%;font-size:10.5px;text-align:center;margin-top:2px;
padding:1px 2px}
.job{border:1px solid var(--border);border-radius:7px;padding:10px;
margin-bottom:8px;background:var(--panel2);cursor:pointer}
.job.sel{border-color:var(--accent)}
.job .top{display:flex;gap:8px;align-items:center;font-size:12.5px}
.job .id{font-family:Consolas,monospace;color:var(--dim)}
.st{font-weight:600}.st.queued{color:var(--dim)}.st.loading{color:var(--warn)}
.st.running{color:var(--accent)}.st.done{color:var(--ok)}
.st.error{color:var(--err)}.st.cancelled{color:var(--dim)}
.bar{height:5px;background:#171717;border-radius:3px;margin-top:6px;overflow:hidden}
.bar>i{display:block;height:100%;background:var(--accent);width:0%;
transition:width .4s}
.job .meta{font-size:11.5px;color:var(--dim);margin-top:5px;
white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#detail video{width:100%;max-height:420px;background:#000;border-radius:6px}
#detail pre{background:#161616;border:1px solid var(--border);border-radius:6px;
padding:8px;font-size:11px;max-height:200px;overflow:auto;white-space:pre-wrap}
.mrow{display:flex;gap:8px;align-items:center}
.hint{font-size:11.5px;color:var(--dim);margin-top:6px;line-height:1.5}
#tokrow{display:none}
a{color:var(--accent)}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--err)}
.dot.on{background:var(--ok)}
</style></head><body>
<header>
  <h1>🎬 SpriteMill VideoLab</h1>
  <span class="badge">モデル実験用 動画生成サーバ</span>
  <span id="conn"><span class="dot" id="dot"></span> <span id="connTx">接続待ち</span></span>
  <span id="gpu"></span>
</header>
<main>
<div><!-- 左: 生成フォーム -->
  <div class="card" id="tokrow">
    <h2>接続トークン</h2>
    <input id="token" placeholder="Colabノートに表示された TOKEN を貼り付け">
    <div class="hint">このサーバはトークン保護されています。URL の ?token=… からも自動取得します。</div>
  </div>
  <div class="card">
    <h2>モデル</h2>
    <div class="mrow"><select id="model"></select></div>
    <div class="hint" id="modelDesc"></div>
  </div>
  <div class="card">
    <h2>生成</h2>
    <div class="tabs">
      <button data-mode="i2v" class="on">画像→動画</button>
      <button data-mode="multikey">キーフレーム→動画</button>
      <button data-mode="t2v">テキスト→動画</button>
    </div>
    <div id="imgArea">
      <div id="drop">クリック / ドラッグ&amp;ドロップで画像を追加<br>
      <span id="dropHint">(i2v: 先頭1枚を開始フレームに使用)</span></div>
      <input type="file" id="file" accept="image/*" multiple hidden>
      <div id="thumbs"></div>
    </div>
    <label>プロンプト</label>
    <textarea id="prompt" placeholder="例: the same chibi character walking in place, fixed camera, no rotation"></textarea>
    <label>ネガティブ (対応モデルのみ)</label>
    <textarea id="negative" style="min-height:40px" placeholder="worst quality, blurry, distorted"></textarea>
    <div class="row">
      <div><label>幅</label><input id="width" type="number" step="32" value="768"></div>
      <div><label>高さ</label><input id="height" type="number" step="32" value="512"></div>
      <div><label>フレーム数</label><input id="frames" type="number" value="97"></div>
    </div>
    <div class="row">
      <div><label>fps</label><input id="fps" type="number" value="24"></div>
      <div><label>ステップ</label><input id="steps" type="number" value="30"></div>
      <div><label>guidance</label><input id="guidance" type="number" step="0.5" value="3"></div>
      <div><label>シード (0=乱数)</label><input id="seed" type="number" value="0"></div>
    </div>
    <label>extra — モデル固有オプション (JSON)</label>
    <input id="extra" placeholder='例: {"walk_lora": "pixel_walk", "motion_score": 3.5}'>
    <div style="margin-top:14px"><button id="go" style="width:100%">▶ 生成開始</button></div>
    <div class="hint" id="formHint"></div>
  </div>
</div>
<div><!-- 右: ジョブ -->
  <div class="card">
    <h2>ジョブ</h2>
    <div id="jobs"><div class="hint">まだジョブがありません。</div></div>
  </div>
  <div class="card" id="detail" style="display:none">
    <h2>結果 <span id="dId" class="badge"></span></h2>
    <div id="dVideo"></div>
    <div style="margin:8px 0" id="dActions"></div>
    <pre id="dLog"></pre>
  </div>
</div>
</main>
<script>
'use strict';
const $=id=>document.getElementById(id);
let MODE='i2v', IMAGES=[], MODELS=[], SELJOB=null, TOKEN='';
const qs=new URLSearchParams(location.search);
if(qs.get('token')){TOKEN=qs.get('token');localStorage.setItem('vl_token',TOKEN);}
else TOKEN=localStorage.getItem('vl_token')||'';
$('token').value=TOKEN;
$('token').addEventListener('input',e=>{TOKEN=e.target.value.trim();
  localStorage.setItem('vl_token',TOKEN);refreshModels();});

function api(path,opt={}){opt.headers=Object.assign({},opt.headers,
  TOKEN?{'Authorization':'Bearer '+TOKEN}:{});
  return fetch(path,opt).then(r=>{
    if(r.status===401)throw new Error('401 トークンが違います');
    if(!r.ok)return r.json().catch(()=>({})).then(b=>{
      throw new Error(b.detail||('HTTP '+r.status));});
    return r.json();});}

// ---- 接続 & モデル一覧 ----
async function refreshHealth(){
  try{const h=await fetch('/health').then(r=>r.json());
    $('dot').className='dot on';$('connTx').textContent='接続OK';
    $('tokrow').style.display=h.auth?'block':'none';
    $('gpu').textContent=h.gpu?`GPU: ${h.gpu.name} (${h.gpu.free_gb}/${h.gpu.vram_gb}GB free)`:'GPU: なし(mockのみ)';
    if(!MODELS.length)refreshModels();
  }catch(e){$('dot').className='dot';$('connTx').textContent='接続エラー';}}
async function refreshModels(){
  try{const m=await api('/api/models');MODELS=m.models;
    const sel=$('model');const cur=sel.value;sel.innerHTML='';
    for(const md of MODELS){const o=document.createElement('option');
      o.value=md.id;o.textContent=md.label+(md.loaded?' ✓':'');sel.appendChild(o);}
    if([...sel.options].some(o=>o.value===cur))sel.value=cur;
    else if(m.current)sel.value=m.current;
    onModel();}
  catch(e){$('modelDesc').textContent='⚠ モデル一覧を取得できません ('+e.message+
    ')。上の「接続トークン」欄に TOKEN を貼るか、?token=付きのURLで開いてください。';}}
function onModel(){const md=MODELS.find(x=>x.id===$('model').value);if(!md)return;
  $('modelDesc').textContent=(md.desc||'')+(md.requires?' 【必要環境: '+md.requires+'】':'');
  for(const k of['width','height','fps','steps','guidance'])
    if(md.defaults&&md.defaults[k]!=null)$(k==='frames'?'frames':k).value=md.defaults[k];
  if(md.defaults&&md.defaults.num_frames!=null)$('frames').value=md.defaults.num_frames;
  document.querySelectorAll('.tabs button').forEach(b=>{
    b.disabled=!md.modes.includes(b.dataset.mode);});
  if(!md.modes.includes(MODE)){setMode(md.modes[0]);}}
$('model').addEventListener('change',onModel);

// ---- モード ----
function setMode(m){MODE=m;
  document.querySelectorAll('.tabs button').forEach(b=>
    b.classList.toggle('on',b.dataset.mode===m));
  $('imgArea').style.display=(m==='t2v')?'none':'block';
  $('dropHint').textContent=m==='multikey'
    ?'(複数枚OK。各サムネ下の数値=動画内の位置% を編集可)'
    :'(i2v: 先頭1枚を開始フレームに使用)';
  renderThumbs();}
document.querySelectorAll('.tabs button').forEach(b=>
  b.addEventListener('click',()=>!b.disabled&&setMode(b.dataset.mode)));

// ---- 画像 ----
$('drop').addEventListener('click',()=>$('file').click());
$('drop').addEventListener('dragover',e=>{e.preventDefault();$('drop').classList.add('over');});
$('drop').addEventListener('dragleave',()=>$('drop').classList.remove('over'));
$('drop').addEventListener('drop',e=>{e.preventDefault();$('drop').classList.remove('over');
  addFiles(e.dataTransfer.files);});
$('file').addEventListener('change',e=>addFiles(e.target.files));
function addFiles(fs){for(const f of fs){if(!f.type.startsWith('image/'))continue;
  const rd=new FileReader();rd.onload=()=>{IMAGES.push({b64:rd.result,pos:null});
    renderThumbs();};rd.readAsDataURL(f);}}
function renderThumbs(){const t=$('thumbs');t.innerHTML='';
  const n=IMAGES.length;
  IMAGES.forEach((im,i)=>{if(im.pos==null)im.pos=n>1?Math.round(i/(n-1)*100):0;
    const d=document.createElement('div');d.className='thumb';
    d.innerHTML=`<img src="${im.b64}"><button class="x">×</button>`+
      (MODE==='multikey'?`<input class="pos" type="number" min="0" max="100" value="${im.pos}">`:'');
    d.querySelector('.x').onclick=()=>{IMAGES.splice(i,1);IMAGES.forEach(x=>x.pos=null);renderThumbs();};
    const p=d.querySelector('.pos');if(p)p.onchange=e=>{im.pos=+e.target.value;};
    t.appendChild(d);});}

// ---- 生成 ----
$('go').addEventListener('click',async()=>{
  $('formHint').textContent='';
  try{
    if(MODE!=='t2v'&&!IMAGES.length)throw new Error('画像を追加してください');
    let extra={};const ex=$('extra').value.trim();
    if(ex){try{extra=JSON.parse(ex);}catch(e){throw new Error('extra が正しいJSONではありません');}}
    const body={model:$('model').value,mode:MODE,prompt:$('prompt').value,extra,
      negative:$('negative').value,
      images_b64:MODE==='t2v'?[]:IMAGES.map(x=>x.b64),
      key_positions:MODE==='multikey'?IMAGES.map(x=>(x.pos||0)/100):[],
      width:+$('width').value,height:+$('height').value,
      num_frames:+$('frames').value,fps:+$('fps').value,
      steps:+$('steps').value,guidance:+$('guidance').value,seed:+$('seed').value};
    $('go').disabled=true;
    const r=await api('/api/generate',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    SELJOB=r.job;refreshJobs();
  }catch(e){$('formHint').textContent='⚠ '+e.message;}
  finally{$('go').disabled=false;}});

// ---- ジョブ一覧 ----
const ST_JP={queued:'待機',loading:'モデル読込',running:'生成中',
  done:'完了',error:'エラー',cancelled:'中止'};
async function refreshJobs(){
  let data;try{data=await api('/api/jobs');}catch(e){return;}
  const box=$('jobs');box.innerHTML='';
  if(!data.jobs.length){box.innerHTML='<div class="hint">まだジョブがありません。</div>';return;}
  for(const j of data.jobs){
    const el=document.createElement('div');
    el.className='job'+(j.id===SELJOB?' sel':'');
    const el1=j.started?((j.finished||Date.now()/1e3)-j.started):0;
    el.innerHTML=`<div class="top"><span class="st ${j.status}">● ${ST_JP[j.status]||j.status}</span>
      <span class="id">${j.id}</span><span style="margin-left:auto">${j.model}/${j.mode}</span>
      ${['queued','loading','running'].includes(j.status)?'<button class="sec cx">中止</button>':''}</div>
      <div class="bar"><i style="width:${Math.round(j.progress*100)}%"></i></div>
      <div class="meta">${j.params.width}x${j.params.height} ${j.params.num_frames}f
       step${j.params.steps} ${el1?Math.round(el1)+'s':''} ${j.detail||''} ${j.prompt||''}</div>`;
    el.addEventListener('click',()=>{SELJOB=j.id;refreshJobs();});
    const cx=el.querySelector('.cx');
    if(cx)cx.addEventListener('click',e=>{e.stopPropagation();
      api('/api/cancel/'+j.id,{method:'POST'});});
    box.appendChild(el);
    if(j.id===SELJOB)showDetail(j);}
  if(SELJOB&&!data.jobs.some(j=>j.id===SELJOB)){$('detail').style.display='none';}}
let LASTV='';
function showDetail(j){$('detail').style.display='block';$('dId').textContent=j.id;
  const tokq=TOKEN?('?token='+encodeURIComponent(TOKEN)):'';
  if(j.status==='done'){
    if(LASTV!==j.id){$('dVideo').innerHTML=
      `<video controls loop autoplay muted src="/result/${j.id}${tokq}"></video>`;
      $('dActions').innerHTML=
      `<a href="/result/${j.id}${tokq}" download="videolab_${j.id}.mp4"><button class="sec">⬇ ダウンロード</button></a>`;
      LASTV=j.id;}
  }else{$('dVideo').innerHTML='';$('dActions').innerHTML='';LASTV='';}
  $('dLog').textContent=(j.log||[]).slice(-40).join('\n')||'(ログなし)';}

setMode('i2v');refreshHealth();refreshModels();
setInterval(refreshHealth,10000);setInterval(refreshJobs,2000);refreshJobs();
</script></body></html>
"""


# ---------------------------------------------------- webUI (工房モード)
# トップ `/` のお友だち用ページ。低レベル操作 (モデル・解像度・プロンプト)
# は一切置かない。トークンの扱いは INDEX_HTML と同じ (?token= /
# localStorage vl_token を共有)。
KOBO_HTML = r"""<!DOCTYPE html>
<html lang="ja"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SpriteMill 工房</title>
<style>
:root{--bg:#1c1c1c;--panel:#252526;--panel2:#2d2d30;--fg:#e8e8e8;--dim:#9a9a9a;
--accent:#57a6ff;--ok:#57d38c;--err:#ff6b6b;--warn:#ffc857;--border:#3c3c3c}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font-family:"Segoe UI","Yu Gothic UI",Meiryo,sans-serif;font-size:15px}
header{display:flex;align-items:center;gap:12px;padding:12px 18px;
background:var(--panel);border-bottom:1px solid var(--border)}
header h1{font-size:18px;margin:0}
header .badge{font-size:12px;color:var(--dim)}
header a{margin-left:auto;font-size:12px;color:var(--dim);text-decoration:none}
header a:hover{color:var(--accent)}
main{max-width:1060px;margin:0 auto;padding:14px}
.card{background:var(--panel);border:1px solid var(--border);
border-radius:10px;padding:16px;margin-bottom:14px}
.card h2{font-size:14px;margin:0 0 12px;color:var(--accent);
letter-spacing:.05em}
.card h3{font-size:13px;margin:14px 0 8px;color:var(--dim)}
input{width:100%;background:var(--panel2);color:var(--fg);
border:1px solid var(--border);border-radius:6px;padding:7px 9px;font-size:13px}
textarea{width:100%;background:var(--panel2);color:var(--fg);
border:1px solid var(--border);border-radius:6px;padding:7px 9px;
font-size:13px;font-family:inherit;resize:vertical;min-height:64px}
input:focus,textarea:focus{outline:1px solid var(--accent)}
label{display:block;font-size:12px;color:var(--dim);margin:10px 0 3px}
.req{background:var(--panel2);border:1px solid var(--border);
border-radius:10px;padding:10px 12px;margin-top:10px;display:flex;
gap:12px;align-items:center;flex-wrap:wrap}
.req .grow{flex:1;min-width:180px}
.req .nm{font-size:13.5px;font-weight:600}
.req .stx{font-size:12.5px;color:var(--dim);margin-top:2px;line-height:1.5}
.req .stx.err{color:var(--err)}
.req .stx.ok{color:var(--ok)}
button{background:var(--accent);color:#0b1320;border:0;border-radius:8px;
padding:10px 18px;font-size:15px;font-weight:600;cursor:pointer}
button:hover{filter:brightness(1.1)}
button.sec{background:var(--panel2);color:var(--fg);
border:1px solid var(--border);font-weight:400;font-size:13px;padding:8px 12px}
button:disabled{opacity:.5;cursor:default}
.hint{font-size:12.5px;color:var(--dim);margin-top:8px;line-height:1.6}
#packs{display:flex;flex-wrap:wrap;gap:12px}
.pack{width:150px;background:var(--panel2);border:1px solid var(--border);
border-radius:10px;padding:10px;text-align:center;cursor:pointer}
.pack:hover{border-color:var(--accent)}
.pack.sel{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent)}
.pack img{height:120px;max-width:100%;object-fit:contain;display:block;
margin:0 auto 8px;image-rendering:auto;background:#161616;border-radius:6px}
.pack .nm{font-size:13px;font-weight:600;overflow:hidden;
text-overflow:ellipsis;white-space:nowrap}
.bar{height:10px;background:#171717;border-radius:5px;margin:10px 0 6px;
overflow:hidden}
.bar>i{display:block;height:100%;background:var(--accent);width:0%;
transition:width .5s}
.st{font-weight:600}.st.queued{color:var(--dim)}.st.running{color:var(--accent)}
.st.done{color:var(--ok)}.st.error{color:var(--err)}.st.cancelled{color:var(--dim)}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:760px){.cols{grid-template-columns:1fr}}
#prev,#pix{max-width:100%;background:
repeating-conic-gradient(#202020 0 25%,#2a2a2a 0 50%) 0 0/24px 24px;
border:1px solid var(--border);border-radius:8px;display:block}
#dvid{width:100%;max-height:430px;background:#000;border-radius:8px;
margin-top:8px}
select{background:var(--panel2);color:var(--fg);border:1px solid var(--border);
border-radius:6px;padding:7px 9px;font-size:13.5px;width:100%}
.actions{display:flex;gap:10px;flex-wrap:wrap;margin-top:16px;
align-items:center}
#tokrow{display:none}
a{color:var(--accent)}
</style></head><body>
<header>
  <h1>🎨 SpriteMill 工房</h1>
  <span class="badge">キャラを選んで、歩くスプライトを作ろう</span>
  <a href="/advanced">上級者向け画面</a>
</header>
<main>
  <div class="card" id="tokrow">
    <h2>あいことば (トークン)</h2>
    <input id="token" placeholder="共有された TOKEN をここに貼り付けてください">
    <div class="hint">リンクに ?token=… が付いていれば自動で入ります。</div>
  </div>
  <div class="card">
    <h2>0. キャラをつくる (おまかせ)</h2>
    <label>キャラのなまえ (必須・40文字まで)</label>
    <input id="rqname" maxlength="40" placeholder="例: ルナ">
    <label>どんな子にする? (コンセプト・500文字まで)</label>
    <textarea id="rqconcept" maxlength="500"
      placeholder="例: 銀髪ツインテールの魔法使いの女の子。青いローブに星のステッキ。"></textarea>
    <label>参考画像 (あれば / PNG・JPEG、20MBまで)</label>
    <input type="file" id="rqfile" accept="image/png,image/jpeg">
    <div class="actions">
      <button id="btnreq">✉ 依頼する</button>
      <span class="hint" id="rqmsg"></span>
    </div>
    <div id="reqs"></div>
  </div>
  <div class="card">
    <h2>1. キャラをえらぶ</h2>
    <div id="packs"><div class="hint">読み込み中…</div></div>
  </div>
  <div class="card" id="work" style="display:none">
    <h2>2. 歩かせる — <span id="wname"></span></h2>
    <div id="startrow">
      <button id="btngo">▶ 歩行スプライトを生成する</button>
      <span class="hint">生成には数分〜十数分かかります (そのまま待っていてOK)</span>
    </div>
    <div id="prog" style="display:none">
      <div><span class="st" id="ptst"></span> <span id="ptxt" class="hint"></span></div>
      <div class="bar"><i id="pbar"></i></div>
      <button class="sec" id="btncancel">中止する</button>
    </div>
    <div id="perr" class="hint" style="display:none;color:var(--err)"></div>
    <div id="result" style="display:none">
      <div class="cols">
        <div>
          <h3>歩きプレビュー (全方向)</h3>
          <img id="prev" alt="preview">
        </div>
        <div>
          <h3>方向ごとのムービー</h3>
          <select id="dirsel"></select>
          <video id="dvid" controls loop autoplay muted playsinline></video>
        </div>
      </div>
      <div id="pixwrap" style="display:none">
        <h3>ドット絵シート</h3>
        <img id="pix" alt="pixel art">
      </div>
      <div class="actions">
        <button id="btnpix">🟦 ドット絵にする</button>
        <button id="btnwebp" class="sec">🔄 WEBPを作り直す</button>
        <a id="dl" download><button class="sec">⬇ まとめてダウンロード</button></a>
        <span class="hint" id="msg"></span>
      </div>
    </div>
  </div>
</main>
<script>
'use strict';
const $=id=>document.getElementById(id);
let TOKEN='',SEL=null,SELNAME='',JOB=null,LASTFILES='',BUSY=false;
const qs=new URLSearchParams(location.search);
if(qs.get('token')){TOKEN=qs.get('token');localStorage.setItem('vl_token',TOKEN);}
else TOKEN=localStorage.getItem('vl_token')||'';
$('token').value=TOKEN;
$('token').addEventListener('input',e=>{TOKEN=e.target.value.trim();
  localStorage.setItem('vl_token',TOKEN);loadPacks();loadRequests();});
const tokq=()=>TOKEN?('?token='+encodeURIComponent(TOKEN)):'';
function api(path,opt={}){opt.headers=Object.assign({},opt.headers,
  TOKEN?{'Authorization':'Bearer '+TOKEN}:{});
  return fetch(path,opt).then(r=>{
    if(r.status===401)throw new Error('あいことば(トークン)が違います');
    if(!r.ok)return r.json().catch(()=>({})).then(b=>{
      throw new Error(b.detail||('HTTP '+r.status));});
    return r.json();});}

const DIRJP={front:'まえ',back:'うしろ',left:'ひだり',right:'みぎ',
  front_left:'ひだりナナメまえ',front_right:'みぎナナメまえ',
  back_left:'ひだりナナメうしろ',back_right:'みぎナナメうしろ'};
const DIRORD=['front','front_left','front_right','left','right',
  'back_left','back_right','back'];
const STJP={queued:'じゅんび中…',running:'生成中…',done:'できあがり!',
  error:'エラー',cancelled:'中止しました'};

async function checkAuth(){
  try{const h=await fetch('/health').then(r=>r.json());
    $('tokrow').style.display=h.auth?'block':'none';}catch(e){}}

// ---- 0. 依頼リレー (キャラをつくる) ----
const RQST={waiting:'母艦の受付待ち (母艦PCが起動している必要があります)',
  claimed:'立ち絵を生成中… (数分かかります)'};
function esc(s){return String(s??'').replace(/[&<>"']/g,
  c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
let REQSIG='';
$('btnreq').addEventListener('click',async()=>{
  const name=$('rqname').value.trim();
  if(!name){$('rqmsg').textContent='⚠ なまえを入れてください';return;}
  const body={name,concept:$('rqconcept').value.trim()};
  const f=$('rqfile').files[0];
  if(f){
    if(f.size>20*1024*1024){
      $('rqmsg').textContent='⚠ 画像が大きすぎます (20MBまで)';return;}
    try{body.image_b64=await new Promise((ok,ng)=>{
      const r=new FileReader();
      r.onload=()=>ok(String(r.result).split(',',2)[1]);
      r.onerror=()=>ng(new Error('画像を読み込めません'));
      r.readAsDataURL(f);});}
    catch(e){$('rqmsg').textContent='⚠ '+e.message;return;}}
  $('btnreq').disabled=true;$('rqmsg').textContent='送信中…';
  try{await api('/api/requests',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    $('rqmsg').textContent='依頼を送りました! 下で進み具合が見られます。';
    $('rqname').value='';$('rqconcept').value='';$('rqfile').value='';
    loadRequests();}
  catch(e){$('rqmsg').textContent='⚠ '+e.message;}
  finally{$('btnreq').disabled=false;}});

async function delRequest(rid){
  try{await api('/api/requests/'+encodeURIComponent(rid)+'/delete',
    {method:'POST'});}catch(e){}
  loadRequests();}

async function loadRequests(){
  let rs,jobs=[];
  try{rs=await api('/api/requests');}catch(e){return;}
  try{jobs=(await api('/api/jobs')).jobs||[];}catch(e){}
  const box=$('reqs');
  // 状態が動いたときだけパック一覧も更新 (pack_ready到着の反映)
  const sig=rs.map(r=>r.request_id+':'+r.status).join(',');
  if(sig!==REQSIG){if(REQSIG)loadPacks();REQSIG=sig;}
  box.innerHTML='';
  for(const r of rs){
    const d=document.createElement('div');d.className='req';
    let st='',cls='',extra='';
    if(r.status==='waiting'||r.status==='claimed'){
      st=RQST[r.status];
      if(r.status==='waiting')extra='<button class="sec" data-del="'+
        r.request_id+'">とりけす</button>';
    }else if(r.status==='failed'){
      st='⚠ 失敗: '+esc(r.detail||'');cls=' err';
      extra='<button class="sec" data-del="'+r.request_id+'">削除</button>';
    }else if(r.status==='pack_ready'){
      const j=jobs.find(x=>x.model==='walkpack'&&x.pack===r.pack_id&&
        ['queued','running'].includes(x.status));
      if(j){st='歩行スプライトを生成中… '+
        Math.round((j.progress||0)*100)+'%';}
      else{
        const je=jobs.find(x=>x.model==='walkpack'&&x.pack===r.pack_id&&
          x.status==='error');
        if(je){st='⚠ 歩行生成に失敗しました: '+esc(je.detail||'');cls=' err';}
        else{st='できあがり!';cls=' ok';}
        extra='<button class="sec" data-open="'+esc(r.pack_id)+
          '" data-nm="'+esc(r.name)+'">ひらく</button>';
      }
    }else{st=esc(r.status);}
    d.innerHTML='<div class="grow"><div class="nm">'+esc(r.name)+
      '</div><div class="stx'+cls+'">'+st+'</div></div>'+extra;
    box.appendChild(d);}
  box.querySelectorAll('[data-del]').forEach(b=>b.addEventListener('click',
    ()=>delRequest(b.dataset.del)));
  box.querySelectorAll('[data-open]').forEach(b=>b.addEventListener('click',
    ()=>{select(b.dataset.open,b.dataset.nm);
      $('work').scrollIntoView({behavior:'smooth'});}));}

// ---- パック一覧 ----
async function loadPacks(){
  let ps;try{ps=await api('/api/packs');}
  catch(e){$('packs').innerHTML='<div class="hint">⚠ '+e.message+'</div>';return;}
  const box=$('packs');box.innerHTML='';
  if(!ps.length){box.innerHTML=
    '<div class="hint">キャラパックがまだありません。配布側のアプリからアップロードしてもらってください。</div>';return;}
  for(const p of ps){
    const d=document.createElement('div');
    d.className='pack'+(p.pack_id===SEL?' sel':'');
    d.innerHTML=`<img src="/api/packs/${encodeURIComponent(p.pack_id)}/thumb.png${tokq()}" alt="">
      <div class="nm">${p.name}</div>`;
    d.addEventListener('click',()=>select(p.pack_id,p.name));
    box.appendChild(d);}}

function select(pid,name){SEL=pid;SELNAME=name;JOB=null;LASTFILES='';
  $('work').style.display='block';$('wname').textContent=name;
  $('result').style.display='none';$('prog').style.display='none';
  $('perr').style.display='none';$('startrow').style.display='block';
  $('msg').textContent='';
  document.querySelectorAll('.pack').forEach(el=>el.classList.remove('sel'));
  loadPacks();loadFiles();poll();}

// ---- 生成 ----
$('btngo').addEventListener('click',async()=>{
  if(!SEL)return;$('btngo').disabled=true;$('perr').style.display='none';
  try{const r=await api('/api/walkpack/'+encodeURIComponent(SEL),{method:'POST'});
    JOB=r.job;$('startrow').style.display='none';$('prog').style.display='block';}
  catch(e){$('perr').textContent='⚠ '+e.message;$('perr').style.display='block';}
  finally{$('btngo').disabled=false;}});
$('btncancel').addEventListener('click',()=>{
  if(JOB)api('/api/cancel/'+JOB,{method:'POST'}).catch(()=>{});});

// ---- 進捗ポーリング ----
async function poll(){
  if(!SEL)return;
  let data;try{data=await api('/api/jobs');}catch(e){return;}
  const j=(data.jobs||[]).find(x=>x.model==='walkpack'&&x.pack===SEL&&
    (JOB?x.id===JOB:true));
  if(!j)return;
  if(['queued','running'].includes(j.status)){
    JOB=j.id;$('startrow').style.display='none';
    $('prog').style.display='block';
    $('ptst').className='st '+j.status;
    $('ptst').textContent=STJP[j.status]||j.status;
    $('ptxt').textContent=j.detail||'';
    $('pbar').style.width=Math.round((j.progress||0)*100)+'%';
  }else if(JOB&&j.id===JOB){
    $('prog').style.display='none';$('startrow').style.display='block';
    if(j.status==='done'){JOB=null;loadFiles(true);}
    else if(j.status==='error'){JOB=null;
      $('perr').textContent='⚠ 生成に失敗しました: '+(j.detail||'');
      $('perr').style.display='block';}
    else{JOB=null;}
  }}
setInterval(poll,2000);

// ---- 結果 ----
async function loadFiles(bust){
  if(!SEL)return;
  let fs;try{fs=await api('/api/walkpack/'+encodeURIComponent(SEL)+'/files');}
  catch(e){return;}
  const key=fs.map(f=>f.name).join(',');
  if(!bust&&key===LASTFILES)return;LASTFILES=key;
  if(!fs.length){$('result').style.display='none';return;}
  const cb=bust?('&t='+Date.now()):'';
  const url=n=>'/api/walkpack/'+encodeURIComponent(SEL)+'/file/'+
    encodeURIComponent(n)+tokq()+(tokq()?cb:(bust?('?t='+Date.now()):''));
  $('result').style.display='block';
  const prev=fs.find(f=>f.kind==='preview');
  $('prev').style.display=prev?'block':'none';
  if(prev)$('prev').src=url(prev.name);
  const mp4s=fs.filter(f=>f.kind==='mp4'&&!f.name.startsWith('canvas_'));
  const sel=$('dirsel');const cur=sel.value;sel.innerHTML='';
  const byDir={};
  for(const f of mp4s){const m=f.name.match(/_\d\d_([a-z_]+)_walkT\.mp4$/);
    if(m)byDir[m[1]]=f.name;}
  for(const d of DIRORD){if(!byDir[d])continue;
    const o=document.createElement('option');
    o.value=byDir[d];o.textContent=DIRJP[d]||d;sel.appendChild(o);}
  if([...sel.options].some(o=>o.value===cur))sel.value=cur;
  if(sel.options.length){$('dvid').src=url(sel.value);
    $('dvid').parentElement.style.display='block';}
  const pix=fs.find(f=>f.name.endsWith('_pixel@2x.png'))||
            fs.find(f=>f.kind==='pixel');
  $('pixwrap').style.display=pix?'block':'none';
  if(pix)$('pix').src=url(pix.name);
  $('dl').href='/api/walkpack/'+encodeURIComponent(SEL)+'/download'+tokq();}
$('dirsel').addEventListener('change',()=>{
  $('dvid').src='/api/walkpack/'+encodeURIComponent(SEL)+'/file/'+
    encodeURIComponent($('dirsel').value)+tokq();});

async function post2(target,label){
  if(!SEL||BUSY)return;BUSY=true;$('msg').textContent=label+'中…';
  $('btnpix').disabled=$('btnwebp').disabled=true;
  try{await api('/api/walkpack/'+encodeURIComponent(SEL)+'/pixelize',
    {method:'POST',headers:{'Content-Type':'application/json'},
     body:JSON.stringify({target})});
    $('msg').textContent=label+'ができました!';loadFiles(true);}
  catch(e){$('msg').textContent='⚠ '+e.message;}
  finally{BUSY=false;$('btnpix').disabled=$('btnwebp').disabled=false;}}
$('btnpix').addEventListener('click',()=>post2('pixel','ドット絵'));
$('btnwebp').addEventListener('click',()=>post2('preview','WEBP'));

checkAuth();loadPacks();loadRequests();
setInterval(checkAuth,15000);
setInterval(loadRequests,3000);
</script></body></html>
"""


# ---------------------------------------------------------------- 起動
def start_server(host: str, port: int, token: str | None):
    import uvicorn
    app = build_app(token)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=worker_loop, daemon=True).start()
    threading.Thread(target=_watchdog_loop, daemon=True).start()
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    return server


def run_in_colab(port: int = 8000, preload: str | None = None):
    """Colabノートブックから呼ぶ: サーバ+cloudflaredトンネルを起動しURL/TOKENを表示。

    セルの再実行に強い設計: トークンは「ファイルが無い時だけ」生成し、
    以後はランタイムが変わるまで常に同じ値を使い回す。こうしないと、
    サーバ起動中(healthがまだ立たない数秒)にセルがもう一度実行された
    とき alive=False 判定で新トークンが発行され、2つ目のサーバは
    ポート占有でスレッド内に静かに死に、トンネルは旧トークンのサーバに
    つながったまま画面には新トークンが印字される(=貼っても必ず401。
    2026-07-12 実障害。Run All連打/自動運転リトライで再現)。
    古い cloudflared だけ張り替える。
    """
    import re as _re
    import urllib.request as _rq
    # Colabのディスクは60GB級モデル2つで枯渇するため、切替時の自動削除を既定ON
    os.environ.setdefault("VIDEOLAB_PURGE_ON_SWITCH", "1")
    tok_file = Path(tempfile.gettempdir()) / "videolab_token.txt"

    def _health_up(timeout=3):
        try:
            with _rq.urlopen(f"http://127.0.0.1:{port}/health",
                             timeout=timeout) as r:
                return r.status == 200
        except Exception:
            return False

    # トークンの決定はサーバ生存判定と独立(何回実行しても同じ値になる)
    token = ""
    if tok_file.is_file():
        token = tok_file.read_text(encoding="utf-8").strip()
    if not token or token.startswith("-"):
        # '-'始まりはCLIでオプションと誤認される (実障害 2026-07-13
        # "-w_v0...": argparseが --videolab-token の値を取れず起動失敗)
        token = secrets.token_urlsafe(16)
        while token.startswith("-"):
            token = secrets.token_urlsafe(16)
        tok_file.write_text(token, encoding="utf-8")
    if _health_up():
        print("既存のサーバを再利用します(トークン据え置き・トンネルのみ再作成)")
    else:
        server = start_server("127.0.0.1", port, token)
        threading.Thread(target=server.run, daemon=True).start()
        # 固定sleepではなく health が立つまで待つ(二重起動の誤判定防止)
        for _ in range(60):
            if _health_up(timeout=2):
                break
            time.sleep(1)
        else:
            print("警告: サーバのhealthが確認できませんでした。"
                  "下のURLで接続できない場合はランタイムを再起動してください")
    if preload and preload in ADAPTERS:
        def _pre():
            ADAPTERS[preload].ensure_loaded(lambda m: print(f"[preload] {m}", flush=True))
            globals()["CURRENT_MODEL"] = preload
        threading.Thread(target=_pre, daemon=True).start()
    # トンネルもトークン同様に「生きていれば据え置き」: セルを何回実行しても
    # webUI行のURLが変わらないようにする。従来は毎回 pkill→新URL だったため、
    # 貼った直後にRun Allがもう一度走るとURLが差し替わり、貼った側は
    # getaddrinfo failed(DNS消滅)で必ず失敗した(2026-07-12実障害)。
    # cloudflaredはカーネルと別プロセスなのでランタイム再起動を生き延びる。
    #
    # 生存判定はDNS登録(DoH)で行う。v0.4.1のトンネル越しHTTPは「VMから
    # 自分のトンネルに届かない」偽陰性で毎回作り直しに、v0.4.2のプロセス
    # 生存のみは「プロセスは生きているがトンネルは死んでいる」ゾンビを
    # 延々と再利用する偽陽性になった(いずれも2026-07-12実障害)。
    # クライアントが必要とするのは公開DNSにホストが存在することなので、
    # dns.google (DoH) での解決可否がちょうど正しい判定になる。
    def _dns_ok(u, timeout=8):
        """DoHでトンネルのホスト名が引けるか。判定不能時は None。"""
        try:
            host = u.split("//", 1)[-1].split("/", 1)[0]
            with _rq.urlopen("https://dns.google/resolve?name="
                             f"{host}&type=A", timeout=timeout) as r:
                return json.load(r).get("Status") == 0
        except Exception:
            return None

    url_file = Path(tempfile.gettempdir()) / "videolab_url.txt"
    url = None
    if url_file.is_file():
        old = url_file.read_text(encoding="utf-8").strip()
        cf_alive = (subprocess.run(["pgrep", "-f", "cloudflared"],
                                   capture_output=True).returncode == 0)
        if old and cf_alive:
            dns = _dns_ok(old)
            if dns is not False:   # DoH不通(None)なら従来どおり据え置き
                url = old
                print("既存のトンネルを再利用します(URL据え置き)")
            else:
                print("既存トンネルのDNSが消えています(ゾンビ) -- 作り直します")
    for attempt in range(1, 4):
        if url is not None:
            break
        subprocess.run(["pkill", "-f", "cloudflared"], capture_output=True)
        time.sleep(1)
        proc = subprocess.Popen(
            ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{port}",
             "--no-autoupdate"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        got = None
        deadline = time.time() + 60
        while time.time() < deadline and got is None:
            line = proc.stdout.readline()
            m = _re.search(r"https://[a-z0-9-]+\.trycloudflare\.com",
                           line or "")
            if m:
                got = m.group(0)
        if not got:
            print(f"トンネルURLを取得できませんでした (試行{attempt}/3)")
            continue
        # 死産検知: URLが印字されてもDNSに載らないトンネルがある
        # (quick tunnelのレート制限/エッジ障害)。載るまで待ち、
        # 駄目なら作り直す
        for _ in range(12):
            dns = _dns_ok(got)
            if dns:
                url = got
                break
            time.sleep(5)
        if url is None:
            print(f"トンネルがDNSに載りません -- 作り直します "
                  f"(試行{attempt}/3: {got})")
    if not url:
        raise RuntimeError(
            "cloudflared のトンネルを作成できませんでした(3回失敗)。"
            "時間を置いてこのセルを再実行してください。")
    url_file.write_text(url, encoding="utf-8")
    print("=" * 62)
    print("SpriteMill VideoLab 起動完了!")
    print(f"  webUI : {url}/?token={token}")
    print(f"  URL   : {url}")
    print(f"  TOKEN : {token}")
    print("=" * 62)
    print("webUI の行をブラウザで開くとそのまま使えます。")
    print("SpriteMill 本体から使う場合は URL と TOKEN を動画AI設定に貼り付け。")
    return url, token


def main(argv=None):
    ap = argparse.ArgumentParser(description="SpriteMill VideoLab server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--token", default=os.environ.get("VIDEOLAB_TOKEN") or None,
                    help="Bearerトークン。未指定ならローカル前提で認証なし")
    ap.add_argument("--preload", default=None, help="起動時に読み込むモデルID")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)
    if args.host != "127.0.0.1" and not args.token:
        print("警告: 外部公開 (--host) するときは --token を指定してください", file=sys.stderr)
    server = start_server(args.host, args.port, args.token)
    if args.preload and args.preload in ADAPTERS:
        def _pre():
            ADAPTERS[args.preload].ensure_loaded(lambda m: print(f"[preload] {m}", flush=True))
            globals()["CURRENT_MODEL"] = args.preload
        threading.Thread(target=_pre, daemon=True).start()
    url = f"http://{args.host}:{args.port}/"
    if args.token:
        url += f"?token={args.token}"
    print(f"SpriteMill VideoLab v{__version__}  ->  {url}", flush=True)
    # GCE作業員モード: 環境変数が立っていればGCSワーカー+アイドル自己停止を
    # 起動する (Colab/ローカルは env 無設定なので素通り)。startup.sh は
    # この main() 経由で入る (--host 0.0.0.0 --token <PD固定トークン>)。
    # (2026-07-19) state公開はバインド成功後: 以前はここで即start_gce_workers
    # していたため、server.run()がバインドに失敗して死んでも (pkill直後の
    # ソケット占有等) vm/state.json が status=up のまま永久に残り、受付が
    # 死んだURLを配り続けた。run_in_gce と同じく /health 応答を確認してから
    # 公開する (バインド失敗ならプロセスごと落ちてこのスレッドも死ぬので
    # 公開されない)。server.run()は主スレッドに残す (uvicornのシグナル処理)。
    if _gcs_active() or _idle_min() > 0:
        def _gce_when_ready():
            for _ in range(60):
                try:
                    with _url_req.urlopen(
                            f"http://127.0.0.1:{args.port}/health",
                            timeout=2) as r:
                        if r.status == 200:
                            break
                except Exception:                 # noqa: BLE001
                    time.sleep(1)
            else:
                # 60秒待っても未応答: run_in_gce と同じく続行する。ここに
                # 来られる=プロセスは生きているので、課金の守り (アイドル
                # 自己停止) を立てない方が実害が大きい
                print("GCE作業員モード: /health 未確認のまま続行 (60s)",
                      flush=True)
            if start_gce_workers(token=args.token or "", port=args.port):
                print("GCE作業員モード: GCSワーカー/アイドル自己停止 稼働",
                      flush=True)
        threading.Thread(target=_gce_when_ready, daemon=True).start()
    if not args.no_browser:
        try:
            import webbrowser
            threading.Timer(1.0, lambda: webbrowser.open(url)).start()
        except Exception:
            pass
    server.run()


if __name__ == "__main__":
    main()
