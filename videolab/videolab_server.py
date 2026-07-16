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
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

# CUDAの断片化緩和(torchの初回import前に効かせる必要があるためここで設定)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

__version__ = "0.9.12"  # 0.9.12: OOM/フリーズしない保証+quant開通 (P0-P2)
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
        try:
            downgrade()
        except Exception as e:      # noqa: BLE001
            log(f"自動降格に失敗 (そのまま判定続行): {str(e)[:120]}")
        try:
            import torch
            free = torch.cuda.mem_get_info()[0] / 2**30
        except Exception:
            return
        need = act + ACT_SAFETY_GB      # 降格後は重みがVRAM外
        if free >= need:
            log(f"VRAM admission[{tag}]: 降格後OK "
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
    if dst.is_file() and dst.stat().st_size > 2**30 and (
            dsrc is None or not dsrc.is_file()
            or dst.stat().st_size == dsrc.stat().st_size):
        log(f"セッションディスクの既存コピーを再利用: {filename}")
        return str(dst)
    code = ("from huggingface_hub import hf_hub_download; "
            f"hf_hub_download({repo!r}, {filename!r})")
    if dsrc is not None and dsrc.is_file() and dsrc.stat().st_size > 2**30:
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
        return path.is_file() and path.stat().st_size > 2**30

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

    def _snap_valid(root: Path) -> str:
        """スナップショットの健全性検査 (v0.9.1-0.9.2)。

        ①.manifest.json (PC配置時に生成した 相対パス→サイズ の台帳) と
        照合 — Drive同期が未完だと大きいシャードが丸ごと欠ける実障害
        (2026-07-14: text_encoderのsafetensorsがFileNotFound)。
        ②全JSONのパース — FUSEは同期未完ファイルを『サイズはあるのに
        読むと空』で返す実障害 (config.json 0バイト)。
        問題があればその相対パスを返す (""=健全)。"""
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
                if not _json.loads(jp.read_text(encoding="utf-8")):
                    return str(jp.relative_to(root))
            except Exception:
                return str(jp.relative_to(root))
        return ""

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
            log(f"Driveキャッシュへベース部品を書き戻し: {repo}")
            shutil.copytree(local, dsrc, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns(".cache"))
            # マニフェスト (v0.9.2): 次セッションが同期未完を検出できる
            import json as _json
            man = {f.relative_to(local).as_posix(): f.stat().st_size
                   for f in local.rglob("*")
                   if f.is_file() and ".cache" not in f.parts
                   and f.name not in (".complete", ".manifest.json")}
            (dsrc / ".manifest.json").write_text(
                _json.dumps(man, ensure_ascii=False), encoding="utf-8")
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
            if not (p.is_file() and p.stat().st_size > 2**30):
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
                if p.is_file() and p.stat().st_size > 2**30:
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


def _pin_step_callback(inner, sched, x0, noise, slots: list, release: float):
    """リファインの毎step後、固定スロットを (1-σ)x0 + σε へ描き戻す。

    SDEdit再デノイズは全フレームを自由に動かすため、歩行周期の位相が
    stage1からわずかに流れて「先頭=終端同位相」(コマ選出の前提) が崩れる。
    描き戻しは初期化と同じ noise を使い、固定スロットにflow matchingの
    直線補間路そのものを歩かせる — モデルから見て軌道上の点なので予測が
    暴れず、終端σ=0で厳密にstage1へ着地する。release>0 ならσ<release の
    終盤stepは描き戻しを止め、質感の馴染ませに開放する。"""
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
        callback_kwargs["latents"] = lat
        return callback_kwargs
    return cb


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

    def _downgrade_to_block(self, log):
        """常駐/モデルオフロード構成をblock offloadへ組み替える (P0-3の
        自動降格)。DiTをCPUへ落としてからgroup offloadフックを付ける。
        失敗しても例外は投げない (admission側が再判定する)。"""
        pipe = self.pipe
        if pipe is None:
            return
        seen = set()
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
            self._try_group_offload(m, log, f"{name}(自動降格)")
        try:
            pipe.vae.enable_tiling()
        except Exception:
            pass
        _free_cuda(log)
        self._offload_mode = "block"

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
        # 中間キーフレーム (images[1:-1]): prepare_latents をフックして
        # 条件テンソルに直接注入する。失敗しても先頭/終端拘束で続行。
        mids = [_fit_image(im, w, h) for im in req.images[1:-1]]
        mid_pos = (list(req.key_positions[1:-1])
                   if len(req.key_positions) == len(req.images)
                   else [(i + 1) / (len(mids) + 1)
                         for i in range(len(mids))])
        orig_prep = None
        if mids:
            orig_prep = self.pipe.prepare_latents

            def _patched(*a, **k):
                latents, condition = orig_prep(*a, **k)
                try:
                    self._inject_mid_keyframes(condition, mids, mid_pos,
                                               n, w, h, log)
                except Exception as e:   # noqa: BLE001
                    log(f"中間キーフレーム注入に失敗 (先頭/終端のみで"
                        f"続行): {str(e)[:200]}")
                return latents, condition
            self.pipe.prepare_latents = _patched
        log(f"生成開始: {w}x{h} {n}f steps={steps} cfg={req.guidance}")
        try:
            out = _call_with_optional_kwargs(
                self.pipe, kw, ["last_image", "callback_on_step_end"], log)
        finally:
            if orig_prep is not None:
                self.pipe.prepare_latents = orig_prep
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
        # Q4_0/Q8_0だけなので _ensure_loaded_impl がHighを自動代替する)
        q = _norm_quant((extra or {}).get("quant")
                        or os.environ.get("VIDEOLAB_ANISORA_QUANT", "Q4_0"))
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
        return bool(e.get("latent_from") or e.get("refine_frames_b64"))

    def _ensure_loaded_impl(self, log):
        _require_deps(log)
        import torch
        from diffusers import (GGUFQuantizationConfig, WanImageToVideoPipeline,
                               WanTransformer3DModel)
        from huggingface_hub import hf_hub_download
        quant, offload = self._resolve_want(getattr(self, "_next_extra", {}))
        lite = self._lite_wanted(getattr(self, "_next_extra", {}))
        # High側の棚在庫はQ4_0/Q8_0のみ (2026-07-16 HF tree API実測)。
        # 下位quant指定時はHighだけQ4_0で代替する (混載ペア。Highは序盤
        # 数stepの構図担当なので品質影響は小さい)。在庫が確認できない
        # 環境 (Drive固定・オフライン) では要求どおり試して404に任せる
        hq = quant
        gguf_high = f"High/Index-Anisora-V3.2-High-{quant}.gguf"
        if not lite and quant not in ("Q4_0", "Q8_0") \
                and _hf_file_missing(self.gguf_repo, gguf_high):
            hq = "Q4_0"
            gguf_high = f"High/Index-Anisora-V3.2-High-{hq}.gguf"
            log(f"High側に{quant}の在庫が無いためHighのみQ4_0で代替します "
                "(Low側は指定どおり)")
        gguf_low = f"Low/Index-Anisora-V3.2-Low-{quant}.gguf"
        qc = GGUFQuantizationConfig(compute_dtype=_pick_dtype())
        if lite:
            log(f"リファイン専用ロード: Low単体 {quant} (High省略で"
                "RAM/DL約9GB節約。VIDEOLAB_ANISORA_LITE=offで従来動作)")
            hi = None
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
            _peak = ((max(_gguf_gb(hi), _gguf_gb(lo)) if _early_off else gg)
                     + 12 + (0 if self._te_lazy else 11) + 6)
            _ram_gate(log, _peak, f"AniSora {quant} 読み込み")
        t_hi = None
        if hi is not None:
            log("transformer(High) 読み込み — configはWan2.2ベースを明示")
            t_hi = WanTransformer3DModel.from_single_file(
                hi, quantization_config=qc, config=snap,
                subfolder="transformer", torch_dtype=_pick_dtype())
            if _early_off:
                self._try_group_offload(t_hi, log, "transformer(早期退避)")
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
        # liteモードはtransformerにもLowを渡す (同一オブジェクトの別名)。
        # None渡しはWanImageToVideoPipeline.__call__内の属性参照で壊れる
        # 可能性があるため、実体共有=追加メモリゼロの別名が安全
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            snap, transformer=(t_hi if t_hi is not None else t_lo),
            transformer_2=t_lo,
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
        self.loaded = True

    def generate(self, req: GenRequest, workdir: Path, log, progress) -> Path:
        # 量子化/オフロード指定がロード時と違う場合は積み替え (共通設定を
        # ジョブ単位で反映するため。切替は数分かかるので必要時のみ)。
        # Low単体(lite)ロード中に通常i2vジョブが来たらHigh込みへ積み替え
        # (逆=High込みでリファインはそのまま賄える)
        want = self._resolve_want(req.extra)
        lite = self._lite_wanted(req.extra)
        if self.loaded and (
                want != (getattr(self, "loaded_quant", None),
                         getattr(self, "loaded_offload", None))
                or (getattr(self, "loaded_lite", False) and not lite)):
            log(f"設定変更 {self.loaded_quant}/{self.loaded_offload}"
                f"{'/Low単体' if getattr(self, 'loaded_lite', False) else ''}"
                f" -> {want[0]}/{want[1]}{'' if lite else '/High込み'}: "
                "モデルを積み替えます")
            self.unload(log)
            _free_cuda(log)
        if not self.loaded:
            self._next_extra = dict(req.extra or {})
            self.ensure_loaded(log)
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
        strength = float(req.extra.get("refine_strength", 0.45))
        strength = max(0.10, min(0.90, strength))
        steps = max(8, int(req.steps))          # スケジュール解像度(既定24)
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
        if (hi_t is not None
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

        def _set_ts(nsteps, device=None, **kw3):
            orig_set(nsteps, device=device, **kw3)
            sig = sched.sigmas                    # CPU float32, 末尾に終端σ
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
            x_t = (1.0 - s0) * x0 + s0 * noise
            log(f"リファイン: {len(frames)}f σ0={s0:.2f} 実行{tail}/{steps}"
                "step (全stepがLow=ディテール側)")
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
            if pin_slots:
                cb = _pin_step_callback(cb, sched, x0, noise,
                                        pin_slots, pin_release)
            kw = dict(image=cond_img,
                      width=w, height=h, num_frames=n,
                      num_inference_steps=steps,
                      guidance_scale=float(req.guidance),
                      latents=x_t,
                      generator=g,
                      output_type="np", return_dict=False,
                      callback_on_step_end=cb)
            kw.update(self._prompt_kwargs(self._build_prompt(req, log),
                                          req.negative or None,
                                          req.guidance, log))
            # latents未対応の古いdiffusersなら明示エラーにする (黙って
            # 落とすと「1段目を無視した素の生成」が静かに走る偽PASS)。
            # latent固定が要求されているときは callback_on_step_end も
            # 必須扱い — 黙って外すと「固定なしの再加工」が静かに走る
            out = _call_with_optional_kwargs(
                self.pipe, kw,
                [] if pin_slots else ["callback_on_step_end"], log)
        finally:
            sched.set_timesteps = orig_set
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
        return _frames_to_mp4(list(out[0][0]), req.fps, workdir, log)


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
      せず VACE-Fun側を維持。patch_mode="slice" はドナー重みの先頭16ch
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
        if k.startswith("patch_embedding."):
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
    kept_vace = sum(1 for k in tgt if k.startswith("vace_"))
    log(f"AniSora移植[{tag}]: {len(take)}キー移植"
        f"{f' (うち保存形式変換{len(coerced)})' if coerced else ''} / "
        f"スキップ{len(skipped)} "
        f"({', '.join(skipped[:3])}{' ...' if len(skipped) > 3 else ''}) / "
        f"vace_*温存 {kept_vace}キー")
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
        if lora not in ("lightning",):
            lora = ""
        if base != "anisora":
            # 移植なしでは experts/patch は不活性 — 正規化してノブ操作
            # だけの無駄な積み替え(数分)を防ぐ
            experts, patch = "both", "fun"
        else:
            # LightningはT2V(=fun)向け蒸留。移植ベースでは意味が無い
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
                _ram_gate(log, _vgg + (_donor if base == "anisora" else 0)
                          + 12 + (0 if self._te_lazy else 11) + 6,
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
                for tgt, tag, sub in ((t_hi, "High", "transformer"),
                                      (t_lo, "Low", "transformer_2")):
                    if experts != "both" and experts != tag.lower():
                        log(f"AniSora移植[{tag}]: スキップ "
                            f"(vace_experts={experts})")
                        continue
                    dq = quant
                    fname = f"{tag}/Index-Anisora-V3.2-{tag}-{dq}.gguf"
                    # AniSora High側の棚在庫はQ4_0/Q8_0のみ — 下位quant時は
                    # HighドナーだけQ4_0で代替 (2026-07-16 HF実測)
                    if (tag == "High" and dq not in ("Q4_0", "Q8_0")
                            and _hf_file_missing(self.anisora_gguf_repo,
                                                 fname)):
                        dq = "Q4_0"
                        fname = f"{tag}/Index-Anisora-V3.2-{tag}-{dq}.gguf"
                        log(f"AniSora移植[High]: {quant}の在庫が無いため"
                            "Q4_0ドナーで代替します")
                    log(f"AniSora GGUF DL: {self.anisora_gguf_repo} {fname}")
                    dp = _hf_download(self.anisora_gguf_repo, fname, log)
                    log(f"AniSora {tag} 読み込み (移植ドナー・config="
                        "Wan2.2-I2Vベースを明示)")
                    donor = WanTransformer3DModel.from_single_file(
                        dp, quantization_config=qc, config=self.base_repo,
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
                "指定してください (高速化は vace_lora=lightning を推奨)")
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
        log(f"生成開始: {w}x{h} {n}f steps={steps} cfg={req.guidance} "
            f"骨格制御{len(control)}f cond={kw['conditioning_scale']}"
            + (" [latent直出し]" if emit_latent else ""))
        try:
            out = _call_with_optional_kwargs(
                self.pipe, kw, ["callback_on_step_end", "conditioning_scale"],
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
    """VACE High -> native AniSora Lowを1本のlatent軌道で直結する。

    完成VACE動画をMP4/JPEG化してVAE再encode・再加ノイズする旧refineとは
    異なり、Wan2.2の通常のHigh/Low expert切替と同じUniPC scheduler上で
    noise latentを一度も終了させない。VACEは16ch、AniSora I2Vは
    latent16+参照条件20=36ch入力だが、両者のnoise predictionは16chなので
    scheduler stateをそのまま継続できる。
    """
    id = "vace_anisora_handoff"
    label = "VACE High → AniSora Low (latent直結・品質本命)"
    desc = ("前半だけOpenPose/VACEで構図と歩行を固定し、同じノイズlatentを"
            "後半の無改造AniSora Lowへ直接渡す。VACE HighにはLightning"
            "低step LoRAを維持。中間動画・VAE再encode・"
            "再ノイズ化なし。extra hybrid_boundary=0.90(既定) / 0.875。"
            "steps既定8 (6ではLow区間不足で黄変、2026-07-13実測)。")
    requires = ("Colab L4で十分 (Q4+動的キャンバス設計でほぼフル品質・"
                "料金はA100の約1/5。A100は最速だが贅沢品)。Q4でVACE High"
                "約8.5GB + AniSora Low約9GBを区間ごとにGPUへ載せ替え。"
                "High側へLightning低step LoRAを適用。DL約77GB。"
                "16GB未満のご家庭GPUはextra offloadでblock offload(実験的)")
    cache_repos = VACEAdapter.cache_repos + ("lightx2v/Wan2.2-Lightning",)
    disk_gb = 77
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
        q = _norm_quant(e.get("quant")
                        or os.environ.get("VIDEOLAB_VACE_QUANT", "Q4_0"))
        try:
            boundary = float(e.get("hybrid_boundary")
                             or os.environ.get("VIDEOLAB_HYBRID_BOUNDARY",
                                               "0.90"))
        except (TypeError, ValueError):
            boundary = 0.90
        return q, max(0.80, min(0.95, boundary))

    @staticmethod
    def _hybrid_lora(extra: dict) -> str:
        """VACE Highに使う低step LoRA。既定lightning、offで比較用無効。"""
        e = extra or {}
        value = str(e.get("vace_lora")
                    or os.environ.get("VIDEOLAB_VACE_LORA", "lightning"))
        value = value.strip().lower()
        return "off" if value in ("off", "none", "0", "false") else "lightning"

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
        qc = GGUFQuantizationConfig(compute_dtype=_pick_dtype())
        vname = (f"HighNoise/Wan2.2-VACE-Fun-A14B-high-noise-"
                 f"{quant}.gguf")
        log(f"latent直結ロード: VACE High {quant}")
        # ベース部品はsnapshot経由 (v0.8.8: Drive固定運転でもhandoffが動く
        # ように。configもここから読む)
        snap_vace = _snapshot_local(self.repo, log)
        snap_base = _snapshot_local(self.base_repo, log)
        vp = _hf_download(self.gguf_repo, vname, log)
        vhigh = WanVACETransformer3DModel.from_single_file(
            vp, quantization_config=qc, config=snap_vace,
            subfolder="transformer", torch_dtype=_pick_dtype())

        # ロード前RAMゲート (P0-2): VACE High + ドナー + AniSora Low の
        # 同時常駐がロードピーク (実測48GB級@Q4)。VACEサイズ+推定で判定
        _vgg1 = _gguf_gb(vp)
        if _vgg1 >= 1.0:
            _dn = 16.0 if quant == "Q8_0" else 9.3
            _ram_gate(log, _vgg1 + _dn * 2 + 12
                      + (0 if _low_ram_vm() else 11) + 6,
                      f"latent直結 {quant} 読み込み")
        # High段もアニメpriorから離れないよう、共有blockをAniSora Highへ
        # 移植する。後半はこのキメラを使わずnative Lowそのものへ切り替える。
        ah_name = f"High/Index-Anisora-V3.2-High-{quant}.gguf"
        aq = quant
        if aq not in ("Q4_0", "Q8_0") \
                and _hf_file_missing(self.anisora_gguf_repo, ah_name):
            # AniSora High側の棚在庫はQ4_0/Q8_0のみ (2026-07-16 HF実測)
            aq = "Q4_0"
            ah_name = f"High/Index-Anisora-V3.2-High-{aq}.gguf"
            log(f"AniSora High移植ドナー: {quant}の在庫が無いため"
                "Q4_0で代替します")
        log(f"latent直結ロード: AniSora High移植ドナー {aq}")
        ahp = _hf_download(self.anisora_gguf_repo, ah_name, log)
        donor = WanTransformer3DModel.from_single_file(
            ahp, quantization_config=qc, config=snap_base,
            subfolder="transformer", torch_dtype=_pick_dtype())
        _transplant_base_weights(vhigh, donor, log, tag="HybridHigh")
        del donor
        gc.collect()

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
        pipe_kwargs = dict(transformer=vhigh, transformer_2=None,
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
        if lora == "lightning":
            # 旧VACE経路で使っていた低step LoRAをHigh側へ維持する。
            # handoff後はnative AniSora Low（自身が蒸留済み）なので、VACE用
            # low_noise LoRAは重ねず、High用だけをVACE transformerへ装着。
            rep = os.environ.get("VIDEOLAB_VACE_LORA_REPO",
                                 "lightx2v/Wan2.2-Lightning")
            fold = os.environ.get(
                "VIDEOLAB_VACE_LORA_DIR",
                "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V2.0")
            log(f"latent直結: VACE HighへLightning低step LoRA適用 "
                f"({rep}/{fold})")
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
                    rep, f"{fold}/high_noise_model.safetensors", log))
                self.pipe.load_lora_weights(
                    str(_hi_p.parent), adapter_name="lightning",
                    weight_name=_hi_p.name)
                self.pipe.set_adapters(["lightning"], [1.0])
            except Exception as e:
                raise RuntimeError(
                    "latent直結のVACE HighへLightning LoRAを適用できません"
                    f"でした: {str(e)[:300]}")
            log("Lightning適用完了: VACE Highのみ（native AniSora Lowは"
                "蒸留済み重みのまま）")
        else:
            log("latent直結: VACE Highの低step LoRAは比較用に無効")
        # VAE/UMT5/tokenizer/schedulerは同一オブジェクトを共有。両repoのVAE
        # safetensorsは同一hashなのでlatent scaleの変換も不要。
        self.ani_pipe = WanImageToVideoPipeline.from_pretrained(
            snap_base, transformer=None, transformer_2=alow,
            vae=self.pipe.vae, text_encoder=self.pipe.text_encoder,
            tokenizer=self.pipe.tokenizer, scheduler=sched,
            torch_dtype=_pick_dtype())
        # load_lora_weights/from_pretrainedが内部で置いたdeviceに依存しない。
        # condition encode前はDiT 2体を必ずCPUへ戻し、VAE/UMT5用の空きを作る。
        for module in (vhigh, alow, self.pipe.vae, self.pipe.text_encoder):
            try:
                module.to("cpu")
            except Exception:
                pass
        _free_cuda(log)
        _log_cuda_state(log, "ロード後CPU待機")
        self.prompt_suffix = AniSoraAdapter.prompt_suffix
        self.loaded_quant = quant
        self.loaded_lora = lora
        self.loaded = True
        log("latent直結モデル準備完了: VACE High + native AniSora Low "
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
        if self.loaded and (quant, lora) != (
                getattr(self, "loaded_quant", None),
                getattr(self, "loaded_lora", None)):
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
        vmodel, amodel = pipe.transformer, apipe.transformer_2
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
            boundary_t = boundary * sched.config.num_train_timesteps
            high_count = sum(float(t) >= boundary_t for t in timesteps)
            if high_count <= 0 or high_count >= len(timesteps):
                raise RuntimeError(
                    f"hybrid_boundary={boundary} ではHigh/Lowの両区間を"
                    f"作れません (timesteps={len(timesteps)}, High={high_count})")
            log(f"latent直結生成: {w}x{h} {n}f / {steps}step — "
                f"VACE High {high_count}step → native AniSora Low "
                f"{len(timesteps) - high_count}step (境界{boundary:.3f})")

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
            for si, t in enumerate(timesteps):
                use_vace = float(t) >= boundary_t
                timestep = t.expand(latents.shape[0])
                if use_vace:
                    latent_input = latents.to(vmodel.dtype)
                    with vmodel.cache_context("cond"):
                        pred = vmodel(
                            hidden_states=latent_input, timestep=timestep,
                            encoder_hidden_states=prompt_embeds,
                            control_hidden_states=control_cond,
                            control_hidden_states_scale=scale,
                            return_dict=False)[0]
                    if guidance > 1.0:
                        with vmodel.cache_context("uncond"):
                            uncond = vmodel(
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
                            vmodel.to("cpu")
                            _free_cuda(log)
                        if not a_hooked and not swap_first:
                            self._move(amodel, device, log,
                                       "native AniSora Low")
                        switched = True
                        log("handoff完了: pixel化せずlatent16chとUniPC履歴を"
                            "native 36ch I2V入力へ接続")
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
                           getattr(pipe, "vae", None), vmodel, amodel):
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
    global CURRENT_MODEL
    while True:
        jid = WORK_Q.get()
        j = JOBS.get(jid)
        if not j or j.get("_cancel"):
            if j:
                j["status"] = "cancelled"
            continue
        # 孤児検知: キュー待ちの間にクライアントが消えたジョブは始めない
        if (j.get("_watch_poll")
                and time.time() - j.get("_last_poll", 0) > ORPHAN_SEC):
            j["status"] = "cancelled"
            j["detail"] = "クライアント切断により自動中止 (ポーリング途絶)"
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

                def _dl_watch(_a=adapter):
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
            # (読み込みブロック中に中止されたジョブの生成を始めない)
            if j.get("_cancel"):
                raise JobCancelled()

            j["status"] = "running"
            j["detail"] = ""
            j["started"] = time.time()
            workdir = WORK_ROOT / jid
            workdir.mkdir(parents=True, exist_ok=True)
            out = adapter.generate(j["_req"], workdir, log, progress)
            j["path"] = str(out)
            j["status"] = "done"
            j["progress"] = 1.0
            log(f"完了: {out}")
        except JobCancelled:
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


def _watchdog_scan(now: float | None = None) -> list:
    """生成watchdog (P1) の1スキャン: running/loading ジョブの鼓動が
    VIDEOLAB_STALL_MIN分 (既定15) 途絶えていたらstalled扱いにする。
    処理したジョブidのリストを返す (テスト用)。

    ハングしたworkerスレッドはPythonからkillできないため、ジョブを
    error化してクライアントの永久待ちを解き、VIDEOLAB_WATCHDOG_EXIT=1
    のときはプロセスごと自沈する (Colab自動運転の張り直しと対にすると
    「無応答のまま課金」の根治になる)。"""
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
        print(f"[{j['id']}] WATCHDOG: {int(age // 60)}分無応答 -> stalled",
              flush=True)
    if hit and os.environ.get("VIDEOLAB_WATCHDOG_EXIT") == "1":
        print("[videolab] VIDEOLAB_WATCHDOG_EXIT=1: ハング検知のため"
              "プロセスを自沈します (張り直しで復旧)", flush=True)
        os._exit(1)
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


# ---------------------------------------------------------------- FastAPI
def build_app(token: str | None):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, HTMLResponse

    app = FastAPI(title="SpriteMill VideoLab", version=__version__)

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
        for _j in JOBS.values():
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
    if not args.no_browser:
        try:
            import webbrowser
            threading.Timer(1.0, lambda: webbrowser.open(url)).start()
        except Exception:
            pass
    server.run()


if __name__ == "__main__":
    main()
