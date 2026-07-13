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

__version__ = "0.6.8"   # 0.6.8: refine_cond_still=リファイン条件画像を原画立ち絵に
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
    # hf_transfer=HF並列DL(Colab⇔HFで500MB/s超。60GB初回が1-2分)
    "ftfy gguf peft hf_transfer",
]

# hf_transferが入っていればHFダウンロードを並列化する (未導入なら従来DL。
# 「毎回DLよりドライブ読みが速いのでは」2026-07-13の疑問への回答は逆で、
# ColabのDriveマウントは30-80MB/s、HF直DLは実測280MB/s+hf_transferで更に上)
try:
    import hf_transfer  # noqa: F401
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
except ImportError:
    pass

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
    """連番PNG -> H.264 mp4(ブラウザ再生互換の yuv420p)。"""
    cmd = [ffmpeg, "-y", "-framerate", str(fps),
           "-i", str(frames_dir / "%05d.png"),
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
           "-movflags", "+faststart", str(dest)]
    r = subprocess.run(cmd, capture_output=True, text=True)
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
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


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

    def unload(self, log):
        self.pipe = None
        self.loaded = False

    def _finalize_pipe(self, log, offload: str = "", footprint_gb=None,
                       vae_tiling: bool = False,
                       resident_margin_gb: float = 6):
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

        # ---- オフロード戦略の決定 (速度の要。2026-07-12 A100で3分問題) ----
        # cuda  = 全モデルをGPU常駐(オフロード無し=最速。A100等でVRAMに
        #         余裕があるとき)。model_cpu_offloadはCPU<->GPU転送が挟まり
        #         A100の性能を殺すため、載るなら常駐が正解。
        # model = 主要モデルを順次GPUへ(中VRAM)。 seq = 逐次(12GB級)。
        mode = (offload or os.environ.get("VIDEOLAB_OFFLOAD", "")).lower()
        mode = {"sequential": "seq", "offload": "model", "full": "cuda",
                "none": "cuda"}.get(mode, mode)
        if mode not in ("seq", "model", "cuda"):
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
        (2026-07-12 お兄さま発案: 終端だけでは中盤で一回転して戻る
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
        img = _fit_image(req.images[0], w, h)
        kw = dict(image=img, prompt=self._build_prompt(req, log),
                  negative_prompt=req.negative or None,
                  width=w, height=h, num_frames=n,
                  num_inference_steps=steps,
                  guidance_scale=float(req.guidance),
                  generator=torch.Generator("cpu").manual_seed(req.seed),
                  output_type="np", return_dict=False,
                  callback_on_step_end=_step_callback(progress, steps))
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
            "VIDEOLAB_ANISORA_QUANT=Q8_0(既定・計32GB)/Q4_0(計18GB・24GB級GPU向け)。"
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
        q = str((extra or {}).get("quant")
                or os.environ.get("VIDEOLAB_ANISORA_QUANT", "Q8_0"))
        if q not in ("Q4_0", "Q8_0"):     # High側はQ4_0/Q8_0のみ存在
            q = "Q8_0"
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
        else:
            off = ""
        return q, off

    def ensure_loaded(self, log):
        _require_deps(log)
        import torch
        from diffusers import (GGUFQuantizationConfig, WanImageToVideoPipeline,
                               WanTransformer3DModel)
        from huggingface_hub import hf_hub_download
        quant, offload = self._resolve_want(getattr(self, "_next_extra", {}))
        gguf_high = f"High/Index-Anisora-V3.2-High-{quant}.gguf"
        gguf_low = f"Low/Index-Anisora-V3.2-Low-{quant}.gguf"
        qc = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)
        log(f"GGUF DL: {self.gguf_repo} {quant} "
            f"(High+Low 各{'15.9GB' if quant == 'Q8_0' else '9GB'})")
        hi = hf_hub_download(self.gguf_repo, gguf_high)
        lo = hf_hub_download(self.gguf_repo, gguf_low)
        log("transformer(High) 読み込み — configはWan2.2ベースを明示")
        t_hi = WanTransformer3DModel.from_single_file(
            hi, quantization_config=qc, config=self.base_repo,
            subfolder="transformer", torch_dtype=torch.bfloat16)
        log("transformer_2(Low) 読み込み")
        t_lo = WanTransformer3DModel.from_single_file(
            lo, quantization_config=qc, config=self.base_repo,
            subfolder="transformer_2", torch_dtype=torch.bfloat16)
        log(f"ベース部品 DL/読み込み: {self.base_repo} (VAE+UMT5 約12GB)")
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            self.base_repo, transformer=t_hi, transformer_2=t_lo,
            torch_dtype=torch.bfloat16)
        # GPU常駐可否の判定用フットプリント(GB): High+Low GGUF常駐 +
        # bf16テキストエンコーダ(~6) + VAE(~0.5) + 生成アクティベーション(~4)
        fp = {"Q4_0": 27, "Q8_0": 42}.get(quant, 42)
        self._finalize_pipe(log, offload=offload, footprint_gb=fp)
        self.loaded_quant, self.loaded_offload = quant, offload
        self.loaded = True

    def generate(self, req: GenRequest, workdir: Path, log, progress) -> Path:
        # 量子化/オフロード指定がロード時と違う場合は積み替え (共通設定を
        # ジョブ単位で反映するため。切替は数分かかるので必要時のみ)
        want = self._resolve_want(req.extra)
        if self.loaded and want != (getattr(self, "loaded_quant", None),
                                    getattr(self, "loaded_offload", None)):
            log(f"設定変更 {self.loaded_quant}/{self.loaded_offload} -> "
                f"{want[0]}/{want[1]}: モデルを積み替えます")
            self.unload(log)
            _free_cuda(log)
        if not self.loaded:
            self._next_extra = dict(req.extra or {})
            self.ensure_loaded(log)
        if req.extra.get("refine_frames_b64"):
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
        frames = load_images_b64(list(req.extra["refine_frames_b64"]))
        n = max(5, ((len(frames) - 1) // 4) * 4 + 1)             # 4k+1
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
        if hi_t is not None and _dev_of(hi_t) == "cuda":
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
        with torch.no_grad():
            vid = torch.cat(
                [pipe.video_processor.preprocess(f, height=h, width=w)
                 .unsqueeze(2) for f in frames], dim=2)
            vid = vid.to(device=dev, dtype=pipe.vae.dtype)
            enc = pipe.vae.encode(vid)
            x0 = (enc.latent_dist.mode() if hasattr(enc, "latent_dist")
                  else enc.latents)
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
            # 条件画像は既定で「1段目動画の先頭フレーム」。ただし
            # refine_cond_still 指定時は原画の立ち絵を条件にする:
            # 1段目の劣化(量子化+蒸留の粘土)がノイズ済みlatentの中では
            # 「正常な構造」に見えてしまい、リファインが安定した粘土に
            # 確定する (2026-07-13お兄さま指摘)。原画をframe0に注入すると
            # 時系列アテンションが劣化前の質感を全フレームから参照できる。
            # 前提=骨格の直立プレフィックス (frame0が直立なので立ち絵と
            # 矛盾しない。旧動画=全フレーム歩行に使うと先頭が跳ねる)
            cond_img = frames[0]
            if req.extra.get("refine_cond_still") and req.images:
                cond_img = _fit_image(req.images[0], w, h)
                log("リファイン条件画像: 原画立ち絵 (粘土を正常と誤認"
                    "させないための参照注入)")
            kw = dict(image=cond_img,
                      prompt=self._build_prompt(req, log),
                      negative_prompt=req.negative or None,
                      width=w, height=h, num_frames=n,
                      num_inference_steps=steps,
                      guidance_scale=float(req.guidance),
                      latents=x_t,
                      generator=g,
                      output_type="np", return_dict=False,
                      callback_on_step_end=_step_callback(progress, tail))
            # latents未対応の古いdiffusersなら明示エラーにする (黙って
            # 落とすと「1段目を無視した素の生成」が静かに走る偽PASS)
            out = _call_with_optional_kwargs(
                self.pipe, kw, ["callback_on_step_end"], log)
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
    お兄さま裁定)。v0.5.0からはコミュニティ定石「AniSoraベース+VACE
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
            "推奨)}。量子化は共通quant (Q8_0既定/Q4_0。AniSora GGUFはこの"
            "2種のみ)。bf16は移植なし=素のVACE-Fun")
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
        q = str(e.get("quant")
                or os.environ.get("VIDEOLAB_VACE_QUANT", "Q8_0"))
        if q not in ("Q4_0", "Q8_0", "bf16"):
            q = "Q8_0"
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

    def ensure_loaded(self, log):
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
        if quant == "bf16":
            # フル精度: 常駐70GB+81f attentionの活性化~9GBはA100-80にも
            # 入らない(2026-07-12実測: 残り324MiBでOOM)ため余白12GBで判定
            # -> 実質オフロード運用。素のVACE-Fun品質のA/B比較用の逃げ道
            # (AniSora移植はGGUF経路のみ。_resolve_wantがbase=funに固定)
            log(f"読み込み開始: {self.repo} (bf16・素のVACE-Fun・DL約70GB)")
            self.pipe = WanVACEPipeline.from_pretrained(
                self.repo, torch_dtype=torch.bfloat16)
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
            qc = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)
            log(f"GGUF DL: {self.gguf_repo} {quant} "
                f"(High+Low 各{'15.5GB' if quant == 'Q8_0' else '8.5GB'})")
            hi = hf_hub_download(self.gguf_repo, gguf_high)
            lo = hf_hub_download(self.gguf_repo, gguf_low)
            log("transformer(HighNoise) 読み込み — configはVACEベースを明示")
            t_hi = WanVACETransformer3DModel.from_single_file(
                hi, quantization_config=qc, config=self.repo,
                subfolder="transformer", torch_dtype=torch.bfloat16)
            log("transformer_2(LowNoise) 読み込み")
            t_lo = WanVACETransformer3DModel.from_single_file(
                lo, quantization_config=qc, config=self.repo,
                subfolder="transformer_2", torch_dtype=torch.bfloat16)
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
                    fname = f"{tag}/Index-Anisora-V3.2-{tag}-{quant}.gguf"
                    log(f"AniSora GGUF DL: {self.anisora_gguf_repo} {fname}")
                    dp = hf_hub_download(self.anisora_gguf_repo, fname)
                    log(f"AniSora {tag} 読み込み (移植ドナー・config="
                        "Wan2.2-I2Vベースを明示)")
                    donor = WanTransformer3DModel.from_single_file(
                        dp, quantization_config=qc, config=self.base_repo,
                        subfolder=sub, torch_dtype=torch.bfloat16)
                    _transplant_base_weights(tgt, donor, log, tag=tag,
                                             patch_mode=patch)
                    del donor
                    gc.collect()
            else:
                log("vace_base=fun: AniSora移植なし (素のVACE-Fun。"
                    "steps30/cfg5.0を明示推奨)")
            log(f"ベース部品 DL/読み込み: {self.repo} (VAE+UMT5 約12GB)")
            self.pipe = WanVACEPipeline.from_pretrained(
                self.repo, transformer=t_hi, transformer_2=t_lo,
                torch_dtype=torch.bfloat16)
            # GGUF常駐 + UMT5(~6.7GB) + VAE + 81f骨格制御の活性化(~9GB)
            fp = {"Q4_0": 27, "Q8_0": 42}.get(quant, 42)
            margin = 10
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
                self.pipe.load_lora_weights(
                    rep, adapter_name="lightning",
                    weight_name=f"{fold}/high_noise_model.safetensors")
                self.pipe.load_lora_weights(
                    rep, adapter_name="lightning_2",
                    weight_name=f"{fold}/low_noise_model.safetensors",
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
                            vae_tiling=True, resident_margin_gb=margin)
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

        kw = dict(video=control, reference_images=[ref],
                  prompt=self._build_prompt(req, log),
                  negative_prompt=req.negative or self.WAN_NEGATIVE,
                  width=w, height=h, num_frames=n,
                  num_inference_steps=steps,
                  guidance_scale=float(req.guidance),
                  conditioning_scale=float(
                      req.extra.get("conditioning_scale", 1.0)),
                  generator=torch.Generator("cpu").manual_seed(req.seed),
                  output_type="np", return_dict=False,
                  callback_on_step_end=cb)
        log(f"生成開始: {w}x{h} {n}f steps={steps} cfg={req.guidance} "
            f"骨格制御{len(control)}f cond={kw['conditioning_scale']}")
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
        return _frames_to_mp4(list(out[0][0]), req.fps, workdir, log)


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

    def ensure_loaded(self, log):
        _require_deps(log)
        import torch
        from diffusers import WanImageToVideoPipeline
        log(f"読み込み開始: {self.base_repo} (DL約126GB・初回は20分前後)")
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            self.base_repo, torch_dtype=torch.bfloat16)
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
            print(f"[{_j['id']}] {msg}", flush=True)

        def progress(p, _j=j):
            _j["progress"] = round(float(p), 4)
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
                adapter.ensure_loaded(log)
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
            j["status"] = "error"
            j["detail"] = str(e)[:600]
            j["log"].append(traceback.format_exc()[-1500:])
            print(f"[{jid}] ERROR: {e}", flush=True)
            # OOM等の失敗断片を回収してから次のジョブへ
            _free_cuda(log)
        finally:
            j["finished"] = time.time()
            j.pop("_req", None)


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
        return {"ok": True, "app": "SpriteMill VideoLab", "version": __version__,
                "auth": bool(token), "queued": WORK_Q.qsize(),
                "current_model": CURRENT_MODEL, "gpu": gpu, "disk": disk,
                "libs": libs}

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
        if mode in ("i2v", "multikey") and not images:
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
        return FileResponse(j["path"], media_type="video/mp4",
                            filename=f"videolab_{jid}.mp4")

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
    if not token:
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
