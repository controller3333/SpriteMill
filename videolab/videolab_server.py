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

__version__ = "0.3.7"   # 0.3.7: 中間キーフレーム注入(往復回転の対策)
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
    "ftfy gguf peft",
]

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


def _purge_model_cache(repo: str, log):
    """指定リポのHFキャッシュを削除してディスクを空ける(Colabの容量対策)。

    VIDEOLAB_PURGE_ON_SWITCH=1 のとき、モデル切替でアンロードした側に適用。
    切替で戻ると再ダウンロードになるが、Colabの~200GBディスクでは
    60GB級モデルを2つ置いた時点で枯渇するため削除優先。
    """
    if not repo:
        return
    hub = Path(os.environ.get("HF_HOME",
                              str(Path.home() / ".cache" / "huggingface"))) / "hub"
    d = hub / ("models--" + repo.replace("/", "--"))
    if d.is_dir():
        try:
            # snapshots/ はblobsへのsymlinkなので実体のみ数える(二重カウント防止)
            sz = sum(f.lstat().st_size for f in d.rglob("*")
                     if f.is_file() and not f.is_symlink()) / 2**30
        except Exception:
            sz = 0
        shutil.rmtree(d, ignore_errors=True)
        log(f"ディスク解放: {repo} のキャッシュ削除 ({sz:.0f}GB)")


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

    def _finalize_pipe(self, log, offload: str = "", footprint_gb=None):
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
                if footprint_gb and total_gb >= footprint_gb + 6:
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
                # 常駐時はVAEタイリング不要(タイリングはVAEデコードを遅くする)
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
            "VIDEOLAB_ANISORA_QUANT=Q8_0(既定・計32GB)/Q4_0(計18GB・24GB級GPU向け)")
    requires = ("Colab A100/L4 / ローカル24GB+ (Q4_0) / 12GB級はQ4_0+"
                "逐次オフロード (低速・RAM32GB推奨)・DL 30〜45GB")
    gguf_repo = "QuantStack/Index-Anisora-V3.2-GGUF"
    cache_repos = ("QuantStack/Index-Anisora-V3.2-GGUF",)
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
        off = "seq" if off in ("seq", "sequential") else ""
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
        return super().generate(req, workdir, log, progress)


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
                        repos = (getattr(prev, "cache_repos", None)
                                 or [getattr(prev, "repo", "")])
                        for r in repos:
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
        except Exception as e:
            j["status"] = "error"
            j["detail"] = str(e)[:600]
            j["log"].append(traceback.format_exc()[-1500:])
            print(f"[{jid}] ERROR: {e}", flush=True)
        finally:
            j["finished"] = time.time()
            j.pop("_req", None)


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

    セルの再実行に強い設計: 既にサーバが動いていればトークンを再利用し
    (新規発行すると旧サーバとトークン不一致で全APIが401になる)、
    古い cloudflared だけ張り替える。
    """
    import re as _re
    import urllib.request as _rq
    # Colabのディスクは60GB級モデル2つで枯渇するため、切替時の自動削除を既定ON
    os.environ.setdefault("VIDEOLAB_PURGE_ON_SWITCH", "1")
    tok_file = Path(tempfile.gettempdir()) / "videolab_token.txt"
    alive = False
    try:
        with _rq.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
            alive = (r.status == 200)
    except Exception:
        pass
    if alive and tok_file.is_file():
        token = tok_file.read_text(encoding="utf-8").strip()
        print("既存のサーバを再利用します(トークン据え置き・トンネルのみ再作成)")
    else:
        token = secrets.token_urlsafe(16)
        tok_file.write_text(token, encoding="utf-8")
        server = start_server("127.0.0.1", port, token)
        threading.Thread(target=server.run, daemon=True).start()
        time.sleep(3)
    if preload and preload in ADAPTERS:
        def _pre():
            ADAPTERS[preload].ensure_loaded(lambda m: print(f"[preload] {m}", flush=True))
            globals()["CURRENT_MODEL"] = preload
        threading.Thread(target=_pre, daemon=True).start()
    subprocess.run(["pkill", "-f", "cloudflared"], capture_output=True)
    time.sleep(1)
    proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{port}",
         "--no-autoupdate"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    url = None
    deadline = time.time() + 60
    while time.time() < deadline and url is None:
        line = proc.stdout.readline()
        m = _re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line or "")
        if m:
            url = m.group(0)
    if not url:
        raise RuntimeError("cloudflared のトンネルURLが取得できませんでした。セルを再実行してください。")
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
