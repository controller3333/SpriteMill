#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""canvas_walk.py -- 4方向(または8方向)動画法。

イラスト歩行フレーム(03_walk_frames/<dir>/00_idle..05_walk.png)を1枚の
キャンバス(グリッド)に並べ、キーフレーム拘束の i2v (/generate_multikey 契約)
で **全方向を1回の生成**にまとめ、方向別セルに切り出す。

- 生成数を 1/N (4方向=1/4, 8方向=1/8) に削減。
- 中間に「一歩(w1..w5)」ポーズを挟むので、回転せず実際に歩く(2026-07-09検証)。
- POST先が /generate_multikey (keys_b64のリスト) を満たせばプロバイダ非依存。
- 出力は build_T_sheet_from_mp4 互換の {char}_{idx}_{d}_walkT.mp4 なので、
  既存の QC(inspect_walk_mp4) と組み立ては無改造で再利用できる。
"""
from __future__ import annotations

import base64
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

MAGENTA = (255, 0, 255)

# レイアウト = (cols, rows, [方向の並び順]). 立ち絵キャンバスとセル分割で共用。
LAYOUT_4 = (2, 2, ["front", "back", "left", "right"])           # = 4P (主要4)
LAYOUT_4P = (2, 2, ["front", "back", "left", "right"])          # 主要4方向
# ★前後半球グルーピング (2026-07-18): 旧4x2は「主要4+斜め4」で、斜め前が
# 後ろ姿2セルと同じキャンバス・同じ生成を共有していた。斜め前の参照
# 立ち絵が正面寄りのキャラ (Codexの斜め45°の壁、真ロップ実測ヨー-13°)
# では顔の骨格信号が弱く、歩行開始フレームでセル間の見た目共有と
# 「歩き去る3/4」事前分布に吸われて頭だけ後頭部化する (20260717_2232
# 真ロップ: 奥耳遮蔽・ヨー追従・透視・compass角セル明文の全対策後も
# 8試行全滅、反転は毎回歩行開始=f6で発生)。キャンバスを「顔が見える
# 4方向」と「顔を見せない3方向+横顔」に分けると、斜め前の隣から後ろ姿が
# 消え、プロンプトでも「全員の顔が最後まで見える」を汚染なしで宣言できる
# (rear語彙の同居汚染は実測済み: compass上段へrear系3/4語彙を足した版は
# 下段の角まで後ろ向きに化けた 2026-07-18)。横顔は全モードで崩れない
# 実績があるため、どちらの半球に入れても安全。
LAYOUT_F4 = (2, 2, ["front", "left", "front_left", "front_right"])   # 前半球
LAYOUT_B4 = (2, 2, ["back", "right", "back_left", "back_right"])     # 後半球
LAYOUT_8 = (4, 2, ["front", "front_left", "front_right", "left",
                   "back", "back_left", "back_right", "right"])  # 8-in-1(低解像度・行=半球)
# 3x3 コンパス配置(2026-07-10): 各方向を向く方位のセルに置き中央は空白(None)。
# index i -> (i%3, i//3) が下の並びのままコンパス位置になる。None は compose で
# 空セル扱い・split でスキップ。pipeline.COMPASS_GRID と対応。
LAYOUT_COMPASS = (3, 3, ["back_left", "back", "back_right",
                         "left", None, "right",
                         "front_left", "front", "front_right"])
# 方向 -> シート内インデックス(pipeline.py DIRECTIONS と一致させる)
IDX = {"front": 1, "left": 2, "right": 3, "back": 4,
       "front_left": 5, "front_right": 6, "back_left": 7, "back_right": 8}
# 8方向の生成戦略
MODES = {
    "4x2": [LAYOUT_F4, LAYOUT_B4],   # 前半球 + 後半球 の2生成(高解像度・生成数1/4)
    "4only": [LAYOUT_4P],            # 主要4方向のみ(1生成)
    "8x1": [LAYOUT_8],               # 8-in-1 の1生成(生成数1/8・低解像度)
    "compass": [LAYOUT_COMPASS],     # 3x3コンパス(中央空白)の1生成(2026-07-10)
}
# idle + 歩行5コマ = 1周期。始点=終点=idle なので生成物はループする。
FULL_CYCLE = ["00_idle.png", "01_walk.png", "02_walk.png", "03_walk.png",
              "04_walk.png", "05_walk.png", "00_idle.png"]

# i2v は「歩き」を回転/オービットで見せたがる。肯定文で「その場足踏み・固定
# カメラ・向き固定」を強く指定(両プロバイダ)。Grokはネガティブを無視するので
# 抑制はすべて肯定文に畳む。Veoには別途 NEGATIVE_ROTATION を parameters で渡す。
CANVAS_PROMPT = (
    "A game character sprite sheet: the same chibi character shown in several "
    "fixed compass cells. Each figure marches in place with an alternating-leg "
    "walk cycle, like walking on a treadmill -- only the legs and arms move, "
    "feet lift and step without travelling. Every figure stays centered in its "
    "own cell and keeps facing its own fixed direction the entire time; the body "
    "and head never turn, rotate, spin, or change direction. Static locked-off "
    "tripod camera, fixed frame, no camera movement, no pan, no zoom, no orbit. "
    "Plain flat magenta background, smooth seamless looping walk animation."
)
# Veo parameters.negativePrompt 用(Grokは非対応)。net の回転/並進/カメラ移動を
# 狙い、"walking"/"movement" 単体は入れない(歩行自体を消さないため)。
NEGATIVE_ROTATION = (
    "rotating, turning around, spinning, pirouette, twirl, 360 turn, turntable, "
    "revolving, changing facing direction, character turning, head turn, body "
    "turn, facing away, back view appearing, side profile appearing, camera "
    "orbit, camera pan, camera rotation, camera roll, dolly, tracking shot, arc "
    "shot, moving camera, drifting, zoom in, zoom out, walking forward, moving "
    "across frame, sliding"
)


def compose_canvas(frame_paths: list[Path], cols: int, rows: int,
                   bg: tuple[int, int, int] = MAGENTA) -> Image.Image:
    """方向別フレーム画像を cols x rows グリッドに中央寄せで敷き詰める。"""
    ims = [Image.open(p).convert("RGB") for p in frame_paths]
    cw = max(im.width for im in ims)
    ch = max(im.height for im in ims)
    canvas = Image.new("RGB", (cw * cols, ch * rows), bg)
    for i, im in enumerate(ims):
        x = (i % cols) * cw + (cw - im.width) // 2
        y = (i // cols) * ch + (ch - im.height) // 2
        canvas.paste(im, (x, y))
    return canvas


def compose_dir_canvas(dir_to_path: dict, layout,
                       bg: tuple[int, int, int] = MAGENTA) -> Image.Image:
    """dir->画像パス の対応から layout グリッドのキャンバスを作る(欠けは空セル)。
    単一画像プロバイダ(Grok/Google)向け=立ち絵を並べて1枚絵で渡す用途。"""
    cols, rows, dirs = layout
    ims = []
    for d in dirs:
        p = dir_to_path.get(d)
        ims.append(Image.open(p).convert("RGB")
                   if p and Path(p).is_file() else None)
    cw = max((im.width for im in ims if im), default=64)
    ch = max((im.height for im in ims if im), default=128)
    canvas = Image.new("RGB", (cw * cols, ch * rows), bg)
    for i, im in enumerate(ims):
        if im is None:
            continue
        x = (i % cols) * cw + (cw - im.width) // 2
        y = (i // cols) * ch + (ch - im.height) // 2
        canvas.paste(im, (x, y))
    return canvas


def run_canvas_walk_single(gen_video_fn, dir_to_path: dict, mode: str,
                           ffmpeg: str, mp4_dir: Path, char_id: str,
                           work_dir: Path, prompt: str, seed: int = 0
                           ) -> list[Path]:
    """単一画像 i2v プロバイダ(Grok/Google等)向け: 立ち絵をキャンバスに並べ、
    1枚絵として `gen_video_fn(image_path, prompt, dest)` で歩行動画化し、方向別
    mp4 ({char}_{idx}_{d}_walkT.mp4) に分割。これらのモデルは拘束なしでも安定
    して歩くのでキーフレーム(multikey)は使わない。生成数を N方向→モード分
    (4x2=2本, 8x1=1本)に削減してクレジット消費を抑える。書いたパス一覧を返す。"""
    work_dir.mkdir(parents=True, exist_ok=True)
    mp4_dir.mkdir(parents=True, exist_ok=True)

    def _one_layout(gi_layout) -> list[Path]:
        gi, layout = gi_layout
        cols, rows, dirs = layout
        present = [d for d in dirs
                   if d and dir_to_path.get(d) and Path(dir_to_path[d]).is_file()]
        if not present:
            print(f"  canvas(1枚) pass {gi}: 立ち絵なし、スキップ", flush=True)
            return []
        canvas = compose_dir_canvas(dir_to_path, layout)
        cpath = work_dir / f"stand_canvas_g{gi}.png"
        canvas.save(cpath)
        cvid = work_dir / f"stand_canvas_g{gi}_walk.mp4"
        print(f"  canvas(1枚) pass {gi}: {present} -> 動画1本生成", flush=True)
        gen_video_fn(cpath, prompt, cvid)
        return split_canvas_video(ffmpeg, cvid, layout, mp4_dir,
                                  char_id, IDX, dirs_subset=present)

    written: list[Path] = []
    items = list(enumerate(MODES[mode]))
    if len(items) <= 1:
        for it in items:
            written += _one_layout(it)
    else:
        # 4x2 等の複数キャンバスは並列生成(2026-07-10ユーザー要望: 動画も
        # 画像と同じ並列キュー。API側処理なのでコストは変わらない)
        from concurrent.futures import ThreadPoolExecutor as _TPE
        with _TPE(max_workers=len(items)) as _ex:
            for w in _ex.map(_one_layout, items):
                written += w
    return written


def compose_pose_canvases(frames_root: Path, layout, poses: list[str],
                          out_dir: Path) -> list[Path]:
    """各ポーズ(00_idle..05_walk)ごとに、layout の全方向を並べたキャンバスを作る。
    キーフレーム列(ポーズ順)のPNGパスを返す。"""
    cols, rows, dirs = layout
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for pose in poses:
        frame_paths = [frames_root / d / pose for d in dirs]
        missing = [p for p in frame_paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"canvas frames missing: {missing}")
        canvas = compose_canvas(frame_paths, cols, rows)
        p = out_dir / f"canvas_{pose}"
        canvas.save(p)
        paths.append(p)
    return paths


def _api_post(url: str, token: str | None, payload: dict, timeout: int = 120):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _api_get(url: str, token: str | None, timeout: int = 60):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def generate_canvas_video(endpoint: str, token: str | None,
                          canvas_paths: list[Path], prompt: str, dest: Path,
                          seed: int = 0, poll_timeout: int = 2400) -> None:
    """キャンバスのキーフレーム列を /generate_multikey に投げ、mp4 を dest に落とす。"""
    base = endpoint.rstrip("/")
    keys = [base64.b64encode(p.read_bytes()).decode("ascii") for p in canvas_paths]
    payload = _api_post(f"{base}/generate_multikey", token,
                        {"keys_b64": keys, "prompt": prompt, "seed": seed},
                        timeout=300)
    job = payload.get("job")
    if not job:
        raise RuntimeError(f"canvas video: no job id: {str(payload)[:300]}")
    deadline = time.time() + poll_timeout
    while time.time() < deadline:
        try:
            st = json.loads(_api_get(f"{base}/status/{job}", token))
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  poll transient error ({e}); retrying", flush=True)
            time.sleep(10)
            continue
        s = st.get("status")
        if s == "done":
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(_api_get(f"{base}/result/{job}", token, timeout=300))
            return
        if s == "error":
            raise RuntimeError(f"canvas video job failed: {st.get('detail')}")
        time.sleep(10)
    raise TimeoutError(f"canvas video poll timeout for job {job}")


def _probe_size(ffprobe: str, mp4: Path) -> tuple[int, int]:
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(mp4)],
        creationflags=0x08000000 if sys.platform == "win32" else 0,
        capture_output=True, text=True, check=True).stdout.strip()
    w, h = (int(x) for x in out.split(",")[:2])
    return w, h


def split_canvas_video(ffmpeg: str, canvas_mp4: Path, layout, out_dir: Path,
                       char_id: str, idx_of: dict[str, int],
                       dirs_subset: list[str] | None = None) -> list[Path]:
    """キャンバス動画を方向別セルにクロップし、build_T_sheet 互換の
    {char_id}_{idx:02d}_{d}_walkT.mp4 として書き出す。書いたパス一覧を返す。"""
    cols, rows, dirs = layout
    out_dir.mkdir(parents=True, exist_ok=True)
    ffprobe = ffmpeg[:-len("ffmpeg")] + "ffprobe" if ffmpeg.endswith("ffmpeg") \
        else "ffprobe"
    w, h = _probe_size(ffprobe, canvas_mp4)
    cw, ch = w // cols, h // rows
    written: list[Path] = []
    win = 0x08000000 if sys.platform == "win32" else 0
    for i, d in enumerate(dirs):
        if d is None:                    # コンパスの空セル(中央)はスキップ
            continue
        if dirs_subset and d not in dirs_subset:
            continue
        x, y = (i % cols) * cw, (i // cols) * ch
        out = out_dir / f"{char_id}_{idx_of[d]:02d}_{d}_walkT.mp4"
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(canvas_mp4),
             "-filter:v", f"crop={cw}:{ch}:{x}:{y}", "-an", str(out)],
            check=True, creationflags=win)
        written.append(out)
    return written


def run_canvas_walk(frames_root: Path, mode: str, endpoint: str,
                    token: str | None, ffmpeg: str, mp4_dir: Path,
                    char_id: str, seed: int, work_dir: Path,
                    poses: list[str] = FULL_CYCLE,
                    prompt: str = CANVAS_PROMPT) -> list[Path]:
    """1キャラの歩行を「モード分の生成(4x2なら2回)」でまとめて作り、方向別
    mp4 ({char}_{idx}_{d}_walkT.mp4) を mp4_dir に書く。書いたパス一覧を返す。
    frames_root には各方向の 00_idle..05_walk.png が要る(codexイラスト歩行)。"""
    work_dir.mkdir(parents=True, exist_ok=True)
    mp4_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for gi, layout in enumerate(MODES[mode]):
        cols, rows, dirs = layout
        # このパスに存在する方向だけを対象に(欠け方向はスキップ)
        present = [d for d in dirs if (frames_root / d / poses[0]).is_file()]
        if not present:
            print(f"  canvas pass {gi}: 対象方向のフレームなし、スキップ", flush=True)
            continue
        sub_layout = (cols, rows, dirs)   # 位置は固定(欠けはマゼンタ空セル)
        cdir = work_dir / f"canvas_g{gi}"
        canvases = compose_pose_canvases(frames_root, sub_layout, poses, cdir)
        cvid = work_dir / f"canvas_g{gi}_walk.mp4"
        print(f"  canvas pass {gi}: {present} -> generate ({len(canvases)} keys)",
              flush=True)
        generate_canvas_video(endpoint, token, canvases, prompt, cvid, seed=seed)
        written += split_canvas_video(ffmpeg, cvid, sub_layout, mp4_dir,
                                      char_id, IDX, dirs_subset=present)
    return written


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames-root", required=True,
                    help="03_walk_frames dir (各方向に 00_idle..05_walk.png)")
    ap.add_argument("--mp4-dir", required=True, help="方向別mp4の出力先")
    ap.add_argument("--work-dir", required=True, help="キャンバス/中間の作業先")
    ap.add_argument("--char", default="C01")
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--token", default=None)
    ap.add_argument("--mode", choices=tuple(MODES), default="4x2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ffmpeg", default="ffmpeg")
    args = ap.parse_args()
    written = run_canvas_walk(
        Path(args.frames_root), args.mode, args.endpoint, args.token,
        args.ffmpeg, Path(args.mp4_dir), args.char, args.seed,
        Path(args.work_dir))
    print(f"CANVAS_WALK done: {len(written)} direction mp4(s)")
    for p in written:
        print(f"  {p}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
