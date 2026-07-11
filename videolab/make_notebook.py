#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_notebook.py — videolab_server.py から Colab ノートブックを生成する。

単一ソース方針: サーバ実装は videolab_server.py だけを正とし、
colab/SpriteMill_video_lab.ipynb はこのスクリプトで再生成する(手編集禁止)。

    python make_notebook.py            # ../colab/SpriteMill_video_lab.ipynb を更新
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVER = HERE / "videolab_server.py"
OUT = HERE.parent / "colab" / "SpriteMill_video_lab.ipynb"


def _cell(kind: str, src: str) -> dict:
    lines = src.splitlines(keepends=True)
    if kind == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": lines}


def main() -> None:
    code = SERVER.read_text(encoding="utf-8")
    m = re.search(r'^__version__\s*=\s*"([^"]+)"', code, re.M)
    version = m.group(1) if m else "?"
    # サーバ本体から Colab 用 pip 行を拾う(COLAB_PIP = [...] を定義しておく)
    mp = re.search(r"^COLAB_PIP\s*=\s*(\[[^\]]*\])", code, re.M | re.S)
    pip_lines = "\n".join(f"!pip -q install {x}" for x in eval(mp.group(1))) if mp else \
        "!pip -q install fastapi uvicorn pillow"

    md0 = f"""# SpriteMill VideoLab (Colab動画生成サーバ) v{version}

モデル差し替え式の動画生成サーバ + webUI。**LTX動画モデル**などをColab GPUで動かし、
ブラウザから直接試したり、SpriteMill本体の動画AIとして接続したりできる。

**ランタイム**: `ランタイム → ランタイムのタイプを変更` で **GPU** を選ぶ。
モデルごとの目安は起動後の webUI のモデル説明欄に表示される
(大型モデルは A100、量子化版は L4/T4 でも可)。

**使い方(Run All を2回)**
1. `すべてのセルを実行` → セル2で一度だけ自動再起動される。
2. 再接続後もう一度 `すべてのセルを実行`。
3. 最後のセルに表示される **webUI の URL** をブラウザで開く(TOKEN付きリンク)。
4. SpriteMill 本体から使う場合は URL と TOKEN をアプリに貼り付け。
5. keep-alive セルは回しっぱなしで放置。

※ 生成された動画・トークンはこのColabセッション内にのみ存在し、
セッション終了で消える。必要な動画は webUI からダウンロードすること。"""

    c1 = f"""# ---- 1) セットアップ: GPU確認 / 依存 / cloudflared ----
!nvidia-smi -L
{pip_lines}
!wget -q -O /usr/local/bin/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
!chmod +x /usr/local/bin/cloudflared
print('setup done -- 次の cell 2 が一度だけ自動再起動します')"""

    c2 = """# ---- 2) 依存を確定させる一度きりの自動再起動 ----
# ★再起動で Run All は一旦止まります。再接続後もう一度【すべてのセルを実行】を。
import os
_sentinel = '/content/_sm_videolab_restarted'
if not os.path.exists(_sentinel):
    open(_sentinel, 'w').close()
    print('依存確定のため一度ランタイムを再起動します… 再接続後もう一度 Run All')
    os.kill(os.getpid(), 9)
else:
    print('再起動済み — cell 3 へ進みます')"""

    c3 = "%%writefile videolab_server.py\n" + code

    c4 = """# ---- 4) サーバ起動 + トンネル公開 (URL/TOKEN が表示される) ----
import videolab_server
url, token = videolab_server.run_in_colab(preload=None)"""

    c5 = """# ---- 5) keep-alive: アイドル切断防止 (回しっぱなしでOK) ----
import time, urllib.request
_i = 0
print('keep-alive 開始。放置でOK。')
while True:
    time.sleep(300)
    _i += 5
    try:
        with urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=10) as r:
            _ok = (r.status == 200)
    except Exception:
        _ok = False
    print(f'alive {_i} min  server={"up" if _ok else "??"}', flush=True)"""

    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "A100"},
            "accelerator": "GPU",
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": [_cell("markdown", md0), _cell("code", c1), _cell("code", c2),
                  _cell("code", c3), _cell("code", c4), _cell("code", c5)],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT} (server v{version}, {len(code.splitlines())} lines embedded)")
    print("※ GUIの「Colabでノートブックを開く」はGitHub上のノートを開きます。"
          "再生成したら github_repo/ へコピーして push するのを忘れずに "
          "(controller3333/SpriteMill)")


if __name__ == "__main__":
    main()
