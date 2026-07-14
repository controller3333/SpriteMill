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
**推奨は L4**(料金はA100の約1/5。Q4量子化+動的キャンバス設計でほぼ
フル品質が出る)。A100は最速だが贅沢品。モデルごとの目安は起動後の
webUI のモデル説明欄に表示される。

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

    c2 = """# ---- 2) 依存の確定再起動 + Google Driveモデルキャッシュ (任意) ----
# ★再起動で Run All は一旦止まります。再接続後もう一度【すべてのセルを実行】を。
import os
_sentinel = '/content/_sm_videolab_restarted'
if not os.path.exists(_sentinel):
    open(_sentinel, 'w').close()
    print('依存確定のため一度ランタイムを再起動します… 再接続後もう一度 Run All')
    os.kill(os.getpid(), 9)
else:
    print('再起動済み — cell 3 へ進みます')
    # モデルは完全Drive固定 (2026-07-14指示「毎回DLは詰まることがわかった」):
    # MyDrive/SpriteMill_models から読み込む。HuggingFaceへは一切行かない
    # (ColabのIP/割当はHFに429で絞られ、毎セッションのDLが成立しないため)。
    # 初回のみDriveの認可ポップアップに「許可」が必要。
    os.environ['VIDEOLAB_DRIVE_ONLY'] = '1'
    import threading
    def _mount_drive():
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            p = '/content/drive/MyDrive/SpriteMill_models'
            os.makedirs(p, exist_ok=True)
            os.environ['VIDEOLAB_DRIVE_CACHE'] = p
            print('Driveモデルキャッシュ有効:', p)
        except Exception as e:
            print('⚠ Driveをマウントできませんでした:', str(e)[:80])
            print('⚠ モデルはDrive固定です — このセルを単独で再実行して'
                  '認可ポップアップに「許可」してください')
    threading.Thread(target=_mount_drive, daemon=True).start()"""

    c3 = "%%writefile videolab_server.py\n" + code

    c35 = """# ---- 3.5) モデルのDrive配置チェック (初回だけHFから取得) ----
# 配置済みなら数秒でスキップします (Run All 毎回実行しても安全)。
# 未配置のときだけ、HFから一度だけ取得してDriveへ配置します (30〜60分)。
# ※共有フォルダ利用の人 (配布リンクからショートカット追加) は常にスキップ
import videolab_server
videolab_server.populate_drive()"""

    c4 = """# ---- 4) サーバ起動 + トンネル公開 (URL/TOKEN が表示される) ----
import sys, importlib
import videolab_server
if "videolab_server" in sys.modules:      # 再実行時に最新コードを反映
    videolab_server = importlib.reload(videolab_server)
url, token = videolab_server.run_in_colab(preload=None)"""

    c5 = """# ---- 5) keep-alive + 生成進捗ゲージ (回しっぱなしでOK) ----
# 生成中はジョブごとに tqdm ゲージ (モデルDLと同じ見た目) がその場で伸びる
# (2026-07-13要望「上みたいなゲージが出るように」)。3秒ごとに /api/jobs を
# 確認し、待機中は5分ごとに alive を印字する。
import time, json, urllib.request
from tqdm.auto import tqdm
try:
    token
except NameError:
    token = ''
_bars = {}
_i = 0
print('keep-alive 開始。生成が始まるとここに進捗ゲージが出ます。')
while True:
    time.sleep(3)
    _i += 3
    try:
        _rq = urllib.request.Request(
            'http://127.0.0.1:8000/api/jobs',
            headers={'Authorization': 'Bearer ' + token} if token else {})
        with urllib.request.urlopen(_rq, timeout=10) as r:
            jobs = json.loads(r.read().decode()).get('jobs') or []
    except Exception:
        if _i % 300 == 0:
            print(f'alive {_i // 60} min  server=?? (応答なし)', flush=True)
        continue
    for j in jobs:
        jid = str(j.get('id') or '')
        st = str(j.get('status') or '')
        p = max(0.0, min(1.0, float(j.get('progress') or 0.0)))
        info = (st + ' ' + str(j.get('detail') or '').strip())[:48]
        if st in ('queued', 'loading', 'running'):
            if jid not in _bars:
                _bars[jid] = tqdm(
                    total=100, desc=f'生成 {jid[:8]}',
                    bar_format=('{desc} {percentage:3.0f}%|{bar}| '
                                '{elapsed} {postfix}'))
            b = _bars[jid]
            b.n = int(p * 100)
            b.set_postfix_str(info, refresh=False)
            b.refresh()
        elif jid in _bars:
            b = _bars.pop(jid)
            if st == 'done':
                b.n = 100
            b.set_postfix_str(st, refresh=False)
            b.refresh()
            b.close()
    if not _bars and _i % 300 == 0:
        print(f'alive {_i // 60} min  server=up (待機中)', flush=True)"""

    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "L4"},
            "accelerator": "GPU",
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": [_cell("markdown", md0), _cell("code", c1), _cell("code", c2),
                  _cell("code", c3), _cell("code", c35),
                  _cell("code", c4), _cell("code", c5)],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT} (server v{version}, {len(code.splitlines())} lines embedded)")
    print("※ GUIの「Colabでノートブックを開く」はGitHub上のノートを開きます。"
          "再生成したら github_repo/ へコピーして push するのを忘れずに "
          "(controller3333/SpriteMill)")


if __name__ == "__main__":
    main()
