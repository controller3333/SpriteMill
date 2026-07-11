#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Colabノートブックの自動運転 -- VideoLabサーバをボタン一発で起動する。

設計は web_drive.py と同じ「CDPアタッチ + 壊れても止まらない」方針:
  - 自動化フラグの付かない普通のEdgeを常駐させ、CDPで接続だけする
    (Googleログインはこの永続プロファイルに保存され、Gemini等の
     ブラウザログインと共有される = 追加ログイン不要のことが多い)。
  - 各工程はベストエフォート。セレクタ/キーが効かなければ案内を出して
    人間に委ね、最後の URL/TOKEN 回収は画面テキストの監視で拾うので、
    途中を手で操作しても正しく回収できる(=半自動に劣化するだけ)。

進捗は print() で標準出力に流す(GUIがストリーム表示)。成功時は
{"url":..., "token":...} を返し、失敗時は例外。

注意: Colabのランタイム種別(GPU)は「ノートブック×アカウント」ごとに
記憶されるため、初回だけ手動でGPU(A100/L4)を選べば以後は自動で足りる。
本ドライバはランタイム種別の変更までは行わない(メニューダイアログが
Colabのバージョンで揺れやすく、誤操作リスクが高いため)。
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

TRYCF = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
TOKEN_Q = re.compile(r"[?&]token=([A-Za-z0-9_\-]+)")
TOKEN_LINE = re.compile(r"TOKEN\s*[:：]\s*([A-Za-z0-9_\-]+)")


def _web_drive():
    """web_drive の Edge/CDP ヘルパを借りる(frozenでは hiddenimport、
    devでは engine/ を都度パスに追加)。"""
    try:
        import web_drive
        return web_drive
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "engine"))
        import web_drive
        return web_drive


def extract_url_token(text: str) -> tuple[str, str] | None:
    """Colab最終セルの出力から trycloudflare URL と TOKEN を拾う。"""
    m = TRYCF.search(text or "")
    if not m:
        return None
    url = m.group(0)
    tok = ""
    mt = TOKEN_Q.search(text)
    if mt:
        tok = mt.group(1)
    else:
        ml = TOKEN_LINE.search(text)
        if ml:
            tok = ml.group(1)
    return url, tok


class ColabDriver:
    def __init__(self, profile_dir: Path, notebook_url: str):
        wd = _web_drive()
        self._wd = wd
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        if not wd.ensure_debug_edge(profile_dir, notebook_url,
                                    open_url=True):
            raise RuntimeError(
                "運転用Edgeを起動できませんでした(Edge未検出/ポート不通)")
        self.browser = self._pw.chromium.connect_over_cdp(
            f"http://127.0.0.1:{wd.DEBUG_PORT}")
        ctx = (self.browser.contexts[0] if self.browser.contexts
               else self.browser.new_context())
        # Colabタブを探す(無ければ既存タブで開く)
        self.page = None
        for p in ctx.pages:
            try:
                if "colab.research.google.com" in (p.url or ""):
                    self.page = p
                    break
            except Exception:
                continue
        if self.page is None:
            self.page = ctx.pages[0] if ctx.pages else ctx.new_page()
            self.page.goto(notebook_url, wait_until="domcontentloaded",
                           timeout=60000)
        self.notebook_url = notebook_url

    def _text(self) -> str:
        try:
            return self.page.inner_text("body", timeout=5000)
        except Exception:
            return ""

    def _needs_login(self) -> bool:
        u = ""
        try:
            u = self.page.url or ""
        except Exception:
            pass
        return ("accounts.google.com" in u
                or "ServiceLogin" in u)

    def _click_any(self, patterns: list[str]) -> bool:
        """テキストが patterns のどれかに一致するボタンをクリック。"""
        rx = re.compile("|".join(patterns), re.I)
        for sel in ("paper-button", "mwc-button", "button",
                    ".goog-buttonset-default", "[role='button']"):
            try:
                loc = self.page.locator(sel)
                n = min(loc.count(), 40)
                for i in range(n):
                    el = loc.nth(i)
                    try:
                        if el.is_visible(timeout=200) and rx.search(
                                el.inner_text(timeout=200) or ""):
                            el.click(timeout=2000)
                            return True
                    except Exception:
                        continue
            except Exception:
                continue
        return False

    def _run_all(self, log) -> None:
        """すべてのセルを実行(Ctrl+F9)。GitHubノート警告が出たら承認。"""
        # GitHub由来ノートの「このまま実行」警告を先に潰す
        self._click_any(["run anyway", "そのまま実行", "このまま実行",
                         "実行する", "run all"])
        try:
            self.page.bring_to_front()
        except Exception:
            pass
        self.page.keyboard.press("Control+F9")
        time.sleep(2)
        # キー直後にも警告/接続ダイアログが出ることがある
        if self._click_any(["run anyway", "そのまま実行", "このまま実行",
                            "実行する"]):
            log("  GitHubノートの実行警告を承認しました")
            time.sleep(1)
            self.page.keyboard.press("Control+F9")

    def run(self, log, poll_timeout: int = 2400) -> dict:
        # 1) ロード待ち + ログイン確認
        log("Colabノートブックを開いています…")
        for _ in range(30):
            if "colab-run-button" in self._text_html_probe():
                break
            if self._needs_login():
                raise RuntimeError(
                    "Googleにログインしていません。開いたEdgeの窓で"
                    "Googleアカウントにサインインし、ノートブックが表示"
                    "されてからもう一度お試しください(ログインは保存され、"
                    "次回からは自動になります)")
            time.sleep(1)
        # 2) 1回目のRun all
        log("すべてのセルを実行します(1回目)…")
        self._run_all(log)
        # 3) セル2の自動再起動を待つ → 2回目のRun all
        log("依存確定のためのランタイム再起動を待っています"
            "(数分かかることがあります)…")
        restarted = self._wait_restart(log, timeout=600)
        if restarted:
            log("再起動を検知。すべてのセルを実行します(2回目)…")
            time.sleep(3)
            self._run_all(log)
        else:
            log("  再起動を検知できませんでした。Colab画面で"
                "「すべてのセルを実行」を一度押してください(自動回収は続行)")
        # 4) URL/TOKEN をポーリング回収
        log("サーバ起動待ち: URL と TOKEN を探しています"
            "(初回はモデルDLで時間がかかります)…")
        deadline = time.time() + poll_timeout
        while time.time() < deadline:
            got = extract_url_token(self._text())
            if got and got[0]:
                log(f"URL/TOKEN を取得しました: {got[0]}")
                return {"url": got[0], "token": got[1]}
            time.sleep(5)
        raise TimeoutError(
            "URL/TOKENを取得できませんでした。Colab画面の最後のセルに"
            "URLが出ているか確認し、出ていれば手でアプリの欄に貼って"
            "ください")

    def _text_html_probe(self) -> str:
        try:
            return self.page.content()[:200000]
        except Exception:
            return ""

    def _wait_restart(self, log, timeout: int) -> bool:
        """cell2の os.kill による再起動を検知する。実行が一旦止まり、
        ランタイムが再接続されるのを、状態バー文言と接続ボタンで判定。"""
        end = time.time() + timeout
        saw_running = False
        while time.time() < end:
            t = self._text()
            # 「再起動」「クラッシュ」「再接続」等の痕跡
            if re.search(r"再起動|restart|crashed|クラッシュ|reconnect|再接続",
                         t, re.I):
                return True
            if re.search(r"実行中|running|busy", t, re.I):
                saw_running = True
            # 実行が始まってから一旦止まったら再起動とみなす
            if saw_running and not re.search(r"実行中|running|busy", t, re.I):
                time.sleep(3)
                return True
            time.sleep(3)
        return False

    def close(self) -> None:
        try:
            self.browser.close()   # 常駐Edgeは閉じない(CDP接続だけ解放)
        except Exception:
            pass


def drive_colab(profile_dir: Path, notebook_url: str, log=print,
                poll_timeout: int = 2400) -> dict:
    """公開エントリ: ノートを開いて起動し {'url','token'} を返す。"""
    drv = None
    try:
        drv = ColabDriver(Path(profile_dir), notebook_url)
        return drv.run(log, poll_timeout=poll_timeout)
    finally:
        if drv is not None:
            drv.close()


if __name__ == "__main__":   # 手動テスト用
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="webview_profile")
    ap.add_argument("--url", required=True)
    a = ap.parse_args()
    print(drive_colab(Path(a.profile), a.url))
