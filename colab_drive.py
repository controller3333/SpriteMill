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
    """Colab最終セルの出力から trycloudflare URL と TOKEN を拾う。

    URLとTOKENは必ず【同じ行】から対で取る(文書全体から別々に拾うと、
    古い出力のURLと新しいTOKENのような偽ペアを作ってしまう)。複数の
    webUI行が見えている場合は最後(=最下部・最新の出力)を優先する。
    """
    text = text or ""
    best = None
    for line in text.splitlines():
        m = TRYCF.search(line)
        if not m:
            continue
        tok = ""
        mt = TOKEN_Q.search(line)
        if mt:
            tok = mt.group(1)
        # token付きの行を最優先、無ければURLだけの行も候補として保持
        if tok or best is None or not best[1]:
            best = (m.group(0), tok)
    if best is None:
        return None
    if not best[1]:
        # 同一行にtokenが無い形式(URL行とTOKEN行が分かれている)への保険。
        # 最後の TOKEN : 行を対にする(出力は上から古い順なので最後=最新)
        toks = TOKEN_LINE.findall(text)
        if toks:
            best = (best[0], toks[-1])
    return best


def _classify_conn(t: str) -> str:
    """colab-connect-button の表示テキストから接続状態を分類する。

    実DOM採取(2026-07-12、WebView2+CDPでColab実機を観測)による実測値:
      未接続       : 「接続 A100 arrow_drop_down」「再接続 A100 …」
                     「新しいランタイムに接続する」
      割り当て中   : 「more_horiz 接続中 arrow_drop_down」
      接続済み(暇) : 「done arrow_drop_down」 (doneはチェックアイコンの
                     リガチャ文字。RAM/ディスクゲージは別要素だった)
      実行中       : 「more_horiz arrow_drop_down」 (接続中の文字なし)
    """
    if not t:
        return "unknown"
    if re.search(r"接続中|しています|connecting|割り当て|allocat|初期化"
                 r"|initializ", t, re.I):
        return "connecting"   # 「再接続しています」等の進行形もここ
    if re.search(r"\bdone\b|RAM|ディスク|disk", t, re.I):
        return "connected"
    if re.search(r"more_horiz", t, re.I):
        return "connected"    # セル実行中のビジー表示(接続はしている)
    if re.search(r"再接続|reconnect|接続|connect", t, re.I):
        return "disconnected"
    return "unknown"


WEBVIEW_CDP_PORT = 9456   # sprite_mill._webview_main と一致させること


def _port_up(port: int) -> bool:
    import urllib.request as _rq
    try:
        with _rq.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=2):
            return True
    except Exception:
        return False


def _probe_health(url: str, timeout: float = 4.0):
    """トンネルURLの /health に届けばそのJSONを返す(不達はNone)。
    v0.9.0: Drive固定運転の状態(drive.only/mounted/ready)も載ってくるので
    ⚡自動運転のマウント承諾待ちゲートに使う。"""
    import json as _json
    import urllib.request as _rq
    try:
        with _rq.urlopen(f"{url}/health", timeout=timeout) as r:
            try:
                return _json.loads(r.read().decode()) or {"ok": True}
            except Exception:
                return {"ok": True}
    except Exception:
        return None


_PW = None
_PW_TID = None


def _playwright():
    """Playwrightドライバの取得。同一スレッド内でだけ使い回す。

    sync版Playwrightは start() したスレッドに束縛され、GUIの⚡は毎回
    新しいワーカースレッドで走るため、前回のドライバを別スレッドから
    使うと「cannot switch to a different thread (which happens to have
    exited)」で必ず落ちる (2026-07-14実障害: ⚡の2回目で毎回発生)。
    スレッドが変わっていたら旧ドライバのnodeプロセスをbest-effortで
    直接killして(stop()も旧スレッド束縛なので呼べない)作り直す。"""
    global _PW, _PW_TID
    import threading as _th
    tid = _th.get_ident()
    if _PW is not None and _PW_TID != tid:
        try:
            # 内部のnodeドライバ子プロセスを直接落とす (Popen.killは
            # スレッド安全。greenlet経由のstop()は死んだスレッド束縛で
            # 呼べないため、孤児化させないにはこれしかない)
            proc = getattr(getattr(getattr(_PW, "_impl_obj", None),
                                   "_connection", None), "_transport", None)
            proc = getattr(proc, "_proc", None)
            if proc is not None:
                proc.kill()
        except Exception:
            pass
        _PW = None
    if _PW is None:
        from playwright.sync_api import sync_playwright
        _PW = sync_playwright().start()
        _PW_TID = tid
    return _PW


def _launch_webview_host(notebook_url: str, log=print) -> bool:
    """完全内蔵のWebView2窓 (SpriteMill --webview) を起動してCDPを待つ。

    Edge運転で「普段使い側の拡張機能が全部無効化される」事故が起きたため
    (2026-07-12 配布テスト)、既定はこちら。WebView2はブラウザのEdgeとは
    別ランタイム・別プロファイルで、普段の環境に一切干渉しない。"""
    import subprocess
    if _port_up(WEBVIEW_CDP_PORT):
        return True
    if getattr(sys, "frozen", False):
        cmd = [sys.executable, "--webview", notebook_url]
    else:
        host = Path(__file__).resolve().parent / "sprite_mill.py"
        cmd = [sys.executable, str(host), "--webview", notebook_url]
    try:
        subprocess.Popen(cmd)
    except OSError as e:
        log(f"内蔵ブラウザを起動できません: {e}")
        return False
    deadline = time.time() + 40
    while time.time() < deadline:
        if _port_up(WEBVIEW_CDP_PORT):
            return True
        time.sleep(0.5)
    return False


class ColabDriver:
    def __init__(self, profile_dir: Path, notebook_url: str,
                 browser: str = "webview2", log=print):
        wd = _web_drive()
        self._wd = wd
        self._pw = _playwright()
        cdp_port = None
        if browser == "webview2":
            if _launch_webview_host(notebook_url, log):
                cdp_port = WEBVIEW_CDP_PORT
                log("内蔵ブラウザ(WebView2)で開きました")
            else:
                log("内蔵ブラウザの起動に失敗 -- Edge運転にフォールバック"
                    "します (config videolab_browser で切替可)")
        if cdp_port is None:
            try:
                ok = wd.ensure_debug_edge(profile_dir, notebook_url,
                                          open_url=True, app_mode=True)
            except TypeError:   # 旧web_drive (app_mode未対応) との互換
                ok = wd.ensure_debug_edge(profile_dir, notebook_url,
                                          open_url=True)
            if not ok:
                raise RuntimeError(
                    "運転用ブラウザを起動できませんでした"
                    "(WebView2/Edgeともに不通)")
            cdp_port = wd.DEBUG_PORT
        self.browser = self._pw.chromium.connect_over_cdp(
            f"http://127.0.0.1:{cdp_port}")
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

    def _click_any(self, patterns: list[str], selectors=None) -> bool:
        """テキストが patterns のどれかに一致するボタンをクリック。

        ColabはMaterial 3へ移行済みで、GitHubノートの実行警告
        「このまま実行」等は md-text-button (2026-07-12 実DOMで確認。
        旧セレクタでは一度も押せていなかった=自動運転不発の主因)。"""
        rx = re.compile("|".join(patterns), re.I)
        for sel in (selectors or ("md-text-button", "md-filled-button",
                                  "md-outlined-button",
                                  "md-filled-tonal-button",
                                  "paper-button", "mwc-button", "button",
                                  ".goog-buttonset-default",
                                  "[role='button']")):
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

    # ---- ランタイム状態の観測 (2026-07-12: 2回目Run All空振り対策) ----

    def _js(self, expr: str, default=None):
        try:
            return self.page.evaluate(expr)
        except Exception:
            return default

    def _conn_state(self) -> str:
        """接続ボタン(colab-connect-button)の状態。shadow DOM 込み。"""
        t = self._js(
            "(() => {"
            "  const b = document.querySelector('colab-connect-button');"
            "  if (!b) return '';"
            "  const s = b.shadowRoot ? (b.shadowRoot.textContent || '') : '';"
            "  return (s + ' ' + (b.textContent || '') + ' '"
            "          + (b.getAttribute('aria-label') || '')).trim();"
            "})()", default="") or ""
        return _classify_conn(t)

    def _dialogs_text(self) -> str:
        """ダイアログ/トースト(左下のクラッシュ通知含む)のテキストを集める。

        cell2のos.killはカーネルだけを殺しVM接続は切れないため、右上の
        接続ボタンは変化せず、左下の「予期せぬクラッシュ」通知が唯一の
        確実な再起動シグナル(2026-07-12ユーザー観測)。通知系の要素を
        広めに拾い、ノートのセル本文・出力は除外する(セルのソースにも
        『クラッシュ』の語が書かれているため)。"""
        return self._js(
            "(() => {"
            "  const sels = \"paper-toast, mwc-snackbar, md-snackbar,"
            " colab-toast, [role='alert'], [role='status'],"
            " [role='alertdialog'], [role='dialog'], mwc-dialog, md-dialog,"
            " paper-dialog\";"
            "  const out = [];"
            "  for (const el of document.querySelectorAll(sels)) {"
            "    if (el.closest('.cell, .notebook-content, .monaco-editor,"
            " colab-static-output-renderer')) continue;"
            "    const t = (el.innerText || el.textContent || '').trim();"
            "    if (t) out.push(t);"
            "  }"
            "  return out.join('\\n');"
            "})()", default="") or ""

    def _cells_running(self) -> bool:
        """セルが実行中/実行待ちかを判定する。

        実測(2026-07-12): colab-run-button の shadowRoot 直下の className が
        実行待ち=「cell-execution … pending」、実行中=「… animating running」
        になる(アイドルは stale / stale error 等)。"""
        return bool(self._js(
            "(() => {"
            "  for (const b of document.querySelectorAll("
            "       'colab-run-button')) {"
            "    const c = b.shadowRoot && b.shadowRoot.firstElementChild"
            "      ? b.shadowRoot.firstElementChild.className : '';"
            "    if (/running|pending|animating|queued/i.test(c))"
            "      return true;"
            "  }"
            "  if (document.querySelector("
            "      '.cell.running, .cell.pending, .cell.executing'))"
            "    return true;"
            "  return false;"
            "})()", default=False))

    def _crash_snackbar_open(self) -> bool:
        """左下のクラッシュ通知(スナックバー)が開いているか。

        実測(2026-07-12): colab-snackbar#message-area 内の
        div.mdc-snackbar が表示中は mdc-snackbar--open クラスを持ち、
        文言は「不明な理由により、セッションがクラッシュしました。」。
        通知は自動では消えず、閉じるまでDOMに残り続けるため、
        検知側は開始時に一度掃除してから監視する。"""
        return bool(self._js(
            "(() => {"
            "  const walk = (root) => {"
            "    for (const el of root.querySelectorAll("
            "         '.mdc-snackbar--open')) {"
            "      if (/クラッシュ|crash/i.test(el.textContent || ''))"
            "        return true;"
            "    }"
            "    for (const el of root.querySelectorAll('*'))"
            "      if (el.shadowRoot && walk(el.shadowRoot)) return true;"
            "    return false;"
            "  };"
            "  return walk(document);"
            "})()", default=False))

    def _close_crash_snackbar(self) -> None:
        """開きっぱなしのクラッシュ通知を閉じる(残骸の掃除用)。"""
        self._js(
            "(() => {"
            "  const walk = (root) => {"
            "    for (const el of root.querySelectorAll("
            "         '.mdc-snackbar--open')) {"
            "      if (!/クラッシュ|crash/i.test(el.textContent || ''))"
            "        continue;"
            "      const b = el.querySelector("
            "        '.mdc-snackbar__dismiss, [class*=\"dismiss\"],"
            " md-icon-button, button');"
            "      if (b) b.click();"
            "    }"
            "    for (const el of root.querySelectorAll('*'))"
            "      if (el.shadowRoot) walk(el.shadowRoot);"
            "  };"
            "  walk(document);"
            "})()", default=None)

    _DIALOG_BTNS = ("mwc-dialog button", "mwc-dialog mwc-button",
                    "md-dialog button", "paper-dialog paper-button",
                    "[role='dialog'] button",
                    "[role='dialog'] [role='button']",
                    "[role='alertdialog'] button")

    def _dismiss_info_dialogs(self) -> bool:
        """クラッシュ通知などの情報ダイアログを OK/閉じる で潰す。"""
        return self._click_any(
            [r"^\s*OK\s*$", r"^\s*閉じる\s*$", r"^\s*了解\s*$",
             r"^\s*Close\s*$", r"^\s*Dismiss\s*$"],
            selectors=self._DIALOG_BTNS)

    RUN_WARN = ["run anyway", "そのまま実行", "このまま実行", "実行する"]

    def _run_all(self, log, tries: int = 3) -> bool:
        """すべてのセルを実行し、実行が本当に始まったかを確認。

        起動はツールバーの実行ボタン(「ノートブック内のすべてのセルを
        実行」md-text-button、2026-07-12 実DOMで確認)のクリックを優先し、
        見つからなければ Ctrl+F9 にフォールバック。始まっていなければ
        ダイアログを潰し直して再試行する。"""
        for attempt in range(1, tries + 1):
            self._dismiss_info_dialogs()
            self._click_any(self.RUN_WARN)   # 先に出ている実行警告
            try:
                self.page.bring_to_front()
            except Exception:
                pass
            if self._cells_running():
                # keep-aliveセル等が回っているとRun Allのキューがその後ろに
                # 詰まって永遠に進まない -- 先に実行を中断してから流す
                log("  実行中のセルを中断してからRun Allします")
                try:
                    self.page.keyboard.press("Escape")
                    self.page.keyboard.press("Control+m")
                    self.page.keyboard.press("i")
                except Exception:
                    pass
                time.sleep(2)
            clicked = self._click_any(
                [r"すべてのセルを実行", r"run\s*all"])
            if not clicked:
                try:
                    # ダイアログ残骸でキーが届かないのを防いでから送る
                    self.page.keyboard.press("Escape")
                    self.page.keyboard.press("Control+F9")
                except Exception:
                    time.sleep(2)
                    continue
            end = time.time() + 15
            while time.time() < end:
                if self._click_any(self.RUN_WARN):
                    log("  GitHubノートの実行警告を承認しました")
                if self._cells_running() or self._conn_state() == "connecting":
                    return True
                time.sleep(1)
            if attempt < tries:
                log(f"  実行開始を確認できません -- 再試行します"
                    f"({attempt}/{tries})")
        return False

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
        # 1.5) サーバが既に生きているなら何も実行せずそのまま採用する。
        # Run Allを押すと (a)キューが無限ループのkeep-aliveセルの後ろに
        # 詰まり (b)キュー投入時にセル出力=URL行が消去される
        # (2026-07-12ユーザー観測「5が回ってるせいで4が実行できてない」)
        got = self._fresh_url_token()
        if got:
            log(f"サーバは既に稼働中です -- そのまま採用: {got[0]}")
            return {"url": got[0], "token": got[1]}
        # 1.7) ランタイムタイプの確認を先に人間へ (2026-07-14要望「GPUの
        # 設定をどのタイミングでやればいいかいつも迷う」— 自動化が走る前に
        # 設定画面を開いて、閉じられるまで待つ)。ヘッドレス検証は
        # prompt_runtime=False でスキップ
        if getattr(self, "prompt_runtime", True):
            self._prompt_runtime_type(log)
        # 2) 1回目のRun all
        log("すべてのセルを実行します(1回目)…")
        if not self._run_all(log):
            log("  実行開始を確認できませんでした。始まっていないようなら"
                "Colab画面で「すべてのセルを実行」を押してください"
                "(自動処理は続行)")
        # 3) セル2の自動再起動を待つ → 再接続の完了を確認して2回目
        log("依存確定のためのランタイム再起動を待っています"
            "(数分かかることがあります)…")
        res = self._wait_restart(log, timeout=600)
        if res == "server-up":
            log("再起動済みのランタイムでした -- そのままサーバ起動を"
                "待ちます")
        elif res == "restarted":
            log("再接続の完了を確認。すべてのセルを実行します(2回目)…")
            time.sleep(3)
            if not self._run_all(log):
                log("  2回目の実行開始を確認できませんでした。Colab画面で"
                    "「すべてのセルを実行」を押してください(自動回収は続行)")
        elif res == "no-gpu":
            raise RuntimeError(
                "GPUランタイムに接続できませんでした(割り当て不可または"
                "使用量上限)。Colab画面でランタイム種別(GPU)を確認するか、"
                "時間を置いてもう一度お試しください")
        else:
            log("  再起動を検知できませんでした。1回目の実行がまだ進行中"
                "ならそのまま待ち、止まっているようならColab画面で"
                "「すべてのセルを実行」を押してください(自動回収は続行)")
        # 4) URL/TOKEN をポーリング回収 (+Drive固定運転の準備完了ゲート)
        log("サーバ起動待ち: URL と TOKEN を探しています"
            "(初回はモデルDLで時間がかかります)…")
        deadline = time.time() + poll_timeout
        while time.time() < deadline:
            got = self._fresh_url_token()
            if got:
                # Drive固定 (v0.8.8+) のサーバは、マウント承諾やモデル配置が
                # 済むまで生成できない。/healthのdrive状態で人間待ちにする
                # (2026-07-14要望「ポップアップまで自動→人間が承諾したら
                # 続きを進める」)
                d = ((getattr(self, "_last_health", None) or {})
                     .get("drive") or {})
                if d.get("only") and not d.get("ready"):
                    self._nudge_drive(log, mounted=bool(d.get("mounted")))
                    deadline = max(deadline, time.time() + 600)
                    time.sleep(5)
                    continue
                log(f"URL/TOKEN を取得しました: {got[0]}")
                return {"url": got[0], "token": got[1]}
            time.sleep(5)
        raise TimeoutError(
            "URL/TOKENを取得できませんでした。Colab画面の最後のセルに"
            "URLが出ているか確認し、出ていれば手でアプリの欄に貼って"
            "ください")

    def _nudge_drive(self, log, mounted: bool) -> None:
        """Drive固定運転の人間待ちアナウンス (状態が変わったときだけ)。"""
        if mounted:
            state = "populate"
            msg = ("モデルをDriveから配置中です — このまま待ちます"
                   " (初回セットアップ中なら30〜60分)")
        else:
            dt = self._dialogs_text()
            if re.search(r"ドライブ|Drive", dt or "", re.I):
                state = "consent"
                msg = ("🖐 Google Driveの接続許可ポップアップが出ています — "
                       "内蔵ブラウザで「Google ドライブに接続」を押して"
                       "ください (承諾されるまで待機します)")
            else:
                state = "unmounted"
                msg = ("Driveマウント待ち — 内蔵ブラウザに認可ポップアップが"
                       "出たら「許可」を押してください (待機します)")
        if getattr(self, "_last_nudge", "") != state:
            self._last_nudge = state
            log(msg)

    def _rt_dialog_open(self) -> bool:
        """ランタイムタイプ設定ダイアログが開いているか。

        Playwrightのtext=ロケータで判定する — shadow DOMを貫通するため。
        (旧実装のJS innerText収集はshadow DOMを貫通せず検出漏れ →
        パレット掃除のEscapeがダイアログを1秒で閉じる実バグ 2026-07-14。
        文言は「ハードウェア アクセラレータ」限定: 「ランタイムのタイプ」は
        ノート冒頭の説明文にも書かれており誤検出する)"""
        try:
            loc = self.page.locator(
                "text=/ハードウェア\\s*アクセラレータ|Hardware accelerator/i")
            for i in range(min(loc.count(), 5)):
                if loc.nth(i).is_visible(timeout=200):
                    return True
        except Exception:
            pass
        return False

    def _prompt_runtime_type(self, log, timeout: int = 600) -> None:
        """「ランタイムのタイプを変更」ダイアログを開いて人間の確認を待つ。

        開く手段はコマンドパレット (Ctrl+Shift+P) — メニューDOMより
        UI変更に強い。開けなければ従来どおり続行する (best effort、
        2026-07-14要望「できるなら」)。閉じられるまで待機=保存/キャンセル
        どちらでも人間が確認したとみなす。"""
        try:
            try:
                self.page.bring_to_front()
            except Exception:
                pass
            opened = False
            for query in ("ランタイムのタイプ", "runtime type"):
                self.page.keyboard.press("Control+Shift+P")
                time.sleep(0.8)
                self.page.keyboard.type(query, delay=25)
                time.sleep(0.8)
                self.page.keyboard.press("Enter")
                for _ in range(10):
                    time.sleep(0.5)
                    if self._rt_dialog_open():
                        opened = True
                        break
                if opened:
                    break
                # Escapeはダイアログも閉じてしまうため、掃除する前に
                # もう一度だけ検出を試す (描画遅延の保険)
                time.sleep(2)
                if self._rt_dialog_open():
                    opened = True
                    break
                self.page.keyboard.press("Escape")
                time.sleep(0.3)
            if not opened:
                log("  ランタイムタイプ画面を自動で開けませんでした — "
                    "必要ならメニューから確認してください (続行します)")
                return
            log("🖐 GPUタイプを確認して「保存」を押してください (推奨: L4) — "
                "押されるまで待機します")
            deadline = time.time() + timeout
            while time.time() < deadline:
                if not self._rt_dialog_open():
                    log("ランタイムタイプの確認を受領 — 続行します")
                    return
                time.sleep(2)
            log("  確認待ちがタイムアウトしました — そのまま続行します")
        except Exception as e:   # noqa: BLE001
            log(f"  ランタイムタイプ確認をスキップ ({str(e)[:80]})")

    def _scroll_to_server_output(self) -> bool:
        """トンネルURLを印字するセル(サーバ起動セル)を画面内に出す。

        Colabは画面外のセル出力を仮想化してDOMから外すため
        (lazy-virtualized、2026-07-12実測: bodyテキストにwebUI行が
        存在しなかった)、スクロールして表示させないと inner_text に
        URL行が現れない。手動コピーが動いていたのは人間が見るために
        スクロールしていたからだった。"""
        try:
            c = self.page.locator(".cell", has_text="run_in_colab(").last
            c.scroll_into_view_if_needed(timeout=3000)
            time.sleep(0.6)   # 仮想化解除の描画待ち
            return True
        except Exception:
            return False

    def _fresh_url_token(self):
        """画面のURL/TOKENを、/health への到達を確認してから返す。

        画面に見えているURLでも信用しない: 前セッションの残骸・
        作り直しで死んだトンネル・DNS未浸透の新トンネルはすべて
        「この端末から応答が取れない」ので、probeが通ったものだけを
        採用する(2026-07-12: 収集したURLがgetaddrinfo failedで全滅する
        事故が続いたため、検証をURLの新旧判定から到達性そのものに変更)。
        失敗したURLは30秒間probeを抑制して回線を無駄にしない。"""
        self._scroll_to_server_output()   # 仮想化された出力を呼び戻す
        got = extract_url_token(self._text())
        if not got or not got[0]:
            return None
        u = got[0]
        now = time.time()
        neg = getattr(self, "_url_probe_neg", None)
        if neg is None:
            neg = self._url_probe_neg = {}
        if now < neg.get(u, 0.0):
            return None
        h = _probe_health(u)
        if h:
            self._last_health = h   # Drive状態ゲート用 (v0.9.0)
            return got
        neg[u] = now + 30
        return None

    def _text_html_probe(self) -> str:
        try:
            return self.page.content()[:200000]
        except Exception:
            return ""

    # GPU割り当て失敗ダイアログの文言(接続前にしか検査しない)
    NO_GPU_RX = (r"GPU\s*(に|への)?接続できません|Cannot connect to a GPU"
                 r"|割り当てられません|使用量上限|usage limits?"
                 r"|バックエンドに接続できません")

    def _wait_restart(self, log, timeout: int) -> str:
        """cell2 の自動再起動を「ランタイム接続状態」の遷移で検知する。

        以前は画面テキスト中の『再起動/再接続』等を探していたが、ノート
        自身の説明文・セルのソースに同じ語が最初から表示されており、
        開始直後に誤検知 → 2回目のRun Allが早すぎて空振りしていた
        (2026-07-12 ユーザー報告)。ここでは接続ボタンの
        connected → 切断 → connected(再接続完了) を待ってから返す。
        接続が確認できている間(=cell1実行中)は期限を延長するので、
        依存インストールが長引いても timeout で誤誘導しない。

        返り値: "restarted"=再接続完了 / "server-up"=再起動不要で
        サーバURLが既に出ている / "no-gpu"=GPU割り当て失敗 /
        "timeout"=検知できず。"""
        start = time.time()
        end = start + timeout
        hard_end = start + max(timeout * 3, 1800)   # 延長の上限
        seen_connected = False
        went_down = False
        crash_seen = False
        down_since = 0.0
        down_polls = 0
        last = ""
        # クラッシュ通知は閉じるまでDOMに残るため、開始時に残骸を掃除して
        # から監視する(以後に開いた通知=本物。2026-07-12実DOM調査)
        if self._crash_snackbar_open():
            log("  前回のクラッシュ通知の残骸を閉じます")
            self._close_crash_snackbar()
            time.sleep(1)
        sticky = self._crash_snackbar_open()   # 閉じられなかった場合の保険
        while time.time() < min(end, hard_end):
            # 再開ランタイム等で再起動が不要なら(新しい)URLが先に出る
            if self._fresh_url_token():
                return "server-up"
            st = self._conn_state()
            if st != last and st != "unknown":
                log(f"  ランタイム状態: {st}")
                last = st
            dtxt = self._dialogs_text()
            snack = self._crash_snackbar_open()
            if sticky:
                if not snack:
                    sticky = False   # 一度閉じた -> 以後に開けば本物
                snack = False
            crashed = snack or bool(
                re.search(r"クラッシュ|crash|再起動|restart", dtxt, re.I))
            # 開始直後(接続前かつ60秒以内)のクラッシュ表示は前セッションの
            # 残骸なので掃除して無視。それ以外のクラッシュ通知は、接続
            # ボタンが変化しなくても再起動の確定シグナルとして扱う
            # (os.killはカーネルのみ殺しVM接続は切れない=右上は変化しない。
            #  2026-07-12ユーザー観測: 左下の「予期せぬクラッシュ」通知のみ)
            if crashed and not seen_connected and time.time() - start < 60:
                self._dismiss_info_dialogs()
                self._close_crash_snackbar()
                crashed = False
            if (not seen_connected
                    and re.search(self.NO_GPU_RX, dtxt, re.I)):
                return "no-gpu"
            if not went_down:
                if st == "connected":
                    seen_connected = True
                    down_polls = 0
                    # cell1(依存インストール)が長引いても待てるように、
                    # 接続を確認できている間は期限を延長する
                    end = max(end, time.time() + 300)
                elif seen_connected and st in ("disconnected", "connecting"):
                    down_polls += 1   # 一瞬の揺らぎは無視(2回で確定)
                if crashed or down_polls >= 2:
                    went_down = True
                    crash_seen = crashed
                    down_since = time.time()
                    log("  ランタイムの再起動を検知 -- カーネルの復帰を"
                        "待ちます…")
            else:
                crash_seen = crash_seen or crashed
                if crashed or st == "disconnected":
                    self._dismiss_info_dialogs()   # クラッシュ通知を閉じる
                    self._close_crash_snackbar()   # (通知は自動では消えない)
                if st == "connected":
                    time.sleep(2)                  # 状態の安定を確認
                    if self._conn_state() == "connected":
                        # 回線の揺らぎとの区別: 本物の再起動なら実行キューは
                        # 全部消えている。まだセルが動いていれば cell1 継続
                        # 中の一時断なので、再起動待ちに戻る
                        if not crash_seen and self._cells_running():
                            log("  一時的な接続の揺らぎでした -- "
                                "再起動待ちを続けます")
                            went_down = False
                            down_polls = 0
                            end = max(end, time.time() + 300)
                            continue
                        # クラッシュ通知の直後はカーネルがまだ復帰中で
                        # Run Allが空振りしやすい(2026-07-12ユーザー指摘:
                        # 「出た直後だと反応しないかも。5〜10秒待機を」)
                        # -- 検知から最低10秒は寝かせてから返す
                        if time.time() - down_since < 10:
                            continue
                        return "restarted"
                elif (st == "unknown" and not crashed
                        and time.time() - down_since >= 10):
                    # 接続ボタンが読めない環境でも、クラッシュ通知が消えて
                    # 10秒経てばカーネル復帰とみなす(2回目のRun All側にも
                    # 実行開始の確認+3回の押し直しがあるので安全)
                    return "restarted"
                elif (st == "disconnected"
                      and time.time() - down_since > 25):
                    # 自動再接続が始まらない -- 接続ボタンを押してみる
                    self._js(
                        "(() => {"
                        "  const b = document.querySelector("
                        "      'colab-connect-button');"
                        "  if (!b) return false;"
                        "  const t = b.shadowRoot && b.shadowRoot"
                        ".querySelector('#connect,button,mwc-button,"
                        "md-text-button');"
                        "  (t || b).click(); return true;"
                        "})()")
                    log("  再接続ボタンを押しました")
                    down_since = time.time()
            time.sleep(3)
        return "timeout"

    def close(self) -> None:
        try:
            self.browser.close()   # 常駐Edgeは閉じない(CDP接続だけ解放)
        except Exception:
            pass


def drive_colab(profile_dir: Path, notebook_url: str, log=print,
                poll_timeout: int = 2400, browser: str = "webview2",
                prompt_runtime: bool = True) -> dict:
    """公開エントリ: ノートを開いて起動し {'url','token'} を返す。

    browser: "webview2"=完全内蔵ブラウザ(既定・普段の環境に不干渉) /
             "edge"=従来のCDPアタッチEdge (config videolab_browser)。"""
    drv = None
    try:
        drv = ColabDriver(Path(profile_dir), notebook_url,
                          browser=browser, log=log)
        drv.prompt_runtime = prompt_runtime
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
