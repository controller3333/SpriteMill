# SpriteMill VideoLab

モデル差し替え式の**動画生成サーバ + webUI**。LTX などのオープンウェイト動画モデルを
Colab GPU またはローカル GPU で動かし、ブラウザからいろいろなモデルを試せる。
SpriteMill 本体の動画AI(歩行 i2v)としてもそのまま接続できる。

```
[ブラウザ webUI] ─┐
                  ├─ HTTP ─→ [videolab_server.py] ─→ [モデルアダプタ (LTX-2.3 / LTX-0.9 / Wan2.2 / mock)]
[SpriteMill 本体] ─┘              (Colab or ローカル)
```

- サーバ実装は `videolab_server.py` の **1ファイルだけ**。Colab ノートブックは
  `make_notebook.py` で自動生成する(ノートの手編集は禁止。直すのは .py 側)。
- API は旧 FramePack サーバ互換
  (`POST /generate_multikey` → `GET /status/{job}` → `GET /result/{job}`、Bearer トークン)。
  SpriteMill の `canvas_walk.py` から無改造で呼べる。

## 使い方 A: Colab (GPU を持っていない人向け・基本ルート)

1. `../colab/SpriteMill_video_lab.ipynb` を Google Colab で開く
2. ランタイム → ランタイムのタイプを変更 → **GPU (A100 推奨、L4 でも可)**
3. **すべてのセルを実行**(セル2で一度自動再起動 → もう一度すべて実行)
4. 最後に表示される **webUI の URL**(`?token=` 付き)をブラウザで開く
5. モデルを選び、画像を放り込んで生成。結果はブラウザで再生・ダウンロード

> ⚠ **Colab 規約の注意**: Colab **無料枠**は「notebook UI を迂回して web UI 主体で
> 使うこと」「SSH 等のリモート制御」を FAQ で明示的に禁止している(有料プランは
> この制限が解除される)。VideoLab を Colab で使うときは **有料プラン
> (Pay-as-you-go $10〜 / Pro)** を使うこと。L4 は約 1.71 CU/h(≒$0.17/h)。
> どのみち無料 T4 はシステム RAM 12GB が LTX 系のロードで枯渇しやすい。

## 使い方 B: ローカル GPU (リッチな環境の人向け)

要件: Windows / Linux、NVIDIA GPU(目安は下表)、Python 3.10〜3.12、ffmpeg。

```bat
cd SpriteMill\videolab
python -m pip install -r requirements.txt
:: 実モデルを使う場合 (CUDA 13.0 版 torch。RTX 5090/Blackwell 含む):
python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
python -m pip install "diffusers>=0.39.0" transformers accelerate safetensors sentencepiece
run_local.bat
```

`http://127.0.0.1:7860` が自動で開く。LAN に公開する場合は
`python videolab_server.py --host 0.0.0.0 --token 好きな文字列`。

Windows の注意(コミュニティ定番のハマりどころ):
- **長パスを有効化**(HFキャッシュのパスが深い): gpedit または レジストリ `LongPathsEnabled=1`
- torch の CUDA 版違い混在(cu118 と cu13x)は DLL エラーの元。入れ直すときは一度 pip uninstall
- flash-attn / triton は**不要**(PyTorch SDPA に自動フォールバックする)
- モデルキャッシュを OneDrive 配下に置かない

## モデル一覧 (2026-07 調査・出典検証済み)

| ID | モデル | VRAM 目安 | 対応モード | 備考 |
|---|---|---|---|---|
| `mock` | 合成アニメ | 不要 (ffmpegのみ) | t2v/i2v/multikey | 疎通・UI確認用 |
| `ltx23` | LTX-2.3 22B distilled | A100 40GB+ (bf16 46GB, offloadで動作) | t2v/i2v/multikey | 8ステップ高速。**本命** |
| `ltx23dev` | LTX-2.3 22B dev | 同上 | t2v/i2v/multikey | 40ステップ高品質 |
| `ltx098` | LTX-Video 0.9.7 distilled 13B | 10〜16GB (fp8化) | t2v/i2v/multikey | 旧世代・軽量。キーフレームを**任意フレーム位置**に置ける |
| `wan22` | Wan 2.2 TI2V-5B | 24GB (offloadで軽減) | i2v のみ | キャラ同一性の評判◎。比較用。Apache 2.0 |

要点(2026-07-11 時点の調査、重要事実は別エージェントで裏取り済み):
- **LTX-2.3 は 2026-03-05 公開の実在モデル**(22B、映像+音声同時生成、W/H は 32 の倍数、
  フレーム数は 8k+1、24/48fps)。HF リポは**ゲートなし**(トークン不要)。
- diffusers は **v0.38.0 以降**で LTX-2.3 対応(v0.39.0 で高速化)。本サーバは
  `LTX2ConditionPipeline` + `LTX2VideoCondition(frames=画像, index=latent位置, strength)` を使用。
- **キーフレーム条件付けの粒度**: LTX-2 系の index は latent 単位=**実フレーム8枚ぶん**。
  さらに「条件画像に対応する8フレームは静止しがち」という公式注意があるため、
  歩行サイクル用途ではコマ抽出前に必ず目視確認する。粒度が問題になる場合は
  `ltx098`(任意フレーム位置指定可)で比較すること。
- ライセンス: LTX-2 Community License = **年商 $10M 未満なら商用無料**。Wan2.2 は Apache 2.0。
- 速度実測の例: RTX 5090 + GGUF Q4 で 832x480・81f が約22秒 / H100 distilled 768x512・5秒分が約2秒。
- さらに低 VRAM にしたい場合は GGUF 量子化(`QuantStack/LTX-2.3-GGUF` Q4_K_M 17.8GB 等)
  + ComfyUI が定番だが、本サーバは diffusers 経路のみ実装(ロードマップ参照)。

## SpriteMill 本体との接続 (2026-07-11 GUI統合済み)

SpriteMill の動画AIで **「VideoLab — AniSora」** を選ぶと、既存の全自動
パイプライン(立ち絵→8方向動画→QC→シート化)の動画生成がこのサーバに飛ぶ。

- **Colab**: ノートブックを実行 → 表示された URL と TOKEN を GUI の
  「VideoLab URL / TOKEN」欄に貼るだけ。
- **ローカルGPU**: GUI の「🖥 ローカルサーバーを起動」ボタン一発。
  システムPythonの検出 → 不足ライブラリの導入(確認あり) → サーバ起動 →
  URL自動設定まで自動。**アプリ終了時にサーバも自動停止する。**
- エンジン側は常に方向別8本生成(AniSoraで実証済みの唯一の安定形)。
  後方3方向は顔可視性ロック+motion 2.5 を自動適用し、walk QC でFAILした
  方向だけをプロンプト強化+シード変更で自動再生成する(最大2周)。
- 生成条件は量産ゲートPASS実証値で固定: 464x848 / 81f / 16fps / 8step。
  変更は config.json の videolab_size / videolab_motion / videolab_model。

## API (旧 FramePack 契約互換)

```
GET  /health                      認証不要。GPU/モデル状態
GET  /api/models                  モデル一覧
POST /api/generate                {model, mode, prompt, negative, images_b64[], key_positions[],
                                   width, height, num_frames, fps, steps, guidance, seed, extra}
POST /generate_multikey           {keys_b64[], prompt, seed}   ← SpriteMill canvas_walk.py 互換
GET  /status/{job}                {status: queued|loading|running|done|error, progress, detail, log}
GET  /result/{job}                mp4 本体
POST /api/cancel/{job}
```

認証: `Authorization: Bearer <TOKEN>`(`?token=` クエリでも可。Colab では自動生成)。

## ロードマップ / 未実装

- **実GPUでの動作検証**: LTX/Wan アダプタは公式ドキュメント+検証済みAPI仕様どおりの
  実装だが、この開発機には GPU がないため**実生成は未検証**。最初の Colab 実行が
  パイロットになる(mock・API 契約・webUI はローカル検証済み)。
- **Index-AniSora V3.2**(Bilibili、Wan2.2ベースのアニメ特化、キーフレーム補間対応、
  Apache 2.0、12GB版あり): ちびキャラ歩行には**最有力候補**だが diffusers 未対応で
  独自リポ組み込みが必要なため未実装。次の実験候補筆頭。
- GGUF 量子化ロード(低VRAM強化)、2段階生成(latent upsample)による品質向上、
  Wan2.2-Animate-14B(テンプレ歩行動画でキャラを駆動する方式)。
