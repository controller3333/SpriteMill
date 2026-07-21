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
| `anisora` | Index-AniSora V3.2 | A100 40GBはQ4推奨 | i2v/multikey/refine | アニメ特化。公開版にネイティブOpenPose入力はない |
| `vace` | VACE-Fun + AniSora重み移植（実験） | A100 40GBはoffload必須 | OpenPose i2v | 骨格追従用。AniSora公式のPose実装ではない |
| `vace_anisora_handoff` | VACE High → native AniSora Low | A100 40GBはQ4推奨 | OpenPose i2v | VACE HighへLightning低step LoRAを適用。同じlatent/UniPC軌道の後半をAniSoraへ直結。SpriteMillの既定 |

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

### AniSoraの「8秒」とOpenPose表記について

- [公式トップREADME](https://github.com/bilibili/Index-anisora#-updates) は
  「5秒・360pを8秒以内」と掲げるが、GPU種別/台数、cold/warm、精度、step数、
  offloadの有無を記載していないため、公開手順だけでは再現条件を確定できない。
- [V3の単GPU例](https://github.com/bilibili/Index-anisora/tree/main/anisoraV3) は
  step数を省略し、公開コード既定は40 stepかつ単GPUでmodel offload有効。
  8 stepを明示するのはマルチGPU例である。V3.2 READMEの「8」は生成秒数でなく
  **推論step数**でもあるので混同しない。
- 同じTesla A100で最小832x480・5秒に数分かかる
  [未回答Issue #51](https://github.com/bilibili/Index-anisora/issues/51) がある。
  A100で5分前後は、少なくとも公開構成から外れた異常値とはいえない。
- トップREADMEはPose guidanceを掲げるが、公開V3/V3.2のCLI・推論コード・配布重みに
  OpenPose/DWPose入力やPose adapterは見当たらない。VACE対応も
  [Issue #10](https://github.com/bilibili/Index-anisora/issues/10) で計画表明のまま。
  したがって本サーバの `vace` は公開AniSoraのネイティブ機能ではなく、
  VACE-Funの制御層とAniSoraの共有ブロックを組み合わせた実験経路である。

## SpriteMill 本体との接続 (2026-07-11 GUI統合済み)

SpriteMill の動画AIで **「VideoLab — AniSora」** を選ぶと、既存の全自動
パイプライン(立ち絵→8方向動画→QC→シート化)の動画生成がこのサーバに飛ぶ。

### 工房の現行AI生成 (0.11.31)

工房の動きの型 `ai` はVACEを使わない。8方向を1方向ずつ8回生成し、
各方向は480x864・既定8推論step (管理範囲4--12) でAniSora High/Lowを
通す。0フレーム目は元画像を全面固定し、以後は顔/推定頭部と範囲外背景を
参照潜在へ固定、体だけを参照潜在の混ざらない純ノイズ (sigma=1.0) から
生成する。Highでは同じnoisy latentをOpenPose条件と画像条件で別々にforwardし、
ノイズ予測だけを既定25% / 75%で合成する。Lowは画像条件で人物として仕上げる。
実験用の一続き8方向生成では `anisora_dual_condition_image_weights` に画素フレームごとの
画像比を渡せる。安定歩行と方向間に挟む旋回フレームでPose/画像の支配率を
変える比較用であり、方向別の人物画像を一本の条件列へ常時入れる方式は残像が
安定区間まで残るため工房本線には採用していない。
`anisora_image_guidance_mask=sparse24` のように指定すると、AniSora公式の
任意時刻画像条件と同じmask構造で24フレームごとの画像だけを既知錨にできる。
黒いOpenPose背景は中立灰へ置換し、生成範囲は固定bboxでなく元絵シルエット+
時刻別Pose周囲の動的maskなので、背景側の暗い帯を作らない。decode後の矩形
画素貼り戻しは行わない。プロンプトには母艦Codexのキャラ別モーション文を
後置する。現行の実効ノブは `refine_total` (4--12、既定8)、`motion`
(2.0--4.0、既定3.0)、`pose_weight` (0.05--0.50、既定0.25)、
`head_release_steps` (0--12、既定0=終端まで固定) の4つ。

受付台0.7.2以降は、旧12択で作られた依頼を「やり直し」すると自動的に
`ai`へ移行する。現行4択の`ai`と加工系3種は選択どおり維持される。
受付台0.8.2以降の完成品ビューアは、依頼時のコンセプト・動きの型・セル
サイズ・テンプレート・ドット絵・造形補助等を右側の「生成カルテ」に表示する。
0.11.29以降に完成した作品では、実際に使ったsteps/motion/Pose・画像比/方式も
完成manifestへスナップショットし、後で管理ノブを変えても過去作の表示を
変えない。

以下のVACE→AniSora説明は、工房のレガシー再生成および通常GUIの比較経路。

- **Colab**: ノートブックを実行 → 表示された URL と TOKEN を GUI の
  「VideoLab URL / TOKEN」欄に貼るだけ。
- **ローカルGPU**: GUI の「🖥 ローカルサーバーを起動」ボタン一発。
  システムPythonの検出 → 不足ライブラリの導入(確認あり) → サーバ起動 →
  URL自動設定まで自動。**アプリ終了時にサーバも自動停止する。**
- 既定の `vace_anisora_handoff` は、選択された方向まとめキャンバス全体を
  **Lightning低step LoRAつきVACE High → native AniSora Low**
  （合計6step・境界0.90）の1軌道で生成する。中間MP4、
  JPEG分解、VAE再encode、再ノイズ化は行わず、最後に共有VAEで1回だけdecodeする。
  `compass` は8方向1ジョブ、`4x2` は2ジョブ、`all` は方向別8ジョブ。
  VACEの参照用time slotだけをhandoff時にlatentとUniPC履歴の
  両方から外し、AniSoraの16ch latent＋20ch I2V条件へ接続する。
- 旧 `compass`（骨格1＋方向別AniSora仕上げ8＝9ジョブ）と方向別SDEdit refineは
  `videolab_pose_hybrid=false` の比較・後退用として残す。既定経路からは呼ばれない。
  小セル化と完成動画の再入力で品質・時間の双方に不利なため、通常運用には使わない。
- 既定は **33f / 16fps / 合計6step**（直立6f＋歩行1周期26f＋終端、
  VACE High側はLightning低step LoRA）。
  シートで使う5歩行コマを満たす最短長で、従来81fの3周期に対し時間、
  外見ドリフト、向き逸脱を減らす。再選択候補が必要な場合だけ49/81fへ戻す。
- latent直結はQ4の VACE High とAniSora Lowを区間ごとにGPUへ載せ替え、UMT5、
  条件encode用VAE、最終decode用VAEも必要な区間だけGPUへ移す。A100 40GBを基準に
  しており、初回DL/ロードを除く`compass`の所要目標は5〜15分、方向別8本は
  10〜20分。実測前の保証値ではない。
- v0.7.1以降はlatent独自ループ全体を`torch.inference_mode()`で実行し、
  condition encode前に両DiTをCPUへ明示退避する。各区間で`VRAM[...]`を記録するため、
  Colab A100のOOMはどの移動・encode・denoiseで発生したか切り分けられる。
  latent直結の方向まとめが失敗した場合も、方向別8ジョブへ自動展開しない。
- 境界は `videolab_pose_hybrid_boundary=0.90`（比較候補0.875）。
  `videolab_pose_hybrid=false` で旧VACE/グリッド/SDEdit経路へ戻せる。

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

- **公式AniSora単体のPose adapterが公開された場合の置換候補**。現行のlatent直結は
  再生成なしの一軌道だが、VACE制御層を前半に使う独自統合であり、AniSora公式機能ではない。
- 別系統の実験候補: [UniAnimate-DiT](https://github.com/ali-vilab/UniAnimate-DiT)
  のWan2.1 I2V構成へ同系統のAniSora V3/V3.1重みを載せるA/B。
  公開済みの組合せではなく互換性・品質検証が必要。V3.2は高/低ノイズ2モデル構成で
  直接互換ではない。
- GGUF 量子化ロード(低VRAM強化)、2段階生成(latent upsample)による品質向上、
  Wan2.2-Animate-14B(テンプレ歩行動画でキャラを駆動する方式)。
