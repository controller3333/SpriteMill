# SpriteMill VideoLab

ゲームキャラの**歩行スプライトシート**を作るための、モデル差し替え式
動画生成サーバ + webUI です。[SpriteMill](https://github.com/controller3333/SpriteMill)
本体の動画AIバックエンドとして、Google Colab またはローカルGPUで動きます。

## Colab で使う (GPU不要・推奨)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/controller3333/SpriteMill/blob/main/colab/SpriteMill_video_lab.ipynb)

1. 上のバッジをクリック(またはSpriteMillアプリの「Colabでノートブックを開く」)
2. ランタイム → **GPU (A100推奨)** → すべてのセルを実行(2回)
3. 表示された URL / TOKEN を SpriteMill の「VideoLab URL / TOKEN」欄に貼る

> ⚠ Colab **無料枠**は「webUI主体の利用」が規約で禁止されています。
> 有料プラン (Pay-as-you-go $10〜 / Pro) で使ってください。

## ローカルGPUで使う

SpriteMill アプリの「🖥 ローカルサーバーを起動」ボタン一発で、
セットアップから起動・接続まで自動で行われます。手動起動する場合は
[videolab/README.md](videolab/README.md) を参照してください。

## 搭載モデル

| モデル | 用途 |
|---|---|
| **AniSora V3.2** (既定) | アニメ特化・2026-07の比較検証でスプライト歩行が圧勝 |
| **VACE High → AniSora Low latent直結** | Lightning低step LoRAつきOpenPose骨格を前半だけ使い、同じノイズ軌道をAniSoraで仕上げる（中間動画の再入力なし） |
| LTX-2.3 22B (distilled/dev) | 高速・汎用 |
| LTX-Video 0.9.7 | 低VRAM (10-16GB) |
| Wan 2.2 TI2V-5B / I2V-A14B+歩行LoRA | 比較・実験用 |
| mock | GPU不要の疎通確認 |

詳細・調査記録・ライセンス注意は [videolab/README.md](videolab/README.md) へ。

## 開発メモ

- サーバ実装は `videolab/videolab_server.py` の1ファイルが正。
  Colabノートブックは `videolab/make_notebook.py` で自動生成(手編集禁止)。
- 各モデルの重みは初回に Hugging Face から自動ダウンロードされます。
  ライセンスは各モデルのものに従ってください
  (AniSora=Apache 2.0 / LTX-2=Community License / Wan=Apache 2.0)。
