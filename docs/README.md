# ドキュメント管理メモ

## ローカルビルド

```bash
julia --project=docs -e '
    using Pkg
    Pkg.develop(PackageSpec(path=pwd()))
    Pkg.instantiate()
    include("docs/make.jl")
'
```

ビルド結果は `docs/build/` に出力される。ブラウザで `docs/build/index.html` を開けば確認できる。

## ファイル構成

```
docs/
├── Project.toml          # Documenter.jl 等の依存
├── Manifest.toml
├── make.jl               # ビルドスクリプト（ページ構成もここ）
└── src/
    ├── index.md           # トップページ（クイックスタート）
    └── api/
        ├── estimators.md  # 推定器の API リファレンス
        ├── policies.md    # 方策の API リファレンス
        ├── environments.md # 環境の API リファレンス
        ├── system.md      # System, Agent, History
        ├── evaluation.md  # 評価関数
        └── internals.md   # 内部ユーティリティ (@ic0 等)
```

## ページを追加・変更するとき

1. `docs/src/` 以下に `.md` ファイルを作成
2. `docs/make.jl` の `pages` 配列にエントリを追加

## docstring を追加したとき

ソースコード中に `"""..."""` で docstring を書けば、対応する `docs/src/api/*.md` 内の
`@docs` ブロックにシンボル名を追加するだけで API リファレンスに反映される。

例: `src/core/estimator/new_est.jl` に `NewEstimator` を追加した場合

```markdown
# docs/src/api/estimators.md に追加
```@docs
NewEstimator
```
```

## CI でのデプロイ

`.github/workflows/CI.yml` の `docs` ジョブが main push 時に自動ビルド・デプロイする。
デプロイ先: `https://tatsujit.github.io/MultiBandits.jl`

初回デプロイ時に必要な設定:
- GitHub リポジトリの Settings → Pages → Source を `gh-pages` ブランチに設定
- `DocumenterTools.genkeys()` で生成した鍵を `DOCUMENTER_KEY` シークレットに登録
