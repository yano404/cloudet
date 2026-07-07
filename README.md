# detpos

FARO Quantum-S で測定した原子核実験用検出器の三次元点群から、検出器の位置・位置関係をリダクションするツール群。既存の ChatGPT 実装（`survey2026/March/DELTA_survey_20260312/`）の再設計版。

## 設計方針

- 計算コア（`detpos/`）は numpy のみ依存、GUI と完全分離、単体テストで検証
- RANSAC は点の選別のみに使い、最終平面は常に直交最小二乗（`robust_fit_plane`: フィット → strict 再選別 → 収束まで反復）
- 統計は inlier（打ち切り）と全点の両方を報告し、打ち切りに頑健な `mad_sigma` を併記
- 乱数はシード固定で再現可能
- 単位は mm、平面は Hesse 標準形 `n·x + d = 0`（`|n|=1`、符号規約あり）

## 構成

```
detpos/
  plane.py      平面フィットコア（LSQ / RANSAC / robust 反復・残差統計）
  mainplane.py  main plane component 抽出（連結成分 + QC ゲート）
  picking.py    クリック駆動の領域抽出ロジック（GUI 非依存）
  plyio.py      PLY 読み書き（double 精度、Open3D 非依存）
  groups.py     group 読込（新形式 + legacy groups_out 互換）
  project.py    プロジェクトディレクトリ I/O（manifest / settings / group 保存）
  pipeline.py   バッチフィット（fit_xxx.json + CSV + 残差 u-v マップ）
  picker_gui.py 対話的 picker（Open3D GUI の薄いシェル）
  cli.py        detpos pick / detpos fit
tests/          合成データによる検証（σ=0.03mm の FARO 条件を模擬）
```

## 使い方（全工程）

```bash
pip install -e ".[dev,gui]"   # Qt picker（PySide6 + PyVista）と u-v マップ
pytest                        # テスト

# 1. 対話的抽出（マウス位置で P キー → group 追加。Fit で多平面分離と QC を即表示）
detpos pick ~/surveys/proj1 --pcd /path/to/scan.ply

# 2. バッチ高精度フィット（picker の保存先をそのまま入力に）
detpos fit ~/surveys/proj1 -o ~/surveys/proj1/fits
```

Qt picker: P ピック / M append 切替 / F アクティブを fit / V solo / Ctrl+S 保存 /
ツリーで名前編集・表示切替・平面ごとの品質確認。
旧 Open3D 版は `--ui open3d`（要 `pip install -e ".[viz]"`）。

## 段階計画

1. [完了] コア
2. [完了] バッチ CLI（main component 抽出 + QC がデフォルト）
3. [未] 位置関係リダクション（面間距離・角度・交線・コーナー）
4. [完了・実機検証待ち] GUI picker 再実装
