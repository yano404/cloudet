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
  plane.py    平面フィットコア（LSQ / RANSAC / robust 反復・残差統計）
  plyio.py    PLY 読み書き（double 精度、Open3D 非依存）
tests/        合成データによる検証（σ=0.03mm の FARO 条件を模擬）
```

## テスト

```
pip install -e .[dev]
pytest
```

## 段階計画

1. コア（本コミット）
2. バッチ CLI: 既存 `groups_out/*.ply` の一括フィット + JSON/CSV 出力 + 残差 u-v マップ
3. 位置関係リダクション（面間距離・角度・交線・コーナー）
4. GUI picker の再実装
