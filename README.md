# cloudet

FARO Quantum-S などで取得した三次元点群から、原子核実験用検出器の位置・位置関係をリダクションするツールです。

## 設計方針

- **抽出の契約: 1クリック = 連結した1つの物理面 = 1 group = 1平面式**。
  クリック点を種に面内へ成長させ、範囲は連結性で決まる（半径では打ち切らない）。
  種の法線は必ず誤差を含む前提で、蓄積 → 再フィット → 再蓄積を収束まで反復する
  （20° 傾いた種からでも面全体を復元することを確認済み）。
  近接平行面の分離は例外モード（GUI「Split into parallel planes」）
- 計算コア（`cloudet/`）は numpy のみ依存、GUI と完全分離、単体テストで検証
- RANSAC は点の選別のみに使い、最終平面は常に直交最小二乗（`robust_fit_plane`: フィット → strict 再選別 → 収束まで反復）
- 統計は inlier（打ち切り）と全点の両方を報告し、打ち切りに頑健な `mad_sigma` を併記
- 乱数はシード固定で再現可能
- 単位は mm、平面は Hesse 標準形 `n·x + d = 0`（`|n|=1`、符号規約あり）

## 構成

```
cloudet/
  plane.py      平面フィットコア（LSQ / RANSAC / robust 反復・残差統計）
  mainplane.py  main plane component 抽出（連結成分 + QC ゲート）
  picking.py    クリック駆動の領域抽出ロジック（GUI 非依存）
  plyio.py      PLY 読み書き（double 精度、Open3D 非依存）
  groups.py     group 読込
  project.py    プロジェクトディレクトリ I/O（manifest / settings / group 保存）
  pipeline.py   残差 u–v マップ（GUI QC 用）
  picker_qt.py  対話的アプリ（PySide6 + PyVista）
  cli.py        cloudet [project] [--cloud ...]（既定はアプリ起動）
tests/          合成データによる検証（σ=0.03mm の FARO 条件を模擬）
```

## 使い方

```bash
pip install -e ".[dev]"       # アプリ一式（Qt UI 含む）
pip install -e ".[dev,fast]"  # 任意: 表示間引きを Open3D で高速化
pytest

# アプリ起動（pick / Fit / 残差 QC / 保存はすべて GUI）
cloudet --cloud /path/to/scan.ply
cloudet ~/surveys/proj1 --cloud /path/to/scan.ply
```

プロジェクト構成::

```text
<project>/
  manifest.json
  settings.json
  groups/
    group_000.ply / .json / _indices.npy
    ...
  vtk.log          # GUI 利用時
```

Qt UI: PROJECT で出力フォルダを指定 / SOURCE で点群を Load /
`P` pick / overlap only `>` farther and `<` nearer /
`M` append toggle / `F` fit active / `V` show only active / `Ctrl+S` save groups /
rename in tree, visibility toggle, and per-plane quality in the tree.
Fit 後は右ドックに pyqtgraph の残差 u–v マップと符号付き残差ヒストグラム（µm）を表示。
Cmd/Ctrl+ドラッグで矩形選択（ハンドルで調整可）。ズーム／パン対応。
Refit selection でその点だけの平面を追加フィットできる（元の fit と矩形は残る）。
Clear refit で追加フィットだけ消せる。平面を選ぶと表示が切り替わる。
The Groups tab mirrors depth controls with navigator buttons.
VTK 自身のエラー・警告は端末ではなく `<project_dir>/vtk.log` に出る
（`CLOUDET_VTK_LOG=0` で PyVista 既定の挙動に戻す。別の値はログ出力先として使う）。

## 段階計画

1. [完了] コア
2. [完了] GUI picker（Fit / 残差 QC / 保存）
3. [未] 位置関係リダクション（面間距離・角度・交線・コーナー）
