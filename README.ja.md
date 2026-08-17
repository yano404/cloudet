# cloudet

English: [README.md](README.md)

FARO Quantum-S などで取得した三次元点群から、原子核実験用検出器の位置・位置関係をリダクションするツールです。

## 設計方針

- **抽出の契約: 1クリック = 連結した1つの物理面 = 1 group = 1平面式**。
  クリック点を種に面内へ成長させ、範囲は連結性で決まる（半径では打ち切らない）。
  種の法線は必ず誤差を含む前提で、蓄積 → 再フィット → 再蓄積を収束まで反復する
  （20° 傾いた種からでも面全体を復元することを確認済み）。
  近接平行面の分離は例外モード（GUI「Extract multiple planes (p0, p1, …)」）
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
  array_backend.py  任意 CuPy GPU バックエンド（無ければ NumPy）
  geometry.py   構築演算（オフセット平面・交差）
  reduce.py     レシピ駆動リダクション → 解析用 geometry.json
  pipeline.py   残差 u–v マップ（GUI QC 用）
  picker_qt.py  対話的アプリ（PySide6 + PyVista）
  cli.py        cloudet [project] [--cloud ...] | reduce | version
tests/          合成データによる検証（σ=0.03mm の FARO 条件を模擬）
```

## 使い方

```bash
pip install -e ".[dev]"       # アプリ一式（Qt UI 含む）
pip install -e ".[dev,open3d]"  # 任意: Open3D 表示間引きと RANSAC バックエンド
pip install -e ".[dev,gpu]"   # 任意: Fit / 残差 QC / 表示 voxel を CuPy で GPU 化
pytest

# アプリ起動（pick / Fit / 残差 QC / 保存はすべて GUI）
cloudet --cloud /path/to/scan.ply
cloudet ~/surveys/proj1 --cloud /path/to/scan.ply

# 構築型リダクション（保存済み fit + レシピ → 解析パラメータ）
cloudet reduce ~/surveys/proj1 --recipe recipe.json -o geometry.json
```

Linux/WSL では `[gpu]` が `cupy-cuda12x[ctk]`（カーネルコンパイル用 CUDA ヘッダ）を入れます。ヘッダ無しで CuPy だけ入っている場合、`auto` では NumPy に自動フォールバックします。

### GPU（任意・NVIDIA + CUDA 12.x）

3D **表示**はもともと VTK/OpenGL で GPU を使います。任意の **CuPy** で、大点群の Fit・残差 u–v マップ・表示用 voxel 間引きを加速できます。

```bat
pip install -e ".[dev,gpu]"
pip install "cupy-cuda12x[ctk]"   # GPU プローブ失敗時（WSL でヘッダ不足になりがち）
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceProperties(0)['name'])"
cloudet --cloud C:\path\to\scan.ply
```

Settings → **Compute backend**: `auto`（CuPy が使えるとき）/ `numpy` / `cupy`。  
**Display downsampling method** の `auto` も CuPy を Open3D より優先します。

CuPy なし（Mac 等）でも従来どおり NumPy のみで動作します。約 5 万点未満は `auto` でも CPU のままです。`CLOUDET_COMPUTE_BACKEND=numpy` で CPU 固定も可能です。

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

## 位置関係リダクション

Fit + 保存後、面は `groups/group_*.json` の `fit.planes[].abcd` に残ります。
解析用パラメータ（仮想軸、ビーム×標的交点、図面オフセット面）は、宣言的レシピで導出します（この段階では点群不要）。

```bash
cloudet reduce <project> --recipe recipe.json -o geometry.json
```

オフセットの符号: 正の `distance_mm` は平面の Hesse 単位法線方向へ移動（Fit と同じ符号規約）。
反対側（例: 外向き法線に対する「内側」）は負の距離を使います。

対応する construct ops: `offset`, `intersect_planes`, `intersect_three_planes`,
`intersect_line_plane`, `intersect_normal_plane`。出力 `geometry.json` は
planes / lines / points と provenance（`scanned` | `offset` | `intersection`）を含みます。

### 対話的リダクション（GUI）

アプリの **Reduction** ドック（Residuals とタブ並び）:

1. まず **操作を選択** — PARAMETERS にその操作の入力だけ出る（面 / 軸の選択、オフセットスライダーなど）
2. **Offset**: PARAMETERS で面を選び、スライダーで距離をプレビュー（緑）→ Apply で確定
3. 交差系: PARAMETERS で必要な面・軸を選んで Apply
4. Entities で表示切替後、**Save recipe…** / **Export geometry…**

## 段階計画

1. [完了] コア
2. [完了] GUI picker（Fit / 残差 QC / 保存）
3. [完了] 構築型リダクション（レシピ → geometry.json; CLI + GUI）
4. [未] より高機能なレシピ編集、検出器剛体 pose ヘルパ
