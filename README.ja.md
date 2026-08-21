<p align="center">
  <img src="docs/assets/cloudet-wordmark.svg" alt="cloudet" width="420">
</p>

# cloudet

[English](README.md) | 日本語

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

`[gpu]` は `cupy-cuda12x[ctk]`（カーネルコンパイル用 CUDA ヘッダ）を入れます。ヘッダ無しで CuPy だけ入っている場合、`auto` では NumPy に自動フォールバックします。

### GPU（任意・NVIDIA + CUDA 12.x）

3D **表示**はもともと VTK/OpenGL で GPU を使います。任意の **CuPy** で、大点群の Fit・残差 u–v マップ・表示用 voxel 間引きを加速できます。

```bat
pip install -e ".[dev,gpu]"
pip install "cupy-cuda12x[ctk]"   # GPU プローブ失敗時（CUDA ヘッダ不足。WSL で起きやすい）
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceProperties(0)['name'])"
cloudet --cloud C:\path\to\scan.ply
```

Settings → **Compute backend**: `auto`（CuPy が使えるとき）/ `numpy` / `cupy`。  
**Display downsampling method** の `auto` も CuPy を Open3D より優先します。

CuPy なし（Mac 等）でも従来どおり NumPy のみで動作します。約 5 万点未満は `auto` でも CPU のままです。`CLOUDET_COMPUTE_BACKEND=numpy` で CPU 固定も可能です。

プロジェクト構成:

```text
<project>/
  manifest.json
  settings.json
  groups/
    group_000.ply / .json / _indices.npy
    group_000_p0_indices.npy   # p0 の fit に使った inlier（任意）
    group_000_cyl0_indices.npy # 円筒 inlier（任意）
    group_000_cir0_indices.npy # 円 inlier（任意）
    ...
  vtk.log          # GUI 利用時
```

Qt UI: PROJECT で出力フォルダを指定 / SOURCE で点群を Load /
`P` pick / overlap only `>` farther and `<` nearer /
`M` append toggle / `F` fit active / `V` show only active / `Ctrl+S` save groups /
rename in tree, visibility toggle, and per-plane quality in the tree.
Fit 後は右ドックに pyqtgraph の残差 u–v マップと符号付き残差ヒストグラム（µm）を表示。
Cmd/Ctrl+ドラッグで矩形選択（ハンドルで調整可）。ズーム／パン対応。
Refit selection でその点だけの平面を追加し、同じ group の p1, p2, … として保存する（元の平面は残る）。Reduction には G6_p1 として取り込める。
Clear refit で追加フィットだけ消せる。平面を選ぶと表示が切り替わる。
The Groups tab mirrors depth controls with navigator buttons.
VTK 自身のエラー・警告は端末ではなく `<project_dir>/vtk.log` に出る
（`CLOUDET_VTK_LOG=0` で PyVista 既定の挙動に戻す。別の値はログ出力先として使う）。

## 位置関係リダクション

Fit + 保存後、面は `groups/group_*.json` の `fit.planes[].normal` + `d` に残ります
（任意で `fit.cylinders[]` / `fit.circles[]` と直径 `diameter_mm`）。
解析用パラメータ（仮想軸、ビーム×標的交点、図面オフセット面）は、宣言的レシピで導出します（この段階では点群不要）。

```bash
cloudet reduce <project> --recipe recipe.json -o geometry.json
cloudet migrate <project> [--dry-run]
```

オフセットの符号: 正の `distance_mm` は平面の Hesse 単位法線方向へ移動（Fit と同じ符号規約）。
反対側（例: 外向き法線に対する「内側」）は負の距離を使います。

旧形式（`abcd`、レシピ v1 の `of` / `a`/`b` など）も読み込み時に変換されます。
保存と `cloudet migrate` は現行キーのみ書き出します。

対応する construct ops: `offset`, `intersect_planes`, `intersect_three_planes`,
`intersect_line_plane`, `intersect_normal_plane`（元の面の法線 ∩ 先の面）、`line_from_point_normal`
（点を通り、面の法線方向の軸）、`line_from_two_points`（2点を通る軸）、
`midpoint_line_planes`（直線を2平面で切った線分の中点）、
`plane_from_plane_point`, `plane_from_line_point`, `plane_from_two_lines`,
`rotate_plane_about_line`、`rotate_point_about_line`、`rotate_line_about_line`
（任意の軸まわりの剛体回転。角度は度、右手系）。

#### `geometry.json`（エクスポート出力）

`geometry.json` はレシピを**実行した結果**に、各 entity の record を載せたものです（点群本体ではありません）。
トップレベルの `planes` / `lines` / `points` は常に**測量（survey）座標**です。
各 record には provenance（`scanned` | `offset` | `intersection` | `constructed`）と
パラメータ（`normal`/`d`, `point`/`direction`, `xyz` など。親参照は `parents`）が入ります。

| キー | 内容 |
|------|------|
| `recipe` | `{ "sha256", "echo" }` — 再現用にレシピ全文 |
| `export` | 解析対象 id のリスト（メタデータ。全 entity は別途列挙） |
| `frame` | 任意。Align Z 姿勢（`axis`, `origin`, `flip_z`, 任意で `yaw_*`） |
| `aligned` | 任意。aligned 座標系の `{ planes, lines, points }` |
| `measures` | 任意。pin した測定（再計算済み `value` / `unit`） |

**aligned の書き出し:** `cloudet reduce` はレシピに `frame` があれば `aligned` と
`frame` を付けます。GUI では **Also write aligned-frame coordinates** にチェックし、
FRAME の **Axis** と **Origin** を設定してください（Export に Align Z は不要）。
GUI は同じフォルダに `geometry_recipe.json` も書きます。

#### `geometry_summary.json`（薄いサマリ）

エクスポート時、`geometry.json` の隣に **`geometry_summary.json`** も書きます
（CLI / GUI）。レシピ echo・provenance・parents は含めず、entity の**名前と座標**
だけです。`aligned` があればその座標を優先し、なければ survey です。
レシピの `export` が空でなければ、その id だけを含めます。

#### aligned triad オペランド（原点 / 軸 / 平面）

`recipe.frame` で `axis` と `origin` を設定すると、construct から次の仮想 id が
使えます（`_store` には入らず、`geometry.json` にも行として出ません）:

| id | 種類 | 幾何 |
|----|------|------|
| `aligned.origin` | 点 | FRAME 原点 |
| `aligned.x` / `aligned.y` / `aligned.z` | 直線 | 原点を通る +X / +Y / +Z |
| `aligned.yz` / `aligned.zx` / `aligned.xy` | 平面 | 原点を通り、法線が +X / +Y / +Z |

GUI では FRAME の Axis/Origin 選択後に、種類に合うコンボへ出ます。FRAME の
axis / origin / yaw には使えません。例:

```json
{
  "frame": { "axis": "beam_axis", "origin": "beam_on_target", "flip_z": false },
  "construct": [
    {
      "id": "tilted",
      "op": "rotate_plane_about_line",
      "plane": "target",
      "line": "aligned.x",
      "angle_deg": 90.0
    },
    {
      "id": "above_xy",
      "op": "offset",
      "plane": "aligned.xy",
      "distance_mm": 10.0
    }
  ]
}
```

出力 `geometry.json` は planes / lines / points と provenance を含みます。

任意のトップレベル `frame` は Align Z のメタデータだけです（`axis` 直線 id、
`origin` 点 id、`flip_z`、任意で `yaw_line` または `yaw_plane` と `yaw_to`
（`x` / `-x` / `y` / `-y`）。直線の向きまたは平面法線の XY 射影で Z まわりを
決めます。construct のステップではなく、測量座標は変わりません。
`cloudet reduce` もトップレベルは測量のまま書き、`frame` があるときは
`aligned` コピーと姿勢を足します。
GUI は Load recipe / Load All でその選択を復元しますが、**Align Z は自動ではかけません**。

### 対話的リダクション（GUI）

右側ドックは **Residuals** / **Reduction** / **Measure** のタブです。

**Reduction** で幾何を構築します:

1. まず **操作を選択** — その操作の入力だけ出る（面 / 軸の選択、オフセットスライダーなど）
2. **Offset**: 面を選び、スライダーで距離をプレビュー（緑）→ Apply で確定
3. 交差系: 必要な面・軸を選んで Apply。
   **Normal ∩ plane → point** は元の面のオーバーレイ位置から法線を別の面に当てた交点です。
4. Entities で表示切替後、**Save recipe…** / **Export geometry…**
5. **FRAME**（表示専用）: Axis（直線）と Origin（点）を選び **Align Z**。
   軸を `(0, 0, 1)` に、原点を `(0, 0, 0)` に移す最小回転を使います。
   任意の **XY** で、**直線**または**平面法線**（水平成分のみ）を ±X / ±Y に
   載せます。XY を空にすれば最小回転のままです。**Survey** で測量座標の表示に戻します。
   Groups・レシピの構築結果・Fit は測量のままです。pick も元の点群から行います。
   Axis/Origin 設定後は、対応するオペランドと Entities 最下部に
   **aligned origin** / **X/Y/Z axis** / **YZ/ZX/XY plane** が出ます
   （表示切替のみ。rename/delete 不可）。3D では原点が球、軸が RGB 矢印
   （+X 赤 / +Y 緑 / +Z 青）、平面が同色のパッチです。
6. **Also write aligned-frame coordinates** ON かつ FRAME の **Axis/Origin** 設定後、
   **Export geometry…** で測量座標に加え `aligned` と `frame` を書けます（Export に
   Align Z は不要）。`cloudet reduce` もレシピに `frame` があれば同じです。
   Load recipe は `recipe.frame` から FRAME の選択を戻します。3D 表示を合わせる
   ときだけ Align Z を押してください。

**Measure** で、構築したエンティティの距離・角度を読みます:

- Distance (point - point / point - plane / point - line) と
  Angle (plane - plane / line - line / line - plane)
- 距離は符号なし（mm）
- line–plane の角は、直線が面に平行なら 0°
- **Add measurement** で `recipe.measures` と `geometry.json` に残る（値は再計算）
- 距離は 3D にティールの線分

## 段階計画

1. [完了] コア
2. [完了] GUI picker（Fit / 残差 QC / 保存）
3. [完了] 構築型リダクション（レシピ → geometry.json; CLI + GUI）
4. [未] より高機能なレシピ編集、検出器剛体 pose ヘルパ
