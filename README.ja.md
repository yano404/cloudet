<p align="center">
  <img src="docs/assets/cloudet-wordmark.svg" alt="cloudet" width="420">
</p>

# cloudet

[English](README.md) | 日本語

FARO Quantum-S などで取得した三次元点群から、原子核実験用検出器の位置と相対的な幾何関係をリダクションするツールです。

## 設計方針

- **既定では、1 クリックで連結した 1 つの物理面を抽出し、1 group、1 平面式として扱います。**
  クリックした点をシードとして面内に領域を成長させます。抽出範囲は連結性で決まり、半径では打ち切りません。
  シードの法線には誤差があることを前提とし、蓄積 → 再フィット → 再蓄積を収束するまで反復します
  （約 20° 傾いたシードからでも面全体を復元できることを確認済みです）。
  近接する平行面の分離には、例外的に **Extract multiple planes (p0, p1, …)** を使用します。
- **cylinder と平面上の circle にも対応しています。** cylinder はダクトやパイプなどを対象とし、Fit kind = `cylinder` を選んで円周上の 3 点をシードにします。
  circle はマーカー穴などを対象とし、Residuals の u–v マップで選択して **Fit circle on selection** を実行します。必要に応じて Fix Φ も使用できます。
  circle の中心は、支持する Groups の plane 上に固定されます。詳細は [幾何リダクションのガイド](docs/guide/reduction.md)を参照してください。
- 計算コア（`cloudet/`）が依存するのは NumPy のみです。GUI から完全に分離されており、単体テストで検証しています。
- RANSAC は点の選別にのみ使用し、最終的な plane は必ず直交最小二乗法で求めます（`robust_fit_plane`: フィット → strict 再選別 → 収束するまで反復）。cylinder と circle も、直径ベース（`diameter_mm`）の API を使用します。
- 統計量は inlier（打ち切り後）と全点の両方について報告し、打ち切りに頑健な `mad_sigma` も併記します。
- 乱数シードを固定し、再現性を確保しています。
- 単位は mm です。plane は Hesse 標準形 `n·x + d = 0`（`|n|=1`）で表し、符号規約を定めています。

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

`[gpu]` extra を指定すると、カーネルのコンパイルに必要な CUDA ヘッダを含む `cupy-cuda12x[ctk]` がインストールされます。CUDA ヘッダなしで CuPy だけがインストールされている場合、`auto` モードでは NumPy に自動的にフォールバックします。

### GPU（任意、NVIDIA + CUDA 12.x）

3D **表示**は VTK/OpenGL 経由で GPU を使います。任意で **CuPy** を入れると、大規模点群の Fit・残差 u–v マップ・表示用 voxel のダウンサンプリングを高速化できます。

```bat
pip install -e ".[dev,gpu]"
pip install "cupy-cuda12x[ctk]"   # GPU プローブ失敗時（CUDA ヘッダ不足。WSL で起きやすい）
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceProperties(0)['name'])"
cloudet --cloud C:\path\to\scan.ply
```

Settings → **Compute backend** では、`auto`（利用可能なら CuPy）、`numpy`、`cupy` を選択できます。
**Display downsampling method** が `auto` の場合も、CuPy がインストールされていれば Open3D より優先されます。

CuPy は必須ではありません。Mac などの CPU のみの環境でも、NumPy を使用して動作します。約 5 万点未満の点群は、`auto` モードでも CPU で処理します。`CLOUDET_COMPUTE_BACKEND=numpy` を設定すれば、常に CPU を使用できます。

プロジェクトの構成は次のとおりです。

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

Qt UI では、PROJECT で出力フォルダを指定し、SOURCE で点群を Load します。
`P` で pick、点が重なる箇所では `>` で奥側、`<` で手前側の点を選択できます。
`M` で append の切り替え、`F` でアクティブな group の Fit、`V` でアクティブな group のみ表示、`Ctrl+S` で groups を保存します。
ツリーでは rename と表示の切り替えができ、plane、cylinder、circle ごとの品質を確認できます。
Fit kind は **plane**（既定）または **cylinder** です。cylinder では円周上の 3 点をシードにし、Esc でクリアします。
Fit 後、右ドックの Residuals には、plane の **u–v** マップまたは cylinder の **s–z** マップと、符号付き残差のヒストグラムが表示されます（ダクトでは µm または mm スケール）。
Cmd/Ctrl+ドラッグで矩形選択でき、ハンドルで範囲を調整できます。ズームとパンにも対応しています。
plane マップでは、**Fit circle on selection** を実行すると `cir0`, `cir1`, … が追加されます。必要に応じて Fix Φ も使用できます。
Refit selection を実行すると、選択した点だけを使った追加の plane が、同じ group に p1, p2, … として追加されます。元の plane は残ります。
Reduction → **Import from Groups** では、plane、cylinder の軸（→ line）、circle の中心（→ point）を取り込めます。
Clear refit では追加の plane フィットだけを削除できます。plane、cylinder、circle を選ぶと、対応する表示に切り替わります。
Groups タブにも、depth を移動するナビゲーションボタンがあります。
詳細は [GUI ガイド](docs/guide/gui.md)と[幾何リダクションのガイド](docs/guide/reduction.md)を参照してください。
VTK 自身が出力するエラーと警告は、端末ではなく `<project_dir>/vtk.log` に記録されます
（`CLOUDET_VTK_LOG=0` を設定すると PyVista の既定の動作に戻り、それ以外の値はログの出力先として使用されます）。

## 位置関係リダクション

Fit して保存すると、face は `groups/group_*.json` の `fit.planes[].normal` と `d` に記録されます
（必要に応じて `fit.cylinders[]`、`fit.circles[]`、直径 `diameter_mm` も記録されます）。
仮想軸、ビームと標的の交点、図面上のオフセット面などの解析用パラメータは、宣言的な recipe から導出します。この段階では点群は不要です。
recipe の詳細と cylinder／circle の bind については、[docs/guide/reduction.md](docs/guide/reduction.md)を参照してください。
tracker plane、マーカー circle、ダクトと壁の交差を扱うサンプル recipe は [examples/recipes/](examples/recipes/) にあります。

```bash
cloudet reduce <project> --recipe recipe.json -o geometry.json
cloudet migrate <project> [--dry-run]
```

オフセットの符号規約では、正の `distance_mm` で plane を Hesse 単位法線の向きに移動します。Fit と同じ符号規約です。
反対側へ移動する場合、たとえば外向き法線に対して「内側」へ移動する場合は、負の距離を指定します。

recipe の例（tracker の壁 → ビーム軸と標的の交点）:

```json
{
  "version": 2,
  "units": "mm",
  "faces": {
    "tracker_left":  { "from": "group", "name": "G0" },
    "tracker_front": { "from": "group", "name": "G1" },
    "target":        { "from": "group", "name": "G2" }
  },
  "construct": [
    { "id": "left_in",  "op": "offset", "plane": "tracker_left",  "distance_mm": 12.0 },
    { "id": "front_in", "op": "offset", "plane": "tracker_front", "distance_mm": 12.0 },
    { "id": "beam_axis", "op": "intersect_planes", "plane_a": "left_in", "plane_b": "front_in" },
    { "id": "beam_on_target", "op": "intersect_line_plane", "line": "beam_axis", "plane": "target" }
  ],
  "export": ["beam_axis", "beam_on_target"],
  "frame": { "axis": "beam_axis", "origin": "beam_on_target", "flip_z": false }
}
```

旧形式の plane `abcd` や recipe v1 の `of`、`a`、`b` なども読み込めます。
保存時と `cloudet migrate` の実行時には、現行のキーだけを書き出します。

対応している construct ops は、`offset`、`intersect_planes`、`intersect_three_planes`、
`intersect_line_plane`、`intersect_normal_plane`（元の plane の法線と移動先の plane との交点）、`line_from_point_normal`
（指定した point を通り、plane の法線方向に延びる軸）、`line_from_two_points`（2 点を通る軸）、
`midpoint_line_planes`（line を 2 つの plane で切った線分の中点）、
`plane_from_plane_point`, `plane_from_line_point`, `plane_from_two_lines`,
`rotate_plane_about_line`、`rotate_point_about_line`、`rotate_line_about_line`
です。回転操作は任意の軸まわりの剛体回転で、角度の単位は度、回転方向は右手系です。

#### `geometry.json`（エクスポート出力）

`geometry.json` には、recipe の**実行結果**と、計算された各 entity の record が格納されます。点群本体は含まれません。
トップレベルの `planes`、`lines`、`points` は、常に**測量（survey）座標**で表されます。
各 record には provenance（`scanned` | `offset` | `intersection` | `constructed`）と
パラメータ（`normal`/`d`、`point`/`direction`、`xyz` など）が格納され、親 entity は `parents` で参照します。

| キー | 内容 |
|------|------|
| `recipe` | `{ "sha256", "echo" }` — 再現用に recipe 全文を保存 |
| `export` | 解析対象 id のリスト（メタデータ。全 entity は別途列挙） |
| `frame` | 任意。Align Z の姿勢（`axis`, `origin`, `flip_z`、必要に応じて `yaw_*`） |
| `aligned` | 任意。aligned 座標系の `{ planes, lines, points }` |
| `measures` | 任意。固定した測定値（再計算済みの `value` / `unit`） |

**aligned 座標の書き出し:** `cloudet reduce` は、recipe に `frame` があれば `aligned` と
`frame` を追加します。GUI では **Also write aligned-frame coordinates** にチェックを入れ、
FRAME の **Axis** と **Origin** を設定してください。エクスポートのために Align Z を実行する必要はありません。
GUI は、同じフォルダに再実行用の `geometry_recipe.json` も書き出します。

#### `geometry_summary.json`（簡易サマリ）

エクスポート時には、`geometry.json` と同じフォルダに **`geometry_summary.json`** も書き出します
（CLI / GUI）。recipe の echo、provenance、parents は含まず、entity の**名前と座標**
だけを記録します。`aligned` があればその座標を優先し、なければ survey 座標を使用します。
recipe の `export` が空でなければ、指定された id だけを含めます。例:

```json
{
  "units": "mm",
  "frame": "aligned",
  "planes": {},
  "lines": {
    "beam_axis": { "point": [0.0, 0.0, 0.0], "direction": [0.0, 0.0, 1.0] }
  },
  "points": {
    "beam_on_target": { "xyz": [0.0, 0.0, 0.0] }
  }
}
```

#### aligned triad のオペランド（原点 / 軸 / plane）

`recipe.frame` で `axis` と `origin` を設定すると、construct から次の仮想 id を
参照できます。これらは `_store` には格納されず、`geometry.json` にも独立した record として出力されません。

| id | 種類 | 幾何 |
|----|------|------|
| `aligned.origin` | point | FRAME の Origin |
| `aligned.x` / `aligned.y` / `aligned.z` | line | Origin を通る +X / +Y / +Z |
| `aligned.yz` / `aligned.zx` / `aligned.xy` | plane | Origin を通り、法線が +X / +Y / +Z |

GUI では、FRAME の Axis と Origin を選択すると、種類の一致するコンボボックスにこれらのオペランドが表示されます。FRAME の
Axis、Origin、yaw には使用できません。例:

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

出力される `geometry.json` には、planes、lines、points、provenance が含まれます。

トップレベルの任意の `frame` は、Align Z のメタデータです。`axis` に line の id、
`origin` に point の id、`flip_z` を指定し、必要に応じて `yaw_line` または `yaw_plane` と `yaw_to`
（`x` / `-x` / `y` / `-y`）も指定します。line の向きまたは plane の法線を XY 平面に射影し、Z 軸まわりの向きを
決定します。`frame` は construct のステップではなく、測量座標を変更しません。
`cloudet reduce` もトップレベルには測量座標を書き出し、`frame` がある場合は
`aligned` 座標のコピーと姿勢を追加します。
GUI は Load recipe / Load All で選択内容を復元しますが、**Align Z は自動では実行しません**。

### GUI での対話的リダクション

右側のドックには、**Residuals**、**Reduction**、**Measure** のタブがあります。

**Reduction** では、次の手順で幾何を構築します。

1. まず操作を選択します。選択した操作に必要な入力項目だけが表示されます（plane／axis の選択、オフセットスライダーなど）。
   **Import from Groups** で plane、cylinder の軸、circle の中心を取り込みます。
2. **Offset** では plane を選び、距離スライダーを動かして緑色の表示をプレビューし、Apply で確定します。
3. 交差を求める操作では、必要な plane や axis を選択して Apply を実行します。
   **Normal ∩ plane → point** は、元の plane のオーバーレイ位置から法線を延ばし、別の plane との交点を求める操作です。
4. Entities で表示を切り替え、**Save recipe…** または **Export geometry…** を実行します。
5. 表示専用の **FRAME** では、Axis（line）と Origin（point）を選択し、**Align Z** を実行します。
   Axis を `(0, 0, 1)` に合わせ、Origin を `(0, 0, 0)` に移す最小回転を使用します。
   任意の **XY** を指定すると、**line** または **plane の法線**の水平成分を ±X / ±Y に
   合わせられます。XY を空にすると、最小回転だけを適用します。**Survey** を押すと、測量座標での表示に戻ります。
   Groups、recipe の construct 結果、Fit は測量座標のままです。pick も元の点群に対して行います。
   Axis と Origin の設定後は、対応するオペランドのコンボボックスと Entities の末尾に
   **aligned origin** / **X/Y/Z axis** / **YZ/ZX/XY plane** が出ます
   （表示の切り替えのみ可能で、rename と delete はできません）。3D 表示では、Origin は球、axis は RGB の矢印
   （+X は赤、+Y は緑、+Z は青）、plane は対応する色のパッチで表されます。
6. **Also write aligned-frame coordinates** にチェックを入れ、FRAME の **Axis** と **Origin** を設定してから
   **Export geometry…** を実行すると、測量座標に加えて `aligned` と `frame` を書き出せます。エクスポートのために
   Align Z を実行する必要はありません。`cloudet reduce` でも、recipe に `frame` があれば同様に出力されます。
   Load recipe は `recipe.frame` から FRAME の選択内容を復元します。3D 表示を aligned 座標に合わせる
   場合にだけ Align Z を押してください。

**Measure** では、構築した entity 間の距離と角度を確認できます。

- Distance（point - point / point - plane / point - line）と
  Angle（plane - plane / line - line / line - plane）に対応しています。
- 距離は符号なしで、単位は mm です。
- line–plane の角度は、line が plane に平行な場合に 0° となります。
- **Add measurement** を実行すると、測定が `recipe.measures` と `geometry.json` に保存されます。値は再計算されます。
- 距離の測定箇所は、3D 表示にティール色の線分で示されます。

## ロードマップ

1. [完了] コア
2. [完了] GUI picker（Fit / 残差 QC / 保存）
3. [完了] 構築型リダクション（recipe → geometry.json、CLI + GUI）
4. [完了] cylinder と circle の Fit（Groups + Reduction への取り込み、Fix Φ / `diameter_mm`）
5. [未完了] recipe 編集機能の強化、検出器の剛体 pose を扱う helper
