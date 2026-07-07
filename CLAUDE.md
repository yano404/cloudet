# detpos プロジェクトメモ

FARO Quantum-S（公称精度 ~30 µm）で測定した原子核実験用検出器の三次元点群から、
検出器の位置・位置関係をリダクションするツール群。
ChatGPT 実装（下記「参照データ」内の `interactive_plane_picker.py` / `high_precision_ransac.py`）の再設計版。

## 絶対に守ること

- **原本不可侵**: `~/work/survey2026/` 以下の既存コード・データは読み取り専用。変更・削除しない。
- 単位は **mm**。平面は Hesse 標準形 `n·x + d = 0`（`|n|=1`、法線の最大成分が正になる符号規約）。
- RANSAC は点の選別のみ。**最終平面は必ず直交最小二乗**（`robust_fit_plane`）。
- 乱数はシード固定。統計は inlier（打ち切り）と全点の両方 + `mad_sigma` を報告。
- コアは numpy のみ依存、GUI と分離、テスト必須（`pytest`、合成データで σ=0.03mm 条件を模擬）。

## 参照データ

- 元データ: `~/work/survey2026/March/DELTA_survey_20260312/`
  - `groups_out/`: 抽出済み 17 group（`group_xxx.ply` double精度 + `groups_summary.json` + 旧形式 `groups_meta.npz`）
  - 元 PCD は4スキャン結合の約6000万点。`_mat.txt` は全て単位行列（PolyWorks でアライン済み）
- レビュー文書: `../code_review_20260707.md`

## 既知の重要知見

- Open3D `segment_plane` は inlier に対し最終再フィットするため、残差が threshold を僅かに超えるのは正常。
- 点数が多いため統計誤差は無視でき、精度は系統誤差（パス間レジストレーション残差など）で決まる。床は器差 ~30 µm。
- 実データ G13/G14 のフィット結果: inlier mad_sigma ≈ 57 µm（公称の約2倍）、inlier 率 47%。
  group 内に平面外の点が多い（picker の accumulate が無限平面スラブで拾うため）。
- 旧 `groups_meta.npz` は pickle 依存（object 配列: names, indices）で、実ファイルには
  来歴キー（source_num_points / source_pcd_path）が欠落 → 整合性チェックは機能していなかった。

## 保存形式（新設計・合意済み）

1 group = `group_xxx.ply`（double点群）+ `group_xxx.json`（メタ・plyのSHA256）
+ `group_xxx_indices.npy`（元PCDインデックス、picker再編集用・オプション）。
全体に `manifest.json`（単位・元スキャン一覧・ハッシュ・結合順序・ツールversion）。
pickle 完全排除。後段処理の契約は ply + json のみ。旧 groups_out は読み取り互換を維持。

## 設定ファイル（新設計・合意済み）

旧 `plane_picker_settings.json` の問題: フラットにアルゴリズム/表示/セッションが混在、
version なし、未知キー黙殺、`settings_path` の自己参照、単位暗黙。
実害: 現物の `accumulate_distance_threshold` は 2.5（コードのデフォルトは 1.0）で、
**groups_out がどの値で抽出されたか確定不能**（G13/G14 の平面外混入 max 3.4mm と符合）。

新設計:
- スキーマ: `{"version": 1, "units": "mm", "detection": {...}, "view": {...}}` とセクション分割。
  数値キーは単位サフィックス付き（例 `local_radius_mm`）
- セッション状態（pcd_path, save_dir, パネル幅）は別ファイル `session.json`（マシンローカル）。
  `settings_path` 自己参照は廃止
- ローダーは dataclass ベース: 未知キーは警告、欠落はデフォルト、version で移行
- **detection の内容は group 保存のたびに manifest へ焼き込む**（再現性は出力側記録で担保。
  可変な設定ファイルに依存しない）

## 段階計画と現状

1. [完了] コア: `detpos/plane.py`（LSQ/RANSAC/robust反復/統計）, `detpos/plyio.py`（PLY I/O）, テスト13本
2. [完了] バッチ CLI: `detpos fit <groups_dir> -o <out>`（`groups.py`=legacy互換読込,
   `pipeline.py`=fit_xxx.json+CSV+u-vマップ, `cli.py`）。全17group実行済み →
   `../results/DELTA_survey_20260312/fits/`。
   結果: 品質は group 間で大差。良: G9=39µm, G16=47µm。悪: G11=1223µm(平面ですらない,
   adaptive threshold が発散して inlier 100%), G3=472µm, G4=364µm, G15=非収束。
   u-v マップでパス間段差（±100µm のコヒーレント構造）が可視化できた。
2.5 [完了] main component 抽出（`mainplane.py`、CLI デフォルト。--simple で旧動作）:
   picker はラフ選択という前提。上限付き適応threshold + 面内(u,v)グリッド連結成分分析
   （クリック点の成分を優先、なければ最大）+ QC ゲート（ok/suspect/fail + reasons）。
   全17group 再実行 → `../results/DELTA_survey_20260312/fits_main/`: ok=9, suspect=8, fail=0。
   G11 は 1223µm のゴミ → main 292k点で 49µm ok に回復。suspect 8つは mad_sigma 100-200µm
   （パス間段差が主因の見込み、u-vマップ参照）
2.7 [完了] 多平面分離（`multiplane.py`、CLI デフォルト。--single-plane で単一平面）:
   **データモデルは 1 group = N 平面**。逐次抽出（fit→inlier除去→繰り返し）、
   平面ごとの threshold は細め（RANSAC 0.1 / 上限 0.15mm）で近接平行面を解像。
   合成テストで 0.4mm 分離を確認。実データ G13 は d=-1733.35/-1733.62/-1733.05 の
   **0.3mm 間隔の3面に分離**（前は 116µm suspect で混合していた）。
   fit_xxx.json は version 2: `planes` 配列（dominant first、各平面に
   plane方程式/status/bimodalフラグ/統計）。CSV は1行=1平面。
   平面方程式は Hesse 標準形 n·x + d = 0 [mm]（`plane.abcd`）。
3. [未] 位置関係リダクション: 面間距離・角度・交線・コーナー → 検出器位置・姿勢。
   平面の安定命名（例 "det1_top"）の仕様決めが必要（GUI からラベル付け予定）
4. [実装済み・実機未検証] GUI picker 再実装（`picker_gui.py` + `picking.py` + `project.py`、
   `detpos pick <project_dir> --pcd <cloud>`)。
   - ロジック（クリック抽出=連結性制限付き accumulate、新形式保存）は GUI から分離しテスト済み
   - 保存(S)時に全 group を extract_main_plane で fit し QC/mad_sigma を即表示、manifest 書込
   - Save All は非表示 group も保存（旧版のバグを踏襲しない）。Load は manifest の
     n_points 照合 + indices 範囲チェック
   - GUI 本体はサンドボックスで起動不可のため**ユーザーの実機での動作確認待ち**。
     Open3D API は旧実装で実績のあるパターンを踏襲

## コマンド

```bash
pip install -e .[dev]     # 初回
pytest                    # テスト
```
