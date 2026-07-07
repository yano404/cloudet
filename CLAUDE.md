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

## 段階計画と現状

1. [完了] コア: `detpos/plane.py`（LSQ/RANSAC/robust反復/統計）, `detpos/plyio.py`（PLY I/O）, テスト13本
2. [次] バッチ CLI: groups_out 全 group 一括フィット + JSON/CSV 出力 + 残差 u-v 空間マップ
   （u-v マップで 57 µm の原因＝パス間段差か波打ちかを切り分ける）
3. 位置関係リダクション: 面間距離・角度・交線・コーナー → 検出器位置・姿勢
4. GUI picker 再実装（計算コアと分離した薄い GUI。旧版の教訓: Save All が非表示 group を
   meta から落とす問題、accumulate の連結性制限が必要）

## コマンド

```bash
pip install -e .[dev]     # 初回
pytest                    # テスト
```
