# 新規利用者向けの名称・API契約・説明の監査 — 2026-09-16

`classical` を利用者向けの分類名として使う箇所を修正した。その後の監査で、
名称・説明と実際の意味が離れている点、開発時の互換処理や実装都合が表面に出ている点を
下記の16件に整理した。**一覧の項目は調査結果・対応案であり、今回変更したのは
`classical` の表記と、それに直接関係する例・リンクのみ。**

## 対象と確認方法

- 基準は取得し直した `origin/main` の `6eb9bdb7222c7baffdc8e259df571218e5fc4cbc`。
  開始時の作業ツリーはクリーン。`codex/numpy-backend-terminology` で作業した。
- 公開サイトは `dev` のみ。`gh-pages` は `ab57a5a` で基準コミットからの公開。
  公開済み603 HTMLを取得し、エラー0件。APIメニューの繰り返しも含め581ページに
  表示上の `classical` があった。[取得結果](published_terminology.json)。
- 取得時点のmainのDocumentation/Test CIは成功。オープンPRは0、Issueは #52 が1件。
  CIの成功は変更前の状態についての記録であり、今回の修正のリモートCI結果ではない。
- パッケージの26 Pythonモジュール、公開API生成対象12モジュール・95 export、
  README、ガイド、公開ノートブック、同梱ノートブック、15 example、38 script、
  テスト・CI・配布設定を対象に、名称、実装分岐、docstring、実行例を照合した。
- 付属の[確認コード](terminology_probes.py)と[出力](terminology_probes.json)は
  軽量なAPI確認と実装分岐の観測。MCMC・新たな科学的校正は実行していない。
- 外部サブモジュールはコミット状態を確認し、第三者コードの内部監査・変更は対象外。
  `jeanspy_paper` は未初期化。凍結済み検証結果・旧レビュー・実行来歴は改変していない。

## 実施した用語修正

- 計算基盤は **NumPy/SciPy backend** または **JAX backend**、推論手法は
  **emcee** または **NumPyro** と明記。README、ガイド、ノートブック、公開docstring、
  API生成メニュー、表示されるコード例、例外メッセージを更新した。
- バックエンド比較ノートブックの変数名を `numpy_backend` / `jax_backend` に変更し、
  再実行して出力・コードハッシュ・実行来歴を保存した。他のノートブックは説明文のみの変更。
- `scripts/example_classical_inference.py` → `scripts/example_numpy_inference.py`。
  軸対称の例は `--backend numpy` / `--backend numpyro` とし、ヘルプで計算基盤と
  サンプラーの対応を説明。README・ガイド・CIの呼び出しも更新した。
- 内部構成メモを `docs/numpy_scipy_backend_structure.md` に改名した。
- 公開ページの旧見出しアンカー4件は、既存のブックマークを維持する非表示アンカーとして保持。
  内部パッケージ名 `_classical`、過去の記録・ハッシュ内のパスは保存した。
- ライブラリ13ファイルの変更について、docstringを除いたASTを比較した。
  変更は `TypeError` の表示文1件のみで、計算式・公開import・関数の引数は同じ。
  ただし現在の再開判定はソース全文を含むため、**この変更でも旧ソースで保存したchainとの
  同一性ハッシュは変わる**。詳細は U16。

## 残っている項目

優先度「高」は、入力や計算対象を取り違える、または説明どおりの呼び出しができない項目。
「中」はAPI選択・モデル移植・再開時の理解を妨げる項目。「低」は公開範囲や内部互換層の整理。
現在のdocstringに制約が書かれている場合も、名前からの誤解が残るものは挙げた。

| ID | 優先度 | 現在の表面 | 利用者が誤解しやすい点 | 対応案 |
| --- | --- | --- | --- | --- |
| U01 | 高 | `Exp3dModel(re_pc=...)` | 純粋な3次元指数分布でもなく、`re_pc` は半光半径でもない | 投影指数分布として統合し、指数スケールと半光半径を明示的に選べるようにする |
| U02 | 高 | `Model.update(..., target=...)` | 更新対象を指定できるように見えるが、引数は無視される | 引数を廃止するか、対象を指定するAPIとして実装・検証する |
| U03 | 高 | ハローの `r_t_pc` と `mass_density_3d` | 球対称では密度を切らず質量等だけ切る。軸対称では密度も切る | 切断済み／未切断密度を明示し、密度・質量・積分の契約を揃える |
| U04 | 高 | `jfactor_ullio2016_simple` / `jfactor_evans2016` | `simple` は同じ積分の高速版ではない。Evans版の `r_t_pc` はLOS密度の切断でもない | 積分領域・近似条件を名前や必須の選択肢に反映する |
| U05 | 高 | 球対称観測データの固定 `float32` | `float64` の入力を渡しても保存時に精度が落ちる | 観測dtypeを明示的な契約にし、保存データ・再開同一性を含めて移行する |
| U06 | 高 | コピーされた汎用docstring | 存在しないメソッドや、そのクラスにはないオプション・別モデルの振る舞いを説明している | クラスごとの実APIに合わせて書き直し、例と実装を対応させる |
| U07 | 中 | `model_numpyro` / `axisymmetric_numpyro` | JAXの前向き計算とNumPyro推論の境界がモジュール名で分からない | `model_jax` / `axisymmetric_jax` 等を検討。`sampler_numpyro` は役割に合う |
| U08 | 中 | 複数の意味を持つ `backend` | 計算基盤、積分手法、核関数の実装、保存形式が同じ語で現れる | LOSは `solver` / `integration_method`、核は `kernel_backend` 等で区別する |
| U09 | 中 | `bfunc_beta_ani` / `bfunc_beta_z` | 名前から `log10(1-beta)` という標本化座標や、その上の事前分布だと分からない | 変換を表す座標名、または明示的なparameter specificationへ移す |
| U10 | 中 | `a,b,g` と `alpha,beta,gamma`、`r_a` | 同じZhao指数が幾何ごとに別名。異方性半径だけpcの接尾辞がない | 物理量の対応を揃える。異なる物理量である `beta_ani` と `beta_z` は分けておく |
| U11 | 中 | `SimpleDSphEstimationModel` / `get_default_estimation_model` | `simple` / `default` では構成・尤度・必須事前分布が分からない。省略引数はCSV生成も起こす | 球対称運動学モデルとPlummer+NFWプリセットを区別し、priorを明示的に渡す |
| U12 | 中 | `SersicModel(method="approx")` / `norm_3d` | 複数の近似のうち `approx` だけがLGMを指し、一般名の正規化もLGM専用 | `lgm`、`norm_3d_lgm` 等、近似法を表す名前にする |
| U13 | 中 | `BaesEta2AnisotropyModel` | 専用kernelを選んだつもりでも `sigmalos2(backend="auto")` はAbelへ進む | 専用kernel利用に必要な指定と実際の分岐を示し、開発予定の説明を置き換える |
| U14 | 低 | `enclosure_mass`、`inverse_temparature`、`_model_impl` | 誤記・別名・「一時的」互換層が残り、正規の入口が増えている | 正式表記と移行期間を決め、不要な別名・内部shimを整理する |
| U15 | 低 | 自動的に公開される実装補助関数 | `hashable` / `memorize` などが利用者向けのサポート対象に見える | 各モジュールの公開APIを明示し、数値APIと実装補助を分ける |
| U16 | 中 | ソース全文を使う再開同一性判定 | docstringのみの修正でも解析が変わった場合と同じ再開エラーになる | 完全な来歴と再開互換性の責務を分ける設計を検討する。保護を単純に弱めない |

### 実装根拠と影響

**U01 — 指数分布と半径の名前。**
[定義と式](../../../src/jeanspy/_classical/profiles.py#L401)では投影密度は
`exp(-R/re)/(2*pi*re**2)`、3次元密度は修正Bessel関数 `K0` による逆投影。
`Exp3dModel(re_pc=100)` の半光半径は `167.834699... pc` で、
`Exp2dModel(re_pc=167.834699...)` と投影・3次元密度が一致することを確認した。
名前の変更は、既存の数値に半径換算を伴わせずに行うと物理モデルを変えてしまう。

**U02 — 何もしない対象引数。**
[`Model.update`](../../../src/jeanspy/_classical/core.py#L334) は `del target` する。
`PlummerModel(re_pc=200).update(target="DMModel", re_pc=300)` がエラーなく
恒星成分を変更することを確認した。対象を限定したつもりで別の値が更新され得る。

**U03 — 切断の不一致。**
[球対称NFW密度](../../../src/jeanspy/_classical/profiles.py#L1000)と
[軸対称Zhao密度](../../../src/jeanspy/axisymmetric.py)を比較した。
`Q=1, rs_pc=100, rhos_Msunpc3=0.1, r_t_pc=500` の共通NFW形で
`r=1000 pc` の球対称密度は `8.26446e-5 Msun/pc^3`、軸対称側は0。
現在のガイドもこの違いを明記しているが、同じ名前で独自の積分を書く際に意味がずれる。
統一は物理的な仕様変更なので今回の用語修正には含めていない。

**U04 — `simple` が隠す積分領域。**
[`jfactor_ullio2016_simple`](../../../src/jeanspy/_classical/profiles.py#L740)は
球形apertureまでの近似で、同じ投影coneに入る外側shellを省く。
[`jfactor_evans2016`](../../../src/jeanspy/_classical/profiles.py#L1079)は
投影apertureを `r_t_pc` で制限するが、LOS方向の密度を切らない。
関連する `roi_deg_max_warning` も名前と異なり上限超過で `ValueError` を出す。
値を固定したまま、どの積分を選ぶAPIなのかを明確にする必要がある。

**U05 — 過去の保存dtype。**
[`dtype = np.float32`](../../../src/jeanspy/_classical/inference.py#L535)と
[`data.astype(self.dtype)`](../../../src/jeanspy/_classical/inference.py#L714)が
通常・sharedの両経路に適用される。入力 `100.123456789 pc` は
`100.12345886230469 pc` になった。JAXのx64設定とは独立の処理。
この差が特定の推論結果をどれだけ変えるかは今回評価していない。

**U06 — 説明テンプレートの実装からの乖離。**
- [`SersicModel`](../../../src/jeanspy/sersic.py#L35)と `Exp3dModel` は
  `logdensity_2d` を説明しているが、両クラス・基底にそのメソッドは存在しない。
- JAXの[`OsipkovMerrittModel`](../../../src/jeanspy/model_numpyro.py#L1509)と
  `BaesAnisotropyModel` はSciPy callbackの選択を説明しているが、各 `kernel` に
  `backend` 引数はない。その選択は `ConstantAnisotropyModel` のもの。
- 球対称の[`SimpleDSphEstimationModel.convert_params`](../../../src/jeanspy/_classical/inference.py#L586)
  に「axisymmetric result」「fixed_params」の説明が混在する。
- NumPyroの[`ParameterSpec`](../../../src/jeanspy/sampler_numpyro.py#L145)の例が
  NumPy/SciPy用 `examples/docs_inference.py` を指す。

API生成時にdocstringの存在は検査しているが、記述と実際のメソッドの一致は保証していない。
ビルド成功だけでは発見できない説明上の問題として、優先して個別に修正する価値がある。

**U07 — JAXとNumPyroの役割。**
[`model_numpyro`](../../../src/jeanspy/model_numpyro.py)と
[`axisymmetric_numpyro`](../../../src/jeanspy/axisymmetric_numpyro.py)は前向き計算を実装し、
推論は `sampler_numpyro` に置かれている。前二者にNumPyroの直接importはない。
packageのoptional dependency案内も `numpyro_cpu` / `numpyro_cuda12` にまとまるため、
名前だけでJAX単独の計算が見つけにくい。モジュール改名ではpickle・import・再開来歴も考慮する。

**U08 — `backend` の粒度。**
[`DSphModel.sigmalos2`](../../../src/jeanspy/model_numpyro.py#L1846)の `backend` は
`auto/kernel/abel` という積分経路を選び、同じモジュールの定異方性kernelでは
`jax/scipy` を選ぶ。さらにJAXのCPU/GPU設定や保存backendもある。
機能の違い自体は必要だが、計算基盤を選ぶつもりで渡せる同名の引数ではない。

**U09 — `bfunc_` の由来を知らないと読めない事前分布。**
[`convert_params`](../../../src/jeanspy/_classical/inference.py#L586)と
軸対称推論はこの接頭辞を `1 - 10**x` と解釈する。
一様なのは `x = log10(1-beta)` 上であり、beta上で一様ではない。
明示的な `ParameterSpec` に相当する表現、または変換が読める名前を検討すべき。
移行時は既存priorの測度・行順・保存済み座標を保存する必要がある。

**U10 — 幾何間のparameter schema。**
[球対称Zhao](../../../src/jeanspy/_classical/profiles.py#L898)の `a,b,g` と
軸対称Zhaoの `alpha,beta,gamma` は同じ式の指数だが、相互の名前を受け付けない。
異方性の `r_a` も他の半径と違い `_pc` がない。
一方、`beta_ani` と `beta_z` は異なる速度分散テンソルを表すので名称統一の対象にすべきではない。

**U11 — `simple/default` と暗黙のファイル操作。**
[`get_default_estimation_model`](../../../src/jeanspy/_classical/inference.py#L852)は
Plummer+NFW+一定異方性・Gaussian LOS尤度・測光priorという具体的な構成。
`config="priorconfig.csv"` がなければ未設定のCSVを作って例外を送出する。
汎用的な「既定の推論」と読める関数名から構成や副作用が分かりにくい。
priorの自動選択を推奨する変更ではなく、プリセットとprior入力を明示する整理が必要。

**U12 — 近似法の一般名。**
[`method="approx"`](../../../src/jeanspy/sersic.py#L525)はLGM法を選ぶが、VM20等も近似法。
[`norm_3d`](../../../src/jeanspy/sersic.py#L211)はLGM専用で、既定の `auto` の
正規化ではない。`vm20bis` は文献側の名称なので、開発中の仮名として一律廃止する理由はない。
近似法名・適用範囲を揃え、一般名だけが特定の古い近似を指す状況を解消するのが適切。

**U13 — 特殊化クラスと有効な計算経路。**
[`BaesEta2AnisotropyModel`](../../../src/jeanspy/baes_eta2.py#L242)は専用kernelを実装するが、
[`auto` 分岐](../../../src/jeanspy/model_numpyro.py#L1904)はBaesの派生型をAbelへ送る。
数値計算部分を置き換えた分岐観測でも `auto -> abel`、`kernel -> kernel` を確認した。
これは精度や速度の比較結果ではない。module docstringの「既定経路へ昇格する前の検証」
という開発予定より、利用時に有効になる経路と必要な指定を説明すべき。

**U14 — 互換名・誤記・期限のない一時shim。**
[`enclosure_mass`](../../../src/jeanspy/_classical/profiles.py#L641)と `enclosed_mass` が併存し、
[`inverse_temparature`](../../../src/jeanspy/_classical/inference.py#L85)の誤記は軸対称側にも引き継がれた。
[`_model_impl`](../../../src/jeanspy/_model_impl.py)は「v0.1.0移行中の一時shim」とされている。
private module名 `_classical` もこの層の整理候補だが、利用者のimport名を変えずに
見える説明を先に直せるため、今回の本体構造変更には含めなかった。

**U15 — 公開範囲の自動拡大。**
[`exports()`](../../../scripts/generate_api_docs.py#L36)は `__all__` がないモジュールから
非underscoreのローカルcallableを公開APIとして列挙する。
[`dequad`](../../../src/jeanspy/dequad.py)の `hashable` / `memorize` 等もAPI一覧に入る。
これらをサポート対象とする意思決定なしに実装補助が公開契約になりやすい。
`dequad` 本体や有用な低水準積分APIを削除すべきという指摘ではない。

**U16 — 文書編集とchain再開の結合。**
[`software_identity`](../../../src/jeanspy/_sampling_identity.py#L190)は全Pythonソースと
データファイルのバイト列をハッシュ化し、[`Sampler`](../../../src/jeanspy/sampler.py#L220)と
NumPyro samplerが再開時に検証する。今回のdocstring修正でもdigestが変わることを確認した。
保守的な解析保護として意図された挙動で、今回判定は変更していない。
旧chainを継続するには元のソース・環境を保持する必要がある。
将来切り分ける場合も、物理モデル・データ・prior・設定・依存関係の不一致を見逃さない
契約と移行方法が必要で、単純にdocstringを除外すれば完了するとは限らない。

## 検証結果

- 関連pytest **65 passed / 3 skipped**。skipはMCMC opt-in対象であり、失敗を隠したものではない。
  `test_docs_examples`, `test_docs_versions`, `test_model_module_structure`,
  `test_source_syntax`, `test_import`, `test_axisymmetric_inference`,
  `test_documentation_notebooks`, `test_demo_notebook`, `test_distribution_documentation` を実行。
- 公開ノートブック7件の保存済み出力とコードハッシュ検査に成功。
  コードを変更したbackendノートブックは新規カーネルで再実行。
- Sphinx HTMLを `-W --keep-going` で構築成功。doctestは **4 passed / 0 failed**。
- 603ページ・333,168ローカルリンクのチェックで不整合0。
  公開境界チェックは除外ページ11件、内部payload 0。版・ソース参照の表示チェックも成功。
- 再生成した603 HTMLで、表示テキスト中の単独語 `classical` は **0件**。
  [旧アンカーを含む確認結果](rendered_terminology.json)。
- 新しいCLI名の `--help` を確認。ライブラリの数式変更、追加MCMC、公開・push・mergeは未実施。

確認コードはリポジトリrootで、docs・numpyro_cpu・devのlocked環境から実行できる。

```bash
uv run --no-sync python docs/reviews/20260916/terminology_probes.py
```

この監査は利用者に見える意味と契約の確認であり、全パラメータ域の数値精度・科学的妥当性の再検証ではない。
