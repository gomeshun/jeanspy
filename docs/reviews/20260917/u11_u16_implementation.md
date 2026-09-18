# U11–U16 実装・確認記録 — 2026-09-17

基準は取得し直した `origin/main` の `e415ffd2cb2b43b173d9b94c86730520fafcb68a`
（U01–U10を扱ったPR #69のマージ）。開始時の作業ツリーはクリーン。
作業ブランチは `codex/explicit-api-release-cleanup`。

## 変更内容

| 監査項目 | 対応 |
| --- | --- |
| U11 | プリセットを `plummer_nfw_constant_anisotropy_model`、汎用球対称推論クラスを `SphericalDSphEstimationModel` に改名。`config` を必須にし、欠損CSVの自動作成を廃止。明示的なテンプレート作成は `FlatPriorModel.write_config_template`。 |
| U12 | Sérsicの `approx` を文献の近似名 `lgm` に変更。LGM専用正規化は `lgm_norm_3d`、線形のb近似は `b_linear`。既定 `auto` のVM20bis/SP04/数値Abel分岐、係数・式・適用域は維持。 |
| U13 | Baes eta=2でも `auto` はAbel、専用Appell-F1 kernelには `solver="kernel"` が必要とmodule/class docstringに明記。自動選択の動作自体は維持。 |
| U14 | 自然な enclosed mass（半径内に含まれる質量）に合わせ `enclosed_mass` に統一。`inverse_temperature` の誤記修正。実装パッケージ `_classical` を `_numpy` に改名し、転送専用 `_model_impl` を削除。 |
| U15 | `dequad` のhashability/memoization helperとJAX quadrature containerをprivate化。全公開moduleの明示的 `__all__` を必須にし、API文書生成の自動推定を廃止。`dequad`、`generate_x_w` と数値hypergeometric関数は公開を維持。 |
| U16 | 再開契約をidentity format 2へ変更。計算コード・解析条件による互換性判定と、ソース原本のバイトハッシュ履歴を分離。旧formatは自動変換しない。 |

## 追加の整理と維持したもの

- 旧LOS入口 `sigmalos2_dequad` / `sigmalos_dequad` を廃止し、公開入口を
  `sigmalos2` / `sigmalos` に統一。double-exponentialの実装はprivateで維持。
- `jeanspy.model` からのprivate Ullio helperの再公開を廃止。内部テストは定義元を参照。
- `configure_runtime` の旧引数用catch-allを廃止し、精度の明示的signatureだけにした。
  廃止済み引数はPythonの `TypeError` で拒否する。
- `sampler` 内のコメントアウトされた試作例と不要import、`dequad` の未使用debug状態・
  plotting scratchpadを除去。数値式は変えない。
- `Parameters` / `DotDict` に互いのAPIが混ざっていた説明も、実際の属性・辞書操作に修正。
- `_numpy` は実装本体であり削除対象ではない。`model` のpublic facadeと
  public module経由のpickle対応は利用者向けの役割があるため維持。
- `vm20bis` は文献上の名称、`kernel_outer_transform="log"` は実際の数値選択肢、
  `JEANSPY_JAX_ENABLE_X64` はサポートする設定として維持。default定数は既定値を示すため維持。
- 凍結した検証結果、protocol、保持ソース、外部submodule、過去の監査probeは変更しない。
  過去の再現用ファイル・protocol内の旧ラベルは、現在の互換APIではない。

## 再開契約

PackageのPython構文からdocstringだけを取り除き、位置情報・コメント・書式を除外して比較する。
実行文字列・数値定数・default・import・assert、moduleの追加/削除/改名、package dataは保持する。
Callable側も、実行する命令、参照される定数、制御フロー・例外範囲、default・closure・宣言状態を比較。
文書の追加/削除による定数表番号の変化と、実行時に返す文字列の変更を区別する。

モデル、データ、prior、パラメータ順序・変換、固定設定、solver、sampler、Python・依存version、
JAXのbackend/precisionの既存保護を維持。source/data/dependency情報をプロセス内でcacheしない。

実行受理時にfull source/dataハッシュの履歴を保存する。NumPyroは `metadata.json` の
`source_provenance`、emceeはHDF5 groupの `jeanspy_source_provenance` dataset。
同一ソースの連続実行は重複記録せず、以前のハッシュと開始iteration等を残す。
HDF5では履歴を伸長可能datasetに格納し、attributeサイズ制限に依存しない。

Format 1以前のchainは元のcheckout・環境で再開する。新APIは新規出力先が必要。
外部ファイルやopaqueな状態、docstring/source textそのものを計算入力にするcustom modelは
`sampling_identity()` にその状態を含める必要がある。ハッシュ記録はソースarchiveの代わりではない。

利用者向けの変更表・移行方法は [API migration](../../source/guides/api-migration.md) を参照。

## 検証

- CPUの全pytest: **555 passed / 37 subtests passed**（MCMCを含む、286.39秒）。
  短いchainの自己相関推定と少数ensembleに由来する警告16件。skip/failureなし。
- 文書変更後のemcee/NumPyro再開とprovenance追記、式・data・dependency変更の拒否、
  旧format/missing metadataの拒否と出力保持、実行文字列の保護、default/closure、
  docstringの追加/削除、複数loopと連続実行の一致を検査した。
- 全体検証で見つかった、密度だけのcustom haloに質量実装を要求する過剰な制約を修正。
  J-factorのみを使うhaloは従来どおり使用でき、質量が必要な呼び出しだけ未実装エラーになる。
- Python 3.13.12 / 3.14.3でも、packageのbytecode正規化関数についてdocstringの
  追加・修正・削除を確認。3.14のdocstring存在flagを除外し、制御用flagは維持。
  これは追加のstdlib単体検証であり、両versionで全MCMCを実行したという意味ではない。
- Sphinx doctest: **5 passed / 0 failed**。HTMLも `-W --keep-going` で構築成功。
- 最終runtimeと一致するwheel（34 files / runtime 29 files）を空の別環境にinstallし、
  JAXなしでimport、canonical API、cutoff、LGM、積分、pickleを確認。
  `examples/docs_inference.py` の実際のemcee保存・再開も成功（合計6 steps）。
  この独立環境はNumPy 2.5.3 / SciPy 1.18.1 / pandas 3.0.5 /
  emcee 3.1.6 / h5py 3.16.0。元のworkspace環境は変更しない。
- 全pytest・公開例のlocked環境はPython 3.12.13、NumPy 2.4.3、SciPy 1.17.1、
  JAX 0.9.1、NumPyro 0.20.0、ArviZ 1.0.0。

- 公開notebook 7件をそれぞれ新しいkernelで再実行。保存出力の検査7件も合格。
  全26 Python source・lock・notebook codeのハッシュ、実行済みcell、error出力なしを確認。
  backend比較cellの実測時間以外は、7件の保存出力が変更前と一致した。
- 両Quickstartを新規出力先 `/tmp/jeanspy-u11-final-quickstart/run-pldzygtw` で実行し、
  保存・再開・図を再生成。source/lock 32 files、出力9 filesのハッシュが一致。
  NumPyroはidentity format 2、full source/data 29 filesのprovenanceを保存し、2 chunkを生成。
- 最終HTML **583 pages / 308,374 local links / broken links 0**。
  公開境界の混入0（除外する内部ページ11件）、dev/mainの表示チェックも合格。
- 一時build cacheを共有しないソースコピーからwheel/sdistを構築。
  **wheel 34 files / sdist 317 files**、runtime 29 filesとsdistのsource 310 filesを照合。
  新しい配布チェックは、削除済みprivate module等の余分なruntimeも拒否する。
  workspaceで生成されたAPI inventoryも意図したsdist payloadとして収録。

数値式、近似係数、既定solver選択は維持する。上記は実装・API・実行と保存再開の検証であり、
短いchainの科学的収束や校正を主張するものではない。

2026-09-17の実装・検証完了時点では、変更はローカルの作業ツリーに保存し、
commit・push・PR・公開・mergeは未実施だった。この記録はその時点の検証結果を示す。

## PR #70 再レビュー — 2026-09-18

[PR #70](https://github.com/gomeshun/jeanspy/pull/70) の初回headは
`e5cb5fa8e17c03de5f18a948b9e55de112622723`、baseは上記 `e415ffd`。
公開API、呼び出し元、移行ガイド、配布内容、両samplerの再開経路と拒否時の出力保持を再確認した。
再レビューで次の残存箇所を修正した。

- DSphModelのclass docstringにあった、削除済みLOS APIが引き続き使えるという説明を削除。
- READMEのZhao有効域に残っていた `a` / `b` を `alpha` / `beta` に訂正。
- 質量メソッドの旧alias比較が、改名によって自己比較になっていたテストを修正。
  旧aliasの削除とNumPy/SciPy・JAX間の数値一致を検査し、float32でも通過した。
- 密度のみを提供するcustom haloをJ-factorに使えることと、Jeans計算では質量が必要なことを
  基底classの説明でも区別。JAX質量メソッドの引数説明の文法も訂正。

初回headのCIは18成功、3件は公開・リリース専用ジョブの予定されたskip。
ローカル再確認は65 passed、追加修正後の関連テストは51 passed / 2 subtests passed。
追加のruntime変更はdocstringのみで、計算同一性は初回headと同じ、原本ハッシュのみが
変わることを確認した。前節の555件は同じ計算コードに対する全体検証である。

追加修正後、公開notebook 7件を新しいkernelで再実行し、保存出力の検査7件も合格。
全26 Python source・lock・notebook codeのハッシュが一致し、error出力はない。
実測時間を表示するbackend比較cell以外の出力は初回headと一致した。
両Quickstartも `/tmp/jeanspy-pr70-review-quickstart/run-xtrxh2mb` で再生成・再開し、
source/lock 32 filesと出力9 filesのハッシュを照合した。

最初の制限付き環境ではJupyter用socketを作成できず、カーネル起動前に停止した。
該当する今回の3ジョブのみを終了し、ローカル通信可能な環境で再実行した結果を上記に記録した。
Sphinxも外部inventoryを取得可能な環境で `-W --keep-going` のHTML構築と5 doctestに合格。
最終HTMLは583 pages / 308,374 local links / broken links 0、公開境界の混入0、
dev/mainの表示チェックも合格した。
