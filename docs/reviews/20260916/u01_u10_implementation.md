# U01–U10 対応記録

基準: `origin/main` / HEAD `6eb9bdb7222c7baffdc8e259df571218e5fc4cbc`。
作業ブランチ: `codex/numpy-backend-terminology`。
先行する[監査](user_facing_technical_debt.md)は、変更前のAPI・観測結果として保存する。
ここでは利用者が承認したU01–U10の対応を記録する。公開用の呼び替え・換算手順は
[API移行ガイド](../../source/guides/api-migration.md)を参照。

| ID | 対応 | 計算・移行上の扱い |
| --- | --- | --- |
| U01 | `ProjectedExponentialModel(r_exp_pc=...)` に統合し、読み取り専用 `re_pc` propertyを追加 | `re_pc=1.67834699001666*r_exp_pc`。投影指数分布とK0逆投影を維持。旧Exp2dの半径は係数で割り、旧Exp3dのスケールは同じ値を渡す。測光priorは半光半径にかかるよう換算 |
| U02 | 無視されていた `Model.update(target=...)` を廃止 | 現行利用箇所は初期化時の内部呼び出しだけ。未知の引数は全更新前に拒否し、一部のパラメータだけ更新されることを防止 |
| U03 | NumPy/SciPy・JAXの球対称NFW/Zhao密度を `r_t_pc` で切断 | 外側の密度は0、境界は内側に含める。質量は同じ切断半径の外で一定。`r_t_pc=np.inf` は有限半径で未切断。従来の球対称LOS計算は質量を使用しており、その計算式は維持 |
| U04 | `jfactor_cone`、`jfactor_spherical_aperture`、`jfactor_small_angle_infinite_los` に改名 | 数値式・積分領域は維持。最後の方法だけLOS密度が未切断で、投影apertureを切る。角度上限は `small_angle_limit_deg`、検証メソッドは `validate_small_angle` |
| U05 | 観測データの固定float32を廃止し `dtype=` を追加 | 既定は3列の共通浮動小数点dtypeを保持、整数のみならfloat64。指定時は指定dtype。共有メモリも同じ精度を使い、resetで型・形状を変える操作は拒否。保存同一性に型も含まれる |
| U06 | 公開クラスとメソッドのdocstringを実APIと照合 | Uniformの存在しないメソッド、Sersicのlogdensity、NFWのZhao用説明、各異方性の引数・戻り形状、JAXのSciPy callback、球対称/軸対称推論・例の混在などを修正 |
| U07 | `model_jax` / `axisymmetric_jax` に改名 | JAX前向き計算と `sampler_numpyro` の推論を区別。現行のimport・例・テスト・API生成を更新。旧モジュールのshimは追加しない |
| U08 | JAXのLOS計算法を `solver`、核関数の実装を `kernel_backend` に分離 | 計算法・auto分岐・既定精度は維持。runtime情報のキーも変更。保存形式は `storage_backend`、軸対称の実行例は `--sampler emcee/numpyro` |
| U09 | `jeanspy.parameters.SamplingParameter` で変換を明示 | sample名、physical名、`identity/pow10/one_minus_pow10/arccos` を指定。接頭辞を解釈しない。NumPyroの既存 `ParameterSpec` は維持。priorはsample座標上の密度のまま。変換仕様を再開同一性に含める |
| U10 | 球対称Zhaoを `alpha,beta,gamma` に統一 | 軸対称と同じ指数名。質量の数式、事前範囲・測度は維持。異なる異方性 `beta_ani` と `beta_z` は区別し、未承認の `r_a` 改名は行わない |

## 条件付き項目の判断

- U02: `target` は実装で破棄されており、現行の更新ロジックに寄与しない。
- U03: 未切断密度を返す必要のある呼び出しは現行の球対称LOSにはない。
  未切断LOSの解析J-factorは専用名と説明で区別し、その式を変更しない。
- U05: 固定float32は共有メモリにも用いられる保存設定だった。
  バッファは入力のバイト数から確保でき、固定float32である必要はない。
  数値ソルバーによる高精度への昇格と、観測入力の無断の丸めを区別した。

## 検証

- 通常テスト（保存出力再生成中の文書ノートブック7件を除外）:
  **500 passed / 30 skipped / 37 subtests passed**。30件はMCMCの明示実行待ち。
- `pytest -m mcmc --run-mcmc`: **30 passed**。emcee/NumPyroの実行、共有・spawn、
  checkpoint、保存形式、再開・不一致拒否を含む。
- 最終の契約・Zhao・数値回帰・構文チェック: **81 passed**（上記と重複）。
- 新しい契約テストは指数分布の投影/3-D正規化、半光半径、切断境界、
  `dM/dr=4*pi*r**2*rho`、JAXの密度勾配、float32/64の通常/共有保存、
  型変更失敗時の非破壊性、任意の座標名とprior測度、指数スケールの測光換算を確認。
- 公開ノートブック7冊を `scripts/run_doc_notebooks.py --include-mcmc --write`
  で再実行。保存出力・配布可能性・関連例の検査は **20 passed / 2 skipped**
  （通常テストとの重複を含み、skipの2件は上記MCMC実行で合格）。
  重複を除くpytestの合計は **537 passed / 37 subtests passed**。
- `scripts/run_quickstart.py` でemcee/NumPyroの図・ログを再生成。
  最終採用ディレクトリは `/tmp/jeanspy-u01-u10-quickstart/run-csea2yyc`。
  `execution.json` のソース33ファイル・出力9ファイルのSHA-256一致を確認。
- 軸対称CLIの `--sampler emcee` / `--sampler numpyro` を新規実行・再開。
  emceeは合計14 steps、NumPyroは合計8 drawsまで保存され、両者とも
  最終summaryで `resumed: true` を確認。
- Sphinx HTMLを `-W --keep-going` でビルドし、doctest **5 passed**。
  593 HTMLの319,560ローカルリンクに破損なし。公開境界の検査は
  内部payload 0、廃止済み内部ページ11件の混入なし。dev/mainの表示同一性も合格。
- wheelとsdistを構築し、収録内容と作業ツリーのファイル一致を確認。
  wheelの新しい `model_jax` / `axisymmetric_jax` / `parameters` を確認し、
  旧モジュール2件が残っていないことも確認。
  wheelを別のディレクトリへインストールし、その実体からimportして、指数分布の
  半光半径、明示変換、float64でのNumPy/JAX切断Zhao密度の一致を検証。

実行環境はPython 3.12.13、NumPy 2.4.3、SciPy 1.17.1、JAX 0.9.1、
NumPyro 0.20.0。`/tmp/jeanspy-terminology-venv` を使用し、既存の環境は変更しない。

新しいテストはCPU環境で実行。短いchainでは自己相関推定やESS等の警告が出る。
実行成功・保存再開の確認を科学的収束・被覆率校正と解釈しない。

## 保存済み解析と履歴

公開APIと保存精度が変わるため、新しい解析出力ディレクトリが必要。
ソース全文を含む既存の再開保護は緩めていない。実際に、docstring修正中に走った
最初のQuickstart例はソース変更を検出してcheckpoint保存を拒否した。
ソース確定後、新しいディレクトリで再実行した結果だけを公開例の出力に採用した。

凍結済み `validation/`、検証プロトコル・結果・保持ソース、外部サブモジュールは変更しない。
先行監査のprobeとJSONは変更前の記録であり、旧APIの基準checkout用である。
U11–U16は今回の承認対象外として残す。この記録は2026-09-16のローカル実装完了時点の
検証結果であり、その時点ではcommit・push・公開・mergeは行っていない。
