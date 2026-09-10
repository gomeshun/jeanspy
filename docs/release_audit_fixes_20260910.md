# リリース監査で見つかった問題の修正 — 2026-09-10

修正前の対象は `719a3165ff26877ce87a19915825f08ea6791ae5`（PR #61 マージ後）。
[元の監査と再現結果](reviews/20260910/README.md) を保存し、
`codex/fix-release-audit-findings` で下記の9件を修正した。
パッケージのバージョンは未公開の `0.1.0` を維持する。
修正PRは [#62](https://github.com/gomeshun/jeanspy/pull/62)。

## 修正内容と検証の対応

| ID | 問題 | 修正 | 主な回帰テスト |
|---|---|---|---|
| R1 / P1 | 整数半径で数値 NFW 質量がゼロになり、NaN 密度も有限の尤度になる | 半径・パラメータを浮動小数点へ昇格。正のスケール等の領域を検証し、無効な値を NaN のまま伝える。有限値への一律置換を廃止 | `test_release_numerical_regressions.py`: 整数と独立な解析質量の比較、ゼロ半径の勾配、JIT 尤度での無効モデル棄却 |
| R2 / P1 | `(N, 1)` の速度と `(N,)` の半径が `(N, N)` へ放送される | 3配列を同じ長さの非空1次元に限定。非有限値、非正半径、負の誤差を JIT 中も棄却。速度平均と分散下限・上限も検証 | 同ファイル: 放送の拒否、1星1尤度項の直接計算、動的な NaN/Inf/負誤差 |
| R3 / P1 | 事前分布・データを変更して古いエネルギーを再利用し、同じ出力に追記する | emcee / NumPyro 共通の内容識別。モデル・事前・データ・順序・設定・実装・主要依存版を比較し、不一致なら書き込み前に停止 | `test_sampling_identity.py`: Normal の中心 0→100、データ変更、メモリ内／再構築後の再開、バイト単位で保存物が不変、同一対象の再開成功 |
| R4 / P1 | 発散する Zhao J-factor を有限値・負値として返す | `g >= 1.5` を明示拒否。正のスケールと密度を検証し、積分不収束もエラー | `test_release_numerical_regressions.py`: `g=1.5,1.6,2.5,3` の拒否、収束する `g=1.4` の独立な重み付き積分 |
| R5 / P1 | Evans 式の `R/rs ≈ 1` で壊滅的な桁落ち | 分子の定数・1次項を解析的に相殺した20次級数を使用 | 同ファイル: 80桁 mpmath に対して両側・中心・級数切替点を `rtol=2e-11` で比較 |
| R6 / P2 | メタデータ上は合法な依存下限で ABI / 保存が壊れる | pandas 2.2.2、h5py 3.11、netCDF4 1.7.4、xarray 2025.3.1、Zarr 3.0.8 へ下限を更新。最小依存構成を CI に追加 | 全テスト、3保存形式の再開、`test_storage_compatibility.py` の独立プロセス内での2通りの import 順序と読み書き |
| R7 / P2 | プラットフォームに存在しない `numpy.float128` を無条件 import | 未使用 import を削除。Linux / Windows / macOS の base 数値・推論検証を追加 | `test_platform_compatibility.py` とネイティブ OS CI |
| R8 / P2 | 打切り表面密度が領域外でも正、Uniform の CDF が1を超える | 支持領域を適用し、打切り密度を正規化。Uniform CDF を0〜1にする | 全平面の規格化積分、領域外の密度ゼロ、CDF 境界 |
| R9 / P2 | `.exp` / `.pow10` で省略可能な物理名を省くと NumPyro site が衝突 | 省略時は `sample_name + '_transformed'` に記録。重複 site / 物理名を明示拒否 | 変換値・site 名・返却辞書キーの検証 |

追加で、emcee の `burn_in()` が既存の事後サンプルを事後密度で再重み付けする処理を削除した。
最後の ensemble から継続し、保存済み warmup は利用側で `discard` する契約を明記した。
WBIC は観測数1では定義できないため、2件以上を要求する。
ライセンスメタデータは BSD-3-Clause の SPDX 表記に更新した。

## 再開時の互換性

NumPyro のチェックポイント形式は2。旧形式・識別情報なしのチェーンを自動的に採用しない。
履歴を保持したまま新しい出力先と新しい MCMC インスタンスで開始する。
`resume=False` は別の推論対象を同じ出力先に混ぜる許可にはならない。
emcee で明示的に `reset=True` とした場合だけ、対応する backend を破棄して新しく開始できる。

Python 関数の閉包、配列の内容、NumPyro 分布、組み込み Jeans モデルは自動識別する。
隠れたファイル内容・サービス・拡張オブジェクトなどを読む独自モデルには、
その全状態を表す `sampling_identity()` を要求する。
任意の Python コードの副作用まで自動的に判定できるという保証はしない。
独自の `sampling_identity()` が状態を省略した場合の正しさは呼び出し側の責任となる。

## 依存下限の修正過程

1. 最初の候補（pandas 2.2.2、h5py 3.11、netCDF4 1.7.2）では import が通り、
   全テストは **326成功・2失敗**。実際の保存試験によって次の問題を検出した。
2. xarray 2024.11 と Zarr 3.0.8 は `Group.create_array(... exists_ok=...)` で失敗した。
   xarray 2025.3.1 に更新すると Zarr の保存・結合・再開が通った。
3. netCDF4 1.7.2 は h5py 3.11 を先に import すると `NetCDF: HDF error` で書き込みが失敗。
   import 順序を反転すると成功した。netCDF4 1.7.4 では両順序が成功した。
   NumPyro や xarray を使わない短い再現でも同じ結果だった。
4. 最小構成は `--resolution lowest-direct --only-binary :all:` で runtime / plotting だけを
   解決し、pytest / mpmath を追加して検証する。宣言下限と実際に選ばれた版を区別する。

[pandas の NumPy 2 対応](https://pandas.pydata.org/pandas-docs/stable/whatsnew/v2.2.2.html) と
[xarray 2025.3.1 の変更履歴](https://docs.xarray.dev/en/v2025.03.1/whats-new.html) も参照した。
上流には [h5py と netCDF4 の同時 import に関する類似報告](https://github.com/Unidata/netcdf4-python/issues/1438)
があるが、本修正の採否は上記の実行結果で判断した。

## 検証結果

最終コード `b721022c3eaa00aa0e5e7dda286fe74d3f84f4e5` で以下を確認した。

- [通常CI](https://github.com/gomeshun/jeanspy/actions/runs/34461603158): **16ジョブすべて成功**。
- [リリースCI](https://github.com/gomeshun/jeanspy/actions/runs/34461603493): **25ジョブ成功**。
  PyPI 公開の1ジョブはタグ以外では実行しない設定どおり skipped。
- Python 3.12 / 3.13 × locked / fresh / lowest の6構成それぞれで、MCMC を含む
  **337テスト + 37 subtests が成功**。Linux / Windows / macOS × Python 2版 × 依存3構成の
  base 数値・推論・optional 依存分離チェックも18構成すべて成功。
- 最終コードの実GPU RTX 3090 / CUDA 検証も **67成功**（139.03秒）。
  数値回帰・NumPyro 保存再開・最終追加分を含む対象識別を確認した。
  [実行コマンドとログ](reviews/20260910/fixes/final_gpu_regressions.txt) を保存した。
- wheel / sdist のビルド、Twine、両配布物の CPU / CUDA extras 解決、
  インストール後の README・同梱データ・NumPyro CPU 検証が CI で成功。
  ローカルでも最終コードから再生成して3つの隔離環境で動作を確認し、
  配布物とインストール先の全23 Python / CSV ファイルがソースとバイト単位で一致した。
- 各ジョブの URL とテスト件数は [CI記録](reviews/20260910/fixes/final_ci_validation.json)、
  ローカル配布物の SHA-256・SPDX・実行結果は
  [配布物記録](reviews/20260910/fixes/final_artifact_validation.json) に保存した。
  アーカイブの時刻情報が異なるため、ローカル生成物のハッシュを CI 生成物のハッシュとは扱わない。

修正過程と数値検証の記録:

- ロック済み Python 3.12 の全テスト: **328成功 + 37 subtests**（235.30秒、追加回帰試験前）。
- 実GPU RTX 3090 / CUDA の数値・sampler 回帰: **62成功**（50.33秒）。
- 完全な246ケースの精度契約: CPU float64 `3.697e-4`、CPU float32 `3.716e-4`、
  GPU float32 の演算設定を CPU で計算した比較 `3.762e-4`。基準 `1e-3` をすべて満たした。
  この最後の精度列は実GPUの測定ではなく、上記の実GPUテストとは別の検証である。
- 最終依存下限を決めた時点の全件実行: **331成功 + 37 subtests**（218.70秒）。
  古い matplotlib / pyparsing の非推奨警告と、netCDF4 import 時の ndarray サイズ警告が残るが、
  数値試験・両 import 順序での保存／読み出し・3形式の再開は成功した。警告は抑制していない。
- NumPy 1.26.0 / SciPy 1.12.0 を含む base 下限: **42成功**（7.00秒）。
- Windows/macOS の base、Linux Python 3.13 の locked/fresh/lowest は最初の CI で成功。
  Linux Python 3.12.3 では `inspect.getclosurevars` が `self.logger` の属性名を
  モジュールのグローバル `logger` と誤認し、ロックを解析しようとして失敗した。
  実際の `LOAD_GLOBAL` / `LOAD_NAME` だけを読み取るよう修正し、同じ Python 3.12.3 で
  旧挙動の失敗と修正後の **28成功**（7.31秒）を確認した。属性名の衝突と入れ子の式を回帰試験に追加。
- 関数の識別に加え、Series のラベル、JIT されたユーザー変換の閉包、同梱 CSV の内容も識別する。
  JAX の実行 backend と精度設定も含め、CPU / GPU の設定差をまたぐ状態の誤用を拒否する。
  明示的な `sampling_identity()` は関数にも指定でき、宣言済みの状態を優先することで
  ロック等を含む独自クラスにも適用できる。この経路の追加後の関連試験は **53成功**。

以上によりコード・依存互換性・配布物はリリース候補として検証済み。
この記録を追加する文書コミットのCIは [PR の checks](https://github.com/gomeshun/jeanspy/pull/62/checks)
で追跡する。下記の PyPI 側設定と実際の公開操作は別途残る。

## 通知された依存関係アラート

ブランチを push した時点で main の `uv.lock` に22件の Dependabot アラートが通知された。
対象は4パッケージに集中していたため、該当するロックだけを更新した。

| パッケージ | 更新前 | 更新後 | 用途 |
|---|---|---|---|
| Pillow | 12.2.0 | 12.3.0 | 描画関連 |
| Tornado | 6.5.5 | 6.5.8 | notebook 開発環境 |
| pytest | 9.0.2 | 9.1.1 | テスト。宣言下限も9.0.3へ更新 |
| Pygments | 2.19.2 | 2.21.0 | 開発・テスト表示 |

GitHub が示した修正版の範囲を満たすことを確認し、JAX・NumPyro・SciPy 等の計算用ロックは維持した。
main のアラート表示が閉じるのはマージ後であり、この記録は GitHub 側での解消済み表示を主張しない。

## 科学的な適用範囲と公開前の手順

球対称 Jeans の数値・推論・保存の実装検証であり、分布関数の非負性や事後区間の被覆率は主張しない。
有限な中心ポテンシャルで必要な `gamma_star >= 2 beta_0` に照らすと、
Plummer コア + NFW + 定数の正の異方性は物理的な分布関数を持てない。
数値ストレス試験としての意味を README に明記し、利用者の科学的事前分布を自動変更しない。
([An & Evans 2006](https://arxiv.org/abs/astro-ph/0511686))

マージ・バージョンタグ・PyPI 公開は別の操作。公開用のタグは検証のためには作らない。
GitHub の `pypi` environment は存在するが、PyPI アカウント側の Trusted Publisher 登録は
利用者の確認待ちであり、本修正では公開していない。
