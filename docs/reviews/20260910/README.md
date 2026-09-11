# リポジトリ再レビュー — 2026-09-10

> これは修正前の `719a316` に対する監査記録です。後続の修正と検証結果は
> [リリース監査の修正記録](../../release_audit_fixes_20260910.md) を参照してください。

対象は `main` の `719a3165ff26877ce87a19915825f08ea6791ae5`、パッケージバージョンは `0.1.0`。
PR #61 のマージ後の状態を、配布物・公開 API・数値計算・推論・保存と再開・CI の観点から確認した。

**判定：現状のままの初回 PyPI 公開は保留を推奨する。** 配布基盤と主要な正常系は動いているが、科学的な出力を誤らせる不具合を新たに再現した。優先度 P1 が 5 件、P2 が 4 件ある。これらは未修正であり、この監査で追加したのは本レポートと再現資料のみ。

## 1. 何ができるか

| 機能 | 現状と確認範囲 |
| --- | --- |
| 球対称 Jeans 計算 | classical NumPy/SciPy と JAX の LOS 二次モーメント計算がある。classical は `sigmar2`・`sigmat2` も提供。公開 Quick Start は wheel / sdist の双方から実行成功。 |
| 恒星分布 | classical は Plummer、Sérsic、投影指数分布等。Sérsic は数値 Abel 反転と文献近似を持つ。JAX の組み込み恒星分布は Plummer。`Uniform2dModel` は 3-D モデルを持たない。 |
| 暗黒物質質量 | NFW、Zhao と truncation。Zhao の `auto/numeric` は cusp を正則化する積分で、`b<=3`、`g<3` の有限半径質量を扱う。NFW の `numeric` には下記 R1 がある。 |
| 速度異方性 | constant、Osipkov–Merritt、Baes–van Hese。JAX は固定 `eta=2` の特殊実装も持つ。 |
| JAX / 自動微分 | NFW の解析質量、Zhao の数値質量、LOS solver に JIT / gradient 経路がある。Zhao の明示的な incomplete-beta 解析経路での shape-parameter 微分は未対応。 |
| classical 推論 | 明示的な有限 prior を使う Plummer + NFW + constant anisotropy、emcee、HDF5 保存、同じモデル・同じデータでの再開。log10 scale/density と `log10(1-beta)` の座標を採用。 |
| NumPyro 推論 | `ParameterSpec`、`JeansLikelihoodModel`、NUTS / AIES のテスト、`NumPyroSampler` による MCMC の保持・再開。R2、R3、R9 の制約が残る。 |
| 保存 | ArviZ DataTree の Zarr / h5netcdf / netCDF4、チャンク結合、非同期書き込み、エラー伝播。同一 target の正常再開は CPU と実 GPU で確認。 |
| J-factor | classical に Ullio の有限 ROI、球形 aperture 近似、NFW Evans 式がある。R4、R5 のため全公開パラメータ域で信頼できる状態ではない。 |
| パッケージ配布 | `py3-none-any` wheel と sdist の生成、twine、base install、CPU extra の smoke、package data の同梱。最新依存では動作確認済み。 |

組み込み機能として、軸対称 Jeans (#52)、回転・固有運動の同時推論、foreground / membership mixture、連星補正、恒星の自己重力を加えた多成分ポテンシャル、観測選択関数付きの空間・速度同時尤度は提供されていない。独自の model / likelihood による拡張は可能だが、既製 API があるという意味ではない。

`R=0` の LOS 中心極限は未対応。classical は例外、JAX は NaN を返す。任意の epsilon で代用する仕様ではない。Sérsic をそのまま JAX / NumPyro の恒星成分として使う経路もない。

## 2. 再現した問題

### R1 — P1：NFW の数値質量が整数入力でゼロになり、不正な密度も正常値へ置換される

対象：[model_numpyro.py:719](https://github.com/gomeshun/jeanspy/blob/719a316/src/jeanspy/model_numpyro.py#L719)、特に dtype 決定と末尾の `nan_to_num`。

`NFWModel.enclosed_mass([100,1000], method="numeric", params=...)` は `[0,0]` を返す。同じ半径の float 入力では `[553056.620, 24271429.497] Msun`、解析解では `[553057.142, 24271590.540] Msun`。半径の整数 dtype が `t_min=1e-6` をゼロにし、中心での `0 * inf` から生じた NaN をゼロへ置換している。

さらに `rhos_Msunpc3=NaN` でも数値質量と LOS 分散がゼロになり、誤差 2 km/s の 1 星に対する対数尤度は有限な `-1.6120857` になる。`+inf` は質量 `1e12` に置換される経路も確認した。Zhao の override で防いだ不正値が、基底数値積分を使う NFW / custom DM 経路には残っている。

対応：計算 dtype をパラメータを含む浮動小数点へ昇格させ、半径とモデル領域を検証する。不正値を質量ゼロや任意の上限へ変換せず、NaN として solver / likelihood の reject に到達させる。

受入条件：整数・float、scalar・vector、eager・JIT の一致、NaN / infinity / 不正 scale の拒否、NFW を `dm_mass_method="numeric"` にした likelihood の `-inf` まで検証する。

### R2 — P1：NumPyro 尤度が観測配列を誤って broadcast し、尤度を多重計上する

対象：[sampler_numpyro.py:245](https://github.com/gomeshun/jeanspy/blob/719a316/src/jeanspy/sampler_numpyro.py#L245)。

3 星の `R.shape=(3,)`、`e.shape=(3,)` に対して `v.shape=(3,1)` を渡すと、観測ごとの log probability が `(3,3)` になる。正しい `(3,)` の対数尤度 `-7.7383280` が `-23.1972803` へ変わる。例外や入力警告は出ず、推論の重みを変えてしまう。負の観測誤差も二乗で正値と同じ結果になる。

対応：3 配列を同じ長さの非空 1-D に統一するか、不一致を fail-fast する。観測値の有限性、誤差の非負性、`sigma2_bounds` の順序と値域も検証する。scalar の扱いは明示的に定義する。

受入条件：`(N,1)`、`(1,N)`、長さ不一致、空配列、非有限値、負の誤差を含むテスト。正しいデータでは 1 星につき 1 尤度項であることを確認する。

### R3 — P1：推論対象が変わっても古い checkpoint / emcee state を再利用する

対象：[sampler_numpyro.py:453](https://github.com/gomeshun/jeanspy/blob/719a316/src/jeanspy/sampler_numpyro.py#L453)、[sampler.py:151](https://github.com/gomeshun/jeanspy/blob/719a316/src/jeanspy/sampler.py#L151)。

出力ディレクトリ、backend、format version は確認するが、モデル、prior、観測データ、parameter schema、solver 設定との一致を確認していない。

同じ `x` サイトを持つ `Normal(0,1)` の保存先を `Normal(100,1)` の新しい sampler に指定すると、NumPyro は `resumed=True` とし、12 draw すべてが古い最終値 `-0.5726053` に固定されたまま保存される。24 draw の結合も成功する。新 target の正しい potential は約 `5058.34` だが、保存されていた値は約 `1.08288`。同じ新 target を fresh に実行すると平均は `100.3866` だった。

emcee でも同じ保存ファイルを別の Gaussian target に使うと、追加した全 step が古い最終座標のままになり、保存 log probability と新 target での再計算に最大約 `5180.62` の差が出る。default のファイル名にはデータ識別子が必須ではないため、異なる対象での偶発的な再利用も起き得る。

対応：明示的な analysis identity / fingerprint を導入し、モデル・prior・観測・数値設定・parameter schema と照合してから再開する。不一致は新しい出力先・fresh run を要求する。target が変わった鎖を単に結合してはならない。

受入条件：同一 target の再開は維持し、prior / data / parameter order / solver 設定が変わった場合は、既存データを変更する前に検出する。両 sampler で検証する。

### R4 — P1：発散する Zhao J-factor に有限値や負値を返す

対象：[profiles.py:256](https://github.com/gomeshun/jeanspy/blob/719a316/src/jeanspy/_classical/profiles.py#L256)、[profiles.py:295](https://github.com/gomeshun/jeanspy/blob/719a316/src/jeanspy/_classical/profiles.py#L295)。

中心で `rho ~ r^(-g)` なら、aperture が中心を含む J-factor は `integral r^(2-2g) dr` を含む。内側 cutoff がなければ有限となる条件は **`g<1.5`** であり、質量の条件 `g<3` とは異なる。

`rs=1000 pc, rho_s=.01 Msun/pc^3, a=1, b=4, r_t=10000 pc, D=100000 pc, ROI=.5 deg` で、full Ullio は `g=1.5` に `1.19048e20`、`g=1.6` に `-4.38321e18`、`g=2.5` に `2.51222e19 GeV^2 cm^-5` を返す。SciPy の IntegrationWarning は出るが、不正な数値がそのまま API の戻り値になる。simple 経路でも同じ問題を再現した。

対応：J-factor 固有の収束条件を検証し、発散を明示的に拒否するか `+inf` として契約化する。有限化のために中心 cutoff / core を導入するなら、その物理的意味と値を利用者に明示して指定してもらう。任意の epsilon で正規化してはならない。

受入条件：`g=1.49,1.5,1.6,2.5`、full / simple の双方。積分 warning / 誤差推定が失敗を示す場合の扱いも定義する。

### R5 — P1：Evans J-factor が `R_aperture / rs ≈ 1` で桁落ちする

対象：[profiles.py:473](https://github.com/gomeshun/jeanspy/blob/719a316/src/jeanspy/_classical/profiles.py#L473)。

式が `delta=(1-y^2)` の二乗で割る形で、近傍級数へ切り替える幅 `1e-8` が狭すぎる。70 桁 mpmath による同じ Evans 式の評価と比較した。

上と同じ NFW scale、`D=100000 pc` で、`y=1-1e-7` では約 `-3.45909e21`、参照値は約 `1.69975e17 GeV^2 cm^-5`。`y=1+1e-6` でも約 479% の相対誤差。これらでは runtime warning も出ない。`y=1` だけを調べるテストでは発見できない。

対応：十分な次数の安定な級数と検証済み切替域、あるいは独立積分への fallback を使う。近傍の両側を log-spaced に評価する回帰テストを追加する。Evans 式の truncation の解釈は full Ullio と別途比較して明記する。

### R6 — P2：宣言された依存範囲で install は成功するが import が壊れる

対象：[pyproject.toml:35](https://github.com/gomeshun/jeanspy/blob/719a316/pyproject.toml#L35)、README の lower-bound support claim。

CPython 3.12 Linux の compatible wheels に対して `uv pip compile --resolution lowest-direct --extra numpyro_cpu --only-binary :all:` を実行した。解決された `numpy=2.0.0, pandas=2.1.1, h5py=3.10.0, netCDF4=1.6.5` は現在の metadata に適合し、`uv pip check` も成功する。しかし pandas / h5py / netCDF4 の各 import が `numpy.dtype size changed ... Expected 96 ... got 88` で失敗し、実際にインストールした JeansPy wheel の `jeanspy.model` も import できない。

最新版の fresh install は成功する。問題は「現在の lock」と「最新」だけでは古い許容組み合わせを検証できないこと。

対応：実際に NumPy 2 と互換な下限・制約へ更新し、下限環境を CI に追加する。単に resolver が成功することをバイナリ互換性の証明にしない。[lowest-cpu.txt](lowest-cpu.txt) が再現用の全バージョン一覧。

### R7 — P2：platform-dependent な `numpy.float128` を無条件 import する

対象：[dequad.py:1](https://github.com/gomeshun/jeanspy/blob/719a316/src/jeanspy/dequad.py#L1)。

`float128` は利用されていないのに無条件 import される。NumPy はこの型を全プラットフォームで提供するわけではない。属性のない環境を模擬してこのモジュールをロードすると、行 1 で ImportError になる。Windows / Apple Silicon 実機のテストは実施していないため、これは feature-absence 再現とソース点検による判定である。

対応：不要な import を除去する。必要なら `longdouble` / feature detection を使い、Windows・macOS の base-install smoke を追加する。`Operating System :: OS Independent` と Linux-only CI の差を埋める。[NumPy の platform-specific type の説明](https://numpy.org/doc/1.21/release/1.21.0-notes.html) も参照。

### R8 — P2：有限領域の空間分布が領域外でも正の確率密度を返す

対象：[profiles.py:30](https://github.com/gomeshun/jeanspy/blob/719a316/src/jeanspy/_classical/profiles.py#L30)、[profiles.py:157](https://github.com/gomeshun/jeanspy/blob/719a316/src/jeanspy/_classical/profiles.py#L157)。

Plummer の `density_2d_truncated` は cutoff 内で規格化するだけで、領域外をゼロにしていない。`re=R_trunc=200 pc` では全半径で積分すると確率が 2 になる。`Uniform2dModel(Rmax_pc=200)` も `R=400 pc` で正の密度、CDF=4 を返す。

対応：外側でゼロ、CDF は上下端で 0 / 1 とするか、入力を明示的に拒否する。負半径と無効 cutoff も契約化する。既に catalog を ROI 内に制限した条件付き計算だけでは、この不具合は表面化しない。

### R9 — P2：`ParameterSpec.exp/pow10` の省略可能引数が site 名の衝突を起こす

対象：[sampler_numpyro.py:190](https://github.com/gomeshun/jeanspy/blob/719a316/src/jeanspy/sampler_numpyro.py#L190)。

`ParameterSpec.exp('log_re', dist.Normal(0,1))` と `pow10(...)` は、`param_name` を省略すると sample と deterministic の両方を `log_re` で登録し、`all sites must have unique names` で失敗する。公開 constructor の必須引数だけでは使用できない。

対応：名前を省略したときの仕様を成立させるか、変換後の別名を必須にして構築時に分かりやすく検証する。現在の回避策は `param_name='re_pc'` 等を明示すること。

## 3. 科学的に保証できないこと

- 数値積分精度、MCMC の実行、保存、chain convergence は、posterior coverage / bias / prior calibration の検証ではない。実データや物理 mock に対する coverage campaign は今回実施していない。
- Jeans 二次モーメントが正でも、非負の phase-space distribution function が存在するとは限らない。有限中心ポテンシャル下では中心 tracer slope と anisotropy に必要条件 `gamma_star >= 2 beta_0` がある。Plummer core (`gamma_star=0`) + NFW の constant `beta>0` は、この条件を満たさない。数値 stress test や `beta=.2` の synthetic 例は、そのまま物理的な平衡分布の実証にはならない。これは [An & Evans (2006)](https://arxiv.org/abs/astro-ph/0511686) の定理をこの組み合わせへ適用した判断である。必要条件だけで DF の十分条件を保証するものでもない。
- 現在の推奨数値精度域は有限個の Plummer + NFW の kernel 評価点で検証されている。Zhao LOS、任意の tracer、全ての連続 prior 点、Baes の `auto` が選ぶ Abel solver に同じ誤差保証を拡張できない。
- `g<3` の有限質量と `g<1.5` の有限 annihilation J-factor は異なる。物理的な cutoff を決めるのは scientific modeling の選択である。
- default classical inference は kinematics-only で、観測値が独立な Gaussian velocity likelihood に従うモデル。membership、binary、selection function、非平衡の影響は自動で評価しない。

## 4. リリース基盤の評価

**整っている点**

- wheel / sdist の双方を `uv build --no-sources` で生成でき、twine が成功する。BSD license と 3 CSV の package data が同梱される。wheel の全 `jeanspy/` payload が今回の source checkout とバイト単位で一致した。
- 通常 CI は Python 3.12 / 3.13 の locked 環境。release 関連 PR / tag は locked と fresh の全テストを追加し、MCMC を含む。
- release workflow は同じ commit の reusable test workflow を呼び、`publish.needs=[build,tests]`。tag と `pyproject.toml` の version を照合し、検証した artifact を publish job に渡す。
- actions は commit SHA 固定。PyPI 書き込み権限は tag-only publish job の OIDC に限定される。PEP 740 attestations の固定 action も点検した。
- GitHub の `pypi` environment は存在する。

**公開前に残る点**

- R1–R5 は誤った科学出力・誤推論に直結するため修正が必要。R6–R9 も初回公開前に対応し、サポート範囲と API の契約を一致させたい。
- PyPI の公式 JSON endpoint はレビュー時 HTTP 404。`jeanspy` はまだ公開されていない。GitHub Release もない。これは初回公開前の状態であり、それ自体を packaging bug とは扱わない。
- PyPI 側の pending / Trusted Publisher 登録は公開 API から確認できない。owner=`gomeshun`, repository=`jeanspy`, workflow=`release.yml`, environment=`pypi` の対応を登録画面で確認する必要がある。[PyPI 公式手順](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)。
- `pypi` environment に現在 required reviewer / branch restriction はない。自動 publish を選ぶ運用なら整合的だが、tag push が公開操作になる点は明確にしておく。
- live PyPI upload / OIDC exchange は実行していない。初回公開の成否まで確認したとは言えない。
- build は成功するが、setuptools が `project.license` の TOML table と license classifier に deprecation warning を出す。`license='BSD-3-Clause'` / `license-files` への移行を計画する。現在の失敗原因ではない。

## 5. 今回の検証結果と限界

| 検証 | 結果 |
| --- | --- |
| 現在の main、Python 3.12、exact locked CPU、`pytest tests --run-mcmc -q` | **275 passed、37 subtests passed、11 warnings、157.15 s** |
| 公開 kernel accuracy contract の full 246 cases | すべて `1e-3` 以内。最大相対誤差：CPU64 `3.697e-4`、CPU32 `3.716e-4`、GPU32 grid の CPU proxy `3.762e-4` |
| RTX 3090、driver 595.84、JAX 0.9.1 / NumPyro 0.20.0、sampler tests | **10 passed、23.61 s**。短い NUTS・3 storage backend・正常再開等。全 GPU model / 全バージョンの保証ではない。 |
| 現在の wheel / sdist + 最新 base dependencies | 両方の Quick Start / packaged data check が成功 |
| 現在の wheel + 最新 CPU extra | fresh 環境で NumPyro / Jeans smoke が成功（JAX 0.11.1 / NumPyro 0.21.0 / ArviZ 1.3.0）。 |
| lower-direct dependencies | resolver と metadata check は成功、installed wheel の import は ABI error で失敗 |
| adversarial numerical / input probes | R1、R2、R4、R5、R8、R9 を再現 |
| target-changing restart probes | R3 を NumPyro / emcee の双方で再現 |
| platform feature-absence probe | R7 の無条件 import 失敗を再現。Windows / macOS 実機は未検証 |

full accuracy 実行では、同一 anisotropy ごとに stateless model インスタンスを再利用して JIT の重複コンパイルを削減した。ケース、全パラメータ、reference / candidate grid、誤差指標、合否基準はリポジトリの script と同一。独立した物理モデルとの比較ではなく、高解像度の同系統 kernel を reference にした数値テストである。

GPU / storage 検証は必要なデバイス・ファイル操作が可能な実行環境で行った。sandbox 内の CUDA device 不可視、ビルド時 DNS 制限は切り分けて再実行しており、ライブラリ不具合には数えていない。

## 6. 再現資料

- [数値・入力検証スクリプト](numerical_reproductions.py) / [記録 JSON](numerical_reproductions.json)
- [target 変更時の再開スクリプト](resume_reproductions.py) / [記録 JSON](resume_reproductions.json)
- [full accuracy runner](full_accuracy.py) / [結果](full_accuracy.log)
- [GPU sampler 結果](gpu_sampler.log)
- [lower dependency versions](lowest-cpu.txt)
- [基準 commit・artifact SHA256・検証サマリー](validation_summary.json)

再現コマンド（repository root、CPU / dev dependencies のある環境）：

```bash
JAX_PLATFORMS=cpu python docs/reviews/20260910/numerical_reproductions.py
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true python docs/reviews/20260910/resume_reproductions.py
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true python docs/reviews/20260910/full_accuracy.py
```

下限環境は既存環境に混ぜず、新しい venv に `lowest-cpu.txt` と built wheel をインストールして検証する。ここにある scripts は現状の問題を記録する監査資料であり、成功を期待する回帰テストとして CI に追加したものではない。

## 7. 推奨する対応順序

1. R1 / R2 / R3：誤推論を防ぐ mass、observation contract、resume identity。
2. R4 / R5：J-factor の収束条件と数値安定性。cutoff が必要な scientific choice は明示的に決定する。
3. R6 / R7：サポートする依存下限・OS と install / import の保証を一致させる。
4. R8 / R9：確率分布の support と公開 convenience API の修正。
5. 追加の scientific policy：DF consistency、coverage calibration、未検証領域の位置付け。
6. 修正を含む commit で通常 CI・release gate・今回の再現条件を検証し、PyPI publisher 登録を確認してから tag を作る。

追加の低優先度点検候補として、`Sampler.burn_in()` は既に posterior を標本化した chain を log posterior で再重み付けして初期位置を選ぶため、単純な posterior からの再標本化とは異なる。burn-in を保存 chain に残す仕様と併せて整理したい。WBIC の `1/log(N)` は `N=1` で未定義だが専用の入力 guard がない。この 2 点については今回、専用の統計検証・実行再現を行っていないため、上の再現済み 9 件には数えていない。
