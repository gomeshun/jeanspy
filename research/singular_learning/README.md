# Jeans解析の識別可能性と学習曲線：研究方針

作成日：2026-09-29。状態：文献確認と研究計画。新規MCMC・RLCTの確定計算・必要星数の算出は未実施。

**中心課題は、有限個の視線速度からγ・ρ(r)・M(r)が学習される速さの違いを、観測写像の幾何、prior、観測設計から説明すること。** 特異学習理論を有力な枠組みとして採用するが、現在の有限次元モデルが生成点で厳密に特異であることや、相転移が起きることを結論として先取りしない。

ここで n は独立な星の速度測定数であり、MCMCの反復数・保存draw数・ESSではない。

## 作業場所と成果物

- worktree：`/tmp/jeanspy-singular-learning-20260929`
- branch：`codex/singular-learning-theory`
- base：最新 `origin/main` を取得・照合した `d28e0afd8fcb8223f9409ccf23ba6dc6d0f67b0c`
- [理論ノート](theory_notes.md)：Gaussian KL、特異性の判定、RLCTと物理量の学習率、prior、crossover。
- [文献と位置づけ](references.md)：一次資料、確認範囲、先行研究との差。
- [入力の記録](source_manifest.json)：参照元のcommit・SHA-256、保存した小さな資料の一覧。
- `snapshots/`：既存mockのprotocol、要約、原稿の該当箇所などの参照用コピー。実験の変更指示ではない。

本worktreeでは計画資料だけを追加した。ソースコード、公開API、原稿submodule、既存のchain、既存環境を変更していない。専用のPython環境と計算コードは今後このworktree内に作る。GPUやMCMCを前提とせず、当面は解析式とCPUの少量の決定論的計算を使う。

## 現在の観測をどう位置づけるか

2026-09-29時点のpaper checkout `eb748b0c3e7f347d5e39aa8d8922b512b026114c` のprotocol・summary・precision addendumを直接確認した。

| 条件 | Draco-like | Segue 1-like |
|---|---:|---:|
| 星数 | 256 | 64 |
| Plummerの射影半光半径 R_e [pc] | 180.301774 | 29 |
| halo scale r_s [pc] | 1000 | 150 |
| M(r_1/2) [M_sun] | 8.2760493 × 10^6 | 5.8 × 10^5 |
| 速度誤差 e [km/s] | 2 | 5 |
| 射影半径の選択 [pc] | 10–1500 | 0–87 |
| 生成halo (α, β_h, γ) | (1,3,1) / (1,3,0) | (1,3,1) / (1,3,0) |
| 生成anisotropy β_a | −0.5 | −0.5 |
| 生成速度 | AGAMAの平衡DF | AGAMAの平衡DF |
| 推論 | 条件付きGaussian LOS、7変数、μを厳密積分 | 同左 |

両方とも星のサイズ等を参考にしたcontrolled mockであり、実在銀河のbest fitではない。同じ真値・設計でnだけを変更した実験ではない。

保存summaryから読み取ったγ中央値は以下の通り。これは学習率の測定ではない。

| mock | coordinate-uniform | joint Jeffreys |
|---|---:|---:|
| Draco-like cusp | 0.855794 | 0.666016 |
| Draco-like core | 0.878723 | 0.459551 |
| Segue-like cusp | 0.688380 | 0.509061 |
| Segue-like core | 0.412391 | 0.408816 |

8対象ともsampling gateは通過している。Segueの元summaryはpreflight failureを保持し、`all_four_posteriors_usable=false` のままである。別のprecision addendumが既知の算術問題を有限の点検で解決し、保存posteriorによる比較図を支持している。元summaryを成功へ書き換えたとは解釈しない。いずれも反復mockによるcoverageや識別性能の検証ではない。

## 渡辺理論を使う位置

基本的な道具は、生成分布q、モデルp、prior測度π、観測設計gを指定したときのKL contrastと、そのゼロ集合周辺の体積である。自由エネルギーの漸近展開にRLCT λ と多重度mが現れる。適用条件と式は[理論ノート](theory_notes.md)、原典は[Watanabe 2013](https://jmlr.org/papers/v14/watanabe13a.html)にまとめた。

このモデルでは、次の区別が最初の成果になる。

| 状態 | 調べる量 | 科学的含意 |
|---|---|---|
| 厳密な非識別性・高次の特異性 | KLのゼロ集合、局所展開、RLCT | nを増やしても個々のパラメータが決まらない／通常と異なる学習率の可能性 |
| full rankだが弱い識別可能性 | 小さな特異値、高次項、priorの幅 | 漸近的には通常の率でも、その領域への到達が遅い可能性 |
| support境界 | 接錐とKLの境界展開 | 切断・歪んだposterior。境界だけではrank欠損を意味しない |
| 浮動小数点・積分誤差 | 独立精度、積分次数、積分端点 | 見かけのrank欠損やpriorの誤差 |
| DFとGaussian尤度のずれ | 条件付きmoment、KL最小集合、score分散 | 幾何が共通でもcoverage・揺らぎが異なる可能性 |

mass–anisotropy degeneracyは重要な候補だが、自由関数としての逆問題の非一意性を、そのまま今回のZhao＋一定anisotropyの7変数モデルの厳密な特異性とは同一視しない。

## 研究の順序と完了条件

### 1. 統計実験と評価対象を定義する

現行7変数、同じsupport、固定Plummer tracer、非切断halo、Gaussian LOS尤度を出発点として記述する。既存の物理的定義を変更しない。主評価量はγ、有限rでのlog ρ(r)、局所傾斜s(r)、log M(r)。σ_los²(R)の予測も併記する。

半径は既存図のpc単位に加え、r/R_eの共通座標でも比較する。恒星が十分観測される範囲と中心・外側への外挿を分ける。半光半径付近の質量の安定性と、中心傾斜の回復は異なる課題である。これは[Wolf et al.](https://arxiv.org/abs/0908.2995)との接続にもなる。

成果物：7変数の仕様、観測設計、生成分布、prior測度、物理量と半径の一覧。未決事項を明示したprotocol案。

### 2. MCMCに依存しないKL幾何を調べる【最初の実作業】

Gaussian conditional truthについて、速度を積分した厳密なKL式を使う。R,eの設計積分を一次元／低次元quadratureで評価し、JeansPyのADからscore行列と高次方向微分を求める。4生成点、同じsupportの内点、境界、近傍を区別する。

最小のFisher固有値に閾値を置いて特異と宣言せず、score関数の線形独立性、KLの増加次数、連続な等価モデル族の有無を調べる。数値探索で見つかった候補を解析式・独立積分・必要な高精度計算で確かめる。有限の点検だけなら「数値的証拠」と明記する。

成果物：各生成点／領域を「正則の証明」「特異の証明」「弱いがfull rankの数値的証拠」「未確定」に分類した表。負の結果も成果とする。

### 3. 簡約モデルのRLCTと物理量の率を解析する

まず正則、純粋な冗長性、高次特異性、近特異性の対照例で式を確立する。Jeans問題から簡約モデルを作る場合は、KLの上下比較や支配項を確認する。内側power law近似の冗長性を、有限r_sの全Jeansモデルにそのまま移さない。

真の特異性が確認された領域では、必要に応じてNewton polyhedronや局所変数変換を使ってλ,mを求める。全モデルのexact RLCTが難しい場合は、証明できた局所結果・上下界・有限nの有効係数を別々に示す。

同時に、弱い方向がγ、ρ(r)、M(r)をどの程度変えるかを調べる。RLCTを求めるだけでは個々の物理量の推定精度に答えられない。

成果物：証明可能な簡約例、全モデルへの移行条件、半径別の識別可能性、物理量ごとの収縮率または限界。

### 4. 星数依存を分解し、crossoverを予測する

最初はpopulation contrastの重み exp(−nK)π を使い、少数次元の積分・局所積分・必要ならQMCで学習曲線を調べる。これは実現データのposteriorではなく、揺らぎを除いた理論対照である。

候補nは32, 64, 128, 256, 512, 1024, 2048, 4096, 8192を起点とし、必要な範囲だけ拡張する。数値は計算設計案であり、必要星数の予測結果ではない。広いn領域を扱える簡約モデルを先行させ、最初から全6次元の高精度周辺積分を目標にしない。

| 比較 | 変えるもの | 固定するもの |
|---|---|---|
| nだけの効果 | 星数 | 真値、g(R,e)、support、固定prior測度 |
| 速度精度 | eまたはe/速度scale | n、半径選択、真値 |
| 半径の情報 | R_min/R_e、R_max/R_e、選択関数 | n、速度誤差、真値 |
| prior感度 | 既存coordinate-uniformとjoint Jeffreysに対応する測度 | 尤度、support、真値、設計 |
| 生成分布 | likelihood-matched Gaussianと既存DF | 比較可能な真の条件付きmomentと設計 |

最後の2行には下記の科学的選択が残る。独立した結果として扱い、一度に全てを変更しない。

複数領域の局所自由エネルギーを比較し、dominantな領域の入れ替わりを探索する。有限nの移動はまずcrossoverと呼ぶ。滑らかなposteriorの変形、境界からの離脱、モード間の重み変化も対照に含め、数学的な相転移を証明したとはしない。

成果物：どの因子がγとρ(r)の違いを説明するかの予測。再現しなければ「この機構では説明できない」と記録する。

### 5. 必要星数とモデル評価へつなぐ

必要星数は N_req(T, ε, δ; q,g,π) として、対象T、許容誤差ε、失敗確率δを伴って定義する。例えばγの絶対誤差、log10 ρ(r)のdex誤差、区間幅・coverageを別々に指定する。nの保証が得られないなら、条件付きの予測や下限として報告する。

core/cuspの判定はγ=0への連続priorの点確率では定義できない。γの領域・decision rule、または別モデルの比較を明示し、後者ならモデルpriorも必要になる。γの境界や現行supportを無断で変更しない。

モデルの評価は、予測損失、物理量のbias・coverage、観測設計に対する頑健性、DFの物理的実現可能性で行う。λが小さいことを「haloモデルとして優秀」の同義語にしない。既に[Jeans＋Fisherによるforecast](https://arxiv.org/abs/2112.05166)があるため、新規性の候補はその適用域と破れ、priorを含む局所幾何、γと密度関数の異なる学習率の説明に置く。

反復Gaussian/DF mockによる実現posteriorのcoverage検証は、この理論段階の後の別実験とする。既存の4カタログは動機と照合用であり、学習曲線やcoverageの測定には使わない。

成果物：条件付き必要星数の定義と理論予測、検証可能な仮説、後続の反復mock設計。RLCT未確定でも、弱い識別可能性や推定不能性が明確なら研究として成立する。

## 実装前に固定する科学的選択（提案）

1. 主評価はγとlog ρ(r)、補助評価は局所傾斜・M(r)・σ_los²。誤差許容値とcore/cuspの判定閾値はまだ固定しない。最初の幾何解析はこれらの閾値に依存せず進められる。
2. 最初の厳密理論は、現行Gaussian尤度内の真値と設計分布gを条件とする。AGAMAのDFをGaussianに置き換えた科学的検証とは呼ばない。既存DFとの橋渡し条件を別に証明する。
3. 星数だけの主対照には固定されたpriorが必要。population設計gからのJeffreys priorを候補とし、現行の各カタログ依存prior π_J,g_n は別の比較とする。これは新しい理論対照の提案であって、既存priorの定義変更ではない。
4. 有限supportは現行のまま出発する。別のsupport、DF-positivity制約、anisotropy family等は感度解析案を具体化してから選ぶ。

この文書は研究方針であり、新しい生成family・prior・大規模計算protocolを採用済みとするものではない。

## 数値計算の境界

新しいコードは `research/singular_learning/` 配下に閉じ、JeansPyの現行公開JAX interfaceを呼ぶ。`src/jeanspy/model_jax.py` の球対称forwardと、`src/jeanspy/_zhao.py` の共通mass kernelを参照する。実験固有のJeffreys helperを公開APIとして扱わない。

ADは離散積分プログラムを微分する。物理的な連続モデルの解析性やrankの証明にはならない。clip、kernelの指数guard、積分の有限端点、境界での分岐、積分下の微分の妥当性を点検する。u_maxを固定した次数倍増だけでtail検証としない。

最新baseの `validation/zhao_derivative_stability/README.md` は微分の桁落ち修正と残る近特異点の誤差を記録している。これを出発点とし、rankを上げるためのjitter・固有値floor・pseudodeterminantを科学的targetへ追加しない。

数学的証明、連続積分への収束、離散プログラムの微分整合性、MCMC収束、科学的calibrationを独立した欄で記録する。
