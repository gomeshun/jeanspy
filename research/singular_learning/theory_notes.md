# 理論ノート：Jeansの観測写像、RLCT、物理量の学習

2026-09-29。以下のJeansへの計算は本計画での導出・研究仮説であり、渡辺の文献がJeansモデルについて証明した結果ではない。全モデルのRLCTは未計算。

## 1. モデルを観測分布として定義する

構造変数を ψ=(log10 ρ_s, log10 r_s, α, β_h, γ, η)、η=−log10(1−β_a)、θ=(ψ,μ) とする。単位は既存protocolと同じ。現行supportは

\[
[-4,4]\times[0,5]\times[0.5,3]\times[3,10]\times[0,1.2]\times[-1,1]\times[-1000,1000].
\]

R_eとhaloの非切断条件は固定。真の連続forwardを f_ψ(R)=σ_los²(R;ψ)、S_ψ(R,e)=f_ψ(R)+e² と書く。

\[
p_θ(v\mid R,e)=\mathcal N(v;\mu,S_ψ(R,e)).
\]

観測設計をg(dR,de)とし、理論ではg p_θというjoint experimentを使える。gはθに依存せず、尤度にphotometry情報を追加する操作ではない。既存カタログに条件付ける場合は g_n=n^{-1}Σ_i δ_(R_i,e_i) を使うが、固定／random designの漸近条件を分ける。

理想的な位置だけのPlummer選択なら、Rの密度は

\[
g_R(R)\propto {2R\over R_e^2}(1+R^2/R_e^2)^{-2}
\mathbf1_{[R_{\min},R_{\max}]}(R),
\]

正規化定数は H(R_max)−H(R_min)、H(R)=R²/(R²+R_e²)。実際の有限AGAMA poolはこれへの近似かつ有限母集団である。無限nの理論のために同じ有限poolを独立な星として繰り返し使わない。

## 2. Gaussian KLは速度のMonte Carloを要しない

同じ設計gの下で、q(v|R,e)=N(μ_0,S_0)を生成分布とすると

\[
K_g(θ)=\frac12\int\left[
\log {S_ψ\over S_0}
+{S_0+(\mu-\mu_0)^2\over S_ψ}-1
\right]g(dR,de).
\tag{1}
\]

これは厳密なKLであり、残るのはR,eの設計積分とJeans forwardのみ。実装ではa=log(S_ψ/S_0)に対して a+expm1(−a) を使い、さらに小さいaでは級数によって相殺を点検する。

正の分散が局所で上下に有界なら

\[
K_g=\tfrac14\int a^2\,dg
+\tfrac12(\mu-\mu_0)^2\int S_0^{-1}\,dg
+\text{higher-order terms}.
\tag{2}
\]

したがってゼロ集合は μ=μ_0 かつ f_ψ(R)=f_ψ0(R) がgの下でほとんど至る所成り立つ集合。複数のθが近い曲線を与えることと、完全に等しい曲線を与えることを区別できる。

\[
I_{ψψ,g}=\tfrac12\int D(R,e)^T D(R,e)\,dg,
\quad D=\partial_ψ\log S_ψ,
\quad I_{μμ,g}=\int S_ψ^{-1}\,dg,
\quad I_{ψμ,g}=0.
\tag{3}
\]

固定設計の全情報は和を取り、n I_gnになる。score関数のrankが一次の局所識別可能性を決める。小さい正の固有値と厳密なゼロは異なる。数値固有値の大きさは座標scaleに依存するため、無次元座標・明示した尺度を使い、raw／scaled scoreの両方を記録する。

同一Rでeだけを変えても、構造scoreは同じ∇f(R)のscalar倍であり新しい方向は増えない。異なるRをk種類しか持たない設計では構造情報rankは高々min(k,6)。一方、64星や256星というだけでrank欠損は導けない。

最初の証明候補は、設計支持内の6半径における6×6の構造score行列の非零行列式である。連続性がありgがその近傍に正の重みを持てば、この非零性はpopulation Gram行列の正定値性に結びつく。ただし浮動小数点のdet≠0だけでは証明にならず、解析式や保証付き誤差で確認する。逆に多くの数値点でdetが小さいことは、恒等的なゼロの証明にはならない。局所full rankが示されても、離れた等価な解の有無は別途調べる。

## 3. 今回のどこが特異かは未確定

パラメータが7個あること、強く相関すること、ρ(r)がr=0で発散することは、統計モデルの特異性の証明にならない。

Zhao密度は

\[
\rho(r)=\rho_s x^{-γ}(1+x^α)^{-(β_h-γ)/α},\quad x=r/r_s.
\tag{4}
\]

有限r>0ではγ=0でも通常はγへの微分が消えない。また、core truthのγ=0だけでなく、core/cusp双方のβ_h=3が現行support境界にある。full-rankの二次KLと通常の全次元接錐なら、境界による切断が起きても積分のn^{-d/2}指数は変わらない。境界にあることだけでλが小さくなるとはいえない。誤指定で制約最適点に一次項が残る場合は別途展開する。

容易に考えられる厳密な冗長性にもdomainの確認が必要：β_h=γなら純粋power lawだが、今回のβ_h≥3、γ≤1.2では不可能。ρ_s=0、r_s=∞、r_s=0、α=0も現行の有限supportに含まれない。近づく領域の有限n効果は研究できるが、その極限モデルのRLCTを有限の真値に代入しない。

r≪r_sでρ≈A r^{-γ}、A=ρ_s r_s^γとなり、ρ_sとr_sの分離や外側shapeが弱くなることは候補機構。ただしLOS積分はRより大きい3次元半径まで伸びる。観測R_max≪r_sだけで全Jeans forwardが厳密にpower lawになるとはいえず、remainderとtailの制御が必要。

## 4. RLCTが言うこと／個々の物理量に別途必要なこと

局所KL contrast K≥0とprior測度πを指定し、ζ(z)=∫K(θ)^zπ(θ)dθの最大の極を−λ、その次数をmとする。標準的な解析性・可積分性・relative finite variance等の条件下で

\[
F_n=-\log\int e^{-nL_n(θ)}\pi(θ)dθ
=nL_n(θ_*)+λ\log n-(m-1)\log\log n+O_p(1).
\tag{5}
\]

ここで L_n=−n^{-1}Σ log p_θ、θ_*はpopulation risk最小点。一般の誤指定ではKはKLそのものではなく最小値を引いたcontrastになる。原典：[Watanabe 2013, §3, Theorem 2](https://jmlr.org/papers/volume14/watanabe13a/watanabe13a.pdf)。

正則内点・正の滑らかなpriorならλ=d/2,m=1。今回full rankなら全7変数でλ=7/2が比較基準。μを解析積分した6変数samplerでも、evidenceにはμ積分のn^{-1/2}因子が残るので、単にλ=3とはしない。構造だけのλを報告するならμの1/2を明示して分離する。

likelihood-matched、通常温度のBayesで所要条件が満たされると、posterior predictiveの平均KL汎化誤差の先頭項はλ/n。これはγのMSE、ρ(r)の区間幅、頻度論的coverageの式ではない。温度を変えた場合などにはsingular fluctuationも関わる。[Watanabe 2010, Lemma 3 / Theorem 2](https://jmlr.org/papers/volume11/watanabe10a/watanabe10a.pdf)。

研究対象Tについて、正則なら

\[
\mathrm{Var}(T\mid D_n)\simeq
{1\over n}\nabla T^T I_g^{-1}\nabla T.
\tag{6}
\]

特異な場合、この逆行列式で済ませない。完全に等価な観測分布の集合上でTが一定でなければ、観測だけによるTの一意な回復は不可能。一次のnull方向でも高次で識別可能な場合があるため、∇Tとnull方向の内積だけでは不可能性を断定できない。

一つの道具として、局所集合Uにおける

\[
\omega_T(\varepsilon)=\sup_{θ\in U:K_g(θ)\le\varepsilon^2}
|T(θ)-T(θ_0)|
\tag{7}
\]

を調べる。posteriorがK=O(1/n)に集中する条件が別途得られれば、ω_T(n^{-1/2})が物理量の学習尺度の候補になる。これだけで有限標本coverageや上限を証明したことにはしない。異なる最小点がある場合はUを超えたglobalな曖昧さも調べる。

## 5. γとρ(r)が違う動きをする理由の候補

式(4)から、自然対数座標で

\[
s(r)=-{d\log\rho\over d\log r}
=γ+(β_h-γ){x^α\over1+x^α},
\tag{8}
\]

\[
{\partial\log\rho\over\partial\log\rho_s}=1,
\quad {\partial\log\rho\over\partial\log r_s}=s(r),
\quad {\partial\log\rho\over\partial γ}
=-\log x+{1\over α}\log(1+x^α).
\tag{9}
\]

ρ(r)はγだけでなくnormalization、scale、外側shapeとそれらのjoint posteriorに依存する。同じγ marginalでもρ(r)のposteriorは変わりうる。逆に、あるrでparameterの相関が相殺すれば、γが広くてもρ(r)が絞れる。

この仮説を、Fisherの弱い方向と∇log ρ(r)、∇s(r)、∇log M(r)の投影、および高次KLで検証する。rごとのpivotは結果として探し、半光半径でρそのものが最適に定まると仮定しない。M(r_1/2)の既知の安定性は別の対照である。

## 6. priorを含む幾何

現行π_J∝√det Iは、μに依存しなくても√I_μμという構造変数依存因子を含む。これを削ると別priorになる。

既存のposterior計算ではpriorの正規化定数を評価していない。保存されたlog priorやMCMC summaryから、直ちに正規化されたevidence・Bayes factorが得られるわけではない。異なるprior／モデルの自由エネルギーを比較する段階ではこの定数も必要になる。

同じ設計gをn倍するならI_n=nI_gなので√det I_n=n^{7/2}√det I_g。このθ非依存因子は**正規化されたprior**では消える。evidenceでこれを消さずに使うと、log n係数を誤る。一方、R,eの経験分布g_nが変わるとpriorの形も変わる。固定priorの漸近定理を、未検証のn依存priorへそのまま使わない。

Jeffreys密度はrank欠損上で消え、vanishing orderがRLCTに影響しうる。全領域でdet I=0なら通常のjoint Jeffreys密度は正規化できず、quotient上の測度等を別途定義する問題になる。pseudodeterminantへの置換は自明な修正ではない。

説明用の厳密な1変数例：p(y|u)=N(u^k,1)、truth u=0、k≥1、有限support。K=u^{2k}/2なので、正で滑らかなpriorならλ=1/(2k)、uの尺度はn^{-1/(2k)}。一方√I=k|u|^{k−1}をpriorとした積分の指数はλ=1/2になる。この結果は直接の積分変数変換で得られる。Jeansのλがこの値になるとは主張しない。

Jeffreys密度の平方根が特異集合で滑らかでない場合、標準定理のprior仮定を局所chartごとに確認する。bounded supportにしただけで全ての適用条件が自動的に満たされるとはしない。

smoothで可逆な再パラメータ化は同じprior測度をpush forwardすれば情報量やRLCTを変えない。log M(r_1/2)をnormalization座標に使うのは有益な表示・計算案だが、その座標で新たにuniformと置けば別priorである。

## 7. 近特異性とcrossoverを先に理解する

説明例 K_ε(u)=ε²u²+u⁴を考える。ε=0ではλ=1/4、固定ε>0では十分大きいnでλ=1/2。quartic項が決めるu∼n^{-1/4}でquadratic項はnε²u²∼ε²n^{1/2}となるため、crossover尺度はn∼ε^{-4}。小さいFisher固有値が正でも、長い有限n領域で特異モデルに似た振る舞いを取りうる。

これは近特異性の制御例であり、今回のJeansモデルについてεを同定した結果ではない。n→∞とε→0の順序を区別する。

population積分

\[
\widetilde Z(n)=\int e^{-nK_g(θ)}\pi(θ)dθ,
\quad \widetilde F=-\log\widetilde Z,
\quad λ_{\mathrm{eff}}(n)={d\widetilde F\over d\log n}
=n\,\widetilde{\mathbb E}_n[K_g]
\tag{10}
\]

はMCMCを使わず簡約モデル等で計算できる。標準的なLaplace型展開が成り立つ場合の漸近的なλと、有限nのλ_effを別に報告する。\widetilde π_n∝exp(−nK_g)πはpopulation contrastの分布で、観測posteriorのcoverageを再現するものではない。

局所領域a,bのpopulation自由エネルギー差の候補は、各局所積分に漸近展開が適用できる場合

\[
\Delta\widetilde F_{ab}(n)
\simeq n\Delta K_{ab}+\Delta λ_{ab}\log n
-\Delta(m-1)_{ab}\log\log n-\Delta\log C_{ab}.
\tag{11}
\]

Cはpriorの重みや体積定数を含む。λだけでは交差するnは決まらない。実現データでは経験損失の揺らぎも必要で、異なる予測分布間の損失差には一般にO_p(√n)の項がありうる。population式を誤差なしの「相転移点」としない。

有限nの正規化積分は通常滑らかであり、dominant領域の交換を直ちに非解析的な相転移と呼ばない。正則モデルでもpriorからlikelihoodへ支配が移ることはある。

## 8. DF mockへの理論的な橋

Gaussian log likelihoodの期待値は条件付きの平均と分散だけで決まる。任意のDF由来qが、測定誤差畳み込み後に E_q[v|R,e]=μ_0、Var_q[v|R,e]=S_0 を厳密に満たすなら

\[
E_q[-\log p_θ]+E_q[\log p_{θ_0}]=K_g(θ)
\tag{12}
\]

となり、右辺は式(1)と同じ。従ってpopulationのGaussian contrast幾何を共有できる可能性がある。q=p_θ0とは限らず、Gaussian posterior predictiveのqへのKLがゼロになるという意味ではない。

AGAMAの有限表現、selection、Jeans積分の誤差がmoment一致を崩す場合は、そのずれを評価する。条件付き三次・四次momentはscoreの揺らぎに影響し、Gaussian期待Fisherと真のscore covarianceは一般に異なる。regular misspecificationでもH=E_q[−∂²log p]とJ=E_q[score score^T]を区別する必要があり、posterior幅と推定量の頻度論的ばらつきが一致するとは限らない。

特異かつ誤指定ではrelative finite variance等の仮定が追加の論点になる。moment一致だけで全てのSLT定理・coverageを移せたとはしない。この橋を検証すること自体が、Gaussian理論と平衡DF mockの役割を整理する成果になる。

## 9. 必要星数の定義と識別限界

例としてpoint estimatorを用いるなら

\[
N_{\rm req}(T,\epsilon,\delta;q,g,\pi)
=\inf\{N:\ \forall n\ge N,\quad
P_{q,g}(|\widehat T_n-T_0|\le\epsilon)\ge1-\delta\}.
\tag{13}
\]

posterior区間については幅とcoverageを別条件にする。式(13)が実際に有限かどうかも研究対象。有限のn gridで測定する場合は「検証した範囲内の必要星数」として、全n≥Nの保証と区別する。ρ(r)のpointwiseと同時band、星の存在する領域と外挿領域も区別する。

二つの候補の一星あたりKLが小さいと、独立n星でもKLはn倍にしかならない。nK≪1での判別の難しさを情報論的な下限として検討できる。ただし複合仮説ではnuisanceを最適化して比較し、simple-hypothesisのnK尺度をそのまま達成可能な必要星数としない。

## 10. 適用条件の監査項目

- gの支持、R→0の扱い、独立性、固定／random designの選択。
- 分散の正値性・momentの可積分性、連続Jeans積分の収束、積分下の微分。
- 同じ予測分布を与える最小集合、真値が内点か境界か、full rankか高次特異性か。
- 正規化可能な固定prior測度、特異集合でのvanishing order、n依存priorとの区別。
- analytic continuous modelと、clip/guardを含む離散実装との差。
- 誤指定の場合、contrastの最小集合・relative finite variance・score covariance。

この監査から正則性が示された場合にも、有限nの弱い識別可能性と物理量依存の精度を研究する道筋は残る。特異性の発見を成功条件にはしない。
