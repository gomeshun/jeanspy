# 文献・位置づけ・確認範囲

参照日：2026-09-29。一次資料を中心に確認した。下記は網羅的なsystematic reviewではなく、研究方針を定めるための出発点。先行研究が存在しないという優先権の主張はしない。

| 資料 | 今回の確認範囲と使い方 |
|---|---|
| [渡辺澄夫：特異学習理論（ユーザー指定ページ）](https://sites.google.com/view/sumiowatanabe/home/特異学習理論) | 概説本文。局所自由エネルギーの競合、Fisherのrank、局所RLCTを入口にする。深層学習に関する図式を、そのまま7変数Jeansモデルの結論にしない。 |
| [Watanabe (2013), A Widely Applicable Bayesian Information Criterion, JMLR 14, 867–897](https://jmlr.org/papers/v14/watanabe13a.html) / [PDF](https://jmlr.org/papers/volume14/watanabe13a/watanabe13a.pdf) | §3のfundamental conditions、normal crossing、Lemma 3、Theorem 2、§4のWBIC、§7の予測とevidenceの区別を確認。主な数学的基礎。 |
| [Watanabe (2010), Asymptotic Equivalence of Bayes Cross Validation and Widely Applicable Information Criterion in Singular Learning Theory, JMLR 11, 3571–3594](https://jmlr.org/papers/v11/watanabe10a.html) / [PDF](https://jmlr.org/papers/volume11/watanabe10a/watanabe10a.pdf) | fundamental conditions、Remark 3、Lemma 3、Theorem 2を確認。λ/nはposterior predictiveの学習曲線についての記述。WAIC/LOOは速度予測の評価に有用だがγ回復の代用ではない。 |
| [Watanabe, Recent Advances in Algebraic Geometry and Bayesian Statistics, arXiv:2211.10049](https://arxiv.org/abs/2211.10049) | abstract・書誌情報を確認。特異点解消とBayesの橋渡しを学ぶ次の総説として位置づける。今回の定理の細部は上の原著PDFに依拠。 |
| [Watanabe & Amari (2002), The Effect of Singularities in a Learning Machine when the True Parameters Do Not Lie on such Singularities](https://papers.nips.cc/paper_files/paper/2002/hash/c2ba1bc54b239208cb37b901c0d3b363-Abstract.html) | abstractを確認。真値が厳密な特異点になくても近傍が有限標本に影響する、という問いの先行研究。個別結果をJeansに移植しない。 |
| [The Local Learning Coefficient: A Singularity-Aware Complexity Measure, arXiv:2308.12108](https://arxiv.org/abs/2308.12108) | 現在のabstract・書誌情報を確認。局所係数の数値推定は後段の候補。近傍・温度・数値アルゴリズム依存の推定量をexact RLCTと呼ばない。まず既知の解析例で検証する。 |
| [Guerra, Geha & Strigari, Forecasts on the Dark Matter Density Profiles of Dwarf Spheroidal Galaxies with Current and Future Kinematic Observations, arXiv:2112.05166](https://arxiv.org/abs/2112.05166) | abstractを確認。Jeans＋Fisherで星数・誤差と密度profile精度を予測する直接関連先行研究。本文のモデル・support・prior・proper motionの条件を比較表にすることを次段階の文献課題とする。報告された必要星数を今回のmockに流用しない。 |
| [Wolf et al. (2010), Accurate masses for dispersion-supported galaxies](https://arxiv.org/abs/0908.2995) / [MNRAS本文](https://academic.oup.com/mnras/article/406/2/1220/1002447) | abstractと§2–3の議論を確認。半光半径付近のenclosed massの比較的良い識別と、density slopeの難しさを結ぶ対照。任意の半径・profileへの万能保証として扱わない。 |

## 本研究で新たに問う内容

1. 有限supportのZhao＋一定anisotropy＋Gaussian LOSモデルは、どの生成点・観測設計で厳密に特異か。full rankなら、どの近特異性が有限nの挙動を支配するか。
2. γとρ(r)の異なる挙動を、同じKL幾何から物理量ごとに説明できるか。
3. coordinate-uniformとjoint Jeffreysの違いを、局所の体積とpriorのvanishing orderから理解できるか。
4. 固定nだけでなく、速度誤差と半径選択を変えることで、同じ星数の情報量がどう変わるか。
5. Gaussian理論のpopulation contrastが、momentの一致するDF mockへどこまで移り、揺らぎ・coverageではどこで破れるか。

自動微分はこれらの問いを検証可能にする手段として位置づける。研究の成果はJAXの使用自体ではなく、証明した条件、反証可能な学習曲線の予測、推定可能な物理量と限界の説明である。
