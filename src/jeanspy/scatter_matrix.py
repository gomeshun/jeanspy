import matplotlib.lines as mlines
import numpy as np
from pandas.plotting._matplotlib.tools import _set_ticks_props, _subplots
from pandas.core.dtypes.missing import notna

def scatter_matrix(
    frame,
    alpha=0.5,
    figsize=None,
    ax=None,
    grid=False,
    diagonal="hist",
    marker=".",
    density_kwds=None,
    hist_kwds=None,
    range_padding=0.05,
    plot_axes="all",  # "all", "lower", "upper"
    ylabel_direction=None,  # "left", "right",
    refresh_labels=False,
    scales=None,  # list of scale ["linear" or "log"]
    **kwds
):
    '''
    Modification of pandas.scatter_matrix.
    '''
    
    if scales is None: scales = ["linear"] * len(frame.columns)
    
    def _get_marker_compat(marker):
        if marker not in mlines.lineMarkers:
            return "o"
        return marker


    df = frame._get_numeric_data()
    n = df.columns.size
    naxes = n * n
    fig, axes = _subplots(naxes=naxes, figsize=figsize, ax=ax, squeeze=False)

    if axes.ndim != 2:  # Note: axに二次元配列入れてても_subplotsは一次元配列を生成する
        axes = axes.reshape((n,n))

    ylabel_direction = ylabel_direction or ("left" if plot_axes != "upper" else "right")  # default: "lower" --> "left", "upper"  --> "right"

    # no gaps between subplots
    fig.subplots_adjust(wspace=0, hspace=0)

    mask = notna(df)

    marker = _get_marker_compat(marker)

    hist_kwds = hist_kwds or {}
    density_kwds = density_kwds or {}

    # GH 14855
    kwds.setdefault("edgecolors", "none")

    boundaries_list = []
    for i,a in enumerate(df.columns):
        values = df[a].values[mask[a].values]
        rmin_, rmax_ = np.min(values), np.max(values)
        if scales[i]=="linear":
            rdelta_ext = (rmax_ - rmin_) * range_padding / 2.0
            boundaries_list.append((rmin_ - rdelta_ext, rmax_ + rdelta_ext))
        elif scales[i]=="log":
            rdelta_ext = np.exp((np.log(rmax_) - np.log(rmin_)) * range_padding / 2.0)
            boundaries_list.append((rmin_ / rdelta_ext, rmax_ * rdelta_ext))


    for i, a in enumerate(df.columns):
        for j, b in enumerate(df.columns):
            ax = axes[i, j]
            ax.set_visible(False)  # 一旦非表示にしておく

            if i == j:
                values = df[a].values[mask[a].values]

                # Deal with the diagonal by drawing a histogram there.
                if diagonal == "hist":
                    if "bins" not in hist_kwds: hist_kwds["bins"]=16
                    _hist_kwds = {key:val for key,val in hist_kwds.items()}
                    if scales[i] == "log" and type(hist_kwds["bins"]) is int:
                        _hist_kwds["bins"] = np.logspace(np.log10(np.min(values)),np.log10(np.max(values)),hist_kwds["bins"])
                    ax.hist(values,**_hist_kwds)
                    

                elif diagonal in ("kde", "density"):
                    from scipy.stats import gaussian_kde

                    y = values
                    gkde = gaussian_kde(y)
                    ind = np.linspace(y.min(), y.max(), 1000)
                    ax.plot(ind, gkde.evaluate(ind), **density_kwds)
                   

                ax.set_xscale(scales[i])
                ax.set_xlim(boundaries_list[i])
                ax.set_visible(True)

            elif plot_axes == "all" or (i > j and plot_axes == "lower") or (i < j and plot_axes == "upper"):
                common = (mask[a] & mask[b]).values

                ax.scatter(
                    df[b][common], df[a][common], marker=marker, alpha=alpha, **kwds
                )
                
                ax.set_xscale(scales[j])
                ax.set_yscale(scales[i])

                ax.set_xlim(boundaries_list[j])
                ax.set_ylim(boundaries_list[i])
                ax.set_visible(True)

            ax.set_xlabel(b)
            ax.set_ylabel(a)

            if refresh_labels:
                ax.xaxis.set_visible(True)
                ax.yaxis.set_visible(True)

            if plot_axes in ("all", "lower"):
                if j != 0:
                    ax.yaxis.set_visible(False)
                if i != n - 1:
                    ax.xaxis.set_visible(False)
            elif plot_axes == "upper":
                if ylabel_direction == "left" and i != j:
                    ax.yaxis.set_visible(False)
                else:  # elif ylabel_direction == "right":
                    if j == n - 1:
                        ax.yaxis.tick_right()
                        ax.yaxis.set_label_position('right') 
                    else:
                        ax.yaxis.set_visible(False)
                if i == 0:
                    ax.xaxis.tick_top()
                    ax.xaxis.set_label_position('top') 
                else:
                    ax.xaxis.set_visible(False)

    if len(df.columns) > 1:
        lim1 = boundaries_list[0]
        locs = axes[0][1].yaxis.get_majorticklocs()
        locs = locs[(lim1[0] <= locs) & (locs <= lim1[1])]
        adj = (locs - lim1[0]) / (lim1[1] - lim1[0])

        lim0 = axes[0][0].get_ylim()
        adj = adj * (lim0[1] - lim0[0]) + lim0[0]
        axes[0][0].yaxis.set_ticks(adj)

        if np.all(locs == locs.astype(int)):
            # if all ticks are int
            locs = locs.astype(int)
        axes[0][0].yaxis.set_ticklabels(locs)

    _set_ticks_props(axes, xlabelsize=8, xrot=90, ylabelsize=8, yrot=0)

    return axes