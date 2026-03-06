import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def scatter3D(self,x,y,z,**args):
    df=self
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter3D(df[x],df[y],df[z],**args)
    ax.set_xlabel(df[x].name)
    ax.set_ylabel(df[y].name)
    ax.set_zlabel(df[z].name)
    return ax

def scatter_matrix(self,*,bins=32,fontsize=8):
    from matplotlib.colors import LogNorm
    df = self
    fig = plt.figure()
    fig.subplots_adjust(wspace=0,hspace=0)
    
    num_param = len(df.columns)
    counts_max = 0

    hists = []
    for i_hist in np.arange(num_param):
        tmp_hist = fig.add_subplot(num_param,num_param,i_hist+1)
        tmp_hist.minorticks_on()
        hists.append(tmp_hist)
        tmp_hist.set_title(df.columns[i_hist],fontsize=fontsize)
        #print(df[df.columns[i_hist]])
        counts,bins,patches = tmp_hist.hist(df[df.columns[i_hist]],bins=bins,histtype='step')
        #print('params_upper_plot[i_hist]:',params_upper_plot[i_hist],type(params_upper_plot[i_hist]))
        #print(params_name[i_hist])
        #if not (np.isnan(params_lower_plot[i_hist]) or np.isnan(params_upper_plot[i_hist])):
        #    print(params_lower_plot[i_hist],':',params_upper_plot[i_hist])
        #   tmp_hist.set_xlim(left=params_lower_plot[i_hist],right=params_upper_plot[i_hist])
        tmp_hist.tick_params(which='both',direction='in',right=True)
    
        counts_max = max(counts_max,np.max(counts))
#[axis.set_ylim(bottom=0,top=counts_max*1.1) for axis in hists ]
#[axis.set_yticklabels([]) for  axis in hists[1:]]
    
   
    axes_scatter = []
    for param_i in np.arange(num_param-1): # {0,1,2,...,num-2}, indicate the x_axis of the figures
        for param_j in np.arange(param_i+1,num_param): # {i+1, i+2, ..., n-1}, indicate the y_axis of the figures 
    # (i,j) = n*j+i (+1), where 0<=i,j<=n-1, where the last '+1' is required to shift to actual posision.
    # We want to start from the bottom raw. The bottom raw of i-th column is (n-1)-i.
    # Then move to above figure. The amount of shift is j-(i+1) because j={i+1,i+2,...,n-1}
    # Condequently, the the vertical axis indicater j' = ((n-1)-i) - (j-(i+1)) = n-2*i+j-2
    # Therefore the actual position is (i',j'), where i' = i.
            pos = num_param*(num_param-param_j)+param_i+1
            axes_scatter.append(fig.add_subplot(num_param,num_param,pos))
            present_axis = axes_scatter[-1]
            present_axis.minorticks_on()
            present_axis.tick_params(which='both',direction='in',right=True,top=True)
        
            hist_tmp = axes_scatter[-1].hist2d(df[df.columns[param_i]],df[df.columns[param_j]],normed=True,bins=bins,norm=LogNorm())
            
            #if not (np.isnan(params_lower_plot[param_i]) or np.isnan(params_upper_plot[param_i])):
            #    #print(param_i,':xlim->',params_lower_plot[param_i],':',params_upper_plot[param_i])
            #    present_axis.set_xlim(left=params_lower_plot[param_i],right=params_upper_plot[param_i])
            #if not (np.isnan(params_lower_plot[param_j]) or np.isnan(params_upper_plot[param_j])):
            #    #print(param_j,':ylim->',params_lower_plot[param_j],':',params_upper_plot[param_j])
            #    present_axis.set_ylim(bottom=params_lower_plot[param_j],top=params_upper_plot[param_j])
            ##hist_tmp = axes_scatter[-1].hist2d(MCparams[:,param_i],MCparams[:,param_j],normed=True,bins=32)
            if param_j != param_i+1:
                present_axis.set_xticklabels([],fontsize=fontsize)
            hists.append(hist_tmp) # for colorbar, but...
            if param_i==0:
                #present_axis.set_ylabel(params_name[param_j],rotation='horizontal')
                #present_axis.set_ylabel(params_name[param_j])
                present_axis.set_ylabel(df.columns[param_j],rotation=70.,fontsize=fontsize)
            else:
                present_axis.set_yticklabels([],fontsize=fontsize)
    return fig


pd.DataFrame.plot_scatter3D = scatter3D
pd.DataFrame.plot_scatter_matrix = scatter_matrix

# seaborn
import seaborn as sns
#sns.set(style="ticks",palette="tab10")
sns.set_style('whitegrid')
sns.set_palette("tab10",10)
import matplotlib

#args_pairplot = {"size":5,"plot_kws":{"s":8,"marker":"o"},"diag_kws":{"bins":32,"histtype":"stepfilled","alpha":0.3}}
args_pairplot = {"plot_kws":{"s":8,"marker":"o"},"diag_kws":{"bins":32,"histtype":"stepfilled","alpha":0.3}}

def mypairplot(
    mcchain,
    scatter_kws={"s":1,"marker":"o","alpha":0.1},
    kde_kws={"shade":True,"shade_lowest":False,"cmap":"jet"},
    hist_kws={"bins":32,"histtype":"stepfilled","alpha":0.3},
    **kws):
    
    print("Note: The following warnings occur, but no problem.\nUserWarning: The following kwargs were not used by contour: 'label', 'color' ")
    g = sns.PairGrid(mcchain,**kws)
    g = g.map_upper(plt.scatter,**scatter_kws)
    g = g.map_lower(sns.kdeplot,**kde_kws)
    g = g.map_diag(plt.hist, **hist_kws)
    return g