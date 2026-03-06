import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy.geometry import Point, Polygon
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D

#from py_header import mylib02 as my
from jeanspy.coord import projected_distance,projected_angle

######## config ########
IMPORT_SDSS_FILENAME = "Draco_SDSS.csv"
SDSS_SKIP_ROWS = 1
CMD_PLOT_TITLE = "SDSS observation of Draco dSph: "
CMD_XLABEL = r'$g-i$'
CMD_YLABEL = r'$i$'
POS_PLOT_TITLE = "SDSS observation of Draco: posision"
EXPORT_FILENAME = "Draco_SDSS_cut_xy"
EXPORT_FILENAME2 = "Draco_SDSS_cut_r"

SLIDER_VAL_INIT = 1.0
SLIDER_VAL_MAX = 1.5

AX_POS_ASPECT = 0.5
AX_POS_XRANGE = 2.1
AX_POS_YRANGE = AX_POS_XRANGE*AX_POS_ASPECT

AX_NUM_RDEGRANGE = SLIDER_VAL_INIT
AX_NUM_BINNUM = 50

EXPORT_FILENAME += str(SLIDER_VAL_INIT)+".csv"
EXPORT_FILENAME2 += str(SLIDER_VAL_INIT)+".csv"

CMD_CUT_FNAME = "Draco_CMDcut.csv"
CMD_CUT = np.loadtxt(CMD_CUT_FNAME,delimiter=",")
#CMD_CUT = np.array([[1.57,15.18],[0.90,17.51],[0.75,18.71],[0.37,19.30],[-0.03,19.11],[-0.42,20.00],[-0.42,20.19],[-0.42,20.99],[0.00,20.47],[0.36,20.62],[0.58,20.25],[0.90,20.28],[1.04,18.93],[1.89,15.82]])
CMD_CUT_C = np.append(CMD_CUT,np.array([CMD_CUT[0]]),axis=0) #2nd argument must have same dimensions to 1st one
#print(CMD_CUT)
#print(CMD_CUT_C)

dSph_prop = pd.read_csv("dSph_property.csv",index_col=0)
draco_prop = dSph_prop.loc["Draco"]

######## def ########
def set_axpos_coordinates():
    ax_pos.set_xlabel(r'$\xi$[deg]')
    ax_pos.set_ylabel(r'$\eta$[deg]')
    ax_pos.set_xlim(-AX_POS_XRANGE,AX_POS_XRANGE)
    ax_pos.set_ylim(-AX_POS_YRANGE,AX_POS_YRANGE)
    ax_pos.set_title(POS_PLOT_TITLE)
    ax_pos.set_aspect(1/AX_POS_ASPECT)

def set_axcmd_coordinates():
    ax_cmd.set_title(CMD_PLOT_TITLE+IMPORT_SDSS_FILENAME)
    ax_cmd.set_xlabel(CMD_XLABEL) # "r" before the string means "raw string"
    ax_cmd.set_ylabel(CMD_YLABEL) # we can use TeX format by sandwitching strings with "$"
    ax_cmd.set_xlim(-0.5,2.5)
    ax_cmd.set_ylim(24,14)
    ax_cmd.set_aspect(0.2)

def set_axpos_refference():
    ax_pos.scatter(0,0,c='red',s=0.5)
    ax_pos.plot(hlr_ra,hlr_dec)

def set_axcmd_refference():
    ax_cmd.plot(CMD_CUT_C.T[0],CMD_CUT_C.T[1])

######## def2 ########
from jeanspy.cmd_utilities import inpoly

######## main routin ######## 


#### import Draco data ####
data = np.loadtxt(IMPORT_SDSS_FILENAME,delimiter = ',',skiprows=SDSS_SKIP_ROWS)
print("imported: "+IMPORT_SDSS_FILENAME+", data shape:" +str(data.shape))

#### split the imported data ####
pos = data[:,1:3]
ra = pos[:,0]
dec = pos[:,1]
g_obs = data[:,6] # 7th item
i_obs = data[:,8] # 9th item
g_err = data[:,11] #12th item
i_err = data[:,13] #14th item
g_ext = data[:,16] #17th item
i_ext = data[:,18] #19th item

g = g_obs - g_ext # mag: brighter < fainter, X_ext means the extinction effect, so the actual star is brighter than X_obs by X_ext.
i = i_obs - i_ext
gmi = g-i

x = ra - draco_prop.RAdeg#my.DRACO_CENTER_RA_DEG 
y = dec - draco_prop.DEdeg#my.DRACO_CENTER_DEC_DEG
xy = np.array([x,y]).T

#### impose CMD cut here ####
ind_cut = np.where(inpoly(gmi,i,CMD_CUT.T[0],CMD_CUT.T[1]))
coind_cut = np.where(np.logical_not(inpoly(gmi,i,CMD_CUT.T[0],CMD_CUT.T[1])))
print(ind_cut)
print(coind_cut)
gmi,co_gmi = gmi[ind_cut],gmi[coind_cut]
i,co_i = i[ind_cut],i[coind_cut]
ra,dec = ra[ind_cut], dec[ind_cut]
xy = xy[ind_cut]
x,co_x = x[ind_cut], x[coind_cut]
y,co_y = y[ind_cut], y[coind_cut]
#co_x = x[coind_cut] #it is not valied because x is already changed
#co_y = y[coind_cut]

theta = np.arange(360)
hlr_ra  = draco_prop.RAdeg*np.cos(np.deg2rad(theta))
hlr_dec = draco_prop.DEdeg*np.sin(np.deg2rad(theta))

#### plot the data ####
fig = plt.figure() # "figure": a figure in TeX, "axis": a pabel in a figure in TeX
ax_pos = fig.add_subplot(2,2,1)
ax_cmd = fig.add_subplot(2,2,2) #I,J,K means shape(I,J), K-th figure 
ax_num = fig.add_subplot(2,2,3)
#ax1 = fig.add_subplot(1,2,1) #I,J,K means shape(I,J), K-th figure 
#ax2 = fig.add_subplot(1,2,2, projection='3d')


set_axpos_coordinates()
pos_copathcol = ax_pos.scatter(co_x,co_y,c='0.9',s=0.1) #c='0.1' means c='gray'
pos_pathcol = ax_pos.scatter(x,y,c='gray',s=0.1) #c='0.5' means c='gray'
set_axpos_refference()


set_axcmd_coordinates()
cmd_copathcol = ax_cmd.scatter(co_gmi,co_i,c='0.9',s=0.1)
cmd_pathcol = ax_cmd.scatter(gmi,i,c='gray',s=0.1)
set_axcmd_refference()

binnum = 100
edges = np.arange(0.00,AX_NUM_RDEGRANGE+1e-8,AX_NUM_RDEGRANGE/AX_NUM_BINNUM)
edge_centers = (edges[1:]+edges[:-1])/2
#print(edge_centers)

#n,bins,patches = ax_num.hist(my.dist2d(np.array([x,y]),np.array([0,0])),bins=edges,histtype='step',normed=False)
#DEBUG print("projected_deg:\n",(projected_angle(ra=ra,de=dec,ra_center=draco_prop.RAdeg,de_center=draco_prop.DEdeg,dtype="deg")))
#DEBUG quit()
n,bins,patches = ax_num.hist(projected_angle(ra=ra,de=dec,ra_center=draco_prop.RAdeg,de_center=draco_prop.DEdeg,dtype="deg"),bins=edges,histtype='step',normed=False)
ax_ndensity = fig.add_subplot(2,2,4) # plot the 2d density (\Sigma(R))
#print(n)
#print(bins)
#ax_ndensity.plot(edge_centers,n/2./np.pi/edge_centers/np.sum(n))

ax_ndensity.errorbar(
    edge_centers,
    n/2./np.pi/edge_centers/np.sum(n), # density: (number of stars in ith bin)/(2*pi*R) where R is the center of ith bin
    yerr=np.sqrt(n)/2./np.pi/edge_centers/np.sum(n),
    fmt='.'
)
#ax_ndensity.set_xscale('log')
ax_ndensity.set_yscale('log')
ax_ndensity.grid(which='both')

ax_slider = fig.add_axes([0.1,0.01,0.8,0.03]) #left, bottom, width, height
ax_button = fig.add_axes([0.92,0.1,0.06,0.03])
slider = Slider(ax_slider,'cut radius [deg]',0,SLIDER_VAL_MAX,valinit=SLIDER_VAL_INIT)
button = Button(ax_button,'export')

def slider_update(slider_val):
    #ax_pos.clear()  
    #set_axpos_coordinates()
    #ind_cut = projected_angle(xy.T,np.array([0,0]))<slider_val
    #ind_res = my.dist2d(xy.T,np.array([0,0]))>=slider_val
    ind_cut = projected_angle(draco_prop.RAdeg,draco_prop.DEdeg,ra,dec,dtype="deg")<slider_val
    ind_res = projected_angle(draco_prop.RAdeg,draco_prop.DEdeg,ra,dec,dtype="deg")>=slider_val
    x_reshaped = x[ind_cut] #[:] means the refference
    y_reshaped = y[ind_cut]
    pos_copathcol.set_offsets(np.concatenate((np.array([x[ind_res],y[ind_res]]).T,np.array([co_x,co_y]).T)))
    pos_pathcol.set_offsets(np.array([x_reshaped,y_reshaped]).T)   
    #set_axpos_refference()
    
    #ax_cmd.clear()
    #set_axcmd_coordinates()
    gmi_cut, gmi_cocut = gmi[ind_cut], gmi[ind_res]
    i_cut, i_cocut = i[ind_cut], i[ind_res]
    
    #ax_cmd.scatter(gmi_cocut,i_cocut,c='0.9',s=0.1)
    cmd_copathcol.set_offsets(np.concatenate((np.array([co_gmi,co_i]).T,np.array([gmi_cocut,i_cocut]).T)))
    #ax_cmd.scatter(co_gmi,co_i,c='0.9',s=0.1)
    cmd_pathcol.set_offsets(np.array([gmi_cut,i_cut]).T)
    #set_axcmd_refference()

    ax_num.clear()
    edges = np.arange(0.00,AX_NUM_RDEGRANGE+1e-8,AX_NUM_RDEGRANGE/AX_NUM_BINNUM)
    #ax_num.hist(my.dist2d(np.array([x_reshaped,y_reshaped]),np.array([0,0])),bins=edges,histtype='step',normed=False)
    ax_num.hist(
        projected_angle(draco_prop.RAdeg,draco_prop.DEdeg,x_reshaped+draco_prop.RAdeg,y_reshaped+draco_prop.DEdeg,dtype="deg"),
        bins=edges,histtype='step',normed=False
    )
    #print(vlos_pathcol)

    fig.canvas.draw()
    #fig.show()
    #fig.canvas.flush_events()
    

def button_clicked(event):
    #print(event)
    #ind_cut = my.dist2d(xy.T,np.array([0,0]))<slider.val
    ind_cut = projected_angle(draco_prop.RAdeg,draco_prop.DEdeg,ra,dec,dtype="deg")<slider.val
    x_reshaped = x[ind_cut] #[:] means the refference
    y_reshaped = y[ind_cut] #note that "slider_update" cannot update the variable x and y inside the function.
    X = np.array([x_reshaped,y_reshaped]).T
    print(X)
    print(X.shape)
    np.savetxt(EXPORT_FILENAME,X,'%.6e,%.6e',delimiter=",",header='x_cut[deg],y_cut[deg]',comments='#')
    print(EXPORT_FILENAME," is exported")

    OUTOF_CIRCLE_SUF = "outof_"
    x_outof_circle = x[np.logical_not(ind_cut)]
    y_outof_circle = y[np.logical_not(ind_cut)]
    X_outof_circle = np.array([x_outof_circle,y_outof_circle]).T
    np.savetxt(OUTOF_CIRCLE_SUF+EXPORT_FILENAME,X_outof_circle,'%.6e,%.6e',delimiter=',',header='x_outof_circle[deg],y_outof_circle[deg]',comments='#')
    print(OUTOF_CIRCLE_SUF+EXPORT_FILENAME,'is exported')

    myheader = 'r_cut[deg]. r_cut = '+str(slider.val)
    #np.savetxt(EXPORT_FILENAME2,my.dist2d(X.T,np.array([0,0])),'%.6e',delimiter=",",header=myheader,comments='#')
    np.savetxt(
        EXPORT_FILENAME2,
        projected_angle(
            draco_prop.RAdeg,draco_prop.DEdeg,x_reshaped+draco_prop.RAdeg,y_reshaped+draco_prop.DEdeg,dtype="deg"
        ),
        '%.6e',delimiter=",",header=myheader,comments='#'
    )
    #projected_angle(draco_prop.RAdeg,draco_prop.DEdeg,ra,dec,dtype="deg")
    print(EXPORT_FILENAME2," is exported")

    myheader = 'r_cut[deg]. r_cut = '+str(slider.val)
    #np.savetxt(OUTOF_CIRCLE_SUF+EXPORT_FILENAME2,my.dist2d(X_outof_circle.T,np.array([0,0])),'%.6e',delimiter=",",header=myheader,comments='#')
    np.savetxt(OUTOF_CIRCLE_SUF+EXPORT_FILENAME2,projected_angle(draco_prop.RAdeg,draco_prop.DEdeg,x_outof_circle+draco_prop.RAdeg,y_outof_circle+draco_prop.DEdeg,dtype="deg"),'%.6e',delimiter=",",header=myheader,comments='#')
    print(OUTOF_CIRCLE_SUF+EXPORT_FILENAME2," is exported")

def onclick(event):
    #print(event)
    if event.inaxes:
        if event.inaxes!=ax_slider and event.inaxes!=ax_button:
            print("[x,y]:[%f:%f]" % (event.xdata,event.ydata))

slider_update(SLIDER_VAL_INIT) # initialize
slider.on_changed(slider_update)
button.on_clicked(button_clicked)

fig.canvas.mpl_connect('button_press_event',onclick)

fig.show()
input("press any key to exit...\n")





















