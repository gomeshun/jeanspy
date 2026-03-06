import numpy as np
from numpy import cos as c 
from numpy import sin as s

def projected_distance(dist,ra_center,de_center,ra,de,dtype="rad"):
    '''
    args:
        ra_center,de_center,ra,de: radian unit
    note:
        calculate the projected distance from the center position:
        dist = dist*sin(theta),
        cos_theta = c(de_center)*c(de)*(c(ra-ra_center))+s(de_center)*s(de)
    '''
    t = np.deg2rad if dtype == "deg" else lambda x:x
    cos_theta = c(t(de_center))*c(t(de))*(c(t(ra-ra_center)))+s(t(de_center))*s(t(de))
    return dist*np.sqrt(1-cos_theta*cos_theta)

def projected_distance_simple(dist,ra_center,de_center,ra,de,dtype="rad"):
    '''
    args:
        ra_center,de_center,ra,de: radian unit
    note:
        calculate the projected distance from the center position:
        dist = dist*sin(theta),
        cos_theta = c(de_center)*c(de)*(c(ra-ra_center))+s(de_center)*s(de)
    '''
    t = np.deg2rad if dtype == "deg" else lambda x:x
    return dist*np.sqrt( (c(t(de_center))*t(ra-ra_center))**2 + (t(de-de_center))**2 )

def projected_angle(ra_center,de_center,ra,de,dtype="rad"):
    '''
    args:
        ra_center,de_center,ra,de: radian unit
    note:
        calculate the projected distance from the center position:
        dist = dist*sin(theta),
        cos_theta = c(de_center)*c(de)*(c(ra-ra_center))+s(de_center)*s(de)
    '''
    t,t_inv = (np.deg2rad,np.rad2deg) if dtype == "deg" else (lambda x:x,lambda x:x)
    cos_theta = c(t(de_center))*c(t(de))*(c(t(ra-ra_center)))+s(t(de_center))*s(t(de))
    return t_inv(np.arccos(cos_theta))

def hms_to_h(h,m,s):
    return h+m/60.+s/3600.

def hms_to_deg(h,m,s):
    return (h+m/60.+s/3600.)*15

def dms_to_deg(d,m,s):
    '''
    return deg from input (d,m,s).  
    
    Note: 1.0/d is positive for positive d, negative for negative d,
          +inf for +0.0 and -inf for -0.0 (IEEE754)
    '''
    sign = np.sign(np.array(1.0)/d)
    return sign*(np.fabs(d)+m/60.+s/3600.)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d as Axes3D
    
    args = {'ra_center':0,'de_center':np.pi/2-0.5}
    ra = (np.arange(-np.pi,np.pi,0.01) + args['ra_center'])
    de = (np.arange(-np.pi,np.pi,0.01) + args['de_center'])
    ras, des = np.meshgrid(ra,de)
    dists = projected_distance(1,ra=ras,de=des,**args)

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.plot_surface(ras,des,dists)
    ax.set_xlabel('ra'),ax.set_ylabel('de'),ax.set_zlabel('dist')
    print(args)
    fig.show()
    input('press any key')
