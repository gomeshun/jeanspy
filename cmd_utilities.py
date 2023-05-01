import numpy as np

def inpoly(x,y,xs_vertex,ys_vertex):
    ''' 
    xs_vertex[:], ys_vertex[:]: polygon
    x, y: point
    '''
    x1 = xs_vertex[:,np.newaxis]
    y1 = ys_vertex[:,np.newaxis]
    x2 = np.roll(x1,-1)
    y2 = np.roll(y1,-1)
    print(np.array([x1[:,0],y1[:,0]]))
    print(np.array([x2[:,0],y2[:,0]]))
    ispointbetweenithsep = (x1-x)*(x2-x)<0
    ispointaboveithsep = (x2-x1)*((x2-x1)*(y-y1)-(y2-y1)*(x-x1))>0
    numofsepbelowpoint = (ispointbetweenithsep*ispointaboveithsep).sum(axis=0)
    isoddnumbersepabovepoint = (numofsepbelowpoint%2 == 1)
    #print(ispointbetweenithsep)
    #print(ispointaboveithsep)
    #print(numofsepbelowpoint)
    #print(isoddnumbersepabovepoint)
    return isoddnumbersepabovepoint

