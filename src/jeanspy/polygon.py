import numpy as np


class Polygon:

    def __init__(self,points):
        """
        points: (n,2) ndarray
        """
        self.reset_points(points)
    
    @property
    def points(self):
        return self._points.copy()
    
    @property
    def x(self):
        return self.points[:,0]
    
    @property
    def y(self):
        return self.points[:,1]
    
    def reset_points(self,new_points):
        self._points = new_points
    
    
    
    def includes(self,points):
        """
        points: (n,2) ndarray
        
        Using Crossing Number Algorithm.
        """
        sx,sy = points.T  # sx.shape = (n,)
        
        # reshaping x and y for numpy.broadcasting:
        #     (m,1) * (n,) -> (m,n)
        # where
        #     n: the number of input points
        #     m: the number of polygon edges ( =len(self.points) )
        _x = self.x[:,np.newaxis]  # x.shape = (n,1)
        _y = self.y[:,np.newaxis]  # y.shape = (n,1)
        
        _x_shift = np.roll(_x,-1)
        _y_shift = np.roll(_y,-1)
        
        is_point_between_ith_sep = (_x-sx)*(sx-_x_shift) > 0
        is_point_above_ith_sep = (_x_shift-_x)*((_x_shift-_x)*(sy-_y)-(_y_shift-_y)*(sx-_x))>0
        n_sep_below_point = (is_point_between_ith_sep*is_point_above_ith_sep).sum(axis=0)
        is_odd_number_sep_above_point = (n_sep_below_point%2 == 1)

        return is_odd_number_sep_above_point