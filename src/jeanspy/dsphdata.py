import pandas as pd
import numpy as np
from astropy.io import ascii
from astropy.coordinates import SkyCoord,Distance
import astropy.units as u
from copy import deepcopy as copy
from importlib.resources import files


DATA_DIR = files("jeanspy").joinpath("data")

class dSphProp:
    '''
    initialize a sampler instance.
    
    input:
        - dsph_name(str)   : dwarf spheroidal name listed in "Nearbygalaxies.dat".
    '''
    def __init__(self,dsph_name,ra=None,dec=None,distance=None,distance_err=None,verbose=True):        
        #self.dsph_prop = None
        self.load_dsph_property(dsph_name,ra,dec,distance,distance_err,verbose)
        
    def load_dsph_property(self,dsph_name,ra=None,dec=None,distance=None,distance_err=None,verbose=True):
        dsph_prop = ascii.read(DATA_DIR.joinpath("NearbyGalaxies.dat")).to_pandas().set_index("GalaxyName").loc[dsph_name]
        self.dsph_prop = dsph_prop
        _ra  = "{}h{}m{}s".format(dsph_prop.RAh,dsph_prop.RAm,dsph_prop.RAs) if (ra  is None) else ra
        _dec = "{}d{}m{}s".format(dsph_prop.DEd,dsph_prop.DEm,dsph_prop.DEs) if (dec is None) else dec
        _distance = Distance(distmod=dsph_prop["(m-M)o"]).pc if (distance is None) else distance
        self.dsph_prop_sc = SkyCoord(ra=_ra,dec=_dec,distance=_distance*u.pc)
        
        if verbose:
            print(self.dsph_prop_sc)
        
        if not distance_err is None:
            self.dsph_prop_sc.distance_err = distance_err
            print("distance_err = {}".format(distance_err))
        
        
    def to_sc(self):
        return copy(self.dsph_prop_sc)

class dSphData:
    '''
    data class of the observed data of dSph.
    
    attributes:
        - 
    
    '''
    idx_photo_default = dict(index_col=0,cut="in_cmdcut",ra_idx="raMean",dec_idx="decMean")
    idx_spec_default  = dict(index_col=0,cut="in_cmdcut",ra_idx="ra",    dec_idx="dec",radial_velocity_idx="v_helio",radial_velocity_err_idx="err_v_helio")
    
    def __init__(self,
                 fname_photo=None,fname_spec=None,
                 idx_photo=None,idx_spec=None):
        '''
        initialize a sampler instance.
        
        input:
            - dsph_name(str)   : dwarf spheroidal name listed in "Nearbygalaxies.dat".
            - fname_photo(str) : file name of photometry data
            - fname_spec(str)  : file name of spectroscopy data
            - idx_photo,idx_spec(dict) : dictionary of argument passed to "self.load_file".
        '''
        idx_photo = dSphData.idx_photo_default if idx_photo is None else idx_photo
        idx_spec = dSphData.idx_spec_default if idx_spec is None else idx_spec

        if not fname_photo is None:
            self.load_csv("photo",fname_photo,**idx_photo)
        if not  fname_spec is None:
            self.load_csv("spec", fname_spec, **idx_spec)
        
    def load_csv(self,name,fname,index_col=None,cut=None,
                 ra_idx="ra",dec_idx="dec",
                 radial_velocity_idx=None,radial_velocity_err_idx=None
                ):
        df_tot = pd.read_csv(fname,index_col=index_col)  # before cut
        df = df_tot if (cut is None) else df_tot[df_tot[cut]].reset_index(drop=True)  # after cut
        
        # prepare args of SkyCoord
        sc_kwargs = dict(ra=df[ra_idx].values*u.deg, dec=df[dec_idx].values*u.deg)
        if not radial_velocity_idx is None:
            sc_kwargs["radial_velocity"] = df[radial_velocity_idx].values*u.km/u.s
        sc = SkyCoord(**sc_kwargs)
        
        # set radial_velocity_err to sc if specified
        if not radial_velocity_err_idx is None:
            sc.radial_velocity_err = df[radial_velocity_err_idx].values*u.km/u.s
            print("radial velocity err loaded.")
            
        setattr(self,"df_"+name,df)
        setattr(self,"sc_"+name,sc)
        #setattr(self,"sep_"+name,self.dsph_prop_sc.separation(sc))

    def to_sc(self,which):
        return copy(getattr(self,"sc_"+which))
        
        