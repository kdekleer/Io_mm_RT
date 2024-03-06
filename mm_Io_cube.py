from mm_Io_atmosphere import mm_Io_atmosphere
from mm_Io_opacity import mm_Io_opacity
from mm_deprob import deprob
import numpy as np
from scipy import ndimage
from astropy import constants as const
from astropy import units as u

import matplotlib.pyplot as plt
import copy
import pdb

def mm_Io_cube(band = 6, nuGHz_min = None, nuGHz_max = None,
               Tsurf       = 115.0, # inputs to atmosphere
               Tgas        = 250.0,
               emiss       = 0.8,
               so2_col     = 5.e16,
               s34mixing   = 0.045, 
               nacl_col =  0.001*5e16,
               kcl_col = 0.0005*5e16,
               cl37mixing = 0.33,
               so_col      = 0.05*5.e16,
               nlev        = 200,
               cell        = 0.1,
               Vrot        = -75.0*1.e2,
               Vdop        = 0.0,
               a_au        = 5.875,
               npang       = 355.0,
               solon       = 260.55,
               solat       = 2.59,
               spec_res    = 5.e6,
               tointerp    = True):

    # -----------------------------------------------------------------------------
    #    Constants and Values
    # -----------------------------------------------------------------------------
    h = const.h.cgs.value
    c = const.c.cgs.value #cm s-1
    kk = const.k_B.cgs.value # [cm2 g s-2 K-1]
    kmperau=const.au.to('km').value

    erad_km  = 1829.4  # jpl horizons
    prad_km  = 1815.7
    # -------------------------------------------------------------------------
    #  set up atmosphere (T-P profile and abundance profiles)
    # -------------------------------------------------------------------------
    atm = mm_Io_atmosphere(so2_col = so2_col,
                               so_col = so_col,
                               s34mixing=s34mixing,
                               nacl_col = nacl_col,
                               kcl_col = kcl_col,
                               cl37mixing = cl37mixing,
                               nlev = nlev,Tgas = Tgas)
    nlay = len(atm['pressm'])
    
    # -------------------------------------------------------------------------
    #  define wavelengths, resolution
    # -------------------------------------------------------------------------
    inu = np.ceil(spec_res/4.)#; perform 1 integrations for every output resolution element
    if band == 6:
            nuGHz_min = 211.
            nuGHz_max = 275.
    elif band == 7:
            nuGHz_min = 275.
            nuGHz_max = 373.
    # over-ride for now:
    else:
        if nuGHz_min is None:
            nuGHz_min = 316.0
        if nuGHz_max is None:
            nuGHz_max = 321.0
 
    nu_min = nuGHz_min*1.e9
    nu_max = nuGHz_max*1.e9
    nnu = np.ceil((nu_max-nu_min)/inu)+2
    nu  = np.arange(nnu)*inu+nu_min
    wn = nu/c
    nwn = len(wn)

    # -------------------------------------------------------------------------
    #  get optical depth at each wavelength
    # -------------------------------------------------------------------------

    tau_gas = mm_Io_opacity(atm,wn, tointerp=True)

    # --- Surface Layer
    nlay = len(atm['temp'])
    TBsurf = emiss*Tsurf
#   --------------------------------------------------------------------------
#   make a cube of mu and x
#   --------------------------------------------------------------------------
    erad_arc = (erad_km)/(a_au*kmperau)*206265
    prad_arc = (prad_km)/(a_au*kmperau)*206265
    erad_pix = erad_arc/cell
    prad_pix = prad_arc/cell
    
    nx = int(np.max([np.ceil(erad_pix),np.ceil(prad_pix)])*2)
    # need nx to be odd
    if nx % 2 == 0:
        nx+=3
    else:
        nx+=2
    ny = int(nx)
    center=[nx-(nx-1)/2-1, ny-(ny-1)/2-1]
    ob_lat,ob_lon, ob_emi,ob_pc=deprob(nx=nx,ny=ny, center=center, erad_pix=erad_pix,prad_pix=prad_pix, so_lon=solon, so_lat=solat,np_ang=npang)
    mu=np.cos(ob_emi.radian)

#   --------------------------------------------------------------------------
#   Calculate Doppler Shifts
#   --------------------------------------------------------------------------
    V   = Vrot*np.cos(ob_lat)*np.sin(ob_lon-(solon*u.deg).to(u.radian))*np.cos((solat*u.deg).to(u.radian))+Vdop
      
# -------------------------------------------------------------------------
#  Calculate B for angle mu [and velocity V]
# -------------------------------------------------------------------------

    z    = spec_res/inu          # standard deviation of gaussian
    
    buff = np.round(z)*2
    smnu = (np.arange(nnu,dtype=int))[int(buff):int(nnu-buff)]
    Bnu  = np.zeros((nwn,nlay))
    Bnu[:,0] = 2*h*nu**3/c**2*1/(np.exp(h*nu/(kk*atm['temp'][0]))-1)
    tau_above = np.zeros(tau_gas.shape) # sum tau from top down
    for j in np.arange(nlay):
        Bnu      [:,j] = 2*h*nu**3/c**2*1/(np.exp(h*nu/(kk*atm['temp'][j]))-1)
        tau_above[:,j] = np.sum(tau_gas[:,0:j],axis=1)

    Tcube=np.zeros((nx,ny,nwn))
    Bcube=np.zeros((nx,ny,nwn))
    
    #integrate
    for j in np.arange(ny/2,dtype=int):
        for i in np.arange(nx/2, dtype=int): #run on only 1 quadrant, mu will be the same in all 4
            if ~np.isnan(mu[i,j]):
                loc=[[int(i),int(j)],[int(i),int(ny-j-1)],[int(nx-i-1),int(j)],[int(nx-i-1),int(ny-j-1)]]
                mmu=np.mean([mu[l[0],l[1]] for l in loc])
                B = Bnu[:,0]*np.exp(-tau_above[:,0]/mmu)*tau_gas[:,0]/mmu
                #
                # --- to print out info for paper; comment out if running mcmc or fits --- #
                #print('tau:',mmu,np.max(tau_above[:,:])/mmu)
                # ---
                for n in range(1,nlay):
                    B+=Bnu[:,n]*np.exp(-tau_above[:,n]/mmu)*tau_gas[:,n]/mmu
                B+=2*h*nu**3/c**2*1/(np.exp(h*nu/(kk*TBsurf))-1) * np.exp(-tau_above[:,nlay-1]/mmu)
                for a in range(len(loc)): # different values of the doppler shift
                    delnu=V[loc[a][0],loc[a][1]]/c*nu
                    shiftnu = nu - delnu 
                    Bdop=np.interp(nu, shiftnu, B)
                    Tdop = 1/(kk/(h*nu)*np.log((2*h*nu**3)/(c**2*Bdop)+1))
                    Tdop[smnu] = (ndimage.filters.uniform_filter(Tdop,z))[smnu] 
                    Bdop[smnu] = (ndimage.filters.uniform_filter(Bdop,z))[smnu] 
                    Tcube[loc[a][0],loc[a][1],:] = Tdop
                    Bcube[loc[a][0],loc[a][1],:] = Bdop
    

    return atm['temp'],atm['pressm'],nu,Tcube,Bcube 


    
