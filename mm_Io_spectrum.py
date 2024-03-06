from mm_Io_atmosphere import mm_Io_atmosphere
from mm_Io_opacity import mm_Io_opacity
import numpy as np
from astropy import constants as const
from astropy import units as u
from scipy import ndimage

import matplotlib.pyplot as plt
import copy


def mm_Io_spectrum(band = 6, nuGHz_min = None, nuGHz_max = None,
               Tsurf       = 115.0, 
               Tgas        = 250.0,
               emiss       = 0.8,
               so2_col     = 1.e16,
               s34mixing   = 0.045, 
               nacl_col =  0.001*1e16,
               kcl_col = 0.0005*1e16,
               cl37mixing = 0.33,
               so_col      = 0.05*1.e16,
               nlev        = 200,
               spec_res    = 5.e6, # Hz
               tointerp    = True,
               mu0         = 0.999):
    # -----------------------------------------------------------------------------
    #    Constants
    # -----------------------------------------------------------------------------
    h = const.h.cgs.value
    c = const.c.cgs.value #cm s-1
    kk = const.k_B.cgs.value # [cm2 g s-2 K-1]

    # -------------------------------------------------------------------------
    #  set up atmosphere (T-P profile and abundance profiles)
    # -------------------------------------------------------------------------
    atm = mm_Io_atmosphere(so2_col = so2_col,so_col = so_col, s34mixing = s34mixing, nacl_col=nacl_col,kcl_col=kcl_col,cl37mixing=cl37mixing,nlev = nlev,Tgas = Tgas)
    nlay = len(atm['pressm'])
    
    # -------------------------------------------------------------------------
    #  define wavelengths, resolution
    # -------------------------------------------------------------------------
    inu = np.ceil(spec_res/4.)
    if band == 6:
            nuGHz_min = 211.
            nuGHz_max = 275.
    elif band == 7:
            nuGHz_min = 275.
            nuGHz_max = 373.
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

    # -------------------------------------------------------------------------
    #  integrate for angle mu
    # -------------------------------------------------------------------------
    Bnu  = np.zeros((nwn,nlay))
    Bnu[:,0] = 2*h*nu**3/c**2*1/(np.exp(h*nu/(kk*atm['temp'][0]))-1)
    tau_above = np.zeros(tau_gas.shape) # sum tau from top down
    for j in np.arange(nlay):
        Bnu      [:,j] = 2*h*nu**3/c**2*1/(np.exp(h*nu/(kk*atm['temp'][j]))-1)
        tau_above[:,j] = np.sum(tau_gas[:,0:j],axis=1)

    Bdisk = Bnu[:,0]*np.exp(-tau_above[:,0]/mu0)*tau_gas[:,0]/mu0
    for j in np.arange(nlay):
        Bdisk += Bnu[:,j]*np.exp(-tau_above[:,j]/mu0)*tau_gas[:,j]/mu0

    Bdisk+=2*h*nu**3/c**2*1/(np.exp(h*nu/(kk*TBsurf))-1) * np.exp(-tau_above[:,nlay-1]/mu0)
    Tdisk = 1/(kk/(h*nu)*np.log((2*h*nu**3)/(c**2*Bdisk)+1))

    z    = spec_res/inu          # standard deviation of gaussian
    buff = np.round(z)*2
    
# Smooth to spec_res. WARNING!! This could be bad near the edges of
# the frequency window.
    smnu = (np.arange(nnu,dtype=int))[int(buff):int(nnu-buff)]
    Tdisk[smnu]  = (ndimage.filters.uniform_filter(Tdisk,z))[smnu]
    Bdisk[smnu] = (ndimage.filters.uniform_filter(Bdisk,z))[smnu] 
    return atm['temp'],atm['pressm'],nu,Tdisk,Bdisk 


    
