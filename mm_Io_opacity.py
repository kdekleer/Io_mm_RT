from astropy.io import fits
import numpy as np
import alpha

def mm_Io_opacity(atm,wn,tointerp=True,usealkalis=True):
    #
    alphaSO2_file = 'init/SO2_out.fits'
    alphaS34O2_file = 'init/S34O2_out.fits'
    alphaSO_file = 'init/SO_out.fits'
    alphaS34O_file = 'init/S34O_out.fits'
    alphaNaCl_file = 'init/NaCl_out.fits'
    alphaNa37Cl_file = 'init/Na37Cl_out.fits'
    alphaKCl_file = 'init/KCl_out.fits'
    alphaK37Cl_file = 'init/K37Cl_out.fits'
    #-----------------------------------------------------------------------------
    #   SO2 Absorption
    # -----------------------------------------------------------------------------
    if tointerp==True:
        alphaSO2 = alpha.alpha_SO2(wn,atm['temp'],atm['pressm'],atm['so2m'])
    else:
        alphaSO2 = fits.getdata(open(alphaSO2_file,'rb'))
    #-----------------------------------------------------------------------------
    #   34SO2 Absorption
    # -----------------------------------------------------------------------------
    if tointerp==True:
        alphaS34O2 = alpha.alpha_S34O2(wn,atm['temp'],atm['pressm'],atm['s34o2m'])
    else:
        alphaS34O2 = fits.getdata(open(alphaS34O2_file,'rb'))
    # -----------------------------------------------------------------------------
    #   SO Absorption
    # -----------------------------------------------------------------------------
    if tointerp==True:
        alphaSO = alpha.alpha_SO(wn,atm['temp'],atm['pressm'],atm['som'])
    else:
        alphaSO = fits.getdata(open(alphaSO_file,'rb'))
    # -----------------------------------------------------------------------------
    #   S34O Absorption
    # -----------------------------------------------------------------------------
    if tointerp==True:
        alphaS34O = alpha.alpha_S34O(wn,atm['temp'],atm['pressm'],atm['s34om'])
    else:
        alphaS34O = fits.getdata(open(alphaS34O_file,'rb'))
    #
    if usealkalis==True:
        # -----------------------------------------------------------------------------
        #   NaCl Absorption
        # -----------------------------------------------------------------------------
        if tointerp==True:
            alphaNaCl = alpha.alpha_NaCl(wn,atm['temp'],atm['pressm'],atm['naclm'])
        else:
            alphaNaCl = fits.getdata(open(alphaNaCl_file,'rb'))
        # -----------------------------------------------------------------------------
        #   Na37Cl Absorption
        # -----------------------------------------------------------------------------
        if tointerp==True:
            alphaNa37Cl = alpha.alpha_Na37Cl(wn,atm['temp'],atm['pressm'],atm['na37clm'])
        else:
            alphaNa37Cl = fits.getdata(open(alphaNa37Cl_file,'rb'))
        # -----------------------------------------------------------------------------
        #   KCl Absorption
        # -----------------------------------------------------------------------------
        if tointerp==True:
            alphaKCl = alpha.alpha_KCl(wn,atm['temp'],atm['pressm'],atm['kclm'])
        else:
            alphaKCl = fits.getdata(open(alphaKCl_file,'rb'))
        # -----------------------------------------------------------------------------
        #   K37Cl Absorption
        # -----------------------------------------------------------------------------
        if tointerp==True:
            alphaK37Cl = alpha.alpha_K37Cl(wn,atm['temp'],atm['pressm'],atm['k37clm'])
        else:
            alphaK37Cl = fits.getdata(open(alphaK37Cl_file,'rb'))
    
    # -----------------------------------------------------------------------------
    #   Total gas optical depth
    # -----------------------------------------------------------------------------
    nlay = len(atm['pressm'])
    dz = atm['dz']
    tau_gas = np.zeros(np.shape(alphaSO2))
    
    tau_so2 = np.zeros(np.shape(alphaSO2))
    for i in np.arange(nlay):
        tau_so2[:,i]  = alphaSO2[:,i]*dz[i]
        tau_gas [:,i]  += alphaSO2[:,i]*dz[i]

    tau_s34o2 = np.zeros(np.shape(alphaS34O2))
    for i in np.arange(nlay):
        tau_s34o2[:,i]  = alphaS34O2[:,i]*dz[i]
        tau_gas [:,i]  += alphaS34O2[:,i]*dz[i]

    tau_so = np.zeros(np.shape(alphaSO))
    for i in np.arange(nlay):
        tau_so[:,i]  = alphaSO[:,i]*dz[i]
        tau_gas [:,i]  += alphaSO[:,i]*dz[i]

    tau_s34o = np.zeros(np.shape(alphaS34O))
    for i in np.arange(nlay):
        tau_s34o[:,i]  = alphaS34O[:,i]*dz[i]
        tau_gas [:,i]  += alphaS34O[:,i]*dz[i]

    if usealkalis==True:

        tau_nacl = np.zeros(np.shape(alphaNaCl))
        for i in np.arange(nlay):
            tau_nacl[:,i]  = alphaNaCl[:,i]*dz[i]
            tau_gas [:,i]  += alphaNaCl[:,i]*dz[i]

        tau_na37cl = np.zeros(np.shape(alphaNa37Cl))
        for i in np.arange(nlay):
            tau_na37cl[:,i]  = alphaNa37Cl[:,i]*dz[i]
            tau_gas [:,i]  += alphaNa37Cl[:,i]*dz[i]

        tau_kcl = np.zeros(np.shape(alphaKCl))
        for i in np.arange(nlay):
            tau_kcl[:,i]  = alphaKCl[:,i]*dz[i]
            tau_gas [:,i]  += alphaKCl[:,i]*dz[i]

        tau_k37cl = np.zeros(np.shape(alphaK37Cl))
        for i in np.arange(nlay):
            tau_k37cl[:,i]  = alphaK37Cl[:,i]*dz[i]
            tau_gas [:,i]  += alphaK37Cl[:,i]*dz[i]

    return tau_gas
    
