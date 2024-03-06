import numpy as np
from astropy import constants as const

def mm_Io_atmosphere(so2_col = 1.e16,
                         so_col  = 0.05*1.e16,
                         s34mixing=0.045,
                         nacl_col =  0.001*1e16,
                         kcl_col = 0.0005*1e16,
                         cl37mixing = 0.33,
                         nlev=200,
                         Tgas=200):
    # -------------------------------------------------------------------------
    #   Constants
    # -------------------------------------------------------------------------
    gg  = 181 #[cm s-2]
    kk  = const.k_B.cgs.value # [cm2 g s-2 K-1]
    mp  = const.m_p.cgs.value
    # Assume a constant SO/SO2 ratio
    na37cl_col = cl37mixing*nacl_col
    k37cl_col = cl37mixing*kcl_col
    s34o2_col = s34mixing*so2_col
    s34o_col = s34mixing*so_col
    #
    tot_col = so2_col + so_col + nacl_col + kcl_col + s34o2_col + s34o_col + na37cl_col + k37cl_col
    #
    so2m = so2_col / tot_col
    s34o2m = s34o2_col/ tot_col
    som  = so_col / tot_col
    s34om  = s34o_col / tot_col
    naclm = nacl_col / tot_col
    na37clm = na37cl_col / tot_col
    kclm = kcl_col / tot_col
    k37clm = k37cl_col / tot_col
    mu   = so2m*64 + s34o2m*66 + som*48 + s34om*50 + naclm*58 + na37clm*60 + kclm*74 + k37clm*76

    # Pressure is force per area = M/A = C*(mu*mp)*g where C = #/A and
    # mu*mp = avg weight of a molecule
    Psurf = mu*mp*gg*(tot_col)*1e-6 # bar
    # Go up 10 scale heights
    Pmin  = Psurf/np.exp(10.)
    pressure = 10**(np.arange(nlev)*(np.log10(Psurf)-np.log10(Pmin))/(nlev-1)+np.log10(Pmin))
    nlay   = nlev-1
    pressb = pressure[1:nlay+1]      # top of layers [bar] 
    presst = pressure[0:nlay]    # bottom of layers [bar]
    pressm = (presst+pressb)/2.    # middle of layers [bar]
    # Mole Fractons: ;same for each layer, but make dimensions equal to nlay
    so2m = np.zeros((nlay))+so2m
    s34o2m = np.zeros((nlay))+s34o2m
    som  = np.zeros((nlay))+som
    s34om  = np.zeros((nlay))+s34om
    naclm  = np.zeros((nlay))+naclm
    na37clm  = np.zeros((nlay))+na37clm
    kclm  = np.zeros((nlay))+kclm
    k37clm  = np.zeros((nlay))+k37clm

    # -------------------------------------------------------------------------
    #   T-P profile
    # -------------------------------------------------------------------------
    temp = np.zeros((nlay)) + Tgas
    # profile of S02 comes from the hydrostatic law with T=Tgas at all heights
    N   = pressm*1.e6/(kk*temp) #[cm-3]
    NSO2 = N*so2m
    NNSO  = N*som
    hh  = kk*temp/(gg*mu*1.67e-24)
    dz  = hh*np.log(pressb/presst)
    
    atm = {'temp': temp,
           'pressm':pressm,
           'pressb':pressb,
           'presst':presst,
           'mu': mu,
           'dz':dz,
           'hh':hh,
           'som':som,
           's34om':s34om,
           'so2m':so2m,
           's34o2m':s34o2m,
           'naclm':naclm,
           'na37clm':na37clm,
           'kclm':kclm,
          'k37clm': k37clm}

    return atm
