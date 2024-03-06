import numpy as np

def alpha_SO2(wn,temp,pressm,so2m):
    
    h   = 6.626068e-27 # g cm2 s-1
    c   = 2.99792458e10 # cm s-1
    kk  = 1.38065e-16 # erg/K
    T0  = 300.0 # K
    eta = 3./2. # symmetric-top molecule or nonlinear molecule or 1 for a linear molecule

    #convert wavenumbers to frequencies in GHz
    nu = wn*c/1.e9
    nnu = len(nu)
    nlay = len(pressm)

    #read in input file; inputs are line center frequencies (GHz), Line
    #intensity (cm-1/mol/cm^2), transition energy (cm-1)
    out = np.loadtxt('refdata/so2_linelist.dat',skiprows=1)
    nu_0 = [out[i,0] for i in range(out.shape[0])]
    S0 = [out[i,1] for i in range(out.shape[0])]
    Eline = [out[i,2] for i in range(out.shape[0])]
    nlines = len(nu_0)
    alpha = np.zeros((nnu,nlay))

    for N in np.arange(nlines):
        if nu_0[N] > np.min(nu)-1 and nu_0[N] < np.max(nu)+1:
            for i in np.arange(nlay):
                phi_so2 = phi_dop(nu,nu_0[N],temp[i], 64.)
                alpha[:,i] += 10.0**6*pressm[i]*so2m[i]/(kk*temp[i])*(T0/temp[i])**(eta+1.0)*S0[N]*np.exp(-(h*c/kk)*Eline[N]*(1/temp[i]-1/T0))*phi_so2
    return alpha

def alpha_S34O2(wn,temp,pressm,s34o2m):
    
    h   = 6.626068e-27 # g cm2 s-1
    c   = 2.99792458e10 # cm s-1
    kk  = 1.38065e-16 # erg/K
    T0  = 300.0 # K
    eta = 3./2. # symmetric-top molecule or nonlinear molecule  or 1 for a linear molecule

    #convert wavenumbers to frequencies in GHz
    nu = wn*c/1.e9
    nnu = len(nu)
    nlay = len(pressm)

    #read in input file; inputs are line center frequencies (GHz), Line
    #intensity (cm-1/mol/cm^2), transition energy (cm-1)
    out = np.loadtxt('refdata/s-34-o2_linelist.dat',skiprows=1)
    nu_0 = [out[i,0] for i in range(out.shape[0])]
    S0 = [out[i,1] for i in range(out.shape[0])]
    Eline = [out[i,2] for i in range(out.shape[0])]
    nlines = len(nu_0)
    alpha = np.zeros((nnu,nlay))

    for N in np.arange(nlines):
        if nu_0[N] > np.min(nu)-1 and nu_0[N] < np.max(nu)+1:
            for i in np.arange(nlay):
                phi_s34o2 = phi_dop(nu,nu_0[N],temp[i], 66.)
                alpha[:,i] += 10.0**6*pressm[i]*s34o2m[i]/(kk*temp[i])*(T0/temp[i])**(eta+1.0)*S0[N]*np.exp(-(h*c/kk)*Eline[N]*(1/temp[i]-1/T0))*phi_s34o2
    return alpha

def alpha_SO(wn,temp,pressm,som):
    
    h   = 6.626068e-27 # g cm2 s-1
    c   = 2.99792458e10 # cm s-1
    kk  = 1.38065e-16 # erg/K
    T0  = 300.0 # K
    eta = 1. # symmetric-top molecule or nonlinear molecule  or 1 for a linear molecule

    #convert wavenumbers to frequencies in GHz
    nu = wn*c/1.e9
    nnu = len(nu)
    nlay = len(pressm)

    #read in input file; inputs are line center frequencies (GHz), Line
    #intensity (cm-1/mol/cm^2), transition energy (cm-1)
    out = np.loadtxt('refdata/so_linelist.dat',skiprows=1)
    nu_0 = [out[i,0] for i in range(out.shape[0])]
    S0 = [out[i,1] for i in range(out.shape[0])]
    Eline = [out[i,2] for i in range(out.shape[0])]
    nlines = len(nu_0)
    alpha = np.zeros((nnu,nlay))

    for N in np.arange(nlines):
        if nu_0[N] > np.min(nu)-1 and nu_0[N] < np.max(nu)+1:
            for i in np.arange(nlay):
                phi_so = phi_dop(nu,nu_0[N],temp[i], 48.)
                alpha[:,i] += 10.0**6*pressm[i]*som[i]/(kk*temp[i])*(T0/temp[i])**(eta+1.0)*S0[N]*np.exp(-(h*c/kk)*Eline[N]*(1/temp[i]-1/T0))*phi_so
    return alpha

def alpha_S34O(wn,temp,pressm,s34om):
    
    h   = 6.626068e-27 # g cm2 s-1
    c   = 2.99792458e10 # cm s-1
    kk  = 1.38065e-16 # erg/K
    T0  = 300.0 # K
    eta = 1. # symmetric-top molecule or nonlinear molecule  or 1 for a linear molecule

    #convert wavenumbers to frequencies in GHz
    nu = wn*c/1.e9
    nnu = len(nu)
    nlay = len(pressm)

    #read in input file; inputs are line center frequencies (GHz), Line
    #intensity (cm-1/mol/cm^2), transition energy (cm-1)
    out = np.loadtxt('refdata/s-34-o_linelist.dat',skiprows=1)
    nu_0 = [out[i,0] for i in range(out.shape[0])]
    S0 = [out[i,1] for i in range(out.shape[0])]
    Eline = [out[i,2] for i in range(out.shape[0])]
    nlines = len(nu_0)
    alpha = np.zeros((nnu,nlay))

    for N in np.arange(nlines):
        if nu_0[N] > np.min(nu)-1 and nu_0[N] < np.max(nu)+1:
            for i in np.arange(nlay):
                phi_s34o = phi_dop(nu,nu_0[N],temp[i], 50.)
                alpha[:,i] += 10.0**6*pressm[i]*s34om[i]/(kk*temp[i])*(T0/temp[i])**(eta+1.0)*S0[N]*np.exp(-(h*c/kk)*Eline[N]*(1/temp[i]-1/T0))*phi_s34o
    return alpha

def alpha_NaCl(wn,temp,pressm,naclm):
    
    h   = 6.626068e-27 # g cm2 s-1
    c   = 2.99792458e10 # cm s-1
    kk  = 1.38065e-16 # erg/K
    T0  = 300.0 # K
    eta = 1. # symmetric-top molecule or nonlinear molecule  or 1 for a linear molecule

    #convert wavenumbers to frequencies in GHz
    nu = wn*c/1.e9
    nnu = len(nu)
    nlay = len(pressm)

    #read in input file; inputs are line center frequencies (GHz), Line
    #intensity (cm-1/mol/cm^2), transition energy (cm-1)
    out = np.loadtxt('refdata/nacl_linelist.dat',skiprows=1)
    nu_0 = [out[i,0] for i in range(out.shape[0])]
    S0 = [out[i,1] for i in range(out.shape[0])] 
    Eline = [out[i,2] for i in range(out.shape[0])]
    nlines = len(nu_0)
    alpha = np.zeros((nnu,nlay))

    for N in np.arange(nlines):
        if nu_0[N] > np.min(nu)-1 and nu_0[N] < np.max(nu)+1:
            for i in np.arange(nlay):
                phi_nacl = phi_dop(nu,nu_0[N],temp[i], 58.)
                alpha[:,i] += 10.0**6*pressm[i]*naclm[i]/(kk*temp[i])*(T0/temp[i])**(eta+1.0)*S0[N]*np.exp(-(h*c/kk)*Eline[N]*(1/temp[i]-1/T0))*phi_nacl
    return alpha

def alpha_Na37Cl(wn,temp,pressm,na37clm):
    
    h   = 6.626068e-27 # g cm2 s-1
    c   = 2.99792458e10 # cm s-1
    kk  = 1.38065e-16 # erg/K
    T0  = 300.0 # K
    eta = 1. # symmetric-top molecule or nonlinear molecule  or 1 for a linear molecule

    #convert wavenumbers to frequencies in GHz
    nu = wn*c/1.e9
    nnu = len(nu)
    nlay = len(pressm)

    #read in input file; inputs are line center frequencies (GHz), Line
    #intensity (cm-1/mol/cm^2), transition energy (cm-1)
    out = np.loadtxt('refdata/nacl-37_linelist.dat',skiprows=1)
    nu_0 = [out[i,0] for i in range(out.shape[0])]
    S0 = [out[i,1] for i in range(out.shape[0])] 
    Eline = [out[i,2] for i in range(out.shape[0])]
    nlines = len(nu_0)
    alpha = np.zeros((nnu,nlay))

    for N in np.arange(nlines):
        if nu_0[N] > np.min(nu)-1 and nu_0[N] < np.max(nu)+1:
            for i in np.arange(nlay):
                phi_na37cl = phi_dop(nu,nu_0[N],temp[i], 60.)
                alpha[:,i] += 10.0**6*pressm[i]*na37clm[i]/(kk*temp[i])*(T0/temp[i])**(eta+1.0)*S0[N]*np.exp(-(h*c/kk)*Eline[N]*(1/temp[i]-1/T0))*phi_na37cl
    return alpha

def alpha_KCl(wn,temp,pressm,kclm):
    
    h   = 6.626068e-27 # g cm2 s-1
    c   = 2.99792458e10 # cm s-1
    kk  = 1.38065e-16 # erg/K
    T0  = 300.0 # K
    eta = 1. # symmetric-top molecule or nonlinear molecule  or 1 for a linear molecule

    #convert wavenumbers to frequencies in GHz
    nu = wn*c/1.e9
    nnu = len(nu)
    nlay = len(pressm)

    #read in input file; inputs are line center frequencies (GHz), Line
    #intensity (cm-1/mol/cm^2), transition energy (cm-1)
    out = np.loadtxt('refdata/kcl_linelist.dat',skiprows=1)
    nu_0 = [out[i,0] for i in range(out.shape[0])]
    S0 = [out[i,1] for i in range(out.shape[0])] 
    Eline = [out[i,2] for i in range(out.shape[0])]
    nlines = len(nu_0)
    alpha = np.zeros((nnu,nlay))

    for N in np.arange(nlines):
        if nu_0[N] > np.min(nu)-1 and nu_0[N] < np.max(nu)+1:
            for i in np.arange(nlay):
                phi_kcl = phi_dop(nu,nu_0[N],temp[i], 74.)
                alpha[:,i] += 10.0**6*pressm[i]*kclm[i]/(kk*temp[i])*(T0/temp[i])**(eta+1.0)*S0[N]*np.exp(-(h*c/kk)*Eline[N]*(1/temp[i]-1/T0))*phi_kcl
    return alpha

def alpha_K37Cl(wn,temp,pressm,k37clm):
    
    h   = 6.626068e-27 # g cm2 s-1
    c   = 2.99792458e10 # cm s-1
    kk  = 1.38065e-16 # erg/K
    T0  = 300.0 # K
    eta = 1. # symmetric-top molecule or nonlinear molecule  or 1 for a linear molecule

    #convert wavenumbers to frequencies in GHz
    nu = wn*c/1.e9
    nnu = len(nu)
    nlay = len(pressm)

    #read in input file; inputs are line center frequencies (GHz), Line
    #intensity (cm-1/mol/cm^2), transition energy (cm-1)
    out = np.loadtxt('refdata/kcl-37_linelist.dat',skiprows=1)
    nu_0 = [out[i,0] for i in range(out.shape[0])]
    S0 = [out[i,1] for i in range(out.shape[0])] 
    Eline = [out[i,2] for i in range(out.shape[0])]
    nlines = len(nu_0)
    alpha = np.zeros((nnu,nlay))

    for N in np.arange(nlines):
        if nu_0[N] > np.min(nu)-1 and nu_0[N] < np.max(nu)+1:
            for i in np.arange(nlay):
                phi_k37cl = phi_dop(nu,nu_0[N],temp[i], 60.)
                alpha[:,i] += 10.0**6*pressm[i]*k37clm[i]/(kk*temp[i])*(T0/temp[i])**(eta+1.0)*S0[N]*np.exp(-(h*c/kk)*Eline[N]*(1/temp[i]-1/T0))*phi_k37cl
    return alpha

# Doppler broaden line profile
def phi_dop(nu,nu0,T,mu):
    c    = 2.99792458e10 # cm s-1
    kk   = 1.38065e-16 # [cm2 g s-2 K-1]
    m    = mu*1.67e-24 # mass of an atom
    delnuD = nu0/c*np.sqrt((2*kk*T)/m ) # Dopplar width
    phi= 1/(delnuD*np.sqrt(np.pi))*np.exp(-(nu-nu0)**2/(delnuD)**2)
    phicm1=phi/(1.e9/c)
    return phicm1
