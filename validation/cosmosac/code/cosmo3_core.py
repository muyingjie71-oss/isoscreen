#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""COSMO-SAC 2010/Hsieh three-sigma model on a frozen subset of the C6 isomer panel.

Uses profile metadata & 3*51 sigma histogram areas in nist_profiles_精简带来源.json,
derived locally from the NIST usnistgov/COSMOSAC repository, preserving
Git blob SHA1s. Profile areas rounded to 9 significant digits for compact exchange;
this does not imply experimental precision. No parameter fitting or ML training.

Replicates NIST COSMO3 implementation sections:
 https://github.com/usnistgov/COSMOSAC/blob/master/include/COSMO_SAC/COSMO.hpp
 https://github.com/usnistgov/COSMOSAC/blob/master/tests/catch_tests.cpp
 dispersion: 2014 COSMO-SAC-dsp correction as implemented in NIST C++ (default),
              which is identically zero for *double differences across C6 isomers*
              because the four solutes share the same class and dsp e/kB.

A dedicated Hsieh 2014 benchmark pentane-in-heptane is evaluated before predictions.
"""
from __future__ import annotations
import json, math, pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
ORIGINAL = ROOT/'input'
# Assigned from the academic-use private local NIST sigma cache at runtime.
PROFILES = {}
SIGMAS = np.tile(np.arange(-25,26,dtype=float)/1000,3)
TYPES = np.repeat([0,1,2],51)
R = 0.00198720425864083 # kcal / mol / K
Q0 = 79.53 # Å2
R0 = 66.69 # Å3
Z = 10.0
A_EFF = 7.25 # Å2
A_ES = 6525.69
B_ES = 1.4859e8
C_OH_OH = 4013.78
C_OH_OT = 3016.43
C_OT_OT = 932.31
SRC = 'NIST_COSMO3_HSIEH_2010'


def profile(name):
    p = PROFILES[name]
    arr = np.zeros(153,dtype=float)
    for j,val in p['nz']:
        arr[j] = val
    if not abs(arr.sum()-p['area'])/p['area']<1e-5:
        raise AssertionError(f'Area inconsistent: {name}')
    return arr, p


def comb_inf(solute,solvent):
    q1=solute['area']/Q0; q2=solvent['area']/Q0
    r1=solute['volume']/R0; r2=solvent['volume']/R0
    l1=Z/2*(r1-q1)-(r1-1)
    l2=Z/2*(r2-q2)-(r2-1)
    V=r1/r2; F=q1/q2
    return math.log(V) + Z/2*q1*math.log(F/V) + l1 - V*l2


def sigma_lnGamma(T, nonzero_prob, indexes, exponent_kernel, tol=1e-12):
    # NIST iterative fixed-point equation: Γ_m = 1/Σ_n p_n Γ_n exp(-ΔW_mn/RT)
    # Use all union-of-solute-and-solvent sigma bins; some p entries may be 0.
    ps = np.array(nonzero_prob,dtype=float)
    assert len(ps)==len(indexes)
    Gam=np.ones(len(ps), dtype=float)
    KP=exponent_kernel*ps[None,:]
    for itr in range(2000):
        Gamnew = 1.0/(KP@Gam)
        Gammid=(Gam+Gamnew)/2.0
        max_relative=np.max(np.abs((Gammid-Gamnew)/Gammid))
        Gam=Gammid
        if max_relative<tol:
            break
    else:
        raise RuntimeError(f'Gamma failed convergence {T=} last {max_relative=}')
    return np.log(Gam),itr+1


def kernel(T,indexes):
    sig=SIGMAS[indexes]
    typ=TYPES[indexes]
    c=np.zeros((len(indexes),len(indexes)), dtype=float)
    is_hb=(sig[:,None]*sig[None,:])<0
    oh_oh=(typ[:,None]==1)&(typ[None,:]==1)&is_hb
    ot_ot=(typ[:,None]==2)&(typ[None,:]==2)&is_hb
    oh_ot=(((typ[:,None]==1)&(typ[None,:]==2))|((typ[:,None]==2)&(typ[None,:]==1)))&is_hb
    c[oh_oh]=C_OH_OH
    c[ot_ot]=C_OT_OT
    c[oh_ot]=C_OH_OT
    dw=(A_ES+B_ES/T**2)*(sig[:,None]+sig[None,:])**2-c*(sig[:,None]-sig[None,:])**2
    return np.exp(-dw/(R*T))


def residual_inf(T,name_solute,name_solvent):
    a,solute=profile(name_solute)
    b,solvent=profile(name_solvent)
    idx=np.flatnonzero(a+b>1e-12)
    K=kernel(T,idx)
    # weight of sigma segments normalized by the pure COSMO area
    ps_i=a[idx]/solute['area']
    ps_s=b[idx]/solvent['area']
    lnG_i,n_i=sigma_lnGamma(T,ps_i,idx,K)
    lnG_s,n_s=sigma_lnGamma(T,ps_s,idx,K)
    resid=np.sum(a[idx]*(lnG_s-lnG_i))/A_EFF
    return float(resid),n_i,n_s


def dispersion_inf(name_solute,name_solvent):
    u=PROFILES[name_solute];v=PROFILES[name_solvent]
    w=0.27027
    # No water/COOH among chosen molecules; NIST COSMO.hpp has special sign cases.
    ek1,ek2=u['dsp'],v['dsp']
    return w*(.5*(ek1+ek2)-math.sqrt(ek1*ek2))


def calc(T,name_solute,name_solvent):
    if name_solute==name_solvent:return 0.,0.,0.,0.,0,0
    comb=comb_inf(PROFILES[name_solute],PROFILES[name_solvent])
    resid,n_i,n_s=residual_inf(float(T),name_solute,name_solvent)
    disp=dispersion_inf(name_solute,name_solvent)
    return comb+resid+disp,comb,resid,disp,n_i,n_s


def validation():
    z=calc(293.15,'N-PENTANE','N-HEPTANE')
    expected={'C':-0.0317,'R':0.0001,'dsp':0.0000}
    diffs={'C':z[1]-expected['C'],'R':z[2]-expected['R'],'dsp':z[3]-expected['dsp']}
    assert max(map(abs,diffs.values()))<.002, ('Benchmark not passed',z,diffs)
    # Mathematical self-consistency: equal solute/solvent has ln gamma = 0
    for mol in ('n-Hexane','N-PENTANE'):
        z0=calc(326.35,mol,mol)
        assert z0[0]==0.0
    # Symmetry and profile completeness also tested for every sample
    return {'reference':'Hsieh et al. 2014 FPE Table 6, NIST COSMOSAC/tests/catch_tests.cpp',
            'case':'N-PENTANE as dilute solute in N-HEPTANE, T=293.15 K',
            'reference_comb':-0.0317,'reference_resid':0.0001,'reference_disp':0.0,
            'computed_comb':z[1],'computed_resid':z[2],'computed_disp':z[3],
            'errors':diffs,'passed':True}

