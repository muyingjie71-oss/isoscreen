"""COSMO-SAC 2010 with three sigma profiles and the 2014 dispersion correction.

Follows the implementation of NIST:
  https://github.com/usnistgov/COSMOSAC/blob/master/include/COSMO_SAC/COSMO.hpp
  https://github.com/usnistgov/COSMOSAC/blob/master/tests/catch_tests.cpp (reference values)

PROFILES maps a standard InChIKey to its sigma profile as read by run_cosmosac.parse_sigma_file:
area (A^2), volume (A^3), dispersion parameter dsp (e/k_B in K), dispersion class flag and the
non-zero bins nz of the three 51-bin profiles (non-hydrogen-bonding, OH, other hydrogen-bonding).
"""
import math
import numpy as np

PROFILES = {}
SIGMAS = np.tile(np.arange(-25, 26, dtype=float) / 1000, 3)
TYPES = np.repeat([0, 1, 2], 51)
R = 0.00198720425864083  # kcal / mol / K
Q0 = 79.53  # A^2
R0 = 66.69  # A^3
Z = 10.0
A_EFF = 7.25  # A^2
A_ES = 6525.69
B_ES = 1.4859e8
C_OH_OH = 4013.78
C_OH_OT = 3016.43
C_OT_OT = 932.31


def profile(key):
    p = PROFILES[key]
    arr = np.zeros(153, dtype=float)
    for j, val in p['nz']:
        arr[j] = val
    if not abs(arr.sum() - p['area']) / p['area'] < 1e-5:
        raise AssertionError(f'Area inconsistent: {key}')
    return arr, p


def comb_inf(solute, solvent):
    """Staverman-Guggenheim combinatorial term at infinite dilution."""
    q1 = solute['area'] / Q0; q2 = solvent['area'] / Q0
    r1 = solute['volume'] / R0; r2 = solvent['volume'] / R0
    l1 = Z / 2 * (r1 - q1) - (r1 - 1)
    l2 = Z / 2 * (r2 - q2) - (r2 - 1)
    V = r1 / r2; F = q1 / q2
    return math.log(V) + Z / 2 * q1 * math.log(F / V) + l1 - V * l2


def sigma_lnGamma(T, prob, indexes, exponent_kernel, tol=1e-12):
    """Segment activity coefficients from the fixed-point equation
    Gamma_m = 1 / sum_n p_n Gamma_n exp(-dW_mn / RT), over the sigma bins of solute and solvent."""
    ps = np.array(prob, dtype=float)
    assert len(ps) == len(indexes)
    Gam = np.ones(len(ps), dtype=float)
    KP = exponent_kernel * ps[None, :]
    for itr in range(2000):
        Gamnew = 1.0 / (KP @ Gam)
        Gammid = (Gam + Gamnew) / 2.0
        max_relative = np.max(np.abs((Gammid - Gamnew) / Gammid))
        Gam = Gammid
        if max_relative < tol:
            break
    else:
        raise RuntimeError(f'Segment activity coefficients did not converge at T = {T} K')
    return np.log(Gam), itr + 1


def kernel(T, indexes):
    sig = SIGMAS[indexes]
    typ = TYPES[indexes]
    c = np.zeros((len(indexes), len(indexes)), dtype=float)
    is_hb = (sig[:, None] * sig[None, :]) < 0
    oh_oh = (typ[:, None] == 1) & (typ[None, :] == 1) & is_hb
    ot_ot = (typ[:, None] == 2) & (typ[None, :] == 2) & is_hb
    oh_ot = (((typ[:, None] == 1) & (typ[None, :] == 2)) | ((typ[:, None] == 2) & (typ[None, :] == 1))) & is_hb
    c[oh_oh] = C_OH_OH
    c[ot_ot] = C_OT_OT
    c[oh_ot] = C_OH_OT
    dw = (A_ES + B_ES / T ** 2) * (sig[:, None] + sig[None, :]) ** 2 - c * (sig[:, None] - sig[None, :]) ** 2
    return np.exp(-dw / (R * T))


def residual_inf(T, solute_key, solvent_key):
    """Residual term at infinite dilution."""
    a, solute = profile(solute_key)
    b, solvent = profile(solvent_key)
    idx = np.flatnonzero(a + b > 1e-12)
    K = kernel(T, idx)
    ps_i = a[idx] / solute['area']   # segment fractions, normalized by the total area
    ps_s = b[idx] / solvent['area']
    lnG_i, n_i = sigma_lnGamma(T, ps_i, idx, K)
    lnG_s, n_s = sigma_lnGamma(T, ps_s, idx, K)
    resid = np.sum(a[idx] * (lnG_s - lnG_i)) / A_EFF
    return float(resid), n_i, n_s


def dispersion_inf(solute_key, solvent_key):
    """2014 dispersion correction at infinite dilution, with the sign rules of COSMO.hpp for water and COOH."""
    u = PROFILES[solute_key]; v = PROFILES[solvent_key]
    c0, c1 = u['flag'], v['flag']; w = 0.27027
    if ('H2O' in (c0, c1) and ('HB-ACCEPTOR' in (c0, c1) or 'COOH' in (c0, c1))) or \
       ('COOH' in (c0, c1) and ('NHB' in (c0, c1) or 'HB-DONOR-ACCEPTOR' in (c0, c1))):
        w = -w
    es = u['dsp']; ev = v['dsp']
    return w * (0.5 * (es + ev) - math.sqrt(max(0, es * ev)))


def ln_gamma_inf(T, solute_key, solvent_key):
    """ln gamma_inf of the solute in the solvent and its combinatorial, residual and dispersion parts."""
    if solute_key == solvent_key:
        return 0.0, 0.0, 0.0, 0.0
    C = comb_inf(PROFILES[solute_key], PROFILES[solvent_key])
    res, _, _ = residual_inf(T, solute_key, solvent_key)
    dsp = dispersion_inf(solute_key, solvent_key)
    z = C + res + dsp
    if not np.isfinite(z):
        raise ValueError('Nonfinite ln gamma')
    return float(z), float(C), float(res), float(dsp)
