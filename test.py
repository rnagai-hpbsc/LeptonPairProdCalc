import numpy as np
import KPmodel_libs

import matplotlib.pyplot as plt 

fig = plt.figure()
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True

logE_leps = np.linspace(5,12,100)
E_leps = np.array([10**logE for logE in logE_leps]) 

calc = KPmodel_libs.XsecCalculator(m_lep=200)

E_lep = 100e3
ymin, ymax = calc.lim_y(E_lep)
print(ymin, ymax)

log_ys = np.linspace(np.log10(ymin),np.log10(ymax),500)
ys = np.array([10**logy for logy in log_ys])
ysE = ys * E_lep
secterm = 4*calc.m_ele/ysE 

print(ys[0], log_ys[0])
print(f'lim_rho    : {calc.lim_rho(ys[0],E_lep)}')
print(f'lim_rho_Ey : {calc.lim_rho_EyParams(ys[0]*E_lep,E_lep)}')
print(f'lim_y      : {calc.lim_y(E_lep)}')
print(f'lim_Ey     : {calc.lim_Ey(E_lep)}')

Xis = np.array([calc.getXi(0.1, y) for y in ys])

dsdy_quad = np.array([calc.getDSigmaDy(y,E_lep,'quad') for y in ys])
dsdy_romb = np.array([calc.getDSigmaDy(y,E_lep,'romberg') for y in ys])

plt.plot(ys, dsdy_quad, label='Quad')
plt.plot(ys, dsdy_romb, label='Romberg')
plt.axvline(ymin, ls=':')
plt.axvline(ymax, ls=':')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Fraction of Energy loss $y$')
plt.ylabel(r'$\frac{d\sigma}{dy}\,(y)$')
plt.legend()

fig2 = plt.figure()
s_quad = np.array([calc.getSigma(E,'quad') for E in E_leps])
s_romb = np.array([calc.getSigma(E,'romberg') for E in E_leps])
plt.plot(E_leps, s_quad, label='Quad')
plt.plot(E_leps, s_romb, label='Romberg')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Energy [GeV]')
plt.ylabel(r'$\sigma_\mathrm{std~rock}$ [cm$^2$]')
plt.legend()

plt.show()
