import numpy as np
import KPmodel_libs

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt 
from tqdm import tqdm
import sys

fig = plt.figure()
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True

yacc = 30
E_leps = np.array([10**logE for logE in np.linspace(5,8,yacc)])

kpmodel = KPmodel_libs.XsecCalculator(m_lep=150) # stau mass of 150 GeV

y = 0.9958649775595848
kpmodel.setRombergTol(1e-8)
kpmodel.setRombergDivmax(50)
kpmodel.setRombergDebug()
print(kpmodel.getRombergTol())

sigmas = np.array([kpmodel.getSigma(E_lep) for E_lep in E_leps])
sigmas_quad = np.array([kpmodel.getSigma(E_lep,method='quad') for E_lep in E_leps])
plt.plot(E_leps,sigmas,label='Python Romberg')
plt.plot(E_leps,sigmas_quad,label='Python Quad')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Energy [GeV]')
plt.ylabel('Total Xsec')
plt.tight_layout()
plt.savefig('plot/comp_sigma.png',bbox_inches='tight',format='png',dpi=300)
#plt.show()

sys.exit()

dsigmadys = np.array([kpmodel.getDSigmaDy_OLD(y,E_lep) for E_lep in E_leps])
print(dsigmadys)
dsigmadys_quad = np.array([kpmodel.getDSigmaDy_OLD(y,E_lep,method='quad') for E_lep in E_leps])
print(dsigmadys_quad)
dsigmadys_factor = np.array([kpmodel.getDSigmaDy(y,E_lep) for E_lep in E_leps])

Niter = 500
intDsDyDrhos = []
for E_lep in E_leps: 
    rhomax = kpmodel.lim_rho(y, E_lep)
    tmprhos = np.linspace(-rhomax, rhomax, Niter)
    drhos = tmprhos[1] - tmprhos[0]
    tmpDsDyDrhos = np.array([kpmodel.getAsymTerm(rho,y,E_lep) for rho in tmprhos])
    intDsDyDrhos.append(np.sum(tmpDsDyDrhos)*drhos)


#print(sigmas)
plt.plot(E_leps,dsigmadys,label='Romberg integral of $\mathrm{d}\sigma/\mathrm{d}y\mathrm{d}\\rho$')
plt.plot(E_leps,dsigmadys_quad,label='Quad integral of $\mathrm{d}\sigma/\mathrm{d}y\mathrm{d}\\rho$')
plt.plot(E_leps,dsigmadys_factor,label='Romberg integral of AsymTerm $\\times$ Factor')
#plt.plot(E_leps,intDsDyDrhos)
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.xlabel('Energy [GeV]')
plt.ylabel('Differential Xsec')
plt.tight_layout()
plt.show()


sys.exit()

yacc = 1000
logE_leps = np.linspace(5,12,yacc)
E_leps = np.array([10**logE for logE in logE_leps]) 

m_lep = 150
calc  = KPmodel_libs.XsecCalculator(m_lep=m_lep)

E_lep = 1e11
ymin, ymax = calc.lim_y(E_lep)
print(ymin, ymax)

log_ys = np.linspace(np.log10(ymin),np.log10(ymax),yacc)
ys = np.array([10**logy for logy in log_ys])
ys[0] = ymin 
ys[len(ys)-1] = ymax
ysE = ys * E_lep
secterm = 4*calc.m_ele/ysE 

print(ys[0], log_ys[0])
print(f'lim_rho    : {calc.lim_rho(ys[0],E_lep)}')
print(f'lim_y      : {calc.lim_y(E_lep)}')
print(f'lim_Ey     : {calc.lim_Ey(E_lep)}')

Xis = np.array([calc.getXi(0.1, y) for y in ys])


# Dsigma/Dy comparison 
dsdy_quad = np.array([calc.getDSigmaDy(y,E_lep,'quad') for y in ys])
dsdy_romb = np.array([calc.getDSigmaDy(y,E_lep,'romberg') for y in ys])

plt.plot(ys, dsdy_quad, label='Quad', color='tab:blue')
plt.plot(ys, dsdy_romb, label='Romberg', color='tab:orange')
plt.axvline(ymin, ls=':', color='blue')
plt.axvline(ymax, ls=':', color='blue')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Fraction of Energy loss $y$')
plt.ylabel(r'$\frac{d\sigma}{dy}\,(y)$')
plt.title(f'E = {E_lep} GeV, Mass = {m_lep} GeV')
plt.legend()
plt.show()

# Dsigma/Dy scan 
fig3d = plt.figure()
Ysacc = 20
logYs_wo = np.linspace(-13,0,Ysacc)
logE_wo = np.linspace(5,8,Ysacc)
X, Y = np.meshgrid(logE_wo, logYs_wo)
Z1 = X+Y # dummy 
Z2 = X+Y # dummy 
for i in tqdm(range(Ysacc)):
    for j in range(Ysacc): 
        _ymin, _ymax = calc.lim_y(10**X[i][j])
        if Y[i][j] > np.log10(_ymax) or Y[i][j] < np.log10(_ymin):
            Z1[i][j] = 1e-50
            Z2[i][j] = 1e-50
            continue
        else: 
            Z1[i][j] = calc.getDSigmaDy(10**Y[i][j],10**X[i][j],'quad')
            Z2[i][j] = calc.getDSigmaDy(10**Y[i][j],10**X[i][j],'romberg')

ax1 = Axes3D(fig3d)
ax1.set_xlabel(r'$\log E$')
ax1.set_ylabel(r'$\log y$')
ax1.set_zlabel(r'$\frac{d\sigma}{dy}$')

ax1.plot_surface(X, Y, Z1)
ax1.plot_surface(X, Y, Z2)
ax1.set_zscale('log')

plt.show()

# Singularity study 
y = 0.99
rhomax = calc.lim_rho(y,E_lep)
rho_ys = np.linspace(0.9, rhomax, yacc)
plt.plot(rho_ys, np.array([calc.getDSigmaDyDrho(rho, y, E_lep) for rho in rho_ys])) 
plt.axvline(rhomax,ls=':',color='blue')
plt.xlim(0.985,1.005)
plt.xlabel(r'$\rho$')
plt.ylabel(r'$\frac{d^2\sigma}{dy\,d\rho}$')
plt.title(f'y = {y}, Mass = {m_lep} GeV')
plt.show()

# comparison Le: 
Std_Le = np.array([calc.getLe(rho,y,E_lep) for rho in rho_ys])
Div_Le = np.array([calc.getLeDiv(rho,y,E_lep) for rho in rho_ys])
plt.plot(rho_ys, Std_Le)
plt.plot(rho_ys, Div_Le)
plt.xlabel(r'$\rho$')
plt.ylabel(r'$L_e$')
plt.show()

sys.exit(0)
# Energy Loss comparison 
s_quad = np.array([calc.getEnergyLoss(E,'quad') for E in E_leps])
s_romb = np.array([calc.getEnergyLoss(E,'romberg') for E in E_leps])
plt.plot(E_leps, s_quad, label='Quad', color='tab:blue')
plt.plot(E_leps, s_romb, label='Romberg', color='tab:orange')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Energy [GeV]')
plt.ylabel(r'$\beta_\mathrm{std~rock}$ [cm$^2$/g]')
plt.title(f'Mass = {m_lep} GeV')
plt.legend()
plt.show()

# 3D surface 
lim_log_ys = np.linspace(np.log10(ymin),np.log10(ymax),yacc)
rhos = np.linspace(0.996,1.001,yacc)
X, Y = np.meshgrid(rhos,lim_log_ys)

Z = X+Y # dummy 
for i in tqdm(range(yacc)):
    for j in range(yacc): 
        if np.abs(X[i][j]) > calc.lim_rho(Y[i][j], E_lep): 
            Z[i][j] = 0
            continue
        else: 
            Z[i][j] = calc.getDSigmaDyDrho(X[i][j],10**Y[i][j],E_lep)
        #print (X[i][j], Y[i][j], Z[i][j])

ax = Axes3D(fig3d)
ax.set_xlabel(r'$\rho$')
ax.set_ylabel(r'$\log y$')
ax.set_zlabel(r'$\frac{d^2\sigma}{dy\,d\rho}$')

ax.plot_surface(X, Y, Z)

plt.show()
