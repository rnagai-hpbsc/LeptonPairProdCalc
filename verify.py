import numpy as np
import KPmodel_libs

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt 
from tqdm import tqdm
import sys
import juliet_read as jr
from juliet_read import getLaTeXStr
from scipy.integrate import quad, romberg

fig = plt.figure()

def main():    
    yacc = 9
    E_leps = np.array([10**logE for logE in np.linspace(5,8,yacc)])
    
    kpmodel = KPmodel_libs.XsecCalculator(m_lep=150) # stau mass of 150 GeV
    taukp = KPmodel_libs.XsecCalculator(m_lep=1.776) # standard model tau
    JlogYs, JlogEs, JlogYDsDys, JDsDy = jr.getIntermedFile('data/stauToePairCreation150GeV.dat')
    JlogYs_mod, JlogEs_mod, JlogYDsDys_mod, JDsDy_mod = jr.getIntermedFile('data/stauToePairCreation150GeV_mod2.dat')
    tauLogYs, tauLogEs, tauLogYDsDys, tauDsDy = jr.getIntermedFile('data/tauToePairCreation.dat')
    
    IntMtx = jr.getIntMtx('data/stauToePairCreationMtx150GeV')
    IntMtx_mod = jr.getIntMtx('data/stauToePairCreationMtx150GeV_mod2')
    tauIntMtx = jr.getIntMtx('data/tauToePairCreationFitMtx')
    E = [10**(5+0.01*iLogE) for iLogE in range(len(IntMtx.sigmaMtx))]
    
    Etab = [10**iLogE for iLogE in JlogEs]
    Ytab = [10**jLogY for jLogY in JlogYs]
    S = []
    for iLogE in range(len(Etab)):
        dS = 0
        for i in range(len(Ytab)-1):
            dx = np.abs(Ytab[i+1] - Ytab[i])
            dS += 0.5 * dx * (JDsDy[iLogE][i+1]+JDsDy[iLogE][i])
        S.append(dS)
    
    tauS = []
    for iLogE in range(len(tauLogEs)):
        dS = 0
        for i in range(len(tauLogYs)-1):
            dx = np.abs(10**tauLogYs[i+1] - 10**tauLogYs[i])
            dS += 0.5 * dx * (tauDsDy[iLogE][i+1]+tauDsDy[iLogE][i])
        tauS.append(dS)
    
    S_mod = []
    for iLogE in range(len(JlogEs_mod)):
        dS = 0
        for i in range(len(JlogYs_mod)-1):
            dx = np.abs(10**JlogYs_mod[i+1] - 10**JlogYs_mod[i])
            dS += 0.5 * dx * (JDsDy_mod[iLogE][i+1]+JDsDy_mod[iLogE][i])
        S_mod.append(dS)
    
    y = 0.9958649775595848
    kpmodel.setRombergTol(1e-8)
    kpmodel.setRombergDivmax(50)
    kpmodel.setRombergDebug()
    factor = kpmodel.getFactor()
    print(kpmodel.getRombergTol())
    print(factor)
    
    #E_lep = E_leps[0]
    #ymin, ymax = kpmodel.lim_y(E_lep)
    #logYs = np.linspace(np.log10(ymin),np.log10(ymax),100)
    #DsigmaDys = [kpmodel.getDSigmaDy(10**logy,E_lep,method='quad') for logy in logYs]
    
    calcdata = np.load('modules/data/calc_150GeV.npz')
    logEs = calcdata['logE']
    logYs = calcdata['logY']
    Ys = calcdata['Y']
    DsigmaDys = calcdata['DsDy']
    sigmapys = calcdata['Sigma']
    
    tauCalc = np.load('modules/data/calc_1.776GeV.npz')
    tauClogEs = tauCalc['logE']
    tauClogYs = tauCalc['logY']
    tauCYs = tauCalc['Y']
    tauCDsigmaDys = tauCalc['DsDy']
    tauCSigmas = tauCalc['Sigma']
    
    energy = 10
    dsdy_logY = [DsigmaDys[energy][i]*np.log(10)*Ys[energy][i] for i in range(len(Ys[energy]))]
    dsdy_tau = [tauCDsigmaDys[energy][i]*np.log(10)*tauCYs[energy][i] for i in range(len(tauCYs[energy]))]
    plt.plot(logYs[energy],dsdy_logY,label='Stau 150 GeV')
    plt.plot(tauClogYs[energy],dsdy_tau,label='SM tau')
    plt.plot(JlogYs,JDsDy[energy]*10**JlogYs*np.log(10),label='Stau 150 GeV (Juliet)')
    #plt.plot(JlogYs_mod,JDsDy_mod[energy]*10**JlogYs_mod*np.log(10),label='Stau 150 GeV (Juliet_mod)')
    plt.xlabel('$\log y$')
    plt.ylabel('$\mathrm{d}\sigma/\mathrm{d}(\log y)$')
    plt.title('Pair Creation $e^+e^-$')
    plt.axvline(JlogYs[0],ls=':',color='magenta')
    plt.axvline(JlogYs[len(JlogYs)-1],ls=':',color='magenta')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plot/verify_integral.pdf')
    plt.show()
    
    sys.exit()
    
    for i in range(int(140/10)):
        index = 10 #i*10 + 10
        ymin, ymax = kpmodel.lim_y(10**logEs[index])
        plt.plot(Ys[index],DsigmaDys[index]*Ys[index], label=f'Python: {getLaTeXStr(sigmapys[index])}')
        plt.plot(10**JlogYs,JDsDy[index]*10**JlogYs, label=f'JULIeT: {getLaTeXStr(S[index])}')
        plt.plot(10**JlogYs_mod,JDsDy_mod[index]*10**JlogYs_mod, ls='--',label=f'JULIeT Mod: {getLaTeXStr(S_mod[index])}')
        plt.xscale('log')
        #plt.yscale('log')
        plt.ylabel('$\mathrm{d}\sigma/\mathrm{d}y$')
        plt.xlabel('Inelasticity $y$')
        plt.axvline(ymin, color='magenta',ls='--')
        plt.axvline(ymax, color='magenta',ls='--')
        #plt.axvline(10**JlogYs[len(JlogYs)-1],color='tab:green',ls=':')
        #plt.axvline(1.e4/(10**logEs[index]), color='tab:green',ls=':')
        #plt.axvline(ymin+2*1e-9, color='tab:green',ls=':')
        print(10**JlogYs[len(JlogYs)-1] * 10**logEs[index])
        plt.title(f'Stau Incident Energy: {getLaTeXStr(10**logEs[index])} GeV')
        plt.legend()
        plt.tight_layout()
        #plt.savefig('plot/vr_dsdy_stau_mod_high.pdf')
        plt.show()
        sys.exit()
    
    for i in range(int(140/10)):
        index = i*10 + 10
        ymin, ymax = taukp.lim_y(10**logEs[index])
        plt.plot(tauCYs[index],tauCDsigmaDys[index], label=f'Python ($\\tau$): {getLaTeXStr(tauCSigmas[index])}')
        plt.plot(10**tauLogYs,tauDsDy[index], label=f'JULIeT ($\\tau$): {getLaTeXStr(tauS[index])}')
        plt.xscale('log')
        plt.ylabel('$\mathrm{d}\sigma/\mathrm{d}y$')
        plt.xlabel('Inelasticity $y$')
        plt.axvline(ymin, color='magenta',ls='--')
        plt.axvline(ymax, color='magenta',ls='--')
        print(10**JlogYs[len(JlogYs)-1] * 10**logEs[index])
        plt.title(f'Tau Incident Energy: {getLaTeXStr(10**logEs[index])} GeV')
        plt.legend()
        plt.tight_layout()
        #plt.savefig('plot/vr_dsdy_tau.pdf')
        plt.show()
        break
    
    
    plt.plot(Etab,sigmapys,label='Python')
    plt.plot(Etab,S,label='JULIeT Table')
    plt.plot(E, IntMtx.sigmaMtx,label='JULIeT Fitted')
    plt.plot(Etab, S_mod, ls='--',label='JULIeT Mod Table')
    plt.plot(E, IntMtx_mod.sigmaMtx,ls='--',label='JULIeT Mod Fitted')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Energy [GeV]')
    plt.ylabel('Total Cross Section [cm$^{2}$]')
    plt.title('Stau 150 GeV')
    plt.ylim(1e-28,2e-24)
    plt.legend()
    plt.tight_layout()
    #plt.savefig('plot/vr_sigma_stau_mod.pdf')
    plt.show()
    
    plt.plot(10**tauClogEs,tauCSigmas,label='Python')
    plt.plot(10**tauLogEs,tauS,label='JULIeT Table')
    plt.plot(E,tauIntMtx.sigmaMtx,label='JULIeT Fitted')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Energy [GeV]')
    plt.ylabel('Total Cross Section [cm$^{2}$]')
    plt.title('Standard Model Tau')
    plt.ylim(1e-28,2e-24)
    plt.legend()
    plt.tight_layout()
    #plt.savefig('plot/vr_sigma_tau.pdf')
    plt.show()
    
    sys.exit()
    
    iLogE = int(index*5)
    JFY = np.array([10**(5+0.01*jLogE)/10**(5+0.01*iLogE) for jLogE in range(iLogE)])
    plt.plot(JFY,IntMtx.transferMtx[iLogE][:len(JFY)],label='JULIeT: transferMtx')
    plt.plot(JFY,IntMtx.transferAMtx[iLogE][:len(JFY)],label='JULIeT: transferAMtx')
    
    
    sigmas = np.array([factor*kpmodel.getSigmaNoFactor(E_lep) for E_lep in E_leps])
    sigmas_quad = np.array([factor*kpmodel.getSigmaNoFactor(E_lep,method='quad') for E_lep in E_leps])
    plt.plot(E_leps,sigmas,label='Python Romberg')
    plt.plot(E_leps,sigmas_quad,label='Python Quad')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Energy [GeV]')
    plt.ylabel('Total Xsec')
    plt.legend()
    plt.tight_layout()
    #plt.savefig('plot/comp_sigma.png',bbox_inches='tight',format='png',dpi=300)
    plt.show()
    
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

if __name__=="__main__":
    main()
