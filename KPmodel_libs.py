import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import quad, romberg


class XsecCalculator:
    
    M_E   = 0.511e-3 #GeV
    M_MU  = 105.658e-3 #GeV
    M_TAU = 1.7768 #GeV

    sqrtE = np.sqrt(np.e) # sqrt of Napier's Constant
    NA = 6.02214e23 # Avogadro Constant 
    R = 189. # Radiation Constant 
    alpha = 1./137. # Fine Structure Constant 
    lambda_e = 3.8616e-11 #cm : reduced compton wavelength of the electron 
    quadlimit = 1000

    def __init__(self, m_lep=M_MU, m_ele=M_E, material='rock', n=1): 
        self.m_lep = m_lep
        self.m_ele = m_ele
        self.n = n
        self.accrho = False
        self.rhoSafetyFactor = 0
        self.ySafetyFactor = 0
        if material=='rock':
            self.Z = 11.
            self.A = 22. 
        else: 
            print('Undefined material seting... We set Z=0, A=0.')
            self.Z = 0.
            self.A = 0.

    def setAccRho(self): 
        self.accrho = True

    def setRhoSafetyFactor(self, value): 
        self.rhoSafetyFactor = float(value)

    def setySafetyFactor(self, value):
        self.ySafetyFactor = float(value)

    def getZ3(self):
        return self.Z**(1./3.)

    def getMassRatio(self):
        return self.m_lep / self.m_ele
    
    def lim_rho(self, y, E_lep): 
        if self.accrho is False: 
            return self.lim_rho_raw(y, E_lep)
        else: 
            return self.lim_rho_Ey(y*E_lep, E_lep)
    
    def lim_rho_raw(self, y, E_lep):
        rho_max = (1.-6.*self.m_lep**2./E_lep**2./(1.-y))*np.sqrt(1.-4.*self.m_ele/E_lep/y)
        return rho_max

    def lim_rho_Ey(self, Ey, E_lep):
        rho_max = (1.-6.*self.m_lep**2/E_lep/(E_lep-Ey))*np.sqrt(1.-4.*self.m_ele/Ey)
        return rho_max
    
    def lim_y(self, E_lep): 
        ymin = 4. * self.m_ele / E_lep
        ymax = 1. - 0.75 * self.m_lep / E_lep * self.sqrtE * self.getZ3() 
        return ymin, ymax

    def lim_Ey(self, E_lep): 
        Eymin = 4. * self.m_ele
        Eymax = E_lep - 0.75 * self.m_lep * self.sqrtE * self.getZ3()
        return Eymin, Eymax

    def getLog1Px(self, x): 
        func = 0
        N = self.n
        if N < 0: 
            func = np.log(1.+x) 
        elif N==0: 
            func = np.log1p(x)
        elif N > 0: 
            for i in range(N+1):
                func += (-1)**i * x**(i+1) / (i+1.)
        return func 

    def getBeta(self, y): 
        return y**2 / (2.*(1.-y))
    
    def getXi(self, rho, y): 
        return 0.5*self.getMassRatio()**2 * self.getBeta(y) * (1-rho**2)
    
    def getYe(self, rho, y): 
        return (5. - rho**2 + 4.*self.getBeta(y)*(1.+rho**2) ) / (2.*(1.+3.*self.getBeta(y))*np.log(3.+1./self.getXi(rho,y)) - rho**2 - 2.*self.getBeta(y)*(2.-rho**2) ) 
    
    def getYl(self, rho, y): 
        return (4. + rho**2 + 3.*self.getBeta(y)*(1.+rho**2) ) / ((1.+rho**2)*(1.5+2.*self.getBeta(y))*np.log(3+self.getXi(rho,y))+1.-1.5*rho**2)

    def getLe(self, rho, y, E_lep): 
        return np.log(self.R/self.getZ3()*np.sqrt((1.+self.getXi(rho,y))*(1.+self.getYe(rho,y))) \
                / (1.+2.*self.m_ele*self.sqrtE/self.getZ3()*(1.+self.getXi(rho,y))*(1.+self.getYe(rho,y)) / (E_lep*y*(1.-rho**2)) )) \
                - 0.5*np.log1p(2.25 * self.getZ3()**2 / self.getMassRatio()**2 * (1.+self.getXi(rho,y)) * (1.+self.getYe(rho,y)))
    
    def getLeDiv(self, rho, y, E_lep):
        return np.log(self.R/self.getZ3()*np.sqrt((1.+self.getXi(rho,y))*(1.+self.getYe(rho,y)))) \
                - np.log1p(2.*self.m_ele*self.sqrtE/self.getZ3()*(1.+self.getXi(rho,y))*(1.+self.getYe(rho,y)) / (E_lep*y*(1.-rho**2))) \
                - 0.5 * np.log1p(2.25 * self.getZ3()**2 / self.getMassRatio()**2 * (1.+self.getXi(rho,y)) * (1.+self.getYe(rho,y))) 

    def getPhie(self, rho, y, E_lep): 
        return (((2.+rho**2)*(1.+self.getBeta(y))+self.getXi(rho,y)*(3.+rho**2))*np.log1p(1./self.getXi(rho,y)) + (1.-rho**2-self.getBeta(y)) / (1.+self.getXi(rho,y)) - (3.+rho**2)) \
                *self.getLe(rho,y,E_lep)

    def getLl(self, rho, y, E_lep):
        return np.log(self.R / self.getZ3()**2 / 1.5 * self.getMassRatio() / (1.+2.*self.m_ele*self.sqrtE*self.R/self.getZ3()*(1.+self.getXi(rho,y))*(1.+self.getYl(rho,y)) / (E_lep*y*(1.-rho**2))) )

    def getPhil(self, rho, y, E_lep):
        return (((1.+rho**2)*(1.+1.5*self.getBeta(y)) - (1.+2.*self.getBeta(y)) * (1.-rho**2)/self.getXi(rho,y)) * np.log1p(self.getXi(rho,y)) \
                + self.getXi(rho,y) * (1.-rho**2-self.getBeta(y))/(1.+self.getXi(rho,y)) + (1.+2.*self.getBeta(y)) * (1.-rho**2)) * self.getLl(rho,y,E_lep)

    def getDSigmaDyDrho(self, rho, y, E_lep): 
        return self.alpha**4 / 1.5 / np.pi * self.Z*(self.Z+1.) * (self.lambda_e*self.M_E/self.m_ele)**2 * (1.-y)/y * (self.getPhie(rho,y,E_lep) + self.m_ele**2 / self.m_lep**2 * self.getPhil(rho,y,E_lep))

    def getDSigmaDy(self, y, E_lep, method): 
        rho_max = self.lim_rho(y,E_lep)
        rhoBound = rho_max - self.rhoSafetyFactor
        #print(rhoBound)
        if method == 'quad': 
            #print("Gauss Quadrature Integration will be performed.")
            return quad(self.getDSigmaDyDrho, -rhoBound, rhoBound, args=(y,E_lep), limit=self.quadlimit)[0]
        elif method == 'romberg':
            #print("Romberg Integration will be performed.")
            dsigmady = romberg(self.getDSigmaDyDrho, -rhoBound, rhoBound, args=(y,E_lep))
            return dsigmady
        else: 
            print("Invalid Integration method. Return 0.")
            return 0

    def getSigma(self, E_lep, method): 
        ymin, ymax = self.lim_y(E_lep)
        yminBound = ymin + self.ySafetyFactor/E_lep
        ymaxBound = ymax #- self.ySafetyFactor
        if method == 'quad':
            #print("Gauss Quadrature Integration will be performed.")
            return quad(self.getDSigmaDy, yminBound, ymaxBound, args=(E_lep, method), limit=self.quadlimit)[0]
        elif method == 'romberg': 
            #print("Romberg Integration will be performed.")
            return romberg(self.getDSigmaDy, yminBound, ymaxBound, args=(E_lep, method))
        else: 
            print("Invalid Integration method. Return 0.")
            return 0

    def getyDSigmaDy(self, y, E_lep, method): 
        ydsdy =  y*self.getDSigmaDy(y, E_lep, method)
        #print(f'{y}, {self.getDSigmaDy(y, E_lep, method)}, {ydsdy}')
        return ydsdy

    def getEnergyLoss(self, E_lep, method): 
        ymin, ymax = self.lim_y(E_lep)
        yminBound = ymin + self.ySafetyFactor/E_lep
        ymaxBound = ymax #- self.ySafetyFactor
        if method == 'quad': 
            #print("Gauss Quadrature Integration will be performed.")
            return self.NA/self.A * quad(self.getyDSigmaDy, yminBound, ymaxBound, args=(E_lep, method))[0]
        elif method == 'romberg':
            #print("Romberg Integration will be performed.")
            energyloss = self.NA/self.A * romberg(self.getyDSigmaDy, yminBound, ymaxBound, args=(E_lep, method))
            return energyloss
        else:
            print("Invalid Integration method. Return 0.")
            return 0

    def getSigmaArray(self, logE_leps, method): 
        return np.array([self.getSigma(10**logE, method) for logE in logE_leps])
    
    def getEnergyLossArray(self, logE_leps, method): 
        return np.array([self.getEnergyLoss(10**logE, method) for logE in logE_leps])


"""
MLEP = 200. # GeV
m_e = 0.511e-3 # GeV

R = 189.
Z = 11.
A = 22.
alpha = 1./137.
lambda_e = 3.8616e-11 #cm : reduced compton wavelength of the electron 

NA = 6.02214e23

sqrtE = np.sqrt(np.e)
print(np.e)
"""



"""
fig = plt.figure()
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True

logE_leps = np.linspace(5,12,20)
E_leps = np.array([10**logE for logE in logE_leps])

n=1
m_lep=MLEP
orig_KP_sigmas_e = np.array([get_orig_KP_sigma(10**logE,m_e) for logE in logE_leps])
simp_KP_sigmas_e = np.array([get_simp_KP_sigma(10**logE,m_e) for logE in logE_leps])
taylor_KP_sigmas_e = np.array([get_Taylor_KP_sigma(10**logE,m_e,n) for logE in logE_leps])
lowlv_KP_sigmas_e = np.array([get_LowLv_KP_sigma(10**logE,m_e) for logE in logE_leps])

plt.plot(E_leps, orig_KP_sigmas_e,label="Original", zorder=0, color="tab:blue")
plt.plot(E_leps, simp_KP_sigmas_e,label="Simplified", ls='--', zorder=1, color="tab:orange")
plt.plot(E_leps, taylor_KP_sigmas_e,label=f"{int(n)}-order Taylor", ls=':', zorder=2, color="tab:green")
plt.plot(E_leps, lowlv_KP_sigmas_e,label="Low-Accuracy", ls=':', zorder=3, color="tab:red")
plt.xlabel(r"E [GeV]")
plt.ylabel(r"$\sigma_{\mathrm{std~rock}}$ [$\mathrm{cm}^2$]")
plt.title(f'Pair Production Total Cross Section for Stau {int(m_lep)} GeV.')
plt.xscale('log')
plt.yscale('log')
plt.legend()
#plt.ylim(1e-7,0.1)
plt.savefig(f"comp_methods_for_{int(m_lep)}GeV.pdf")

#plt.plot(rho_x, simplified_KP(E_lep, y, rho_x))
#plt.show()

fig2 = plt.figure()

m_mu = 105.e-3
m_tau = 1.78

orig_KP_sigmas_e = np.array([get_orig_KP_sigma(10**logE,m_e) for logE in logE_leps])
orig_KP_sigmas_mu = np.array([get_orig_KP_sigma(10**logE,m_mu) for logE in logE_leps])
orig_KP_sigmas_tau = np.array([get_orig_KP_sigma(10**logE,m_tau) for logE in logE_leps])
plt.plot(E_leps, orig_KP_sigmas_e,label=r"$e^+e^-$",color='tab:green')
plt.plot(E_leps, orig_KP_sigmas_mu,label=r"$\mu^+\mu^-$",color='tab:red')
plt.plot(E_leps, orig_KP_sigmas_tau,label=r"$\tau^+\tau^-$",color='tab:purple')
plt.xlabel(r"E [GeV]")
plt.ylabel(r"$\sigma_{\mathrm{std~rock}}$ [$\mathrm{cm}^2$]")
plt.title(f'Pair Production Total Cross Section for Stau {int(m_lep)} GeV.')
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-38,1e-26)
plt.legend()
#plt.savefig(f'pairproduction_for_stau{int(m_lep)}GeV.pdf')
#plt.show()

fig3 = plt.figure()

m_lep=200.
ee_KP_sigmas_200 = np.array([get_orig_KP_sigma(10**logE,m_e) for logE in logE_leps])
mm_KP_sigmas_200 = np.array([get_orig_KP_sigma(10**logE,m_mu) for logE in logE_leps])
tt_KP_sigmas_200 = np.array([get_orig_KP_sigma(10**logE,m_tau) for logE in logE_leps])
m_lep=400.
ee_KP_sigmas_400 = np.array([get_orig_KP_sigma(10**logE,m_e) for logE in logE_leps])
mm_KP_sigmas_400 = np.array([get_orig_KP_sigma(10**logE,m_mu) for logE in logE_leps])
tt_KP_sigmas_400 = np.array([get_orig_KP_sigma(10**logE,m_tau) for logE in logE_leps])
m_lep=600.
ee_KP_sigmas_600 = np.array([get_orig_KP_sigma(10**logE,m_e) for logE in logE_leps])
mm_KP_sigmas_600 = np.array([get_orig_KP_sigma(10**logE,m_mu) for logE in logE_leps])
tt_KP_sigmas_600 = np.array([get_orig_KP_sigma(10**logE,m_tau) for logE in logE_leps])
plt.plot(E_leps, ee_KP_sigmas_200,color='tab:green')
plt.plot(E_leps, mm_KP_sigmas_200,color='tab:red')
plt.plot(E_leps, tt_KP_sigmas_200,color='tab:purple')
plt.plot(E_leps, ee_KP_sigmas_400,color='tab:green',ls='--')
plt.plot(E_leps, mm_KP_sigmas_400,color='tab:red',ls='--')
plt.plot(E_leps, tt_KP_sigmas_400,color='tab:purple',ls='--')
plt.plot(E_leps, ee_KP_sigmas_600,color='tab:green',ls=':')
plt.plot(E_leps, mm_KP_sigmas_600,color='tab:red',ls=':')
plt.plot(E_leps, tt_KP_sigmas_600,color='tab:purple',ls=':')
plt.plot(0,0,label=r'$e^+e^-$',color='tab:green')
plt.plot(0,0,label=r'$\mu^+\mu^-$',color='tab:red')
plt.plot(0,0,label=r'$\tau^+\tau^-$',color='tab:purple')
plt.plot(0,0,label=r'Stau 200 GeV',color='tab:gray')
plt.plot(0,0,label=r'Stau 400 GeV',color='tab:gray',ls='--')
plt.plot(0,0,label=r'Stau 600 GeV',color='tab:gray',ls=':')
plt.xlabel(r"E [GeV]")
plt.ylabel(r"$\sigma_{\mathrm{std~rock}}$ [$\mathrm{cm}^2$]")
plt.title(f'Pair Production Total Cross Section for Stau.')
plt.xscale('log')
plt.yscale('log')
plt.ylim(5e-35,2e-30)
plt.legend(ncol=2)
#plt.savefig(f'pairproduction_for_staus.pdf')

fig4 = plt.figure()
m_lep=MLEP
E_lep=100e6
Z=11
ymin, ymax = lim_y(E_lep,m_e)
y=0.99
n=1
rho_max = lim_rho(y,E_lep,m_e)
#print(y, rho_max, get_simp_KP_dsigmady(y,E_lep,m_e))
rho_x = np.linspace(-1.*rho_max,rho_max,200)
#print(rho_x)
#print(simplified_KP(rho_x, y, E_lep, m_e))
plt.plot(rho_x, getLowLevelPhiE(E_lep, y, rho_x, m_e), label="Using log(1+1/x)")
plt.plot(rho_x, getPhiE(E_lep, y, rho_x, m_e), label="Using log1p(1/x)")
plt.plot(rho_x, getTaylorPhiE(E_lep, y, rho_x, m_e, n), label=f"{int(n)}-order Taylor series")
plt.xlabel(r"Asymmetric Parameter $\rho$")
plt.ylabel(r"$\phi_e$")
#plt.ylim(-0.7e-31,1.5e-31)
plt.title(r"$\phi_e$ behaviors. $E_\mathrm{lep} = 10^8~\mathrm{GeV}$.")
plt.legend()
plt.savefig(f'DsigmaDyDrho-rho_comp_{int(m_lep)}GeV.pdf')
plt.show()

"""


