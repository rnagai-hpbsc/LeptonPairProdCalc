import numpy as np 
import matplotlib.pyplot as plt
import scipy.integrate

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

def lim_rho(y, E_lep, m_ele): 
    rho_max = (1.-6.*m_lep**2./E_lep**2./(1.-y))*np.sqrt(1.-4.*m_ele/E_lep/y)
    return rho_max

def lim_y(E_lep, m_ele): 
    ymin = 4. * m_ele / E_lep
    ymax = 1. - 0.75 * m_lep / E_lep * sqrtE * (Z**(1./3.)) 
    return ymin, ymax

def getPhiE(E_lep, y, rho, m_ele): 
    beta = y**2/(2.*(1.-y))
    xi = 0.5*(m_lep/m_ele)**2 * beta * (1.-rho**2)
    Y_e = (5. - rho**2. + 4.*beta*(1.+rho**2) ) / (2.*(1.+3.*beta)*np.log(3.+1./xi) - rho**2 - 2.*beta*(2.-rho**2) )
    L_e = np.log(R*Z**(-1./3.)*np.sqrt((1.+xi)*(1.+Y_e))/(1.+2.*m_ele*sqrtE*Z**(-1./3.)*(1.+xi)*(1.+Y_e)/(E_lep*y*(1.-rho**2)))) - 0.5*np.log1p((9./4.) * Z**(2./3.) * (m_ele/m_lep)**2 * (1.+xi) * (1.+Y_e))
    phi_e = (((2.+rho**2)*(1.+beta)+xi*(3.+rho**2))*np.log1p(1./xi) + (1.-rho**2-beta)/(1.+xi)-(3.+rho**2))*L_e
    return phi_e

def getLowLevelPhiE(E_lep, y, rho, m_ele): 
    beta = y**2/(2.*(1.-y))
    xi = 0.5*(m_lep/m_ele)**2 * beta * (1.-rho**2)
    Y_e = (5. - rho**2. + 4.*beta*(1.+rho**2) ) / (2.*(1.+3.*beta)*np.log(3.+1./xi) - rho**2 - 2.*beta*(2.-rho**2) )
    L_e = np.log(R*Z**(-1./3.)*np.sqrt((1.+xi)*(1.+Y_e))/(1.+2.*m_ele*sqrtE*Z**(-1./3.)*(1.+xi)*(1.+Y_e)/(E_lep*y*(1.-rho**2)))) - 0.5*np.log1p((9./4.) * Z**(2./3.) * (m_ele/m_lep)**2 * (1.+xi) * (1.+Y_e))
    phi_e = (((2.+rho**2)*(1.+beta)+xi*(3.+rho**2))*np.log(1.+1./xi) + (1.-rho**2-beta)/(1.+xi)-(3.+rho**2))*L_e
    return phi_e

def getTaylorLog1px(x, n): 
    func = 0
    for i in range(n+1): 
        func += (-1)**i * x**(i+1) / (i+1.)
    return func

def getTaylorPhiE(E_lep,y,rho,m_ele,n):
    beta = y**2/(2.*(1.-y))
    xi = 0.5*(m_lep/m_ele)**2 * beta * (1.-rho**2)
    Y_e = (5. - rho**2. + 4.*beta*(1.+rho**2) ) / (2.*(1.+3.*beta)*np.log(3.+1./xi) - rho**2 - 2.*beta*(2.-rho**2) )
    L_e = np.log(R*Z**(-1./3.)*np.sqrt((1.+xi)*(1.+Y_e))/(1.+2.*m_ele*sqrtE*Z**(-1./3.)*(1.+xi)*(1.+Y_e)/(E_lep*y*(1.-rho**2)))) - 0.5*np.log1p((9./4.) * Z**(2./3.) * (m_ele/m_lep)**2 * (1.+xi) * (1.+Y_e))
    phi_e = (((2.+rho**2)*(1.+beta)+xi*(3.+rho**2))*getTaylorLog1px(1./xi, n) + (1.-rho**2-beta)/(1.+xi)-(3.+rho**2))*L_e
    return phi_e

def getPhiL(E_lep, y, rho, m_ele):
    beta = y**2/(2*(1-y))
    xi = 0.5*(m_lep/m_ele)**2 * beta * (1-rho**2)
    Y_L = (4+rho**2+3*beta*(1+rho**2))/((1+rho**2)*(1.5+2*beta)*np.log(3+xi)+1-1.5*rho**2)
    L_L = np.log(R*Z**(-2/3)*2./3.*m_lep/m_ele/(1+2*m_ele*sqrtE*R*Z**(-1/3)*(1+xi)*(1+Y_L)/(E_lep*y*(1-rho**2))))
    phi_L = (((1+rho**2)*(1+1.5*beta)-(1+2*beta)*(1-rho**2)/xi)*np.log1p(xi)+xi*(1-rho**2-beta)/(1+xi)+(1+2*beta)*(1-rho**2))*L_L
    return phi_L

def simplified_KP(rho, y , E_lep, m_ele): 
    dsigmadydrho = alpha**4 * (2./3.) / np.pi * Z*(Z+1.) *(lambda_e*0.511e-3/m_ele)**2 * (1.-y)/y * getPhiE(E_lep, y, rho, m_ele)
    return dsigmadydrho

def original_KP(rho, y, E_lep, m_ele): 
    dsigmadydrho = alpha**4 * (2/3) / np.pi * Z*(Z+1.) *(lambda_e*0.511e-3/m_ele)**2 * (1.-y)/y * (getPhiE(E_lep, y, rho, m_ele) + m_ele**2/m_lep**2*getPhiL(E_lep, y, rho, m_ele))
    return dsigmadydrho

def taylor_KP(rho, y, E_lep, m_ele, n):
    dsigmadydrho = alpha**4 * (2/3) / np.pi * Z*(Z+1.) *(lambda_e*0.511e-3/m_ele)**2 * (1.-y)/y * (getTaylorPhiE(E_lep, y, rho, m_ele, n) + m_ele**2/m_lep**2*getPhiL(E_lep, y, rho, m_ele))
    return dsigmadydrho

def lowlv_KP(rho, y, E_lep, m_ele): 
    return alpha**4 * (2/3) / np.pi * Z*(Z+1.) *(lambda_e*0.511e-3/m_ele)**2 * (1.-y)/y * (getLowLevelPhiE(E_lep, y, rho, m_ele) + m_ele**2/m_lep**2*getPhiL(E_lep, y, rho, m_ele))

def get_simp_KP_dsigmady(y,E_lep,m_ele): 
    rho_max = lim_rho(y,E_lep,m_ele)
    return scipy.integrate.quad(simplified_KP, -rho_max, rho_max, args=(y,E_lep,m_ele))[0]

def get_orig_KP_dsigmady(y,E_lep,m_ele):
    rho_max = lim_rho(y,E_lep,m_ele)
    return scipy.integrate.quad(original_KP, -rho_max, rho_max, args=(y,E_lep,m_ele))[0]

def get_Taylor_KP_dsigmady(y,E_lep,m_ele,n):
    rho_max = lim_rho(y,E_lep,m_ele)
    return scipy.integrate.quad(taylor_KP, -rho_max, rho_max, args=(y,E_lep,m_ele,n))[0]

def get_LowLv_KP_dsigmady(y,E_lep,m_ele):
    rho_max = lim_rho(y,E_lep,m_ele)
    return scipy.integrate.quad(lowlv_KP, -rho_max, rho_max, args=(y,E_lep,m_ele))[0]

def get_simp_KP_sigma(E_lep,m_ele): 
    ymin, ymax = lim_y(E_lep,m_ele)
    return scipy.integrate.quad(get_simp_KP_dsigmady, ymin, ymax, args=(E_lep,m_ele))[0]

def get_orig_KP_sigma(E_lep,m_ele):
    ymin, ymax = lim_y(E_lep,m_ele)
    return scipy.integrate.quad(get_orig_KP_dsigmady, ymin, ymax, args=(E_lep,m_ele))[0]

def get_Taylor_KP_sigma(E_lep,m_ele,n):
    ymin, ymax = lim_y(E_lep,m_ele)
    return scipy.integrate.quad(get_Taylor_KP_dsigmady, ymin, ymax, args=(E_lep,m_ele,n))[0]

def get_LowLv_KP_sigma(E_lep,m_ele):
    ymin, ymax = lim_y(E_lep,m_ele)
    return scipy.integrate.quad(get_LowLv_KP_dsigmady, ymin, ymax, args=(E_lep,m_ele))[0]

def get_simp_KP_ydsigmady(y,E_lep,m_ele): 
    return y*get_simp_KP_dsigmady(y,E_lep,m_ele)

def get_orig_KP_ydsigmady(y,E_lep,m_ele):
    return y*get_orig_KP_dsigmady(y,E_lep,m_ele)

def get_simp_KP_beta(E_lep,m_ele): 
    ymin, ymax = lim_y(E_lep,m_ele)
    return scipy.integrate.quad(get_simp_KP_ydsigmady, ymin, ymax, args=(E_lep,m_ele))[0]

def get_orig_KP_beta(E_lep,m_ele):
    ymin, ymax = lim_y(E_lep,m_ele)
    return scipy.integrate.quad(get_orig_KP_ydsigmady, ymin, ymax, args=(E_lep,m_ele))[0]

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




