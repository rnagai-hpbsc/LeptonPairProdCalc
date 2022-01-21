import numpy as np 
import matplotlib.pyplot as plt
import scipy.integrate
from scipy.integrate import quad, romberg
import sys
from scipy.interpolate import interp1d


MLEP = 150. # GeV
m_e = 0.511e-3 # GeV

R = 189.
Z = 11.
A = 22.
alpha = 1./137.
lambda_e = 3.8616e-11 #cm : reduced compton wavelength of the electron 

NA = 6.02214e23

sqrtE = np.sqrt(np.e)
print(np.e)

iLogE = 0
jLogY = 0
y = 0.9958649775595848
#y = 0.99

E_lep = 10**(5+0.01*iLogE) # GeV

def lim_rho(y, E_lep, m_ele): 
    rho_max = (1.-6.*m_lep**2./E_lep**2./(1.-y))*np.sqrt(1.-4.*m_ele/E_lep/y)
    return rho_max

def lim_y(E_lep, m_ele): 
    ymin = 4. * m_ele / E_lep
    ymax = 1. - 0.75 * m_lep / E_lep * sqrtE * (Z**(1./3.)) 
    return ymin, ymax

def getYmax(E_lep, m_ele, m):
    return 1. - 0.75 * m / E_lep * sqrtE * (Z**(1./3.)) 

def getYE(E_lep, y, rho, m_ele):
    beta = y**2/(2.*(1.-y))
    xi = 0.5*(m_lep/m_ele)**2 * beta * (1.-rho**2)
    return (5. - rho**2. + 4.*beta*(1.+rho**2) ) / (2.*(1.+3.*beta)*np.log(3.+1./xi) - rho**2 - 2.*beta*(2.-rho**2) )

def getLE(E_lep, y, rho, m_ele):
    beta = y**2/(2.*(1.-y))
    xi = 0.5*(m_lep/m_ele)**2 * beta * (1.-rho**2)
    Y_e = getYE(E_lep,y,rho,m_ele)
    return np.log(R*Z**(-1/3)*np.sqrt((1.+xi)*(1.+Y_e))/(1.+2.*m_ele*sqrtE*Z**(-1/3)*(1.+xi)*(1.+Y_e)/(E_lep*y*(1.-rho**2)))) - 0.5*np.log1p(1.25 * Z**(2./3.) * (m_ele/m_lep)**2 * (1.+xi) * (1.+Y_e))

def getLEmod(E_lep,y,rho,m_ele):
    beta = y**2/(2.*(1.-y))
    xi = 0.5*(m_lep/m_ele)**2 * beta * (1.-rho**2)
    Y_e = getYE(E_lep,y,rho,m_ele)
    return np.log(R*Z**(-1/3)*np.sqrt((1.+xi)*(1.+Y_e))) - np.log1p(2.*m_ele*sqrtE*Z**(-1/3)*(1.+xi)*(1.+Y_e)/(E_lep*y*(1.-rho**2))) - 0.5*np.log1p(1.25 * Z**(2./3.) * (m_ele/m_lep)**2 * (1.+xi) * (1.+Y_e))


def getPhiE(E_lep, y, rho, m_ele): 
    beta = y**2/(2.*(1.-y))
    xi = 0.5*(m_lep/m_ele)**2 * beta * (1.-rho**2)
    L_e = getLE(E_lep,y,rho,m_ele)
    phi_e = (((2.+rho**2)*(1.+beta)+xi*(3.+rho**2))*np.log1p(1./xi) + (1.-rho**2-beta)/(1.+xi)-(3.+rho**2))*L_e
    return phi_e

def getYL(E_lep, y, rho, m_ele):
    beta = y**2/(2*(1-y))
    xi = 0.5*(m_lep/m_ele)**2 * beta * (1-rho**2)
    return (4+rho**2+3*beta*(1+rho**2))/((1+rho**2)*(1.5+2*beta)*np.log(3+xi)+1-1.5*rho**2)

def getYLmod(E_lep,y,rho,m_ele):
    beta = y**2/(2*(1-y))
    xi = 0.5*(m_lep/m_ele)**2 * beta * (1-rho**2)
    return ((4+rho**2)/beta+3*(1+rho**2))/((1+rho**2)*(1.5/beta+2)*np.log(3+xi)+(1-1.5*rho**2)/beta)

def getLL(E_lep, y, rho, m_ele):
    beta = y**2/(2*(1-y))
    xi = 0.5*(m_lep/m_ele)**2 * beta * (1-rho**2)
    Y_L = getYL(E_lep, y, rho, m_ele)
    return np.log(R*Z**(-2/3)*2/3*m_lep/m_ele/(1+2*m_ele*sqrtE*R*Z**(-1/3)*(1+xi)*(1+Y_L)/(E_lep*y*(1-rho**2))))

def getLLmod(E_lep,y,rho,m_ele):
    beta = y**2/(2*(1-y))
    xi = 0.5*(m_lep/m_ele)**2 * beta * (1-rho**2)
    Y_L = getYL(E_lep, y, rho, m_ele)
    delta = 2*m_ele*(1+xi)/E_lep/y/(1-rho**2)
    gamma = R*delta/Z**(1/3)
    return np.log(R*Z**(-2/3)*2/3*m_lep/m_ele/(1+gamma*sqrtE*(1+Y_L)))


def LLdenomNum(E_lep, y, rho, m_ele):
    beta = y**2/(2*(1-y))
    xi = 0.5*(m_lep/m_ele)**2 * beta * (1-rho**2)
    Y_L = getYL(E_lep, y, rho, m_ele)
    delta = 2*m_ele**2*(1+xi)/E_lep/y/(1-rho**2)
    gamma = R*delta/m_ele/Z**(1/3)
    return 2*m_ele*sqrtE*R*Z**(-1/3)*(1+xi)*(1+Y_L)/E_lep/y/(1-rho**2),delta,gamma, gamma*sqrtE*(1+Y_L), 2*m_ele*sqrtE*R/Z**(1/3)*(1+xi)*(1+Y_L)/(E_lep*y*(1-rho**2))

def MinusLL(E_lep,y,rho,m_ele):
    Y_L = getYL(E_lep,y,rho,m_ele)
    return -np.log(3*Z**(1/3)*(.5*m_ele*Z**(1/3)/m_lep/R + m_ele**2*sqrtE*(1+Y_L)/m_lep/E_lep/y/(1-rho**2) + m_lep*y*sqrtE*(1+Y_L)/4/E_lep/(1-y)))

def MinusLL_eachTerm(E_lep,y,rho,m_ele):
    Y_L = getYL(E_lep,y,rho,m_ele)
    return .5*m_ele*Z**(1/3)/m_lep/R, m_ele**2*sqrtE*(1+Y_L)/m_lep/E_lep/y/(1-rho**2), m_lep*y*sqrtE*(1+Y_L)/4/E_lep/(1-y)

def getLL_denom(E_lep, y, rho, m_ele):
    beta = y**2/(2*(1-y))
    xi = 0.5*(m_lep/m_ele)**2 * beta * (1-rho**2)
    Y_L = getYL(E_lep, y, rho, m_ele)
    return 2*m_ele*sqrtE*R*Z**(-1/3)*(1+xi)*(1+Y_L)/(E_lep*y*(1-rho**2))

def getPhiL(E_lep, y, rho, m_ele):
    beta = y**2/(2*(1-y))
    xi = 0.5*(m_lep/m_ele)**2 * beta * (1-rho**2)
    L_L = getLL(E_lep, y, rho, m_ele)
    phi_L = (((1+rho**2)*(1+1.5*beta)-(1+2*beta)*(1-rho**2)/xi)*np.log1p(xi)+xi*(1-rho**2-beta)/(1+xi)+(1+2*beta)*(1-rho**2))*L_L
    return phi_L

def AsymTerm(rho, y, E_lep, m_ele):
    return getPhiE(E_lep,y,rho,m_ele) + (m_ele/m_lep)**2 * getPhiL(E_lep,y,rho,m_ele)

def original_KP(rho, y, E_lep, m_ele): 
    dsigmadydrho = alpha**4 * (2/3) / np.pi * Z*(Z+1.) *(lambda_e*0.511e-3/m_ele)**2 * (1.-y)/y * (getPhiE(E_lep, y, rho, m_ele) + m_ele**2/m_lep**2*getPhiL(E_lep, y, rho, m_ele))
    return dsigmadydrho

def get_orig_KP_dsigmady(y,E_lep,m_ele):
    rho_max = lim_rho(y,E_lep,m_ele)
    return scipy.integrate.quad(original_KP, -rho_max, rho_max, args=(y,E_lep,m_ele))[0]

def get_orig_KP_sigma(E_lep,m_ele):
    ymin, ymax = lim_y(E_lep,m_ele)
    return scipy.integrate.quad(get_orig_KP_dsigmady, ymin, ymax, args=(E_lep,m_ele))[0]

def get_orig_KP_beta(E_lep,m_ele):
    ymin, ymax = lim_y(E_lep,m_ele)
    return scipy.integrate.quad(get_orig_KP_ydsigmady, ymin, ymax, args=(E_lep,m_ele))[0]

fig = plt.figure()
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['axes.formatter.use_mathtext'] = True

logE_leps = np.linspace(5,12,20)
E_leps = np.array([10**logE for logE in logE_leps])

m_lep=MLEP
rhomax = lim_rho(y, E_lep, m_e)
rhos = np.linspace(-rhomax, rhomax, 100)
DsDyDrhos = np.array([AsymTerm(rho,y,E_lep,m_e) for rho in rhos])
PhiEs = np.array([getPhiE(E_lep,y,rho,m_e) for rho in rhos])
LEs = np.array([getLE(E_lep,y,rho,m_e) for rho in rhos])
PhiLs = np.array([getPhiL(E_lep,y,rho,m_e) for rho in rhos])
LLs = np.array([getLL(E_lep,y,rho,m_e) for rho in rhos])
YLs = np.array([getYL(E_lep,y,rho,m_e) for rho in rhos])
LL_denom = np.array([getLL_denom(E_lep,y,rho,m_e) for rho in rhos])

print(romberg(AsymTerm, -rhomax, rhomax, args=(y,E_lep,m_e)), np.sum(DsDyDrhos))

ymin, ymax = lim_y(E_lep,m_e)
print(ymax, y, 1-m_lep/E_lep)
print(f'Max mass for y: {4*(1-ymax)/sqrtE/Z**(1/3)/3*E_lep}')

IntDsDyDrhos = []
minrange = 5
maxrange = 100
xdata = [i+minrange for i in range(maxrange-minrange)]
for i in range(maxrange-minrange):
    tmprhos = np.linspace(-rhomax, rhomax, i+minrange)
    if len(tmprhos)>1: 
        drhos = tmprhos[1] - tmprhos[0]
    else: 
        drhos = 2*rhomax
    tmpDsDyDrhos = np.array([AsymTerm(rho,y,E_lep,m_e) for rho in tmprhos])
    IntDsDyDrhos.append(np.sum(tmpDsDyDrhos)*drhos)

print(LLdenomNum(E_lep,y,0,m_e),getYL(E_lep,y,0,m_e))
print(getLL(E_lep,y,0,m_e), MinusLL(E_lep,y,0,m_e))
print(getLL(E_lep,y,0,m_e), getLLmod(E_lep,y,0,m_e))
print(getLE(E_lep,y,0,m_e), getLEmod(E_lep,y,0,m_e))
a,b,c=MinusLL_eachTerm(E_lep,y,0,m_e)
print(-np.log(3*Z**(1/3))-np.log(a+b+c),-np.log(3*Z**(1/3)*(a+b+c)))

rhoscanDefLL = np.array([getLL(E_lep,y,rho,m_e) for rho in rhos])
rhoscanModLL = np.array([getLLmod(E_lep,y,rho,m_e) for rho in rhos])

plt.plot(rhos,rhoscanDefLL,label='Default Calc of $L_\mathrm{lep}$')
#plt.plot(rhos,rhoscanModLL,label='Modified Calc of $L_\mathrm{lep}$')
#plt.legend()
plt.xlabel('Asymmetry Parameter $\\rho$')
plt.ylabel('$L_\mathrm{lep}$')
plt.tight_layout()
plt.savefig('plot/rhoscanLL.png',bbox_inches='tight',format='png',dpi=300)
plt.show()

yscanrange = np.linspace(ymin,ymax,100)
yscanDefLL = np.array([getLL(E_lep,y,0,m_e) for y in yscanrange])
yscanModLL = np.array([getLLmod(E_lep,y,0,m_e) for y in yscanrange])

yscanDefLL_rhomax = np.array([getLL(E_lep,y,lim_rho(y,E_lep,m_e),m_e) for y in yscanrange])

plt.plot(yscanrange,yscanDefLL,label='$L_\mathrm{lep}$ at $\\rho = 0$')
plt.plot(yscanrange,yscanDefLL_rhomax,label='$L_\mathrm{lep}$ at $\\rho = \\rho_\mathrm{max}$',ls='--')
#plt.plot(yscanrange,yscanModLL,label='Modified Calc of $L_\mathrm{lep}$')
plt.axhline(0,color='black')
plt.xlabel('Inelasticity $y$')
plt.ylabel('Cross section w/o factor')
plt.legend()
plt.tight_layout()
plt.savefig('plot/yscanLL.png',bbox_inches='tight',format='png',dpi=300)
plt.show()

sys.exit()

plt.plot(yscanrange,yscanDefLL-yscanModLL)
plt.xlabel('Inelasticity $y$')
plt.ylabel('Difference Def - Mod')
plt.tight_layout()
plt.show()

yscanDefYL = np.array([getYL(E_lep,y,0,m_e) for y in yscanrange])
yscanModYL = np.array([getYLmod(E_lep,y,0,m_e) for y in yscanrange])
plt.plot(yscanrange,yscanDefYL,label='Default Calc of $Y_\mathrm{lep}$')
plt.plot(yscanrange,yscanModYL,label='Modified Calc of $Y_\mathrm{lep}$')
plt.xlabel('Inelasticity $y$')
plt.ylabel('Value')
plt.legend()
plt.tight_layout()
plt.show()

sys.exit()

plt.plot(yscanrange,(yscanDefYL-yscanModYL)/yscanDefYL)
plt.xlabel('Inelasticity $y$')
plt.ylabel('Difference Def - Mod')
plt.tight_layout()
plt.show()

#sys.exit()

plt.plot(xdata,IntDsDyDrhos,label='Sum of the elements')
plt.axhline(0,color='black')
plt.axhline(romberg(AsymTerm, -rhomax, rhomax, args=(y,E_lep,m_e)),color='magenta',linestyle=':',label='Python Romberg default (tol: 1.48e-8)')
plt.axhline(romberg(AsymTerm, -rhomax, rhomax, args=(y,E_lep,m_e), tol=1e-20,divmax=100),color='magenta',linestyle='--',label='Python Romberg (tol: 1e-20)')
plt.axhline(quad(AsymTerm, -rhomax, rhomax, args=(y,E_lep,m_e))[0],color='blue',linestyle=':',label='Python Quad')
plt.axhline(-7.3889462891444814e-9,color='tab:orange',linestyle=':',label='Juliet Romberg')

plt.xlim(0, maxrange)
plt.title(f'$y$ = {y}')
plt.xlabel('Number of samples')
plt.ylabel('Integrated Asymmetry Term Value')
plt.legend()
plt.tight_layout()
plt.savefig('plot/asymvalue.png',bbox_inches='tight',format='png',dpi=300)
plt.show()

plt.plot(rhos,DsDyDrhos,label='Asym Term')
plt.plot(rhos,PhiEs,label='$\\phi_e$')
plt.plot(rhos,((m_e/m_lep)**2)*PhiLs,label='$\\phi_\\ell$ w/ mass scale')
plt.xlabel('Asymmetry Parameter $\\rho$')
plt.ylabel('Differential Cross Section w/o factor')
plt.title(f'$y$ = {y}')
plt.axhline(0,color='black')
plt.legend()
plt.tight_layout()
plt.savefig('plot/asymshape.png',bbox_inches='tight',format='png',dpi=300)
plt.show()

plt.plot(rhos,LLs)
plt.tight_layout()
#plt.axhline(np.log(2/3*m_lep/m_e*R*Z**(-2/3)))
plt.show()

yscans = [getYmax(E_lep,m_e,4*m+2) for m in range(60)]
LLs_rho0 = [getLL(E_lep,yscan,0,m_e) for yscan in yscans]
f = interp1d(LLs_rho0,yscans,kind='cubic')
print(f(0))
plt.plot(yscans, LLs_rho0)
plt.axvline(ymax,color='magenta',linestyle=':')
plt.axvline(f(0),color='tab:green',linestyle=':')
y_min, y_max = plt.gca().get_ylim()
y_loc = .8*y_max + .2*y_min
y_loc2 = .7*y_max + .3*y_min
plt.text(ymax,y_loc,'150 GeV',color='magenta',verticalalignment='center',horizontalalignment='center')
plt.text(f(0),y_loc2,f'{4/3*E_lep*(1-f(0))/sqrtE/Z**(1/3):.1f} GeV',color='tab:green',verticalalignment='center',horizontalalignment='center')
for m in range(4):
    mass = 200-50*m
    if mass==150:
        continue
    ymax4mass = getYmax(E_lep,m_e,mass)
    plt.axvline(ymax4mass,color='gray',linestyle=':',linewidth=1)
    plt.text(ymax4mass,y_loc,f'{mass} GeV',color='gray', verticalalignment='center',horizontalalignment='center')
plt.axhline(0,color='black')
plt.xlabel('Inelasticity $y$')
plt.ylabel('$L_\mathrm{lep}$ value at $\\rho = 0$')
plt.tight_layout()
plt.savefig('plot/yscan.png',bbox_inches='tight',format='png',dpi=300)
plt.show()

ynum = 10
ydiv = (ymax - f(0))/ynum
fsum = []
fys = []
for i in range(ynum):
    fy = ymax-ydiv*i
    fys.append(fy)
    frhomax = lim_rho(fy,E_lep,m_e)
    frhos = np.linspace(-frhomax, frhomax, 100)
    fsum.append(quad(AsymTerm,-frhomax,frhomax, args=(fy,E_lep,m_e))[0])
    func = [AsymTerm(frho, fy, E_lep, m_e) for frho in frhos]
    plt.plot(frhos,func)
plt.axhline(0,color='black')
plt.tight_layout()
plt.show()

plt.plot(fys,fsum)
plt.show()
sys.exit()

plt.plot(rhos,LEs)
plt.axhline(0,color='black')
plt.show()
plt.plot(rhos,PhiEs/LEs)
plt.axhline(0,color='black')
plt.show()
plt.plot(rhos,LLs)
plt.axhline(0,color='black')
plt.show()
plt.plot(rhos,YLs)
plt.axhline(0,color='black')
plt.show()


'''
n=1
m_lep=MLEP
orig_KP_sigmas_e = np.array([get_orig_KP_sigma(10**logE,m_e) for logE in logE_leps])

plt.plot(E_leps, orig_KP_sigmas_e,label="Original", zorder=0, color="tab:blue")
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
'''



