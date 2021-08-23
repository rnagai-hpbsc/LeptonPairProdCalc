import numpy as np 
import matplotlib.pyplot as plt
import scipy.integrate

m_lep = 200 # GeV
m_ele = 0.511e-3#511e-6 # GeV

R = 189
Z = 22
alpha = 1./137.
lambda_e = 3.8616e-11 #cm : reduced compton wavelength of the electron 

N_A = 6.02214e23

sqrtE = np.sqrt(np.e)

def calc_params(E_lep, y):
    ymin = 4. * m_ele / E_lep
    ymax = 1. - 0.75 * m_lep / E_lep * sqrtE * (Z**(1./3.)) 
    rho_max = (1.-6.*m_lep**2./E_lep**2./(1.-y))*np.sqrt(1.-4.*m_ele/E_lep/y)

    return ymin, ymax, rho_max

def lim_rho(y, E_lep): 
    rho_max = (1.-6.*m_lep**2./E_lep**2./(1.-y))*np.sqrt(1.-4.*m_ele/E_lep/y)
    return rho_max

def lim_y(E_lep): 
    ymin = 4. * m_ele / E_lep
    ymax = 1. - 0.75 * m_lep / E_lep * sqrtE * (Z**(1./3.)) 
    return ymin, ymax

print(calc_params(100e6, 0.99))

def getPhiE(E_lep, y, rho): 
    beta = y**2/(2*(1-y))
    xi = 0.5*(m_lep/m_ele)**2 * beta * (1-rho**2)
    Y_e = (5 - rho**2 + 4*beta*(1+rho**2) ) / (2*(1+3*beta)*np.log(3+1/xi) - rho**2 - 2*beta*(2-rho**2) )
    L_e = np.log(R*Z**(-1/3)*np.sqrt((1+xi)*(1+Y_e))/(1+2*m_ele*sqrtE*Z**(-1/3)*(1+xi)*(1+Y_e)/(E_lep*y*(1-rho**2)))) - 0.5*np.log1p((9./4.) * Z**(2/3) * (m_ele/m_lep)**2 * (1+xi) * (1+Y_e))
    phi_e = (((2+rho**2)*(1+beta)+xi*(3+rho**2))*np.log(1+1/xi) + (1-rho**2-beta)/(1+xi)-(3+rho**2))*L_e
    #print(beta, Y_e, L_e)
    return phi_e

def getPhiL(E_lep, y, rho):
    beta = y**2/(2*(1-y))
    xi = 0.5*(m_lep/m_ele)**2 * beta * (1-rho**2)
    Y_L = (4+rho**2+3*beta*(1+rho**2))/((1+rho**2)*(1.5+2*beta)*np.log(3+xi)+1-1.5*rho**2)
    L_L = np.log(R*Z**(-2/3)*2./3.*m_lep/m_ele/(1+2*m_ele*sqrtE*R*Z**(-1/3)*(1+xi)*(1+Y_L)/(E_lep*y*(1-rho**2))))
    phi_L = (((1+rho**2)*(1+1.5*beta)-(1+2*beta)*(1-rho**2)/xi)*np.log1p(xi)+xi*(1-rho**2-beta)/(1+xi)+(1+2*beta)*(1-rho**2))*L_L
    return phi_L

def simplified_KP(E_lep, y, rho): 
    dsigmadydrho = alpha**4 * (2/3) / np.pi * (Z*lambda_e)**2 * (1-y)/y * getPhiE(E_lep, y, rho)
    return dsigmadydrho

def original_KP(E_lep, y, rho): 
    dsigmadydrho = alpha**4 * (2/3) / np.pi * (Z*lambda_e)**2 * (1-y)/y * (getPhiE(E_lep, y, rho) + m_ele**2/m_lep**2*getPhiL(E_lep, y, rho))
    return dsigmadydrho

print(simplified_KP(100e6,0.99,0))

def easy_KP(E_lep,y): # xi >> 1 case
    F_lep = (((4./3.)-(4./3.)*y+y**2)*(np.log(m_lep**2 * y**2/(m_ele**2 * (1-y)))-2)+10./9.*(1-y))*np.log(2./3.*m_lep/m_ele*R*Z**(-2/3)/(1+R*sqrtE*Z**(-1/3)*m_lep**2*y/(2*m_ele*E_lep*(1-y))))
    dsigmady = 4./(3*np.pi) * (Z*alpha*alpha*lambda_e*m_ele/m_lep)**2/y*F_lep
    return dsigmady

def easy_KP_for_int(y, E_lep):
    return easy_KP(E_lep, y)

def get_rho_int_simplified_KP(rho, E_lep, y):
    return simplified_KP(E_lep,y,rho)

def get_rho_int_original_KP(rho, E_lep, y):
    return original_KP(E_lep,y,rho)

E_lep = 100e6
y = 0.99 

#ymin, ymax, rho_max = calc_params(E_lep, y)
rho_max = lim_rho(y, E_lep)
rho_x = np.linspace(-1*rho_max,rho_max,200)

fig = plt.figure()

#print(f'dsigma/dy = {easy_KP(E_lep,y)}')

def get_easy_KP_sigma(E_lep):
    ymin, ymax = lim_y(E_lep)
    return scipy.integrate.quad(easy_KP_for_int, 0.1, ymax, args=(E_lep))[0]

def get_simp_KP_dsigmady(y,E_lep): 
    rho_max = lim_rho(y,E_lep)
    return scipy.integrate.quad(get_rho_int_simplified_KP, -rho_max, rho_max, args=(E_lep,y))[0]

def get_orig_KP_dsigmady(y,E_lep):
    rho_max = lim_rho(y,E_lep)
    return scipy.integrate.quad(get_rho_int_original_KP, -rho_max, rho_max, args=(E_lep,y))[0]

def get_simp_KP_sigma(E_lep): 
    ymin, ymax = lim_y(E_lep)
    return scipy.integrate.quad(get_simp_KP_dsigmady, ymin, ymax, args=(E_lep))[0]

def get_orig_KP_sigma(E_lep):
    ymin, ymax = lim_y(E_lep)
    return scipy.integrate.quad(get_orig_KP_dsigmady, ymin, ymax, args=(E_lep))[0]

def get_simp_KP_ydsigmady(y,E_lep): 
    return y*get_simp_KP_dsigmady(y,E_lep)

def get_orig_KP_ydsigmady(y,E_lep):
    return y*get_orig_KP_dsigmady(y,E_lep)

def get_simp_KP_beta(E_lep): 
    ymin, ymax = lim_y(E_lep)
    return scipy.integrate.quad(get_simp_KP_ydsigmady, ymin, ymax, args=(E_lep))[0]

def get_orig_KP_beta(E_lep):
    ymin, ymax = lim_y(E_lep)
    return scipy.integrate.quad(get_orig_KP_ydsigmady, ymin, ymax, args=(E_lep))[0]

logE_leps = np.linspace(6,12,100)

simp_KP_sigmas = np.array([get_simp_KP_sigma(10**logE) for logE in logE_leps])
orig_KP_sigmas = np.array([get_orig_KP_sigma(10**logE) for logE in logE_leps])

#plt.plot(logE_leps, simp_KP_sigmas,label="Simplified KP")
#plt.plot(logE_leps, orig_KP_sigmas,label="Original KP")
#plt.yscale('log')
#plt.legend()
#plt.ylim(1e-7,0.1)

print(rho_x)
print(simplified_KP(E_lep, y, rho_x))
plt.plot(rho_x, simplified_KP(E_lep, y, rho_x))
plt.show()

