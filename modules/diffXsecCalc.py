import numpy as np
import KPmodel_libs as kp

from tqdm import tqdm 
import sys  
import click

@click.command()
@click.option('--accuracy','-a',type=int,default=100)
@click.option('--edivision','-e',type=int,default=140)
@click.option('--mass','-m',type=float,default=150)
@click.option('--oparticle',type=str,default='electron')
def main(accuracy,edivision,mass,oparticle):
    logYs = []
    Ys = []
    DSigmaDys = []
    Sigmas = []

    if oparticle == 'electron':
        m_ele = 0.511e-3
        ofn = ''
    elif oparticle == 'muon':
        m_ele = 105.658e-3
        ofn = '_muon'
    elif oparticle == 'tau':
        m_ele = 1.7768
        ofn = '_tau'
    else:
        m_ele = 0.511e-3
        ofn = ''
    kpmodel = kp.XsecCalculator(m_lep=mass, m_ele=m_ele)
    logEs = np.linspace(5,12,edivision)
    for i in tqdm(range(edivision)):
        ymin, ymax = kpmodel.lim_y(10**logEs[i])
        logYArray = np.linspace(np.log10(ymin),np.log10(ymax),accuracy)
        YArray = 10**logYArray
        if YArray[0] < ymin: 
            YArray[0] = ymin
        if YArray[len(YArray)-1] > ymax:
            YArray[len(YArray)-1] = ymax
        DSigmaDy = np.array([kpmodel.getDSigmaDy(y,10**logEs[i],method='quad') for y in YArray])

        sigma = 0
        for j in range(len(YArray)-1):
            dx = np.abs(YArray[j+1]-YArray[j])
            sigma += 0.5 * dx * (DSigmaDy[j+1]+DSigmaDy[j])

        logYs.append(logYArray)
        Ys.append(YArray)
        DSigmaDys.append(DSigmaDy)
        Sigmas.append(sigma)
        kpmodel.setRombergDivmax(100)
        tqdm.write(f"{10**logEs[i]:.3e} GeV: {sigma:.3e}")

    if mass == int(mass):
        np.savez(f'data/calc_{int(mass)}GeV{ofn}.npz',logY=np.array(logYs),Y=np.array(Ys),DsDy=np.array(DSigmaDys),Sigma=np.array(Sigmas),logE=logEs)
    else:
        np.savez(f'data/calc_{mass:.3f}GeV{ofn}.npz',logY=np.array(logYs),Y=np.array(Ys),DsDy=np.array(DSigmaDys),Sigma=np.array(Sigmas),logE=logEs)

    return 

if __name__ == '__main__':
    main()

