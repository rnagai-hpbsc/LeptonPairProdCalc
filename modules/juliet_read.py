import javaobj
import numpy as np

def getIntMtx(javafile):
    print(f'Read {javafile}...')
    with open(javafile,"rb") as f:
        sobj = f.read()

    intMtx = javaobj.loads(sobj)
    return intMtx

