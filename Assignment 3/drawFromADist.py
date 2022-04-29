import numpy as np

def drawFromADist(p):

    if np.sum(p) == 0 :
        p = 0.05 * np.ones((1,len(p)))

    p = p / (np.sum(p))
    c = np.cumsum(p)

    idx = np.where((np.random.uniform()- c)<0)
    ##print(idx)
    sample = np.min(idx)
    ##print(p.shape)
    out = np.zeros(len(p))
    ##print(out)
    out[sample] = 1

    return out


