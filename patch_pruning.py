import numpy as np 

def patch_pruning(Xh, Xl):
    pvars = np.var(Xh, axis=0)
    threshold = np.percentile(pvars, 10)
    idx = pvars > threshold
    # print(pvars)
    Xh = Xh[:, idx]
    Xl = Xl[:, idx]
    return Xh, Xl