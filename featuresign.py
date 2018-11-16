import numpy as np 
from scipy import sparse

def fss_yang(lmbd, A, b):

    """
    L1QP_FeatureSign solves nonnegative quadradic programming 
    using Feature Sign. 

    min  0.5*x'*A*x+b'*x+\lambda*|x|

    [net,control]=NNQP_FeatureSign(net,A,b,control)
    """
 
    EPS = 1e-9
    x = np.zeros((A.shape[1], 1))
    # print('X =', x.shape)
    grad = np.dot(A, x) + b 
    # print('GRAD =', grad.shape)
    ma = np.amax(np.multiply(abs(grad), np.isin(x, 0).T), axis=0)
    mi = np.zeros(grad.shape[1])
    for j in range(grad.shape[1]):
        for i in range(grad.shape[0]):
            if grad[i, j] == ma[j]:
                mi[j] = i
                break
    mi = mi.astype(int)
    # print(grad[mi])
    while True:

        if np.all(grad[mi]) > lmbd + EPS:
            x[mi] = (lmbd - grad[mi]) / A[mi, mi]
        elif np.all(grad[mi]) < - lmbd - EPS:
            x[mi] = (- lmbd - grad[mi]) / A[mi, mi]
        else:
            if np.all(x == 0):
                break

        while True:
            
            a = np.where(x != 0)
            Aa = A[a, a]
            ba = b[a]
            xa = x[a]

            vect = -lmbd * np.sign(xa) - ba
            x_new = np.linalg.lstsq(Aa, vect)
            idx = np.where(x_new != 0)
            o_new = np.dot((vect[idx] / 2 + ba[idx]).T, x_new[idx]) + lmbd * np.sum(abs(x_new[idx]))

            s = np.where(np.multiply(xa, x_new) < 0) 
            if np.all(s == 0):
                x[a] = x_new
                loss = o_new
                break
            x_min = x_new
            o_min = o_new
            d = x_new - xa
            t = np.divide(d, xa)
            for zd in s.T:
                x_s = xa - d / t[zd]
                x_s[zd] = 0
                idx = np.where(x_s == 0)
                o_s = np.dot((np.dot(Aa[idx, idx], x_s[idx]) / 2 + ba[idx]).T, x_s[idx] + lmbd * np.sum(abs(x_s[idx])))
                if o_s < o_min:
                    x_min = x_s
                    o_min = o_s
            
            x[a] = x_min
            loss = o_min
        
        grad = np.dot(A, sparse.csc_matrix(x)) + b

        ma, mi = max(np.multiply(abs(grad), np.where(x == 0)))
        if ma <= lmbd + EPS:
            break
    
    return x

