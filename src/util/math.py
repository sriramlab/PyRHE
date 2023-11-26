import scipy
import numpy as np

def _simulate_geno_from_random(p_j):
    np.random.seed(0)
    rval = np.random.random()
    dist_pj = [(1-p_j)*(1-p_j), 2*p_j*(1-p_j), p_j*p_j]
    
    if rval < dist_pj[0]:
        return 0
    elif rval >= dist_pj[0] and rval < (dist_pj[0] + dist_pj[1]):
        return 1
    else:
        return 2
    

def impute_geno(X, simulate_geno: bool = True):
    N = X.shape[0]
    M = X.shape[1]
    X_imp = X
    if simulate_geno:
        for m in range(M):
            observed_mask = ~np.isnan(X[:, m])
            observed_values = X[observed_mask, m]
            
            observed_ct = observed_values.size
            observed_sum = np.sum(observed_values)
            observed_mean = 0.5 * observed_sum / observed_ct
            
            missing_mask = np.isnan(X[:, m])
            X_imp[missing_mask, m] = _simulate_geno_from_random(observed_mean)
    
    means = np.mean(X_imp, axis=0)
    stds = 1/np.sqrt(means*(1-0.5*means))
    X_imp = (X_imp - means) * stds
    return X_imp

def solve_linear_equation(X, y):
        '''
        Solve least square
        '''
        sigma = np.linalg.lstsq(X, y, rcond=None)[0]
        return sigma


def solve_linear_qr(X, y):
        '''
        Solve least square using QR decomposition
        '''
        Q, R = scipy.linalg.qr(X)
        sigma = scipy.linalg.solve_triangular(R, np.dot(Q.T, y))
        return sigma

