import numpy as np


def _simulate_geno_from_random(p_j):
    rval = np.random.random()
    dist_pj = [(1-p_j)*(1-p_j), 2*p_j*(1-p_j), p_j*p_j]
    
    if rval < dist_pj[0]:
        return 0
    elif rval >= dist_pj[0] and rval < (dist_pj[0] + dist_pj[1]):
        return 1
    else:
        return 2
    

def impute_geno(X):
    N = X.shape[0]
    M = X.shape[1]
    X_imp = X.copy()

    for m in range(M):
        
        observed_sum = 0
        observed_ct = 0
        for n in range(N):
            if not np.isnan(X[n, m]):
                observed_ct += 1
                observed_sum += X[n, m]
        
        observed_sum = (observed_sum  / observed_ct)* 0.5

        for j in range(N):
            if np.isnan(X[j,m]):
                X_imp[j, m] = _simulate_geno_from_random(observed_sum)
                
    # standardize
    X_imp = (X_imp-np.mean(X_imp, axis=0))/np.std(X_imp, axis=0)

    return X_imp