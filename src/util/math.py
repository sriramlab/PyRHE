import scipy
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
    

def impute_geno(X, simulate_geno: bool = False):
    N = X.shape[0]
    M = X.shape[1]
    X_imp = X
    if simulate_geno:
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

    means = np.mean(X_imp, axis=0)
    stds = 1/np.sqrt(means*(1-0.5*means))
    X_imp = (X_imp - means) * stds
    return X_imp


def compute_XXz(bin, all_zb, mailman):
    # TODO: mailman == True
    num_snp = bin.index
    gen = bin.gen

    res = gen @ all_zb
    Nz = np.shape(all_zb)[1]

    means = np.mean(gen, axis=1)

    # stds = np.std(gen, axis=1)
    stds = 1 / np.sqrt(means * (1 - 0.5 * means)) # a sample’s value is a binomial random variable with n = 2

    zb_sum = np.sum(all_zb, axis=0)

    zb_sum = np.array(zb_sum).reshape(-1, 1)

    for j in range(num_snp):
        for k in range(Nz):
            res[j, k] = res[j, k] * stds[j]

    inter = np.array(means * stds).reshape(-1, 1)
    resid =  inter @ zb_sum.T
    inter_zb = res - resid

    for k in range(Nz):
        for j in range(num_snp):
            inter_zb[j, k] = inter_zb[j, k] * stds[j]

    new_zb = inter_zb.T
    new_res = new_zb @ gen

    
    new_resid = new_zb @ np.array(means).reshape(-1, 1)
    new_resid = new_resid * np.ones((1, np.shape(gen)[1]))

    temp = new_res - new_resid
    return temp.T


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

