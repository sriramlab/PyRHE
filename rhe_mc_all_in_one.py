import numpy as np
import pandas as pd
import argparse
from src.core.rhe import RHE
from bed_reader import open_bed
from src.util.math import *



def rhe_mc(y, X, annot, k=10, jackbin=2, cov=None, outfile=""):
    use_cov = (cov is not None)
    N, M = X.shape
    Nbin = annot.shape[1]
    
    if use_cov:
        Ncov = cov.shape[1]
    
    step_size, step_size_rem = (M // jackbin), (M % jackbin)
    ## standardize phenotype
    # y = y - np.mean(y)
    ## project y onto the residual of cov
    if use_cov:
        Q = np.linalg.inv(cov.T @ cov)
        y_res = cov @ (Q @ (cov.T @ y))
        y_new = y - y_res
        
    yy = np.dot(y.T, y).flatten()
    Nc = N
    if use_cov:
        yy = yy - np.dot(y.T, y_res).flatten()
        Nc = Nc - Ncov
        

    M_list = []
    for num_bin in range(Nbin):
        M_cur = len(np.where(annot[:, num_bin] == 1)[0])
        print(f"Number of SNPs in bin #{num_bin}: {M_cur}")
        M_list.append(M_cur)
    
    
    np.random.seed(0)
    all_zb = np.random.normal(size=(N, k))
    if use_cov:
        all_Uzb = np.zeros((N, k))
        for num_vec in range(k):
            all_Uzb[:, num_vec] = cov @ (Q @ (cov.T @ all_zb[:, num_vec]))

    
    M_mat = np.zeros((jackbin+1, Nbin))
    M_mat[jackbin] = M_list
    XXz = np.zeros((jackbin+1, Nbin, N, k))
    yXXy = np.zeros((jackbin+1, Nbin))
    
    if use_cov:
        UXXz, XXUz = np.zeros((jackbin+1, Nbin, N, k)), np.zeros((jackbin+1, Nbin, N, k))
        yVXXVy = np.zeros((jackbin+1, Nbin))
    
    
        
        
    for num_jackbin in range(jackbin):
        if num_jackbin == jackbin - 1: 
            cur_step_size = step_size+step_size_rem
        else:
            cur_step_size = step_size
        X_block = X[:, num_jackbin*step_size:(num_jackbin*step_size+cur_step_size)]
        annot_block = annot[num_jackbin*step_size:(num_jackbin*step_size+cur_step_size)]
        

        for num_bin in range(Nbin):
            X_block_k = X_block[:, np.where(annot_block[:, num_bin] == 1)[0]]
            M_mat[num_jackbin, num_bin] = X_block_k.shape[1]
            # means = X_block_k.mean(axis=0)
            # stds = 1/np.sqrt(means*(1-0.5*means))
            # Z_block_k = (X_block_k - means) * stds

            # print(Z_block_k)
            
            Z_block_k = impute_geno(X_block_k)
            XXz_k = np.zeros((N, k))
            for num_vec in range(k):
                XXz_k[:, num_vec] = Z_block_k @ (Z_block_k.T @ all_zb[:, num_vec]) ## normalize later
            XXz[num_jackbin, num_bin] = XXz_k
            if use_cov:
                for num_vec in range(k):
                    UXXz_k, XXUz_k = np.zeros((N, k)), np.zeros((N, k))
                    for num_vec in range(k):

                        UXXz_k[:, num_vec] = cov @ (Q @ (cov.T @ XXz_k[:, num_vec]))
                        XXUz_k[:, num_vec] = Z_block_k @ (Z_block_k.T @ all_Uzb[:, num_vec]) ## normalize later
                UXXz[num_jackbin, num_bin] = UXXz_k
                XXUz[num_jackbin, num_bin] = XXUz_k
            yXXy[num_jackbin, num_bin] = np.sum(np.square(Z_block_k.T @ y))
            if use_cov:
                yVXXVy[num_jackbin, num_bin] = np.sum(np.square(Z_block_k.T @ y_new))
            print(f"Reading and computing bin {num_bin} of {num_jackbin}-th is finished")

            
    XXz_sum = np.sum(XXz[:jackbin], axis=0)
    yXXy_sum = np.sum(yXXy[:jackbin], axis=0)
    M_sum = M_mat[jackbin]
    
    for num_jackbin in range(jackbin):
        XXz[num_jackbin] = XXz_sum - XXz[num_jackbin]
        yXXy[num_jackbin] = yXXy_sum - yXXy[num_jackbin]
        M_mat[num_jackbin] = M_sum - M_mat[num_jackbin]
    XXz[jackbin], yXXy[jackbin] = XXz_sum, yXXy_sum
        
    print(f"Reading and computing of all blocks are finished")

    if use_cov:
        UXXz_sum, XXUz_sum = np.sum(UXXz[:jackbin], axis=0), np.sum(XXUz[:jackbin], axis=0)
        yVXXVy_sum = np.sum(yVXXVy[:jackbin], axis=0)
        for num_jackbin in range(jackbin):
            UXXz[num_jackbin] = UXXz_sum - UXXz[num_jackbin]
            XXUz[num_jackbin] = XXUz_sum - XXUz[num_jackbin]
            yVXXVy[num_jackbin] = yVXXVy_sum - yVXXVy[num_jackbin]
        
        UXXz[jackbin], XXUz[jackbin], yVXXVy[jackbin] = UXXz_sum, XXUz_sum, yVXXVy_sum
    

    # print(UXXz_sum[0])
    

    jack = np.zeros((jackbin, Nbin+1))
    for num_jackbin in range(jackbin+1):
        XXz_block, yXXy_block = XXz[num_jackbin], yXXy[num_jackbin]
        M_block = M_mat[num_jackbin]
        if use_cov:
            UXXz_block, XXUz_block, yVXXVy_block = UXXz[num_jackbin], XXUz[num_jackbin], yVXXVy[num_jackbin]

        trKK = np.zeros((Nbin, Nbin))
            
        for bin1 in range(Nbin):
            XXz_1 = XXz_block[bin1]
            if use_cov: 
                UXXz_1 = UXXz_block[bin1]
            M_1 = M_block[bin1]
            for bin2 in range(Nbin):
                XXz_2 = XXz_block[bin2]
                M_2 = M_block[bin2]
                trkj = np.sum(np.multiply(XXz_1, XXz_2))
                if use_cov:
                    XXUz_2 = XXUz_block[bin2]
                    temp1 = np.sum(np.multiply(XXz_1, XXUz_2))
                    temp2 = np.sum(np.multiply(UXXz_1, XXUz_2))
                    trkj = trkj + temp2 - 2*temp1
                trKK[bin1, bin2] = 1/(k*M_1*M_2)*trkj ## normalize
                
        
        b_trK = np.ones((Nbin, 1)) * N
        if use_cov:
            for num_bin in range(Nbin):
                XXz_k = XXz_block[num_bin]
                M_k = M_block[num_bin]
                b_trK_res = 0
                for num_vec in range(k):
                    b_trK_res = b_trK_res + np.sum(all_Uzb[:, num_vec:(num_vec+1)].T @ XXz_k)
                b_trK[num_bin, 0] -= 1/(k*M_k)*b_trK_res ## normalize
        
        if use_cov:
            c_yKy = np.divide(yVXXVy_block, M_block) ## normalize
        else:
            c_yKy = np.divide(yXXy_block, M_block) ## normalize
        
        norm_eq_A = np.vstack([np.hstack([trKK, b_trK]), [np.append((b_trK).T, Nc)]])

        norm_eq_b = np.concatenate([c_yKy, yy])
        
        # if num_jackbin == jackbin:
        #     print(f"X_l:\n {norm_eq_A}")
        #     print(f"Y_l:\n {norm_eq_b}")

        sigma_vec = np.linalg.solve(norm_eq_A, norm_eq_b)
        
        if num_jackbin == jackbin:
            sigmas = sigma_vec
        else:
            jack[num_jackbin] = sigma_vec
        
    se = np.std(jack, axis=0) * np.sqrt(jackbin-1)
        
    if outfile:
        f = open(outfile, 'w')
        
    print("OUTPUT: ")
    
    print("Variances: ")
    f.write("OUTPUT: \n")
    f.write("Variances: \n")
    for num_bin in range(Nbin):
        print(f"Sigma^2_{num_bin}: {sigmas[num_bin]}, SE: {se[num_bin]}")
        f.write(f"Sigma^2_{num_bin}: {sigmas[num_bin]}, SE: {se[num_bin]}\n")
    print(f"Sigma_e^2: {sigmas[Nbin]}, SE: {se[Nbin]}")
    f.write(f"Sigma_e^2: {sigmas[Nbin]}, SE: {se[Nbin]}\n")
    

    her = sigmas/np.sum(sigmas)
    for num_bin in range(Nbin):
        jack[num_bin] = jack[num_bin]/np.sum(jack[num_bin])
    
    her_ldsc = np.zeros(Nbin)
    jack_ldsc = np.zeros((jackbin, Nbin))
    
    her_per_snp_in_bin = np.zeros(Nbin)
    
    for num_jackbin in range(jackbin+1):
        if num_jackbin == jackbin:
            her_per_snp_in_bin = np.divide(her[:Nbin], M_mat[num_jackbin])
        else:
            her_per_snp_in_bin = np.divide(jack[num_jackbin, :Nbin], M_mat[num_jackbin])
        her_per_snp = np.zeros(M)
        for num_snp in range(M):
            annot_at_snp = annot[num_snp]
            for num_bin in range(Nbin):
                if annot_at_snp[num_bin] == 1:
                    her_per_snp[num_snp] += her_per_snp_in_bin[num_bin]
                    
            for num_bin in range(Nbin):
                if annot_at_snp[num_bin] == 1:
                    if num_jackbin == jackbin:
                        her_ldsc[num_bin] += her_per_snp[num_snp]
                    else:
                        jack_ldsc[num_jackbin, num_bin] += her_per_snp[num_snp]
        
    se_her_ldsc = np.std(jack_ldsc, axis=0) * np.sqrt(jackbin-1)
    
    total_her = np.sum(her_ldsc)
    total_jack = np.sum(jack_ldsc, axis=1)
    se_total = np.std(total_jack) * np.sqrt(jackbin-1)
    
    for num_bin in range(Nbin):
        print(f"h^2_{num_bin}: {her_ldsc[num_bin]}, SE: {se_her_ldsc[num_bin]}")
        f.write(f"h^2_{num_bin}: {her_ldsc[num_bin]}, SE: {se_her_ldsc[num_bin]}\n")
    print(f"Total h^2: {total_her}, SE: {se_total}")
    f.write(f"Total h^2: {total_her}, SE: {se_total}\n")
    
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--geno_file', help='genotype file')
    parser.add_argument('--pheno_file', help='phenotype file')
    parser.add_argument('--cov_file', help='covariate file')
    parser.add_argument('--annot_file', help='annotation file')
    parser.add_argument('--output_file', default="output.txt", help='output file')
    args = parser.parse_args()

    geno_path = "/u/home/j/jiayini/project-sriram/RHE_project/data/simple/actual_geno_1"

    bed = open_bed(geno_path + ".bed")
    
    X = bed.read()

    rhe = RHE(
        geno_file=geno_path,
        annot_file='/u/home/j/jiayini/project-sriram/RHE_project/data/simple/annot.txt',
        cov_file='/u/home/j/jiayini/project-sriram/RHE_project/data/simple/small_covariate_file.cov',
        num_bin=8,
        num_jack=2,
        )

    print("Simulating Phenotype...")

    y, _ = rhe.simulate_pheno(sigma_list=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    print(y)

    annot = rhe.annot_matrix

    cov = rhe.cov_matrix


    rhe_mc(y=y, X=X, annot=annot, cov=cov, outfile=args.output_file)
