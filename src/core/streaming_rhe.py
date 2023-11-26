import numpy as np
from src.core.rhe import RHE
from src.util.math import *
from typing import List, Tuple

class StreamingRHE(RHE):
    def __init__(
        self,
        geno_file: str,
        annot_file: str = None,
        pheno_file: str = None,
        cov_file: str = None,
        num_bin: int = 8,
        num_jack: int = 1,
        num_random_vec: int = 10,
        verbose: bool = True,
        seed: int = 0,
    ):
        super().__init__(
            geno_file=geno_file,
            annot_file=annot_file,
            pheno_file=pheno_file,
            cov_file=cov_file,
            num_bin=num_bin,
            num_jack=num_jack,
            num_random_vec=num_random_vec,
            verbose=verbose,
            seed=seed
        )

        self.XXz = None
        self.yXXy = None
        self.UXXz = None
        self.XXUz = None
        
        self.XXz_sum = np.zeros((self.num_bin, 1, self.num_random_vec, self.num_indv))
        self.yXXy_sum = np.zeros((self.num_bin, 1))
        self.UXXz_sum = np.zeros((self.num_bin, 1, self.num_random_vec, self.num_indv))
        self.XXUz_sum = np.zeros((self.num_bin, 1, self.num_random_vec, self.num_indv))


    def pre_compute(self):
        """
        Only Compute the Sum
        """
    
        for j in range(self.num_jack):
            print(f"Precompute for jackknife sample {j}")
            subsample, sub_annot = self._get_jacknife_subsample(j)
            subsample = impute_geno(subsample, simulate_geno=True)
            all_gen = self.partition_bins(subsample, sub_annot)
                    
            for k, X_kj in enumerate(all_gen):
                self.M[j][k] = self.M[self.num_jack][k] - X_kj.shape[1] # store the dimension with the corresponding block
                for b in range(self.num_random_vec):
                    XXz_kjb = self._compute_XXz(b, X_kj)
                    self.XXz_sum[k][0][b] += XXz_kjb
                    
                if self.use_cov:
                    for b in range(self.num_random_vec):
                        UXXz_kjb = self._compute_UXXz(b, X_kj, XXz_kjb)
                        XXUz_kjb = self._compute_XXUz(b, X_kj)
                        self.UXXz_sum[k][0][b] += UXXz_kjb
                        self.XXUz_sum[k][0][b] += XXUz_kjb

                yXXy_kj = self._compute_yXXy(X_kj, y=self.pheno)
                self.yXXy_sum[k][0] += yXXy_kj

                del X_kj
        
    def estimate(self, method: str = "QR") -> Tuple[List[List], List]:
        """
        Actual RHE estimation for sigma^2
        Returns: 
        sigma_est_jackknife: 
            rows: jackknife sample
            cols: sigma^2 for one jackknife sample [sigma_1^2 sigma_2^2 ... sigma_e^2]
            e.g.,
            sigma_est_jackknife[0][0]: sigma_2 estimate for the 1st bin in the 1st jacknife sample
            sigma_est_jackknife[0][1]: sigma_2 estimate for the 1st bin in the 2st jacknife sample

        sigma_ests_total
            sigma^2 for the whole genotype matrix [sigma_1^2 sigma_2^2 ... sigma_e^2]

        """
        sigma_ests = []

        for j in range(self.num_jack + 1):
            print(f"Estimate for jackknife sample {j}")

            if j != self.num_jack:
                subsample, sub_annot = self._get_jacknife_subsample(j)
                subsample = impute_geno(subsample, simulate_geno=True)
                all_gen = self.partition_bins(subsample, sub_annot)

            T = np.zeros((self.num_bin+1, self.num_bin+1))
            q = np.zeros((self.num_bin+1, 1))

            for k_k, X_kkj in enumerate(all_gen):
                for k_l, X_klj in enumerate(all_gen):
                    M_k = self.M[j][k_k]
                    M_l = self.M[j][k_l]

                    B1 = np.zeros((self.num_random_vec, self.num_indv))
                    B2 = np.zeros((self.num_random_vec, self.num_indv))
                    for b in range(self.num_random_vec):
                        if j != self.num_jack:
                            XXz_kkjb = self._compute_XXz(b, X_kkj)
                            jack_XXz_kkjb = self.XXz_sum[k_k][0][b] - XXz_kkjb
                            B1[b, :] = jack_XXz_kkjb
                        else:
                            B1[b, :] = self.XXz_sum[k_k][0][b]
                        
                        if j != self.num_jack:
                            XXz_kljb = self._compute_XXz(b, X_klj)
                            jack_XXz_kljb = self.XXz_sum[k_l][0][b] - XXz_kljb
                            B2[b, :] = jack_XXz_kljb
                        else:
                            B2[b, :] = self.XXz_sum[k_l][0][b]          
                    
                    if not self.use_cov:
                        T[k_k, k_l] += np.sum(B1 * B2)

                    else:
                        h1 = self.cov_matrix.T @ B1.T
                        h2 = self.Q @ h1
                        h3 = self.cov_matrix @ h2
                        trkij_res1 = np.sum(h3.T * B2)


                        B1 = np.zeros((self.num_random_vec, self.num_indv))
                        B2 = np.zeros((self.num_random_vec, self.num_indv))

                        for b in range(self.num_random_vec):
                            if j != self.num_jack:
                                XXUz_kkjb = self._compute_XXUz(b, X_kkj)
                                jack_XXUz_kkjb = self.XXUz_sum[k_k][0][b] - XXUz_kkjb
                                B1[b, :] = jack_XXUz_kkjb
                            else:
                                B1[b, :] = self.XXUz_sum[k_k][0][b]

                            if j != self.num_jack:
                                UXXz_kljb = self._compute_UXXz(b, X_klj)
                                jack_UXXz_kljb = self.XXUz_sum[k_k][0][b] - UXXz_kljb
                                B2[b, :] = jack_UXXz_kljb 
                            else:
                                B2[b, :] = self.XXUz_sum[k_k][0][b]

                        trkij_res2 = np.sum(B1 * B2)
                        T[k_k, k_l] += (trkij_res2 - 2 * trkij_res1)


                    T[k_k, k_l] /= (self.num_random_vec)
                    T[k_k, k_l] =  T[k_k, k_l] / (M_k * M_l) if (M_k * M_l) != 0 else 0
            

            for k, X_kj in enumerate(all_gen):
                M_k = self.M[j][k]

                if not self.use_cov:
                    T[k, self.num_bin] = self.num_indv
                    T[self.num_bin, k] = self.num_indv

                else:
                    B1 = np.zeros((self.num_random_vec, self.num_indv))
                    for b in range(self.num_random_vec):
                        if j != self.num_jack:
                            XXz_kjb = self._compute_XXz(b, X_kj)
                            jack_XXz_kkjb = self.XXz_sum[k][0][b] - XXz_kjb
                            B1[:, b] = jack_XXz_kkjb 
                        else:
                            B1[:, b] = self.XXz_sum[k][0][b]

                    C1 = self.all_Uzb
                    tk_res = np.sum(B1 * C1.T)
                    tk_res = 1/(self.num_random_vec * M_k) * tk_res

                    T[k, self.num_bin] = self.num_indv - tk_res
                    T[self.num_bin, k] = self.num_indv - tk_res

                if j != self.num_jack:
                    yXXy_kj = self._compute_yXXy(X_kj, y=self.pheno)
                    jack_yXXy_kj = self.yXXy_sum[k][0] - yXXy_kj
                    q[k] = jack_yXXy_kj / M_k if M_k != 0 else 0
                else:
                    q[k] = self.yXXy_sum[k][0] / M_k if M_k != 0 else 0
            
            
            T[self.num_bin, self.num_bin] = self.num_indv if not self.use_cov else self.num_indv - self.cov_matrix.shape[1]
            pheno = self.pheno if not self.use_cov else self.regress_pheno(self.cov_matrix, self.pheno)
            q[self.num_bin] = pheno.T @ pheno 

            if method == "QR":
                sigma_est = solve_linear_equation(T,q)
                sigma_est = np.ravel(sigma_est).tolist()
                sigma_ests.append(sigma_est)
            elif method == "lstsq":
                sigma_est = solve_linear_qr(T,q)
                sigma_est = np.ravel(sigma_est).tolist()
                sigma_ests.append(sigma_est)
            else:
                raise ValueError("Unsupported method for solving linear equation")
            
        sigma_ests = np.array(sigma_ests)

        sigma_est_jackknife, sigma_ests_total = sigma_ests[:-1, :], sigma_ests[-1, :]
            
        return sigma_est_jackknife, sigma_ests_total