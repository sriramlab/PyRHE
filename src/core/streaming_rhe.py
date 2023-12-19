import time
import torch
from src.core.rhe import RHE
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
        device: torch.device = torch.device("cpu"),
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
            device=device,
            verbose=verbose,
            seed=seed
        )
    

    def pre_compute(self):
        """
        Only Compute the Sum
        """
        self.XXz_per_jack = self.np.zeros((self.num_bin, 2, self.num_random_vec, self.num_indv), dtype=self.np.float64)
        self.yXXy_per_jack = self.np.zeros((self.num_bin, 2), dtype=self.np.float64)
        self.UXXz_per_jack = self.np.zeros((self.num_bin, 2, self.num_random_vec, self.num_indv), dtype=self.np.float64)
        self.XXUz_per_jack = self.np.zeros((self.num_bin, 2, self.num_random_vec, self.num_indv), dtype=self.np.float64)

        for j in range(self.num_jack):
            print(f"Precompute (pass 1) for jackknife sample {j}")
            start_whole = time.time()
            subsample, sub_annot = self._get_jacknife_subsample(j)
            subsample = self.impute_geno(subsample, simulate_geno=True)
            all_gen = self.partition_bins(subsample, sub_annot)
                    
            for k, X_kj in enumerate(all_gen):
                self.M[j][k] = self.M[self.num_jack][k] - X_kj.shape[1] # store the dimension with the corresponding block
                for b in range(self.num_random_vec):
                    XXz_kjb = self._compute_XXz(b, X_kj)
                    self.XXz_per_jack[k][0][b] += XXz_kjb

                    if self.use_cov:
                        UXXz_kjb = self._compute_UXXz(XXz_kjb)
                        XXUz_kjb = self._compute_XXUz(b, X_kj)
                        self.UXXz_per_jack[k][0][b] += UXXz_kjb
                        self.XXUz_per_jack[k][0][b] += XXUz_kjb
                
            
                yXXy_kj = self._compute_yXXy(X_kj, y=self.pheno)
                self.yXXy_per_jack[k][0] += yXXy_kj[0][0]

                del X_kj
            
            end_whole = time.time()
            print(f"jackknife precompute (pass 1) total time: {end_whole-start_whole}")

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
            print(f"Precompute (pass 2) for jackknife sample {j}")
            start_whole = time.time()
            if j != self.num_jack:
                subsample, sub_annot = self._get_jacknife_subsample(j)
                subsample = self.impute_geno(subsample, simulate_geno=True)
                all_gen = self.partition_bins(subsample, sub_annot)

            # calculate the stats for one jacknife subsample
            for k in range(self.num_bin):
                X_k = all_gen[k] if j != self.num_jack else 0
                for b in range (self.num_random_vec):
                    XXz_kb = self._compute_XXz(b, X_k) if j != self.num_jack else 0
                    if self.use_cov:
                        UXXz_kb = self._compute_UXXz(XXz_kb) if j != self.num_jack else 0
                        self.UXXz_per_jack[k][1][b] = self.UXXz_per_jack[k][0][b] - UXXz_kb
                        XXUz_kb = self._compute_XXUz(b, X_k) if j != self.num_jack else 0
                        self.XXUz_per_jack[k][1][b] = self.XXUz_per_jack[k][0][b] - XXUz_kb
                    self.XXz_per_jack[k][1][b] = self.XXz_per_jack[k][0][b] - XXz_kb
                
                yXXy_k = (self._compute_yXXy(X_k, y=self.pheno))[0][0] if j != self.num_jack else 0
                self.yXXy_per_jack[k][1] = self.yXXy_per_jack[k][0] - yXXy_k


            print(f"Estimate for jackknife sample {j}")
            T = self.np.zeros((self.num_bin+1, self.num_bin+1))
            q = self.np.zeros((self.num_bin+1, 1))

            for k_k in range(self.num_bin):
                for k_l in range(self.num_bin): # TODO: optimize 
                    M_k = self.M[j][k_k]
                    M_l = self.M[j][k_l]
                    B1 = self.XXz_per_jack[k_k][1]
                    B2 = self.XXz_per_jack[k_l][1]
                    T[k_k, k_l] += self.np.sum(B1 * B2)

                    if self.use_cov:
                        h1 = self.cov_matrix.T @ B1.T
                        h2 = self.Q @ h1
                        h3 = self.cov_matrix @ h2
                        trkij_res1 = self.np.sum(h3.T * B2)

                        B1 = self.XXUz_per_jack[k_k][1]
                        B2 = self.UXXz_per_jack[k_l][1]
                        trkij_res2 = self.np.sum(B1 * B2)
                    
                        T[k_k, k_l] += (trkij_res2 - 2 * trkij_res1)


                    T[k_k, k_l] /= (self.num_random_vec)
                    T[k_k, k_l] =  T[k_k, k_l] / (M_k * M_l) if (M_k * M_l) != 0 else 0


            for k in range(self.num_bin):
                M_k = self.M[j][k]

                if not self.use_cov:
                    T[k, self.num_bin] = self.num_indv
                    T[self.num_bin, k] = self.num_indv

                else:
                    B1 = self.XXz_per_jack[k][1]
                    C1 = self.all_Uzb
                    tk_res = self.np.sum(B1 * C1.T)
                    tk_res = 1/(self.num_random_vec * M_k) * tk_res

                    T[k, self.num_bin] = self.num_indv - tk_res
                    T[self.num_bin, k] = self.num_indv - tk_res

                q[k] = self.yXXy_per_jack[k][1] / M_k if M_k != 0 else 0
    
            
            
            T[self.num_bin, self.num_bin] = self.num_indv if not self.use_cov else self.num_indv - self.cov_matrix.shape[1]
            pheno = self.pheno if not self.use_cov else self.regress_pheno(self.cov_matrix, self.pheno)
            q[self.num_bin] = pheno.T @ pheno 
        

            if method == "lstsq":
                sigma_est = self.solve_linear_equation(T,q)
                sigma_est = self.np.ravel(sigma_est).tolist()
                sigma_ests.append(sigma_est)
            elif method == "QR":
                sigma_est = self.solve_linear_qr(T,q)
                sigma_est = self.np.ravel(sigma_est).tolist()
                sigma_ests.append(sigma_est)
            else:
                raise ValueError("Unsupported method for solving linear equation")
            

            end_whole = time.time()
            print(f"estimate time for jackknife subsample: {end_whole - start_whole}")
            
        sigma_ests = self.np.array(sigma_ests)

        sigma_est_jackknife, sigma_ests_total = sigma_ests[:-1, :], sigma_ests[-1, :]
            
        return sigma_est_jackknife, sigma_ests_total

