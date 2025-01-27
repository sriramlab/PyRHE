import numpy as np
from . import Base
from pyrhe.src.util.file_processing import read_env_file


class GENIE(Base):
    def __init__(
        self,
        env_file: str,
        model: str,
        **kwargs
    ):
         
        super().__init__(**kwargs) 

        self.num_env, self.env = read_env_file(env_file)
        self.env = self.env[:, np.newaxis]
        self.num_gen_env_bin = self.num_bin * self.num_env
        self.model = model

        self.log._log(f"Number of environments: {self.num_env}")
        self.log._log(f"Model: {self.model}")


    def shared_memory(self):
        self.get_num_estimates()
        self.shared_memory_arrays = {
                "XXz": ((self.num_estimates, self.num_jack + 1, self.num_random_vec, self.num_indv), np.float64),
                "yXXy": ((self.num_estimates, self.num_jack + 1), np.float64),
                "UXXz": ((self.num_estimates, self.num_jack + 1, self.num_random_vec, self.num_indv), np.float64),
                "XXUz": ((self.num_estimates, self.num_jack + 1, self.num_random_vec, self.num_indv), np.float64),
                "M": ((self.num_jack + 1, self.num_estimates), np.int64)
            }

        if self.model == "G":
            self.M_last_row = self.len_bin
        elif self.model == "G+GxE":
            self.M_last_row = np.concatenate((self.len_bin, self.len_bin * self.num_env))
        elif self.model == "G+GxE+NxE":
            self.M_last_row = np.concatenate((self.len_bin, self.len_bin * self.num_env, [1] * self.num_env))
        else:
            raise ValueError("Unsupported GENIE model type")


    def get_num_estimates(self):
        if self.model == "G":
            self.num_estimates = self.num_bin
        elif self.model == "G+GxE":
            self.num_estimates = self.num_bin + self.num_gen_env_bin
        elif self.model == "G+GxE+NxE":
            self.num_estimates = self.num_bin + self.num_gen_env_bin + self.num_env
        else:
            raise ValueError("Unsupported GENIE model type")

    def pre_compute_jackknife_bin(self, j, all_gen):
        for k, X_kj in enumerate(all_gen): 
            self.M[j][k] = self.M[self.num_jack][k] - X_kj.shape[1]
            print(f"k = {k}, M = {self.M[j][k]}")
            for b in range(self.num_random_vec):
                self.XXz[k, j, b, :] = self._compute_XXz(b, X_kj)
                if self.use_cov:
                    self.UXXz[k, j, b, :] = self._compute_UXXz(self.XXz[k][j][b])
                    self.XXUz[k, j, b, :] = self._compute_XXUz(b, X_kj)
            
            self.yXXy[k][j] = self._compute_yXXy(X_kj, y=self.pheno)
        
        
        # GxE
        if self.model == "G+GxE" or self.model == "G+GxE+NxE":
            for e in range(self.num_env):
                for k, X_kj in enumerate(all_gen): 
                    k = (e + 1) * k + self.num_bin
                    self.M[j][k] = self.M[self.num_jack][k] - X_kj.shape[1]
                    X_kj = X_kj * (self.env[:, e]).reshape(-1, 1)
                    for b in range(self.num_random_vec):
                        self.XXz[k, j, b, :] = self._compute_XXz(b, X_kj)

                        if self.use_cov:
                            self.UXXz[k, j, b, :] = self._compute_UXXz(self.XXz[k][j][b])
                            self.XXUz[k, j, b, :] = self._compute_XXUz(b, X_kj)
                    
                    self.yXXy[k][j] = self._compute_yXXy(X_kj, y=self.pheno)
    
                
        # NxE
        if self.model == "G+GxE+NxE":
            for e in range(self.num_env):
                k = e + self.num_bin + self.num_gen_env_bin
                self.M[j][k] = 1
        
    
    def run(self, method):
        sigma_est_jackknife, sigma_ests_total = self.estimate(method=method)
        sig_errs = self.estimate_error(sigma_est_jackknife)

        self.log._log("Variance components: ")


        for i, est in enumerate(sigma_ests_total):
            if self.model == "G":
                self.log._log(f"Sigma^2_g[{i}] : {est}  SE : {sig_errs[i]}")

            elif self.model == "G+GxE":
                if i < self.num_bin:
                    self.log._log(f"Sigma^2_g[{i}] : {est}  SE : {sig_errs[i]}")
                else:
                    self.log._log(f"Sigma^2_gxe[{i - self.num_bin}] : {est}  SE : {sig_errs[i]}")

            elif self.model == "G+GxE+NxE":
                if i < self.num_bin:
                    self.log._log(f"Sigma^2_g[{i}] : {est}  SE : {sig_errs[i]}")
                elif i < self.num_bin + self.num_gen_env_bin:
                    self.log._log(f"Sigma^2_gxe[{i - self.num_bin}] : {est}  SE : {sig_errs[i]}")
                elif i < self.num_bin + self.num_gen_env_bin + self.num_env:
                    self.log._log(f"Sigma^2_nxe[{i - self.num_bin - self.num_gen_env_bin}] : {est}  SE : {sig_errs[i]}")

        self.log._log(f"Sigma^2_e : {sigma_ests_total[-1]}  SE : {sig_errs[-1]}")

        return {
            "sigma_ests_total": sigma_ests_total,
            "sig_errs": sig_errs,
        }