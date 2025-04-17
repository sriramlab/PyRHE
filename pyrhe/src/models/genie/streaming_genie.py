from pyrhe.src.base import StreamingBase
from pyrhe.src.models.genie import GENIE
from pyrhe.src.util.mat_mul import *


class StreamingGENIE(GENIE, StreamingBase):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs) 

    def pre_compute_jackknife_bin(self, j, all_gen, worker_num):
        # G
        for k, X_kj in enumerate(all_gen): 
            X_kj = self.standardize_geno(X_kj)
            self.M[j][k] = self.M[self.num_jack][k] - X_kj.shape[1]
            for b in range(self.num_random_vec):
                self.XXz[k][worker_num][b] += self._compute_XXz(b, X_kj)

                if self.use_cov:
                    self.UXXz[k][worker_num][b] += self._compute_UXXz(self.XXz[k][worker_num][b])
                    self.XXUz[k][worker_num][b] += self._compute_XXUz(b, X_kj)
            
            yXXy_kj = self._compute_yXXy(X_kj, y=self.pheno)
            self.yXXy[k][worker_num] += yXXy_kj[0][0]
                
        
        # GxE
        if self.genie_model == "G+GxE" or self.genie_model == "G+GxE+NxE":
            for e in range(self.num_env):
                for k, X_kj in enumerate(all_gen): 
                    X_kj = self.standardize_geno(X_kj)
                    k_gxe = (e + 1) * k + self.num_bin
                    self.M[j][k_gxe] = self.M[self.num_jack][k_gxe] - X_kj.shape[1]
                    X_kj_gxe = elem_mul(X_kj, self.env[:, e].reshape(-1, 1), device=self.device)  # Avoid modifying X_kj
                    for b in range(self.num_random_vec):
                        self.XXz[k_gxe][worker_num][b] += self._compute_XXz(b, X_kj_gxe)

                        if self.use_cov:
                            self.UXXz[k_gxe][worker_num][b] += self._compute_UXXz(self.XXz[k_gxe][worker_num][b])
                            self.XXUz[k_gxe][worker_num][b] += self._compute_XXUz(b, X_kj_gxe)
                    
                    yXXy_kj = self._compute_yXXy(X_kj_gxe, y=self.pheno)
                    self.yXXy[k_gxe][worker_num] += yXXy_kj[0][0]
                
        # NxE
        if self.genie_model == "G+GxE+NxE":
            for e in range(self.num_env):
                k = e + self.num_bin + self.num_gen_env_bin
                self.M[j][k] = 1

    def pre_compute_jackknife_bin_pass_2(self, j, all_gen):
        # G
        for k, X_kj in enumerate(all_gen): 
            X_kj = self.standardize_geno(X_kj)
            for b in range (self.num_random_vec):
                XXz_kb = self._compute_XXz(b, X_kj) if j != self.num_jack else 0
                if self.use_cov:
                    UXXz_kb = self._compute_UXXz(XXz_kb) if j != self.num_jack else 0
                    self.UXXz[k][1][b] = self.UXXz[k][0][b] - UXXz_kb
                    XXUz_kb = self._compute_XXUz(b, X_kj) if j != self.num_jack else 0
                    self.XXUz[k][1][b] = self.XXUz[k][0][b] - XXUz_kb
                self.XXz[k][1][b] = self.XXz[k][0][b] - XXz_kb
            
            yXXy_k = (self._compute_yXXy(X_kj, y=self.pheno))[0][0] if j != self.num_jack else 0
            self.yXXy[k][1] = self.yXXy[k][0] - yXXy_k
        
        # GxE
        if self.genie_model == "G+GxE" or self.genie_model == "G+GxE+NxE":
            for e in range(self.num_env):
                for k, X_kj in enumerate(all_gen): 
                    X_kj = self.standardize_geno(X_kj)
                    k_gxe = (e + 1) * k + self.num_bin
                    X_kj_gxe = elem_mul(X_kj, self.env[:, e].reshape(-1, 1), device=self.device)
                    for b in range(self.num_random_vec):
                        XXz_kb = self._compute_XXz(b, X_kj_gxe) if j != self.num_jack else 0
                        if self.use_cov:
                            UXXz_kb = self._compute_UXXz(XXz_kb) if j != self.num_jack else 0
                            self.UXXz[k_gxe][1][b] = self.UXXz[k_gxe][0][b] - UXXz_kb
                            XXUz_kb = self._compute_XXUz(b, X_kj_gxe) if j != self.num_jack else 0
                            self.XXUz[k_gxe][1][b] = self.XXUz[k_gxe][0][b] - XXUz_kb
                        self.XXz[k_gxe][1][b] = self.XXz[k_gxe][0][b] - XXz_kb
                    yXXy_k = (self._compute_yXXy(X_kj_gxe, y=self.pheno))[0][0] if j != self.num_jack else 0
                    self.yXXy[k_gxe][1] = self.yXXy[k_gxe][0] - yXXy_k