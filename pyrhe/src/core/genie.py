import numpy as np
from . import Base
from pyrhe.src.util.file_processing import read_env_file


class GENIE(Base):
    def __init__(
        self,
        env_file: str,
        **kwargs
    ):
         
        super().__init__(**kwargs) 

        self.num_env, self.env = read_env_file(env_file)
        self.env = self.env[:, np.newaxis]
        self.num_gen_env_bin = self.num_bin * self.num_env

        print(self.num_env)
        print(self.env)


    def shared_memory(self):
        self.shared_memory_arrays = {
                "XXz": ((self.num_bin + self.num_gen_env_bin + self.num_env, self.num_jack + 1, self.num_random_vec, self.num_indv), np.float64),
                "yXXy": ((self.num_bin + self.num_gen_env_bin + self.num_env, self.num_jack + 1), np.float64),
                "UXXz": ((self.num_bin + self.num_gen_env_bin + self.num_env, self.num_jack + 1, self.num_random_vec, self.num_indv), np.float64),
                "XXUz": ((self.num_bin + self.num_gen_env_bin + self.num_env, self.num_jack + 1, self.num_random_vec, self.num_indv), np.float64),
                "M": ((self.num_jack + 1, self.num_bin + self.num_gen_env_bin + self.num_env), np.int64)
            }
        
        self.M_last_row = self.len_bin + self.len_bin + [1] * self.num_env

    def get_num_estimates(self):
        self.num_estimates = self.num_bin + self.num_gen_env_bin + self.num_env

    def pre_compute_jackknife_bin(self, j, all_gen):
        for k, X_kj in enumerate(all_gen): 
            self.M[j][k] = self.M[self.num_jack][k] - X_kj.shape[1]
            for b in range(self.num_random_vec):
                self.M[j][k] = self.M[self.num_jack][k] - X_kj.shape[1]
                self.XXz[k, j, b, :] = self._compute_XXz(b, X_kj)
                if self.use_cov:
                    self.UXXz[k, j, b, :] = self._compute_UXXz(self.XXz[k][j][b])
                    self.XXUz[k, j, b, :] = self._compute_XXUz(b, X_kj)
            
            self.yXXy[k][j] = self._compute_yXXy(X_kj, y=self.pheno)
        
        # GxE (TODO: consider more environments)
        for k, X_kj in enumerate(all_gen): 
            k = k + self.num_bin
            self.M[j][k] = self.M[self.num_jack][k] - X_kj.shape[1]
            for b in range(self.num_random_vec):
                self.XXz[k, j, b, :] = self._compute_XXz(b, X_kj * self.env)

                if self.use_cov:
                    self.UXXz[k, j, b, :] = self._compute_UXXz(self.XXz[k][j][b])
                    self.XXUz[k, j, b, :] = self._compute_XXUz(b, X_kj)
            
            self.yXXy[k][j] = self._compute_yXXy(X_kj, y=self.pheno)
        
        # NxE
        for k in range(self.num_env):
            N = self.M[j][k]
            k = k + self.num_bin + self.num_gen_env_bin
            self.M[j][k] = 1 # No normalization
            for b in range(self.num_random_vec):
                self.XXz[k, j, b, :] = self._compute_XXz(b, self.env * np.eye(N))

                if self.use_cov:
                    self.UXXz[k, j, b, :] = self._compute_UXXz(self.XXz[k][j][b])
                    self.XXUz[k, j, b, :] = self._compute_XXUz(b, X_kj)
            
            self.yXXy[k][j] = self._compute_yXXy(X_kj, y=self.pheno)

        

    
    

