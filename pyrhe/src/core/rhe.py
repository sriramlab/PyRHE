import numpy as np
from . import Base



class RHE(Base):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs) 

    def shared_memory(self):
        self.shared_memory_arrays = {
                "XXz": ((self.num_bin, self.num_jack + 1, self.num_random_vec, self.num_indv), np.float64),
                "yXXy": ((self.num_bin, self.num_jack + 1), np.float64),
                "UXXz": ((self.num_bin, self.num_jack + 1, self.num_random_vec, self.num_indv), np.float64),
                "XXUz": ((self.num_bin, self.num_jack + 1, self.num_random_vec, self.num_indv), np.float64),
            }

    def pre_compute_jackknife_bin(self, j, k, X_kj):
        self.M[j][k] = self.M[self.num_jack][k] - X_kj.shape[1]
        for b in range(self.num_random_vec):
            self.XXz[k, j, b, :] = self._compute_XXz(b, X_kj)
            if self.use_cov:
                self.UXXz[k, j, b, :] = self._compute_UXXz(self.XXz[k][j][b])
                self.XXUz[k, j, b, :] = self._compute_XXUz(b, X_kj)
        self.yXXy[k][j] = self._compute_yXXy(X_kj, y=self.pheno)
    
    def aggregate(self):
        for k in range(self.num_bin):
            for j in range(self.num_jack):
                for b in range (self.num_random_vec):
                    self.XXz[k][self.num_jack][b] += self.XXz[k][j][b]
                self.yXXy[k][self.num_jack] += self.yXXy[k][j]
            
            for j in range(self.num_jack):
                for b in range (self.num_random_vec):
                    self.XXz[k][j][b] = self.XXz[k][self.num_jack][b] - self.XXz[k][j][b]
                self.yXXy[k][j] = self.yXXy[k][self.num_jack] - self.yXXy[k][j]
        
        if self.use_cov:
            for k in range(self.num_bin):
                for j in range(self.num_jack):
                    for b in range (self.num_random_vec):
                        self.UXXz[k][self.num_jack][b] += self.UXXz[k][j][b]
                        self.XXUz[k][self.num_jack][b] += self.XXUz[k][j][b]
                                        
                for j in range(self.num_jack):
                    for b in range (self.num_random_vec):
                        self.UXXz[k][j][b] = self.UXXz[k][self.num_jack][b] - self.UXXz[k][j][b]
                        self.XXUz[k][j][b] = self.XXUz[k][self.num_jack][b] - self.XXUz[k][j][b]
