import numpy as np
from . import StreamingBase


class StreamingRHE(StreamingBase):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs) 

    def shared_memory(self):
        self.shared_memory_arrays = {
            "XXz": ((self.num_bin, self.num_workers, self.num_random_vec, self.num_indv), np.float64),
            "yXXy": ((self.num_bin, self.num_workers), np.float64),
            "UXXz": ((self.num_bin, self.num_workers, self.num_random_vec, self.num_indv), np.float64),
            "XXUz": ((self.num_bin, self.num_workers, self.num_random_vec, self.num_indv), np.float64),
        }
    
    def pre_compute_jackknife_bin(self, j, k, X_kj, worker_num):
        self.M[j][k] = self.M[self.num_jack][k] - X_kj.shape[1]
        for b in range(self.num_random_vec):
            XXz_kjb = self._compute_XXz(b, X_kj)
            self.XXz[k][worker_num][b] += XXz_kjb

            if self.use_cov:
                UXXz_kjb = self._compute_UXXz(XXz_kjb)
                XXUz_kjb = self._compute_XXUz(b, X_kj)
                self.UXXz[k][worker_num][b] += UXXz_kjb
                self.XXUz[k][worker_num][b] += XXUz_kjb
                
        yXXy_kj = self._compute_yXXy(X_kj, y=self.pheno)
        self.yXXy[k][worker_num] += yXXy_kj[0][0]
    

        del X_kj

    def aggregate(self):
        if self.multiprocessing:
            self.XXz_per_jack = np.zeros((self.num_bin, 2, self.num_random_vec, self.num_indv), dtype=np.float64)
            self.yXXy_per_jack = np.zeros((self.num_bin, 2), dtype=np.float64)
            self.UXXz_per_jack = np.zeros((self.num_bin, 2, self.num_random_vec, self.num_indv), dtype=np.float64) if self.use_cov else None
            self.XXUz_per_jack = np.zeros((self.num_bin, 2, self.num_random_vec, self.num_indv), dtype=np.float64) if self.use_cov else None
            for k in range(self.num_bin):
                self.yXXy_per_jack[k][0] = np.sum(self.yXXy[k, :])
                for b in range(self.num_random_vec):
                    self.XXz_per_jack[k, 0, b, :] = np.sum(self.XXz[k, :, b, :], axis=0)
                    if self.UXXz_per_jack is not None: 
                        self.UXXz_per_jack[k, 0, b, :] = np.sum(self.UXXz[k, :, b, :], axis=0) 
                    if self.XXUz_per_jack is not None:
                        self.XXUz_per_jack[k, 0, b, :] = np.sum(self.XXUz[k, 0, b, :], axis=0)
            
            self.XXz = self.XXz_per_jack
            self.yXXy = self.yXXy_per_jack 
            self.UXXz = self.UXXz_per_jack
            self.XXUz = self.XXUz_per_jack

    def pre_compute_jackknife_bin_pass_2(self, j, k, X_kj):
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