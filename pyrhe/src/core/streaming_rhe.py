import numpy as np
from . import StreamingBase
from . import RHE


class StreamingRHE(RHE, StreamingBase):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs) 
    
    def shared_memory(self):
        self.get_num_estimates()
        self.shared_memory_arrays = {
            "XXz": ((self.num_estimates, self.num_workers, self.num_random_vec, self.num_indv), np.float64),
            "yXXy": ((self.num_estimates, self.num_workers), np.float64),
            "M": ((self.num_jack + 1, self.num_estimates), np.int64)
        }

        if self.use_cov:
            self.shared_memory_arrays.update({
                "UXXz": ((self.num_estimates, self.num_workers, self.num_random_vec, self.num_indv), np.float64),
                "XXUz": ((self.num_estimates, self.num_workers, self.num_random_vec, self.num_indv), np.float64),
            })

        if self.multiprocessing:
            # Resulting matrix after aggregating the results from each worker
            self.shared_memory_arrays.update({
                "XXz_per_jack": ((self.num_estimates, 2, self.num_random_vec, self.num_indv), np.float64),
                "yXXy_per_jack": ((self.num_estimates, 2), np.float64),
            })
            if self.use_cov:
                self.shared_memory_arrays.update({
                    "UXXz_per_jack": ((self.num_estimates, 2, self.num_random_vec, self.num_indv), np.float64),
                    "XXUz_per_jack": ((self.num_estimates, 2, self.num_random_vec, self.num_indv), np.float64),
                })

        self.M_last_row = self.len_bin
    
    def pre_compute_jackknife_bin(self, j, all_gen, worker_num):
        for k, X_kj in enumerate(all_gen): 
            self.M[j][k] = self.M[self.num_jack][k] - X_kj.shape[1]
            for b in range(self.num_random_vec):
                self.XXz[k][worker_num][b] += self._compute_XXz(b, X_kj)

                if self.use_cov:
                    self.UXXz[k][worker_num][b] += self._compute_UXXz(self.XXz[k][worker_num][b])
                    self.XXUz[k][worker_num][b] += self._compute_XXUz(b, X_kj)
                    
            yXXy_kj = self._compute_yXXy(X_kj, y=self.pheno)
            self.yXXy[k][worker_num] += yXXy_kj[0][0]
    

        del X_kj

    def pre_compute_jackknife_bin_pass_2(self, j, all_gen):
        for k in range(self.num_estimates):
            X_kj = all_gen[k] if j != self.num_jack else 0
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