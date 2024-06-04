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
    
    def setup_lhs_rhs_jackknife(self, j):
        T = np.zeros((self.num_bin+1, self.num_bin+1))
        q = np.zeros((self.num_bin+1, 1))

        for k_k in range(self.num_bin):
            for k_l in range(self.num_bin):
                M_k = self.M[j][k_k]
                M_l = self.M[j][k_l]
                B1 = self.XXz[k_k][j]
                B2 = self.XXz[k_l][j]
                T[k_k, k_l] += np.sum(B1 * B2)

                if self.use_cov:
                    h1 = self.cov_matrix.T @ B1.T
                    h2 = self.Q @ h1
                    h3 = self.cov_matrix @ h2
                    trkij_res1 = np.sum(h3.T * B2)

                    B1 = self.XXUz[k_k][j]
                    B2 = self.UXXz[k_l][j]
                    trkij_res2 = np.sum(B1 * B2)
                
                    T[k_k, k_l] += (trkij_res2 - 2 * trkij_res1)


                T[k_k, k_l] /= (self.num_random_vec)
                T[k_k, k_l] =  T[k_k, k_l] / (M_k * M_l) if (M_k * M_l) != 0 else 0


        for k in range(self.num_bin):
            M_k = self.M[j][k]

            if not self.use_cov:
                T[k, self.num_bin] = self.num_indv
                T[self.num_bin, k] = self.num_indv

            else:
                B1 = self.XXz[k][j]
                C1 = self.all_Uzb
                tk_res = np.sum(B1 * C1.T)
                tk_res = 1/(self.num_random_vec * M_k) * tk_res

                T[k, self.num_bin] = self.num_indv - tk_res
                T[self.num_bin, k] = self.num_indv - tk_res

            q[k] = self.yXXy[(k, j)] / M_k if M_k != 0 else 0
        
        
        T[self.num_bin, self.num_bin] = self.num_indv if not self.use_cov else self.num_indv - self.cov_matrix.shape[1]

        pheno = self.pheno if not self.use_cov else self.regress_pheno(self.cov_matrix, self.pheno)
        q[self.num_bin] = pheno.T @ pheno

        return T, q