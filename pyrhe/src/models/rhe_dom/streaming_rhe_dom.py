import numpy as np
from pyrhe.src.base import StreamingBase
from pyrhe.src.models.rhe_dom import RHE_DOM
from pyrhe.src.util.mat_mul import *


class StreamingRHE_DOM(RHE_DOM, StreamingBase):
    
    def pre_compute_jackknife_bin(self, j, all_gen, worker_num):
        for k, X_kj in enumerate(all_gen):
            # Original genotypes
            X_kj = self.standardize_geno(X_kj) # Standardize
            self.M[j][k] = self.M[self.num_jack][k] - X_kj.shape[1]
            for b in range(self.num_random_vec):
                self.XXz[k][worker_num][b] += self._compute_XXz(b, X_kj)
                if self.use_cov:
                    self.UXXz[k][worker_num][b] += self._compute_UXXz(self.XXz[k][worker_num][b])
                    self.XXUz[k][worker_num][b] += self._compute_XXUz(b, X_kj)
            yXXy_kj = self._compute_yXXy(X_kj, y=self.pheno)
            self.yXXy[k][worker_num] += yXXy_kj[0][0]
            
            # Encoded genotypes
            X_kj_original = all_gen[k]
            X_kj_encoded, maf = self._encode_geno(X_kj_original)
            # Standardize the encoded genotypes using maf
            X_kj_encoded = self.standardize_geno_dom(maf, X_kj_encoded)
                
            self.M[j][k + self.num_bin] = self.M[self.num_jack][k + self.num_bin] - X_kj_encoded.shape[1]
            for b in range(self.num_random_vec):
                self.XXz[k + self.num_bin][worker_num][b] += self._compute_XXz(b, X_kj_encoded)
                if self.use_cov:
                    self.UXXz[k + self.num_bin][worker_num][b] += self._compute_UXXz(self.XXz[k + self.num_bin][worker_num][b])
                    self.XXUz[k + self.num_bin][worker_num][b] += self._compute_XXUz(b, X_kj_encoded)
            yXXy_kj = self._compute_yXXy(X_kj_encoded, y=self.pheno)
            self.yXXy[k + self.num_bin][worker_num] += yXXy_kj[0][0]

    def pre_compute_jackknife_bin_pass_2(self, j, all_gen):
        for k, X_kj in enumerate(all_gen):
            X_kj = self.standardize_geno(all_gen[k]) if j != self.num_jack else 0
            for b in range(self.num_random_vec):
                XXz_kb = self._compute_XXz(b, X_kj) if j != self.num_jack else 0
                if self.use_cov:
                    UXXz_kb = self._compute_UXXz(XXz_kb) if j != self.num_jack else 0
                    self.UXXz[k][1][b] = self.UXXz[k][0][b] - UXXz_kb
                    XXUz_kb = self._compute_XXUz(b, X_kj) if j != self.num_jack else 0
                    self.XXUz[k][1][b] = self.XXUz[k][0][b] - XXUz_kb
                self.XXz[k][1][b] = self.XXz[k][0][b] - XXz_kb

            yXXy_k = (self._compute_yXXy(X_kj, y=self.pheno))[0][0] if j != self.num_jack else 0
            self.yXXy[k][1] = self.yXXy[k][0] - yXXy_k

            # Encoded genotypes
            X_kj_original = all_gen[k]
            X_kj_encoded, maf = self._encode_geno(X_kj_original)
            # Standardize the encoded genotypes using maf
            X_kj_encoded = self.standardize_geno_dom(maf, X_kj_encoded)

            for b in range(self.num_random_vec):
                XXz_kb = self._compute_XXz(b, X_kj_encoded) if j != self.num_jack else 0
                if self.use_cov:
                    UXXz_kb = self._compute_UXXz(XXz_kb) if j != self.num_jack else 0
                    self.UXXz[k + self.num_bin][1][b] = self.UXXz[k + self.num_bin][0][b] - UXXz_kb
                    XXUz_kb = self._compute_XXUz(b, X_kj_encoded) if j != self.num_jack else 0
                    self.XXUz[k + self.num_bin][1][b] = self.XXUz[k + self.num_bin][0][b] - XXUz_kb
                self.XXz[k + self.num_bin][1][b] = self.XXz[k + self.num_bin][0][b] - XXz_kb
            
            yXXy_k = (self._compute_yXXy(X_kj_encoded, y=self.pheno))[0][0] if j != self.num_jack else 0
            self.yXXy[k + self.num_bin][1] = self.yXXy[k + self.num_bin][0] - yXXy_k
