"""
Streaming version of RHE
"""
import numpy as np
from typing import Optional
from src.util.math import *

from .rhe import RHE

class StreamingRHE(RHE):
    def __init__(self, 
        num_streaming_block: Optional[int] = None,
        *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: Here we assume num_streaming_block == num_jack

        self.num_streaming_block = self.num_jack if num_streaming_block is None else num_streaming_block

    def pre_compute(self):
        """
        Compute U and V 
        """
        random_vecs = [np.random.randn(self.num_indv, 1) for _ in range(self.num_random_vec)]
        if self.use_cov:
            for b in range(self.num_random_vec):
                random_vecs[b] = self.cov_matrix @ (self.cov_matrix_aug @ self.cov_matrix.T @ random_vecs[b])

        for j in range(self.num_jack):
            print(f"Precompute for jackknife sample {j}")
            _, excluded_cols = self._get_jacknife_subsample(self.geno, j)
            all_gen = self.partition_bins(self.geno, excluded_cols)
                    
            for k, geno in enumerate(all_gen):
                X_kj = geno.gen

                X_kj = impute_geno(X_kj) # TODO: modify impute geno
                for b, random_vec in enumerate(random_vecs):
                    self.Z[k, j, b, :] = (X_kj @ X_kj.T @ random_vec).flatten()

                if self.use_cov:
                    for b in range(self.num_random_vec):
                        self.cov_Z_1[k, j, b, :] = (self.cov_matrix @ (self.cov_matrix_aug @ (self.cov_matrix.T @ self.Z[k][j][b]))).flatten()
                        self.cov_Z_2[k, j, b, :] = X_kj @ X_kj.T @ random_vecs[b].flatten()

                v = X_kj.T @ self.pheno if not self.use_cov else X_kj.T @ (self.regress_pheno(self.cov_matrix, self.pheno))
                self.H[k][j] = v.T @ v

                del X_kj
        
        for k in range(self.num_bin):
            for j in range(self.num_jack):
                for b, random_vec in enumerate(random_vecs):
                    self.U[k][self.num_jack][b] += self.Z[k][j][b]
                self.V[k][self.num_jack] += self.H[k][j]
            
            for j in range(self.num_jack):
                for b, random_vec in enumerate(random_vecs):
                    self.U[k][j][b] = self.U[k][self.num_jack][b] - self.Z[k][j][b]
                self.V[k][j] = self.V[k][self.num_jack] - self.H[k][j]

        del self.Z
        del self.H
    