"""
Streaming version of RHE
"""
import numpy as np
from typing import Optional

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

        for j in range(self.num_jack):
            X_j = self._get_jacknife_subsample(self.geno, j)
            all_gen = self.partition_bins(X_j)
                    
            for k, geno in enumerate(all_gen):
                X_kj = geno.gen
                # M = X_kj.shape[1]
                # X_kj = X_kj / np.sqrt(M)
                for b, random_vec in enumerate(random_vecs):
                    self.Z[(k, j, b)] = X_kj @ X_kj.T @ random_vec
            
                v = X_kj.T @ self.pheno
                self.H[(k, j)] = v.T @ v

                del X_kj
        
        for k in range(self.num_bin):
            for j in range(self.num_jack):
                for b, random_vec in enumerate(random_vecs):
                    if (k, b, self.num_jack) not in self.U:
                        self.U[(k, b, self.num_jack)] = 0
                    self.U[(k, b, self.num_jack)] += self.Z[(k, j, b)] 

                if (k, self.num_jack) not in self.V:
                    self.V[(k, self.num_jack)] = 0
                self.V[(k, self.num_jack)] += self.H[(k, j)]
            

            for j in range(self.num_jack):
                for b, random_vec in enumerate(random_vecs):
                    self.U[(k, b, j)] = self.U[(k, b, self.num_jack)] - self.Z[(k, j, b)]
                self.V[(k, j)] = self.V[(k, self.num_jack)] - self.H[(k, j)] 
    