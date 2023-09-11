"""
Streaming version of RHE
"""
import numpy as np

from .RHE import RHE

class StreamingRHE(RHE):
    def __init__(
        self,
    ):     
        super(StreamingRHE, self).__init__()

    def pre_compute(self):
        """
        Compute the U tensor and the V matrix
        """

        # TODO: Here we assume num_streaming_block == num_jack

        random_vecs = [np.random.rand(self.num_indv, 1) for _ in range(self.num_random_vec)]

        Z = np.zeros((self.num_bin, self.num_jack, self.num_random_vec)) 

        H = np.zero((self.num_bin, self.num_jack))
        
        for j in range(self.num_jack):
            X_j = self._get_jacknife_subsample(self.geno, j)
            all_gen = self.partition_bins(X_j)
            for k, geno in enumerate(all_gen):
                X_kj = geno.gen
                for b, random_vec in random_vecs:
                    Z[k][j][b] = X_kj @ X_kj.T @ random_vec 

                v = X_kj.T @ self.pheno
                H[k][j] = v.T @ v

                del X_kj
        

        for k in range(self.num_bin):
            for b, random_vec in random_vecs:
                for j in range(self.num_jack):
                    self.U[k][b][-1] += Z[k][b][j]
                    self.V[k][-1] += H[k][j]

                for j in range(self.num_jack):
                    self.U[k][b][j] = self.U[k][b][-1] - Z[k][b][j]
                    del Z[k][b][j]
                    self.V[k][j] = self.H[k][-1] - H[k][j]
                    del H[k][j]