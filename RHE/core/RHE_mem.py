"""
Streaming version of RHE
"""
import numpy as np

from .rhe import RHE

class StreamingRHE(RHE):
    def __init__(
        self,
    ):     
        super(StreamingRHE, self).__init__()
        # TODO: Here we assume num_streaming_block == num_jack

    def pre_compute(self):
        """
        Compute U and V 
        """

        random_vecs = [np.random.rand(self.num_indv, 1) for _ in range(self.num_random_vec)]

        Z = {}
        H = {}        
        for j in range(self.num_jack):
            X_j = self._get_jacknife_subsample(self.geno, j)
            all_gen = self.partition_bins(X_j)

            if self.verbose:
                print("Partitioned gen:")
                for gen in all_gen:
                    print(gen.gen)
                    print(gen.num_snp)
                    
            for k, geno in enumerate(all_gen):
                X_kj = geno.gen
                for b, random_vec in enumerate(random_vecs):
                    Z[(k, j, b)] = X_kj @ X_kj.T @ random_vec
            
                v = X_kj.T @ self.pheno
                H[(k, j)] = v.T @ v
                del X_kj
            
        # if self.verbose:
        #     print("Z", Z)
        #     print("H", H)
        
        for k in range(self.num_bin):
            for j in range(self.num_jack):
                for b, random_vec in enumerate(random_vecs):
                    if (k, b, self.num_jack) not in self.U:
                        self.U[(k, b, self.num_jack)] = 0
                    self.U[(k, b, self.num_jack)] += Z[(k, j, b)] 

                if (k, self.num_jack) not in self.V:
                    self.V[(k, self.num_jack)] = 0
                self.V[(k, self.num_jack)] += H[(k, j)]
            

            for j in range(self.num_jack):
                for b, random_vec in enumerate(random_vecs):
                    self.U[(k, b, j)] = self.U[(k, b, self.num_jack)] - Z[(k, j, b)]
                    del Z[(k, j, b)]
                self.V[(k, j)] = self.V[(k, self.num_jack)] - H[(k, j)] 
                del H[(k, j)] 
    