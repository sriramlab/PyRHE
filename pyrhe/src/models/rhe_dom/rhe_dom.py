from pyrhe.src.base import Base
import numpy as np
from typing import List, Tuple, Optional

class RHE_DOM(Base):

    def get_num_estimates(self):
        """Get number of estimates (2x number of bins for original and encoded)"""
        return self.num_bin * 2

    def get_M_last_row(self):
        """Get last row of M matrix"""
        return np.concatenate([self.len_bin, self.len_bin])
    
    def standardize_geno_dom(self, maf, geno_encoded: np.ndarray) -> np.ndarray:
        """
        Standardize genotypes using dominant frequency
        """
        means = np.mean(geno_encoded, axis=0)
        stds = 1 / (2 * maf * (1 - maf))
        return (geno_encoded - means) * stds
        
    def _encode_geno(self, geno: np.ndarray) -> np.ndarray:
        """
        Encode genotypes using dominant frequency
        
        Args:
            geno: Original genotype matrix
            
        Returns:
            Encoded genotype matrix
        """
        # Calculate MAF for each SNP
        maf = np.mean(geno, axis=0) / 2
        # Encode based on MAF values
        encoded = np.zeros_like(geno, dtype=np.float64)
        # Vectorized version to make it faster
        encoded += (geno == 1) * (2 * maf[np.newaxis, :])
        encoded += (geno == 2) * (4 * maf[np.newaxis, :] - 2)
                
        return encoded, maf

    def pre_compute_jackknife_bin(self, j, all_gen):
        """Pre-compute for jackknife bin"""
        for k, X_kj in enumerate(all_gen):
            # Original genotypes
            X_kj = self.standardize_geno(X_kj) # Standardize
            self.M[j][k] = self.M[self.num_jack][k] - X_kj.shape[1]
            for b in range(self.num_random_vec):
                self.XXz[k, j, b, :] = self._compute_XXz(b, X_kj)
                if self.use_cov:
                    self.UXXz[k, j, b, :] = self._compute_UXXz(self.XXz[k][j][b])
                    self.XXUz[k, j, b, :] = self._compute_XXUz(b, X_kj)
            self.yXXy[k][j] = self._compute_yXXy(X_kj, y=self.pheno)
            
            # Encoded genotypes
            X_kj_original = all_gen[k]
            X_kj_encoded, maf = self._encode_geno(X_kj_original)
            # Standardize the encoded genotypes using maf
            X_kj_encoded = self.standardize_geno_dom(maf, X_kj_encoded)
                
            self.M[j][k + self.num_bin] = self.M[self.num_jack][k + self.num_bin] - X_kj_encoded.shape[1]
            for b in range(self.num_random_vec):
                self.XXz[k + self.num_bin, j, b, :] = self._compute_XXz(b, X_kj_encoded)
                if self.use_cov:
                    self.UXXz[k + self.num_bin, j, b, :] = self._compute_UXXz(self.XXz[k + self.num_bin][j][b])
                    self.XXUz[k + self.num_bin, j, b, :] = self._compute_XXUz(b, X_kj_encoded)
            self.yXXy[k + self.num_bin][j] = self._compute_yXXy(X_kj_encoded, y=self.pheno)
        
    def b_trace_calculation(self, k, j, b_idx):
        # Trace can be directly calculated as self.num_indv since the genotype is standardized
        b_trk = self.num_indv
        return b_trk
        

    def run(self, method):
        """
        Run the RHE_DOM model
        """
        sigma_est_jackknife, sigma_ests_total = self.estimate(method=method)
        sig_errs = self.estimate_error(sigma_est_jackknife)

        self.log._log("Variance components: ")
        for i, est in enumerate(sigma_ests_total):
            if i == len(sigma_ests_total) - 1:
                self.log._log(f"Sigma^2_e : {est}  SE : {sig_errs[i]}")
            else:
                self.log._log(f"Sigma^2_g[{i}] : {est}  SE : {sig_errs[i]}")
        
        h2_jackknife, h2_total = self.compute_h2_nonoverlapping(sigma_est_jackknife, sigma_ests_total)
        h2_errs = self.estimate_error(h2_jackknife)

        self.log._log("*****")
        self.log._log("Heritabilities:")
        for i, est_h2 in enumerate(h2_total):
            if i == len(h2_total) - 1:
                self.log._log(f"Total h2 : {est_h2} SE: {h2_errs[i]}")
            else:
                self.log._log(f"h2_g[{i}] : {est_h2} : {h2_errs[i]}")

        self.log._log("*****")
        self.log._log("Enrichments: ")

        enrichment_jackknife, enrichment_total = self.compute_enrichment(h2_jackknife, h2_total)
        enrichment_errs = self.estimate_error(enrichment_jackknife)

        for i, est_enrichment in enumerate(enrichment_total):
            self.log._log(f"Enrichment g[{i}] : {est_enrichment} SE : {enrichment_errs[i]}")

        return {
            "sigma_ests_total": sigma_ests_total,
            "sig_errs": sig_errs,
            "h2_total": h2_total,
            "h2_errs": h2_errs,
            "enrichment_total": enrichment_total,
            "enrichment_errs": enrichment_errs,
        }