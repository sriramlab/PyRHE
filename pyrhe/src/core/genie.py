import numpy as np
from . import Base
from pyrhe.src.util.file_processing import read_env_file
from pyrhe.src.util.mat_mul import *


class GENIE(Base):
    def __init__(
        self,
        env_file: str,
        genie_model: str,
        **kwargs
    ):
         
        super().__init__(**kwargs) 

        self.num_env, self.env = read_env_file(env_file)
        self.env = self.env[:, np.newaxis]
        self.num_gen_env_bin = self.num_bin * self.num_env
        self.genie_model = genie_model

        self.log._log(f"Number of environments: {self.num_env}")
        self.log._log(f"GENIE model: {self.genie_model}")

    def get_num_estimates(self):
        if self.genie_model == "G":
            return self.num_bin
        elif self.genie_model == "G+GxE":
            return self.num_bin + self.num_gen_env_bin
        elif self.genie_model == "G+GxE+NxE":
            return self.num_bin + self.num_gen_env_bin + self.num_env
        else:
            raise ValueError("Unsupported GENIE genie_model type")
    
    def get_M_last_row(self):
        if self.genie_model == "G":
            return self.len_bin
        elif self.genie_model == "G+GxE":
            return np.concatenate((self.len_bin, self.len_bin * self.num_env))
        elif self.genie_model == "G+GxE+NxE":
            return np.concatenate((self.len_bin, self.len_bin * self.num_env, [1] * self.num_env))
        else:
            raise ValueError("Unsupported GENIE genie_model type")

    def pre_compute_jackknife_bin(self, j, all_gen):
        for k, X_kj in enumerate(all_gen): 
            X_kj = self.standardize_geno(X_kj)
            self.M[j][k] = self.M[self.num_jack][k] - X_kj.shape[1]
            print(f"k = {k}, M = {self.M[j][k]}")
            for b in range(self.num_random_vec):
                self.XXz[k, j, b, :] = self._compute_XXz(b, X_kj)
                if self.use_cov:
                    self.UXXz[k, j, b, :] = self._compute_UXXz(self.XXz[k][j][b])
                    self.XXUz[k, j, b, :] = self._compute_XXUz(b, X_kj)
            
            self.yXXy[k][j] = self._compute_yXXy(X_kj, y=self.pheno)
        
        
        # GxE
        if self.genie_model == "G+GxE" or self.genie_model == "G+GxE+NxE":
            for e in range(self.num_env):
                for k, X_kj in enumerate(all_gen): 
                    X_kj = self.standardize_geno(X_kj)
                    k_gxe = (e + 1) * k + self.num_bin
                    self.M[j][k_gxe] = self.M[self.num_jack][k_gxe] - X_kj.shape[1]
                    X_kj_gxe = elem_mul(X_kj, self.env[:, e].reshape(-1, 1), device=self.device)
                    for b in range(self.num_random_vec):
                        self.XXz[k_gxe, j, b, :] = self._compute_XXz(b, X_kj_gxe)

                        if self.use_cov:
                            self.UXXz[k_gxe, j, b, :] = self._compute_UXXz(self.XXz[k_gxe][j][b])
                            self.XXUz[k_gxe, j, b, :] = self._compute_XXUz(b, X_kj_gxe)
                    
                    self.yXXy[k_gxe][j] = self._compute_yXXy(X_kj_gxe, y=self.pheno)
    
                
        # NxE
        if self.genie_model == "G+GxE+NxE":
            for e in range(self.num_env):
                k = e + self.num_bin + self.num_gen_env_bin
                self.M[j][k] = 1
    
    def b_trace_calculation(self, k, j, b_idx):
        if k >= self.num_bin:
            # Actual trace calculation
            M_k = self.M[j][k]
            B1 = self.XXz[k][b_idx]
            b_trk = np.sum(B1 * self.all_zb.T) / (self.num_random_vec * M_k)
        else:
            # Trace can be directly calculated as self.num_indv since the genotype is standardized
            b_trk = self.num_indv

        return b_trk

    def run(self, method):
        sigma_est_jackknife, sigma_ests_total = self.estimate(method=method)
        sig_errs = self.estimate_error(sigma_est_jackknife)

        self.log._log("Variance components: ")

        for i, est in enumerate(sigma_ests_total):
            if self.genie_model == "G":
                if i != len(sigma_ests_total) - 1:
                    self.log._log(f"Sigma^2_g[{i}] : {est}  SE : {sig_errs[i]}")

            elif self.genie_model == "G+GxE":
                if i < self.num_bin:
                    self.log._log(f"Sigma^2_g[{i}] : {est}  SE : {sig_errs[i]}")
                else:
                    self.log._log(f"Sigma^2_gxe[{i - self.num_bin}] : {est}  SE : {sig_errs[i]}")

            elif self.genie_model == "G+GxE+NxE":
                if i < self.num_bin:
                    self.log._log(f"Sigma^2_g[{i}] : {est}  SE : {sig_errs[i]}")
                elif i < self.num_bin + self.num_gen_env_bin:
                    self.log._log(f"Sigma^2_gxe[{i - self.num_bin}] : {est}  SE : {sig_errs[i]}")
                elif i < self.num_bin + self.num_gen_env_bin + self.num_env:
                    self.log._log(f"Sigma^2_nxe[{i - self.num_bin - self.num_gen_env_bin}] : {est}  SE : {sig_errs[i]}")

        self.log._log(f"Sigma^2_e : {sigma_ests_total[-1]}  SE : {sig_errs[-1]}")

        return {
            "sigma_ests_total": sigma_ests_total,
            "sig_errs": sig_errs,
        }

    # def run(self, method): # TODO: Fix
    #     sigma_est_jackknife, sigma_ests_total = self.estimate(method=method)
    #     sig_errs = self.estimate_error(sigma_est_jackknife)

    #     self.log._log("Variance components: ")
    #     for i, est in enumerate(sigma_ests_total):
    #         if i == len(sigma_ests_total) - 1:
    #             self.log._log(f"Sigma^2_e : {est}  SE : {sig_errs[i]}")
    #         else:
    #             self.log._log(f"Sigma^2_g[{i}] : {est}  SE : {sig_errs[i]}")
        
    #     h2_jackknife, h2_total = self.compute_h2_nonoverlapping(sigma_est_jackknife, sigma_ests_total)
    #     h2_errs = self.estimate_error(h2_jackknife)

    #     self.log._log("*****")
    #     self.log._log("Heritabilities:")
    #     for i, est_h2 in enumerate(h2_total):
    #         if i == len(h2_total) - 1:
    #             self.log._log(f"Total h2 : {est_h2} SE: {h2_errs[i]}")
    #         else:
    #             self.log._log(f"h2_g[{i}] : {est_h2} : {h2_errs[i]}")

    #     self.log._log("*****")
    #     self.log._log("Enrichments: ")

    #     enrichment_jackknife, enrichment_total = self.compute_enrichment(h2_jackknife, h2_total)
    #     enrichment_errs = self.estimate_error(enrichment_jackknife)

    #     for i, est_enrichment in enumerate(enrichment_total):
    #         self.log._log(f"Enrichment g[{i}] : {est_enrichment} SE : {enrichment_errs[i]}")

    #     self.log._log("*****\n*****\nHeritabilities and enrichments computed based on overlapping setting")

    #     h2_jackknife_overlap, h2_total_overlap = self.compute_h2_overlapping(sigma_est_jackknife, sigma_ests_total)
    #     h2_errs_overlap = self.estimate_error(h2_jackknife_overlap)

    #     self.log._log("Heritabilities:")
    #     for i, est_h2 in enumerate(h2_total_overlap):
    #         if i == len(h2_total) - 1:
    #             self.log._log(f"Total h2 : {est_h2} SE: {h2_errs_overlap[i]}")
    #         else:
    #             self.log._log(f"h2_g[{i}] : {est_h2} : {h2_errs_overlap[i]}")
        
    #     self.log._log("Enrichments (overlapping def):")
    #     enrichment_jackknife_overlap, enrichment_total_overlap = self.compute_enrichment(h2_jackknife_overlap, h2_total_overlap)
    #     enrichment_errs_overlap = self.estimate_error(enrichment_jackknife_overlap)

    #     for i, est_enrichment in enumerate(enrichment_total_overlap):
    #         self.log._log(f"Enrichment g[{i}] : {est_enrichment} SE : {enrichment_errs_overlap[i]}")


    #     if self.binary_pheno:
    #         self.log._log("*****")
    #         self.log._log("Liability Scale h2 for binary phenotype:")
    #         for i, est_h2 in enumerate(h2_total):
    #             if i == len(h2_total) - 1:
    #                 output = self.calculate_liability_h2(h2_total, h2_errs)
    #                 self.log._log(f"Total Liability-scale h2 : {output[0]}, SE: {output[1]}, p-value: {output[2]}")
    #             else:
    #                 output = self.calculate_liability_h2(est_h2, h2_errs[i])
    #                 self.log._log(f"Liability-scale h2_g[{i}] : {output[0]}, SE: {output[1]}, p-value: {output[2]}")
        
    #     return {
    #         "sigma_ests_total": sigma_ests_total,
    #         "sig_errs": sig_errs,
    #         "h2_total": h2_total,
    #         "h2_errs": h2_errs,
    #         "enrichment_total": enrichment_total,
    #         "enrichment_errs": enrichment_errs,
    #         "h2_total_overlap": h2_total_overlap,
    #         "h2_errs_overlap": h2_errs_overlap,
    #         "enrichment_total_overlap": enrichment_total_overlap,
    #         "enrichment_errs_overlap": enrichment_errs_overlap
    #     }