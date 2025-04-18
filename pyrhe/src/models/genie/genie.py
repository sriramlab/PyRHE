import numpy as np
from typing import Tuple, List
from pyrhe.src.base import Base
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
    

    def estimate(self, method: str = "lstsq") -> Tuple[List[List], List]:
        """
        Adjust the estimate method in the base class to include the effect of traces for heritability calculation
        """
        sigma_ests = []
        sigma_ests_adj = []
        if self.get_trace:
            trace_sums = np.zeros((self.num_jack + 1, self.num_estimates, self.num_estimates))
        else:
            trace_sums = None

        for j in range(self.num_jack + 1):
            self.log._debug(f"Estimate for jackknife sample {j}")

            if self.num_jack == 1 and j == 0:
                T, q = self.setup_lhs_rhs_jackknife(1, trace_sums)
            
            else:
                T, q = self.setup_lhs_rhs_jackknife(j, trace_sums)
            
            if method == "lstsq":
                sigma_est = self.solve_linear_equation(T,q)
                sigma_est = np.ravel(sigma_est).tolist()
                sigma_ests.append(sigma_est)
            elif method == "QR":
                sigma_est = self.solve_linear_qr(T,q)
                sigma_est = np.ravel(sigma_est).tolist()
                sigma_ests.append(sigma_est)
            else:
                raise ValueError("Unsupported method for solving linear equation")
            
            sigma_est_adj = []
            for i in range(len(sigma_est)):
                sigma_est_adj.append(sigma_est[i] * T[i, self.num_estimates])
            sigma_ests_adj.append(sigma_est_adj)
            
        sigma_ests = np.array(sigma_ests)

        sigma_est_jackknife, sigma_ests_total = sigma_ests[:-1, :], sigma_ests[-1, :]

        # Adjust the estimate by the effect of traces for heritability calculation
        sigma_ests_adj = np.array(sigma_ests_adj)
        sigma_est_jackknife_adj, sigma_ests_total_adj = sigma_ests_adj[:-1, :], sigma_ests_adj[-1, :]

        if self.get_trace:
            self.get_trace_summary(trace_sums)
            
        return sigma_est_jackknife, sigma_ests_total, sigma_est_jackknife_adj, sigma_ests_total_adj

    def compute_h2_nonoverlapping(self, sigma_est_jackknife, sigma_ests_total):
        sigma_ests = np.vstack([sigma_est_jackknife, sigma_ests_total[np.newaxis, :]])
        h2 = []
        
        for j in range(self.num_jack + 1):
            sigma_ests_jack = sigma_ests[j]
            h2_list = []
            
            total_var = 0
            for k in range (self.num_estimates):
                total_var += sigma_ests_jack[k]
            
            assert sigma_ests_jack[-1] == sigma_ests_jack[k+1]

            g_total = 0
            for i in range(self.num_bin):
                h2_i = sigma_ests_jack[i] / (total_var + sigma_ests_jack[-1])
                h2_list.append(h2_i)
                g_total += h2_i
            
            gxe_total = 0
            if self.genie_model in ["G+GxE", "G+GxE+NxE"]:
                for i in range(self.num_gen_env_bin):
                    h2_i = sigma_ests_jack[self.num_bin + i] / (total_var + sigma_ests_jack[-1])
                    h2_list.append(h2_i)
                    gxe_total += h2_i
            
            nxe_total = 0
            if self.genie_model == "G+GxE+NxE":
                for i in range(self.num_env):
                    h2_i = sigma_ests_jack[self.num_bin + self.num_gen_env_bin + i] / (total_var + sigma_ests_jack[-1])
                    h2_list.append(h2_i)
                    nxe_total += h2_i
            
            total_h2 = g_total + gxe_total + nxe_total
            h2_list.append(total_h2)  # Total h2
            h2_list.append(g_total)   # Total h2_g
            if self.genie_model in ["G+GxE", "G+GxE+NxE"]:
                h2_list.append(gxe_total)  # Total h2_gxe
            
            h2.append(h2_list)
        
        h2 = np.array(h2)
        return h2[:-1, :], h2[-1, :]

    def compute_enrichment(self, h2_jackknife, h2_total):
        """Compute enrichment estimates"""
        enrichment = []
        M = np.sum(self.M[-1][:self.num_bin])
        
        for j in range(self.num_jack + 1):
            enrichment_list = []
            h2_j = h2_jackknife[j] if j < len(h2_jackknife) else h2_total
            
            # Calculate total h2 and total SNPs for genetic components only
            total_h2 = 0
            total_snps = 0
            for i in range(self.num_bin):
                total_h2 += h2_j[i]
                total_snps += self.M[-1][i]
            
            # Calculate denominator (average h2 per SNP)
            denom = total_h2 / total_snps
            
            # Calculate enrichment for each bin
            for i in range(self.num_bin):
                M_k = self.M[-1][i]
                h2_g = h2_j[i]
                enrichment_list.append((h2_g / M_k) / denom)
            
            enrichment.append(enrichment_list)
        
        enrichment = np.array(enrichment)
        return enrichment[:-1, :], enrichment[-1, :]

    def run(self, method):
        sigma_est_jackknife, sigma_ests_total, sigma_est_jackknife_adj, sigma_ests_total_adj = self.estimate(method=method)
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
        
        h2_jackknife, h2_total = self.compute_h2_nonoverlapping(sigma_est_jackknife_adj, sigma_ests_total_adj)
        h2_errs = self.estimate_error(h2_jackknife)

        self.log._log("*****")
        self.log._log("Heritabilities:")
        for i, est_h2 in enumerate(h2_total):
            if self.genie_model == "G":
                if i < self.num_bin:
                    self.log._log(f"h2_g[{i}] : {est_h2} SE : {h2_errs[i]}")
                elif i == self.num_bin:
                    self.log._log(f"Total h2 : {est_h2} SE: {h2_errs[i]}")
                elif i == self.num_bin + 1:
                    self.log._log(f"Total h2_g : {est_h2} SE: {h2_errs[i]}")
            elif self.genie_model == "G+GxE":
                if i < self.num_bin:
                    self.log._log(f"h2_g[{i}] : {est_h2} SE : {h2_errs[i]}")
                elif i < self.num_bin + self.num_gen_env_bin:
                    self.log._log(f"h2_gxe[{i - self.num_bin}] : {est_h2} SE : {h2_errs[i]}")
                elif i == self.num_bin + self.num_gen_env_bin:
                    self.log._log(f"Total h2 : {est_h2} SE: {h2_errs[i]}")
                elif i == self.num_bin + self.num_gen_env_bin + 1:
                    self.log._log(f"Total h2_g : {est_h2} SE: {h2_errs[i]}")
                elif i == self.num_bin + self.num_gen_env_bin + 2:
                    self.log._log(f"Total h2_gxe : {est_h2} SE: {h2_errs[i]}")
            elif self.genie_model == "G+GxE+NxE":
                if i < self.num_bin:
                    self.log._log(f"h2_g[{i}] : {est_h2} SE : {h2_errs[i]}")
                elif i < self.num_bin + self.num_gen_env_bin:
                    self.log._log(f"h2_gxe[{i - self.num_bin}] : {est_h2} SE : {h2_errs[i]}")
                elif i < self.num_bin + self.num_gen_env_bin + self.num_env:
                    self.log._log(f"h2_nxe[{i - self.num_bin - self.num_gen_env_bin}] : {est_h2} SE : {h2_errs[i]}")
                elif i == self.num_bin + self.num_gen_env_bin + self.num_env:
                    self.log._log(f"Total h2 : {est_h2} SE: {h2_errs[i]}")
                elif i == self.num_bin + self.num_gen_env_bin + self.num_env + 1:
                    self.log._log(f"Total h2_g : {est_h2} SE: {h2_errs[i]}")
                elif i == self.num_bin + self.num_gen_env_bin + self.num_env + 2:
                    self.log._log(f"Total h2_gxe : {est_h2} SE: {h2_errs[i]}")

        self.log._log("*****")
        self.log._log("Enrichments:")
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
            "enrichment_errs": enrichment_errs
        }