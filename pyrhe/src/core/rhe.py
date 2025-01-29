import numpy as np
from . import Base



class RHE(Base):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs) 

    def shared_memory(self):
        self.get_num_estimates()
        self.shared_memory_arrays = {
                "XXz": ((self.num_estimates, self.num_jack + 1, self.num_random_vec, self.num_indv), np.float64),
                "yXXy": ((self.num_estimates, self.num_jack + 1), np.float64),
                "UXXz": ((self.num_estimates, self.num_jack + 1, self.num_random_vec, self.num_indv), np.float64),
                "XXUz": ((self.num_estimates, self.num_jack + 1, self.num_random_vec, self.num_indv), np.float64),
                "M": ((self.num_jack + 1, self.num_estimates), np.int64)
            }
        
        self.M_last_row = self.len_bin
    
    def get_num_estimates(self):
        self.num_estimates = self.num_bin      


    def pre_compute_jackknife_bin(self, j, all_gen):
        for k, X_kj in enumerate(all_gen): 
            self.M[j][k] = self.M[self.num_jack][k] - X_kj.shape[1]
            for b in range(self.num_random_vec):
                self.XXz[k, j, b, :] = self._compute_XXz(b, X_kj)
                if self.use_cov:
                    self.UXXz[k, j, b, :] = self._compute_UXXz(self.XXz[k][j][b])
                    self.XXUz[k, j, b, :] = self._compute_XXUz(b, X_kj)
            self.yXXy[k][j] = self._compute_yXXy(X_kj, y=self.pheno)
    

    def run(self, method):
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

        self.log._log("*****\n*****\nHeritabilities and enrichments computed based on overlapping setting")

        h2_jackknife_overlap, h2_total_overlap = self.compute_h2_overlapping(sigma_est_jackknife, sigma_ests_total)
        h2_errs_overlap = self.estimate_error(h2_jackknife_overlap)

        self.log._log("Heritabilities:")
        for i, est_h2 in enumerate(h2_total_overlap):
            if i == len(h2_total) - 1:
                self.log._log(f"Total h2 : {est_h2} SE: {h2_errs_overlap[i]}")
            else:
                self.log._log(f"h2_g[{i}] : {est_h2} : {h2_errs_overlap[i]}")
        
        self.log._log("Enrichments (overlapping def):")
        enrichment_jackknife_overlap, enrichment_total_overlap = self.compute_enrichment(h2_jackknife_overlap, h2_total_overlap)
        enrichment_errs_overlap = self.estimate_error(enrichment_jackknife_overlap)

        for i, est_enrichment in enumerate(enrichment_total_overlap):
            self.log._log(f"Enrichment g[{i}] : {est_enrichment} SE : {enrichment_errs_overlap[i]}")


        if self.binary_pheno:
            self.log._log("*****")
            self.log._log("Liability Scale h2 for binary phenotype:")
            for i, est_h2 in enumerate(h2_total):
                if i == len(h2_total) - 1:
                    output = self.calculate_liability_h2(h2_total, h2_errs)
                    self.log._log(f"Total Liability-scale h2 : {output[0]}, SE: {output[1]}, p-value: {output[2]}")
                else:
                    output = self.calculate_liability_h2(est_h2, h2_errs[i])
                    self.log._log(f"Liability-scale h2_g[{i}] : {output[0]}, SE: {output[1]}, p-value: {output[2]}")
        
        return {
            "sigma_ests_total": sigma_ests_total,
            "sig_errs": sig_errs,
            "h2_total": h2_total,
            "h2_errs": h2_errs,
            "enrichment_total": enrichment_total,
            "enrichment_errs": enrichment_errs,
            "h2_total_overlap": h2_total_overlap,
            "h2_errs_overlap": h2_errs_overlap,
            "enrichment_total_overlap": enrichment_total_overlap,
            "enrichment_errs_overlap": enrichment_errs_overlap
        }