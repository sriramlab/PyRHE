import time
import torch
import numpy as np
from typing import List, Tuple, Optional
from bed_reader import open_bed
from src.util.math import *
from src.util.file_processing import *


class GenoChunk:
    def __init__(self):
        self.gen = None 
        self.num_snp = 0 

class RHE:
    def __init__(
        self,
        geno_file: str,
        annot_file: str = None,
        pheno_file: str = None,
        cov_file: str = None,
        num_bin: int = 8,
        num_jack: int = 1,
        num_random_vec: int = 10,
        verbose: bool = True,
        seed: int = 0,
    ):
        """
        Initialize the RHE algorithm (creating geno matrix, pheno matrix, etc.)
        """
        np.random.seed(seed)
        self.num_bin= num_bin
        self.num_jack = num_jack
        self.num_random_vec = num_random_vec
        self.verbose = verbose

        if annot_file is None:
            self.num_bin = num_bin
        else:
            self.num_bin = None

        # read fam and bim file
        self.geno_bed = open_bed(geno_file + ".bed")
        self.num_indv = read_fam(geno_file + ".fam")
        self.num_snp = read_bim(geno_file + ".bim")

        # read pheno file
        if pheno_file is not None:
            self.pheno = read_pheno(pheno_file)
            assert self.num_indv == self.pheno.shape[0]
        else:
            self.pheno = None

         # read annot file, generate one if not exist
        if annot_file is None:
            if self.num_bin is None:
                raise ValueError("Must specify number of bins if annot file is not provided")

            annot_file = "generated_annot"

            generate_annot(annot_file, self.num_snp, self.num_bin)

        self.num_bin, self.annot_matrix, len_bin = read_annot(annot_file, self.num_jack)

        # read covariance file
        if cov_file is None:
            self.use_cov = False
            self.cov_matrix = None
            self.Q = None
        else:
            self.use_cov = True
            self.cov_matrix, _ = read_cov(cov_file)
            self.Q = np.linalg.inv(self.cov_matrix.T @ self.cov_matrix)
        
        # all_zb
        self.all_zb = np.random.randn(self.num_indv, self.num_random_vec)
        if self.use_cov:
            self.all_zb = self.cov_matrix @ self.Q @ (self.cov_matrix.T @ self.all_zb)

        self.XXz = np.zeros((self.num_bin, self.num_jack + 1, self.num_random_vec, self.num_indv))
        self.yXXy = np.zeros((self.num_bin, self.num_jack + 1))
        self.cov_H = np.zeros((self.num_bin, self.num_jack + 1)) if self.use_cov else None
        self.cov_Z_1 = np.zeros((self.num_bin, self.num_jack + 1, self.num_random_vec, self.num_indv)) if self.use_cov else None
        self.cov_Z_2 = np.zeros((self.num_bin, self.num_jack + 1, self.num_random_vec, self.num_indv)) if self.use_cov else None
        self.cov_U_1 = np.zeros((self.num_bin, self.num_jack + 1, self.num_random_vec, self.num_indv)) if self.use_cov else None
        self.cov_U_2 = np.zeros((self.num_bin, self.num_jack + 1, self.num_random_vec, self.num_indv)) if self.use_cov else None

        self.M = np.zeros((self.num_jack + 1, self.num_bin))
        self.M[self.num_jack] = len_bin
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def simulate_pheno(self, sigma_list: List):
        '''
        Simulate phenotype y from X and sigma_list
        '''

        geno = self.geno_bed.read()
        geno = impute_geno(geno, simulate_geno=True)

        if len(sigma_list) != self.num_bin:
            
            raise ValueError("Number of elements in sigma list should be equal to number of bins")

        if len(sigma_list) == 1:
                sigma = sigma_list[0]
                y = np.zeros((self.num_indv,1))
                sigma_epsilon=1 - sigma # environmental effect sizes
                betas = np.random.randn(self.num_snp,1)*np.sqrt(sigma) # additive SNP effect sizes
                y += X@betas/np.sqrt(M)
                #print(f'sigma_epislon={sigma_epsilon}')
                y += np.random.randn(self.num_snp,1)*np.sqrt(sigma_epsilon) # add the effect sizes

        else:
            all_gen = self.partition_bins(geno=geno)

            h = 1 - sum(sigma_list) # residual covariance
            sigma_epsilon = np.random.multivariate_normal([0] * self.num_indv, np.diag(np.full(self.num_indv, h)))
            sigma_epsilon = np.array(sigma_epsilon).reshape((len(sigma_epsilon), 1))
            y = np.zeros((self.num_indv,1))

            betas = []

            for i, data in enumerate(all_gen):
                X = data.gen
                M = X.shape[1]
                sigma = sigma_list[i]
                beta = np.random.multivariate_normal([0] * M, np.diag(np.full(M, sigma / M)))
                beta = np.array(beta).reshape((len(beta), 1))
                betas.append(beta)
                y += X@beta
            
            y +=  sigma_epsilon 

        self.pheno = y

        return y, betas

    def _bin_to_snp(self): 
        bin_to_snp_indices = []

        for bin_index in range(self.num_bin):
            snp_indices = np.nonzero(self.annot_matrix[:, bin_index])[0]
            bin_to_snp_indices.append(snp_indices.tolist())

        return bin_to_snp_indices
        

    def partition_bins(self, excluded_columns: Optional[List] = None, geno: Optional[np.ndarray] = None):
        """
        Partition the genotype matrix into num_bin bins
        """

        bin_to_snp_indices = self._bin_to_snp()
        
        all_gen = [GenoChunk() for _ in range(len(bin_to_snp_indices))]

        for i, data in enumerate(all_gen):
            snp_indices = bin_to_snp_indices[i]   
            valid_indices = list(set(snp_indices) - set(excluded_columns)) if excluded_columns else snp_indices
            data.num_snp = len(valid_indices)
            if geno is not None:
                data.gen = geno[:, valid_indices]
            else:
                data.gen = self.geno_bed.read(index=np.s_[:, valid_indices])
        
        return all_gen

        
    def _get_jacknife_subsample(self, jack_index: int) -> np.ndarray:
        """
        Get the jacknife subsample (imputed)
        """

        step_size = self.num_snp // self.num_jack
        step_size_rem = self.num_snp % self.num_jack
        chunk_size = step_size if jack_index < (self.num_jack - 1) else step_size + step_size_rem
        start = jack_index * step_size
        end = start + chunk_size

         # read bed file
        try:
            subsample = self.geno_bed.read(index=np.s_[::1,start:end])
            subsample = impute_geno(subsample, simulate_geno=True)
        except Exception as e:
            raise Exception(f"Error occurred: {e}")

        excluded_columns = list(set(range(self.num_snp)) - set(range(start, end)))
        return subsample, excluded_columns

    def _to_tensor(self, mat):
        return torch.from_numpy(mat).float().to(self.device) if self.device == torch.device("cuda") else mat

    def mat_mul(self, *mats, to_numpy=True):
        if not mats:
            raise ValueError("At least one matrix is required.")

        if self.device == torch.device("cuda"):
            result_tensor = self._to_tensor(mats[0])
            for mat in mats[1:]:
                tensor = self._to_tensor(mat)
                result_tensor = result_tensor @ tensor
            if to_numpy:
                return result_tensor.cpu().numpy()
            else:
                return result_tensor
        else:
            result = mats[0]
            for mat in mats[1:]:
                result = result @ mat
            return result

    def regress_pheno(self, cov_matrix, pheno):
        """
        Project y onto the column space of the covariance matrix and compute the residual
        """
        Q = np.linalg.inv(cov_matrix.T @ cov_matrix)
        return pheno - cov_matrix @ Q @ cov_matrix.T @ pheno
        
    def pre_compute(self, verbose: bool=False):
        """
        Compute U and V 
        """
        
        Z = np.zeros((self.num_bin, self.num_jack + 1, self.num_random_vec, self.num_indv))
        cov_Z_1 = np.zeros((self.num_bin, self.num_jack + 1, self.num_random_vec, self.num_indv))
        cov_Z_2 = np.zeros((self.num_bin, self.num_jack + 1, self.num_random_vec, self.num_indv))

        H = np.zeros((self.num_bin, self.num_jack + 1))

        for j in range(self.num_jack):
            print(f"Precompute for jackknife sample {j}")
            _, excluded_cols = self._get_jacknife_subsample(j)
            all_gen = self.partition_bins(excluded_cols)
                    
            for k, geno in enumerate(all_gen):
                X_kj = geno.gen
                print(X_kj)
                # X_kj = impute_geno(X_kj, simulate_geno=True)
                self.M[j][k] = self.M[self.num_jack][k] - geno.num_snp # store the dimension with the corresponding block
                for b in range(self.num_random_vec):
                    start = time.time()       
                    random_vec = self.all_zb[:, b].reshape(-1, 1)
                    Z[k, j, b, :] = (self.mat_mul(X_kj, self.mat_mul(X_kj.T, random_vec))).flatten()
                    end = time.time()
                    if verbose:
                        print(f"XXz time: {end - start}")
                    
                if self.use_cov:
                    for b in range(self.num_random_vec):
                        random_vec = self.all_zb[:, b].reshape(-1, 1)
                        cov_Z_1[k, j, b, :] = self.mat_mul(self.cov_matrix, self.mat_mul(self.Q, self.mat_mul(self.cov_matrix.T, Z[k][j][b]))).flatten()
                        cov_Z_2[k, j, b, :] = self.mat_mul(X_kj, self.mat_mul(X_kj.T, random_vec)).flatten()

                pheno = self.pheno if not self.use_cov else self.regress_pheno(self.cov_matrix, self.pheno)
                v = self.mat_mul(X_kj.T, pheno)
                H[k][j] = self.mat_mul(v.T, v)
                del X_kj
            
        
        if not self.use_cov:
            for k in range(self.num_bin):
                for j in range(self.num_jack):
                    for b in range (self.num_random_vec):
                        self.XXz[k][self.num_jack][b] += Z[k][j][b]
                    self.yXXy[k][self.num_jack] += H[k][j]
                
                for j in range(self.num_jack):
                    for b in range (self.num_random_vec):
                        self.XXz[k][j][b] = self.XXz[k][self.num_jack][b] - Z[k][j][b]
                    self.yXXy[k][j] = self.yXXy[k][self.num_jack] - H[k][j]
        else:
            for k in range(self.num_bin):
                for j in range(self.num_jack):
                    for b in range (self.num_random_vec):
                        self.UXXz[k][self.num_jack][b] += cov_Z_1[k][j][b]
                        self.XXUz[k][self.num_jack][b] += cov_Z_2[k][j][b]
                    self.yXXy[k][self.num_jack] += H[k][j]
                
                for j in range(self.num_jack):
                    for b in range (self.num_random_vec):
                        self.UXXz[k][j][b] = self.UXXz[k][self.num_jack][b] - cov_Z_1[k][j][b]
                        self.XXUz[k][j][b] = self.XXUz[k][self.num_jack][b] - cov_Z_2[k][j][b]
                    self.yXXy[k][j] = self.yXXy[k][self.num_jack] - H[k][j]

     
    def estimate(self, method: str = "QR") -> Tuple[List[List], List]:
        """
        Actual RHE estimation for sigma^2
        Returns: 
        sigma_est_jackknife: 
            rows: jackknife sample
            cols: sigma^2 for one jackknife sample [sigma_1^2 sigma_2^2 ... sigma_e^2]
            e.g.,
            sigma_est_jackknife[0][0]: sigma_2 estimate for the 1st bin in the 1st jacknife sample
            sigma_est_jackknife[0][1]: sigma_2 estimate for the 1st bin in the 2st jacknife sample

        sigma_ests_total
            sigma^2 for the whole genotype matrix [sigma_1^2 sigma_2^2 ... sigma_e^2]

        """

        sigma_ests = []

        for j in range(self.num_jack + 1):
            print(f"Estimate for jackknife sample {j}")

            T = np.zeros((self.num_bin+1, self.num_bin+1))
            q = np.zeros((self.num_bin+1, 1))

            for k_k in range(self.num_bin):
                for k_l in range(self.num_bin):
                    M_k = self.M[j][k_k]
                    M_l = self.M[j][k_l]
                    for b in range(self.num_random_vec):
                        B1 = self.XXz[k_k][j][b]
                        B2 = self.XXz[k_l][j][b]
                        if not self.use_cov:
                            T[k_k, k_l] += np.sum(B1*B2)
                        else:
                            trkij_res1 = np.sum((self.cov_matrix @ (self.Q @ (self.cov_matrix.T @ B1)) * B2).sum(axis=0))
                            B1 = self.XXUz[k_k][j][b]
                            B2 = self.UXXz[k_l][j][b]
                            trkij_res2 = np.sum(B1 * B2)
                            T[k_k, k_l] += (trkij_res2 - trkij_res1)


                    T[k_k, k_l] /= (self.num_random_vec)
                    T[k_k, k_l] =  T[k_k, k_l] / (M_k * M_l) if (M_k * M_l) != 0 else 0

            for k in range(self.num_bin):
                M_k = self.M[j][k]
                T[k, self.num_bin] = self.num_indv
                T[self.num_bin, k] = self.num_indv

                if self.use_cov:
                    b_trK_res = 0
                    for b in range(self.num_random_vec):
                        b_trK_res = b_trK_res + np.sum(self.all_zb[:, b:(b+1)].T @ self.XXz[k_k][j])
                    T[k, self.num_bin] -= 1/(k * M_k) * b_trK_res
                    T[self.num_bin, k] -= 1/(k * M_k) * b_trK_res
                    
                        
                q[k] = self.yXXy[(k, j)] / M_k if M_k != 0 else 0
            
            
            T[self.num_bin, self.num_bin] = self.num_indv if not self.use_cov else self.num_indv - self.cov_matrix.shape[1]
            pheno = self.pheno if not self.use_cov else self.regress_pheno(self.cov_matrix, self.pheno)
            q[self.num_bin] = pheno.T @ pheno 

            if method == "QR":
                sigma_est = solve_linear_equation(T,q)
                sigma_est = np.ravel(sigma_est).tolist()
                sigma_ests.append(sigma_est)
            elif method == "lstsq":
                sigma_est = solve_linear_qr(T,q)
                sigma_est = np.ravel(sigma_est).tolist()
                sigma_ests.append(sigma_est)
            else:
                raise ValueError("Unsupported method for solving linear equation")
            
        sigma_ests = np.array(sigma_ests)

        sigma_est_jackknife, sigma_ests_total = sigma_ests[:-1, :], sigma_ests[-1, :]
            
        return sigma_est_jackknife, sigma_ests_total

    def estimate_error(self, ests: List[List]):
            """
            Estimate the standard error
            Parameters:
                ests: estimates for each jackknife sample
            Returns: 
                SE: standard error for the estimation for each bin
                    e.g., 
                        SE[0]: standard error for sigma_1^2
                        SE[1]: standard error for sigma_2^2
            """

            SE = []

            for bin_index in range(ests.shape[1]):
                bin_values = ests[:, bin_index]
                mean_bin = np.mean(bin_values)
                sq_diffs = 0
                for bin_value in bin_values:
                    sq_diffs += (bin_value - mean_bin)**2
                se_bin = np.sqrt((self.num_jack - 1) * sq_diffs / self.num_jack)
                SE.append(se_bin)

            return SE

    def compute_h2(self, sigma_est_jackknife: List[List], sigma_ests_total: List) -> Tuple[List[List], List]:
        """
        Compute h^2

        Returns: 
        h2_jackknife
            rows: jackknife sample
            cols: h^2 for one jackknife sample [h_1^2 h_2^2 ... h_SNP^2]
            e.g.,
            h2_jackknife[0][0]: h^2 for the 1st bin in the 1st jacknife sample
            h2_jackknife[0][1]: h^2 for the 1st bin in the 2st jacknife sample

        h2_total
            h^2 for the whole genotype matrix [h_1^2 h_2^2 ... h_SNP^2]

        """

        sigma_ests = np.vstack([sigma_est_jackknife, sigma_ests_total[np.newaxis, :]])

        h2 = []

        for j in range(self.num_jack + 1):
            sigma_ests_jack = sigma_ests[j]
            total = 0
            for k in range (self.num_bin):
                total += sigma_ests_jack[k]
            assert sigma_ests_jack[-1] == sigma_ests_jack[k+1]
            h2_list = [x / (total + sigma_ests_jack[-1])  for x in sigma_ests_jack]
            # compute h_SNP^2
            h_SNP_2 = total / (total + sigma_ests_jack[-1])
            h2_list.append(h_SNP_2)
            h2.append(h2_list)
        
        h2 = np.array(h2)
        
        h2_jackknife, h2_total = h2[:-1, :], h2[-1, :]

        return h2_jackknife, h2_total

    def compute_enrichment(self, h2_jackknife, h2_total):
        """
        Compute enrichment

        Returns: 
        enrichment_jackknife
            rows: jackknife sample
            cols: enrichment for one jackknife sample [enrichment_1^2 ... enrichment_n^2]
            e.g.,
            enrichment_jackknife[0][0]: enrichment for the 1st bin in the 1st jacknife sample
            enrichment_jackknife[0][1]: enrichment for the 1st bin in the 2st jacknife sample

        enrichment_total
            enrichment for the whole genotype matrix [enrichment_1^2 ... enrichment_n^2]

        """
        h2 = np.vstack([h2_jackknife, h2_total[np.newaxis, :]])

        enrichment = []

        for j in range(self.num_jack + 1):
            h2_jack = h2[j]
            enrichment_jack_bin = []
            for k in range (self.num_bin):
                hk_2 = h2_jack[k]
                h_SNP_2 = h2_jack[-1]
                M_k = self.M[j][k]
                M = sum(self.M[j])
                e_k = (hk_2 / h_SNP_2) / (M_k + M) if M_k + M != 0 else 0
                enrichment_jack_bin.append(e_k)
        
            enrichment.append(enrichment_jack_bin)

        enrichment = np.array(enrichment)
        
        enrichment_jackknife, enrichment_total = enrichment[:-1, :], enrichment[-1, :]

        return enrichment_jackknife, enrichment_total

    def __call__(self, method: str = "QR"):
        """
        whole RHE process for printing etc
        """

        self.pre_compute(verbose=True)

        sigma_est_jackknife, sigma_ests_total = self.estimate(method=method)

        sig_errs = self.estimate_error(sigma_est_jackknife)

        for i, est in enumerate(sigma_ests_total):
            if i == len(sigma_ests_total) - 1:
                print(f"residual variance: {est}, SE: {sig_errs[i]}")
            else:
                print(f"sigma^2 estimate for bin {i}: {est}, SE: {sig_errs[i]}")
        
        h2_jackknife, h2_total = self.compute_h2(sigma_est_jackknife, sigma_ests_total)

        h2_errs = self.estimate_error(h2_jackknife)

        for i, est_h2 in enumerate(h2_total):
            print(f"h^2 for bin {i}: {est_h2}, SE: {h2_errs[i]}")

        enrichment_jackknife, enrichment_total = self.compute_enrichment(h2_jackknife, h2_total)

        enrichment_errs = self.estimate_error(enrichment_jackknife)

        for i, est_enrichment in enumerate(enrichment_total):
            print(f"enrichment for bin {i}: {est_enrichment}, SE: {enrichment_errs[i]}")

        return sigma_ests_total, sig_errs, h2_total, h2_errs, enrichment_total, enrichment_errs

    def get_yxxy(self):
        if not self.yXXy:  
            raise ValueError("yxxy has not been computed yet")
        return self.yXXy