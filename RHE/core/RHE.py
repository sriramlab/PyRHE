"""
Non streaming version of RHE
"""
import numpy as np
from typing import List, Tuple
from numpy import ndarray
from bed_reader import open_bed
from RHE.util.math import *
from RHE.util.file_processing import *


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
        num_bin: int = 8,
        num_jack: int = 1,
        num_random_vec: int = 10,
        standardize: bool = True,
        verbose: bool = True,
        seed: int = 0
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

        # read bed file
        try:
            bed = open_bed(geno_file + ".bed")
            self.geno = bed.read()
        except FileNotFoundError:
            print("The .bed file could not be found.")
        except IOError:
            print("An IO error occurred while reading the .bed file.")
        except Exception as e:
            print(f"Error occurred: {e}")

        self.num_indv = self.geno.shape[0]
        self.num_snp = self.geno.shape[1]
        
        # impute geno
        self.geno = impute_geno(self.geno, standardize=standardize)

        # read bim file
        assert self.num_snp == read_bim(geno_file + ".bim")

        # read fam file
        assert self.num_indv == read_fam(geno_file + ".fam")

        # read pheno file
        if pheno_file is not None:
            self.pheno = read_pheno(pheno_file)
            assert self.num_ind == self.pheno[0]
        else:
            self.pheno = None

         # read annot file, generate one if not exist
        if annot_file is None:
            if self.num_bin is None:
                raise ValueError("Must specify number of bins if annot file is not provided")

            annot_file = "generated_annot"

            generate_annot(annot_file, self.num_snp, self.num_bin)

        self.num_bin, self.annot_matrix = read_annot(annot_file, self.num_jack)

        self.H = {}
        self.Z = {}
        self.U = {}
        self.V = {}
        self.M = np.zeros((self.num_jack + 1, self.num_bin)) # number of geno in each bin in each jackknife subsample 
                                                             # last dim is total

        self.M = np.zeros((self.num_jack + 1, self.num_bin))
    
    def simulate_pheno(self, sigma_list: List):
        '''
        Simulate phenotype y from X and sigma_list
        '''

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
            all_gen = self.partition_bins(self.geno)
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
        

    def partition_bins(self, gen):
        """
        Partition the genotype matrix into num_bin bins
        """

        bin_to_snp_indices = self._bin_to_snp()
        
        all_gen = [GenoChunk() for _ in range(len(bin_to_snp_indices))]

        for i, data in enumerate(all_gen):
            snp_indices = bin_to_snp_indices[i]            
            valid_indices = [idx for idx in snp_indices if not np.isnan(gen[0][idx])]
            data.num_snp = len(valid_indices)
            data.gen = gen[:, valid_indices]
        
        return all_gen

        
    def _get_jacknife_subsample(self, gen: np.ndarray, jack_index: int) -> np.ndarray:
        """
        Get the genotype matrix with the (jack_index)th subsample masked
        """

        step_size = self.num_snp // self.num_jack
        step_size_rem = self.num_snp % self.num_jack
        chunk_size = step_size if jack_index < (self.num_jack - 1) else step_size + step_size_rem
        start = jack_index * step_size
        end = start + chunk_size
        
        gen_masked = gen.copy()
        gen_masked[:, :start] = np.nan
        gen_masked[:, end:] = np.nan
        
        return gen_masked
    

    def _get_actual_jacknife_subsample(self, gen: np.ndarray, jack_index: int) -> np.ndarray:
        """
        Get the actual jackknife subsample by excluding the jackknife subsample
        obtained using _get_jacknife_subsample
        """

        step_size = self.num_snp // self.num_jack
        step_size_rem = self.num_snp % self.num_jack
        chunk_size = step_size if jack_index < (self.num_jack - 1) else step_size + step_size_rem
        start = jack_index * step_size
        end = start + chunk_size
        
        gen_masked = gen.copy()
        gen_masked[:, start:end] = np.nan
        
        return gen_masked
        
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
            if j != self.num_jack:
                geno_chunk = self._get_actual_jacknife_subsample(self.geno, j)
                geno_chunk_all_gen = self.partition_bins(geno_chunk)
            else:
                geno_chunk_all_gen = self.partition_bins(self.geno)


            T = np.zeros((self.num_bin+1, self.num_bin+1))
            q = np.zeros((self.num_bin+1, 1))

            for k_k, geno_k in enumerate(geno_chunk_all_gen):
                for k_l, geno_l in enumerate(geno_chunk_all_gen):
                    M_k = geno_k.num_snp
                    M_l = geno_l.num_snp
                    for b in range(self.num_random_vec):
                        T[k_k, k_l] += (self.U[(k_k, b, j)]).T @ self.U[(k_l, b, j)]
                    T[k_k, k_l] /= (self.num_random_vec)
                    T[k_k, k_l] =  T[k_k, k_l] / (M_k * M_l) if (M_k * M_l) != 0 else 0

            for k, geno in enumerate(geno_chunk_all_gen):
                M_k = geno.num_snp
                self.M[j][k] = M_k # record this so don't have to extract the corresponding num_snp for the chunk every time
                Xi = (geno.gen).copy()/np.sqrt(M_k)
                T[k, self.num_bin] = np.trace(Xi@Xi.T) if M_k != 0 else 0
                T[self.num_bin, k] = np.trace(Xi@Xi.T) if M_k != 0 else 0
                q[k] = self.V[(k, j)]
                q[k] = q[k] / M_k if M_k != 0 else 0

            T[self.num_bin, self.num_bin] = self.num_indv
            q[self.num_bin] = self.pheno.T @ self.pheno

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

        self.pre_compute()

        sigma_est_jackknife, sigma_ests_total = self.estimate(method=method)

        sig_errs = self.estimate_error(sigma_est_jackknife)

        for i, est in enumerate(sigma_ests_total):
            if i == len(sigma_ests_total) - 1:
                print(f"residual variance: {est}, SE: {sig_errs[i]}")
            else:
                print(f"sigma^2 estimate for bin {i}: {est}, SE: {sig_errs[i]}")
        
        h2_jackknife, h2_total = self.compute_h2(sigma_est_jackknife, sigma_ests_total)

        sig_errs_h2 = self.estimate_error(h2_jackknife)

        for i, est_h2 in enumerate(h2_total):
            print(f"h^2 for bin {i}: {est_h2}, SE: {sig_errs_h2[i]}")

        enrichment_jackknife, enrichment_total = self.compute_enrichment(h2_jackknife, h2_total)

        sig_errs_enrichment = self.estimate_error(enrichment_jackknife)

        for i, est_enrichment in enumerate(enrichment_total):
            print(f"enrichment for bin {i}: {est_enrichment}, SE: {sig_errs_enrichment[i]}")

    def get_yxxy(self):
        if not self.H:  
            raise ValueError("yxxy has not been computed yet")
        return self.H