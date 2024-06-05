import os
import time
import torch
import traceback
import scipy
import atexit
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from typing import List, Tuple
from bed_reader import open_bed
from pyrhe.src.util.file_processing import *
from tqdm import tqdm
import multiprocessing
from multiprocessing import shared_memory, Manager
from pyrhe.src.core.mp_handler import MultiprocessingHandler
from pyrhe.src.util.mat_mul import *
from pyrhe.src.util.types import *
from pyrhe.src.util.logger import Logger




class Base(ABC):
    def __init__(
        self,
        geno_file: str,
        annot_file: str = None,
        pheno_file: str = None,
        cov_file: str = None,
        num_bin: int = 8,
        num_jack: int = 1,
        num_random_vec: int = 10,
        geno_impute_method: GenoImputeMethod = "binary",
        cov_impute_method: CovImputeMethod = "ignore",
        device: str = "cpu",
        cuda_num: Optional[int] = None,
        num_workers: Optional[int] = None,
        multiprocessing: bool = True,
        seed: Optional[int] = None,
        get_trace: bool = False,
        trace_dir: Optional[str] = None,
        samp_prev: Optional[float] = None,
        pop_prev: Optional[float] = None,
        log: Optional[Logger] = None
    ):

        self.num_jack = num_jack
        self.num_blocks = num_jack
        self.num_random_vec = num_random_vec
        self.num_bin = num_bin if annot_file is None else None
        self.geno_impute_methods = geno_impute_method
    

        ## Config
        self.log = log
        if self.log is None:
            self.log = Logger(debug_mode=False)

        self.log._debug(f"Using device {device}")
        self.multiprocessing = multiprocessing

        # atexit
        if self.multiprocessing:
            atexit.register(self._finalize)

        # Seed
        self.seed = int(time.process_time()) if seed is None else seed
        np.random.seed(self.seed)

        self.device_name, self.cuda_num = device, cuda_num
        self._init_device(self.device_name, self.cuda_num)

        # num workers: auto detect num cores 
        if self.device.type == 'cuda':
            total_workers = torch.cuda.get_device_properties(0).multi_processor_count
        else:
            total_workers = os.cpu_count()
        
        if num_workers is not None:
            if num_workers > total_workers:
                raise ValueError(f"The device only have {total_workers} cores but tried to specify {num_workers} workers")
            else:
                if num_workers > total_workers // 10:
                    self.log._debug(f"The device only have {total_workers} cores but tried to specify {num_workers} workers, recommend to decrease the worker count to avoid memory issues.")
                self.num_workers = num_workers
        else:
            self.num_workers = total_workers // 10
        self.log._debug(f"Number of workers: {self.num_workers}")


        # read fam and bim file
        self.geno_bed = open_bed(geno_file + ".bed")
        self.num_indv_original, fam_df = read_fam(geno_file + ".fam")
        self.num_snp = read_bim(geno_file + ".bim")


        # read annot file, generate one if not exist
        if annot_file is None:
            if self.num_bin is None:
                raise ValueError("Must specify number of bins if annot file is not provided")

            annot_file = "generated_annot"

            generate_annot(annot_file, self.num_snp, self.num_bin)

        self.num_bin, self.annot_matrix, self.len_bin = read_annot(annot_file, self.num_jack)

        # read pheno file and center
        self.pheno_file = pheno_file
        if pheno_file is not None:
            self.pheno, missing_indv, self.binary_pheno = read_pheno(pheno_file)
        else:
            self.pheno = None
            missing_indv = []
        
        self.num_traits = self.pheno.shape[1]
        self.log._log(f"Number of traits: {self.num_traits}")
    
        # read covariate file
        if cov_file is None:
            self.use_cov = False
            self.cov_matrix = None
            self.Q = None
            self.missing_indv = missing_indv
            if self.pheno is not None:
                self.pheno = np.delete(self.pheno, self.missing_indv, axis=0)
        else:
            self.use_cov = True
            self.cov_matrix, self.missing_indv = read_cov(cov_file, missing_indvs=missing_indv, cov_impute_method=cov_impute_method)
            if self.pheno is not None:
                self.pheno = np.delete(self.pheno, self.missing_indv, axis=0)
            self.Q = np.linalg.inv(self.cov_matrix.T @ self.cov_matrix)

        if self.pheno is not None:
            self.pheno = self.pheno - np.mean(self.pheno, axis=0) # center phenotype
        
        self.num_indv = self.num_indv_original - len(self.missing_indv)
        for idx, missing_idx in enumerate(self.missing_indv, start=1):
            col0_value = fam_df.iloc[missing_idx, 0]
            col1_value = fam_df.iloc[missing_idx, 1]
            self.log._log(f"missing individual {idx}: FID:{col0_value} IID:{col1_value}")
        
        self.log._log(f"Number of individuals after filtering: {self.num_indv}")
        if self.cov_matrix is not None:
            self.log._log(f"Number of covariates: {self.cov_matrix.shape[1]}")
        
        self.log._log("*****")
        for i, num_snps in enumerate(self.len_bin):
            self.log._log(f"Number of features in bin {i} : {num_snps}")

        # track subsample size
        if not self.multiprocessing:
            self.M = np.zeros((self.num_jack + 1, self.num_bin))
            self.M[self.num_jack] = self.len_bin

        # all_zb
        self.all_zb = np.random.randn(self.num_indv, self.num_random_vec)
        if self.use_cov:
            self.all_Uzb = self.cov_matrix @ self.Q @ (self.cov_matrix.T @ self.all_zb)

        # trace info
        self.get_trace = get_trace
        self.trace_dir = trace_dir
        if self.get_trace:
            if self.num_bin > 1:
                raise ValueError("Save trace failed, only supports saving tracing for single bin case.")
        
        # liability scale
        self.samp_prev = samp_prev
        self.pop_prev = pop_prev

        self.pheno_cp = self.pheno.copy()

        self.num_estimates = None
  
    
    def _init_device(self, device, cuda_num):
        if device != "cpu":
            if not torch.cuda.is_available():
                self.log._debug("cuda not available, fall back to cpu")
                self.device = torch.device("cpu")
            else:
                if cuda_num is not None and cuda_num > -1:
                    self.device = torch.device(f"cuda:{cuda_num}")
                else:
                    self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
    
    def simulate_pheno(self, sigma_list: List):
        '''
        Simulate phenotype y from X and sigma_list
        '''

        geno = self.geno_bed.read()
        geno = self.impute_geno(geno, simulate_geno=True)

        if len(sigma_list) != self.num_bin:
            
            raise ValueError("Number of elements in sigma list should be equal to number of bins")

        if len(sigma_list) == 1:
                sigma = sigma_list[0]
                y = np.zeros((self.num_indv,1))
                sigma_epsilon=1 - sigma # environmental effect sizes
                betas = np.random.randn(self.num_snp,1)*np.sqrt(sigma) # additive SNP effect sizes
                y += X@betas/np.sqrt(M)
                #self.log._debug(f'sigma_epislon={sigma_epsilon}')
                y += np.random.randn(self.num_snp,1)*np.sqrt(sigma_epsilon) # add the effect sizes

        else:
            all_gen = self.partition_bins(geno=geno, annot=self.annot_matrix)

            h = 1 - sum(sigma_list) # residual covariance
            sigma_epsilon = np.random.multivariate_normal([0] * self.num_indv, np.diag(np.full(self.num_indv, h)))
            sigma_epsilon = np.array(sigma_epsilon).reshape((len(sigma_epsilon), 1))
            y = np.zeros((self.num_indv,1))

            betas = []

            for i, data in enumerate(all_gen):
                X = data
                M = X.shape[1]
                sigma = sigma_list[i]
                beta = np.random.multivariate_normal([0] * M, np.diag(np.full(M, sigma / M)))
                beta = np.array(beta).reshape((len(beta), 1))
                betas.append(beta)
                y += X@beta
            
            y +=  sigma_epsilon 

        self.pheno = y

        if self.use_cov:
            Ncov = self.cov_matrix.shape[1]
            # Assume fixed gamma
            y += self.cov_matrix @ np.ones((Ncov, 1))
            # y += self.cov_matrix @ np.full((Ncov, 1), 0.1)

        self.pheno = np.delete(self.pheno, self.missing_indv, axis=0)
        self.pheno = self.pheno - np.mean(self.pheno)
        self.binary_pheno = False

        return y, betas

        
    def _simulate_geno_from_random(self, p_j):
        rval = np.random.random()
        dist_pj = [(1-p_j)*(1-p_j), 2*p_j*(1-p_j), p_j*p_j]
        
        if rval < dist_pj[0]:
            return 0
        elif rval >= dist_pj[0] and rval < (dist_pj[0] + dist_pj[1]):
            return 1
        else:
            return 2
        

    def impute_geno(self, X, simulate_geno: bool = True):
        N = X.shape[0]
        M = X.shape[1]
        X_imp = X
        if simulate_geno:
            for m in range(M):
                missing_mask = np.isnan(X[:, m])
                if self.geno_impute_methods == GenoImputeMethod.BINARY.value:
                    observed_mean = np.nanmean(X[:, m])
                    X_imp[missing_mask, m] = self._simulate_geno_from_random(observed_mean * 0.5)
                else:
                    X_imp[missing_mask, m] = 0 # mean imputation is just filling with 0 after standardization
                        
        means = np.mean(X_imp, axis=0)
        stds = 1/np.sqrt(means*(1-0.5*means))

        X_imp = (X_imp - means) * stds
              
        return X_imp

    def solve_linear_equation(self, X, y):
        '''
        Solve least square
        '''
        sigma = np.linalg.lstsq(X, y, rcond=None)[0]
        return sigma


    def solve_linear_qr(self, X, y):
        '''
        Solve least square using QR decomposition
        '''
        Q, R = scipy.linalg.qr(X)
        sigma = scipy.linalg.solve_triangular(R, np.dot(Q.T, y))
        return sigma


    def _bin_to_snp(self, annot): 
        bin_to_snp_indices = []

        for bin_index in range(self.num_bin):
            snp_indices = np.nonzero(annot[:, bin_index])[0]
            bin_to_snp_indices.append(snp_indices.tolist())

        return bin_to_snp_indices
        

    def partition_bins(self, geno: np.ndarray, annot: np.ndarray):
        """
        Partition the genotype matrix into num_bin bins
        """

        bin_to_snp_indices = self._bin_to_snp(annot)

        all_gen = []
        for i in range(self.num_bin):
            all_gen.append(geno[:, bin_to_snp_indices[i]])

        return all_gen
    
    def read_geno(self, start, end):
        try:
            if len(self.missing_indv) == 0:
                subsample = self.geno_bed.read(index=np.s_[::1,start:end])
            else:
                subsample_temp = self.geno_bed.read(index=np.s_[::1,start:end])
                subsample = np.delete(subsample_temp, self.missing_indv, axis=0)
                del subsample_temp
        except Exception as e:
            raise Exception(f"Error occurred: {e}")
        
        return subsample


    def _get_jacknife_subsample(self, jack_index: int) -> np.ndarray:
        """
        Get the jacknife subsample 
        """

        step_size = self.num_snp // self.num_jack
        step_size_rem = self.num_snp % self.num_jack
        chunk_size = step_size if jack_index < (self.num_jack - 1) else step_size + step_size_rem
        start = jack_index * step_size
        end = start + chunk_size

         # read bed file
        start_time = time.time()
        subsample = self.read_geno(start, end)
        end_time = time.time()
        # self.log._debug(f"read geno time: {end_time - start_time}")

        sub_annot = self.annot_matrix[start:end]

        return subsample, sub_annot
    

    def _get_annot_subsample(self, jack_index: int) -> np.ndarray:
        step_size = self.num_snp // self.num_jack
        step_size_rem = self.num_snp % self.num_jack
        chunk_size = step_size if jack_index < (self.num_jack - 1) else step_size + step_size_rem
        start = jack_index * step_size
        end = start + chunk_size
        mask = np.ones(self.num_snp, dtype=bool)
        mask[start:end] = False

        sub_annot = self.annot_matrix[mask]

        return sub_annot


    def regress_pheno(self, cov_matrix, pheno):
        """
        Project y onto the column space of the covariate matrix and compute the residual
        """
        Q = np.linalg.inv(cov_matrix.T @ cov_matrix)
        return pheno - cov_matrix @ (Q @ (cov_matrix.T @ pheno))
    
    def _compute_XXz(self, b, X_kj):
        random_vec = self.all_zb[:, b].reshape(-1, 1)
        return (mat_mul(X_kj, mat_mul(X_kj.T, random_vec, device=self.device), device=self.device)).flatten()
    
    def _compute_UXXz(self, XXz_kjb):
        return mat_mul(self.cov_matrix, mat_mul(self.Q, mat_mul(self.cov_matrix.T, XXz_kjb, device=self.device), device=self.device), device=self.device).flatten()
    
    def _compute_XXUz(self, b, X_kj):
        random_vec_cov = self.all_Uzb[:, b].reshape(-1, 1)
        return mat_mul(X_kj, mat_mul(X_kj.T, random_vec_cov, device=self.device), device=self.device).flatten()  

    def _compute_yXXy(self, X_kj, y):
        pheno = y if not self.use_cov else self.regress_pheno(self.cov_matrix, y)
        v = mat_mul(X_kj.T, pheno, device=self.device)
        return mat_mul(v.T, v, device=self.device)

    ### Shared Memory
    @abstractmethod
    def shared_memory(self):
        pass

    def create_shared_memory(self, arrays):
        for name, (shape, dtype) in arrays.items():
            shm = shared_memory.SharedMemory(create=True, size=np.prod(shape) * np.dtype(dtype).itemsize) if self.multiprocessing else None
            setattr(self, name, np.ndarray(shape, dtype=dtype, buffer=shm.buf if self.multiprocessing else None))
            if self.multiprocessing:
                setattr(self, f"{name}_shm", shm)

    def _setup_shared_memory(self):
        self.shared_memory()
        self.create_shared_memory(self.shared_memory_arrays)

        for name in self.shared_memory_arrays.keys():
            if name != "M":
                getattr(self, name).fill(0)

        self.M = getattr(self, "M")
        self.M.fill(0)
        self.M[self.num_jack] = self.M_last_row
    
    
    @abstractmethod
    def pre_compute_jackknife_bin(self, j, k, X_kj):
        pass
    
    @abstractmethod
    def get_num_estimates(self):
        pass
    
    def aggregate(self):
        for k in range(self.num_estimates):
            for j in range(self.num_jack):
                for b in range (self.num_random_vec):
                    self.XXz[k][self.num_jack][b] += self.XXz[k][j][b]
                self.yXXy[k][self.num_jack] += self.yXXy[k][j]
            
            for j in range(self.num_jack):
                for b in range (self.num_random_vec):
                    self.XXz[k][j][b] = self.XXz[k][self.num_jack][b] - self.XXz[k][j][b]
                self.yXXy[k][j] = self.yXXy[k][self.num_jack] - self.yXXy[k][j]
        
        if self.use_cov:
            for k in range(self.num_estimates):
                for j in range(self.num_jack):
                    for b in range (self.num_random_vec):
                        self.UXXz[k][self.num_jack][b] += self.UXXz[k][j][b]
                        self.XXUz[k][self.num_jack][b] += self.XXUz[k][j][b]
                                        
                for j in range(self.num_jack):
                    for b in range (self.num_random_vec):
                        self.UXXz[k][j][b] = self.UXXz[k][self.num_jack][b] - self.UXXz[k][j][b]
                        self.XXUz[k][j][b] = self.XXUz[k][self.num_jack][b] - self.XXUz[k][j][b]

    
    def _pre_compute_worker(self, worker_num, start_j, end_j):
        try:
            if self.multiprocessing:
                self._init_device(self.device_name, self.cuda_num)
            
            for j in range(start_j, end_j):
                self.log._debug(f"Worker {multiprocessing.current_process().name} processing jackknife sample {j}")
                np.random.seed(self.seed)
                start_whole = time.time()

                subsample, sub_annot = self._get_jacknife_subsample(j)

                start = time.time()
                subsample = self.impute_geno(subsample, simulate_geno=True)

                assert subsample.shape[0] == self.num_indv

                end = time.time()
                # self.log._debug(f"impute time: {end - start}")
                all_gen = self.partition_bins(subsample, sub_annot)

                self.pre_compute_jackknife_bin(j, all_gen)
                    
                end_whole = time.time()
                self.log._debug(f"jackknife {j} precompute total time: {end_whole - start_whole}")
        except Exception as e:
            self.log._debug(f"Error in worker {worker_num} processing range {start_j}-{end_j}: {e}")
            raise


    def _distribute_work(self, num_jobs, num_workers):
        jobs_per_worker = np.ceil(num_jobs / num_workers).astype(int)
        ranges = [(i * jobs_per_worker, min((i + 1) * jobs_per_worker, num_jobs)) for i in range(num_workers)]
        return ranges

    def pre_compute(self):
        from . import StreamingRHE

        start_whole = time.time()

        self._setup_shared_memory()

        if self.multiprocessing:
            work_ranges = self._distribute_work(self.num_jack, self.num_workers)

            processes = []

            mp_handler = MultiprocessingHandler(target=self._pre_compute_worker, work_ranges=work_ranges, device=self.device)
            mp_handler.start_processes()
            mp_handler.join_processes()
            
        else:
            for j in tqdm(range(self.num_jack), desc="Preprocessing jackknife subsamples..."):
                self._pre_compute_worker(0, j, j + 1)
        

        end_whole = time.time()

        self.log._debug(f"Precompute total time: {end_whole - start_whole}")
        self.aggregate()

    def _finalize(self):
        if self.multiprocessing:
            for name in self.shared_memory_arrays.keys():
                if hasattr(self, f'{name}_shm'):
                    shm = getattr(self, f'{name}_shm')
                    shm.close()
                    shm.unlink()
            
            # if hasattr(self, 'M_shm'):
            #     self.M_shm.close()
            #     self.M_shm.unlink()

    def setup_lhs_rhs_jackknife(self, j):
        T = np.zeros((self.num_estimates+1, self.num_estimates+1))
        q = np.zeros((self.num_estimates+1, 1))

        for k_k in range(self.num_estimates):
            for k_l in range(self.num_estimates):
                M_k = self.M[j][k_k]
                M_l = self.M[j][k_l]
                B1 = self.XXz[k_k][j]
                B2 = self.XXz[k_l][j]
                T[k_k, k_l] += np.sum(B1 * B2)

                if self.use_cov:
                    h1 = self.cov_matrix.T @ B1.T
                    h2 = self.Q @ h1
                    h3 = self.cov_matrix @ h2
                    trkij_res1 = np.sum(h3.T * B2)

                    B1 = self.XXUz[k_k][j]
                    B2 = self.UXXz[k_l][j]
                    trkij_res2 = np.sum(B1 * B2)
                
                    T[k_k, k_l] += (trkij_res2 - 2 * trkij_res1)


                T[k_k, k_l] /= (self.num_random_vec)
                T[k_k, k_l] =  T[k_k, k_l] / (M_k * M_l) if (M_k * M_l) != 0 else 0


        for k in range(self.num_estimates):
            M_k = self.M[j][k]

            if not self.use_cov:
                T[k, self.num_estimates] = self.num_indv
                T[self.num_estimates, k] = self.num_indv

            else:
                B1 = self.XXz[k][j]
                C1 = self.all_Uzb
                tk_res = np.sum(B1 * C1.T)
                tk_res = 1/(self.num_random_vec * M_k) * tk_res

                T[k, self.num_estimates] = self.num_indv - tk_res
                T[self.num_estimates, k] = self.num_indv - tk_res

            q[k] = self.yXXy[(k, j)] / M_k if M_k != 0 else 0
        
        
        T[self.num_estimates, self.num_estimates] = self.num_indv if not self.use_cov else self.num_indv - self.cov_matrix.shape[1]

        pheno = self.pheno if not self.use_cov else self.regress_pheno(self.cov_matrix, self.pheno)
        q[self.num_estimates] = pheno.T @ pheno

        return T, q
            
    def estimate(self, method: str = "lstsq") -> Tuple[List[List], List]:
        """
        Actual RHE estimation for sigma^2
        Returns: 
        sigma_est_jackknife: 
            rows: jackknife sample
            cols: sigma^2 for one jackknife sample [sigma_1^2 sigma_2^2 ... sigma_e^2]
            e.g.,
            sigma_est_jackknife[0][0]: sigma_2 estimate for the 1st bin in the 1st jacknife sample
            sigma_est_jackknife[0][1]: sigma_2 estimate for the 1st bin in the 2nd jacknife sample

        sigma_ests_total
            sigma^2 for the whole genotype matrix [sigma_1^2 sigma_2^2 ... sigma_e^2]

        """
        sigma_ests = []
        if self.get_trace:
            trace_dict = {}

        for j in range(self.num_jack + 1):
            self.log._debug(f"Estimate for jackknife sample {j}")

            T, q = self.setup_lhs_rhs_jackknife(j)

            if self.get_trace:
                trace_dict[j] = ((T[0][0], self.M[j]))

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
            
        sigma_ests = np.array(sigma_ests)

        sigma_est_jackknife, sigma_ests_total = sigma_ests[:-1, :], sigma_ests[-1, :]

        if self.get_trace:
            self.get_trace_summary(trace_dict)
            
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

    def compute_h2_nonoverlapping(self, sigma_est_jackknife: List[List], sigma_ests_total: List) -> Tuple[List[List], List]:
        """
        Compute h^2

        Returns: 
        h2_jackknife
            rows: jackknife sample
            cols: h^2 for one jackknife sample [h_1^2 h_2^2 ... h_SNP^2]
            e.g.,
            h2_jackknife[0][0]: h^2 for the 1st bin in the 1st jacknife sample
            h2_jackknife[0][1]: h^2 for the 1st bin in the 2nd jacknife sample

        h2_total
            h^2 for the whole genotype matrix [h_1^2 h_2^2 ... h_SNP^2]

        """

        sigma_ests = np.vstack([sigma_est_jackknife, sigma_ests_total[np.newaxis, :]])

        h2 = []

        for j in range(self.num_jack + 1):
            sigma_ests_jack = sigma_ests[j]
            total = 0
            for k in range (self.num_estimates):
                total += sigma_ests_jack[k]
            assert sigma_ests_jack[-1] == sigma_ests_jack[k+1]
            h2_list = [x / (total + sigma_ests_jack[-1])  for x in sigma_ests_jack[:-1]]
            # compute h_SNP^2
            h_SNP_2 = total / (total + sigma_ests_jack[-1])
            h2_list.append(h_SNP_2)
            h2.append(h2_list)
        
        h2 = np.array(h2)
        
        h2_jackknife, h2_total = h2[:-1, :], h2[-1, :]

        return h2_jackknife, h2_total

    def compute_h2_overlapping(self, sigma_est_jackknife, sigma_ests_total):

        sigma_ests = np.vstack([sigma_est_jackknife, sigma_ests_total[np.newaxis, :]])

        h2 = []

        for j in range(self.num_jack + 1):
            sigma_ests_jack = sigma_ests[j]
            total = 0
            for k in range (self.num_estimates):
                total += sigma_ests_jack[k]
            assert sigma_ests_jack[-1] == sigma_ests_jack[k+1]

            h2_list = []

            sub_annot = self._get_annot_subsample(jack_index=j)

            for k in range(self.num_estimates):
                snp_indices = np.where(sub_annot[:, k] == 1)[0]
                total_k = 0
                for snp_idx in snp_indices:
                    snp_bins = np.where(sub_annot[snp_idx, :] == 1)[0].tolist() # A list of bins that this snp belongs to 
                    total_inner = 0
                    for bin in snp_bins:
                        total_inner += sigma_ests_jack[bin] / self.M[j][bin] if self.M[j][bin] != 0 else 0
                    total_k += total_inner

                total_k /= (total + sigma_ests_jack[-1])
                h2_list.append(total_k)
            
            del sub_annot

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
            enrichment_jackknife[0][1]: enrichment for the 1st bin in the 2nd jacknife sample

        enrichment_total
            enrichment for the whole genotype matrix [enrichment_1^2 ... enrichment_n^2]

        """
        h2 = np.vstack([h2_jackknife, h2_total[np.newaxis, :]])

        enrichment = []

        for j in range(self.num_jack + 1):
            h2_jack = h2[j]
            enrichment_jack_bin = []
            for k in range (self.num_estimates):
                hk_2 = h2_jack[k]
                h_SNP_2 = h2_jack[-1]
                M_k = self.M[j][k]
                M = sum(self.M[j])
                e_k = (hk_2 / h_SNP_2) / (M_k / M) if (M != 0 and M_k != 0) else 0
                enrichment_jack_bin.append(e_k)
        
            enrichment.append(enrichment_jack_bin)

        enrichment = np.array(enrichment)
        
        enrichment_jackknife, enrichment_total = enrichment[:-1, :], enrichment[-1, :]

        return enrichment_jackknife, enrichment_total
    
    def get_trace_summary(self, trace_dict):
        pheno_path = os.path.basename(self.pheno_file) if self.pheno_file is not None else None
        trace_filename = f"run_{pheno_path}.trace"
        mn_filename = f"run_{pheno_path}.MN"
        
        if self.trace_dir and os.path.isdir(self.trace_dir):
            trace_filepath = os.path.join(self.trace_dir, trace_filename)
            mn_filepath = os.path.join(self.trace_dir, mn_filename)
        else:
            trace_filepath = trace_filename
            mn_filepath = mn_filename
        
        with open(trace_filepath, 'w') as file:
            file.write("TRACE,NSNPS_JACKKNIFE\n")
            for key in sorted(trace_dict.keys()):
                value = trace_dict[key]
                line = f"{value[0]:.1f},{int(value[1][0])}\n"
                file.write(line)

        mn_content = f"NSAMPLE,NSNPS,NBLKS\n{self.num_indv},{self.num_snp},{self.num_blocks}"
        with open(mn_filepath, 'w') as file:
            file.write(mn_content)

        self.log._debug(f"Trace saved to {trace_filepath}")
        self.log._debug(f"MN data saved to {mn_filepath}")   

    def _compute_liability_h2(self, h2, seh2): 
        # https://gist.github.com/nievergeltlab/fb8a20feded72030907a9b4e81d1c6ea 
        from scipy.stats import norm, chi2
        K = self.pop_prev
        P = self.samp_prev
        zv = norm.pdf(norm.ppf(K))

        h2_liab = h2 * K**2 * (1 - K)**2 / P / (1 - P) / zv**2
        var_h2_liab = (seh2 * K**2 * (1 - K)**2 / P / (1 - P) / zv**2) ** 2
        p_liab = chi2.sf(h2_liab**2 / var_h2_liab, 1)

        return h2_liab, var_h2_liab**0.5, p_liab

    def __call__(self, trait, method: str = "QR"):
        
        self.pheno = self.pheno_cp[:, trait].reshape(-1, 1)
        self.log._log("*****")
        self.log._log(f"OUTPUT FOR TRAIT {trait}: ")
        self.pre_compute()

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
        
        if trait < self.num_traits - 1:
            self._finalize()

        return sigma_ests_total, sig_errs, h2_total, h2_errs, enrichment_total, enrichment_errs, h2_total_overlap, h2_errs_overlap, enrichment_total_overlap, enrichment_errs_overlap
    

    ############################ Other Functionalities #########################################

    def block_Xz(self, bins_info, block_results_dict, j):
        num_snps = self.step_size
        if j == (self.num_blocks - 1):
            num_snps = self.step_size + (self.num_snp % self.step_size)

        Zs = np.random.normal(size=(num_snps, self.num_random_vec))

        start_index = j*self.step_size
        end_index = start_index + num_snps
        
        bins_info[j] = (start_index, end_index, num_snps)
        geno = self.read_geno(start_index, end_index)
        geno = self.impute_geno(geno, simulate_geno=True)
        
        if self.jackknife_blocks:
            block_results_dict[j] = geno @ Zs

        return geno @ Zs
    

    def block_XtXz(self, bins_info, results_dict, block_results_dict, all_blocks_results_dict, j):
        start_index, end_index, num_snps = bins_info[j]
        geno = self.read_geno(start_index, end_index)
        geno = self.impute_geno(geno, simulate_geno=True)
        
        results_dict[j] = (geno.T) @ self.temp_results

        if self.jackknife_blocks:
            for i in range(self.num_blocks):
                if i == j: continue
                else:
                    all_blocks_results_dict[(j, i)] = (geno.T) @ block_results_dict[j]

    def get_XtXz(self, output, jackknife_blocks=True):
        """
        Get XtXz trace estimate
        """
        from itertools import repeat

        self.jackknife_blocks = jackknife_blocks
        self.temp_results = np.zeros((self.num_indv, self.num_random_vec))
        results = np.zeros((self.num_snp, self.num_random_vec))
        self.step_size = self.num_snp // self.num_blocks

        bins_info = Manager().dict()
        results_dict = Manager().dict()

        if jackknife_blocks:
            block_results_dict = Manager().dict()

        with multiprocessing.Pool(self.num_workers) as pool:
            with tqdm(total=self.num_blocks) as pbar:
                pbar.set_description("Calculating Xz")
                args_iter = zip(repeat(bins_info), repeat(block_results_dict), range(self.num_blocks))
                for i, partial in enumerate(pool.starmap(self.block_Xz, args_iter)):
                    self.temp_results += partial
                    pbar.update()
        pool.join()

        if jackknife_blocks:
            for j in range(self.num_blocks):
                block_results_dict[j] = self.temp_results - block_results_dict[j]
            
            all_blocks_results_dict = Manager().dict()

        with multiprocessing.Pool(self.num_workers) as pool:
            with tqdm(total=self.num_blocks) as pbar:
                pbar.set_description("Calculating XtXz")
                args_iter = zip(repeat(bins_info), repeat(results_dict), repeat(block_results_dict), repeat(all_blocks_results_dict), range(self.num_blocks))
                for i, partial in enumerate(pool.starmap(self.block_XtXz, args_iter)):
                    pbar.update()
        pool.join()

        for j in range(self.num_blocks):
            start_index, end_index, _ = bins_info[j]
            results[start_index:end_index, :] = results_dict[j]
            
        
        if jackknife_blocks:
            jackknife_results = dict()
            for j in range(self.num_blocks):
                temp = []
                for i in range(self.num_blocks):
                    if i == j: continue
                    else:
                        temp.append(all_blocks_results_dict[(j, i)])
                jackknife_results[j] = np.row_stack(temp)

        trace_est = np.square(results).sum()/(self.num_random_vec*self.num_snp*self.num_snp)
        self.log._debug(f"The trace estimate is {trace_est}")

        if jackknife_blocks:
            for j in range(self.num_blocks):
                jackknife_result = jackknife_results[j]
                M_snps = np.shape(jackknife_result)[0]
                jackknife_trace_est = np.square(jackknife_result).sum()/(self.num_random_vec*M_snps*M_snps)
                
                self.log._debug(f"The trace estimate of {j}-th jackknife block is {jackknife_trace_est}")

        with open(f"{output}.txt.bin", 'wb') as f:
            results.tofile(f)

        if jackknife_blocks:
            for j in range(self.num_blocks):
                with open(f"{output}.jack_{j}.txt.bin", "wb") as f:
                    jackknife_results[j].tofile(f)