import time
import numpy as np
from abc import abstractmethod
import multiprocessing
from tqdm import tqdm
from typing import List, Tuple
from . import Base
from pyrhe.src.core.mp_handler import MultiprocessingHandler
from pyrhe.src.util.types import *
from pyrhe.src.util.mat_mul import *


class StreamingBase(Base):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs) 
    
    
    def aggregate(self):
        # [k][0][b] stores the sum
        if self.multiprocessing:
            for k in range(self.num_estimates):
                if k < self.num_estimates - self.num_env: # The hetero noise does not have to be calculated in this way
                    self.yXXy_per_jack[k][0] = np.sum(self.yXXy[k, :])
                    for b in range(self.num_random_vec):
                        self.XXz_per_jack[k, 0, b, :] = np.sum(self.XXz[k, :, b, :], axis=0)
                        if self.UXXz_per_jack is not None: 
                            self.UXXz_per_jack[k, 0, b, :] = np.sum(self.UXXz[k, :, b, :], axis=0) 
                        if self.XXUz_per_jack is not None:
                            self.XXUz_per_jack[k, 0, b, :] = np.sum(self.XXUz[k, :, b, :], axis=0)
            
            self.XXz = self.XXz_per_jack
            self.yXXy = self.yXXy_per_jack 
            self.UXXz = self.UXXz_per_jack
            self.XXUz = self.XXUz_per_jack
        
            del self.XXz_per_jack
            del self.yXXy_per_jack 
            del self.UXXz_per_jack
            del self.XXUz_per_jack

        # Only a single calculation for the hetero noise. No need to aggregate across the workers.  
        for e in range(self.num_env):
            k = e + self.num_bin + self.num_gen_env_bin
            X_kj = elem_mul(np.eye(self.num_indv), self.env[:, e].reshape(-1, 1), device=self.device)
            for b in range(self.num_random_vec):
                self.XXz[k][0][b] = self._compute_XXz(b, X_kj)
                self.XXz[k][1][b] = self.XXz[k][0][b] # No need to do the jackknife sampling for the hetero noise. 
            self.yXXy[k][0] = self._compute_yXXy(X_kj, y=self.pheno)
            self.yXXy[k][1] = self.yXXy[k][0]

            if self.use_cov:
                self.UXXz[k][0][b] = self._compute_UXXz(self.XXz[k][0][b])
                self.UXXz[k][1][b] = self.UXXz[k][0][b]
                self.XXUz[k][0][b] = self._compute_XXUz(b, X_kj)
                self.XXUz[k][1][b] = self.XXUz[k][0][b]

    
    def shared_memory(self):
        self.shared_memory_arrays = {
            "XXz":  ((self.num_estimates, self.num_workers, self.num_random_vec, self.num_indv), np.float64),
            "yXXy": ((self.num_estimates, self.num_workers), np.float64),
            "M":    ((self.num_jack + 1, self.num_estimates), np.int64)
        }
        if self.use_cov:
            self.shared_memory_arrays.update({
                "UXXz": ((self.num_estimates, self.num_workers, self.num_random_vec, self.num_indv), np.float64),
                "XXUz": ((self.num_estimates, self.num_workers, self.num_random_vec, self.num_indv), np.float64),
            })
        
        if self.multiprocessing:
            # Resulting matrix after aggregating the results from each worker
            self.shared_memory_arrays.update({
                "XXz_per_jack": ((self.num_estimates, 2, self.num_random_vec, self.num_indv), np.float64),
                "yXXy_per_jack": ((self.num_estimates, 2), np.float64),
            })
            if self.use_cov:
                self.shared_memory_arrays.update({
                    "UXXz_per_jack": ((self.num_estimates, 2, self.num_random_vec, self.num_indv), np.float64),
                    "XXUz_per_jack": ((self.num_estimates, 2, self.num_random_vec, self.num_indv), np.float64),
                })

    def _pre_compute_worker(self, worker_num, start_j, end_j, total_sample_queue=None):
        if self.multiprocessing:
            self._init_device(self.device_name, self.cuda_num)
        
        for j in range(start_j, end_j):
            if not self.multiprocessing:
                worker_num = 0
            self.log._debug(f"Worker {multiprocessing.current_process().name} processing jackknife sample {j}")
            start_whole = time.time()
            subsample, sub_annot = self._get_jacknife_subsample(j)
            subsample = self.impute_geno(subsample)
            if total_sample_queue is not None:
                total_sample_queue.put((worker_num, subsample.shape[0]))
            else:
                self.total_num_sample = subsample.shape[0]
            all_gen = self.partition_bins(subsample, sub_annot)
            self.pre_compute_jackknife_bin(j, all_gen, worker_num)
                
            end_whole = time.time()
            self.log._debug(f"jackknife {j} precompute (pass 1) total time: {end_whole-start_whole}")

    @abstractmethod
    def pre_compute_jackknife_bin_pass_2(self, j, k, X_kj):
        pass

    def _estimate_worker(self, worker_num, method, start_j, end_j, result_queue, trace_sum):
        sigma_ests = []
        for j in range(start_j, end_j):
            self.log._debug(f"Precompute (pass 2) for jackknife sample {j}")
            start_whole = time.time()
            if j != self.num_jack:
                subsample, sub_annot = self._get_jacknife_subsample(j)
                subsample = self.impute_geno(subsample)
                all_gen = self.partition_bins(subsample, sub_annot)
            self.pre_compute_jackknife_bin_pass_2(j, all_gen)

            self.log._debug(f"Estimate for jackknife sample {j}")
            
            T, q = self.setup_lhs_rhs_jackknife(j, trace_sum, is_streaming=True)
            if self.get_trace:
                trace_sum[j] = ((T[0][0], self.M[j]))
        
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

            end_whole = time.time()
            self.log._debug(f"estimate time for jackknife subsample: {end_whole - start_whole}")

        if self.multiprocessing:
            result_queue.put((worker_num, sigma_ests)) # ensure in order
        else:
            result_queue.extend(sigma_ests)


    def estimate(self, method: str = "lstsq") -> Tuple[List[List], List]:
        if self.multiprocessing:
            work_ranges = self._distribute_work(self.num_jack + 1, self.num_workers)
            manager = multiprocessing.Manager()
            trace_sums = manager.dict() if self.get_trace else None

            mp_handler = MultiprocessingHandler(target=self._estimate_worker, work_ranges=work_ranges, device=self.device, trace_sums=trace_sums, method=method, streaming_estimate=True)            
            mp_handler.start_processes()
            mp_handler.join_processes()
            results = mp_handler.get_queue()
            results.sort(key=lambda x: x[0])
            all_results = [item for _, result in results for item in result]

        else:
            self.result_queue = []
            trace_sums = np.zeros((self.num_jack+1, self.num_bin, self.num_bin)) if self.get_trace else None
            for j in tqdm(range(self.num_jack + 1), desc="Estimating..."):
                self._estimate_worker(0, method, j, j + 1, self.result_queue, trace_sums)
            all_results = self.result_queue
            del self.result_queue

        if self.get_trace:
            self.get_trace_summary(trace_sums)

        # Aggregate results
        sigma_ests = np.array(all_results)
        sigma_est_jackknife, sigma_ests_total = sigma_ests[:-1, :], sigma_ests[-1, :]
            
        return sigma_est_jackknife, sigma_ests_total