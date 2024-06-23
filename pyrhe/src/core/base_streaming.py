import time
import numpy as np
from abc import ABC, abstractmethod
import multiprocessing
from typing import Optional
from tqdm import tqdm
from typing import List, Tuple


from . import Base
from pyrhe.src.core.mp_handler import MultiprocessingHandler
from pyrhe.src.util.types import *
from pyrhe.src.util.logger import Logger


class StreamingBase(Base):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs) 
    
    @abstractmethod
    def aggregate(self):
        pass
    
    @abstractmethod
    def shared_memory(self):
        pass

    def _pre_compute_worker(self, worker_num, start_j, end_j, total_sample_queue=None):
        if self.multiprocessing:
            self._init_device(self.device_name, self.cuda_num)
        
        for j in range(start_j, end_j):
            if not self.multiprocessing:
                worker_num = 0
            self.log._debug(f"Worker {multiprocessing.current_process().name} processing jackknife sample {j}")
            start_whole = time.time()
            subsample, sub_annot = self._get_jacknife_subsample(j)
            subsample = self.impute_geno(subsample, simulate_geno=True)
            if total_sample_queue is not None:
                total_sample_queue.put((worker_num, subsample.shape[0]))
            else:
                self.total_num_sample = subsample.shape[0]
            all_gen = self.partition_bins(subsample, sub_annot)

            for k, X_kj in enumerate(all_gen):
                self.M[j][k] = self.M[self.num_jack][k] - X_kj.shape[1]
                self.pre_compute_jackknife_bin(j, k, X_kj, worker_num)
                
            end_whole = time.time()
            self.log._debug(f"jackknife {j} precompute (pass 1) total time: {end_whole-start_whole}")

    @abstractmethod
    def pre_compute_jackknife_bin_pass_2(self, j, k, X_kj):
        pass
    
    @abstractmethod
    def setup_lhs_rhs_jackknife(self, j):
        pass


    def _estimate_worker(self, worker_num, method, start_j, end_j, result_queue, trace_dict):
        sigma_ests = []
        for j in range(start_j, end_j):
            self.log._debug(f"Precompute (pass 2) for jackknife sample {j}")
            start_whole = time.time()
            if j != self.num_jack:
                subsample, sub_annot = self._get_jacknife_subsample(j)
                subsample = self.impute_geno(subsample, simulate_geno=True)
                all_gen = self.partition_bins(subsample, sub_annot)

            # calculate the stats for one jacknife subsample
            for k in range(self.num_bin):
                X_kj = all_gen[k] if j != self.num_jack else 0
                self.pre_compute_jackknife_bin_pass_2(j, k, X_kj)

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
            trace_dict = manager.dict() if self.get_trace else None

            mp_handler = MultiprocessingHandler(target=self._estimate_worker, work_ranges=work_ranges, device=self.device, trace_dict=trace_dict, method=method, streaming_estimate=True)
            mp_handler.start_processes()
            mp_handler.join_processes()
            results = mp_handler.get_queue()
            results.sort(key=lambda x: x[0])
            all_results = [item for _, result in results for item in result]

        else:
            self.result_queue = []
            trace_dict = {} if self.get_trace else None
            for j in tqdm(range(self.num_jack + 1), desc="Estimating..."):
                self._estimate_worker(0, method, j, j + 1, self.result_queue, trace_dict)
            all_results = self.result_queue
            del self.result_queue

        if self.get_trace:
            self.get_trace_summary(trace_dict)

        # Aggregate results
        sigma_ests = np.array(all_results)
        sigma_est_jackknife, sigma_ests_total = sigma_ests[:-1, :], sigma_ests[-1, :]
            
        return sigma_est_jackknife, sigma_ests_total