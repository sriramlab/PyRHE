from pyrhe.src.base import StreamingBase
from pyrhe.src.models.genie import GENIE
from pyrhe.src.util.mat_mul import *
from typing import Tuple, List
import time
import multiprocessing
from pyrhe.src.util import MultiprocessingHandler
from tqdm import tqdm

class StreamingGENIE(GENIE, StreamingBase):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs) 

    def pre_compute_jackknife_bin(self, j, all_gen, worker_num):
        # G
        for k, X_kj in enumerate(all_gen): 
            X_kj = self.standardize_geno(X_kj)
            self.M[j][k] = self.M[self.num_jack][k] - X_kj.shape[1]
            for b in range(self.num_random_vec):
                self.XXz[k][worker_num][b] += self._compute_XXz(b, X_kj)

                if self.use_cov:
                    self.UXXz[k][worker_num][b] += self._compute_UXXz(self.XXz[k][worker_num][b])
                    self.XXUz[k][worker_num][b] += self._compute_XXUz(b, X_kj)
            
            yXXy_kj = self._compute_yXXy(X_kj, y=self.pheno)
            self.yXXy[k][worker_num] += yXXy_kj[0][0]
                
        
        # GxE
        if self.genie_model == "G+GxE" or self.genie_model == "G+GxE+NxE":
            for e in range(self.num_env):
                for k, X_kj in enumerate(all_gen): 
                    X_kj = self.standardize_geno(X_kj)
                    k_gxe = (e + 1) * k + self.num_bin
                    self.M[j][k_gxe] = self.M[self.num_jack][k_gxe] - X_kj.shape[1]
                    X_kj_gxe = elem_mul(X_kj, self.env[:, e].reshape(-1, 1), device=self.device)  # Avoid modifying X_kj
                    for b in range(self.num_random_vec):
                        self.XXz[k_gxe][worker_num][b] += self._compute_XXz(b, X_kj_gxe)

                        if self.use_cov:
                            self.UXXz[k_gxe][worker_num][b] += self._compute_UXXz(self.XXz[k_gxe][worker_num][b])
                            self.XXUz[k_gxe][worker_num][b] += self._compute_XXUz(b, X_kj_gxe)
                    
                    yXXy_kj = self._compute_yXXy(X_kj_gxe, y=self.pheno)
                    self.yXXy[k_gxe][worker_num] += yXXy_kj[0][0]
                
        # NxE
        if self.genie_model == "G+GxE+NxE":
            for e in range(self.num_env):
                k = e + self.num_bin + self.num_gen_env_bin
                self.M[j][k] = 1

    def pre_compute_jackknife_bin_pass_2(self, j, all_gen):
        # G
        for k, X_kj in enumerate(all_gen): 
            X_kj = self.standardize_geno(X_kj)
            for b in range (self.num_random_vec):
                XXz_kb = self._compute_XXz(b, X_kj) if j != self.num_jack else 0
                if self.use_cov:
                    UXXz_kb = self._compute_UXXz(XXz_kb) if j != self.num_jack else 0
                    self.UXXz[k][1][b] = self.UXXz[k][0][b] - UXXz_kb
                    XXUz_kb = self._compute_XXUz(b, X_kj) if j != self.num_jack else 0
                    self.XXUz[k][1][b] = self.XXUz[k][0][b] - XXUz_kb
                self.XXz[k][1][b] = self.XXz[k][0][b] - XXz_kb
            
            yXXy_k = (self._compute_yXXy(X_kj, y=self.pheno))[0][0] if j != self.num_jack else 0
            self.yXXy[k][1] = self.yXXy[k][0] - yXXy_k
        
        # GxE
        if self.genie_model == "G+GxE" or self.genie_model == "G+GxE+NxE":
            for e in range(self.num_env):
                for k, X_kj in enumerate(all_gen): 
                    X_kj = self.standardize_geno(X_kj)
                    k_gxe = (e + 1) * k + self.num_bin
                    X_kj_gxe = elem_mul(X_kj, self.env[:, e].reshape(-1, 1), device=self.device)
                    for b in range(self.num_random_vec):
                        XXz_kb = self._compute_XXz(b, X_kj_gxe) if j != self.num_jack else 0
                        if self.use_cov:
                            UXXz_kb = self._compute_UXXz(XXz_kb) if j != self.num_jack else 0
                            self.UXXz[k_gxe][1][b] = self.UXXz[k_gxe][0][b] - UXXz_kb
                            XXUz_kb = self._compute_XXUz(b, X_kj_gxe) if j != self.num_jack else 0
                            self.XXUz[k_gxe][1][b] = self.XXUz[k_gxe][0][b] - XXUz_kb
                        self.XXz[k_gxe][1][b] = self.XXz[k_gxe][0][b] - XXz_kb
                    yXXy_k = (self._compute_yXXy(X_kj_gxe, y=self.pheno))[0][0] if j != self.num_jack else 0
                    self.yXXy[k_gxe][1] = self.yXXy[k_gxe][0] - yXXy_k
    
    def _estimate_worker(self, worker_num, method, start_j, end_j, result_queue, trace_sum):
        """
        Adjust the estimate method to also pass in the adjusted estimates by traces for heritability calculation
        """
        sigma_ests = []
        sigma_ests_adj = []
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

            sigma_est_adj = []
            for i in range(len(sigma_est)):
                sigma_est_adj.append(sigma_est[i] * T[i, self.num_estimates])
            sigma_ests_adj.append(sigma_est_adj)

            end_whole = time.time()
            self.log._debug(f"estimate time for jackknife subsample: {end_whole - start_whole}")

        if self.multiprocessing:
            result_queue.put((worker_num, sigma_ests, sigma_ests_adj)) # ensure in order
        else:
            result_queue.extend((sigma_ests, sigma_ests_adj))


    def estimate(self, method: str = "lstsq") -> Tuple[List[List], List]:
        """
        Adjust the estimate method in the base class to include the effect of traces for heritability calculation
        """
        if self.multiprocessing:
            work_ranges = self._distribute_work(self.num_jack + 1, self.num_workers)
            manager = multiprocessing.Manager()
            trace_sums = manager.dict() if self.get_trace else None

            mp_handler = MultiprocessingHandler(target=self._estimate_worker, work_ranges=work_ranges, device=self.device, trace_sums=trace_sums, method=method, streaming_estimate=True)            
            mp_handler.start_processes()
            mp_handler.join_processes()
            results = mp_handler.get_queue()
            results.sort(key=lambda x: x[0])
            all_results = [item for _, result, _ in results for item in result]
            all_results_adj = [item for _, _, result in results for item in result]
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
        sigma_ests_adj = np.array(all_results_adj)
        sigma_est_jackknife_adj, sigma_ests_total_adj = sigma_ests_adj[:-1, :], sigma_ests_adj[-1, :]
            
        return sigma_est_jackknife, sigma_ests_total, sigma_est_jackknife_adj, sigma_ests_total_adj