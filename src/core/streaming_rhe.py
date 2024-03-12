import time
import torch
import queue
from tqdm import tqdm
from src.core.rhe import RHE
from typing import List, Tuple
import numpy as np
import multiprocessing
from multiprocessing import shared_memory, Queue
from src.core.mp_handler import MultiprocessingHandler



class StreamingRHE(RHE):
    def __init__(
        self,
        geno_file: str,
        annot_file: str = None,
        pheno_file: str = None,
        cov_file: str = None,
        num_bin: int = 8,
        num_jack: int = 1,
        num_random_vec: int = 10,
        device: torch.device = torch.device("cpu"),
        cuda_num: int = None,
        num_workers: int = None,
        multiprocessing: bool = True,
        verbose: bool = True,
        seed: int = None,
        get_trace: bool = False,
        trace_dir: str = None,
    ):
        super().__init__(
            geno_file=geno_file,
            annot_file=annot_file,
            pheno_file=pheno_file,
            cov_file=cov_file,
            num_bin=num_bin,
            num_jack=num_jack,
            num_random_vec=num_random_vec,
            device=device,
            cuda_num=cuda_num,
            num_workers=num_workers,
            multiprocessing=multiprocessing,
            verbose=verbose,
            seed=seed,
            get_trace=get_trace,
            trace_dir=trace_dir
        )
    
    def _aggregate(self):
        self.XXz_per_jack = np.zeros((self.num_bin, 2, self.num_random_vec, self.num_indv), dtype=np.float64)
        self.yXXy_per_jack = np.zeros((self.num_bin, 2), dtype=np.float64)
        self.UXXz_per_jack = np.zeros((self.num_bin, 2, self.num_random_vec, self.num_indv), dtype=np.float64) if self.use_cov else None
        self.XXUz_per_jack = np.zeros((self.num_bin, 2, self.num_random_vec, self.num_indv), dtype=np.float64) if self.use_cov else None
        for k in range(self.num_bin):
            self.yXXy_per_jack[k][0] = np.sum(self.yXXy[k, :])
            for b in range(self.num_random_vec):
                self.XXz_per_jack[k, 0, b, :] = np.sum(self.XXz[k, :, b, :], axis=0)
                if self.UXXz_per_jack is not None: 
                    self.UXXz_per_jack[k, 0, b, :] = np.sum(self.UXXz[k, :, b, :], axis=0) 
                if self.XXUz_per_jack is not None:
                    self.XXUz_per_jack[k, 0, b, :] = np.sum(self.XXUz[k, 0, b, :], axis=0)

    def _pre_compute_worker(self, worker_num, start_j, end_j, total_sample_queue=None):
        if self.multiprocessing:
            self._init_device(self.device_name, self.cuda_num)
        if self.multiprocessing:
            # set up shared memory in child
            XXz_shm = shared_memory.SharedMemory(name=self.XXz_shm.name)
            XXz = np.ndarray((self.num_bin, self.num_workers, self.num_random_vec, self.num_indv), dtype=np.float64, buffer=XXz_shm.buf)
            XXz.fill(0)

            yXXy_shm = shared_memory.SharedMemory(name=self.yXXy_shm.name)
            yXXy = np.ndarray((self.num_bin, self.num_workers), dtype=np.float64, buffer=yXXy_shm.buf)
            yXXy.fill(0)

            if self.use_cov:
                UXXz_shm = shared_memory.SharedMemory(name=self.UXXz_shm.name)
                UXXz = np.ndarray((self.num_bin, self.num_workers, self.num_random_vec, self.num_indv), dtype=np.float64, buffer=UXXz_shm.buf)
                UXXz.fill(0)

                XXUz_shm = shared_memory.SharedMemory(name=self.XXUz_shm.name)
                XXUz = np.ndarray((self.num_bin, self.num_workers, self.num_random_vec, self.num_indv), dtype=np.float64, buffer=XXUz_shm.buf)
                XXUz.fill(0)
            
            M_shm = shared_memory.SharedMemory(name=self.M_shm.name)
            M =  np.ndarray((self.num_jack + 1, self.num_bin), buffer=M_shm.buf)
            M.fill(0)
            M[self.num_jack] = self.len_bin
        else:
            XXz_per_jack = self.XXz_per_jack
            yXXy_per_jack = self.yXXy_per_jack
            UXXz_per_jack = self.UXXz_per_jack
            XXUz_per_jack = self.XXUz_per_jack
            M = self.M
        
        for j in range(start_j, end_j):
            if self.multiprocessing:
                worker_num = worker_num
            else:
                worker_num = 0
            print(f"Worker {multiprocessing.current_process().name} processing jackknife sample {j}")
            start_whole = time.time()
            subsample, sub_annot = self._get_jacknife_subsample(j)
            subsample = self.impute_geno(subsample, simulate_geno=True)
            if total_sample_queue is not None:
                total_sample_queue.put((worker_num, subsample.shape[0]))
            else:
                self.total_num_sample = subsample.shape[0]
            all_gen = self.partition_bins(subsample, sub_annot)

            for k, X_kj in enumerate(all_gen):
                M[j][k] = M[self.num_jack][k] - X_kj.shape[1] # store the dimension with the corresponding block
                for b in range(self.num_random_vec):
                    XXz_kjb = self._compute_XXz(b, X_kj)
                    if self.multiprocessing:
                        XXz[k][worker_num][b] += XXz_kjb
                    else:
                        self.XXz_per_jack[k][worker_num][b] += XXz_kjb

                    if self.use_cov:
                        UXXz_kjb = self._compute_UXXz(XXz_kjb)
                        XXUz_kjb = self._compute_XXUz(b, X_kj)
                        if self.multiprocessing:
                            UXXz[k][worker_num][b] += UXXz_kjb
                            XXUz[k][worker_num][b] += XXUz_kjb
                        else:
                            self.UXXz_per_jack[k][worker_num][b] += UXXz_kjb
                            self.XXUz_per_jack[k][worker_num][b] += XXUz_kjb
                
            
                yXXy_kj = self._compute_yXXy(X_kj, y=self.pheno)
                if self.multiprocessing:
                    yXXy[k][worker_num] += yXXy_kj[0][0]
                else:
                    self.yXXy_per_jack[k][worker_num] += yXXy_kj[0][0]

                del X_kj

            end_whole = time.time()
            print(f"jackknife {j} precompute (pass 1) total time: {end_whole-start_whole}")


    def _estimate_worker(self, worker_num, method, start_j, end_j, result_queue, trace_dict):
        sigma_ests = []
        for j in range(start_j, end_j):
            print(f"Precompute (pass 2) for jackknife sample {j}")
            start_whole = time.time()
            if j != self.num_jack:
                subsample, sub_annot = self._get_jacknife_subsample(j)
                subsample = self.impute_geno(subsample, simulate_geno=True)
                all_gen = self.partition_bins(subsample, sub_annot)

            # calculate the stats for one jacknife subsample
            for k in range(self.num_bin):
                X_k = all_gen[k] if j != self.num_jack else 0
                for b in range (self.num_random_vec):
                    XXz_kb = self._compute_XXz(b, X_k) if j != self.num_jack else 0
                    if self.use_cov:
                        UXXz_kb = self._compute_UXXz(XXz_kb) if j != self.num_jack else 0
                        self.UXXz_per_jack[k][1][b] = self.UXXz_per_jack[k][0][b] - UXXz_kb
                        XXUz_kb = self._compute_XXUz(b, X_k) if j != self.num_jack else 0
                        self.XXUz_per_jack[k][1][b] = self.XXUz_per_jack[k][0][b] - XXUz_kb
                    self.XXz_per_jack[k][1][b] = self.XXz_per_jack[k][0][b] - XXz_kb
                
                yXXy_k = (self._compute_yXXy(X_k, y=self.pheno))[0][0] if j != self.num_jack else 0
                self.yXXy_per_jack[k][1] = self.yXXy_per_jack[k][0] - yXXy_k


            print(f"Estimate for jackknife sample {j}")
            T = np.zeros((self.num_bin+1, self.num_bin+1))
            q = np.zeros((self.num_bin+1, 1))

            for k_k in range(self.num_bin):
                for k_l in range(self.num_bin): 
                    M_k = self.M[j][k_k]
                    M_l = self.M[j][k_l]
                    B1 = self.XXz_per_jack[k_k][1]
                    B2 = self.XXz_per_jack[k_l][1]
                    T[k_k, k_l] += np.sum(B1 * B2)

                    if self.use_cov:
                        h1 = self.cov_matrix.T @ B1.T
                        h2 = self.Q @ h1
                        h3 = self.cov_matrix @ h2
                        trkij_res1 = np.sum(h3.T * B2)

                        B1 = self.XXUz_per_jack[k_k][1]
                        B2 = self.UXXz_per_jack[k_l][1]
                        trkij_res2 = np.sum(B1 * B2)
                    
                        T[k_k, k_l] += (trkij_res2 - 2 * trkij_res1)


                    T[k_k, k_l] /= (self.num_random_vec)
                    T[k_k, k_l] =  T[k_k, k_l] / (M_k * M_l) if (M_k * M_l) != 0 else 0


            for k in range(self.num_bin):
                M_k = self.M[j][k]

                if not self.use_cov:
                    T[k, self.num_bin] = self.num_indv
                    T[self.num_bin, k] = self.num_indv

                else:
                    B1 = self.XXz_per_jack[k][1]
                    C1 = self.all_Uzb
                    tk_res = np.sum(B1 * C1.T)
                    tk_res = 1/(self.num_random_vec * M_k) * tk_res

                    T[k, self.num_bin] = self.num_indv - tk_res
                    T[self.num_bin, k] = self.num_indv - tk_res

                q[k] = self.yXXy_per_jack[k][1] / M_k if M_k != 0 else 0
    
            T[self.num_bin, self.num_bin] = self.num_indv if not self.use_cov else self.num_indv - self.cov_matrix.shape[1]

            if trace_dict is not None:
                trace_dict[j] = ((T[0][0], self.M[j]))
            
            pheno = self.pheno if not self.use_cov else self.regress_pheno(self.cov_matrix, self.pheno)
            q[self.num_bin] = pheno.T @ pheno 
        
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
            print(f"estimate time for jackknife subsample: {end_whole - start_whole}")

        if self.multiprocessing:
            result_queue.put((worker_num, sigma_ests)) # ensure in order
        else:
            result_queue.extend(sigma_ests)


    def estimate(self, method: str = "lstsq") -> Tuple[List[List], List]:
        if self.multiprocessing:
            work_ranges = self._distribute_work(self.num_jack + 1, self.num_workers)
            manager = multiprocessing.Manager()
            trace_dict = manager.dict() if self.get_trace else None

            mp_handler = MultiprocessingHandler(target=self._estimate_worker, work_ranges=work_ranges, device=self.device, trace_dict=trace_dict, method=method)
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
                

'''
    def estimate(self, method: str = "lstsq") -> Tuple[List[List], List]:
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
            print(f"Precompute (pass 2) for jackknife sample {j}")
            start_whole = time.time()
            if j != self.num_jack:
                subsample, sub_annot = self._get_jacknife_subsample(j)
                subsample = self.impute_geno(subsample, simulate_geno=True)
                all_gen = self.partition_bins(subsample, sub_annot)

            # calculate the stats for one jacknife subsample
            for k in range(self.num_bin):
                X_k = all_gen[k] if j != self.num_jack else 0
                for b in range (self.num_random_vec):
                    XXz_kb = self._compute_XXz(b, X_k) if j != self.num_jack else 0
                    if self.use_cov:
                        UXXz_kb = self._compute_UXXz(XXz_kb) if j != self.num_jack else 0
                        self.UXXz_per_jack[k][1][b] = self.UXXz_per_jack[k][0][b] - UXXz_kb
                        XXUz_kb = self._compute_XXUz(b, X_k) if j != self.num_jack else 0
                        self.XXUz_per_jack[k][1][b] = self.XXUz_per_jack[k][0][b] - XXUz_kb
                    self.XXz_per_jack[k][1][b] = self.XXz_per_jack[k][0][b] - XXz_kb
                
                yXXy_k = (self._compute_yXXy(X_k, y=self.pheno))[0][0] if j != self.num_jack else 0
                self.yXXy_per_jack[k][1] = self.yXXy_per_jack[k][0] - yXXy_k


            print(f"Estimate for jackknife sample {j}")
            T = np.zeros((self.num_bin+1, self.num_bin+1))
            q = np.zeros((self.num_bin+1, 1))

            for k_k in range(self.num_bin):
                for k_l in range(self.num_bin):
                    M_k = self.M[j][k_k]
                    M_l = self.M[j][k_l]
                    B1 = self.XXz_per_jack[k_k][1]
                    B2 = self.XXz_per_jack[k_l][1]
                    T[k_k, k_l] += np.sum(B1 * B2)

                    if self.use_cov:
                        h1 = self.cov_matrix.T @ B1.T
                        h2 = self.Q @ h1
                        h3 = self.cov_matrix @ h2
                        trkij_res1 = np.sum(h3.T * B2)

                        B1 = self.XXUz_per_jack[k_k][1]
                        B2 = self.UXXz_per_jack[k_l][1]
                        trkij_res2 = np.sum(B1 * B2)
                    
                        T[k_k, k_l] += (trkij_res2 - 2 * trkij_res1)


                    T[k_k, k_l] /= (self.num_random_vec)
                    T[k_k, k_l] =  T[k_k, k_l] / (M_k * M_l) if (M_k * M_l) != 0 else 0


            for k in range(self.num_bin):
                M_k = self.M[j][k]

                if not self.use_cov:
                    T[k, self.num_bin] = self.num_indv
                    T[self.num_bin, k] = self.num_indv

                else:
                    B1 = self.XXz_per_jack[k][1]
                    C1 = self.all_Uzb
                    tk_res = np.sum(B1 * C1.T)
                    tk_res = 1/(self.num_random_vec * M_k) * tk_res

                    T[k, self.num_bin] = self.num_indv - tk_res
                    T[self.num_bin, k] = self.num_indv - tk_res

                q[k] = self.yXXy_per_jack[k][1] / M_k if M_k != 0 else 0
    
            
            
            T[self.num_bin, self.num_bin] = self.num_indv if not self.use_cov else self.num_indv - self.cov_matrix.shape[1]
            pheno = self.pheno if not self.use_cov else self.regress_pheno(self.cov_matrix, self.pheno)
            q[self.num_bin] = pheno.T @ pheno 
        

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
            print(f"estimate time for jackknife subsample: {end_whole - start_whole}")
            
        sigma_ests = np.array(sigma_ests)

        sigma_est_jackknife, sigma_ests_total = sigma_ests[:-1, :], sigma_ests[-1, :]
            
        return sigma_est_jackknife, sigma_ests_total
'''

