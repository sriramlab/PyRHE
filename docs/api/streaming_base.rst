Streaming Base Class
===================

The ``StreamingBase`` class extends the ``Base`` class to provide memory-efficient processing of large-scale genomic data. 

In the Base class, we store the pre-computed statistics for each jackknife subsample in the tensors ``XXz`` and ``yXXy``. 
However, the size of these tensors scale with the number of jackknife subsamples, which can be very large. 
Thus, in the ``StreamingBase`` class, we use a two-pass approach. Specifically:

1. First Pass (Aggregate the statistics across all jackknife subsamples)
   - Processes each block to compute initial statistics
   - Accumulates results across workers
   - Stores sums in shared memory

2. Second Pass (Compute the leave-one-out statistics)
   - Recomputes statistics for each jackknife sample
   - Subtracts block contributions for leave-one-out estimates
   - Calculates final estimates and standard errors

Shared Memory Arrays
--------------------

Since the shared memory arrays are of different sizes than the base class, we need to redefine their sizes.

- ``self.M``: Size: (num_jack + 1, num_estimates).

    It contains the number of SNPs in each jackknife subsample in each bin. The last row is the total number of SNPs in each bin (Same as the base class).

To support multiple workers without race condition, we first store each worker's results in a separate array and then aggregate the results.

The initial size of the shared memory arrays are:

- ``self.XXz``: Size: (num_estimates, num_workers, num_random_vec, num_indv).

    The second dimension is the number of workers instead of num_jack + 1

    It contains the ``XXz`` value for the jackknife subsamples within each worker.

- ``self.yXXy``: Size: (num_estimates, num_workers).

    It contains the ``yXXy`` value for the jackknife subsamples within each worker.

If the covariate file is provided, the UXXz and XXUz will be calculated:

- ``self.UXXz``: Size: (num_estimates, num_workers, num_random_vec, num_indv).

    It contains the ``UXXz`` value for the jackknife subsamples within each worker.

- ``self.XXUz``: Size: (num_estimates, num_workers, num_random_vec, num_indv).

    It contains the ``XXUz`` value for the jackknife subsamples within each worker.

Then, we aggregate the results across the workers and first temporarily store the aggregated results in the tensors ``XXz_per_jack``, ``yXXy_per_jack``, ``UXXz_per_jack``, and ``XXUz_per_jack``.

- ``self.XXz_per_jack``: Size: (num_estimates, 2, num_random_vec, num_indv).

- ``self.yXXy_per_jack``: Size: (num_estimates, 2).

- ``self.UXXz_per_jack``: Size: (num_estimates, 2, num_random_vec, num_indv).

- ``self.XXUz_per_jack``: Size: (num_estimates, 2, num_random_vec, num_indv).

Note that the second dimension is 2 because it only stores the total statistics and one leave-one-out statistics at a time. 

Then, we reassign those tensors to the original shared memory arrays ``self.XXz``, ``self.yXXy``, ``self.UXXz``, and ``self.XXUz``, and delete the temporary tensors.

Thus, the final size of the shared memory arrays are:

- ``self.XXz``: Size: (num_estimates, 2, num_random_vec, num_indv).

- ``self.yXXy``: Size: (num_estimates, 2).

- ``self.UXXz``: Size: (num_estimates, 2, num_random_vec, num_indv).

- ``self.XXUz``: Size: (num_estimates, 2, num_random_vec, num_indv).

Key Methods
---------

.. py:method:: aggregate()

   Aggregates results across workers and store the sum in the tensors ``XXz`` and ``yXXy``.

.. py:method:: _pre_compute_worker(worker_num, start_j, end_j, total_sample_queue)

   Worker process for the first pass:

   - Processes assigned jackknife samples
   - Computes initial statistics through the abstract ``pre_compute_jackknife_bin`` method
   - Updates shared memory arrays


.. py:method:: estimate(self, method: str = "lstsq")

The ``estimate`` method in the streaming base class implements memory-efficient estimation of variance components using either multiprocessing or single-process mode. Here's a detailed breakdown:

.. code-block:: python

   def estimate(self, method: str = "lstsq") -> Tuple[List[List], List]:
       """Estimate variance components using streaming approach.
       
       The estimation process involves:
       1. Distributing work across processes (if multiprocessing is enabled)
       2. Computing estimates for each jackknife sample
       3. Aggregating results across samples
       
       Args:
           method: Estimation method to use ("lstsq" or "qr")
       Returns:
           Tuple containing:
           - List of jackknife estimates
           - List of total estimates
       """
       
       # 1. Handle multiprocessing vs single-process mode
       if self.multiprocessing:
           # 1.1 Distribute work across available workers
           work_ranges = self._distribute_work(self.num_jack + 1, self.num_workers)
           
           # 1.2 Set up shared memory for trace calculations if needed
           manager = multiprocessing.Manager()
           trace_sums = manager.dict() if self.get_trace else None
           
           # 1.3 Initialize multiprocessing handler
           mp_handler = MultiprocessingHandler(
               target=self._estimate_worker,
               work_ranges=work_ranges,
               device=self.device,
               trace_sums=trace_sums,
               method=method,
               streaming_estimate=True  # Enable streaming mode
           )
           
           # 1.4 Start and manage worker processes
           mp_handler.start_processes()
           mp_handler.join_processes()
           
           # 1.5 Collect and sort results
           results = mp_handler.get_queue()
           results.sort(key=lambda x: x[0])  # Sort by work range index
           all_results = [item for _, result in results for item in result]
           
       else:
           # 2. Single-process mode
           self.result_queue = []
           trace_sums = np.zeros((self.num_jack+1, self.num_bin, self.num_bin)) if self.get_trace else None
           
           # 2.1 Process each jackknife sample sequentially
           for j in tqdm(range(self.num_jack + 1), desc="Estimating..."):
               self._estimate_worker(
                   0,  # worker_id
                   method,
                   j,  # start index
                   j + 1,  # end index
                   self.result_queue,
                   trace_sums
               )
           
           all_results = self.result_queue
           del self.result_queue  # Clean up
       
       # 3. Process trace calculations if enabled
       if self.get_trace:
           self.get_trace_summary(trace_sums)
       
       # 4. Aggregate and return results
       sigma_ests = np.array(all_results)
       # Separate jackknife estimates from total estimates
       sigma_est_jackknife, sigma_ests_total = sigma_ests[:-1, :], sigma_ests[-1, :]
       
       return sigma_est_jackknife, sigma_ests_total

Abstract Methods
-------------

.. py:method:: pre_compute_jackknife_bin_pass_2(j, k, X_kj)

   Abstract method for the second pass:

   - Computes leave-one-out statistics
   - Updates final estimates


Example for Extending the Streaming Base Class to New Streaming Models
------------------------------------------------------------------------

To create a new streaming model, you need to extend both your base model class and the ``StreamingBase`` class. Here's an example based on the ``StreamingRHE`` implementation.
The new streaming model will inherit the methods like ``get_num_estimates()`` and ``get_M_last_row()`` from the ``NewModel`` class. 
However, you should still define the ``pre_compute_jackknife_bin()`` and ``pre_compute_jackknife_bin_pass_2()`` methods.

.. code-block:: python

   from pyrhe.src.new_model import NewModel
   from pyrhe.src.streaming_base import StreamingBase

   class NewStreamingModel(NewModel, StreamingBase):
       
       def pre_compute_jackknife_bin(self, j, all_gen, worker_num):
           """Implement the first pass computation for jackknife estimates.
           
           This method should:
           1. Process each genotype block
           2. Compute necessary statistics
           3. Store results in shared memory arrays
           
           Args:
               j: Jackknife sample index
               all_gen: List of genotype matrices for each bin
               worker_num: Worker process identifier
           """
           # Example: 
           for k, X_kj in enumerate(all_gen): 
           # 1. Process genotype data
            X_kj = self.standardize_geno(X_kj)

            # 2. Update M matrix
            self.M[j][k] = self.M[self.num_jack][k] - X_kj.shape[1]

            # Compute statistics
            for b in range(self.num_random_vec):
                self.XXz[k][worker_num][b] += self._compute_XXz(b, X_kj) # The statistics are store in self.XXz[k][worker_num][b] instead of self.XXz[k][j][b]

                if self.use_cov:
                    self.UXXz[k][worker_num][b] += self._compute_UXXz(self.XXz[k][worker_num][b])
                    self.XXUz[k][worker_num][b] += self._compute_XXUz(b, X_kj)
                    
            yXXy_kj = self._compute_yXXy(X_kj, y=self.pheno)
            self.yXXy[k][worker_num] += yXXy_kj[0][0]

            # The streaming base class will handle the aggregation of the statistics across workers.
         
       def pre_compute_jackknife_bin_pass_2(self, j, all_gen):
           """Implement second pass computation for jackknife estimates.
           
           This method should:
           1. Process each genotype block for the second pass
           2. Compute leave-one-out statistics
           3. Update shared memory arrays
           
           Args:
               j: Jackknife sample index
               all_gen: List of genotype matrices for each bin
           """
           # Example:
           for k in range(self.num_estimates):
               # Recompute the statistics: 
               X_kj = self.standardize_geno(all_gen[k]) if j != self.num_jack else 0
               for b in range (self.num_random_vec):
                  XXz_kb = self._compute_XXz(b, X_kj) if j != self.num_jack else 0
                  if self.use_cov:
                     UXXz_kb = self._compute_UXXz(XXz_kb) if j != self.num_jack else 0
                     self.UXXz[k][1][b] = self.UXXz[k][0][b] - UXXz_kb # Calculate the leave-one-out statistics
                     XXUz_kb = self._compute_XXUz(b, X_kj) if j != self.num_jack else 0
                     self.XXUz[k][1][b] = self.XXUz[k][0][b] - XXUz_kb # Calculate the leave-one-out statistics
                  self.XXz[k][1][b] = self.XXz[k][0][b] - XXz_kb # Calculate the leave-one-out statistics
            
               yXXy_k = (self._compute_yXXy(X_kj, y=self.pheno))[0][0] if j != self.num_jack else 0
               self.yXXy[k][1] = self.yXXy[k][0] - yXXy_k # Calculate the leave-one-out statistics