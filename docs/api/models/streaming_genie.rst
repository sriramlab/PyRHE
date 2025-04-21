Streaming GENIE Model
====================

The ``StreamingGENIE`` class extends both ``GENIE`` and ``StreamingBase`` to provide memory-efficient processing of gene-environment interaction analysis for large-scale genomic data.

Model Types
-----------

Similar to the `GENIE` model, the ``StreamingGENIE`` model supports three streaming configurations:

- ``G``: Basic genetic effects with streaming
- ``G+GxE``: Genetic and gene-environment interactions with streaming
- ``G+GxE+NxE``: Full model with noise-environment interactions in streaming mode

Class Inheritance
---------------

.. code-block:: python

   class StreamingGENIE(GENIE, StreamingBase):
       """Memory-efficient implementation of GENIE."""


Key Methods
---------

.. py:method:: pre_compute_jackknife_bin(j, all_gen, worker_num)

   Pre-computes statistics for each jackknife sample:

   :param int j: Jackknife sample index
   :param list all_gen: List of genotype matrices for each bin
   :param int worker_num: Worker process identifier

   .. code-block:: python

      def pre_compute_jackknife_bin(j, all_gen, worker_num):
          # Process genetic effects
          for k, X_kj in enumerate(all_gen):
              X_kj = standardize_geno(X_kj)
              M[j][k] = M[num_jack][k] - X_kj.shape[1]
              
              # Compute genetic statistics
              for b in range(num_random_vec):
                  XXz[k][worker_num][b] += compute_XXz(b, X_kj)
                  
                  if use_cov:
                      UXXz[k][worker_num][b] += compute_UXXz(XXz[k][worker_num][b])
                      XXUz[k][worker_num][b] += compute_XXUz(b, X_kj)
              
              yXXy[k][worker_num] += compute_yXXy(X_kj, pheno)[0][0]
          
          # Process GxE interactions if enabled
          if genie_model in ["G+GxE", "G+GxE+NxE"]:
              for e in range(num_env):
                  for k, X_kj in enumerate(all_gen):
                      X_kj_gxe = elem_mul(X_kj, env[:, e].reshape(-1, 1))
                      k_gxe = (e + 1) * k + num_bin
                      
                      for b in range(num_random_vec):
                          XXz[k_gxe][worker_num][b] += compute_XXz(b, X_kj_gxe)
                          
                          if use_cov:
                              UXXz[k_gxe][worker_num][b] += compute_UXXz(XXz[k_gxe][worker_num][b])
                              XXUz[k_gxe][worker_num][b] += compute_XXUz(b, X_kj_gxe)
                      
                      yXXy[k_gxe][worker_num] += compute_yXXy(X_kj_gxe, pheno)[0][0]

.. py:method:: pre_compute_jackknife_bin_pass_2(j, all_gen)

   Performs second pass computation for jackknife estimates:

   :param int j: Jackknife sample index
   :param list all_gen: List of genotype matrices for each bin

   .. code-block:: python

      def pre_compute_jackknife_bin_pass_2(j, all_gen):
          # Process genetic effects
          for k, X_kj in enumerate(all_gen):
              X_kj = standardize_geno(X_kj) if j != num_jack else 0
              process_genetic_pass_2(X_kj, k, j)
          
          # Process GxE interactions
          if genie_model in ["G+GxE", "G+GxE+NxE"]:
              for e in range(num_env):
                  for k, X_kj in enumerate(all_gen):
                      X_kj_gxe = compute_gxe_interaction(X_kj, e, j)
                      process_gxe_pass_2(X_kj_gxe, k, e, j)

Other Methods to Override the Base Class
---------------------------------------

Similar to the `GENIE` model which override the base class's ``estimate`` method to return also the adjusted sigma results based on the traces, the ``StreamingGENIE`` model overrides the following methods from the ``BaseStreaming`` class.

Since the ``estimate`` method calls the ``_estimate_worker`` method, we need to override the ``_estimate_worker`` method as well.

.. py:method:: _estimate_worker(self, worker_num, method, start_j, end_j, result_queue, trace_sum):

    This method is overridden because for StreamingGENIE, the heritability should be computed with the traces. 
    Thus, the adjusted sigma estimated based on the traces is also returned.

    .. code-block:: python

      def _estimate_worker(self, worker_num, method, start_j, end_j, result_queue, trace_sum):
          # ... existing code from the streaming base class ...
          sigma_ests_adj = []

          for j in range(self.num_jack):
            # ... existing code from the base class ...
            # Adjust the estimate by the effect of traces for heritability calculation
            sigma_est_adj = []
            for i in range(len(sigma_est)):
                sigma_est_adj.append(sigma_est[i] * T[i, self.num_estimates])

          for i in range(len(sigma_est)):
              sigma_est_adj.append(sigma_est[i] * T[i, self.num_estimates])
          sigma_ests_adj.append(sigma_est_adj)
          # Put the sigma_ests_adj also in the result_queue
          if self.multiprocessing:
              result_queue.put((worker_num, sigma_ests, sigma_ests_adj)) # ensure in order
          else:
              result_queue.extend((sigma_ests, sigma_ests_adj))
    
      def estimate(self, method: str = "lstsq"):
        # ... existing code from the streaming base class ...

        # ... Unpack the results by also including the all_results_adj
            all_results = [item for _, result, _ in results for item in result]
            all_results_adj = [item for _, _, result in results for item in result

        # ... existing code from the streaming base class ...
        # Also aggregate the results from the all_results_adj
        sigma_ests_adj = np.array(all_results_adj)
        sigma_est_jackknife_adj, sigma_ests_total_adj = sigma_ests_adj[:-1, :], sigma_ests_adj[-1, :]
            
        return sigma_est_jackknife, sigma_ests_total, sigma_est_jackknife_adj, sigma_ests_total_adj


UUsage Example
------------

.. code-block:: python

   from pyrhe.models import StreamingGENIE

   # Initialize model
   streaming_genie_model = StreamingGENIE(
        genie_model="G+GxE+NxE",
       geno_file="path/to/genotype",
       annot_file="path/to/annotation",
       pheno_file="path/to/phenotype",
       cov_file="path/to/covariate",
       num_bins=10,
       num_jack=100,
       num_random_vec=10,
       num_workers=5,
       ...
   )

   # Run analysis
   results = streaming_genie_model()

   # Access results
   # The outputs are automatically logged in the output file. 
   # In addition, you can also access the results:
   print(results)
   print(results['sigma_ests_total'])
   # The results are stored in a dictionary. The keys are:
   # - sigma_ests_total: Estimated variance components
   # - sig_errs: Standard errors of variance components
   # - h2_total: Heritability estimates
   # - h2_errs: Standard errors of heritability
   # - enrichment_total: Enrichment scores
   # - enrichment_errs: Standard errors of enrichment

   