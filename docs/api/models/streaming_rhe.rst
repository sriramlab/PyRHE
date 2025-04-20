Streaming RHE Model
==================

The ``StreamingRHE`` class extends both ``RHE`` and ``StreamingBase`` to provide memory-efficient processing of large-scale genotype data. It implements a streaming version of the Randomized Haseman-Elston regression.

Class Inheritance
---------------

.. code-block:: python

   class StreamingRHE(RHE, StreamingBase):
       """Memory-efficient implementation of RHE."""


.. py:method:: pre_compute_jackknife_bin(j, all_gen, worker_num)

   Pre-computes statistics for each jackknife sample:

   :param int j: Jackknife sample index
   :param list all_gen: List of genotype matrices for each bin
   :param int worker_num: Worker process identifier

   .. code-block:: python

      def pre_compute_jackknife_bin(j, all_gen, worker_num):
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

.. py:method:: pre_compute_jackknife_bin_pass_2(j, all_gen)

   Performs second pass computation for jackknife estimates:

   :param int j: Jackknife sample index
   :param list all_gen: List of genotype matrices for each bin

   .. code-block:: python

      def pre_compute_jackknife_bin_pass_2(j, all_gen):
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

Usage Example
------------

.. code-block:: python

   from pyrhe.models import StreamingRHE

   # Initialize model
   streaming_rhe_model = StreamingRHE(
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
   results = streaming_rhe_model()

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
   # - h2_jackknife_overlap: Jackknife heritability estimates computed based on overlapping setting
   # - h2_errs_overlap: Standard errors of jackknife heritability computed based on overlapping setting
   # - h2_total_overlap: Overlapping heritability estimates computed based on overlapping setting
   # - h2_errs_total_overlap: Standard errors of overlapping heritability computed based on overlapping setting