RHE Model
=========

The ``RHE`` class implements the Randomized Haseman-Elston regression model for heritability estimation. It inherits from the ``Base`` class and provides efficient estimation of additive genetic variance components.

Variance Components
~~~~~~~~~~~~~~~~~~~
The RHE model estimates:

- Additive genetic variance components (σ²g) for each bin
- Environmental variance component (σ²e)
- Total heritability (h²)
- Enrichment scores

Class Inheritance
---------------

.. code-block:: python

   class RHE(Base):
       """Randomized Haseman-Elston regression model."""

Key Methods
-----------

.. py:method:: get_num_estimates()

   Returns the number of variance components to estimate:
   
   - Returns ``num_bin`` for additive genetic effects

.. py:method:: get_M_last_row()

   Specifies the last row of the M matrix:
   
   - Returns ``self.len_bin`` for additive effects
   - ``self.len_bin`` is provided by the base class, which is the number of SNPs in each bin.

.. py:method:: pre_compute_jackknife_bin(j, all_gen)

   Pre-computes statistics for jackknife estimation:

   :param int j: Jackknife sample index
   :param list all_gen: List of genotype matrices for each bin

   .. code-block:: python

      def pre_compute_jackknife_bin(j, all_gen):
          for k, X_kj in enumerate(all_gen): # Loop through the partitioned genotype matrix
               # 1. Process genotype data
               X_kj = self.standardize_geno(X_kj)
               
               # 2. Update M matrix
               self.M[j][k] = self.M[self.num_jack][k] - X_kj.shape[1]
               
               # 3. Compute statistics
               for b in range(self.num_random_vec):
                   self.XXz[k, j, b, :] = self._compute_XXz(b, X_kj)
                   if self.use_cov:
                       self.UXXz[k, j, b, :] = self._compute_UXXz(self.XXz[k][j][b])
                       self.XXUz[k, j, b, :] = self._compute_XXUz(b, X_kj)
               
               # 4. Compute phenotype-related statistics
               self.yXXy[k][j] = self._compute_yXXy(X_kj, y=self.pheno)

.. py:method:: b_trace_calculation(k, j, b_idx)

   Calculates trace terms for variance estimation:

   - Returns ``num_indv`` for standardized genotypes

.. py:method:: run(method)

   Runs the complete RHE analysis:

   :param str method: Estimation method ("lstsq" or "QR")
   :returns: Dictionary containing:
      - sigma_ests_total: Estimated variance components
      - sig_errs: Standard errors of variance components
      - h2_total: Heritability estimates
      - h2_errs: Standard errors of heritability
      - enrichment_total: Enrichment scores
      - enrichment_errs: Standard errors of enrichment
      - h2_jackknife_overlap: Jackknife heritability estimates computed based on overlapping setting
      - h2_errs_overlap: Standard errors of jackknife heritability computed based on overlapping setting
      - h2_total_overlap: Overlapping heritability estimates computed based on overlapping setting
      - h2_errs_total_overlap: Standard errors of overlapping heritability computed based on overlapping setting


Usage Example
------------

.. code-block:: python

   from pyrhe.models import RHE

   # Initialize model
   rhe_model = RHE(
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
   results = rhe_model()

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
