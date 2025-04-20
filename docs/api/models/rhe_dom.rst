RHE-DOM Model
============

The ``RHE_DOM`` class extends the ``Base`` class to implement Randomized Haseman-Elston regression with dominance effects.
It provides joint estimation of additive and dominant genetic variance components

Variance Components
~~~~~~~~~~~~~~~~~~~
The RHE-DOM model estimates:

- Additive genetic variance components (σ²g) for each bin
- Dominance variance components (σ²d) for each bin
- Total heritability (h²) including both effects
- Enrichment scores

Key Methods
-----------

.. py:method:: get_num_estimates()

   Returns number of variance components:
   
   - Returns ``num_bin * 2`` for both additive and dominance effects

.. py:method:: get_M_last_row()

   Specifies the last row of the M matrix:
   
   - Returns concatenated ``[len_bin, len_bin]``
   - First part for additive effects
   - Second part for dominance effects


.. py:method:: standardize_geno_dom(maf, geno_encoded)

   Standardizes dominance-encoded genotypes:

   :param array maf: Major allele frequencies
   :param array geno_encoded: Dominance-encoded genotypes
   :returns: Standardized dominance genotypes

   .. code-block:: python

      def standardize_geno_dom(maf, geno_encoded):
          means = np.mean(geno_encoded, axis=0)
          stds = 1 / (2 * maf * (1 - maf))
          return (geno_encoded - means) * stds

.. py:method:: _encode_geno(geno)

   Encodes genotypes for dominance effects:

   :param array geno: Original genotype matrix
   :returns: Tuple of (encoded genotypes, MAF)

   .. code-block:: python

      def _encode_geno(geno):
          # Calculate MAF
          maf = np.mean(geno, axis=0) / 2
          
          # Encode genotypes
          encoded = np.zeros_like(geno, dtype=np.float64)
          # This part is vectorized for computational efficiency
          encoded += (geno == 1) * (2 * maf[np.newaxis, :])
          encoded += (geno == 2) * (4 * maf[np.newaxis, :] - 2)
          
          return encoded, maf

.. py:method:: pre_compute_jackknife_bin(j, all_gen)

   Pre-computes statistics for jackknife estimation:

   :param int j: Jackknife sample index
   :param list all_gen: List of genotype matrices for each bin

   .. code-block:: python

      def pre_compute_jackknife_bin(j, all_gen):
          for k, X_kj in enumerate(all_gen):
            # Original genotypes
            X_kj = self.standardize_geno(X_kj) # Standardize
            self.M[j][k] = self.M[self.num_jack][k] - X_kj.shape[1]
            for b in range(self.num_random_vec):
                self.XXz[k, j, b, :] = self._compute_XXz(b, X_kj)
                if self.use_cov:
                    self.UXXz[k, j, b, :] = self._compute_UXXz(self.XXz[k][j][b])
                    self.XXUz[k, j, b, :] = self._compute_XXUz(b, X_kj)
            self.yXXy[k][j] = self._compute_yXXy(X_kj, y=self.pheno)
            
            # Encoded genotypes
            X_kj_original = all_gen[k]
            X_kj_encoded, maf = self._encode_geno(X_kj_original)
            # Standardize the encoded genotypes using maf
            X_kj_encoded = self.standardize_geno_dom(maf, X_kj_encoded)
                
            self.M[j][k + self.num_bin] = self.M[self.num_jack][k + self.num_bin] - X_kj_encoded.shape[1]
            for b in range(self.num_random_vec):
                self.XXz[k + self.num_bin, j, b, :] = self._compute_XXz(b, X_kj_encoded)
                if self.use_cov:
                    self.UXXz[k + self.num_bin, j, b, :] = self._compute_UXXz(self.XXz[k + self.num_bin][j][b])
                    self.XXUz[k + self.num_bin, j, b, :] = self._compute_XXUz(b, X_kj_encoded)
            self.yXXy[k + self.num_bin][j] = self._compute_yXXy(X_kj_encoded, y=self.pheno)

.. py:method:: run(method)

   Runs complete RHE-DOM analysis:

   :param str method: Estimation method ("lstsq" or "QR")
   :returns: Dictionary containing:
      - sigma_ests_total: Estimated variance components
      - sig_errs: Standard errors of variance components
      - h2_total: Heritability estimates
      - h2_errs: Standard errors of heritability
      - enrichment_total: Enrichment scores
      - enrichment_errs: Standard errors of enrichment

Usage Example
------------

.. code-block:: python

   from pyrhe.models import RHE_DOM

   # Initialize model
   rhe_dom_model = RHE_DOM(
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
   results = rhe_dom_model()

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