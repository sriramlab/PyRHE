GENIE Model
===========

The ``GENIE`` class implements the Gene-ENvironment Interaction Estimator. It extends the ``Base`` class to estimate gene-environment interaction effects in genomic data.

Core Components
--------------

Model Types
~~~~~~~~~~~

The GENIE model supports three types of analysis:

- ``G``: Basic genetic effects only
- ``G+GxE``: Genetic effects plus gene-environment interactions
- ``G+GxE+NxE``: Full model with noise-environment interactions

Variance Components
~~~~~~~~~~~~~~~~~~~

The model estimates multiple variance components:

- σ²g: Genetic variance
- σ²gxe: Gene-environment interaction variance
- σ²nxe: Noise-environment interaction variance
- σ²e: Environmental variance

Class Inheritance
---------------

.. code-block:: python

   class GENIE(Base):
       """Gene-ENvironment Interaction Estimator."""

Initialization
--------------

To initialize the GENIE model, you should also specify the environment file and the genie model to use.
You must set the  ``self.num_env`` and ``self.env`` member variables, which should be read from the environment file.
You must set the ``self.num_gen_env_bin`` as ``self.num_bin * self.num_env``.


.. code-block:: python
def __init__(
        self,
        env_file: str,
        genie_model: str,
        **kwargs
    ):
         
        super().__init__(**kwargs) 

        self.num_env, self.env = read_env_file(env_file)
        self.env = self.env[:, np.newaxis]
        self.num_gen_env_bin = self.num_bin * self.num_env
        self.genie_model = genie_model

Key Methods
----------

.. py:method:: get_num_estimates()

   Returns the number of variance components based on model type:

   .. code-block:: python

      def get_num_estimates(self):
          if self.genie_model == "G":
            return self.num_bin
        elif self.genie_model == "G+GxE":
            return self.num_bin + self.num_gen_env_bin
        elif self.genie_model == "G+GxE+NxE":
            return self.num_bin + self.num_gen_env_bin + self.num_env
        else:
            raise ValueError("Unsupported GENIE genie_model type")

.. py:method:: get_M_last_row()

   Specifies the last row of the M matrix based on the model type:

   .. code-block:: python

      def get_M_last_row(self):
        if self.genie_model == "G":
            return self.len_bin
        elif self.genie_model == "G+GxE":
            return np.concatenate((self.len_bin, self.len_bin * self.num_env))
        elif self.genie_model == "G+GxE+NxE":
            return np.concatenate((self.len_bin, self.len_bin * self.num_env, [1] * self.num_env))
        else:
            raise ValueError("Unsupported GENIE genie_model type")

.. py:method:: pre_compute_jackknife_bin(j, all_gen)

   Pre-computes statistics for each jackknife sample:

   :param int j: Jackknife sample index
   :param list all_gen: List of genotype matrices for each bin

   .. code-block:: python

      def pre_compute_jackknife_bin(self, j, all_gen):
          for k, X_kj in enumerate(all_gen):
              # Process genetic effects
              X_kj = self.standardize_geno(X_kj)
              self.M[j][k] = self.M[self.num_jack][k] - X_kj.shape[1]
              
              for b in range(self.num_random_vec):
                  self.XXz[k, j, b, :] = self._compute_XXz(b, X_kj)
                  if self.use_cov:
                      self.UXXz[k, j, b, :] = self._compute_UXXz(self.XXz[k][j][b])
                      self.XXUz[k, j, b, :] = self._compute_XXUz(b, X_kj)
              
              self.yXXy[k][j] = self._compute_yXXy(X_kj, self.pheno)
              
              # Process GxE effects if needed
              if self.model_type in ["G+GxE", "G+GxE+NxE"]:
                  X_kj_gxe = self._compute_gxe_effects(X_kj)
                  self.M[j][k + self.num_bin] = self.M[self.num_jack][k + self.num_bin] - X_kj_gxe.shape[1]
                  
                  for b in range(self.num_random_vec):
                      self.XXz[k + self.num_bin, j, b, :] = self._compute_XXz(b, X_kj_gxe)
                      if self.use_cov:
                          self.UXXz[k + self.num_bin, j, b, :] = self._compute_UXXz(self.XXz[k + self.num_bin][j][b])
                          self.XXUz[k + self.num_bin, j, b, :] = self._compute_XXUz(b, X_kj_gxe)
                  
                  self.yXXy[k + self.num_bin][j] = self._compute_yXXy(X_kj_gxe, self.pheno)
              
              # Process NxE effects if needed
              if self.model_type == "G+GxE+NxE":
                  X_kj_nxe = self._compute_nxe_effects(X_kj)
                  self.M[j][k + 2 * self.num_bin] = self.M[self.num_jack][k + 2 * self.num_bin] - X_kj_nxe.shape[1]
                  
                  for b in range(self.num_random_vec):
                      self.XXz[k + 2 * self.num_bin, j, b, :] = self._compute_XXz(b, X_kj_nxe)
                      if self.use_cov:
                          self.UXXz[k + 2 * self.num_bin, j, b, :] = self._compute_UXXz(self.XXz[k + 2 * self.num_bin][j][b])
                          self.XXUz[k + 2 * self.num_bin, j, b, :] = self._compute_XXUz(b, X_kj_nxe)
                  
                  self.yXXy[k + 2 * self.num_bin][j] = self._compute_yXXy(X_kj_nxe, self.pheno)

.. py:method:: b_trace_calculation(k, j, b_idx)

   Calculates trace terms for estimation:

   :param int k: Bin index
   :param int j: Jackknife sample index
   :param int b_idx: Random vector index
   :return: Trace value

   .. code-block:: python

      def b_trace_calculation(self, k, j, b_idx):
        # Trace for the interaction terms
        if k >= self.num_bin:
            # Actual trace calculation
            M_k = self.M[j][k]
            B1 = self.XXz[k][b_idx]
            b_trk = np.sum(B1 * self.all_zb.T) / (self.num_random_vec * M_k)
        else:
            # Trace can be directly calculated as self.num_indv since the genotype is standardized
            b_trk = self.num_indv

        return b_trk

.. py:method:: run(method)

   Runs the complete GENIE analysis:

   :param str method: Estimation method ("lstsq" or "QR")
   :returns: Dictionary containing:
      - sigma_ests_total: Estimated variance components
      - sig_errs: Standard errors of variance components
      - h2_total: Heritability estimates
      - h2_errs: Standard errors of heritability
      - enrichment_total: Enrichment scores
      - enrichment_errs: Standard errors of enrichment

Other Methods to Override the Base Class
---------------------------------------

In addition to the methods above, the ``GENIE`` class also overrides the following methods from the ``Base`` class:

.. py:method:: estimate(self, method)

    This method is overridden because for GENIE, the heritability should be computed with the traces. 
    Thus, the adjusted sigma estimated based on the traces is also returned.

    .. code-block:: python

      def estimate(self, method: str = "lstsq"):
        # ... existing code from the base class ...
        sigma_ests_adj = []

        for j in range(self.num_jack):
            # ... existing code from the base class ...
            # Adjust the estimate by the effect of traces for heritability calculation
            sigma_est_adj = []
            for i in range(len(sigma_est)):
                sigma_est_adj.append(sigma_est[i] * T[i, self.num_estimates])

        # ... existing code from the base class ...
        # Also include the adjusted sigma_ests_adj
        sigma_ests_adj = np.array(sigma_ests_adj)
        sigma_est_jackknife_adj, sigma_ests_total_adj = sigma_ests_adj[:-1, :], sigma_ests_adj[-1, :]

        sigma_ests_adj.append(sigma_est_adj)

        return sigma_est_jackknife, sigma_ests_total, sigma_est_jackknife_adj, sigma_ests_total_adj

.. py:method:: compute_h2_nonoverlapping(self, sigma_est_jackknife, sigma_ests_total):

    Since the heritability is computed based on the traces and the heritability is computed separately for G and GxE, we also need to override the method to compute the heritability.
    A future work is to also override the method to compute the heritability for the overlapping cases.


.. py:method:: compute_enrichment(self, h2_jackknife, h2_total):

    The enrichment is computed separately for G only based on the original GENIE algorithm. Thus, we also override the enrichment computation method.

Usage Example
------------

.. code-block:: python

   from pyrhe.models import GENIE

   # Initialize model
   genie_model = GENIE(
        genie_model="G+GxE+NxE",
       geno_file="path/to/genotype",
       annot_file="path/to/annotation",
       pheno_file="path/to/phenotype",
       cov_file="path/to/covariate",
       env_file="path/to/environment",
       num_bins=10,
       num_jack=100,
       num_random_vec=10,
       num_workers=5,
       ...
   )

   # Run analysis
   results = genie_model()

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

   