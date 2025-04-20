Base Class
==========

The ``Base`` class serves as the foundation for all models in PyRHE, providing common infrastructure and extensible interfaces for model-specific implementations.

The core components of the RHE-based algorithms are comprised of two main components: 

1. Precomputation. This part uses jackknife subsampling to calculate the statistics like XXz and yXXy for each jackknife blocks. 

2. Estimation. This part uses the previously calculated statistics to construct and solve normal equations for methods-of-moments estimation.

Mainly, we delegate the precomputation part to each sub-classes, and the data input processing, aggregation of pre-computation results, and estimation part are implemented in the ``Base`` class. 

Core Infrastructure
-----------------

Data Input and Preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``Base`` class handles:

- Reading PLINK-format genotype files (.bed, .bim, .fam)
- Processing phenotype files and covariates
- Handling missing data and imputation

A key component of processing the genotype matrix is that the internal genotype matrix is not fully loaded into memory; instead, it is streamed
in blocks as needed. This part is abstracted in the `Base` class.

Jackknife Resampling
~~~~~~~~~~~~~~~~~~

Jackknife resampling is used for

1. Calculating standard errors. 

2. Process the genotype matrix in blocks and parallelize processing them by block.

The Base class provides the functionalities for:

- Partitioning of SNPs into jackknife subsamples (The partitioned genotype matrix is stored as `all_gen` which are used by the model-specific classes)
- Aggregation of precomputation results across subsamples
- Calculation of standard errors based on the jackknife subsamples.

Tensor-based Computation
~~~~~~~~~~~~~~~~~~~~~~~

- Methods like ``_compute_XXz`` and ``_compute_yXXy``, which are used to calculate the statistics like ``XXz`` and ``yXXy`` for each jackknife blocks using tensor operations, are defined in the Base class for individual model to use.

Multiprocessing and Memory Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Base class also provides the functionalities for multiprocessing and memory management, specifically: 

- Each jackknife block is assigned to a worker for parallel computation.
- The base class provides basic memory arrays for large tensors and shared memory management.
- Aggregation of indivdual worker results is abstracted

Estimation Logic
~~~~~~~~~~~~~~

To construct the estimation part, the Base class provides the following functionalities:
- Construction of normal equations
- Multiple solver options (least squares, QR decomposition)
- Computation of heritability and enrichment along with their standard errors
- Liability-scale corrections

Key Member Variables in the Base Class:
-------------------------------------

- ``self.M``: Matrix representing the number of SNPs in each jackknife subsample in each bin. The last row is the total number of SNPs in each bin.
- ``self.len_bin``: Total number of SNPs in each bin. This should be specified by the model-specific classes in the ``get_M_last_row()`` method.
- ``self.XXz / self.yXXy / self.UXXz / self.XXUz``: Jackknife statistics. If the covariate is provided, the UXXz and XXUz will be calculated.
- ``self.num_estimates``: Number of estimates. This should be specified by the model-specific classes in the ``get_num_estimates()`` method.

Class Interface
-------------

Initialization
~~~~~~~~~~~~

.. py:class:: Base

   .. py:method:: __init__(self, model, geno_file[, annot_file, pheno_file, ...])

      Initialize the Base class.

      :param str model: The model to use (e.g., "rhe", "rhe_dom", "genie")
      :param str geno_file: Path to the genotype file
      :param str annot_file: Path to the annotation file (Optional)
      :param str pheno_file: Path to the phenotype file (Optional)
         - If the phenotype file is not provided, the model will simulate the phenotypes.
      :param str cov_file: Path to the covariate file (Optional)
      :param int num_bin: Number of bins for partitioning (default: 8)
         - If the annotation file is provided, the number of bins will be the number of unique annotations.
         - If not, annotation file will be automatically generated and saved.
      :param int num_jack: Number of jackknife samples (default: 1)
      :param int num_random_vec: Number of random vectors (default: 10)
      :param str geno_impute_method: Method for genotype imputation (default: "binary")
      :param str cov_impute_method: Method for covariate imputation (default: "ignore")
      :param bool cov_one_hot_conversion: Convert categorical covariates to one-hot encoding (default: False)
      :param int categorical_threshhold: Threshold for categorical variables (default: 100)
      :param str device: Computation device ("cpu" or "cuda") (default: "cpu")
      :param int cuda_num: Specific CUDA device number (optional)
      :param int num_workers: Number of parallel workers (optional)
      :param bool multiprocessing: Enable multiprocessing (default: True)
      :param int seed: Random seed for reproducibility (optional)
      :param bool get_trace: Compute and store trace estimates (default: False)
      :param str trace_dir: Directory for trace estimates (optional)
      :param float samp_prev: Sample prevalence for binary traits (optional)
      :param float pop_prev: Population prevalence for binary traits (optional)
      :param Logger log: Logger instance for progress tracking (optional)

Key Methods
~~~~~~~~~

.. py:method:: read_geno(start, end)

   Read genotype data from the specified range. This is to process genotype data in blocks.

.. py:method:: impute_geno(X)

   Impute missing genotype data using either binary MAF-based sampling or mean imputation.

.. py:method:: standardize_geno(geno)

   The base class provides a standardization method to use for the standardization of the genotype matrix assuming it follows binomial distribu- tion. If not, individual models can define their own standardization methods.

.. py:method:: partition_bins(self, geno: np.ndarray, annot: np.ndarray):

    Partition the genotype matrix into num_bin bins. The partitioned genotype matrix is stored as `all_gen` which are used by the model-specific classes.

.. py:method:: _get_jacknife_subsample(self, jack_index: int):

    Get the jackknife subsample of the genotype matrix.

.. py:method:: aggregate():

    Aggregate the precomputation results across subsamples.

.. py:method:: setup_lhs_rhs_jackknife()

    Set up the left-hand side and right-hand side of the normal equations for each jackknife subsample.

.. py:method:: estimate(method="lstsq")

   Estimate model parameters.

Abstract Methods
-------------

To extend the Base class for new models, the following methods need to be implemented:

.. py:method:: get_num_estimates()

   Returns the number of variance components to estimate.

   - RHE: Returns ``num_bin`` for additive effects
   - RHE-DOM: Returns ``num_bin * 2`` for both additive and dominant effects
   - GENIE: Returns varying numbers based on model type

.. py:method:: get_M_last_row()

   Specifies the structure of the M matrix.

   - RHE: Returns ``len_bin`` for additive effects
   - RHE-DOM: Returns concatenated ``[len_bin, len_bin]`` for both effects
   - GENIE: Returns varying structures based on model components

.. py:method:: pre_compute_jackknife_bin(j, all_gen)

   Implements model-specific pre-computation. j is the index of the jackknife subsample. all_gen is the partitioned genotype matrix.

   - RHE: Standardizes genotypes and computes basic statistics
   - RHE-DOM: Handles both original and encoded genotypes
   - GENIE: Manages environment-specific computations

.. py:method:: b_trace_calculation()

   Handles trace calculations.

   - RHE: Returns ``num_indv`` for standardized genotypes
   - RHE-DOM: Similar to RHE since the genotype matrix is also standardized
   - GENIE: Varies based on model components

.. py:method:: run(method)

   How to run the model and what statistics to return (E.g., sigma estimates, heritability, enrichment, etc.)

Example for Extending the Base Class to New Models
--------------------------------------------------

To create a new model in PyRHE, you need to extend the ``Base`` class and implement the required abstract methods. Here's an example of how to create a new model called ``NewModel``:

.. code-block:: python

   from pyrhe.src.base import Base

   class NewModel(Base):
       """A new model extending the Base class."""
       
       def get_num_estimates(self):
           """Return the number of variance components to estimate.
           This should match the number of components in your model.

           Returns:
               int: Number of variance components
           """
           return self.num_bin  # Adjust based on your model's needs
       
       def get_M_last_row(self):
           """Specify the last row of the M matrix.
           
           
           Returns:
               np.ndarray: the last row of the M matrix
           """
           return self.len_bin  # Adjust based on your model's structure
       
       def pre_compute_jackknife_bin(self, j, all_gen):
           """Implement model-specific pre-computation.

           ``all_gen`` is the partitioned genotype matrix. j is the index of the jackknife subsample.
           
           This method should:
           1. Process each genotype block
           2. Compute necessary statistics
           3. Store results in appropriate arrays
           
           Args:
               j: Jackknife sample index
               all_gen: List of genotype matrices for each bin
           """
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
       
       def b_trace_calculation(self, k, j, b_idx):
           """Calculate trace terms for estimation.
           
           Args:
               k: Bin index
               j: Jackknife sample index
               b_idx: Random vector index
               
           Returns:
               float: Trace value
           """
           return self.num_indv  # Adjust based on your model's needs
       
       def run(self, method="lstsq"):
           """Run the complete analysis and return results.
           
           For example, you can return the following statistics:
           1. Estimate parameters
           2. Compute heritability and other statistics
           3. Handle special cases (e.g., binary traits)
           4. Return comprehensive results
           
           Args:
               method: Estimation method to use
               
           Returns:
               dict: Dictionary of results
           """
           # 1. Estimate variance components
           sigma_est_jackknife, sigma_ests_total = self.estimate(method=method)
           sig_errs = self.estimate_error(sigma_est_jackknife)
           
           # 2. Compute heritability
           h2_jackknife, h2_total = self.compute_h2_nonoverlapping(sigma_est_jackknife, sigma_ests_total)
           h2_errs = self.estimate_error(h2_jackknife)
           
           # 3. Compute enrichments
           enrichment_jackknife, enrichment_total = self.compute_enrichment(h2_jackknife, h2_total)
           enrichment_errs = self.estimate_error(enrichment_jackknife)
           
           # 4. Handle binary traits if needed
           if self.binary_pheno:
               # Compute liability-scale heritability
               liability_results = self.calculate_liability_h2(h2_total, h2_errs)
           
           # 5. Return comprehensive results
           return {
               "sigma_ests_total": sigma_ests_total,
               "sig_errs": sig_errs,
               "h2_total": h2_total,
               "h2_errs": h2_errs,
               "enrichment_total": enrichment_total,
               "enrichment_errs": enrichment_errs,
               # Add any model-specific results
           }