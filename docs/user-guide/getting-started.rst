Getting Started with PyRHE
=========================

This guide will help you get started with PyRHE by walking you through installation, configuration, and basic usage examples.

.. contents:: Table of Contents
   :local:
   :depth: 2

Installation
-----------


1. Create a virtual environment:

.. code-block:: bash

   python -m venv pyrhe-env
   source pyrhe-env/bin/activate  # On Windows, use: pyrhe-env\Scripts\activate
   # Alternatively, you can also use conda:
   conda create -n pyrhe-env python=3.10
   conda activate pyrhe-env

2. Install PyRHE:

.. code-block:: bash

   pip install pyrhe

3. Install PyTorch:

Install proper version of PyTorch from https://pytorch.org/ 

4. Verify installation:

.. code-block:: python

   import pyrhe
   print(pyrhe.__version__)

Configuration Guide
-----------------

Data Input
~~~~~~~~~

- ``model``: The model to run
  - Options: ``rhe``, ``rhe_dom``, ``genie``
  - Example: ``model="rhe"``

- ``genotype`` (``-g``): Path to PLINK BED genotype file
  - Format: PLINK BED format
  - Example: ``genotype="path/to/genotype.bed"``

- ``phenotype`` (``-p``): Path to phenotype file
  - Format: Space/tab-delimited text file
  - Example: ``phenotype="path/to/phenotype.txt"``

- ``covariate`` (``-c``): Path to covariate file
  - Format: Space/tab-delimited text file
  - Example: ``covariate="path/to/covariates.txt"``

- ``annotation`` (``-annot``): Path to genotype annotation file
  - Format: Space/tab-delimited text file
  - Example: ``annotation="path/to/annotations.txt"``

Model Settings
~~~~~~~~~~~~~
- ``num_vec`` (``-k``): Number of random vectors
  - Default: 10
  - Example: ``num_vec=10``

- ``num_block`` (``-jn``): Number of jackknife blocks
  - Default: 100
  - Note: Higher values increase memory usage
  - Example: ``num_block=100``

- ``output`` (``-o``): Output file
  - Example: ``output="results/analysis"``

- ``streaming``: Use streaming version
  - Example: ``streaming=True``

- ``num_workers``: Number of parallel workers
  - Default: Number of CPU cores
  - Example: ``num_workers=4``

- ``device``: Computation device
  - Options: ``"cpu"``, ``"gpu"``
  - Default: ``"cpu"``
  - Note: CPU already provides good performance. You can further improve performance using GPU
  - Example: ``device="gpu"``

- ``cuda_num``: CUDA device number for GPU
  - Required when using GPU
  - Example: ``cuda_num=0``

- ``seed`` (``-s``): Random seed
  - Example: ``seed=42``

Data Processing
~~~~~~~~~~~~~

- ``geno_impute_method``: Genotype imputation method
  - Options: ``"binary"``, ``"mean"``
  - Default: ``"binary"``
  - Example: ``geno_impute_method="mean"``

- ``cov_impute_method``: Covariate imputation method
  - Options: ``"ignore"``, ``"mean"``
  - Default: ``"ignore"``
  - Example: ``cov_impute_method="mean"``

Binary Trait Settings
~~~~~~~~~~~~~~~~~~

- ``samp_prev``: Sample prevalence for binary traits
  - Required for liability scale conversion
  - Example: ``samp_prev=0.1``

- ``pop_prev``: Population prevalence for binary traits
  - Required for liability scale conversion
  - Example: ``pop_prev=0.01``

Trace Estimation
~~~~~~~~~~~~~~

- ``trace`` (``-tr``): Save trace estimates
  - Default: False
  - Saves trace summary statistics (.trace) and metadata (.MN)
  - Example: ``trace=True``

- ``trace_dir``: Directory for trace estimates
  - Required when trace=True
  - Example: ``trace_dir="results/traces"``

Example Running the Model
-------------------------

Run PyRHE as follows:

.. code-block:: bash

   python run_rhe.py <command_line arguments>

Alternatively, you may run PyRHE using a newline-separated config file. It is recommended as this makes the configuration cleaner and easier to manage.

.. code-block:: bash

   python run_rhe.py --config <config file>

Example Configuration File
------------------------

.. code-block:: text

   [PyRHE_Config]
    model = rhe
    genotype = test
    phenotype = test.pheno
    annotation=single.annot
    covariate=test.cov
    cov_one_hot_conversion = yes
    output=outputs/rhe/no_streaming_bin_1.txt
    num_vec = 10
    num_bin =  1
    num_workers = 5
    num_block = 100
    streaming = no
    debug = yes
    benchmark_runtime = no
    geno_impute_method = binary
    cov_impute_method = ignore
    trace = yes

Example Output
-------------

.. code-block:: text

    ##################################
    #                                #
    #          PyRHE (v1.0.0)        #
    #                                #
    ##################################


    Active essential options:
        -g (genotype) test
        -annot (annotation) single.annot
        -p (phenotype) test.pheno
        -c (covariates) test.cov
        -o (output) outputs/rhe/no_streaming_bin_1.txt
        -k (# random vectors) 10
        -jn (# jackknife blocks) 100
        --num_workers 5
        --device cpu
        --geno_impute_method binary
        --cov_impute_method ignore


    Number of traits: 1
    Rank of the covariate matrix: 5
    Number of individuals after filtering: 5000
    Number of covariates: 5
    *****
    Number of features in bin 0 : 10000
    *****
    OUTPUT FOR TRAIT 0: 
    Saved trace summary into run_test.pheno(.tr/.MN)
    Variance components: 
    Sigma^2_g[0] : 0.16241267611557975  SE : 0.030938069538554984
    Sigma^2_e : 0.7969651822679482  SE : 0.030947296404198268
    *****
    Heritabilities:
    h2_g[0] : 0.16928958146817316 : 0.03224962407730982
    Total h2 : 0.16928958146817316 SE: 0.03224962407730982
    *****
    Enrichments: 
    Enrichment g[0] : 1.0 SE : 0.0
    *****
    *****
    Heritabilities and enrichments computed based on overlapping setting
    Heritabilities:
    h2_g[0] : 0.16928958146814096 : 0.03224962407728962
    Total h2 : 0.16928958146817316 SE: 0.03224962407730982
    Enrichments (overlapping def):
    Enrichment g[0] : 0.9999999999998098 SE : 1.2257069904382272e-12
    Runtime:  3.243225574493408


Please refer to the `example`_ folder for a list of configuration files and their corresponding output files.

Example Using the Class
----------------------
Alternatively, you can run the model by using the class directly and integrate the results into your own project.


.. code-block:: python

   from pyrhe.models import RHE

   # Initialize model with configuration
   model = RHE(
       model="rhe",
       genotype="data/genotype.bed",
       phenotype="data/phenotype.txt",
       covariate="data/covariates.txt",
       annotation="data/annotations.txt",
       num_vec=10,
       num_block=100,
       output="results/analysis",
       streaming=True,
       num_workers=4,
       device="cpu",
       geno_impute_method="binary",
       cov_impute_method="ignore",
       trace=True,
       trace_dir="results/traces"
   )

   # Run analysis
   results = model()

   # Access results
   print(results)
   print(results['sigma_ests_total'])
