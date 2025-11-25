# PyRHE

[![Documentation Status](https://readthedocs.org/projects/pyrhe/badge/?version=latest)](https://pyrhe.readthedocs.io/en/latest/?badge=latest)

PyRHE is a unified and efficient Python framework for genomics heritability estimation. It provides a modular and extensible platform for implementing various genetic architecture estimation models and computation optimizations for large-scale genomic data. 

The full documentation is available at the [PyRHE Documentation](https://pyrhe.readthedocs.io/).

## Key Features

- **High computational efficiency** through distributed jackknife subsamples and parallelized genotype I/O and large-scale matrix operations.
- **Tensor-based computation** with automatic conversion of large matrices to PyTorch tensors, designed to run efficiently on both CPU and CUDA-enabled GPU architectures.
- **Memory-efficient streaming support** through the `StreamingBase` class, enabling memory-efficient processing of large-scale genomic data.
- **Modular, extensible design** with abstract base classes (`Base`, `StreamingBase`) that provide interfaces for adding new models.
- **Multiple models in one framework**, including [RHE](https://www.nature.com/articles/s41467-020-17576-9), [RHE-DOM](https://www.sciencedirect.com/science/article/pii/S0002929721001026), and [GENIE](https://www.nature.com/articles/s41467-020-17576-9), all sharing common infrastructure.

# Installation 

```
pip install pyrhe
# Also install proper version of PyTorch from https://pytorch.org/ 
```

# Example Usage

```python
from pyrhe.models import (
    RHE,
    StreamingRHE,
    GENIE,
    StreamingGENIE,
    RHE_DOM,
    StreamingRHE_DOM,
)

# Standard RHE
rhe_model = RHE(
    geno_file="path/to/genotype",
    annot_file="path/to/annotation",
    pheno_file="path/to/phenotype",
    # other arguments...
)
rhe_results = rhe_model()

# Streaming RHE
streaming_rhe_model = StreamingRHE(
    geno_file="path/to/genotype",
    annot_file="path/to/annotation",
    pheno_file="path/to/phenotype",
    # other arguments...
)
streaming_results = streaming_rhe_model()
```
Each model (e.g., RHE, GENIE, RHE_DOM and their streaming version) follows the same pattern:
initialize with file paths and options, then call the model instance to run estimation and return results.

# Run analysis

After installing the package, you can run PyRHE directly from the command line:
```
python run_rhe.py <command-line arguments>
```
Alternatively, you may run PyRHE using a newline-separated config file:
```
python run_rhe.py --config <config file>
```

# Parameters
```
model: The model to run (e.g., rhe, rhe_dom, genie).
genotype (-g): The path of PLINK BED genotype file
phenotype (-p): The path of phenotype file
covariate (-c): The path of covariate file
annotation (-annot): The path of genotype annotation file.
num_vec (-k): The number of random vectors (10 is recommended). 
num_block (-jn): The number of jackknife blocks (100 is recommended). 
    The higher the number of jackknife blocks, the higher the memory usage.
output (-o): The path of the output file prefix
streaming: Whether to use the streaming version or not
num_workers: The number of workers
seed (-s): The random seed
device: Device to use (cpu or gpu)
      Using CPU already enables great performance. You can further improve performance using GPU
cuda_num: CUDA number of GPU
geno_impute_method: How to impute missing genotype ("binary" (binary imputation) or "mean" (mean imputation))
cov_impute_method: How to impute missing covariate ("ignore" (ignore individuals with missing covariate) or "mean" (mean imputation))
samp_prev: Sample prevalence of binary phenotype (for conversion to liability scale)
pop_prev: Population prevalence of binary phenotype (for conversion to liability scale)
trace (-tr): Save the stochastic trace estimates as trace summary statistics (.trace) with metadata (.MN)
trace_dir: Directory to save the trace estimates
```

Please refer to the [example](https://github.com/sriramlab/PyRHE/tree/main/example) for a list of configuration files for running RHE, RHE_DOM, and GENIE, and their respective outputs.

# Accuracy of Estimation

<img width="848" height="566" alt="Screenshot 2025-11-24 at 21 06 56" src="https://github.com/user-attachments/assets/2dd6ce55-1ebb-4168-8c5d-e99da3c19635" /> 

# Runtime Comparison

<img width="863" height="394" alt="Screenshot 2025-11-24 at 21 07 33" src="https://github.com/user-attachments/assets/62b74ca6-8545-4f71-a009-78425ca645da" />
<img width="518" height="506" alt="Screenshot 2025-11-24 at 21 07 44" src="https://github.com/user-attachments/assets/370694f9-cda2-4e87-ac91-eec6decd6655" />
