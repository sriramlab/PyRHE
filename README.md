# PyRHE

[![Documentation Status](https://readthedocs.org/projects/pyrhe/badge/?version=latest)](https://pyrhe.readthedocs.io/en/latest/?badge=latest)

PyRHE is a unified and efficient Python framework for genomics heritability estimation. It provides a modular and extensible platform for implementing various genetic architecture estimation models and computation optimizations for large-scale genomic data.

The full documentation is available at [pyrhe.readthedocs.io](https://pyrhe.readthedocs.io/).

# Installation 

```
pip install pyrhe
# Also install proper version of PyTorch from https://pytorch.org/ 
```

# Example Usage

Run PyRHE as follows:
```
python run_rhe.py <command_line arguments>
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
