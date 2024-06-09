# PyRHE

[PyRHE](https://pypi.org/project/pyrhe/) is an efficient and portable Python package for RHE-mc (Randomized Haseman–Elston regression for Multi-variance Components). It converts large matrix into tensors to accelerate large matrix multiplication and incorporates multiprocessing to process Jackknife blocks in parallel. It is designed to run on both CPU and CUDA-enabled GPU, and is easy to install and integrate into other applications.

# Installation 

```
pip install rhe
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
See the [example](https://github.com/sriramlab/PyRHE/tree/main/example) folder for an example usage.

# Parameters
```
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

PyRHE is easily incorporated in your own project. Here is [an example notebook](https://github.com/sriramlab/PyRHE/blob/main/small_sample.ipynb) for how to do so.


# Comparison between PyRHE & Original RHE
## Accuracy of Estimation
<img width="1141" alt="image" src="https://github.com/jiayini1119/PyRHE/assets/105399924/030dd9f0-c359-409c-934f-8d6114709582">

<img width="1141" alt="image" src="https://github.com/jiayini1119/RHE_project/assets/105399924/71b5cff8-2fdf-42ae-bf29-50e8b93e75f4">

## Runtime Comparision
<img width="862" alt="image" src="https://github.com/jiayini1119/RHE_project/assets/105399924/a629685e-7d56-4b24-a41c-95a0d7c477a5">

<img width="498" alt="image" src="https://github.com/jiayini1119/RHE_project/assets/105399924/e3375751-fe0a-4c24-9bf8-84dff9d91634">

# Example testing pipeline:
Here is the example testing pipeline. You can run testing pipelines when some files are missing (e.g., annotation file). 

**1. Set Up**
Create a `.env` file and specify the `RESULT_DIR` (where you store the results) and `DATA_DIR` (store the simulated phenotype, generated annotation file, etc.)

**2. Generate Annotation File:**  
```
cd core
python generate_annot.py -g {geno_path} -b {num_bin} -o {output_file}
```
**3. Simulate Phenotype:**  

Use the [Simulator](https://github.com/sriramlab/Simulator) to simulate phenotype without covariate.
If want to add covariate, do 
```
cd core
python simulate_pheno.py -b {num_bin} -c {cov_file_path}
```

**4. Run original RHE**  
Running the [original RHE](https://github.com/sriramlab/RHE-mc) using
```
python run_original.py -g {geno_path} -b {num_bin} -c {cov_file_path} -k {num_vec} -jn {num_block} --output {output_file}
```
Then parse the outputs using `parse_output.py`

**5. Run PyRHE**
```
python run_rhe.py -g {geno_path} -b {num_bin} -k {num_vec} -c {cov_file_path} -jn {num_block} --output {output_file}
```
