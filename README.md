# PyRHE

PyRHE is an efficient and portable Python package for RHE-mc (Randomized Haseman–Elston regression for Multi-variance Components). It uses `PyTorch` to convert large matrix into tensors to accelerate large matrix multiplication and incorporates multiprocessing to process Jackknife blocks in parallel. It is designed to run on both CPU and CUDA-enabled GPU, and is easy to install and integrate into other applications.

# Installation 

```
git clone git@github.com:jiayini1119/PyRHE.git
pip install pyrhe/
# Also install proper version of PyTorch from https://pytorch.org/ 
```


# Example Usage
```
python run_rhe.py -g {geno_path} --p {pheno_path} -c {covariate_path} --annot {annot_path}-b {num_bin} -k {num_vec} -jn {num_block} --device {device} --cuda_num {cuda_num} --output {output_file} (--streaming)
```
Or you can incorporate RHE in your own project using:

```python
from pyrhe.src.core import RHE

rhe = RHE(
      geno_file={geno_file_path},
      annot_file={annot_file_path},
      pheno_file={pheno_file},
      cov_file={covariate_file_path},
      num_jack={num_jackknife_blocks},
      num_bin={num_bins},
      num_random_vec={num_random_vecs},
      device={device},
      cuda_num ={cuda_number},
      multiprocessing={whether_to_use_multiprocessing}
      num_workers={num_workers_for_multiprocessing},
      seed={seed},
      get_trace={whether_to_get_trace_estimate_summary}
      )

sigma_ests_total, sig_errs, h2_total, h2_errs, enrichment_total, enrichment_errs = rhe()

```

You can call `rhe()` once to get all the estimation summary. In addition, `rhe()` call is composed of several functions that can be used individually (e.g., error estimation). 

```python
def __call__(self, method: str = "QR"):
    # Precompute, should be called before you do any estimation
    self.pre_compute()

    # Get sigma^2 estimation for each jackknife samples and the whole sample
    sigma_est_jackknife, sigma_ests_total = self.estimate(method=method)
    # Get standard error for sigma^2 estimation
    sig_errs = self.estimate_error(sigma_est_jackknife)

    for i, est in enumerate(sigma_ests_total):
        if i == len(sigma_ests_total) - 1:
            print(f"residual variance: {est}, SE: {sig_errs[i]}")
        else:
            print(f"sigma^2 estimate for bin {i}: {est}, SE: {sig_errs[i]}")

    # Get the heritability estimation for each jackknife samples and the whole sample
    h2_jackknife, h2_total = self.compute_h2(sigma_est_jackknife, sigma_ests_total)
    # Get standard error for the heritability estimation
    h2_errs = self.estimate_error(h2_jackknife)

    # Get the enrichment estimation for each jackknife samples and the whole sample
    enrichment_jackknife, enrichment_total = self.compute_enrichment(h2_jackknife, h2_total)
     # Get standard error for the enrichment estimation
    enrichment_errs = self.estimate_error(enrichment_jackknife)

    return sigma_ests_total, sig_errs, h2_total, h2_errs, enrichment_total, enrichment_errs

```

# Other Functionalities 
- Memory efficient (Streaming) estimation（Replace the `RHE` with `StreamingRHE` in the above to get the streaming version）
- Simulating phenotype 
- Getting summary of trace estimate
- Getting other summary statistics (e.g., XtXz)
- To be added

# Comparison between PyRHE & Original RHE
## Accuracy of Estimation
<img width="1141" alt="image" src="https://github.com/jiayini1119/PyRHE/assets/105399924/030dd9f0-c359-409c-934f-8d6114709582">

<img width="1141" alt="image" src="https://github.com/jiayini1119/RHE_project/assets/105399924/71b5cff8-2fdf-42ae-bf29-50e8b93e75f4">

## Runtime Comparision
<img width="862" alt="image" src="https://github.com/jiayini1119/RHE_project/assets/105399924/a629685e-7d56-4b24-a41c-95a0d7c477a5">

<img width="498" alt="image" src="https://github.com/jiayini1119/RHE_project/assets/105399924/e3375751-fe0a-4c24-9bf8-84dff9d91634">

# Example testing pipeline:
The package also supports you to run testing pipelines when some files are missing (e.g., annotation file). In addition, it supports running the [original RHE](https://github.com/sriramlab/RHE-mc) within the package. Here is the example testing pipeline.

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
```
python run_original.py -g {geno_path} -b {num_bin} -c {cov_file_path} -k {num_vec} -jn {num_block} --output {output_file}
```
Then parse the outputs using `parse_output.py`

**5. Run PyRHE**
```
python run_rhe.py -g {geno_path} -b {num_bin} -k {num_vec} -c {cov_file_path} -jn {num_block} --output {output_file}
```
