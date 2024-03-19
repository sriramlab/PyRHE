# RHE_project

PyRHE is a Python package for RHE-mc (Randomized Haseman–Elston regression for Multi-variance Components). It is easily portable and efficient. It uses `PyTorch` to convert large matrix into tensors to accelerate large matrix multiplication (supporting both CPU and CUDA version) and uses `multiprocessing` to process jackknife blocks in parallel. 

# Installation (TODO)

```
git clone xxxxxxx
pip install -r requirements.txt
pip install pyrhe/
```

# Example Usage

```python
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

The package also supports streaming (memory-efficient) version of RHE. Replace the `RHE` with `StreamingRHE` in the above to get the streaming version. 

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

# Other Functionalities (TODO: ADD)

# Comparison between PyRHE & Original RHE
## Accuracy of Estimation
<img width="1141" alt="image" src="https://github.com/jiayini1119/RHE_project/assets/105399924/71b5cff8-2fdf-42ae-bf29-50e8b93e75f4">

## Runtime Comparision
<img width="862" alt="image" src="https://github.com/jiayini1119/RHE_project/assets/105399924/a629685e-7d56-4b24-a41c-95a0d7c477a5">

<img width="498" alt="image" src="https://github.com/jiayini1119/RHE_project/assets/105399924/e3375751-fe0a-4c24-9bf8-84dff9d91634">

# Example testing pipeline:
The package also supports you to run testing pipelines when some files are missing (e.g., annotation file). Here is the example testing pipeline.
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
**5. Run python RHE**
```
python run_rhe.py -g {geno_path} -b {num_bin} -k {num_vec} -c {cov_file_path} -jn {num_block} --output {output_file}
```

**6. Visualize**
run cells in `plotting.ipynb`.
