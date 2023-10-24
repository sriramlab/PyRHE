import subprocess
import os
import json
import glob
import re 
import time

geno_path = "/u/scratch/b/bronsonj/geno/25k_allsnps"
annot_path = "/u/home/j/jiayini/project-sriram/RHE_project/data/annot/annot_1"
pheno_dir = "/u/home/j/jiayini/project-sriram/RHE_project/data/pheno/bin_1/"
rhemc_mem_path = "/u/home/j/jiayini/project-sriram/RHE-mc/build/RHEmc_mem"  
output_file = "original_results.json"

assert os.access(rhemc_mem_path, os.X_OK), f"{rhemc_mem_path} is not executable"


pheno_files = glob.glob(os.path.join(pheno_dir, "*.phen"))

all_results = {}

pattern = re.compile(r"Sigma\^2_(\d): (\d+\.\d+)  SE: (\d+\.\d+)")

for pheno_file in pheno_files:
    print(f"processing {pheno_file}")
    start_time = time.time() 
    cmd = [
        rhemc_mem_path,
        "-g", geno_path,
        "-p", pheno_file,
        "-annot", annot_path,
        "-k", "10",
        "-jn", "100",
    ]
    
    result = subprocess.run(cmd, text=True, capture_output=True)

    if result.returncode == 0:
        print(result.stdout)
        matches = pattern.findall(result.stdout)
        sigmas = [{"index": int(m[0]), "sigma": float(m[1]), "SE": float(m[2])} for m in matches]
        
        pheno_results = {}
        for sigma in sigmas:
            pheno_results[f"sigma_est_{sigma['index']}"] = sigma['sigma']
            pheno_results[f"sigma_err_{sigma['index']}"] = sigma['SE']
        
        all_results[pheno_file] = pheno_results

    else:
        print(f"Error with {pheno_file}:")
        print(result.stderr)

    end_time = time.time()
    print(f"RHE original runtime: {end_time - start_time:.5f} seconds")

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=4)

print("Processing complete.")