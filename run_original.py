import subprocess
import os
import argparse
import json
import glob
import re 
import time
from constant import RESULT_DIR, DATA_DIR

def main(args):
    annot_path = f"{DATA_DIR}/annot/annot_{args.num_bin}"
    pheno_dir = f"{DATA_DIR}/pheno/bin_{args.num_bin}/"
    rhemc_mem_path = "/u/home/j/jiayini/project-sriram/RHE-mc/build/RHEmc_mem"  

    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    output_file = f"{RESULT_DIR}_{args.output}_{args.num_bin}.json"

    assert os.access(rhemc_mem_path, os.X_OK), f"{rhemc_mem_path} is not executable"


    pheno_files = glob.glob(os.path.join(pheno_dir, "*.phen"))

    all_results = {}

    pattern = re.compile(r"Sigma\^2_(\d): (\d+\.\d+)  SE: (\d+\.\d+)")

    for pheno_file in pheno_files:
        print(f"processing {pheno_file}")
        start_time = time.time() 
        cmd = [
            rhemc_mem_path,
            "-g", args.geno,
            "-p", pheno_file,
            "-annot", annot_path,
            "-k", str(args.num_vec),
            "-jn", str(args.num_block),
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Original RHE') 
    parser.add_argument('--geno', '-g', type=str, default="/u/scratch/b/bronsonj/geno/25k_allsnps", help='genotype file path')
    parser.add_argument('--covariate', '-c', type=str, default=None, help='covariance file path')
    parser.add_argument('--num_vec', '-k', type=int, default=10, help='The number of random vectors (10 is recommended).')
    parser.add_argument('--num_bin', '-b', type=int, default=8, help='Number of bins')
    parser.add_argument('--num_block', '-jn', type=int, default=100, help='The number of jackknife blocks. (100 is recommended). The higher number of jackknife blocks the higher the memory usage.')
    parser.add_argument("--output", type=str, default="original_result", help='output of the file')
    
    args = parser.parse_args()

    main(args)