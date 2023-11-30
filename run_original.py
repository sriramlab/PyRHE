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
    pheno_dir = f"{DATA_DIR}/pheno_with_cov/bin_{args.num_bin}/" if args.covariate else f"{DATA_DIR}/pheno/bin_{args.num_bin}/"
    rhe_path = "/u/home/j/jiayini/project-sriram/RHE-mc/build/RHEmc_mem"  

    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    output_file = f"{RESULT_DIR}/{args.output}_{args.num_bin}.json"
    assert os.access(rhe_path, os.X_OK), f"{rhe_path} is not executable"
    pheno_files = glob.glob(os.path.join(pheno_dir, "*.phen"))

    sigma_pattern = re.compile(r"Sigma\^2_(\d): (-?\d+\.\d+)  SE: (-?\d+\.\d+)")
    h2_pattern = re.compile(r"h\^2 of bin (\d) : (-?\d+\.\d+) SE: (-?\d+\.\d+)")
    enrichment_pattern = re.compile(r"Enrichment of bin (\d) : (-?\d+\.\d+) SE: (-?\d+\.\d+)")

    all_results = {}

    for pheno_file in pheno_files:
        print(f"processing {pheno_file}")
        start_time = time.time() 
        cmd = [
            rhe_path,
            "-g", args.geno,
            "-p", pheno_file,
            "-annot", annot_path,
            "-c", args.covariate,
            "-k", str(args.num_vec),
            "-jn", str(args.num_block),
        ]
        
        result = subprocess.run(cmd, text=True, capture_output=True)
        end_time = time.time()

        if result.returncode != 0:
            print(f"Error with {pheno_file}:")
            print(result.stderr)
            continue

        print(result.stdout)
        print(f"RHE original runtime: {end_time - start_time:.5f} seconds")

        sigma_matches = sigma_pattern.findall(result.stdout)
        h2_matches = h2_pattern.findall(result.stdout)
        enrichment_matches = enrichment_pattern.findall(result.stdout)

        pheno_results = {
            "sigma": [{ "index": int(m[0]), "sigma": float(m[1]), "SE": float(m[2])} for m in sigma_matches],
            "h2": [{ "bin": int(m[0]), "h2": float(m[1]), "SE": float(m[2])} for m in h2_matches],
            "enrichment": [{ "bin": int(m[0]), "enrichment": float(m[1]), "SE": float(m[2])} for m in enrichment_matches]
        }
        
        all_results[pheno_file] = pheno_results

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

    print("Processing complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Original RHE') 
    parser.add_argument('--geno', '-g', type=str, default="/u/scratch/b/bronsonj/geno/25k_allsnps", help='genotype file path')
    parser.add_argument('--covariate', '-c', type=str, default="/u/home/j/jiayini/project-sriram/RHE_project/data/cov_25k.cov", help='covariance file path')
    parser.add_argument('--num_vec', '-k', type=int, default=10, help='The number of random vectors (10 is recommended).')
    parser.add_argument('--num_bin', '-b', type=int, default=8, help='Number of bins')
    parser.add_argument('--num_block', '-jn', type=int, default=100, help='The number of jackknife blocks. (100 is recommended). The higher number of jackknife blocks the higher the memory usage.')
    parser.add_argument("--output", type=str, default="original_result", help='output of the file')
    
    args = parser.parse_args()
    main(args)
