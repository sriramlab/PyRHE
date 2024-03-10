import subprocess
import os
import argparse
import time
from constant import RESULT_DIR, DATA_DIR

def main(args):
    annot_path = f"{DATA_DIR}/annot/annot_{args.num_bin}"
    rhe_path = "/u/home/j/jiayini/project-sriram/RHE-mc/build/RHEmc_mem"  

    output_dir = f"{RESULT_DIR}/original_result/{'cov' if args.covariate else 'no_cov'}/bin_{args.num_bin}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = f"{output_dir}/{args.output}.txt"
    assert os.access(rhe_path, os.X_OK), f"{rhe_path} is not executable"

    pheno_file = args.pheno

    print(f"processing {pheno_file}")
    start_time = time.time() 

    cmd = [
        rhe_path,
        "-g", args.geno,
        "-p", pheno_file,
        "-annot", annot_path,
        "-k", str(args.num_vec),
        "-jn", str(args.num_block),
    ]

    if args.covariate is not None:
        cmd += ["-c", args.covariate]

    result = subprocess.run(cmd, text=True, capture_output=True)
    end_time = time.time()

    if result.returncode != 0:
        print(f"Error with {pheno_file}:")
        print(result.stderr)
        return

    runtime = end_time - start_time
    print(f"RHE original runtime: {runtime:.5f} seconds")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(result.stdout)
        f.write(f"\nruntime: {runtime:.5f} seconds\n")  

    print("Processing complete for " + pheno_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyRHE') 
    parser.add_argument('--streaming', action='store_true', help='use streaming version')
    parser.add_argument('--geno', '-g', type=str, default="/home/jiayini1119/data/200k_allsnps", help='genotype file path')
    parser.add_argument('--pheno', '-p', type=str, help='phenotype file path')
    parser.add_argument('--covariate', '-c', type=str, help='Covariate file path')
    parser.add_argument('--num_vec', '-k', type=int, default=10, help='The number of random vectors.')
    parser.add_argument('--num_bin', '-b', type=int, default=8, help='Number of bins')
    parser.add_argument('--num_block', '-jn', type=int, default=100, help='The number of jackknife blocks.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--device', type=int, help="gpu number")
    parser.add_argument("--output", '-o', type=str, default="test", help='output of the file')

    for i in range(25):
        args = parser.parse_args()
        cov = "_with_cov" if args.covariate else ""
        base_pheno_path = f"/home/jiayini1119/RHE_project/data_200k/pheno{cov}/bin_{args.num_bin}"
        args.pheno = os.path.join(base_pheno_path, f"{i}.phen")  
        args.seed = i
        args.output = f"output_{i}"  
        main(args)

