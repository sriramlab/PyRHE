import subprocess
import os
import numpy as np
import argparse
import time
from constant import RESULT_DIR, DATA_DIR
from run_rhe import parse_config, convert_to_correct_type

def main(args):
    annot_path = f"{DATA_DIR}/annot/annot_{args.num_bin}" if args.annot is None else args.annot
    rhe_path = "/u/home/j/jiayini/project-sriram/RHE-mc/build/RHEmc_mem" if args.streaming else "/u/home/j/jiayini/project-sriram/RHE-mc/build/RHEmc"  

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

    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in proc.stdout:
                print(line, end='')  
                f.write(line)  
            for line in proc.stderr:
                print(line, end='')  

    end_time = time.time()
    runtime = end_time - start_time
    print(f"RHE original runtime: {runtime:.5f} seconds")

    if not args.benchmark_runtime:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"\nruntime: {runtime:.5f} seconds\n")
    else:
        return runtime

    print("Processing complete for " + pheno_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Original_RHE') 
    parser.add_argument('--streaming', action='store_true', help='use streaming version')
    parser.add_argument('--benchmark_runtime', action='store_true', help='benchmark the runtime')
    parser.add_argument('--geno', '-g', type=str, help='genotype file path')
    parser.add_argument('--pheno', '-p', type=str, help='phenotype file path')
    parser.add_argument('--covariate', '-c', type=str, default=None, help='Covariate file path')
    parser.add_argument('--annot',  type=str, default=None, help='Annotation file path')
    parser.add_argument('--num_vec', '-k', type=int, default=10, help='The number of random vectors.')
    parser.add_argument('--num_bin', '-b', type=int, default=8, help='Number of bins')
    parser.add_argument('--num_block', '-jn', type=int, default=100, help='The number of jackknife blocks.')
    parser.add_argument("--output", '-o', type=str, default="test", help='output of the file')
    parser.add_argument('--config', type=str, help='Configuration file path')

    args = parser.parse_args()

    if args.config:
        config_args = parse_config(args.config, 'Original_RHE_Config')
        for key, default in vars(args).items():
            if key in config_args:
                setattr(args, key, convert_to_correct_type(config_args[key], default))
    
    
    if args.benchmark_runtime:
        runtimes = [] 
        for i in range(3):
            args = parser.parse_args()
            cov = "_with_cov" if args.covariate else ""
            base_pheno_path = f"{args.pheno}/pheno{cov}/bin_{args.num_bin}"
            args.pheno = os.path.join(base_pheno_path, f"{i}.phen")  
            runtime = main(args)
            runtimes.append(runtime)
        
        mean_runtime = np.mean(runtimes)
        std_runtime = np.std(runtimes)
        print(f"runtime: {mean_runtime:.2f} ± {std_runtime:.2f} seconds")

    else:
        main(args)


