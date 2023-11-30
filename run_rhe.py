import argparse
from src.core import RHE, StreamingRHE
from constant import DATA_DIR, RESULT_DIR
import os
import numpy as np
import json
import time
def main(args):

    pheno_file = args.pheno
    annot_path = f"{DATA_DIR}/annot/annot_{args.num_bin}"
    
    print(f"processing {pheno_file}")
    if args.streaming:
        rhe = StreamingRHE(
            geno_file=args.geno,
            annot_file=annot_path,
            pheno_file=pheno_file,
            cov_file=args.covariate,
            num_jack=args.num_block,
            num_bin=args.num_bin,
            num_random_vec=args.num_vec,
        )

    else:
        rhe = RHE(
            geno_file=args.geno,
            annot_file=annot_path,
            pheno_file=pheno_file,
            cov_file=args.covariate,
            num_jack=args.num_block,
            num_bin=args.num_bin,
            num_random_vec=args.num_vec,
        )

    # RHE
    start = time.time()

    sigma_ests_total, sig_errs, h2_total, h2_errs, enrichment_total, enrichment_errs = rhe()

    end = time.time()

    runtime = end - start
    result = {
        "sigma_ests_total": sigma_ests_total.tolist() if isinstance(sigma_ests_total, np.ndarray) else sigma_ests_total,
        "sig_errs": sig_errs.tolist() if isinstance(sig_errs, np.ndarray) else sig_errs,
        "h2_total": h2_total.tolist() if isinstance(h2_total, np.ndarray) else h2_total,
        "h2_errs": h2_errs.tolist() if isinstance(h2_errs, np.ndarray) else h2_errs,
        "enrichment_total": enrichment_total.tolist() if isinstance(enrichment_total, np.ndarray) else enrichment_total,
        "enrichment_errs": enrichment_errs.tolist() if isinstance(enrichment_errs, np.ndarray) else enrichment_errs,
        "runtime": runtime
    }

    use_cov = "cov" if args.covariate is not None else "no_cov"
    result_dir = f"{RESULT_DIR}/pyrhe_output/{use_cov}/bin_{args.num_bin}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    

    output_file_path = os.path.join(result_dir, f"{args.output}.json")

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyRHE') 
    parser.add_argument('--streaming', action='store_true', help='use streaming version')
    parser.add_argument('--geno', '-g', type=str, default="/u/scratch/b/bronsonj/geno/25k_allsnps", help='genotype file path')
    parser.add_argument('--pheno', '-p', type=str, default=None, help='phenotype file path')
    parser.add_argument('--covariate', '-c', type=str, default="/u/home/j/jiayini/project-sriram/RHE_project/data/cov_25k.cov", help='covariance file path')
    parser.add_argument('--num_vec', '-k', type=int, default=10, help='The number of random vectors (10 is recommended).')
    parser.add_argument('--num_bin', '-b', type=int, default=8, help='Number of bins')
    parser.add_argument('--num_block', '-jn', type=int, default=100, help='The number of jackknife blocks. (100 is recommended). The higher number of jackknife blocks the higher the memory usage.')
    parser.add_argument("--output", '-o', type=str, default="test", help='output of the file')

    
    # args = parser.parse_args()

    # main(args)

    base_pheno_path = "/u/home/j/jiayini/project-sriram/RHE_project/data/pheno_with_cov/bin_8"

    for i in range(25):
        args = parser.parse_args()
        args.pheno = os.path.join(base_pheno_path, f"{i}.phen")  
        args.output = f"output_{i}"  
        main(args)
