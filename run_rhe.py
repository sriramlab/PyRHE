import argparse
import os
import torch
import numpy as np
from src.core import RHE, StreamingRHE
from constant import DATA_DIR, RESULT_DIR
import json
import time


def main(args):

    print(args)

    # Device
    device = "cpu"
    if torch.cuda.is_available():
        if args.device >= 0:
            device = f"cuda:{args.device}"
        else:
            device = "cuda"
    else:
        print("cuda not available, fall back to cpu")
        device = "cpu"
    device = torch.device(device)

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
            device=device,
            multiprocessing=args.multiprocessing,
            num_workers=args.num_workers,
            seed=args.seed,
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
            device=device,
            multiprocessing=args.multiprocessing,
            num_workers=args.num_workers,
            seed=args.seed,
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
    parser.add_argument('--multiprocessing', action='store_true', help='use streaming version')

    parser.add_argument('--geno', '-g', type=str, default="/home/jiayini1119/data/200k_allsnps", help='genotype file path')
    parser.add_argument('--pheno', '-p', type=str, default=None, help='phenotype file path')
    parser.add_argument('--covariate', '-c', type=str, default=None, help='Covariate file path')
    parser.add_argument('--num_vec', '-k', type=int, default=10, help='The number of random vectors (10 is recommended).')
    parser.add_argument('--num_bin', '-b', type=int, default=8, help='Number of bins')
    parser.add_argument('--num_workers', type=int, default=10, help='Number of workers')
    parser.add_argument('--num_block', '-jn', type=int, default=100, help='The number of jackknife blocks. (100 is recommended). The higher number of jackknife blocks the higher the memory usage.')
    parser.add_argument('--seed', default=0, help='Random seed')
    parser.add_argument('--device', type=int, default=-1, help="gpu number")
    parser.add_argument("--output", '-o', type=str, default="test", help='output of the file')

    
    # args = parser.parse_args()

    # main(args)


    for i in range(25):
        args = parser.parse_args()
        if args.covariate is not None:
            cov = "_with_cov"
        else:
            cov = ""
        base_pheno_path = f"{DATA_DIR}/pheno{cov}/bin_{args.num_bin}"
        args.pheno = os.path.join(base_pheno_path, f"{i}.phen")  
        args.seed = i
        args.output = f"output_{i}"  
        args.multiprocessing = True
        main(args)
