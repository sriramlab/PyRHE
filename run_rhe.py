import argparse
import os
import numpy as np
from pyrhe.src.core import RHE, StreamingRHE
from constant import DATA_DIR, RESULT_DIR
import json
import time
import configparser


def parse_config(config_path, config_name):
    config = configparser.ConfigParser()
    config.read(config_path)
    return dict(config.items(config_name))

def convert_to_correct_type(value, default):
    if value.lower() == 'none':
        return None
    elif isinstance(default, bool):
        return value.lower() in ['true', '1', 't', 'y', 'yes']
    elif isinstance(default, int):
        return int(value)
    else:
        return value

def main(args):

    print(args)
    pheno_file = args.pheno
    annot_path = f"{DATA_DIR}/annot/annot_{args.num_bin}" if args.annot is None else args.annot

    if args.num_workers <= 1:
        args.multiprocessing = False
    else:
        args.multiprocessing = True
    
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
            device=args.device,
            cuda_num =args.cuda_num,
            multiprocessing=args.multiprocessing,
            num_workers=args.num_workers,
            seed=args.seed,
            trace_dir=args.get_trace,
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
            device=args.device,
            cuda_num =args.cuda_num,
            multiprocessing=args.multiprocessing,
            num_workers=args.num_workers,
            seed=args.seed,
            get_trace=args.get_trace,
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

    if not args.benchmark_runtime:
        use_cov = "cov" if args.covariate is not None else "no_cov"
        result_dir = f"{RESULT_DIR}/pyrhe_output/{use_cov}/bin_{args.num_bin}"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        
        output_file_path = os.path.join(result_dir, f"{args.output}.json")

        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
    
    else:
        return runtime
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyRHE') 
    parser.add_argument('--streaming', action='store_true', help='use streaming version')
    parser.add_argument('--get_trace', action='store_true', help='get the trace estimate')
    parser.add_argument('--benchmark_runtime', action='store_true', help='benchmark the runtime')

    parser.add_argument('--geno', '-g', type=str, help='genotype file path')
    parser.add_argument('--pheno', '-p', type=str, default=None, help='phenotype file path')
    parser.add_argument('--covariate', '-c', type=str, default=None, help='Covariate file path')
    parser.add_argument('--annot', type=str, default=None, help='Annotation file path')
    parser.add_argument('--num_vec', '-k', type=int, default=10, help='The number of random vectors (10 is recommended).')
    parser.add_argument('--num_bin', '-b', type=int, default=8, help='Number of bins')
    parser.add_argument('--num_workers', type=int, default=5, help='Number of workers')
    parser.add_argument('--num_block', '-jn', type=int, default=100, help='The number of jackknife blocks. (100 is recommended). The higher number of jackknife blocks the higher the memory usage.')
    parser.add_argument('--seed', default=None, help='Random seed')
    parser.add_argument('--device', type=str, default="cpu", help="device to use")
    parser.add_argument('--cuda_num', type=int, default=None, help='cuda number')
    parser.add_argument("--output", '-o', type=str, default="test", help='output of the file')
    parser.add_argument('--config', type=str, help='Configuration file path')

    args = parser.parse_args()

    if args.config:
        config_args = parse_config(args.config, 'PyRHE_Config')
        for key, default in vars(args).items():
            if key in config_args:
                setattr(args, key, convert_to_correct_type(config_args[key], default))

    if args.benchmark_runtime:
        runtimes = [] 
        for i in range(3):
            args = parser.parse_args()
            if args.covariate is not None:
                cov = "_with_cov"
            else:
                cov = ""
            base_pheno_path = f"{args.pheno}/pheno{cov}/bin_{args.num_bin}"
            args.pheno = os.path.join(base_pheno_path, f"{i}.phen")  
            runtime = main(args) 
            runtimes.append(runtime)

        mean_runtime = np.mean(runtimes)
        std_runtime = np.std(runtimes)
        print(f"runtime: {mean_runtime:.2f} ± {std_runtime:.2f} seconds")
    else:
        main(args)