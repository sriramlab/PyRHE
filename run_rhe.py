import argparse
import os
import numpy as np
from pyrhe.src.core import RHE, StreamingRHE, GENIE, StreamingGENIE, RHE_DOM, StreamingRHE_DOM
from pyrhe.src.util import Logger
from constant import DATA_DIR
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
        return value.lower() in ['true', '1','yes']
    elif isinstance(default, int):
        return int(value)
    else:
        return value

def main(args):

    log = Logger(output_file = args.output, suppress = args.suppress, debug_mode = args.debug)

    header = [
        "##################################",
        "#                                #",
        "#          PyRHE (v1.0.0)        #",
        "#                                #",
        "##################################"
    ]
    for line in header:
        log._log(line)

    log._log("\n")
    options = {
        "-g (genotype)": args.genotype,
        "-annot (annotation)": args.annotation,
        "-p (phenotype)": args.phenotype,
        "-c (covariates)": args.covariate,
        "-o (output)": args.output,
        "-k (# random vectors)": args.num_vec,
        "-jn (# jackknife blocks)": args.num_block,
        "--num_workers": args.num_workers,
        "--device": args.device,
        "--geno_impute_method": args.geno_impute_method,
        "--cov_impute_method": args.cov_impute_method,
    }

    log._log("Active essential options:")
    for flag, desc in options.items():
        log._log(f"\t{flag} {desc}")
    
    log._log("\n")
    log._debug(args)
    pheno_file = args.phenotype
    annot_path = f"{DATA_DIR}/annot/annot_{args.num_bin}" if args.annotation is None else args.annotation

    if args.num_workers <= 1:
        args.multiprocessing = False
    else:
        args.multiprocessing = True
    
    if (args.samp_prev is not None) != (args.pop_prev is not None):
        raise ValueError('Must set both or neither of --samp-prev and --pop-prev.')

    results = {}

    log._debug(f"processing {pheno_file}")


    params = {
        'model': args.model,
        'geno_file': args.genotype,
        'annot_file': annot_path,
        'pheno_file': pheno_file,
        'cov_file': args.covariate,
        'num_jack': args.num_block,
        'num_bin': args.num_bin,
        'num_random_vec': args.num_vec,
        'geno_impute_method': args.geno_impute_method,
        'cov_impute_method': args.cov_impute_method,
        'cov_one_hot_conversion': args.cov_one_hot_conversion,
        'categorical_threshhold': args.categorical_threshhold,
        'device': args.device,
        'cuda_num': args.cuda_num,
        'multiprocessing': args.multiprocessing,
        'num_workers': args.num_workers,
        'seed': args.seed,
        'get_trace': args.trace,
        'trace_dir': args.trace_dir,
        'samp_prev': args.samp_prev,
        'pop_prev': args.pop_prev,
        'log': log,
    }

    if args.model == "rhe":
        if args.streaming:
            rhe = StreamingRHE(**params)
        else:
            rhe = RHE(**params)
    
    elif args.model == "genie":
        params['env_file'] = args.env
        params['genie_model'] = args.genie_model
        if args.streaming:
            rhe = StreamingGENIE(**params)
        else:
            rhe = GENIE(**params)
    
    elif args.model == "rhe_dom":
        if args.streaming:
            rhe = StreamingRHE_DOM(**params)
        else:
            rhe = RHE_DOM(**params)
    else:
        raise ValueError("Unsupported Model")

    for trait in range(rhe.num_traits):
        start = time.time()
        res_dict = rhe(trait=trait)
        end = time.time()

        runtime = end - start

        results[f"Trait{trait}"] = {
            **res_dict,
            "runtime": runtime
        }


    log._log("Runtime: ", runtime)
    log._save_log()


    # if not args.benchmark_runtime:
    #     use_cov = "cov" if args.covariate is not None else "no_cov"
    #     result_dir = f"{RESULT_DIR}/pyrhe_output/{use_cov}/bin_{args.num_bin}"
    #     if not os.path.exists(result_dir):
    #         os.makedirs(result_dir)
        
    #     output_file_path = os.path.join(result_dir, f"{args.debug_output}.json")

    #     with open(output_file_path, 'w', encoding='utf-8') as f:
    #         json.dump(results, f, ensure_ascii=False, indent=4)
    
    # else:
    #     return runtime
    

if __name__ == '__main__':
    # TODO: use config file instead of argparse
    parser = argparse.ArgumentParser(description='PyRHE') 
    parser.add_argument('--model', type=str, default="rhe", choices=['rhe', 'genie', 'rhe_dom'])
    parser.add_argument('--genie_model', type=str, default="G+GxE+NxE", choices=['G', 'G+GxE', 'G+GxE+NxE'])
    parser.add_argument('--streaming', action='store_true', help='use streaming version')

    parser.add_argument('--trace', '-tr', action='store_true', help='get the trace estimate')
    parser.add_argument('--trace_dir', type=str, default="", help='directory to save the trace information')
    parser.add_argument('--benchmark_runtime', action='store_true', help='benchmark the runtime')

    parser.add_argument('--genotype', '-g', type=str, help='genotype file path')
    parser.add_argument('--phenotype', '-p', type=str, default=None, help='phenotype file path')
    parser.add_argument('--covariate', '-c', type=str, default=None, help='Covariate file path')
    parser.add_argument('--cov_one_hot_conversion', action='store_true', help='convert categorical variables in the covariate file into one-hot encoding')
    parser.add_argument('--categorical_threshhold', type=int, default=100, help='Maximum number of distinct values that should be considered categorical rather than quantitative')
    parser.add_argument('--env', '-e', type=str, default=None, help='Environment file path')
    parser.add_argument('--annotation', '-annot', type=str, default=None, help='Annotation file path')
    parser.add_argument('--num_vec', '-k', type=int, default=10, help='The number of random vectors (10 is recommended).')
    parser.add_argument('--num_bin', '-b', type=int, default=8, help='Number of bins')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--num_block', '-jn', type=int, default=100, help='The number of jackknife blocks. (100 is recommended). The higher number of jackknife blocks the higher the memory usage.')
    parser.add_argument('--seed', '-s', default=None, help='Random seed')
    parser.add_argument('--device', type=str, default="cpu", help="device to use")
    parser.add_argument('--cuda_num', type=int, default=None, help='cuda number')
    parser.add_argument("--output", '-o', type=str, default="test.out", help='output of the file')
    parser.add_argument('--geno_impute_method', type=str, default="binary", choices=['binary', 'mean'])
    parser.add_argument('--cov_impute_method', type=str, default="ignore", choices=['ignore', 'mean'])
    parser.add_argument('--samp_prev',default=None, help='Sample prevalence of binary phenotype (for conversion to liability scale).')
    parser.add_argument('--pop_prev',default=None, help='Population prevalence of binary phenotype (for conversion to liability scale).')
    parser.add_argument("--suppress", action="store_true", help="do not print out the outputs to stdout, log file only")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--debug_output", type=str, default="test", help='debug output of the file (for benchmarking)')


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
            base_pheno_path = f"{args.phenotype}/pheno{cov}/bin_{args.num_bin}"
            args.phenotype = os.path.join(base_pheno_path, f"{i}.phen")  
            runtime = main(args) 
            runtimes.append(runtime)

        mean_runtime = np.mean(runtimes)
        std_runtime = np.std(runtimes)
        print(f"runtime: {mean_runtime:.2f} ± {std_runtime:.2f} seconds")
    else:
        main(args)