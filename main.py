import argparse
import glob
import os
import pickle
from src.core.rhe import RHE
from constant import RESULT_DIR, DATA_DIR

def main(args):

    all_results = {}

    if args.test_mult_files:
        pheno_files = glob.glob(f"{DATA_DIR}/pheno/bin_{args.num_bin}/*")
    else:
        pheno_files = [args.pheno]

    annot_path = f"{DATA_DIR}/annot/annot_{args.num_bin}"

    
    for pheno_file in pheno_files:
        rhe = RHE(
            geno_file=args.geno,
            annot_file=annot_path,
            pheno_file=pheno_file,
            num_jack=args.num_block,
            num_bin=args.num_bin,
            num_random_vec=args.num_vec,
            streaming=not args.streaming
        )

        # RHE
        sigma_ests_total, sig_errs, h2_total, h2_errs, enrichment_total, enrichment_errs = rhe()
        result = {
            "sigma_ests_total": sigma_ests_total,
            "sig_errs": sig_errs,
            "h2_total": h2_total,
            "h2_errs": h2_errs,
            "enrichment_total": enrichment_total,
            "enrichment_errs": enrichment_errs,
        }

        all_results[pheno_file] = result

    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    output = f"{RESULT_DIR}/{args.output}_{args.num_bin}.pkl"
    with open(output, 'wb') as output_file:
        pickle.dump(all_results, output_file)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyRHE') 
    parser.add_argument('--no-streaming', action='store_false', dest='streaming')
    parser.add_argument('--geno', '-g', type=str, default="/u/scratch/b/bronsonj/geno/25k_allsnps", help='genotype file path')
    parser.add_argument('--pheno', '-p', type=str, default=None, help='phenotype file path')
    parser.add_argument('--covariate', '-c', type=str, default=None, help='covariance file path')
    parser.add_argument('--num_vec', '-k', type=int, default=10, help='The number of random vectors (10 is recommended).')
    parser.add_argument('--num_bin', '-b', type=int, default=8, help='Number of bins')
    parser.add_argument('--num_block', '-jn', type=int, default=100, help='The number of jackknife blocks. (100 is recommended). The higher number of jackknife blocks the higher the memory usage.')
    parser.add_argument("--test_mult_files", action='store_true', help='Whether to do estimation for multiple phenotype files')
    parser.add_argument("--output", type=str, default="python_rhe", help='output of the file')
    
    args = parser.parse_args()

    main(args)
