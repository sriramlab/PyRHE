import argparse
import glob
import pickle
from src.core.rhe import RHE
from src.core.rhe_mem import StreamingRHE

def main(args):

    all_results = {}

    if args.test_mult_files:
        pheno_files = glob.glob(f"/u/home/j/jiayini/project-sriram/RHE_project/data/pheno/bin_{args.num_bin}/*")
    else:
        pheno_files = [args.pheno]

    
    for pheno_file in pheno_files:
        if args.method == "StreamingRHE":
            rhe = StreamingRHE(
                geno_file=args.geno,
                annot_file=args.annot,
                pheno_file=pheno_file,
                num_jack=args.num_block,
                num_random_vec=args.num_vec,
            )
        else:
            rhe = RHE(
                geno_file=args.geno,
                annot_file=args.annot,
                pheno_file=pheno_file,
                num_jack=args.num_block,
                num_random_vec=args.num_vec,
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

    if not args.test_mult_files:
        return all_results
    else:
        with open(args.output, 'wb') as output_file:
            pickle.dump(all_results, output_file)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Randomized Haseman-Elston regression for Multi-variance Components') 
    parser.add_argument('--method', type=str, default="StreamingRHE", choices=['StreamingRHE', 'RHE'], help='method to estimate')
    parser.add_argument('--geno', '-g', type=str, default="/u/scratch/b/bronsonj/geno/25k_allsnps", help='genotype file path')
    parser.add_argument('--pheno', '-p', type=str, default=None, help='phenotype file path')
    parser.add_argument('--covariate', '-c', type=str, default=None, help='covariance file path')
    parser.add_argument('--annot', type=str, default=None, help='annotation file path')
    parser.add_argument('--num_vec', '-k', type=int, default=10, help='The number of random vectors (10 is recommended).')
    parser.add_argument('--num_bin', '-b', type=int, default=8, help='Number of bins')
    parser.add_argument('--num_block', '-jn', type=int, default=100, help='The number of jackknife blocks. (100 is recommended). The higher number of jackknife blocks the higher the memory usage.')
    parser.add_argument("--test_mult_files", action='store_true', help='Whether to do estimation for multiple phenotype files')
    parser.add_argument("--output", type=str, default="python_rhe", help='output of the file')
    
    args = parser.parse_args()

    main(args)
