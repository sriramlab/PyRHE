import argparse
import numpy as np
from typing import List
from bed_reader import open_bed
from RHE.util.file_processing import *
from RHE.core.RHE import *
from RHE.core.RHE_mem import *


def main(args):

    np.random.seed(args.seed)

    ##############################################################
    # File Reading
    ##############################################################

    # read bed file
    try:
        bed = open_bed(args.genotype + ".bed")
        X = bed.read()
    except FileNotFoundError:
        print("Error: The .bed file could not be found.")
    except IOError:
        print("Error: An IO error occurred while reading the .bed file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    num_ind = X[0].size()
    num_snp = X[1].size

    # read bim file
    assert num_snp == read_bim(args.genotype + ".bim")

    # read fam file
    assert num_ind == read_fam(args.genotype + ".fam")

    # read pheno file
    if args.pheno is not None:
        y = read_pheno(args.phenotype)
        assert num_ind == y[0]
    else:
        y = None

    # read annot file, generate one if not exist
    if args.annotation is None:
        if args.num_bin is None:
            raise ValueError("Must specify number of bins if annot file is not provided")

        annotation_file = "generated_annot"

        generate_annot(annotation_file, args.num_snp, args.num_bin)
    
    else:
        annotation_file = args.annotation
        
    num_bin, annot_matrix = read_annot(annotation_file, args.num_jack)

    ##############################################################
    # RHE
    ##############################################################

    rhe = RHE(
        geno=X,
        pheno=y,
        annot_file_name=annotation_file,
        annot_matrix=annot_matrix,
        num_bin=num_bin,
        num_random_vec=args.num_vec,
    )

    if y is None:
        y, _ = rhe.simulate_pheno()

    RHE()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process genotype, phenotype, covariate, and annotation files.')

    parser.add_argument('-g', '--genotype', type=str, required=True,
                        help='The path of genotype file.')
    parser.add_argument('-p', '--phenotype', type=str, required=False,
                        help='The path of phenotype file.')
    parser.add_argument('-c', '--covariate', type=str, required=False,
                        help='The path of covariate file.')
    parser.add_argument('-annot', '--annotation', type=str, default=None, required=False,
                        help='The path of annotation file.')
    parser.add_argument('--num_bin', type=int, default=None, required=False,
                        help='Only specify this if the annotation file is not provided')
    parser.add_argument('--sigma_list', type=List, default=None, required=False,
                        help='Only specify if you want to simulate phenotype file')
    parser.add_argument('-k', '--num_vec', type=int, default=10,
                        help='The number of random vectors.')
    parser.add_argument('-jn', '--num_jack', type=int, default=100,
                        help='The number of jackknife blocks.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for randomness.')

    args = parser.parse_args()
    main(args)