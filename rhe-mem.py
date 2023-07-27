import argparse
from RHE.util.file_processing import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process genotype, phenotype, covariate, and annotation files.')

    parser.add_argument('-g', '--genotype', type=str, required=True,
                        help='The path of genotype file.')
    parser.add_argument('-p', '--phenotype', type=str, required=True,
                        help='The path of phenotype file.')
    parser.add_argument('-c', '--covariate', type=str, required=False,
                        help='The path of covariate file.')
    parser.add_argument('-annot', '--annotation', type=str, required=True,
                        help='The path of annotation file.')
    parser.add_argument('-k', '--num_vec', type=int, default=10,
                        help='The number of random vectors.')
    parser.add_argument('-jn', '--num_jack', type=int, default=100,
                        help='The number of jackknife blocks.')
    parser.add_argument('-o', '--out_put', type=str, required=False,
                        help='The path of the output file.')

    args = parser.parse_args()

    genotype_bed_file = args.genotype + ".bed"
    genotype_bim_file = args.genotype + ".bim"
    genotype_fam_file = args.genotype + ".fam"

    phenotype_file = args.phenotype
    covariate_file = args.covariate
    annotation_file = args.annotation
    num_vectors = args.num_vec
    num_jacks = args.num_jack
    output_file = args.out_put

    # file reading

    read_bim(genotype_bim_file)

    annot_bool, jack_bin = read_annot(annotation_file, num_jacks)

    # TODO: read cov

    Nind = count_pheno(phenotype_file)

    if (count_fam(genotype_fam_file) != Nind):
        raise ValueError("# indvs in fam file and pheno file does not match")

    # genotype stream pass