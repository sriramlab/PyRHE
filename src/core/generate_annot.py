"""
Generate annotation file
"""
import sys
from constant import DATA_DIR, HOME_DIR
sys.path.insert(0, HOME_DIR)
import os
import random
import argparse
from bed_reader import open_bed


def create_annot_file(geno_file, num_bin, filename):
    random.seed(0)
    try:
        bed = open_bed(geno_file + ".bed")
        geno = bed.read()
    except Exception as e:
        raise Exception(f"Error occurred: {e}")

    num_snp = geno.shape[1]

    with open(filename, 'w') as f:
        for _ in range(num_snp):
            row = [0] * num_bin
            random_col = random.randint(0, num_bin - 1)
            row[random_col] = 1
            f.write(' '.join(str(val) for val in row) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate annot')
    parser.add_argument('--geno', '-g', type=str, default=None, help='path of the genotype file')
    parser.add_argument('--num_bin', '-b', type=int, default=1, help='number of bins')
    parser.add_argument('--output_dir', '-o', type=str, default=f'{DATA_DIR}/annot', help='directory to store the generated annot')

    args = parser.parse_args()

    filename = f'{args.output_dir}/annot_{args.num_bin}'
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    create_annot_file(args.geno, args.num_bin, filename)