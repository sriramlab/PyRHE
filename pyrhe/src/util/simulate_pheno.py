"""
Add covariate component to the simulated phenotype
"""
import argparse
import sys
import os
import numpy as np
sys.path.insert(0, '/home/jiayini1119/RHE_project')
from src.util.file_processing import read_pheno, read_cov
from constant import DATA_DIR

def simulate_pheno(pheno_file, cov_file, output_file):
    with open(pheno_file, 'r') as file:
        lines = file.readlines()
    header = lines[0]
    data = [line.strip().split() for line in lines[1:]]

    y = read_pheno(pheno_file)
    print(y.shape)
    cov = read_cov(std=True, filename=cov_file)
    print(cov.shape[0])
    print(cov.shape[1])
    Ncov = cov.shape[1]
    y_new = y + cov @ np.ones((Ncov, 1))

    with open(output_file, 'w') as file:
        file.write(header)
        for i, line in enumerate(data):
            file.write(f"{line[0]} {line[1]} {y_new[i][0]:.6f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate phenotype data with covariates.')
    parser.add_argument('--num_bins', '-b', type=int, default=1, help='Number of bins to process')
    parser.add_argument('--covariate', '-c', type=str, help='covariate file path')

    args = parser.parse_args()

    pheno_dir = f'{DATA_DIR}/pheno/bin_{args.num_bins}'
    output_dir = f'{DATA_DIR}pheno_with_cov/bin_{args.num_bins}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    covariate_file = args.covariate

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(pheno_dir):
        if filename.endswith('.phen'):
            original_pheno_file = os.path.join(pheno_dir, filename)
            new_pheno_file = os.path.join(output_dir, filename)
            simulate_pheno(original_pheno_file, covariate_file, new_pheno_file)