import sys
import os
import numpy as np
sys.path.insert(0, '/u/home/j/jiayini/project-sriram/RHE_project')
from src.util.file_processing import read_pheno, read_cov

def simulate_pheno(pheno_file, cov_file, output_file):
    with open(pheno_file, 'r') as file:
        lines = file.readlines()
    header = lines[0]
    data = [line.strip().split() for line in lines[1:]]

    y = read_pheno(pheno_file)
    cov = read_cov(std=True, filename=cov_file)
    print(cov.shape[1])
    Ncov = cov.shape[1]
    y_new = y + cov @ np.ones((Ncov, 1))

    with open(output_file, 'w') as file:
        file.write(header)
        for i, line in enumerate(data):
            file.write(f"{line[0]} {line[1]} {y_new[i][0]:.6f}\n")


if __name__ == '__main__':
    pheno_dir = '/u/home/j/jiayini/project-sriram/RHE_project/data/pheno/bin_1'
    output_dir = '/u/home/j/jiayini/project-sriram/RHE_project/data/pheno_with_cov/bin_1'
    covariate_file = '/u/home/j/jiayini/project-sriram/RHE_project/data/cov_25k.cov'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(pheno_dir):
        if filename.endswith('.phen'):
            original_pheno_file = os.path.join(pheno_dir, filename)
            new_pheno_file = os.path.join(output_dir, filename)
            simulate_pheno(original_pheno_file, covariate_file, new_pheno_file)
