
import numpy as np
import os
from bed_reader import open_bed
from src.util import *
from constant import DATA_DIR

def read_pheno_reference(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [line.split()[:2] for line in lines]

def write_pheno_file(fid_iid, y_values, file_path):
    with open(file_path, 'w') as file:
        for (fid, iid), y in zip(fid_iid, y_values):
            file.write(f"{fid} {iid} {y[0]}\n")

def simulate(geno_path, num_bin, sigma_g, cov_path=None, num_samples=25):
    reference_path = "/u/home/j/jiayini/project-sriram/RHE_project/data/ref.phen"
    fid_iid = read_pheno_reference(reference_path)
    bed = open_bed(geno_path + ".bed")
    ori_geno = bed.read()
    N = ori_geno.shape[0]
    # geno = impute_geno(geno, simulate_geno=True)

    sigma_e = 1 - np.sum(sigma_g)
    Nbin = len(sigma_g)

    annot_path = f"{DATA_DIR}/annot/annot_{num_bin}"

    _, annot, M_list = read_annot(annot_path, Njack=100)

    for i in range(num_samples):
        print(f"Processing {i}")
        np.random.seed(i)

        noise = np.random.normal(0, np.sqrt(sigma_e), (N, 1))

        betas = [np.random.normal(0, np.sqrt(sigma_g[n]/M_list[n]), (M_list[n], 1)) for n in range(Nbin)]

        y = noise
        for n in range(Nbin):
            geno = ori_geno[:, np.where(annot[:, n] == 1)[0]]
            geno = impute_geno(geno, simulate_geno=True)
            y += geno @ betas[n]

        if cov_path is not None:
            cov = read_cov(cov_path)
            y += cov @ np.full((cov.shape[1], 1), 0.05)

        if cov_path is not None:
            fn = "_with_cov"
        else:
            fn = ""
        output_dir = f"/u/home/j/jiayini/project-sriram/RHE_project/data/pheno{fn}/bin_{num_bin}" if cov_path else f"/u/home/j/jiayini/project-sriram/RHE_project/data/pheno/bin_{num_bin}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_file = os.path.join(output_dir, f"{i}.phen")
        write_pheno_file(fid_iid, y, output_file)

if __name__ == '__main__':
    geno_path = "/u/scratch/b/bronsonj/geno/25k_allsnps"
    num_bin = 1
    sigma_g = [0.2]
    cov_path = "/u/home/j/jiayini/project-sriram/RHE_project/data/cov_25k.cov"


    simulate(geno_path=geno_path, num_bin=num_bin, sigma_g=sigma_g, cov_path=cov_path, num_samples=25)