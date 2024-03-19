import sys
import numpy as np
sys.path.insert(0, '/u/home/j/jiayini/project-sriram/RHE_project')
import numpy as np
from bed_reader import open_bed

def generate_covariate_file(geno_file):
    bed = open_bed(geno_file + ".bed")
    geno = bed.read()
    num_individuals = geno.shape[0]
    num_covariates = 4

    covariate_data = np.random.randint(0, 2, size=(num_individuals, num_covariates))

    fid = np.arange(1, num_individuals + 1)
    iid = np.ones(num_individuals)

    combined_data = np.column_stack((fid, iid, covariate_data))

    with open("small_covariate_file.cov", 'w') as file:
        file.write("FID IID X1 X2 X3 X4\n")
        for row in combined_data:
            file.write(" ".join(map(str, row.astype(int))) + "\n")

generate_covariate_file("/u/home/j/jiayini/project-sriram/RHE_project/data/simple/actual_geno_1")
