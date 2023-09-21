from RHE.core.rhe import RHE
geno_path="/Users/nijiayi/RHE_project/data/test2/actual_geno_1"

# import numpy as np
# from RHE.core.rhe import RHE

# X = np.array([
#         [0, 1, 1, 2, 1, 2, 1],
#         [2, 1, 1, 0, 2, 1, 2],
#         [1, 2, 2, 1, 1, 2, 1],
#         [1, 0, 1, 0, 1, 0, 1],
#         [2, 1, 0, 1, 1, 2, 0],
#     ])


# from bed_reader import to_bed
# plink_bed_file = "test_geno.bed"
    
# to_bed(plink_bed_file, X)

# geno_path="test_geno"
rhe = RHE(
    geno_file=geno_path,
    num_bin=8
)

# assert rhe.geno.shape[0] == X.shape[0]
# assert rhe.geno.shape[1] == X.shape[1]
# assert rhe.pheno is None


print("Simulating Phenotype...")

y, _ = rhe.simulate_pheno(sigma_list=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
print(y)

rhe()