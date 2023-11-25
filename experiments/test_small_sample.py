
import sys
sys.path.insert(0, '/u/home/j/jiayini/project-sriram/RHE_project')
from src.core.rhe import RHE
geno_path="/u/home/j/jiayini/project-sriram/RHE_project/data/simple/actual_geno_1"

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

rhe = RHE(
    geno_file=geno_path,
    annot_file='/u/home/j/jiayini/project-sriram/RHE_project/data/simple/annot.txt',
    cov_file='/u/home/j/jiayini/project-sriram/RHE_project/data/simple/small_covariate_file.cov',
    num_bin=8,
    num_jack=2,
)

print("Simulating Phenotype...")

y, _ = rhe.simulate_pheno(sigma_list=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
print(y)

rhe()