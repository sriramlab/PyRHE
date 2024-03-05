
import sys
sys.path.insert(0, '/home/jiayini1119/RHE_project')
from src.core.rhe import RHE
from src.core.streaming_rhe import StreamingRHE
geno_path="/home/jiayini1119/RHE_project/data_25k/simple/actual_geno_1"

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

rhe = StreamingRHE(
    geno_file=geno_path,
    annot_file='/home/jiayini1119/RHE_project/data_25k/simple/annot.txt',
    cov_file='/home/jiayini1119/RHE_project/data_25k/simple/small_covariate_file.cov',
    num_bin=1,
    device="cpu",
    num_jack=8,
    num_workers=2,
    get_trace=True,
)

print("Simulating Phenotype...")

y, _ = rhe.simulate_pheno(sigma_list=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])





print(y)

rhe()