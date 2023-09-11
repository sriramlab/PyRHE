from RHE.core.RHE import RHE
geno_path="/Users/nijiayi/RHE_project/data/test2/actual_geno_1"
rhe = RHE(
    geno_file=geno_path,
)

print("Simulating Phenotype...")

y, _ = rhe.simulate_pheno(sigma_list=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

rhe()
