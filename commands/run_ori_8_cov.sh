#!/bin/bash
geno_path="/u/home/j/jiayini/data/200k_allsnps"
pheno_dir_path="/u/home/j/jiayini/project-sriram/RHE_project/data_200k/pheno_with_cov/bin_8"
covariate_path="/u/home/j/jiayini/data/200k.covar"
num_vec=10
num_bin=8
num_block=100
num_replica=25


for ((i=0; i<num_replica; i++))
do
  pheno_path="${pheno_dir_path}/${i}.phen"
    output="output_${i}"
  
  cmd="python run_original.py --geno ${geno_path} --pheno ${pheno_path} -c ${covariate_path} -k ${num_vec} -b ${num_bin} -jn ${num_block} --output ${output}"
  
  echo "Executing: $cmd"
  eval $cmd
done