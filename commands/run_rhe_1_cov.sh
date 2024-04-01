#!/bin/bash
geno_path="/u/home/j/jiayini/data/200k_allsnps"
pheno_dir_path="/u/home/j/jiayini/project-sriram/PyRHE/data_200k/pheno_with_cov/bin_1"
covariate_path="/home/jiayini1119/data/200k.covar"
num_vec=10
num_bin=1
num_block=100
streaming=0
num_replica=25


for ((i=0; i<num_replica; i++))
do
  pheno_path="${pheno_dir_path}/${i}.phen"
    output="output_${i}"
  
  if [ "${streaming}" -eq 1 ]; then
    cmd="python /u/home/j/jiayini/project-sriram/PyRHE/run_rhe.py --streaming"
  else
    cmd="python /u/home/j/jiayini/project-sriram/PyRHE/run_rhe.py"
  fi
  
  cmd+=" --geno ${geno_path} --pheno ${pheno_path} -c ${covariate_path} -k ${num_vec} -b ${num_bin} -jn ${num_block} --output ${output}"
  
  echo "Executing: $cmd"
  eval $cmd
done
