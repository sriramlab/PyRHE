#!/bin/bash
i=`expr $SGE_TASK_ID - 1`

rhe_path=/u/home/j/jiayini/project-sriram/RHE_project/run_rhe.py
gen=/u/scratch/b/bronsonj/geno/25k_allsnps
phen=/u/home/j/jiayini/project-sriram/RHE_project/data/pheno_cov/bin_1
cov=/u/home/j/jiayini/project-sriram/RHE_project/data/cov_25k.cov
output=/u/home/j/jiayini/project-sriram/RHE_project/results/pyrhe_output/cov/bin_1

python $rhe_path -g $gen -p $phen/${i}.phen -k 10 -jn 100 -o $output/result_${i}.txt