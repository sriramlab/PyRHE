#!/bin/bash
i=`expr $SGE_TASK_ID - 1`

rhe_path=/u/home/j/jiayini/project-sriram/RHE_project/py_rhe.py
gen=/u/scratch/b/bronsonj/geno/25k_allsnps
phen=/u/home/j/jiayini/project-sriram/RHE_project/data/pheno/bin_1
cov=/u/home/j/jiayini/project-sriram/RHE_project/data/cov_25k.cov
output=/u/home/j/jiayini/project-sriram/RHE_project/results/pyrhe_output/bin_1

python $rhe_path -g $gen -p $phen/${i}.phen -k 100 -jn 1000 -o $output/result_${i}.txt