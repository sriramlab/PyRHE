qsub -cwd -V -N DG -l highp,gpu,h_data=5G,h_rt=05:00:00 -t 1-25 -pe shared 5 runrhe.sh