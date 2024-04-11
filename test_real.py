import os
import configparser
import subprocess
import argparse
import fcntl
from constant import RESULT_DIR

def main(args):
    phenotypes_to_test = [
        "basal_metabolic", "bmi", "blood_wbc_count", "alcohol_freq", "blood_mscv",
        "height", "urate", "creatinine", "fvc", "overall_health", "fev1_fvc",
        "testosterone", "blood_mch", "whr", "blood_lymphocyte", "alka_phos",
        "bp_systolic", "shbg", "blood_rbc_distrib_width", "ggt", "ala_at",
        "cystatin_c", "phosphate", "bp_diastolic", "blood_reticulocyte", "igf1",
        "asp_at", "hba1c", "urea", "potassium", "sodium_urine", "bmd_heel_tscore",
        "blood_monocyte", "c_reactive_prot", "cholesterol", "microalbumin",
        "blood_platelet_distrib_width", "creatinine_urine", "ldl", "albumin",
        "vitamin_d.R", "blood_platelet", "triglycerides", "blood_rbc_count",
        "blood_eosinophil", "calcium", "blood_mpv", "hdl", "apo_a", "glucose"
    ]
   

    config_file_path = "./config_real_original.txt" if args.use_original else "./config_real_pyrhe.txt"
    phenotype_files_dir = "/u/project/sriram/alipazok/Data/new_ukkb_phenotypes"

    for phenotype in phenotypes_to_test:

        cov_status = "no_cov" if args.no_cov else "cov"

        if args.use_original:
            result_file_path = os.path.join(RESULT_DIR, "original_result", cov_status, f"bin_{args.num_bin}", f"{phenotype}.txt")
        else:
            result_file_path = os.path.join(RESULT_DIR, "pyrhe_output", cov_status, f"bin_{args.num_bin}", f"{phenotype}.json")
        
        if os.path.exists(result_file_path):
            print(f"Skipping {phenotype} as result file already exists.")
            continue
            
        print(f"Processing {phenotype}...")

        with open(config_file_path, "r+") as config_file:
            fcntl.flock(config_file, fcntl.LOCK_EX)
            config = configparser.ConfigParser()
            config.read(config_file_path)
            config_name = "PyRHE_Config" if not args.use_original else "Original_RHE_Config"
            config.set(config_name, "pheno", os.path.join(phenotype_files_dir, f"{phenotype}.pheno"))
            cov = "None" if args.no_cov else os.path.join(phenotype_files_dir, f"{phenotype}.covar")
            config.set(config_name, "covariate", cov)
            config.set(config_name, "output", f"{phenotype}")
            config.set(config_name, "num_bin", str(args.num_bin))
            if args.num_bin == 1:
                config.set(config_name, "annot", "/u/scratch/z/zhengton/data/single.annot.txt")
            else:
                config.set(config_name, "annot", "/u/scratch/z/zhengton/data/2maf_4ld_array_annot.txt")
            config_file.seek(0) 
            config.write(config_file)  
            config_file.truncate()  

            fcntl.flock(config_file, fcntl.LOCK_UN)

        if args.use_original:
            subprocess.run(["python", "./run_original.py", "--config", config_file_path])
        else:
            subprocess.run(["python", "./run_rhe.py", "--config", config_file_path])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyRHE') 
    parser.add_argument('--num_bin', '-b', type=int, default=1, help='Number of bins')
    parser.add_argument('--use_original', action='store_true')
    parser.add_argument('--no_cov', action='store_true')
   
    args = parser.parse_args()

    main(args)
