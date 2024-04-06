import os
import configparser
import subprocess
import argparse

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
        config = configparser.ConfigParser()
        config.read(config_file_path)
        
        config_name = "PyRHE_Config" if not args.use_original else "Original_RHE_Config"

        config[config_name]["pheno"] = os.path.join(phenotype_files_dir, f"{phenotype}.pheno")
        config[config_name]["covariate"] = os.path.join(phenotype_files_dir, f"{phenotype}.covar")
        config[config_name]["output"] = f"{phenotype}"
        config[config_name]["num_bin"] = str(args.num_bin)
        if args.num_bin == 1:
            config[config_name]["annot"] = "/u/scratch/z/zhengton/data/single.annot.txt"
        else:
            config[config_name]["annot"] = "/u/scratch/z/zhengton/data/2maf_4ld_array_annot.txt"


        with open(config_file_path, "w") as config_file:
            config.write(config_file)

        if args.use_original:
            subprocess.run(["python", "./run_original.py", "--config", config_file_path])
        else:
            subprocess.run(["python", "./run_rhe.py", "--config", config_file_path])



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyRHE') 
    parser.add_argument('--num_bin', '-b', type=int, default=1, help='Number of bins')
    parser.add_argument('--use_original', action='store_true')
   
    args = parser.parse_args()

    main(args)
