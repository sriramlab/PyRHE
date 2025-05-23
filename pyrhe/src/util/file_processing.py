import numpy as np
import pandas as pd
from pyrhe.src.util.types import *


def read_bim(filename):
    try:
        with open(filename, 'r') as inp:
            linenum = 0
            for line in inp:
                linenum += 1
                if line.startswith('#'):
                    continue

    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The bim file {filename} could not be found.")
    except IOError:
        raise
    except Exception as e:
        raise e

    num_snp = linenum
    return num_snp

def read_fam(filename):
    try:
        df = pd.read_csv(filename, sep='\s+', header=None)
        num_individuals = df.shape[0]
        return num_individuals, df
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The fam file '{filename}' could not be found.")
    except IOError:
        raise
    except Exception as e:
        raise e
    
def read_annot(filename, Njack):
    Nbin = 0
    annot_bool = []
    try: 
        with open(filename, 'r') as file:
            for linenum, line in enumerate(file):
                if line.startswith('#'):
                    continue

                tokens = list(map(int, line.strip().split(' ')))

                if linenum == 0:
                    Nbin = len(tokens)
                    len_bin = np.zeros(Nbin, dtype=int)

                for i, val in enumerate(tokens):
                    if val == 1:
                        len_bin[i] += 1

                annot_bool.append(tokens)
        
    except FileNotFoundError:
        raise FileNotFoundError("Error: The annotation file could not be found.")
    except IOError:
        raise
    except Exception as e:
        raise e
    
    annot_bool = np.array(annot_bool)

    Nsnp = linenum + 1

    return Nbin, annot_bool, len_bin


def read_pheno(filename):
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()

        header = lines[0].strip().split() # Number of phenotypes
        num_phenotypes = len(header) - 2
        lines = lines[1:]  # Exclude header

        y = [] 
        missing_indv = []

        all_binary = True
        valid_values = {0, 1, 2}

        for i, line in enumerate(lines):
            columns = line.strip().split()
            if "NA" in columns[2:] or -9 in [float(val) if val != "NA" else -9 for val in columns[2:]]:
                phenotypes = [-9] * num_phenotypes
                missing_indv.append(i)
            else:
                phenotypes = [float(val) for val in columns[2:]]
                if not all(p in valid_values for p in phenotypes):
                    all_binary = False

            y.append(phenotypes)

        y = np.array(y)
        return y, missing_indv, all_binary

    except FileNotFoundError:
        raise FileNotFoundError("Error: The pheno file could not be found.")
    except IOError:
        raise
    except Exception as e:
        raise

def generate_annot(filename, num_snp, num_bin):
    try:
        with open(filename, 'w') as f:
            for _ in range(num_snp):
                row = [0] * num_bin
                random_col = np.random.randint(0, num_bin)
                row[random_col] = 1
                f.write(' '.join(str(val) for val in row) + '\n')
    except Exception as e:
        raise 


def read_cov(
        filename, 
        std: bool=False, 
        missing_indvs: list = None, 
        cov_impute_method: str = "ignore", 
        one_hot_conversion: bool = False,
        categorical_threshold: int = 100,
        logger = None
        ):
    try: 
        df = pd.read_csv(filename, sep='\s+')
        
        if missing_indvs is None:
            missing_indvs = []

        if missing_indvs:
            df = df.drop(index=missing_indvs, errors='ignore')

        if 'FID' in df.columns:
            df.drop('FID', axis=1, inplace=True)
        if 'IID' in df.columns:
            df.drop('IID', axis=1, inplace=True)
        
        is_missing = df.replace('NA', np.nan).isin([np.nan, -9]).any(axis=1)
        newly_missing_indvs = df.index[is_missing].tolist()

        if cov_impute_method == "ignore":
            df = df[~is_missing]

        else:  # mean imputation
            df = df.replace({'NA': np.nan, '-9': np.nan})
            for column in df.columns:
                mean_val = df[column].mean()
                df[column].fillna(mean_val, inplace=True)
        
        all_missing_indvs = missing_indvs + newly_missing_indvs

        # for one-hot encoding
        df_one_hot = df.copy()

        # detect categorical variables and handle one-hot encoding
        for column in df.columns:
            num_unique_values = df[column].nunique()
            if num_unique_values <= categorical_threshold:
                if one_hot_conversion:
                    if logger:
                        logger._debug(f"Column '{column}' detected as categorical with {num_unique_values} unique values.")  
                    one_hot = pd.get_dummies(df[column], prefix=column, drop_first=False)

                    one_hot = one_hot.astype(int)

                    # save to a separate one-hot cov file
                    file_name = f"{column}_one_hot.cov"
                    one_hot.to_csv(file_name, index=False, sep=' ', header=False)
                    
                    if logger:
                        logger._debug(f"One-hot encoded values for '{column}' stored in '{file_name}'")

                    # drop the original column from the one-hot DataFrame
                    df_one_hot = df_one_hot.drop(column, axis=1).join(one_hot)
            else:
                if logger:
                    logger._debug(
                        f"Column '{column}' contains quantitative values "
                        f"(number of unique values is {num_unique_values} while categorical threshold is {categorical_threshold})"
                    )

        if std:
            df = (df - df.mean()) / df.std(ddof=1)
        
        if logger:
            logger._debug("Convert categorical variables in the covariate file into one-hot encoding")
            logger._debug(f"The maximum number of distinct values that should be considered categorical rather than quantitative: {categorical_threshold}")
        
        return df.values, all_missing_indvs

    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The covariate file '{filename}' could not be found.")
    except IOError as e:
        raise e
    except Exception as e:
        raise e

    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The covariate file '{filename}' could not be found.")
    except IOError:
        raise
    except Exception as e:
        raise e


def read_env_file(file_path):
    try:
        df = pd.read_csv(file_path, sep='\s+')

        num_env = len(df.columns) - 2  

        env_vector = df['env'].to_numpy()
    
        return num_env, env_vector
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file '{file_path}' could not be found.")
    except IOError:
        raise
    except Exception as e:
        raise e
