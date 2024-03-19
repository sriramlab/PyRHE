import numpy as np
import pandas as pd

def read_bim(filename):
    try:
        with open(filename, 'r') as inp:
            linenum = 0
            for line in inp:
                linenum += 1
                if line.startswith('#'):
                    continue

    except FileNotFoundError:
        raise FileNotFoundError("Error: The bim file could not be found.")
    except IOError:
        raise
    except Exception as e:
        raise 

    num_snp = linenum
    return num_snp

def read_fam(filename):
    try: 
        with open(filename, 'r') as file:
            i = sum(1 for line in file)
        return i
    except FileNotFoundError:
        raise FileNotFoundError("Error: The fam file could not be found.")
    except IOError:
        raise
    except Exception as e:
        raise 

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
        raise FileNotFoundError("Error: The annot file could not be found.")
    except IOError:
        raise
    except Exception as e:
        raise 
    
    annot_bool = np.array(annot_bool)

    Nsnp = linenum + 1

    print("Number of SNPs per block:", Nsnp // Njack)

    for i, num_snps in enumerate(len_bin):
        print(num_snps, "SNPs in", i, "-th bin")
        
    return Nbin, annot_bool, len_bin


def read_pheno(filename):
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
        
        lines = lines[1:]
        
        y = [] 
        missing_indv = []

        for i, line in enumerate(lines):
            columns = line.strip().split()
            if columns == "NA" or float(columns[2]) == -9:
                missing_indv.append(i)
            else:
                y.append(float(columns[2]))
        
        y = np.array(y).reshape(-1, 1)
        return y, missing_indv
    
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


def read_cov(filename, covname="", std: bool=False):
    df = pd.read_csv(filename, delim_whitespace=True)

    if 'FID' in df.columns:
        df.drop('FID', axis=1, inplace=True)
    if 'IID' in df.columns:
        df.drop('IID', axis=1, inplace=True)

    if covname:
        df = df[[covname]]

    df.replace(['NA', -9], np.nan, inplace=True)

    df.fillna(df.mean(), inplace=True)

    if std:
        df = (df - df.mean()) / df.std(ddof=1)

    return df.values
