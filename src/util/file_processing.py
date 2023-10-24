import numpy as np

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
        
        N = len(lines)
        y = np.zeros((N, 1))

        for i, line in enumerate(lines):
            columns = line.strip().split()
            y[i] = float(columns[2])
        
        return y
    
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


def read_cov(std, Nind, filename, covname=""):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    header = lines[0].split()
    cov_indices = [i for i, b in enumerate(header) if b not in ["FID", "IID"]]
    
    covNum = len(cov_indices)
    if covname:
        covIndex = header.index(covname)
    else:
        covIndex = 0
    
    cov_sum = np.zeros(covNum)
    if covname == "":
        covariate = np.zeros((Nind, covNum))
        print(f"Read in {covNum} Covariates..")
    else:
        covariate = np.zeros((Nind, 1))
        print(f"Read in covariate {covname}")
    
    missing = [[] for _ in range(covNum)]
    for j, line in enumerate(lines[1:]):
        data = line.split()
        for k, index in enumerate(cov_indices):
            temp = data[index]
            if temp == "NA" or float(temp) == -9:
                missing[k].append(j)
                continue
            cur = float(temp)
            cov_sum[k] += cur
            if covname == "":
                covariate[j][k] = cur
            elif k == covIndex:
                covariate[j][0] = cur

    # compute cov mean and impute
    for a in range(covNum):
        missing_num = len(missing[a])
        cov_sum[a] /= (Nind - missing_num)
        
        for b in missing[a]:
            if covname == "":
                covariate[b][a] = cov_sum[a]
            elif a == covIndex:
                covariate[b][0] = cov_sum[a]

    if std:
        cov_std = np.zeros(covNum)
        sum_ = np.sum(covariate, axis=0)
        sum2 = np.sum(covariate**2, axis=0)
        for b in range(covNum):
            cov_std[b] = sum2[b] + Nind * cov_sum[b]**2 - 2 * cov_sum[b] * sum_[b]
            cov_std[b] = np.sqrt((Nind - 1) / cov_std[b])
            scalar = cov_std[b]
            for j in range(Nind):
                covariate[j][b] -= cov_sum[b]
                covariate[j][b] *= scalar

    return covariate, covNum