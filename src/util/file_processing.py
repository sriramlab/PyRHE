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
                    jack_bin = np.zeros((Njack, Nbin), dtype=int)
                    len_bins = np.zeros(Nbin, dtype=int)

                for i, val in enumerate(tokens):
                    if val == 1:
                        len_bins[i] += 1

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

    for i, num_snps in enumerate(len_bins):
        print(num_snps, "SNPs in", i, "-th bin")

    # step_size = Nsnp // Njack
    # jack_bin = np.zeros((Njack, Nbin), dtype=int)

    # for i, snp_row in enumerate(annot_bool):
    #     for j, val in enumerate(snp_row):
    #         if val == 1:
    #             temp = i // step_size
    #             if temp >= Njack:
    #                 temp = Njack - 1
    #             jack_bin[temp][j] += 1
                
    # return annot_bool, jack_bin
    return Nbin, annot_bool


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