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
        print("Error reading file", filename)
        exit(1)

    Nsnp = linenum
    print("#SNP in bim file", Nsnp)


def count_fam(filename):
    with open(filename, 'r') as file:
        i = sum(1 for line in file)
    return i


def read_annot(filename, Njack):
    Nbin = 0
    annot_bool = []
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
    
    annot_bool = np.array(annot_bool)

    Nsnp = linenum + 1

    print("Number of SNPs per block:", Nsnp // Njack)

    for i, num_snps in enumerate(len_bins):
        print(num_snps, "SNPs in", i, "-th bin")

    step_size = Nsnp // Njack
    jack_bin = np.zeros((Njack, Nbin), dtype=int)

    for i, snp_row in enumerate(annot_bool):
        for j, val in enumerate(snp_row):
            if val == 1:
                temp = i // step_size
                if temp >= Njack:
                    temp = Njack - 1
                jack_bin[temp][j] += 1
                
    return annot_bool, jack_bin


def count_pheno(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    Nindv = len(lines) - 1

    return Nindv
