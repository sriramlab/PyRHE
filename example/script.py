def duplicate_pheno_values(input_filename, output_filename):
    try:
        with open(input_filename, 'r') as infile:
            lines = infile.readlines()

        with open(output_filename, 'w') as outfile:
            header = lines[0].strip().split()
            new_header = header + [header[-1] + "_dup"]
            outfile.write(' '.join(new_header) + '\n')

            for line in lines[1:]:
                parts = line.strip().split()
                new_line = parts + [parts[-1]]  
                outfile.write(' '.join(new_line) + '\n')

    except FileNotFoundError:
        print("Error: The input file could not be found.")
    except Exception as e:
        print(f"An error occurred: {e}")

duplicate_pheno_values('/u/home/j/jiayini/project-sriram/PyRHE/example/test.pheno', '/u/home/j/jiayini/project-sriram/PyRHE/example/test.pheno.multi')