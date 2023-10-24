# RHE_project

**basic testing pipeline:**

1. **Set Up**

Create a .env file and specify the RESULT_DIR (where you store the results) and DATA_DIR (store the simulated phenotype, generated annotation file, etc.)

2. **Generate Annotation File:**  
```
cd core
python generate_annot.py -g {geno_path} -b {num_bin} -o {output_file}
```

3. **Simulate Phenotype:**  

Use the [Simulator](https://github.com/sriramlab/Simulator).

4. **Run original RHE**  
```
cd ..
python run_original.py -g {geno_path} -b {num_bin} -k {num_vec} -jn {num_block} --output {output_file}
```

5. **Run python RHE**
```
cd ..
python main.py -g {geno_path} -b {num_bin} -k {num_vec} -jn {num_block} --output {output_file}
```

6. **Visualize**
run cells in `plotting.ipynb`.


