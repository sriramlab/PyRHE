import argparse
import os
import json
import re

parser = argparse.ArgumentParser(description='parse output file for original RHE')
parser.add_argument('--folder_path', '-p', type=str)

args = parser.parse_args()

def parse_file_content(file_content):
    sigma_pattern = r"Sigma\^2_(\d+): (-?[\d.]+(?:e-?\d+)?) SE: (-?[\d.]+(?:e-?\d+)?)"
    sigma_matches = re.findall(sigma_pattern, file_content)

    h2_pattern = r"h\^2(?:_i/h\^2_t)? of bin (\d+) : (-?[\d.]+(?:e-?\d+)?) SE: (-?[\d.]+(?:e-?\d+)?)"
    h2_matches = re.findall(h2_pattern, file_content)

    enrichment_pattern = r"Enrichment of bin (\d+) : (-?[\d.]+(?:e-?\d+)?) SE: (-?[\d.]+(?:e-?\d+)?)"
    enrichment_matches = re.findall(enrichment_pattern, file_content)

    runtime_pattern = r"runtime: (-?[\d.]+(?:e-?\d+)?) seconds"
    runtime_match = re.search(runtime_pattern, file_content)

    h2_data = {}
    for match in h2_matches:
        bin_number = int(match[0])
        if bin_number not in h2_data:
            h2_data[bin_number] = {"bin": bin_number, "h2": float(match[1]), "SE": float(match[2])}

    data = {
        "sigma": [{"index": int(m[0]), "sigma": float(m[1]), "SE": float(m[2])} for m in sigma_matches],
        "h2": list(h2_data.values()),
        "enrichment": [{"bin": int(m[0]), "enrichment": float(m[1]), "SE": float(m[2])} for m in enrichment_matches],
        "runtime": float(runtime_match.group(1)) if runtime_match else None
    }

    return data

def process_folder(folder_path):
    all_data = {}
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    content = f.read()
                    file_data = parse_file_content(content)
                    all_data[file_path] = file_data

    return all_data

folder_path = args.folder_path
data = process_folder(folder_path)

output_file_path = os.path.join(folder_path, "summary.json")

with open(output_file_path, "w") as json_file:
    json.dump(data, json_file, indent=4)
