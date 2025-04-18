import os
import sys
import subprocess
import re
import tempfile
import shutil
from pathlib import Path
import pytest

GROUND_TRUTH = {
    'sigma2_g': {
        'value': 0.123913,
        'se': 0.0204703
    },
    'sigma2_gxe': {
        'value': 0.270141,
        'se': 0.129187
    },
    'sigma2_nxe': {
        'value': 0.139967,
        'se': 0.128535
    },
    'sigma2_e': {
        'value': 0.715668,
        'se': 0.0205994
    },
    'h2_g': {
        'value': 0.12382,
        'se': 0.0204589
    },
    'h2_gxe': {
        'value': 0.106168,
        'se': 0.0507162
    },
    'h2_nxe': {
        'value': 0.0552029,
        'se': 0.0506941
    },
    'total_h2': {
        'value': 0.285191,
        'se': 0.0205747
    },
    'total_h2_g': {
        'value': 0.12382,
        'se': 0.0204589
    },
    'total_h2_gxe': {
        'value': 0.106168,
        'se': 0.0507162
    },
    'enrichment_g': {
        'value': 1.0,
        'se': 0.0
    }
}

def parse_output(output_file):
    """Parse the output file and extract the calculated values."""
    results = {}
    with open(output_file, 'r') as f:
        content = f.read()
        
        # Extract sigma2_g
        sigma2_g_match = re.search(r'Sigma\^2_g\[0\] : ([\d.]+)  SE : ([\d.]+)', content)
        if sigma2_g_match:
            results['sigma2_g'] = {
                'value': float(sigma2_g_match.group(1)),
                'se': float(sigma2_g_match.group(2))
            }
            
        # Extract sigma2_gxe
        sigma2_gxe_match = re.search(r'Sigma\^2_gxe\[0\] : ([\d.]+)  SE : ([\d.]+)', content)
        if sigma2_gxe_match:
            results['sigma2_gxe'] = {
                'value': float(sigma2_gxe_match.group(1)),
                'se': float(sigma2_gxe_match.group(2))
            }
            
        # Extract sigma2_nxe
        sigma2_nxe_match = re.search(r'Sigma\^2_nxe\[0\] : ([\d.]+)  SE : ([\d.]+)', content)
        if sigma2_nxe_match:
            results['sigma2_nxe'] = {
                'value': float(sigma2_nxe_match.group(1)),
                'se': float(sigma2_nxe_match.group(2))
            }
            
        # Extract sigma2_e
        sigma2_e_match = re.search(r'Sigma\^2_e : ([\d.]+)  SE : ([\d.]+)', content)
        if sigma2_e_match:
            results['sigma2_e'] = {
                'value': float(sigma2_e_match.group(1)),
                'se': float(sigma2_e_match.group(2))
            }

        # Extract h2_g
        h2_g_match = re.search(r'h2_g\[0\] : ([\d.]+) SE : ([\d.]+)', content)
        if h2_g_match:
            results['h2_g'] = {
                'value': float(h2_g_match.group(1)),
                'se': float(h2_g_match.group(2))
            }

        # Extract h2_gxe
        h2_gxe_match = re.search(r'h2_gxe\[0\] : ([\d.]+) SE : ([\d.]+)', content)
        if h2_gxe_match:
            results['h2_gxe'] = {
                'value': float(h2_gxe_match.group(1)),
                'se': float(h2_gxe_match.group(2))
            }

        # Extract h2_nxe
        h2_nxe_match = re.search(r'h2_nxe\[0\] : ([\d.]+) SE : ([\d.]+)', content)
        if h2_nxe_match:
            results['h2_nxe'] = {
                'value': float(h2_nxe_match.group(1)),
                'se': float(h2_nxe_match.group(2))
            }

        # Extract total h2
        total_h2_match = re.search(r'Total h2 : ([\d.]+) SE: ([\d.]+)', content)
        if total_h2_match:
            results['total_h2'] = {
                'value': float(total_h2_match.group(1)),
                'se': float(total_h2_match.group(2))
            }

        # Extract total h2_g
        total_h2_g_match = re.search(r'Total h2_g : ([\d.]+) SE: ([\d.]+)', content)
        if total_h2_g_match:
            results['total_h2_g'] = {
                'value': float(total_h2_g_match.group(1)),
                'se': float(total_h2_g_match.group(2))
            }

        # Extract total h2_gxe
        total_h2_gxe_match = re.search(r'Total h2_gxe : ([\d.]+) SE: ([\d.]+)', content)
        if total_h2_gxe_match:
            results['total_h2_gxe'] = {
                'value': float(total_h2_gxe_match.group(1)),
                'se': float(total_h2_gxe_match.group(2))
            }

        # Extract enrichment g
        enrichment_g_match = re.search(r'Enrichment g\[0\] : ([\d.]+) SE : ([\d.]+)', content)
        if enrichment_g_match:
            results['enrichment_g'] = {
                'value': float(enrichment_g_match.group(1)),
                'se': float(enrichment_g_match.group(2))
            }
            
    return results

def is_within_range(calculated, ground_truth):
    """Check if the ranges of calculated and ground truth values overlap."""
    # Calculate ranges for both values
    calc_lower = calculated['value'] - calculated['se']
    calc_upper = calculated['value'] + calculated['se']
    truth_lower = ground_truth['value'] - ground_truth['se']
    truth_upper = ground_truth['value'] + ground_truth['se']
    
    # Check if ranges overlap
    return (calc_lower <= truth_upper and calc_upper >= truth_lower)

@pytest.mark.parametrize("config_name", [
    "no_streaming_bin_1.txt",
    "streaming_bin_1.txt"
])
def test_genie_output(config_name):
    """Run the GENIE test and verify the output values for both streaming and non-streaming cases."""
    config_path = os.path.abspath(f'example/configs/genie/{config_name}')
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_config = os.path.join(temp_dir, config_name)
        with open(config_path, 'r') as f:
            config_content = f.read()
        config_content = config_content.replace('trace = yes', 'trace = no')
        config_content = config_content.replace('output=outputs/genie/', f'output={temp_dir}/')
        with open(temp_config, 'w') as f:
            f.write(config_content)
            
        cmd = f"cd example && ./test.sh --config {temp_config}"
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            print(f"Error running test: {stderr.decode()}")
            assert False, f"Test execution failed for {config_name}"
            
        output_file = os.path.join(temp_dir, config_name)
        if not os.path.exists(output_file):
            print(f"Output file not found at: {output_file}")
            print("Current directory contents:")
            print(os.listdir(temp_dir))
            assert False, f"Output file not found at {output_file}"
            
        results = parse_output(output_file)
        
        for key in GROUND_TRUTH:
            assert key in results, f"Missing {key} in results for {config_name}"
            assert is_within_range(results[key], GROUND_TRUTH[key]), \
                f"{key} value {results[key]['value']} is not within range {GROUND_TRUTH[key]['value']} ± {GROUND_TRUTH[key]['se']} for {config_name}"

if __name__ == '__main__':
    test_genie_output("no_streaming_bin_1.txt")
    test_genie_output("streaming_bin_1.txt") 