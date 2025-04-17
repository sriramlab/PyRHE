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
        'value': 0.12410478106518931,
        'se': 0.032512953600579333
    },
    'sigma2_gxe': {
        'value': 0.26624928556220623,
        'se': 0.09731811438356684
    },
    'sigma2_nxe': {
        'value': 0.142724456970253,
        'se': 0.09675039553897923
    },
    'sigma2_e': {
        'value': 0.6757598693113297,
        'se': 0.03268802802730129
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
            
    return results

def is_within_range(calculated, ground_truth):
    """Check if the calculated value is within the ground truth value ± SE."""
    lower_bound = ground_truth['value'] - ground_truth['se']
    upper_bound = ground_truth['value'] + ground_truth['se']
    return lower_bound <= calculated['value'] <= upper_bound

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