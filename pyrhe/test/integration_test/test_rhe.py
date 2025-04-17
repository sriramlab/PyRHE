import os
import sys
import subprocess
import re
import tempfile
import shutil
from pathlib import Path
import pytest

GROUND_TRUTH_1_BIN = {
    'sigma2_g': {
        'value': 0.16087,
        'se': 0.0296942
    },
    'sigma2_e': {
        'value': 0.798576,
        'se': 0.0296937
    },
    'h2_g': {
        'value': 0.16767,
        'se': 0.0309492
    },
    'total_h2': {
        'value': 0.16767,
        'se': 0.0309492
    },
    'enrichment_g': {
        'value': 1.0,
        'se': 0.0
    }
}

GROUND_TRUTH_8_BINS = {
    'sigma2_g': [
        {'value': 0.0305974, 'se': 0.0110474},
        {'value': 0.0171042, 'se': 0.0112948},
        {'value': 0.012402, 'se': 0.00981038},
        {'value': 0.0179935, 'se': 0.00954467},
        {'value': 0.0237989, 'se': 0.0111944},
        {'value': 0.0331766, 'se': 0.0135935},
        {'value': 0.000183239, 'se': 0.00809398},
        {'value': 0.0426152, 'se': 0.0112811}
    ],
    'sigma2_e': {
        'value': 0.7816,
        'se': 0.0336697
    },
    'h2_g': [
        {'value': 0.0318898, 'se': 0.011514},
        {'value': 0.0178267, 'se': 0.0117716},
        {'value': 0.0129259, 'se': 0.0102248},
        {'value': 0.0187536, 'se': 0.00994782},
        {'value': 0.0248042, 'se': 0.0116671},
        {'value': 0.034578, 'se': 0.0141675},
        {'value': 0.000190979, 'se': 0.00843588},
        {'value': 0.0444153, 'se': 0.0117577}
    ],
    'total_h2': {
        'value': 0.1853,
        'se': 0.0336697
    },
    'enrichment_g': [
        {'value': 1.37616, 'se': 0.50122},
        {'value': 0.769286, 'se': 0.466571},
        {'value': 0.556908, 'se': 0.412129},
        {'value': 0.797795, 'se': 0.401199},
        {'value': 1.09134, 'se': 0.478408},
        {'value': 1.47798, 'se': 0.538007},
        {'value': 0.007882, 'se': 0.35385},
        {'value': 2.02181, 'se': 0.532216}
    ]
}

def parse_output(output_file, num_bins=1):
    """Parse the output file and extract the calculated values."""
    results = {}
    if num_bins > 1:
        results = {
            'sigma2_g': [],
            'h2_g': [],
            'enrichment_g': []
        }
    
    with open(output_file, 'r') as f:
        content = f.read()
        print(f"Debug: Content length: {len(content)}")
        print(f"Debug: First 500 chars: {content[:500]}")
        
        # Extract sigma2_e first since it's common to both cases
        sigma2_e_match = re.search(r'Sigma\^2_e : ([\d.]+)  SE : ([\d.]+)', content)
        if sigma2_e_match:
            print(f"Debug: Found sigma2_e match: {sigma2_e_match.groups()}")
            results['sigma2_e'] = {
                'value': float(sigma2_e_match.group(1)),
                'se': float(sigma2_e_match.group(2))
            }
        else:
            print("Debug: No sigma2_e match found")
            # Try alternative pattern
            sigma2_e_match = re.search(r'Sigma\^2_e\s*:\s*([\d.]+)\s*SE\s*:\s*([\d.]+)', content)
            if sigma2_e_match:
                print(f"Debug: Found sigma2_e match with alternative pattern: {sigma2_e_match.groups()}")
                results['sigma2_e'] = {
                    'value': float(sigma2_e_match.group(1)),
                    'se': float(sigma2_e_match.group(2))
                }
            else:
                print("Debug: No sigma2_e match found with alternative pattern")
        
        if num_bins == 1:
            # Extract sigma2_g
            sigma2_g_match = re.search(r'Sigma\^2_g\[0\] : ([\d.]+)  SE : ([\d.]+)', content)
            if sigma2_g_match:
                results['sigma2_g'] = {
                    'value': float(sigma2_g_match.group(1)),
                    'se': float(sigma2_g_match.group(2))
                }
                
            # Extract h2_g
            h2_g_match = re.search(r'h2_g\[0\] : ([\d.]+) : ([\d.]+)', content)
            if h2_g_match:
                results['h2_g'] = {
                    'value': float(h2_g_match.group(1)),
                    'se': float(h2_g_match.group(2))
                }
                
            # Extract enrichment_g
            enrichment_match = re.search(r'Enrichment g\[0\] : ([\d.]+) SE : ([\d.]+)', content)
            if enrichment_match:
                results['enrichment_g'] = {
                    'value': float(enrichment_match.group(1)),
                    'se': float(enrichment_match.group(2))
                }
        else:
            # Extract sigma2_g for all bins
            sigma2_g_matches = re.finditer(r'Sigma\^2_g\[(\d+)\] : ([\d.-]+)  SE : ([\d.]+)', content)
            for match in sigma2_g_matches:
                bin_idx = int(match.group(1))
                value = float(match.group(2))
                se = float(match.group(3))
                results['sigma2_g'].append({'value': value, 'se': se})
                
            # Extract h2_g for all bins
            h2_g_matches = re.finditer(r'h2_g\[(\d+)\] : ([\d.-]+) : ([\d.]+)', content)
            for match in h2_g_matches:
                bin_idx = int(match.group(1))
                value = float(match.group(2))
                se = float(match.group(3))
                results['h2_g'].append({'value': value, 'se': se})
                
            # Extract enrichment_g for all bins
            enrichment_matches = re.finditer(r'Enrichment g\[(\d+)\] : ([\d.-]+) SE : ([\d.]+)', content)
            for match in enrichment_matches:
                bin_idx = int(match.group(1))
                value = float(match.group(2))
                se = float(match.group(3))
                results['enrichment_g'].append({'value': value, 'se': se})
            
        # Extract total_h2 (common to both cases)
        total_h2_match = re.search(r'Total h2 : ([\d.]+) SE: ([\d.]+)', content)
        if total_h2_match:
            results['total_h2'] = {
                'value': float(total_h2_match.group(1)),
                'se': float(total_h2_match.group(2))
            }
            
    print(f"Debug: Final results: {results}")
    return results

def is_within_range(calculated, ground_truth):
    """Check if the calculated value is within the ground truth value ± SE."""
    lower_bound = ground_truth['value'] - ground_truth['se']
    upper_bound = ground_truth['value'] + ground_truth['se']
    return lower_bound <= calculated['value'] <= upper_bound

@pytest.mark.parametrize("config_name,num_bins,ground_truth", [
    ("no_streaming_bin_1.txt", 1, GROUND_TRUTH_1_BIN),
    ("streaming_bin_1.txt", 1, GROUND_TRUTH_1_BIN),
    ("no_streaming_bin_8.txt", 8, GROUND_TRUTH_8_BINS),
    ("streaming_bin_8.txt", 8, GROUND_TRUTH_8_BINS)
])
def test_rhe_output(config_name, num_bins, ground_truth):
    """Run the RHE test and verify the output values for both streaming and non-streaming cases."""
    config_path = os.path.abspath(f'example/configs/rhe/{config_name}')
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_config = os.path.join(temp_dir, config_name)
        with open(config_path, 'r') as f:
            config_content = f.read()
        config_content = config_content.replace('trace = yes', 'trace = no')
        config_content = config_content.replace('output=outputs/rhe/', f'output={temp_dir}/')
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
            
        results = parse_output(output_file, num_bins)
        
        if num_bins == 1:
            for key in ground_truth:
                assert key in results, f"Missing {key} in results for {config_name}"
                assert is_within_range(results[key], ground_truth[key]), \
                    f"{key} value {results[key]['value']} is not within range {ground_truth[key]['value']} ± {ground_truth[key]['se']} for {config_name}"
        else:
            # Check sigma2_g for all bins
            for i, (calculated, truth) in enumerate(zip(results['sigma2_g'], ground_truth['sigma2_g'])):
                assert is_within_range(calculated, truth), \
                    f"sigma2_g[{i}] value {calculated['value']} is not within range {truth['value']} ± {truth['se']} for {config_name}"
            
            # Check sigma2_e
            assert is_within_range(results['sigma2_e'], ground_truth['sigma2_e']), \
                f"sigma2_e value {results['sigma2_e']['value']} is not within range {ground_truth['sigma2_e']['value']} ± {ground_truth['sigma2_e']['se']} for {config_name}"
            
            # Check h2_g for all bins
            for i, (calculated, truth) in enumerate(zip(results['h2_g'], ground_truth['h2_g'])):
                assert is_within_range(calculated, truth), \
                    f"h2_g[{i}] value {calculated['value']} is not within range {truth['value']} ± {truth['se']} for {config_name}"
            
            # Check total_h2
            assert is_within_range(results['total_h2'], ground_truth['total_h2']), \
                f"total_h2 value {results['total_h2']['value']} is not within range {ground_truth['total_h2']['value']} ± {ground_truth['total_h2']['se']} for {config_name}"
            
            # Check enrichment_g for all bins
            for i, (calculated, truth) in enumerate(zip(results['enrichment_g'], ground_truth['enrichment_g'])):
                assert is_within_range(calculated, truth), \
                    f"enrichment_g[{i}] value {calculated['value']} is not within range {truth['value']} ± {truth['se']} for {config_name}" 