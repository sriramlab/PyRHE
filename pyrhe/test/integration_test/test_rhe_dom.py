import os
import sys
import subprocess
import re
import tempfile
import shutil
from pathlib import Path
import pytest

GROUND_TRUTH_1_BIN = {
    'sigma2_g': [
        {'value': 0.169788, 'se': 0.0315156},
        {'value': 0.0330849, 'se': 0.027581}
    ],
    'sigma2_e': {
        'value': 0.756049,
        'se': 0.0420903
    },
    'h2_g': [
        {'value': 0.177061, 'se': 0.0328657},
        {'value': 0.0345022, 'se': 0.0287626}
    ],
    'total_h2': {
        'value': 0.211564,
        'se': 0.0438933
    },
    'enrichment_g': [
        {'value': 1.67384, 'se': 0.237277},
        {'value': 0.326164, 'se': 0.237277}
    ]
}

GROUND_TRUTH_8_BINS = {
    'sigma2_g': [
        {'value': 0.0314795, 'se': 0.0116701},
        {'value': 0.0122328, 'se': 0.0101368},
        {'value': 0.0103779, 'se': 0.00940871},
        {'value': 0.01697, 'se': 0.00955496},
        {'value': 0.0209867, 'se': 0.0105554},
        {'value': 0.0288463, 'se': 0.0125491},
        {'value': -0.00129655, 'se': 0.00761921},
        {'value': 0.0428965, 'se': 0.0115559},
        {'value': 0.0285394, 'se': 0.0267265}
    ],
    'sigma2_e': {
        'value': 0.767889,
        'se': 0.0393888
    },
    'h2_g': [
        {'value': 0.032828, 'se': 0.0121701},
        {'value': 0.0127569, 'se': 0.0105711},
        {'value': 0.0108224, 'se': 0.00981175},
        {'value': 0.0176969, 'se': 0.00996427},
        {'value': 0.0218858, 'se': 0.0110076},
        {'value': 0.030082, 'se': 0.0130867},
        {'value': -0.00135209, 'se': 0.0079456},
        {'value': 0.0447341, 'se': 0.012051},
        {'value': 0.029762, 'se': 0.0278714}
    ],
    'total_h2': {
        'value': 0.199216,
        'se': 0.0410761
    },
    'enrichment_g': [
        {'value': 2.63658, 'se': 1.00069},
        {'value': 1.02457, 'se': 0.819405},
        {'value': 0.867813, 'se': 0.750605},
        {'value': 1.40115, 'se': 0.798591},
        {'value': 1.79216, 'se': 0.911399},
        {'value': 2.39306, 'se': 1.06393},
        {'value': -0.103857, 'se': 0.624482},
        {'value': 3.78989, 'se': 1.12316},
        {'value': 0.298791, 'se': 0.249484}
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
        
        # Extract sigma2_e first since it's common to both cases
        sigma2_e_match = re.search(r'Sigma\^2_e : ([\d.]+)  SE : ([\d.]+)', content)
        if sigma2_e_match:
            results['sigma2_e'] = {
                'value': float(sigma2_e_match.group(1)),
                'se': float(sigma2_e_match.group(2))
            }
        
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
            enrichment_g_match = re.search(r'Enrichment g\[0\] : ([\d.]+) SE : ([\d.]+)', content)
            if enrichment_g_match:
                results['enrichment_g'] = {
                    'value': float(enrichment_g_match.group(1)),
                    'se': float(enrichment_g_match.group(2))
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
            enrichment_g_matches = re.finditer(r'Enrichment g\[(\d+)\] : ([\d.-]+) SE : ([\d.]+)', content)
            for match in enrichment_g_matches:
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

@pytest.mark.parametrize("config_name,num_bins,ground_truth", [
    ("no_streaming_bin_1.txt", 1, GROUND_TRUTH_1_BIN),
    ("streaming_bin_1.txt", 1, GROUND_TRUTH_1_BIN),
    ("no_streaming_bin_8.txt", 8, GROUND_TRUTH_8_BINS),
    ("streaming_bin_8.txt", 8, GROUND_TRUTH_8_BINS)
])
def test_rhe_dom_output(config_name, num_bins, ground_truth):
    """Run the RHE-DOM test and verify the output values for both streaming and non-streaming cases."""
    config_path = os.path.abspath(f'example/configs/rhe_dom/{config_name}')
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_config = os.path.join(temp_dir, config_name)
        with open(config_path, 'r') as f:
            config_content = f.read()
        config_content = config_content.replace('trace = yes', 'trace = no')
        config_content = config_content.replace('output=outputs/rhe_dom/', f'output={temp_dir}/')
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
        
        # Verify sigma2_e
        assert 'sigma2_e' in results, "sigma2_e not found in results"
        assert is_within_range(results['sigma2_e'], ground_truth['sigma2_e']), \
            f"sigma2_e value {results['sigma2_e']} not within range of ground truth {ground_truth['sigma2_e']}"
            
        # Verify total_h2
        assert 'total_h2' in results, "total_h2 not found in results"
        assert is_within_range(results['total_h2'], ground_truth['total_h2']), \
            f"total_h2 value {results['total_h2']} not within range of ground truth {ground_truth['total_h2']}"
            
        if num_bins == 1:
            # Verify single bin case
            assert 'sigma2_g' in results, "sigma2_g not found in results"
            assert is_within_range(results['sigma2_g'], ground_truth['sigma2_g'][0]), \
                f"sigma2_g value {results['sigma2_g']} not within range of ground truth {ground_truth['sigma2_g'][0]}"
                
            assert 'h2_g' in results, "h2_g not found in results"
            assert is_within_range(results['h2_g'], ground_truth['h2_g'][0]), \
                f"h2_g value {results['h2_g']} not within range of ground truth {ground_truth['h2_g'][0]}"
                
            assert 'enrichment_g' in results, "enrichment_g not found in results"
            assert is_within_range(results['enrichment_g'], ground_truth['enrichment_g'][0]), \
                f"enrichment_g value {results['enrichment_g']} not within range of ground truth {ground_truth['enrichment_g'][0]}"
        else:
            # Verify multiple bins case
            assert len(results['sigma2_g']) >= 2, "Expected at least 2 bins for sigma2_g"
            assert len(results['h2_g']) >= 2, "Expected at least 2 bins for h2_g"
            assert len(results['enrichment_g']) >= 2, "Expected at least 2 bins for enrichment_g"
            
            # Check all values for both bins
            # Bin 0
            assert is_within_range(results['sigma2_g'][0], ground_truth['sigma2_g'][0]), \
                f"sigma2_g[0] value {results['sigma2_g'][0]} not within range of ground truth {ground_truth['sigma2_g'][0]}"
            assert is_within_range(results['h2_g'][0], ground_truth['h2_g'][0]), \
                f"h2_g[0] value {results['h2_g'][0]} not within range of ground truth {ground_truth['h2_g'][0]}"
            assert is_within_range(results['enrichment_g'][0], ground_truth['enrichment_g'][0]), \
                f"enrichment_g[0] value {results['enrichment_g'][0]} not within range of ground truth {ground_truth['enrichment_g'][0]}"
            
            # Bin 1
            assert is_within_range(results['sigma2_g'][1], ground_truth['sigma2_g'][1]), \
                f"sigma2_g[1] value {results['sigma2_g'][1]} not within range of ground truth {ground_truth['sigma2_g'][1]}"
            assert is_within_range(results['h2_g'][1], ground_truth['h2_g'][1]), \
                f"h2_g[1] value {results['h2_g'][1]} not within range of ground truth {ground_truth['h2_g'][1]}"
            assert is_within_range(results['enrichment_g'][1], ground_truth['enrichment_g'][1]), \
                f"enrichment_g[1] value {results['enrichment_g'][1]} not within range of ground truth {ground_truth['enrichment_g'][1]}"

if __name__ == '__main__':
    test_rhe_dom_output("no_streaming_bin_1.txt", 1, GROUND_TRUTH_1_BIN)
    test_rhe_dom_output("streaming_bin_1.txt", 1, GROUND_TRUTH_1_BIN)
    test_rhe_dom_output("no_streaming_bin_8.txt", 8, GROUND_TRUTH_8_BINS)
    test_rhe_dom_output("streaming_bin_8.txt", 8, GROUND_TRUTH_8_BINS) 