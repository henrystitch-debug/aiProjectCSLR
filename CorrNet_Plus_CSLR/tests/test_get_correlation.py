"""
Simple unit test for Get_Correlation multi-frame aggregation and boundary handling.
This script does not require extra dependencies (parses baseline.yaml for 'neighbors').
Run with: python tests/test_get_correlation.py
"""
import os
import torch
from corr_map_generation import resnet

# locate baseline.yaml
BASE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'baseline.yaml')
neighbors = 3
try:
    with open(BASE, 'r') as f:
        for line in f:
            if line.strip().startswith('neighbors:'):
                try:
                    neighbors = int(line.split(':',1)[1].strip())
                except Exception:
                    pass
                break
except FileNotFoundError:
    pass

print(f"Using neighbors from config: {neighbors}")

# minimal temporal length must be >= 2*neighbors + 1
T = max(2 * neighbors + 1, 9)
N, C, H, W = 1, 64, 14, 14
x = torch.randn(N, C, T, H, W)

m = resnet.Get_Correlation(channels=C, neighbors=neighbors, agg_mode='concat_conv')
print('Module:', m)

out = m(x, return_affinity=True)
if isinstance(out, tuple):
    features, affinities = out
    print('features.shape =', features.shape)
    print('affinities.shape =', affinities.shape)
else:
    print('output.shape =', out.shape)

print('Test finished.')
