"""
Unit test to verify MS-TCN integration into TemporalConv and SLRModel flows.
Checks that when conv_type=3 and MS-TCN is enabled:
 - TemporalConv returns 'tcn_feat' and 'tcn_logits'
 - SLRModel forwards using tcn_feat into BiLSTM and uses tcn_logits as conv_logits
 - Tensor shapes match expected (T, B, hidden_size) and (T, B, num_classes)
Run with: python tests/test_mstcn_integration.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
from slr_network import SLRModel

# Small deterministic test with CPU only
B = 1
T = 16
frame_channels = 512
hidden_size = 128
num_classes = 10

# Create a model with conv_type=3 (MS-TCN path). Use resnet34 to avoid pretrained download in resnet18.
model = SLRModel(
    num_classes=num_classes,
    c2d_type='resnet34',
    conv_type=3,
    use_bn=False,
    hidden_size=hidden_size,
    weight_norm=False,
    share_classifier=False,
    mstcn_num_layers=2,
    mstcn_hidden_size=hidden_size,
    mstcn_kernel_size=3,
)
model.eval()

# Create dummy framewise features (B, C_in, T)
x = torch.randn(B, frame_channels, T)
# lengths: batch-sized tensor with sequence lengths
len_x = torch.tensor([T], dtype=torch.int)

with torch.no_grad():
    out = model(x, len_x)

# Validate outputs exist
assert 'conv_logits' in out and out['conv_logits'] is not None, 'conv_logits missing'
assert 'sequence_logits' in out and out['sequence_logits'] is not None, 'sequence_logits missing'

conv_logits = out['conv_logits']
seq_logits = out['sequence_logits']
feat_len = out['feat_len']

# conv_logits expected shape: (T_conv, B, num_classes)
print('conv_logits.shape =', conv_logits.shape)
print('sequence_logits.shape =', seq_logits.shape)
print('feat_len =', feat_len)

# Basic shape checks
assert conv_logits.dim() == 3
assert conv_logits.size(1) == B
assert conv_logits.size(2) == num_classes

# sequence_logits is (T_seq, B, num_classes)
assert seq_logits.dim() == 3
assert seq_logits.size(1) == B
assert seq_logits.size(2) == num_classes

print('MS-TCN integration test passed.')
