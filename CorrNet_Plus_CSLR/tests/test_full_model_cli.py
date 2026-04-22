"""
Small CLI test to construct the full SLRModel via main. It loads configs/baseline.yaml and builds the Processor which instantiates the model.
Run from repo root: python3 tests/test_full_model_cli.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import yaml
from main import Processor
import argparse
import utils

# load baseline config
with open('./configs/baseline.yaml','r') as f:
    cfg = yaml.safe_load(f)

# build a minimal args namespace using utils.get_parser defaults
parser = utils.get_parser()
parser.set_defaults(**cfg)
args = parser.parse_args([])
# dataset_info is needed by Processor; load dataset config
with open(f"./configs/{args.dataset}.yaml", 'r') as f:
    args.dataset_info = yaml.load(f, Loader=yaml.FullLoader)

# override heavy operations
args.phase = 'test'
args.load_weights = None
args.load_checkpoints = None
# avoid GPU allocation and side-effects: force CPU only and use a fresh work_dir
import tempfile, time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
args.device = ''
args.work_dir = os.path.join(tempfile.gettempdir(), f'corrnet_test_{int(time.time())}')
print('Using temporary work_dir:', args.work_dir)

# patch utils to avoid GPU allocation / CUDA initialization during test
class _DummyGpuDataParallel:
    def __init__(self):
        self.gpu_list = []
        self.output_device = 'cpu'
    def set_device(self, device):
        # ignore any device requests and stay on CPU
        self.gpu_list = []
        self.output_device = 'cpu'
    def to(self, *args, **kwargs):
        return

utils.GpuDataParallel = _DummyGpuDataParallel

print('Constructing Processor and model (no weights loaded).')
proc = Processor(args)
print('Model constructed:')
print(proc.model)
print('Done.')
