import sys, os
from pathlib import Path
class MockNN: pass
import types
mock_torch = types.ModuleType("torch")
mock_nn = types.ModuleType("torch.nn")
mock_optim = types.ModuleType("torch.optim")
mock_optim.Optimizer = type("Optimizer", (), {})
mock_torch.nn = mock_nn
mock_torch.optim = mock_optim
mock_nn.Module = MockNN
mock_torch.save = lambda state, path: open(path, 'w').write(str(state))
sys.modules['torch'] = mock_torch
sys.modules['torch.nn'] = mock_nn

# Also need to mock utils.logger
mock_utils = types.ModuleType("utils")
mock_logger = types.ModuleType("utils.logger")
mock_logger.get_logger = lambda name: lambda *args: None
sys.modules['utils'] = mock_utils
sys.modules['utils.logger'] = mock_logger

from training.checkpoint import save_checkpoint
class DummyModel(MockNN):
    def state_dict(self): return {'mat': [1,2,3]}
class DummyOpt(mock_optim.Optimizer):
    def state_dict(self): return {'opt': True}

save_checkpoint(
    model=DummyModel(),
    epoch=1,
    best_f1=0.8,
    is_best=False,
    optimizer=DummyOpt(),
    save_dir='outputs/checkpoints/visual/fold1'
)

save_checkpoint(
    model=DummyModel(),
    epoch=1,
    best_f1=0.8,
    is_best=True,
    optimizer=DummyOpt(),
    save_dir='outputs/checkpoints/visual/fold1'
)
print("Tree:")
import glob
for p in Path('outputs/checkpoints/visual').rglob('*.pt'):
    print("   ", p)
