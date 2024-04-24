__version__ = "0.0.2"

import os
import torch
from stable_baselines3_extensions.her.her_replay_buffer import HerReplayBufferExt

__all__ = [
    "HerReplayBufferExt",
]

if torch.cuda.is_available():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)