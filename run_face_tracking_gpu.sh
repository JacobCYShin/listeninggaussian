#!/usr/bin/env bash
set -e
export TORCH_CUDA_ARCH_LIST=8.9
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST_OVERRIDE=8.9
python - <<'PY'
import os
import torch
from torch.utils.cpp_extension import CUDA_HOME
print("cuda_available", torch.cuda.is_available())
print("cuda_home", CUDA_HOME)
print("arch_list", os.environ.get("TORCH_CUDA_ARCH_LIST"))
PY
python data_utils/process.py data/acting/acting_challenge_25fps.mp4 --task 8