#!/usr/bin/env bash
set -e

docker run --gpus all --rm -v $(pwd):/workspace talking-gaussian /bin/bash -lc "python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY"