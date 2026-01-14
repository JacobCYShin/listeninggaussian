#!/usr/bin/env bash
set -e

docker run --gpus all --rm -v $(pwd):/workspace talking-gaussian /bin/bash -lc "pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable' --no-build-isolation && python data_utils/face_tracking/face_tracker.py --path=data/may/ori_imgs --img_h=512 --img_w=512 --frame_num=200"