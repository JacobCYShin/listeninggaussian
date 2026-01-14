# Diff-Listener: Probabilistic 3D Listening Head Generation via Diffusion-based Gaussian Splatting

This repository implements **Diff-Listener**, a probabilistic 3D listening head generation system that combines diffusion-based motion generation with high-fidelity 3D Gaussian Splatting rendering.

## Overview

Diff-Listener generates natural and diverse listener reactions by processing multi-modal inputs (speaker audio and visual expression) through a two-stage architecture:

- **Brain (Motion Generator)**: A diffusion-based model that generates FLAME parameters from speaker audio and visual features
- **Body (Renderer)**: A modified TalkingGaussian-based 3D Gaussian Splatting renderer that produces high-fidelity facial animations from FLAME parameters

## Architecture

### Brain: Probabilistic Motion Generator

The motion generator is a 1D diffusion model that processes:
- **Speaker Audio**: Wav2Vec 2.0 features (768-dim)
- **Speaker Visual**: EMOCA expression codes (50-dim)
- **Listener History**: Previous frame motion (temporal consistency)

**Output**: Listener FLAME parameters (Expression 50 + Pose 6 = 56-dim)

### Body: High-Fidelity 3D Renderer

The renderer is based on TalkingGaussian, modified to accept FLAME parameters as input instead of audio features. It maintains fine-grained expression control essential for natural listener reactions.

## Key Features

- **High-Fidelity Rendering**: 3D Gaussian Splatting enables 512x512+ resolution without 2D warping artifacts
- **Diverse Reactions**: Diffusion-based generation produces varied, natural responses to identical inputs
- **Multi-modal Awareness**: Incorporates both audio and visual speaker information for context-aware reactions
- **Two-Stage Training**: Decoupled training of renderer and motion generator for stability

## Installation

Tested on Ubuntu 18.04/22.04, CUDA 11.3, PyTorch 1.12.1

```bash
git clone <repository-url> --recursive
conda env create --file environment.yml
conda activate talking_gaussian
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install tensorflow-gpu==2.8.0
```

### Dependencies

Install submodules and prepare models:

```bash
bash scripts/prepare.sh
```

For face parsing (EasyPortrait):

```bash
conda activate talking_gaussian
pip install -U openmim
mim install mmcv-full==1.7.1
cd data_utils/easyportrait
wget "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/easyportrait/experiments/models/fpn-fp-512.pth"
```

## Data Preparation

### Body Training Data

For listener identity learning, prepare a video (3-5 minutes, 25 FPS, 512x512 resolution):

1. Extract frames and landmarks:
```bash
python data_utils/process.py data/<ID>/<ID>.mp4
```

2. Extract FLAME parameters using EMOCA:
```bash
# Extract listener visual (GT): [Total_Frames, 56]
# Expression (50) + Pose (6: Jaw + Neck + Head)
```

### Brain Training Data (ViCo Dataset)

1. Extract speaker audio features (Wav2Vec 2.0):
   - Output: `[Total_Frames, 768]` numpy array

2. Extract speaker visual features (EMOCA):
   - Output: `[Total_Frames, 50]` numpy array (Expression only)

3. Extract listener GT (EMOCA):
   - Output: `[Total_Frames, 56]` numpy array

**Critical**: Ensure frame synchronization across all three modalities.

## Training

### Stage 1: Body (Renderer) Training

Train the 3D Gaussian Splatting renderer to reconstruct faces from FLAME parameters:

```bash
# Modify deformation network to accept FLAME parameters instead of audio
# Train with GT FLAME parameters -> original images
bash scripts/train_xx.sh data/<ID> output/<project_name> <GPU_ID>
```

### Stage 2: Brain (Motion Generator) Training

Train the diffusion model to generate FLAME parameters from multi-modal inputs:

```bash
# Train diffusion model: (Audio, Visual) -> FLAME Parameters
# Loss: MSE in parameter space
python train_diffusion.py --config configs/brain_config.yaml
```

## Inference

Generate listening head videos from speaker audio and visual features:

```bash
python synthesize_listener.py \
    --audio <speaker_audio_features>.npy \
    --visual <speaker_visual_features>.npy \
    --body_checkpoint output/<project_name>/chkpnt_fuse_latest.pth \
    --brain_checkpoint output/brain_model/checkpoint.pth \
    --output output/listener_video.mp4
```

## Project Structure

```
├── scene/
│   ├── gaussian_model.py          # 3D Gaussian model
│   ├── motion_net.py              # Motion networks (legacy, to be modified)
│   └── dataset_readers.py         # Data loading
├── gaussian_renderer/             # Rendering pipeline
├── data_utils/                    # Data preprocessing
├── study/                         # Documentation and study materials
└── scripts/                       # Training and utility scripts
```

## Development Plan

See `개발계획_v2.md` for detailed development phases and implementation guidelines.

## Research Direction

See `연구방향.md` for research motivation, methodology, and contributions.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{difflistener2025,
  title={Diff-Listener: Probabilistic 3D Listening Head Generation via Diffusion-based Gaussian Splatting},
  author={...},
  journal={...},
  year={2025}
}
```

## Acknowledgments

This work builds upon:
- [TalkingGaussian](https://github.com/Fictionarry/TalkingGaussian) (ECCV 2024)
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [EMOCA](https://github.com/radekd91/emoca) for FLAME parameter extraction

## License

This code is provided for research purposes only. See the original TalkingGaussian repository for licensing details.

## Legacy Documentation

The original TalkingGaussian README is preserved in `README_LEGACY.md` for reference.
