# 1. Base: match project requirement (CUDA 11.3 + PyTorch 1.12.1)
# Ubuntu 20.04 이미지를 사용해 torch 1.12.1 + cu113 빌드/런타임을 맞춘다.
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# 2. 시스템 패키지 설치
# portaudio19-dev (pyaudio 빌드용 필수), alsa-utils 포함
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    cmake \
    build-essential \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsndfile1 \
    portaudio19-dev \
    alsa-utils \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# 3. Python 설정 (python3.8 기본)
RUN ln -sf /usr/bin/python3 /usr/bin/python
RUN pip3 install --upgrade pip setuptools wheel ninja

# 4. 사전 다운로드한 wheel로 PyTorch 스택 설치 (torch/vision/audio, mmcv-full)
COPY wheels /tmp/wheels
RUN pip3 install /tmp/wheels/torch-1.12.1+cu113-cp38-cp38-linux_x86_64.whl \
    /tmp/wheels/torchvision-0.13.1+cu113-cp38-cp38-linux_x86_64.whl \
    /tmp/wheels/torchaudio-0.12.1+cu113-cp38-cp38-linux_x86_64.whl \
    /tmp/wheels/mmcv_full-1.7.1-cp38-cp38-manylinux1_x86_64.whl

# 5. 프로젝트 의존성 기본 설치 (requirements.txt 기반)
WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
RUN pip3 install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

# 6. 추가 패키지: tensorflow-gpu 2.8.0, openmim, mmcv-full 1.7.1
RUN pip3 install --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    tensorflow-gpu==2.8.0 openmim \
    && mim install mmcv-full==1.7.1

# 7. PyTorch3D (prebuilt wheel, torch 1.12.1 + cu113 + py38)
ENV TORCH_CUDA_ARCH_LIST="8.6+PTX"
ENV FORCE_CUDA="1"
RUN pip3 install /tmp/wheels/pytorch3d-0.7.2-cp38-cp38-linux_x86_64.whl

WORKDIR /workspace
