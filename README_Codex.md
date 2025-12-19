# Codex 작업 메모 (TalkingGaussian)

최근 환경 재구성·디버깅 내역을 정리했습니다. 왜/무엇을/어떻게 했는지 한눈에 볼 수 있도록 작성했습니다.

## 핵심 변경 요약
- **Docker 스택을 프로젝트 요구사항( CUDA 11.3 + torch 1.12.1 )으로 재구성**
  - `torch/torchvision/torchaudio` 1.12.1 + cu113 휠을 미리 다운로드 후 Docker 빌드 시 사용.
  - `mmcv-full 1.7.1` 휠, `pytorch3d 0.7.2` (py38/cu113/torch1.12.1) 휠 사용.
- **커스텀 CUDA 확장 재설치**
  - `submodules/diff-gaussian-rasterization` / `submodules/simple-knn` / `gridencoder` 를 editable 설치.
  - `sys.path`에 소스 경로가 빠져 임포트 실패하던 문제를 `.pth` 로 경로 추가하여 해결.
- **런타임 필수 패키지 설치 및 호환성 수정**
  - `plyfile` 설치.
  - `protobuf`를 3.20.3으로 다운그레이드( tensorboard proto 오류 방지 ).
- **LPIPS용 가중치 캐시 + SSL 우회**
  - 사내/사설 인증서 환경에서 `alexnet-owt-7be5be79.pth` 다운로드 실패 → 미리 받아서 `/workspace/.cache/torch/hub/checkpoints` 에 저장.
  - `TORCH_HOME=/workspace/.cache/torch` 로 통일(컨테이너 안에서 export 필요).

## Dockerfile 관련
- 베이스 이미지를 `nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04` 로 교체.
- 미리 받아둔 휠(`wheels/` 디렉터리)로 torch/vision/audio, mmcv-full, pytorch3d 설치.
- 이후 `pip install -r requirements.txt` + tensorflow-gpu/openmim 등 추가 의존성 설치.

## 컨테이너에서 실행한 주요 명령
- 커스텀 확장 재설치 (경로: `/workspace`):
  ```
  export TORCH_CUDA_ARCH_LIST=8.6+PTX
  pip install -e submodules/diff-gaussian-rasterization
  pip install -e submodules/simple-knn
  pip install -e gridencoder
  ```
- 경로 문제 해결: `/usr/lib/python3.8/site-packages/*.egg-link` 가 기본 `sys.path`에 잡히지 않아 `.pth` 추가.  
  (현재 컨테이너에는 `/usr/local/lib/python3.8/dist-packages/local-projects.pth` 로 반영됨.)
- 필수 패키지:
  ```
  pip install plyfile
  pip install 'protobuf<3.21'
  ```
- LPIPS 가중치 사전 캐시(SSL 실패 방지):
  ```
  mkdir -p /workspace/.cache/torch/hub/checkpoints
  curl -k -L -o /workspace/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth \
       https://download.pytorch.org/models/alexnet-owt-7be5be79.pth
  export TORCH_HOME=/workspace/.cache/torch  # 학습 실행 전에 설정
  ```

## 여전히 유의할 점
- 사설 인증서 환경에서 추가 모델/체크포인트 다운로드가 필요하면 `TORCH_HOME` 아래에 미리 넣거나, 일시적으로 `PYTHONHTTPSVERIFY=0` 로 우회해야 함.
- 학습 파이프라인(`scripts/train_xx.sh`)은 순차로 mouth → face → fuse → metrics 를 실행하므로, 앞 단계가 실패하면 이후 단계에서 `chkpnt_*` 파일을 찾지 못해 바로 종료됨. 먼저 SSL/가중치 캐시를 확보해야 정상 진행됨.

## 현재 상태
- Docker 이미지 `talkinggaussian:cu113` 빌드 완료.
- 커스텀 확장 3종 임포트 OK.
- LPIPS용 AlexNet 가중치가 `/workspace/.cache/torch/hub/checkpoints` 에 존재하면 SSL 없이도 바로 학습 가능. 그렇지 않으면 위의 캐시 절차 또는 `PYTHONHTTPSVERIFY=0` 사용 필요.

## 참고: 실행 예시
```
docker exec -it 3bdbcd343309 /bin/bash
export TORCH_HOME=/workspace/.cache/torch
export TORCH_CUDA_ARCH_LIST=8.6+PTX
bash scripts/train_xx.sh data/macron output/macron_project 0
```
