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

## 타임라인 & 해결 요약 (2025-12-19 이후)
- CUDA/torch 불일치 → Dockerfile을 CUDA 11.3 + torch 1.12.1/cu113 휠로 재작성, 빌드 완료.
- 커스텀 확장 미로딩 → diff-gaussian-rasterization / simple-knn / gridencoder 재설치, `.pth` 경로 추가로 임포트 성공.
- SSL 인증서로 LPIPS 가중치 다운로드 실패 → `alexnet-owt-7be5be79.pth`를 `curl -k`로 수동 캐시, `TORCH_HOME=/workspace/.cache/torch` 기본값 적용.
- tensorboard proto 에러 → protobuf 3.20.3으로 다운그레이드.
- Pillow ANTIALIAS 제거 → Pillow는 10 미만 유지(필요 시 `pip install "pillow<10"`).
- tensorboard 히스토그램 dtype 에러 → opacity 텐서 `detach().cpu().view(-1).float().numpy()` 후 로깅, 실패 시 경고만 출력하도록 방어 코드 추가.
- DeepSpeech pb 혼용 → v0.1.0이 아닌 pb를 쓰면 `deepspeech/logits:0` KeyError → 잘못된 pb 제거 후 `get_deepspeech_model_file()`로 v0.1.0 재다운로드하거나 `--deepspeech ~/.tensorflow/models/deepspeech-0_1_0-b90017e8.pb`로 명시하여 해결.
- 인퍼런스 시 오디오 npy 미지정 → transforms에 기록된 기본 오디오 특징 사용. 렌더 mp4에는 오디오가 없으므로 ffmpeg로 오디오를 별도 mux 필요.
- 렌더 속도 이슈 → `--eval`로 test 세트만 렌더, `--fast` 사용, 필요 시 카메라 슬라이스로 뷰 수 축소.

## 주요 이슈별 정리 (원인 → 최종 해결)
- **PyTorch/CUDA 버전 충돌**  
  - 원인: 컨테이너 기본이 CUDA 12.4/torch 2.6 → 프로젝트 요구(CUDA 11.3/torch 1.12.1)와 불일치.  
  - 해결: Dockerfile을 `nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04` 베이스로 교체, 사전 다운로드한 cu113 휠로 torch 스택/ mmcv-full / pytorch3d 설치.

- **커스텀 확장 임포트 실패**  
  - 원인: editable 설치가 `/usr/lib/python3.8/site-packages/*.egg-link`에 깔리고 `sys.path`에 잡히지 않음.  
  - 해결: diff-gaussian-rasterization / simple-knn / gridencoder를 다시 설치하고, `/usr/local/lib/python3.8/dist-packages/local-projects.pth`에 소스 경로를 추가.

- **SSL 인증서로 모델 가중치 다운로드 실패**  
  - 원인: 사설 인증서 체인으로 LPIPS용 AlexNet 가중치 다운로드 오류.  
  - 해결: `curl -k`로 `/workspace/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth` 캐시, `TORCH_HOME=/workspace/.cache/torch`로 통일. (scripts/train_xx.sh에 기본값 설정)

- **TensorBoard 관련 오류**  
  - proto 오류: protobuf 5.x → tensorboard proto 불일치 → `pip install 'protobuf<3.21'`.  
  - 이미지 로깅: Pillow 10에서 ANTIALIAS 제거 → Pillow <10 유지.  
  - 히스토그램 로깅: numpy ufunc dtype 에러 → opacity 텐서를 CPU/float/numpy로 변환하고, 실패 시 경고만 출력하도록 수정.

- **체크포인트 없음으로 fuse/metrics 실패**  
  - 원인: 앞 단계(mouth/face)가 SSL/로깅 오류로 중단되어 `chkpnt_face_latest.pth` / `chkpnt_fuse_latest.pth`가 생성되지 않음.  
  - 해결: 상기 SSL/로깅 문제 해결 후 mouth → face → fuse 순서로 재학습하여 체크포인트 생성.
- **DeepSpeech 특징 추출 실패**  
  - 원인: v0.1.0이 아닌 pb를 로드해 `deepspeech/logits:0` 노드가 없어 KeyError 발생.  
  - 해결: 잘못된 pb를 제거 후 `deepspeech-0_1_0-b90017e8.pb`를 `get_deepspeech_model_file()`로 다시 받아 지정하거나 `--deepspeech ~/.tensorflow/models/deepspeech-0_1_0-b90017e8.pb`로 명시.
- **렌더 mp4에 오디오 없음**  
  - 원인: `synthesize_fuse.py`는 영상만 생성, 오디오 트랙을 mux하지 않음.  
  - 해결: ffmpeg로 오디오 wav와 mp4를 mux (예: `ffmpeg -i video.mp4 -i audio.wav -c:v copy -c:a aac -shortest out.mp4`).

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
