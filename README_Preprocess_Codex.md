# TalkingGaussian 전처리 환경 재현 가이드 (Codex 기록)

이 문서는 `talking-gaussian:latest (1a42d90b59a3)` 이미지 기반으로 전처리 환경을 다시 만들고,
`data/may` 데이터 전처리를 끝까지 재현할 수 있도록 정리한 상세 기록이다.

## 1. 전제 조건
- Docker + NVIDIA Container Toolkit 설치 완료
- GPU 드라이버 정상 (예: `nvidia-smi` 동작)
- 작업 경로: `~/code/TalkingGaussian`

## 2. 이미지/컨테이너 준비
### 2.1 기존 이미지로 컨테이너 실행
```bash
docker run --gpus all -it --name tg_preprocess \
  -v ~/code/TalkingGaussian:/workspace \
  talking-gaussian:latest /bin/bash
```

이미 실행 중인 컨테이너가 있으면 다음으로 접속:
```bash
docker exec -it 04a5 /bin/bash
```

### 2.2 GPU 확인
```bash
nvidia-smi
python - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
PY
```

## 3. 전처리 환경 구성 (컨테이너 내부)
### 3.1 기본 환경 변수
```bash
export TORCH_HOME=/workspace/.cache/torch
```

### 3.2 pytorch3d 설치 (전처리 face tracking 용)
```bash
pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable' --no-build-isolation
```

## 4. 코드 수정 사항 (재현을 위해 필요)
현재 repo에 반영된 수정 사항을 기준으로 정리한다.

### 4.1 face tracking device 오류 수정
오류: `AttributeError: 'Face_3DMM' object has no attribute 'device'`

수정 파일:
- `data_utils/face_tracking/facemodel.py`

수정 요지:
- `Face_3DMM.__init__` 시작 부분에 `self.device`를 먼저 설정하도록 변경
  ```py
  if device is None:
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  self.device = device
  ```

### 4.2 AU CSV 헤더 공백 문제
오류: OpenFace `au.csv` 헤더에 공백이 섞여 키 매칭 실패

수정 파일:
- `scene/dataset_readers.py`

수정 요지:
- 컬럼명을 strip 처리하고, 키 접근을 공백 없는 키로 통일

### 4.3 메모리 과다 사용 방지
오류: preload 기본값이 True라 큰 데이터에서 OOM 가능성

수정 파일:
- `scene/dataset_readers.py`

수정 요지:
- preload 기본값을 False로 변경

### 4.4 gridencoder 빌드 플래그 문제
오류: C++14 설정으로 빌드 실패/경고 발생

수정 파일:
- `gridencoder/backend.py`
- `gridencoder/setup.py`

수정 요지:
- `-std=c++14` -> `-std=c++17`

## 5. 전처리 실행 절차 (may 데이터 기준)
데이터: `data/may/may.mp4`

### 5.1 이미 완료된 전처리 산출물 확인
```bash
ls -la data/may
```
확인 대상:
- `aud.wav`, `aud_ds.npy`
- `ori_imgs/`, `gt_imgs/`, `torso_imgs/`, `parsing/`
- `track_params.pt`

### 5.2 face tracking (task 8)
```bash
python data_utils/face_tracking/face_tracker.py \
  --path=data/may/ori_imgs --img_h=512 --img_w=512 --frame_num=6000
```
성공 시: `params saved` 출력 및 `data/may/track_params.pt` 생성

### 5.3 transforms 저장 (task 9)
```bash
python data_utils/process.py data/may/may.mp4 --task 9
```
성공 시: `data/may/transforms_*.json` 생성

## 6. 전처리 완료 체크리스트
아래가 모두 존재하면 전처리 완료로 판단 가능:
- `data/may/ori_imgs/`
- `data/may/gt_imgs/`
- `data/may/torso_imgs/`
- `data/may/parsing/`
- `data/may/aud.wav`
- `data/may/aud_ds.npy`
- `data/may/track_params.pt`
- `data/may/transforms_train.json` (및 관련 transforms 파일)

## 7. 컨테이너 변경사항 보존 방법
### 7.1 현재 컨테이너 상태가 유지되는가?
- 컨테이너가 **삭제되지 않는 한** 내부 변경사항은 유지됨.
- 하지만 컨테이너를 삭제하면 **변경사항은 사라짐**.

### 7.2 변경사항을 이미지로 고정 (권장)
```bash
docker commit 04a5 talking-gaussian:preprocess-fixed
```
이렇게 하면 이후에 새 컨테이너를 띄워도 동일 환경을 재사용 가능.

### 7.3 코드/데이터는 볼륨 마운트로 보존
이미 `-v ~/code/TalkingGaussian:/workspace`로 마운트했다면,
`/workspace` 아래 변경사항은 컨테이너 삭제와 무관하게 **호스트에 그대로 남음**.

## 8. 자주 겪은 문제 & 해결 요약
- **CUDA/torch mismatch**: PyTorch 빌드된 CUDA 버전과 실제 CUDA 드라이버 버전 불일치 시 빌드 실패
- **face tracking device 에러**: `self.device` 초기화 누락 문제 (코드 수정 필요)
- **SSL 문제로 모델 다운로드 실패**: `TORCH_HOME` 캐시 경로 지정 및 사전 다운로드로 해결 가능
- **may 학습 실패 원인**: `data/may/au.csv`와 `data/may/teeth_mask/*.npy`가 없어 학습 중단
- **teeth_mask 생성 실패**: `mmcv._ext` 모듈이 빠진 mmcv 설치 + `prettytable` 누락
- **torch 2.6 체크포인트 로딩 오류**: `torch.load` 기본값이 `weights_only=True`로 바뀌어
  `fpn-fp-512.pth` 로딩 실패 (호환 로직 추가)

---

이 문서는 현재 상태를 기준으로 한 “재현 가능한 전처리 환경 구축 기록”이다.
컨테이너 변경사항을 영구화하려면 반드시 `docker commit` 또는 Dockerfile 기반 재빌드가 필요하다.

---

## 9. may 전처리/학습 문제 원인 & 해결 상세

### 9.1 문제 원인 요약
1) `au.csv` 누락  
   `train_xx.sh` 실행 시 `FileNotFoundError: /workspace/data/may/au.csv`
2) `teeth_mask` 누락  
   `train_mouth.py`에서 `FileNotFoundError: data/may/teeth_mask/NN.npy`
3) `mmcv._ext` 누락  
   `mmcv-full`이 CUDA ops 없이 설치되어 `ModuleNotFoundError: mmcv._ext`
4) torch 2.6 체크포인트 로딩 실패  
   `fpn-fp-512.pth` 로딩 시 `Weights only load failed` 발생

### 9.2 해결 방법
- `au.csv` 생성 (OpenFace):
  ```bash
  docker run --rm --entrypoint /bin/bash \
    -v ~/code/TalkingGaussian:/workspace -w /workspace \
    idinteraction/openface -lc \
    "/idinteraction/OpenFace/build/bin/FeatureExtraction -fdir data/may/ori_imgs -aus -of data/may/au.csv -q"
  ```
- `mmcv-full` CUDA ops 포함 설치 (mmcv._ext 필요):
  ```bash
  CUDA_HOME=/usr/local/cuda MMCV_WITH_OPS=1 FORCE_CUDA=1 \
  pip install --no-build-isolation --no-binary mmcv-full --force-reinstall mmcv-full==1.7.2
  ```
- 누락 패키지:
  ```bash
  pip install prettytable
  ```
- torch 2.6 체크포인트 로딩 호환:
  - `data_utils/easyportrait/create_teeth_mask.py`에 `torch.load(..., weights_only=False)` 강제

### 9.3 최종 전처리 체크리스트
- `data/may/au.csv` (OpenFace)
- `data/may/teeth_mask/*.npy` (EasyPortrait)
- `data/may/transforms_*.json`, `track_params.pt` 등 기존 전처리 산출물
