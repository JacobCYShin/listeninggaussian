# EMOCA/FLAME 진행 기록 (설치 + 디버깅 + 결과)

이 문서는 EMOCA/FLAME 추출을 위해 수행한 설치, 디버깅, 실행 결과를 정리한 기록입니다.

---

## 1) 환경 구성

- 작업 위치: `\\wsl$\Ubuntu-22.04\home\hanati\code\TalkingGaussian`
- 데이터 전처리 컨테이너: `emoca-preprocess`
- 베이스 이미지: `talking-gaussian:preprocess-fixed-v2`
- GPU 사용: 컨테이너 실행 시 `--gpus all`

### 컨테이너 생성/접속 요약

- 새 컨테이너로 EMOCA 설치(기존 이미지 환경 오염 방지).
- 내부 경로 기준으로 작업.

---

## 2) EMOCA 설치/설정

### 2.1 소스 코드

- EMOCA repo: `https://github.com/radekd91/emoca`
- 컨테이너 내부 설치 위치: `/workspace/emoca`

### 2.2 Conda 환경

- Miniconda 설치 위치: `/opt/conda`
- Conda env 이름: `work38`
- Python 3.8 기반으로 구성

### 2.3 PyTorch 및 라이브러리

- PyTorch: 1.12.1
- torchvision: 0.13.1
- torchaudio: 0.12.1

설치 과정에서 다음 파일 수정/설정:

- `conda-environment_py38_cu11_ubuntu.yml`에 맞춰 설치
- `requirements38.txt` 수정:
  - `onnxruntime-gpu==1.13.1`
  - `mediapipe==0.10.11`
  - `flatbuffers>=2.0`
  - `mmcv-full` 주석 처리(빌드 이슈)

추가 설치:

- `opencv-python==4.5.5.64`로 교체 (opencv-contrib 제거)
- `numpy==1.23.5`로 다운그레이드
- `pytorch3d`는 로컬 wheel로 설치:
  - `/workspace/wheels/pytorch3d-0.7.2-cp38-cp38-linux_x86_64.whl`
- `pip` 버전 24.0으로 고정

### 2.4 EMOCA 패키지 설치

- `pip install -e /workspace/emoca`

---

## 3) EMOCA Asset 다운로드/라이선스 처리

### 3.1 모델/에셋 다운로드

- EMOCA asset은 라이선스 동의가 필요함.
- 자동 입력 시 반복 "Please answer yes or no" 문제가 발생해서
  사용자가 직접 동의 입력으로 해결.

### 3.2 SSL 인증서 문제

사내 인증서로 인해 기본 다운로드가 실패:

- 해결: `wget --no-check-certificate`

다운로드/추출 위치:

- `/workspace/emoca/assets/DECA`
- `/workspace/emoca/assets/EMOCA/models`
- `/workspace/emoca/assets/FLAME`
- `/workspace/emoca/assets/FaceRecognition`

---

## 4) 추가 파일 캐시/모델 다운로드 문제 해결

### 4.1 face_alignment 모델 다운로드 오류

SSL 문제로 자동 다운로드 실패.

해결:

- `s3fd` / `2DFAN4` 모델을 로컬 파일로 준비 후 캐시에 복사
  - `/root/.cache/torch/hub/checkpoints/s3fd-619a316812.pth`
  - `/root/.cache/torch/hub/checkpoints/2DFAN4-cd938726ad.zip`

### 4.2 EMOCA 모델 cfg.yaml 0 byte 문제

`/workspace/emoca/assets/EMOCA/models/EMOCA_v2_lr_mse_20/cfg.yaml`이 0 bytes여서
`OmegaConf.load` 실패.

해결:

- 기존 폴더 삭제 후 zip 재압축 해제
- `cfg.yaml` 정상 복구 확인 (9KB 이상)

### 4.3 VGG/ResNet 사전학습 가중치 다운로드 SSL 오류

- `vgg19-dcbb9e9d.pth`, `resnet50-0676ba61.pth` 다운로드 시
  SSL 인증서 오류 발생.

해결:

- `wget --no-check-certificate`로 직접 다운로드하여 캐시에 저장
  - `/root/.cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth`
  - `/root/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth`

---

## 5) 코드 수정 (컨테이너 내부)

### 5.1 face_alignment LandmarksType 호환

`LandmarksType._2D` 사용 불가 문제 발생.

해결:

- `/workspace/emoca/gdl/utils/FaceDetector.py`에서
  `LandmarksType._2D` -> `LandmarksType.TWO_D`로 변경

### 5.2 DataLoader shm 오류 해결

`ERROR: Unexpected bus error encountered in worker.`

해결:

- `num_workers=4` -> `num_workers=0`
- 수정 파일:
  - `/workspace/emoca/gdl_apps/EMOCA/demos/test_emoca_on_video.py`

---

## 6) EMOCA 실행 로그 및 결과

### 6.1 입력 비디오

- Macron 원본: `/workspace/data/macron/macron.mp4`
- 길이: 약 5분 50초

### 6.2 실행 명령

```
docker exec emoca-preprocess /opt/conda/bin/mamba run -n work38 \
python /workspace/emoca/gdl_apps/EMOCA/demos/test_emoca_on_video.py \
  --input_video /workspace/data/macron/macron.mp4 \
  --output_folder /workspace/output/emoca_macron \
  --model_name EMOCA_v2_lr_mse_20 \
  --save_codes True --save_images False --save_mesh False \
  --include_original False --include_rec False --include_transparent False \
  --processed_subfolder processed_2026_Jan_14_02-34-04
```

### 6.3 실행 중 오류 및 처리

- 초기 실행은 타임아웃/SSL 문제로 중단됨.
- `nohup` + 로그 파일로 백그라운드 실행.

마지막 종료 오류:

- `IndexError: list index out of range` (영상 렌더링 단계)
- 원인: `save_images=False` 설정으로 이미지가 없어 영상 생성 실패.
- 하지만 `exp.npy`, `pose.npy`는 정상 저장됨.

---

## 7) EMOCA 출력 확인

### 7.1 출력 위치

- 결과 폴더:  
  `/workspace/output/emoca_macron/EMOCA_v2_lr_mse_20/`

각 프레임별 폴더:

- `000001_000`, `000002_000`, ... `008732_000`
- 각 폴더 안에 `exp.npy`, `pose.npy`

### 7.2 프레임 수 확인

- 총 프레임 수: 8732
- exp.npy 개수: 8732개

---

## 8) FLAME 파라미터 생성

### 8.1 생성 규칙

- `exp.npy` (50) + `pose.npy` (6) 결합
- 최종 shape: `[8732, 56]`

### 8.2 저장 위치

- `data/macron/flame_params.npy`

### 8.3 생성 명령 (WSL)

```
python3 -c "import os, glob, numpy as np; \
root='/home/hanati/code/TalkingGaussian/output/emoca_macron/EMOCA_v2_lr_mse_20'; \
dirs=[d for d in glob.glob(os.path.join(root,'*')) if os.path.isdir(d) and os.path.basename(d).split('_')[0].isdigit()]; \
dirs=sorted(dirs); \
out=[np.concatenate([np.load(os.path.join(d,'exp.npy')).reshape(-1)[:50], np.load(os.path.join(d,'pose.npy')).reshape(-1)[:6]]) for d in dirs]; \
arr=np.stack(out, axis=0); \
out_path='/home/hanati/code/TalkingGaussian/data/macron/flame_params.npy'; \
np.save(out_path, arr); \
print(out_path, arr.shape)"
```

출력:

- `/home/hanati/code/TalkingGaussian/data/macron/flame_params.npy (8732, 56)`

---

## 9) 현재 상태 요약

- EMOCA 추출 완료 (exp/pose 전 프레임 정상 생성)
- FLAME 파라미터 파일 생성 완료
- 영상 렌더링 단계 에러는 무시 가능 (이미지 저장을 끈 상태였음)

---

## 10) 다음 단계 (Phase 1 준비)

- `flame_params.npy`로 TalkingGaussian Body 학습 준비
- `deformation_network.py` 입력 56-dim 변경
- 오버피팅 학습 후 재구성 테스트

---

## 부록: 참고 사항

- 컨테이너 내부에서 수정한 파일은 호스트와 분리되어 있음.
- 추후 재현 시에는 `num_workers=0` 상태 유지 권장.
- SSL 문제가 계속될 경우, 모든 모델 다운로드는 `--no-check-certificate`로 처리 필요.

## test_512 EMOCA/FLAME ���� (data/test/test_512.mp4)

### EMOCA ���� (emoca-preprocess)
```
docker exec -it emoca-preprocess bash -lc "cd /workspace && \
export TORCH_CUDA_ARCH_LIST=8.9+PTX; export CUDA_VISIBLE_DEVICES=0; export PYTHONPATH=/workspace/emoca; \
python scripts/emoca_extract_codes.py \
  --input_video data/test/test_512.mp4 \
  --output_folder output/emoca_test_512 \
  --path_to_models /workspace/emoca/assets/EMOCA/models \
  --model_name EMOCA_v2_lr_mse_20 \
  --mode detail \
  --batch_size 1 \
  --device cuda"
```

### FLAME �Ķ���� ����
```
python - <<'PY'
import glob
import os
import numpy as np
base = "/workspace/output/emoca_test_512"
dirs = sorted([d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d)])
valid = [d for d in dirs if os.path.exists(os.path.join(d, "exp.npy")) and os.path.exists(os.path.join(d, "pose.npy"))]
if not valid:
    raise SystemExit("No valid EMOCA frame dirs found")
out = [np.concatenate([np.load(os.path.join(d, "exp.npy")).reshape(-1)[:50],
                       np.load(os.path.join(d, "pose.npy")).reshape(-1)[:6]]) for d in valid]
arr = np.stack(out, axis=0)
np.save("/workspace/data/test/flame_params.npy", arr)
np.savez("/workspace/data/test/flame_params_stats.npz", mean=arr.mean(axis=0), std=arr.std(axis=0))
print(arr.shape)
PY
```

### ���
- `data/test/flame_params.npy`: shape (635, 56)
- `data/test/flame_params_stats.npz`

## EMOCA �ڵ� FLAME ���� ������Ʈ
- `scripts/emoca_extract_codes.py` ���� �� `flame_params.npy`/`flame_params_stats.npz`�� �ڵ� �����ϵ��� ����.
- �⺻ ��Ģ: `output/emoca_<DATA>` �����̸� `/workspace/data/<DATA>`�� ����.
- ���� ó��: ��Ī�Ǵ� data ������ ������ `output` ������ ����.
- ���� ���� �ɼ�: `--flame_out_dir /workspace/data/<DATA>`
