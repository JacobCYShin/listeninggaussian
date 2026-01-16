# TalkingGaussian 실행 가이드 (전처리/OpenFace/EMOCA/학습/추론)

작업 경로: `/home/hanati/code/TalkingGaussian`

## 0) 사용자 입력 템플릿
아래 변수를 실제 값으로 바꿔서 사용한다.

```bash
DATA_NAME="<데이터이름>"                 # 예: pds
VIDEO_PATH="data/<데이터이름>/<영상>.mp4" # 예: data/pds/pds.mp4
EMOCA_OUT="output/emoca_<데이터이름>"     # 예: output/emoca_pds
MODEL_OUT="output/<프로젝트명>"           # 예: output/pds_project_flame
NUM_FRAMES="<전체프레임수>"               # 예: 635
```

## 1) 전처리 (process.py, 전체)
### 1.1 템플릿
```bash
docker run --gpus all --rm -v /home/hanati/code/TalkingGaussian:/workspace talking-gaussian:preprocess-fixed-v2 \
  bash -lc "cd /workspace && \
  export TORCH_CUDA_ARCH_LIST=8.9+PTX; export CUDA_VISIBLE_DEVICES=0; \
  PYTHONUNBUFFERED=1 stdbuf -oL python data_utils/process.py ${VIDEO_PATH} --task -1 \
  2>&1 | tee data/${DATA_NAME}/preprocess_full.log"
```

### 1.2 예시
```bash
docker run --gpus all --rm -v /home/hanati/code/TalkingGaussian:/workspace talking-gaussian:preprocess-fixed-v2 \
  bash -lc "cd /workspace && \
  export TORCH_CUDA_ARCH_LIST=8.9+PTX; export CUDA_VISIBLE_DEVICES=0; \
  PYTHONUNBUFFERED=1 stdbuf -oL python data_utils/process.py data/pds/pds.mp4 --task -1 \
  2>&1 | tee data/pds/preprocess_full.log"
```

### 입력/출력
입력: `data/<데이터이름>/<영상>.mp4`  
출력(예: `data/pds/`):
- `aud.wav`, `aud.npy`(또는 `aud_ds.npy`)
- `ori_imgs/`, `parsing/`, `gt_imgs/`, `torso_imgs/`
- `track_params.pt`
- `transforms_train.json`, `transforms_val.json`

## 2) OpenFace AU 추출 (au.csv)
### 2.1 템플릿
```bash
docker run --rm --entrypoint /bin/bash -v /home/hanati/code/TalkingGaussian:/workspace idinteraction/openface \
  -lc "/idinteraction/OpenFace/build/bin/FeatureExtraction -fdir /workspace/data/${DATA_NAME}/ori_imgs -aus -of /workspace/data/${DATA_NAME}/au.csv -q"
```

### 2.2 예시
```bash
docker run --rm --entrypoint /bin/bash -v /home/hanati/code/TalkingGaussian:/workspace idinteraction/openface \
  -lc "/idinteraction/OpenFace/build/bin/FeatureExtraction -fdir /workspace/data/pds/ori_imgs -aus -of /workspace/data/pds/au.csv -q"
```

### 입력/출력
입력: `data/<데이터이름>/ori_imgs/*.jpg`  
출력: `data/<데이터이름>/au.csv`

## 3) EMOCA 추출 (flame_params 자동 생성)
### 3.1 템플릿
```bash
docker exec -it emoca-preprocess bash -lc "cd /workspace && \
export TORCH_CUDA_ARCH_LIST=8.9+PTX; export CUDA_VISIBLE_DEVICES=0; export PYTHONPATH=/workspace/emoca; \
python scripts/emoca_extract_codes.py \
  --input_video ${VIDEO_PATH} \
  --output_folder ${EMOCA_OUT} \
  --path_to_models /workspace/emoca/assets/EMOCA/models \
  --model_name EMOCA_v2_lr_mse_20 \
  --mode detail \
  --batch_size 1 \
  --device cuda"
```

### 3.2 예시
```bash
docker exec -it emoca-preprocess bash -lc "cd /workspace && \
export TORCH_CUDA_ARCH_LIST=8.9+PTX; export CUDA_VISIBLE_DEVICES=0; export PYTHONPATH=/workspace/emoca; \
python scripts/emoca_extract_codes.py \
  --input_video data/pds/pds.mp4 \
  --output_folder output/emoca_pds \
  --path_to_models /workspace/emoca/assets/EMOCA/models \
  --model_name EMOCA_v2_lr_mse_20 \
  --mode detail \
  --batch_size 1 \
  --device cuda"
```

### 입력/출력
입력: `data/<데이터이름>/<영상>.mp4`  
출력:
- `output/emoca_<데이터이름>/*/exp.npy`, `pose.npy`  
- 자동 생성: `data/<데이터이름>/flame_params.npy`, `data/<데이터이름>/flame_params_stats.npz`

## 4) 학습 (FLAME)
### 4.0 전처리 산출물 체크
```bash
python scripts/check_preprocess.py --data ${DATA_NAME}
```

### 4.1 템플릿
```bash
docker exec -it tg113 bash -lc "cd /workspace && \
export TORCH_HOME=/workspace/.cache/torch; export TORCH_CUDA_ARCH_LIST=8.9+PTX; \
python train_face.py -s data/${DATA_NAME} -m ${MODEL_OUT} --audio_extractor flame"
```

### 4.2 예시
```bash
docker exec -it tg113 bash -lc "cd /workspace && \
export TORCH_HOME=/workspace/.cache/torch; export TORCH_CUDA_ARCH_LIST=8.9+PTX; \
python train_face.py -s data/pds -m output/pds_project_flame --audio_extractor flame"
```

### 입력/출력
입력:
- `data/<데이터이름>/transforms_*.json`
- `data/<데이터이름>/flame_params.npy`
- `data/<데이터이름>/au.csv`
출력:
- `output/<프로젝트명>/` (체크포인트/로그/결과)

## 5) 추론
### 5.1 학습 데이터 기준 (train 프레임)
#### 템플릿
```bash
docker exec -it tg113 bash -lc "cd /workspace && \
export TORCH_HOME=/workspace/.cache/torch; \
python synthesize_face.py -s data/${DATA_NAME} -m ${MODEL_OUT} --use_train --audio_extractor flame"
```

#### 예시
```bash
docker exec -it tg113 bash -lc "cd /workspace && \
export TORCH_HOME=/workspace/.cache/torch; \
python synthesize_face.py -s data/pds -m output/pds_project_flame --use_train --audio_extractor flame"
```

### 5.2 전체 프레임 지정
#### 템플릿
```bash
docker exec -it tg113 bash -lc "cd /workspace && \
export TORCH_HOME=/workspace/.cache/torch; \
python synthesize_face.py -s data/${DATA_NAME} -m ${MODEL_OUT} --audio_extractor flame --start_idx 0 --num_frames ${NUM_FRAMES}"
```

#### 예시
```bash
docker exec -it tg113 bash -lc "cd /workspace && \
export TORCH_HOME=/workspace/.cache/torch; \
python synthesize_face.py -s data/pds -m output/pds_project_flame --audio_extractor flame --start_idx 0 --num_frames 635"
```

### 입력/출력
입력:
- `data/<데이터이름>/transforms_*.json`
- `data/<데이터이름>/flame_params.npy`
출력:
- `output/<프로젝트명>/pds/ours_None/renders/out.mp4`

## 6) 오디오 합치기 (선택)
### 템플릿
```bash
ffmpeg -i ${MODEL_OUT}/pds/ours_None/renders/out.mp4 \
  -i data/${DATA_NAME}/aud.wav -c:v copy -c:a aac -shorpds \
  ${MODEL_OUT}/pds/ours_None/renders/out_with_audio.mp4
```

### 예시
```bash
ffmpeg -i output/pds_project_flame/pds/ours_None/renders/out.mp4 \
  -i data/pds/aud.wav -c:v copy -c:a aac -shorpds \
  output/pds_project_flame/pds/ours_None/renders/out_with_audio.mp4
```
