TalkingGaussian 작업 메모 (EMOCA/전처리/학습/추론)

0) 공통
- 작업 루트: /home/hanati/code/TalkingGaussian
- 데이터 이름: acting
- 원본 비디오(25fps 변환본): data/acting/acting_challenge_25fps.mp4

1) 컨테이너 시작
docker start tg113
docker start emoca-preprocess

2) 전처리 (tg113 컨테이너)
docker exec -it tg113 /bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate GaussianTalker

전체 전처리(오디오/프레임/파싱/배경/랜드마크/트래킹/트랜스폼):
python data_utils/process.py "/home/hanati/code/TalkingGaussian/data/acting/acting_challenge_25fps.mp4" --task -1

트래킹만 재실행 시:
export TORCH_CUDA_ARCH_LIST=8.9+PTX
python data_utils/process.py "/home/hanati/code/TalkingGaussian/data/acting/acting_challenge_25fps.mp4" --task 8
python data_utils/process.py "/home/hanati/code/TalkingGaussian/data/acting/acting_challenge_25fps.mp4" --task 9

3) EMOCA 추출 (emoca-preprocess 컨테이너)
docker exec -it emoca-preprocess /bin/bash
export TORCH_CUDA_ARCH_LIST=8.9+PTX
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/workspace/emoca

EMOCA 코드 추출:
python scripts/emoca_extract_codes.py \
  --input_video "data/acting/acting_challenge_25fps.mp4" \
  --output_folder "output/emoca_acting" \
  --path_to_models "/workspace/emoca/assets/EMOCA/models" \
  --model_name "EMOCA_v2_lr_mse_20" \
  --mode detail \
  --batch_size 1 \
  --device cuda

flame_params.npy 생성(EMOCA exp+pose 결합):
python - <<'PY'
import glob, os, numpy as np
base = "/workspace/output/emoca_acting"
dirs = sorted([d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d)])
out = [np.concatenate([np.load(os.path.join(d, "exp.npy")).reshape(-1)[:50],
                       np.load(os.path.join(d, "pose.npy")).reshape(-1)[:6]]) for d in dirs]
arr = np.stack(out, axis=0)
out_path = "/workspace/data/acting/flame_params.npy"
np.save(out_path, arr)
mean = arr.mean(axis=0)
std = arr.std(axis=0)
np.savez("/workspace/data/acting/flame_params_stats.npz", mean=mean, std=std)
print(out_path, arr.shape)
PY

4) 학습 (tg113 컨테이너)
docker exec -it tg113 /bin/bash
export TORCH_HOME=/workspace/.cache/torch
export TORCH_CUDA_ARCH_LIST=8.9+PTX
python train_face.py -s "data/acting" -m "output/acting_project_flame" --audio_extractor flame

5) 추론 (tg113 컨테이너)
export TORCH_HOME=/workspace/.cache/torch
python synthesize_face.py -s "data/acting" -m "output/acting_project_flame" --fast

6) 오디오 합치기(선택)
ffmpeg -i "output/acting_project_flame/test/ours_None/renders/out.mp4" -i "data/acting/aud.wav" -c:v copy -c:a aac -shortest "output/acting_project_flame/test/ours_None/renders/out_with_audio.mp4"
