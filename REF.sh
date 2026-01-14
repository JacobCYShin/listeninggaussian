 docker run --gpus all -it   -v $(pwd):/workspace   --entrypoint /bin/bash talking-gaussian

docker exec -it <컨테이너이름 또는 ID> /bin/bash
docker exec -it 04a5 /bin/bash

source /opt/conda/etc/profile.d/conda.sh
conda activate GaussianTalker

bash scripts/train_xx.sh data/macron output/macron_project 0

export TORCH_CUDA_ARCH_LIST=8.6+PTX
python train_mouth.py -s data/macron -m output/macron_project --audio_extractor deepspeech


docker exec -it 3bdbcd343309 /bin/bash
export TORCH_HOME=/workspace/.cache/torch
bash scripts/train_xx.sh data/macron output/macron_project 0
bash scripts/train_xx.sh data/may output/may_project 0

tensorboard --logdir output/macron_project

# may inference (fuse)
python synthesize_fuse.py -s data/may -m output/may_project --fast

# mux audio into rendered video (example for macron)
ffmpeg -i output/macron_project/test/ours_None/renders/out.mp4 -i data/macron/aud.wav -c:v copy -c:a aac -shortest output/macron_project/test/ours_None/renders/out_with_audio.mp4

# 20s audio + DeepSpeech features (macron example)
ffmpeg -y -ss 0 -t 20 -i data/macron/aud.wav -ac 1 -ar 16000 data/macron/aud_20s.wav
python data_utils/deepspeech_features/extract_ds_features.py --input data/macron/aud_20s.wav --output data/macron/aud_20s_ds.npy

# render test (video only) with 20s ds features
export TORCH_HOME=/workspace/.cache/torch
python synthesize_fuse.py -S data/macron -M output/macron_project --eval --fast --audio data/macron/aud_20s_ds.npy
python synthesize_fuse.py -S data/may -M output/may_project --eval --fast --audio data/may/aud_20s_ds.npy

# mux audio into rendered test video
ffmpeg -y -i output/macron_project/test/ours_None/renders/out.mp4 -i data/macron/aud_20s.wav -c:v copy -c:a aac -shortest output/macron_project/test/ours_None/renders/out_with_audio.mp4
ffmpeg -y -i output/may_project/test/ours_None/renders/out.mp4 -i data/may/aud_20s.wav -c:v copy -c:a aac -shortest output/may_project/test/ours_None/renders/out_with_audio.mp4
