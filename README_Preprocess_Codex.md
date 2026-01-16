# TalkingGaussian 전처리 가이드

이 문서는 전처리 파이프라인을 간단하고 명료하게 정리한 문서다.

## 1) 전처리 기본 흐름
1. 영상 → 오디오 추출
2. 오디오 특징 추출
3. 프레임 추출
4. 파싱(semantic) 생성
5. 배경 추출
6. torso/gt 이미지 생성
7. 랜드마크 추출
8. face tracking
9. transforms 저장
10. AU(OpenFace) 추출(별도 단계)

## 2) 공통 환경
- 작업 경로: `/home/hanati/code/TalkingGaussian`
- 사용 이미지: `talking-gaussian:preprocess-fixed-v2`
- GPU: RTX4090 (TORCH_CUDA_ARCH_LIST=8.9+PTX)

## 3) 전체 전처리 실행 (task -1)
```bash
docker run --gpus all --rm -v /home/hanati/code/TalkingGaussian:/workspace talking-gaussian:preprocess-fixed-v2 \
  bash -lc "cd /workspace && \
  export TORCH_CUDA_ARCH_LIST=8.9+PTX; export CUDA_VISIBLE_DEVICES=0; \
  PYTHONUNBUFFERED=1 stdbuf -oL python data_utils/process.py data/test/test_512.mp4 --task -1 \
  2>&1 | tee data/test/preprocess_full.log"
```

## 4) 필수 산출물 확인
```bash
ls -la data/test
```
필수:
- `aud.wav`, `aud.npy` (또는 `aud_ds.npy`)
- `ori_imgs/`, `parsing/`, `gt_imgs/`, `torso_imgs/`
- `track_params.pt`
- `transforms_train.json`, `transforms_val.json`

## 5) AU(OpenFace) 추출 (수동)
`process.py`에 OpenFace 호출이 포함되어 있지 않으므로 별도 수행 필요.
```bash
docker run --rm --entrypoint /bin/bash -v /home/hanati/code/TalkingGaussian:/workspace idinteraction/openface \
  -lc "/idinteraction/OpenFace/build/bin/FeatureExtraction -fdir /workspace/data/test/ori_imgs -aus -of /workspace/data/test/au.csv -q"
```

## 6) 문제 발생 시
- SSL 인증서 오류: resnet/s3fd/alexnet 다운로드에서 발생할 수 있음 (SSL 우회 적용됨)
- face tracking GPU 실패: 컨테이너별 torch CUDA 아키텍처 지원 여부 확인 필요
- AU 누락: OpenFace 수동 실행 필요
