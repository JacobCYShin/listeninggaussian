# Diff-Listener: 확산 기반 가우시안 스플래팅을 이용한 확률적 3D 청자 생성

이 저장소는 확산 기반 모션 생성과 고화질 3D Gaussian Splatting 렌더링을 결합한 **Diff-Listener**를 구현합니다.

**English**: [README.md](README.md)

## 개요

Diff-Listener는 다중 모달 입력(화자 오디오 및 시각적 표정)을 두 단계 아키텍처로 처리하여 자연스럽고 다양한 청자 반응을 생성합니다:

- **Brain (모션 생성기)**: 화자 오디오 및 시각적 특징에서 FLAME 파라미터를 생성하는 확산 기반 모델
- **Body (렌더러)**: FLAME 파라미터에서 고화질 얼굴 애니메이션을 생성하는 수정된 TalkingGaussian 기반 3D Gaussian Splatting 렌더러

## 아키텍처

### Brain: 확률적 모션 생성기

모션 생성기는 다음을 처리하는 1D 확산 모델입니다:
- **화자 오디오**: Wav2Vec 2.0 특징 (768차원)
- **화자 시각 정보**: EMOCA 표현 코드 (50차원)
- **청자 이력**: 이전 프레임 모션 (시간적 일관성)

**출력**: 청자 FLAME 파라미터 (표현 50 + 포즈 6 = 56차원)

### Body: 고화질 3D 렌더러

렌더러는 오디오 특징 대신 FLAME 파라미터를 입력으로 받도록 수정된 TalkingGaussian을 기반으로 합니다. 자연스러운 청자 반응에 필수적인 미세한 표정 제어를 유지합니다.

## 주요 특징

- **고화질 렌더링**: 3D Gaussian Splatting을 통해 2D 워핑 아티팩트 없이 512x512 이상의 해상도 달성
- **다양한 반응**: 확산 기반 생성으로 동일한 입력에 대해 다양한 자연스러운 반응 생성
- **다중 모달 인식**: 맥락 인식 반응을 위해 오디오와 시각적 화자 정보 모두 통합
- **2단계 학습**: 안정성을 위한 렌더러와 모션 생성기의 분리된 학습

## 설치

Ubuntu 18.04/22.04, CUDA 11.3, PyTorch 1.12.1에서 테스트됨

```bash
git clone <repository-url> --recursive
conda env create --file environment.yml
conda activate talking_gaussian
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install tensorflow-gpu==2.8.0
```

### 의존성

서브모듈 및 모델 준비:

```bash
bash scripts/prepare.sh
```

얼굴 파싱(EasyPortrait)용:

```bash
conda activate talking_gaussian
pip install -U openmim
mim install mmcv-full==1.7.1
cd data_utils/easyportrait
wget "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/easyportrait/experiments/models/fpn-fp-512.pth"
```

## 데이터 준비

### Body 학습 데이터

청자 신원 학습을 위해 비디오 준비 (3-5분, 25 FPS, 512x512 해상도):

1. 프레임 및 랜드마크 추출:
```bash
python data_utils/process.py data/<ID>/<ID>.mp4
```

2. EMOCA를 사용하여 FLAME 파라미터 추출:
```bash
# 청자 시각 정보 (GT) 추출: [Total_Frames, 56]
# Expression (50) + Pose (6: Jaw + Neck + Head)
```

### Brain 학습 데이터 (ViCo 데이터셋)

1. 화자 오디오 특징 추출 (Wav2Vec 2.0):
   - 출력: `[Total_Frames, 768]` numpy 배열

2. 화자 시각 특징 추출 (EMOCA):
   - 출력: `[Total_Frames, 50]` numpy 배열 (Expression만)

3. 청자 GT 추출 (EMOCA):
   - 출력: `[Total_Frames, 56]` numpy 배열

**중요**: 세 가지 모달리티 모두에서 프레임 동기화를 확인하세요.

## 학습

### Stage 1: Body (렌더러) 학습

FLAME 파라미터에서 얼굴을 재구성하도록 3D Gaussian Splatting 렌더러 학습:

```bash
# 변형 네트워크를 오디오 대신 FLAME 파라미터를 받도록 수정
# GT FLAME 파라미터 -> 원본 이미지로 학습
bash scripts/train_xx.sh data/<ID> output/<project_name> <GPU_ID>
```

### Stage 2: Brain (모션 생성기) 학습

다중 모달 입력에서 FLAME 파라미터를 생성하도록 확산 모델 학습:

```bash
# 확산 모델 학습: (오디오, 시각) -> FLAME 파라미터
# 손실: 파라미터 공간의 MSE
python train_diffusion.py --config configs/brain_config.yaml
```

## 추론

화자 오디오 및 시각 특징에서 청자 헤드 비디오 생성:

```bash
python synthesize_listener.py \
    --audio <speaker_audio_features>.npy \
    --visual <speaker_visual_features>.npy \
    --body_checkpoint output/<project_name>/chkpnt_fuse_latest.pth \
    --brain_checkpoint output/brain_model/checkpoint.pth \
    --output output/listener_video.mp4
```

## 프로젝트 구조

```
├── scene/
│   ├── gaussian_model.py          # 3D Gaussian 모델
│   ├── motion_net.py              # 모션 네트워크 (레거시, 수정 예정)
│   └── dataset_readers.py         # 데이터 로딩
├── gaussian_renderer/             # 렌더링 파이프라인
├── data_utils/                    # 데이터 전처리
├── study/                         # 문서 및 학습 자료
└── scripts/                       # 학습 및 유틸리티 스크립트
```

## 개발 계획

자세한 개발 단계 및 구현 가이드라인은 `개발계획_v2.md`를 참조하세요.

## 연구 방향

연구 동기, 방법론 및 기여는 `연구방향.md`를 참조하세요.

## 인용

연구에 이 코드를 사용하는 경우 다음을 인용해 주세요:

```bibtex
@article{difflistener2025,
  title={Diff-Listener: Probabilistic 3D Listening Head Generation via Diffusion-based Gaussian Splatting},
  author={...},
  journal={...},
  year={2025}
}
```

## 감사의 글

이 작업은 다음을 기반으로 합니다:
- [TalkingGaussian](https://github.com/Fictionarry/TalkingGaussian) (ECCV 2024)
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [EMOCA](https://github.com/radekd91/emoca) (FLAME 파라미터 추출용)

## 라이선스

이 코드는 연구 목적으로만 제공됩니다. 라이선스 세부사항은 원본 TalkingGaussian 저장소를 참조하세요.

