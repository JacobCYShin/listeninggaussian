# TalkingGaussian 전체 아키텍처 개요

## 프로젝트 소개

TalkingGaussian은 **3D Gaussian Splatting** 기반의 Talking Head Synthesis 시스템입니다.
오디오 입력을 받아서 얼굴이 말하는 듯한 3D 비디오를 생성합니다.

**핵심 논문**: [TalkingGaussian: Structure-Persistent 3D Talking Head Synthesis via Gaussian Splatting (ECCV 2024)](https://arxiv.org/abs/2404.15264)

## 핵심 아이디어

1. **Static 3D Gaussian Representation**: 얼굴을 3D Gaussian 포인트 클라우드로 표현
2. **Motion Network**: 오디오 특징 → Gaussian 변형 (이동, 회전, 스케일)
3. **이중 모델 구조**: 
   - **Face Model**: 얼굴 전체 움직임 제어 (눈 깜빡임, 전체 표정)
   - **Mouth Model**: 입 부분 세밀한 움직임 제어 (입 모양)
4. **Gaussian Splatting Rendering**: 변형된 Gaussian을 이미지로 렌더링

## 전체 파이프라인

```
[비디오 입력]
    ↓
[데이터 전처리]
    ├─ 프레임 추출
    ├─ 얼굴 랜드마크 추출
    ├─ Face Parsing
    ├─ Action Units (AU) 추출
    └─ 오디오 특징 추출
    ↓
[학습 단계]
    ├─ Gaussian 모델 학습 (얼굴 3D 구조)
    ├─ Face Motion Network 학습
    └─ Mouth Motion Network 학습
    ↓
[추론 단계]
    ├─ [오디오 입력] 
    ├─ [오디오 특징 추출] (DeepSpeech/HuBERT)
    ├─ [Motion Network] → [Gaussian 변형] (위치, 회전, 스케일)
    ├─ [Gaussian Splatting 렌더링]
    └─ [최종 이미지/비디오]
```

## 주요 컴포넌트

### 1. 데이터 전처리 (`data_utils/process.py`)
- 비디오에서 프레임 추출 (25 FPS)
- 얼굴 랜드마크 추출 (Face Alignment, 68 points)
- Face Parsing (얼굴 영역 분할: 얼굴, 목, 몸통, 배경)
- Action Units (AU) 추출 (OpenFace)
- 오디오 특징 추출 (DeepSpeech/HuBERT)
- 배경 이미지 생성
- Torso 이미지 생성

### 2. 데이터셋 (`scene/dataset_readers.py`)
- 카메라 정보 로드 (`transforms_train.json`, `transforms_val.json`)
- 각 프레임별 정보:
  - 오디오 특징 (`auds`)
  - Action Units (`au_exp`, `blink`, `au25`)
  - 마스크 정보 (`face_mask`, `mouth_mask`, `hair_mask`)
  - 랜드마크 정보 (`lips_rect`, `lhalf_rect`)
- `CameraInfo` 구조로 저장

### 3. Gaussian 모델 (`scene/gaussian_model.py`)
3D Gaussian 파라미터 관리:
- `_xyz`: 위치 (N, 3)
- `_features_dc`, `_features_rest`: Spherical Harmonics 색상
- `_scaling`: 스케일 (N, 3)
- `_rotation`: 회전 quaternion (N, 4)
- `_opacity`: 투명도 (N, 1)
- `_identity`: (학습/추론에 사용)

### 4. Motion Network (`scene/motion_net.py`)
- **MotionNetwork**: 얼굴 전체 움직임
  - 입력: 오디오 특징 + AU (Action Units)
  - 출력: `d_xyz`, `d_rot`, `d_scale`, `d_opa`
- **MouthMotionNetwork**: 입 부분 움직임
  - 입력: 오디오 특징
  - 출력: `d_xyz` (위치 변형만)

### 5. 렌더링 (`gaussian_renderer/__init__.py`)
- `render_motion`: Face 모델 렌더링
- `render_motion_mouth`: Mouth 모델 렌더링
- 두 결과를 Alpha Blending으로 합성

## 추론 파이프라인 (핵심!)

**엔트리 포인트**: `synthesize_fuse.py`

### 전체 흐름

```python
# 1. 모델 로드
gaussians = GaussianModel(...)              # Face Gaussian
gaussians_mouth = GaussianModel(...)        # Mouth Gaussian
motion_net = MotionNetwork(...)             # Face Motion Network
motion_net_mouth = MouthMotionNetwork(...)  # Mouth Motion Network

# 2. 체크포인트 로드
checkpoint = torch.load("chkpnt_fuse_latest.pth")
# 구조: (model_params, motion_params, model_mouth_params, motion_mouth_params)

# 3. 각 카메라 뷰에 대해 렌더링
for view in cameras:
    # 4. Face 렌더링
    render_face = render_motion(view, gaussians, motion_net, ...)
    # Motion Network가 Gaussian을 변형 → 렌더링
    
    # 5. Mouth 렌더링
    render_mouth = render_motion_mouth(view, gaussians_mouth, motion_net_mouth, ...)
    
    # 6. 합성
    final_image = render_face + render_mouth * (1 - render_face.alpha)
```

### 데이터 흐름

1. **전처리 단계**:
   - `data/<ID>/<ID>.mp4` 
   - → 프레임, 랜드마크, 파싱, AU, 오디오 특징
   - → `transforms_train.json`, `transforms_val.json` 생성
   
2. **학습 단계**:
   - 각 프레임에서 Gaussian 파라미터 학습
   - Motion Network 학습 (오디오 → 변형)
   
3. **추론 단계**:
   - 새로운 오디오 특징 
   - → Motion Network 
   - → Gaussian 변형 (d_xyz, d_rot, d_scale)
   - → 렌더링
   - → 최종 이미지

## 파일 구조

```
TalkingGaussian/
├── synthesize_fuse.py          # 추론 엔트리 포인트 ⭐
├── train_face.py               # Face 모델 학습
├── train_mouth.py              # Mouth 모델 학습
├── train_fuse.py               # 최종 통합 학습
├── data_utils/
│   ├── process.py              # 데이터 전처리 ⭐
│   ├── deepspeech_features/    # DeepSpeech 특징 추출
│   ├── hubert.py               # HuBERT 특징 추출
│   └── face_tracking/          # 얼굴 추적
├── scene/
│   ├── __init__.py             # Scene 클래스
│   ├── dataset_readers.py      # 데이터셋 로더 ⭐
│   ├── gaussian_model.py       # Gaussian 모델 ⭐
│   ├── motion_net.py           # Motion Network ⭐
│   └── cameras.py              # 카메라 클래스
└── gaussian_renderer/
    └── __init__.py             # 렌더링 함수 ⭐
```

## 핵심 개념

### 1. 3D Gaussian Splatting
- 3D 공간의 각 포인트를 Gaussian으로 표현
- 각 Gaussian은 위치, 색상(SH), 스케일, 회전, 투명도를 가짐
- 렌더링 시 각 Gaussian을 2D로 투영하여 합성

### 2. Motion Network
- 오디오 특징을 입력으로 받아 Gaussian 변형을 예측
- Hash Grid Encoding 사용 (빠른 위치 인코딩)
- 공간적으로 제한된 영역만 변형 (bound = 0.15)

### 3. 이중 모델 구조
- **Face Model**: 큰 영역 (얼굴 전체)
- **Mouth Model**: 작은 영역 (입 부분)
- 두 모델을 Alpha Blending으로 합성

## 학습 과정

1. **Face 모델 학습** (`train_face.py`):
   - 얼굴 전체 Gaussian 학습
   - Face Motion Network 학습
   
2. **Mouth 모델 학습** (`train_mouth.py`):
   - 입 부분 Gaussian 학습
   - Mouth Motion Network 학습
   
3. **통합 학습** (`train_fuse.py`):
   - 두 모델을 함께 fine-tuning

## 다음 단계

각 컴포넌트에 대한 상세 분석:
- `01_Inference_Pipeline.md`: 추론 코드 상세 분석
- `02_Data_Preprocessing.md`: 데이터 전처리 과정
- `03_Dataset_Loader.md`: 데이터 로딩 메커니즘
- `04_Gaussian_Model.md`: Gaussian 모델 구조
- `05_Motion_Network.md`: Motion Network 아키텍처
- `06_Rendering.md`: 렌더링 과정

