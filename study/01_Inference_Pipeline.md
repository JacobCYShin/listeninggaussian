# 추론 파이프라인 상세 분석

## 개요

`synthesize_fuse.py`가 추론의 엔트리 포인트입니다.
학습된 모델을 로드하여 새로운 비디오를 생성합니다.

## 코드 구조 분석

### 메인 함수: `render_sets()`

```88:111:synthesize_fuse.py
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, use_train : bool, fast, dilate):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        gaussians_mouth = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, shuffle=False)

        motion_net = MotionNetwork(args=dataset).cuda()
        motion_net_mouth = MouthMotionNetwork(args=dataset).cuda()

        (model_params, motion_params, model_mouth_params, motion_mouth_params) = torch.load(os.path.join(dataset.model_path, "chkpnt_fuse_latest.pth"))
        motion_net.load_state_dict(motion_params, strict=False)
        gaussians.restore(model_params, None)

        motion_net_mouth.load_state_dict(motion_mouth_params, strict=False)
        gaussians_mouth.restore(model_mouth_params, None)

        
        # motion_net.fix(gaussians.get_xyz.cuda())
        # motion_net_mouth.fix(gaussians_mouth.get_xyz.cuda())

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        render_set(dataset.model_path, "test" if not use_train else "train", scene.loaded_iter, scene.getTestCameras() if not use_train else scene.getTrainCameras(), gaussians, motion_net, gaussians_mouth, motion_net_mouth, pipeline, background, fast, dilate)
```

**핵심 포인트**:
- **이중 모델 구조**: Face 모델과 Mouth 모델이 분리되어 있음
- **체크포인트 구조**: 4개의 파라미터 그룹 저장
  - `model_params`: Face Gaussian 파라미터
  - `motion_params`: Face Motion Network 파라미터
  - `model_mouth_params`: Mouth Gaussian 파라미터
  - `motion_mouth_params`: Mouth Motion Network 파라미터
- `torch.no_grad()`: 추론 시 그래디언트 계산 불필요

### 렌더링 함수: `render_set()`

```35:85:synthesize_fuse.py
def render_set(model_path, name, iteration, views, gaussians, motion_net, gaussians_mouth, motion_net_mouth, pipeline, background, fast, dilate):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    all_preds = []
    all_gts = []

    all_preds_face = []
    all_preds_mouth = []


    for idx, view in enumerate(tqdm(views, desc="Rendering progress", ascii=True)):
        if view.original_image == None:
            view = loadCamOnTheFly(copy.deepcopy(view))
        with torch.no_grad():
            render_pkg = render_motion(view, gaussians, motion_net, pipeline, background, frame_idx=0)
            render_pkg_mouth = render_motion_mouth(view, gaussians_mouth, motion_net_mouth, pipeline, background, frame_idx=0)
        # gt = view.original_image[0:3, :, :]
        # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        if dilate:
            alpha_mouth = dilate_fn(render_pkg_mouth["alpha"][None])[0]
        else:
            alpha_mouth = render_pkg_mouth["alpha"]
            
        mouth_image = render_pkg_mouth["render"] + view.background.cuda() / 255.0 * (1.0 - alpha_mouth)

        # alpha = gaussian_blur(render_pkg["alpha"], [3, 3], 2)
        alpha = render_pkg["alpha"]
        image = render_pkg["render"] + mouth_image * (1.0 - alpha)

        pred = (image[0:3, ...].clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()* 255).astype(np.uint8)
        all_preds.append(pred)
        
        if not fast:
            all_preds_face.append((render_pkg["render"].clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()* 255).astype(np.uint8))
            all_preds_mouth.append((render_pkg_mouth["render"].clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()* 255).astype(np.uint8))

            all_gts.append(view.original_image.permute(1, 2, 0).cpu().numpy().astype(np.uint8))
    
    imageio.mimwrite(os.path.join(render_path, 'out.mp4'), all_preds, fps=25, quality=8, macro_block_size=1)
    if not fast:
        imageio.mimwrite(os.path.join(gts_path, 'out.mp4'), all_gts, fps=25, quality=8, macro_block_size=1)

        imageio.mimwrite(os.path.join(render_path, 'out_face.mp4'), all_preds_face, fps=25, quality=8, macro_block_size=1)
        imageio.mimwrite(os.path.join(render_path, 'out_mouth.mp4'), all_preds_mouth, fps=25, quality=8, macro_block_size=1)
```

**합성 과정 상세 설명**:

1. **Face 렌더링**: 얼굴 전체 (입 제외)
   ```python
   render_pkg = render_motion(view, gaussians, motion_net, ...)
   # render_pkg["render"]: 렌더링된 이미지 [3, H, W]
   # render_pkg["alpha"]: 알파 채널 [H, W]
   ```

2. **Mouth 렌더링**: 입 부분만
   ```python
   render_pkg_mouth = render_motion_mouth(view, gaussians_mouth, motion_net_mouth, ...)
   ```

3. **Mouth를 배경과 합성**
   ```python
   mouth_image = render_pkg_mouth["render"] + view.background * (1.0 - alpha_mouth)
   ```
   - Mouth 렌더링 결과가 투명한 부분은 배경으로 채움

4. **Face 위에 Mouth 올리기 (Alpha Blending)**
   ```python
   image = render_pkg["render"] + mouth_image * (1.0 - alpha)
   ```
   - Face가 투명한 부분에 Mouth 이미지 적용

### Alpha Blending 수식

최종 이미지 = Face + Mouth × (1 - Face_alpha)

- Face alpha가 1이면: Face만 표시
- Face alpha가 0이면: Mouth 표시
- 중간값이면: 두 이미지 블렌딩

## View 객체 구조

각 `view`는 `Camera` 객체로, 다음 정보를 포함:

```python
view.talking_dict = {
    'auds': audio_features,      # 오디오 특징 [29, 16] 또는 [1, 29, 16]
    'au_exp': action_units,      # AU 특징 [6] - 얼굴 표정
    'blink': blink_value,        # 깜빡임 값 (0~1)
    'au25': [value, ...],        # 입 벌림 정도
    'img_id': frame_id,          # 프레임 ID
    'face_mask': mask,           # 얼굴 마스크
    'mouth_mask': mask,          # 입 마스크
    'hair_mask': mask,           # 머리 마스크
    'lips_rect': [xmin, xmax, ymin, ymax],  # 입술 영역
    'lhalf_rect': [xmin, xmax, ymin, ymax], # 얼굴 하반부 영역
    'mouth_bound': [...],        # 입 경계 정보
}
```

### 오디오 특징 (`auds`)

- **DeepSpeech**: [29, 16] 형태
  - 29: 특징 차원
  - 16: 시간 윈도우
- **HuBERT**: [1024, 16] 형태
- `get_audio_features()` 함수로 프레임별 특징 추출

## 추론 실행 방법

### 기본 추론 (테스트 세트)
```bash
python synthesize_fuse.py -S data/<ID> -M output/<project_name> --eval
```
- 테스트 세트의 오디오 특징 사용
- 결과: `output/<project_name>/test/ours_None/renders/out.mp4`

### 커스텀 오디오 사용
```bash
python synthesize_fuse.py -S data/<ID> -M output/<project_name> --use_train --audio <audio_features>.npy
```
- `--audio` 옵션으로 다른 오디오 특징 파일 지정
- 오디오 특징 파일은 `.npy` 형식

### Fast 모드 (디버그용)
```bash
python synthesize_fuse.py -S data/<ID> -M output/<project_name> --eval --fast
```
- 중간 결과(face만, mouth만) 저장 안 함
- 최종 결과만 저장

### Dilate 옵션
```bash
python synthesize_fuse.py -S data/<ID> -M output/<project_name> --eval --dilate
```
- Mouth 마스크 dilation 적용
- 입 영역을 약간 확장하여 경계를 부드럽게

## 출력 파일

렌더링 결과는 다음 위치에 저장:

```
output/<project_name>/test/ours_None/
├── renders/
│   ├── out.mp4              # 최종 합성 비디오 ⭐
│   ├── out_face.mp4         # Face만 (--fast 없을 때)
│   └── out_mouth.mp4        # Mouth만 (--fast 없을 때)
└── gt/
    └── out.mp4              # Ground Truth 비디오
```

## 주요 파라미터

### 명령줄 인자

- `-S, --source_path`: 데이터 소스 경로 (`data/<ID>`)
- `-M, --model_path`: 모델 경로 (`output/<project_name>`)
- `--eval`: 평가 모드 (테스트 세트 사용)
- `--use_train`: 학습 세트 사용
- `--fast`: 빠른 모드 (중간 결과 저장 안 함)
- `--dilate`: Mouth 마스크 dilation 적용
- `--audio`: 커스텀 오디오 특징 파일 경로
- `--iteration`: 특정 iteration의 체크포인트 사용 (기본: -1, 최신)

### ModelParams

- `sh_degree`: Spherical Harmonics degree (기본: 2)
- `white_background`: 흰색 배경 사용 여부
- `audio_extractor`: 오디오 특징 추출기 (`deepspeech` 또는 `hubert`)

## 코드 실행 흐름

```
synthesize_fuse.py (main)
    ↓
render_sets()
    ├─ GaussianModel 초기화 (Face, Mouth)
    ├─ Scene 로드 (카메라 정보)
    ├─ MotionNetwork 초기화 (Face, Mouth)
    ├─ 체크포인트 로드 및 복원
    └─ render_set()
        ├─ 각 view에 대해:
        │   ├─ render_motion() (Face)
        │   ├─ render_motion_mouth() (Mouth)
        │   └─ Alpha Blending 합성
        └─ 비디오 저장
```

## 체크포인트 구조

`chkpnt_fuse_latest.pth` 파일 구조:

```python
(
    model_params,          # Face Gaussian 파라미터
    motion_params,         # Face Motion Network state_dict
    model_mouth_params,    # Mouth Gaussian 파라미터
    motion_mouth_params    # Mouth Motion Network state_dict
)
```

각 `model_params`는 튜플:
```python
(
    active_sh_degree,
    _xyz,
    _features_dc,
    _features_rest,
    _identity,
    _scaling,
    _rotation,
    _opacity,
    max_radii2D,
    xyz_gradient_accum,
    denom,
    optimizer_state_dict,
    spatial_lr_scale
)
```

## 디버깅 팁

1. **중간 결과 확인**:
   - `--fast` 옵션 없이 실행하면 `out_face.mp4`, `out_mouth.mp4` 확인 가능
   
2. **특정 프레임 확인**:
   - 코드에서 `imageio.mimwrite` 전에 특정 프레임을 이미지로 저장

3. **오디오 특징 확인**:
   - `view.talking_dict['auds']`의 shape 확인
   - DeepSpeech: [29, 16], HuBERT: [1024, 16]

4. **메모리 부족 시**:
   - `preload=False`로 데이터 로딩 (더 느림)

## 다음 단계

- `06_Rendering.md`: `render_motion` 함수 상세 분석
- `05_Motion_Network.md`: Motion Network가 어떻게 오디오를 변형으로 변환하는지
- `04_Gaussian_Model.md`: Gaussian 모델의 구조와 파라미터


