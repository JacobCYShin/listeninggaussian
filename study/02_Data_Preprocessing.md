# 데이터 전처리 과정 상세 분석

## 개요

`data_utils/process.py`는 비디오를 입력받아 학습에 필요한 모든 데이터를 추출합니다.
전처리 과정은 여러 단계로 나뉘어 있으며, 각 단계가 순차적으로 실행됩니다.

## 실행 방법

```bash
python data_utils/process.py data/<ID>/<ID>.mp4
```

## 전처리 단계

전처리는 다음 순서로 진행됩니다:

1. **오디오 추출**
2. **프레임 추출**
3. **얼굴 랜드마크 추출**
4. **Face Parsing (의미 분할)**
5. **Face Tracking (3DMM 기반)**
6. **Action Units (AU) 추출** (별도 실행 필요)
7. **오디오 특징 추출**
8. **배경 이미지 생성**
9. **Torso 이미지 생성**
10. **Teeth 마스크 생성** (별도 실행 필요)
11. **Transform JSON 생성**

## 1. 오디오 추출

```9:14:data_utils/process.py
def extract_audio(path, out_path, sample_rate=16000):
    
    print(f'[INFO] ===== extract audio from {path} to {out_path} =====')
    cmd = f'ffmpeg -i {path} -f wav -ar {sample_rate} {out_path}'
    os.system(cmd)
    print(f'[INFO] ===== extracted audio =====')
```

- **입력**: 비디오 파일 (`<ID>.mp4`)
- **출력**: 오디오 파일 (`aud.wav`, 16kHz)
- **도구**: FFmpeg
- **목적**: 오디오 특징 추출 및 비디오와 동기화

## 2. 프레임 추출

```29:34:data_utils/process.py
def extract_images(path, out_path, fps=25):

    print(f'[INFO] ===== extract images from {path} to {out_path} =====')
    cmd = f'ffmpeg -i {path} -vf fps={fps} -qmin 1 -q:v 1 -start_number 0 {os.path.join(out_path, "%d.jpg")}'
    os.system(cmd)
    print(f'[INFO] ===== extracted images =====')
```

- **입력**: 비디오 파일
- **출력**: 프레임 이미지들 (`ori_imgs/0.jpg`, `1.jpg`, ...)
- **FPS**: 25 (고정)
- **목적**: 각 프레임을 개별 이미지로 저장

## 3. 얼굴 랜드마크 추출

```45:65:data_utils/process.py
def extract_landmarks(ori_imgs_dir):

    print(f'[INFO] ===== extract face landmarks from {ori_imgs_dir} =====')

    import face_alignment
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device)
    except:
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=device)
    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))
    for image_path in tqdm.tqdm(image_paths):
        input = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) # [H, W, 3]
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        preds = fa.get_landmarks(input)
        if len(preds) > 0:
            lands = preds[0].reshape(-1, 2)[:,:2]
            np.savetxt(image_path.replace('jpg', 'lms'), lands, '%f')
    del fa
    print(f'[INFO] ===== extracted face landmarks =====')
```

- **도구**: Face Alignment (2D landmarks)
- **출력**: 각 이미지마다 `.lms` 파일 (68개 랜드마크 포인트)
- **형식**: `[x, y]` 좌표 (68, 2)
- **목적**: 입술 영역, 얼굴 영역 식별

## 4. Face Parsing (의미 분할)

```37:42:data_utils/process.py
def extract_semantics(ori_imgs_dir, parsing_dir):

    print(f'[INFO] ===== extract semantics from {ori_imgs_dir} to {parsing_dir} =====')
    cmd = f'python data_utils/face_parsing/test.py --respath={parsing_dir} --imgpath={ori_imgs_dir}'
    os.system(cmd)
    print(f'[INFO] ===== extracted semantics =====')
```

- **도구**: Face Parsing 모델
- **출력**: `parsing/<id>.png` (각 영역별 색상으로 분할)
- **영역**: 얼굴, 목, 몸통, 배경 등
- **목적**: 각 영역 마스크 생성

## 5. Face Tracking (3DMM)

```247:261:data_utils/process.py
def face_tracking(ori_imgs_dir):

    print(f'[INFO] ===== perform face tracking =====')
```

- **도구**: 3DMM (3D Morphable Model) 기반 얼굴 추적
- **출력**: `track_params.pt` (포즈, 위치, 회전 정보)
- **목적**: 각 프레임의 카메라 포즈 계산

## 6. Action Units (AU) 추출

**주의**: 이 단계는 별도로 실행해야 합니다.

```bash
# OpenFace 사용 (별도 설치 필요)
FeatureExtraction -f data/<ID>/aud.wav -of data/<ID>/au.csv
```

- **도구**: OpenFace
- **출력**: `au.csv` (각 프레임의 Action Units 값)
- **AU 종류**: AU1, AU4, AU5, AU6, AU7, AU45 (눈 깜빡임), AU25 (입 벌림)
- **목적**: 얼굴 표정 및 눈 깜빡임 정보

## 7. 오디오 특징 추출

**DeepSpeech 사용**:
```bash
python data_utils/deepspeech_features/extract_ds_features.py --input data/<ID>/aud.wav
# 출력: data/<ID>/aud_ds.npy
```

**HuBERT 사용** (비영어 오디오 권장):
```bash
python data_utils/hubert.py --wav data/<ID>/aud.wav
# 출력: data/<ID>/aud_hu.npy
```

- **DeepSpeech**: [N, 29, 16] - 29차원 특징, 16 프레임 윈도우
- **HuBERT**: [N, 1024, 16] - 1024차원 특징
- **목적**: Motion Network 입력

## 8. 배경 이미지 생성

```68:122:data_utils/process.py
def extract_background(base_dir, ori_imgs_dir):
    
    print(f'[INFO] ===== extract background image from {ori_imgs_dir} =====')

    from sklearn.neighbors import NearestNeighbors

    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))
    # only use 1/20 image_paths 
    image_paths = image_paths[::20]
    # read one image to get H/W
    tmp_image = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED) # [H, W, 3]
    h, w = tmp_image.shape[:2]

    # nearest neighbors
    all_xys = np.mgrid[0:h, 0:w].reshape(2, -1).transpose()
    distss = []
    for image_path in tqdm.tqdm(image_paths):
        parse_img = cv2.imread(image_path.replace('ori_imgs', 'parsing').replace('.jpg', '.png'))
        bg = (parse_img[..., 0] == 255) & (parse_img[..., 1] == 255) & (parse_img[..., 2] == 255)
        fg_xys = np.stack(np.nonzero(~bg)).transpose(1, 0)
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(fg_xys)
        dists, _ = nbrs.kneighbors(all_xys)
        distss.append(dists)

    distss = np.stack(distss)
    max_dist = np.max(distss, 0)
    max_id = np.argmax(distss, 0)

    bc_pixs = max_dist > 5
    bc_pixs_id = np.nonzero(bc_pixs)
    bc_ids = max_id[bc_pixs_id]

    imgs = []
    num_pixs = distss.shape[1]
    for image_path in image_paths:
        img = cv2.imread(image_path)
        imgs.append(img)
    imgs = np.stack(imgs).reshape(-1, num_pixs, 3)

    bc_img = np.zeros((h*w, 3), dtype=np.uint8)
    bc_img[bc_pixs_id, :] = imgs[bc_ids, bc_pixs_id, :]
    bc_img = bc_img.reshape(h, w, 3)

    max_dist = max_dist.reshape(h, w)
    bc_pixs = max_dist > 5
    bg_xys = np.stack(np.nonzero(~bc_pixs)).transpose()
    fg_xys = np.stack(np.nonzero(bc_pixs)).transpose()
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(fg_xys)
    distances, indices = nbrs.kneighbors(bg_xys)
    bg_fg_xys = fg_xys[indices[:, 0]]
    bc_img[bg_xys[:, 0], bg_xys[:, 1], :] = bc_img[bg_fg_xys[:, 0], bg_fg_xys[:, 1], :]

    cv2.imwrite(os.path.join(base_dir, 'bc.jpg'), bc_img)

    print(f'[INFO] ===== extracted background image =====')
```

- **알고리즘**: Nearest Neighbor 기반 배경 픽셀 선택
- **출력**: `bc.jpg` (배경 이미지)
- **목적**: 렌더링 시 배경으로 사용

## 9. Torso 이미지 생성

```125:244:data_utils/process.py
def extract_torso_and_gt(base_dir, ori_imgs_dir):

    print(f'[INFO] ===== extract torso and gt images for {base_dir} =====')

    from scipy.ndimage import binary_erosion, binary_dilation
```

- **출력**: 
  - `gt_imgs/<id>.jpg`: Ground Truth 이미지 (얼굴 + 배경)
  - `torso_imgs/<id>.png`: Torso 이미지 (RGBA)
- **목적**: 
  - GT: 학습 시 손실 계산
  - Torso: 렌더링 시 목/몸통 부분

## 10. Teeth 마스크 생성

**주의**: 이 단계는 별도로 실행해야 합니다.

```bash
export PYTHONPATH=./data_utils/easyportrait 
python ./data_utils/easyportrait/create_teeth_mask.py ./data/<ID>
```

- **도구**: EasyPortrait
- **출력**: `teeth_mask/<id>.npy` (이빨 마스크)
- **목적**: 입 렌더링 시 이빨 제외

## 11. Transform JSON 생성

```264:347:data_utils/process.py
def save_transforms(base_dir, ori_imgs_dir):
    print(f'[INFO] ===== save transforms =====')

    import torch

    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))
    
    # read one image to get H/W
    tmp_image = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED) # [H, W, 3]
    h, w = tmp_image.shape[:2]

    params_dict = torch.load(os.path.join(base_dir, 'track_params.pt'))
    focal_len = params_dict['focal']
    euler_angle = params_dict['euler']
    trans = params_dict['trans'] / 10.0
    valid_num = euler_angle.shape[0]

    def euler2rot(euler_angle):
        # ... 회전 행렬 변환

    # train_val_split = int(valid_num * 10 / 11)
    train_ids = torch.arange(0, train_val_split)
    val_ids = torch.arange(train_val_split, valid_num)

    rot = euler2rot(euler_angle)
    rot_inv = rot.permute(0, 2, 1)
    trans_inv = -torch.bmm(rot_inv, trans.unsqueeze(2))

    pose = torch.eye(4, dtype=torch.float32)
    save_ids = ['train', 'val']
    train_val_ids = [train_ids, val_ids]
    mean_z = -float(torch.mean(trans[:, 2]).item())

    for split in range(2):
        transform_dict = dict()
        transform_dict['focal_len'] = float(focal_len[0])
        transform_dict['cx'] = float(w/2.0)
        transform_dict['cy'] = float(h/2.0)
        transform_dict['frames'] = []
        ids = train_val_ids[split]
        save_id = save_ids[split]

        for i in ids:
            i = i.item()
            frame_dict = dict()
            frame_dict['img_id'] = i
            frame_dict['aud_id'] = i

            pose[:3, :3] = rot_inv[i]
            pose[:3, 3] = trans_inv[i, :, 0]

            frame_dict['transform_matrix'] = pose.numpy().tolist()

            transform_dict['frames'].append(frame_dict)

        with open(os.path.join(base_dir, 'transforms_' + save_id + '.json'), 'w') as fp:
            json.dump(transform_dict, fp, indent=2, separators=(',', ': '))

    print(f'[INFO] ===== finished saving transforms =====')
```

- **입력**: `track_params.pt` (Face Tracking 결과)
- **출력**: 
  - `transforms_train.json`: 학습 세트 카메라 정보
  - `transforms_val.json`: 검증 세트 카메라 정보
- **분할 비율**: 10:1 (학습:검증)
- **목적**: 각 프레임의 카메라 포즈 저장

## 최종 데이터 구조

전처리 완료 후 `data/<ID>/` 디렉토리 구조:

```
data/<ID>/
├── <ID>.mp4                 # 원본 비디오
├── aud.wav                  # 추출된 오디오
├── aud_ds.npy              # DeepSpeech 특징 (또는 aud_hu.npy)
├── au.csv                  # Action Units (OpenFace)
├── bc.jpg                  # 배경 이미지
├── track_params.pt         # Face Tracking 결과
├── transforms_train.json   # 학습 세트 카메라 정보
├── transforms_val.json     # 검증 세트 카메라 정보
├── ori_imgs/               # 원본 프레임
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
├── gt_imgs/                # Ground Truth 이미지
│   ├── 0.jpg
│   └── ...
├── torso_imgs/             # Torso 이미지
│   ├── 0.png
│   └── ...
├── parsing/                # Face Parsing 결과
│   ├── 0.png
│   └── ...
├── teeth_mask/             # 이빨 마스크
│   ├── 0.npy
│   └── ...
└── *.lms                   # 랜드마크 파일 (ori_imgs와 같은 디렉토리)
```

## Transform JSON 구조

`transforms_train.json` 예시:

```json
{
  "focal_len": 512.0,
  "cx": 256.0,
  "cy": 256.0,
  "frames": [
    {
      "img_id": 0,
      "aud_id": 0,
      "transform_matrix": [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -1.0],
        [0.0, 0.0, 0.0, 1.0]
      ]
    },
    ...
  ]
}
```

- `focal_len`: 카메라 초점 거리
- `cx`, `cy`: 주점 (이미지 중심)
- `transform_matrix`: 4x4 카메라-to-world 변환 행렬

## 주의사항

1. **필수 단계**:
   - 오디오 추출
   - 프레임 추출
   - 랜드마크 추출
   - Face Parsing
   - Face Tracking
   - Transform JSON 생성
   - 오디오 특징 추출

2. **선택 단계** (별도 실행):
   - AU 추출 (OpenFace)
   - Teeth 마스크 생성 (EasyPortrait)

3. **비디오 요구사항**:
   - FPS: 25 (권장)
   - 해상도: 약 512x512
   - 길이: 1-5분
   - 모든 프레임에 말하는 사람이 포함되어야 함

## 다음 단계

- `03_Dataset_Loader.md`: 전처리된 데이터를 어떻게 로드하는지
- `01_Inference_Pipeline.md`: 추론 파이프라인


