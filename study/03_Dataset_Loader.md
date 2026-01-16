# 데이터셋 로더 상세 분석

## 개요

`scene/dataset_readers.py`는 전처리된 데이터를 로드하여 학습/추론에 사용할 수 있는 형태로 변환합니다.
각 프레임의 카메라 정보, 오디오 특징, Action Units 등을 `CameraInfo`로 저장합니다.

## 데이터 로딩 흐름

```
Scene.__init__()
    ↓
readNerfSyntheticInfo()
    ↓
readCamerasFromTransforms()
    ↓
CameraInfo 생성
    ↓
Camera 객체로 변환 (cameraList_from_camInfos)
```

## 핵심 함수: `readCamerasFromTransforms()`

```98:256:scene/dataset_readers.py
def readCamerasFromTransforms(path, transformsfile, white_background, extension=".jpg", audio_file='', audio_extractor='deepspeech', preload=False):
    cam_infos = []
    postfix_dict = {"deepspeech": "ds", "esperanto": "eo", "hubert": "hu"}

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        focal_len = contents["focal_len"]
        bg_img = np.array(Image.open(os.path.join(path, 'bc.jpg')).convert("RGB"))

        frames = contents["frames"]
        
        if audio_file == '':
            aud_features = np.load(os.path.join(path, 'aud_{}.npy'.format(postfix_dict[audio_extractor])))
        else:
            aud_features = np.load(audio_file)
        aud_features = torch.from_numpy(aud_features)
        aud_features = aud_features.float().permute(0, 2, 1)
        auds = aud_features

        au_info = pd.read_csv(os.path.join(path, 'au.csv'))
        au_info.columns = [c.strip() for c in au_info.columns]
        au_blink = au_info['AU45_r'].values
        au25 = au_info['AU25_r'].values
        au25 = np.clip(au25, 0, np.percentile(au25, 95))

        au25_25, au25_50, au25_75, au25_100 = np.percentile(au25, 25), np.percentile(au25, 50), np.percentile(au25, 75), au25.max()

        au_exp = []
        for i in [1,4,5,6,7,45]:
            _key = 'AU' + str(i).zfill(2) + '_r'
            au_exp_t = au_info[_key].values
            if i == 45:
                au_exp_t = au_exp_t.clip(0, 2)
            au_exp.append(au_exp_t[:, None])
        au_exp = np.concatenate(au_exp, axis=-1, dtype=np.float32)

        ldmks_lips = []
        ldmks_mouth = []
        ldmks_lhalf = []
        
        for idx, frame in tqdm(enumerate(frames)):
            lms = np.loadtxt(os.path.join(path, 'ori_imgs', str(frame['img_id']) + '.lms')) # [68, 2]
            lips = slice(48, 60)
            mouth = slice(60, 68)
            xmin, xmax = int(lms[lips, 1].min()), int(lms[lips, 1].max())
            ymin, ymax = int(lms[lips, 0].min()), int(lms[lips, 0].max())

            ldmks_lips.append([int(xmin), int(xmax), int(ymin), int(ymax)])
            ldmks_mouth.append([int(lms[mouth, 1].min()), int(lms[mouth, 1].max())])

            lh_xmin, lh_xmax = int(lms[31:36, 1].min()), int(lms[:, 1].max()) # actually lower half area
            xmin, xmax = int(lms[:, 1].min()), int(lms[:, 1].max())
            ymin, ymax = int(lms[:, 0].min()), int(lms[:, 0].max())
            # self.face_rect.append([xmin, xmax, ymin, ymax])
            ldmks_lhalf.append([lh_xmin, lh_xmax, ymin, ymax])
            
        ldmks_lips = np.array(ldmks_lips)
        ldmks_mouth = np.array(ldmks_mouth)
        ldmks_lhalf = np.array(ldmks_lhalf)
        mouth_lb = (ldmks_mouth[:, 1] - ldmks_mouth[:, 0]).min()
        mouth_ub = (ldmks_mouth[:, 1] - ldmks_mouth[:, 0]).max()



        for idx, frame in tqdm(enumerate(frames)):
            cam_name = os.path.join(path, 'gt_imgs', str(frame["img_id"]) + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)
            w, h = image.size[0], image.size[1]
            if preload:
                image = np.array(image.convert("RGB"))
            else:
                image = None

            torso_img_path = os.path.join(path, 'torso_imgs', str(frame['img_id']) + '.png')
            if preload:
                torso_img = np.array(Image.open(torso_img_path).convert("RGBA")) * 1.0
                bg = torso_img[..., :3] * torso_img[..., 3:] / 255.0 + bg_img * (1 - torso_img[..., 3:] / 255.0)
                bg = bg.astype(np.uint8)
            else:
                bg = None
            # bg = Image.fromarray(np.array(bg, dtype=np.byte), "RGB")
            # bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            talking_dict = {}
            talking_dict['img_id'] = frame['img_id']

            if preload:
                teeth_mask_path = os.path.join(path, 'teeth_mask', str(frame['img_id']) + '.npy')
                teeth_mask = np.load(teeth_mask_path)

                mask_path = os.path.join(path, 'parsing', str(frame['img_id']) + '.png')
                mask = np.array(Image.open(mask_path).convert("RGB")) * 1.0
                talking_dict['face_mask'] = (mask[:, :, 2] > 254) * (mask[:, :, 0] == 0) * (mask[:, :, 1] == 0) ^ teeth_mask
                talking_dict['hair_mask'] = (mask[:, :, 0] < 1) * (mask[:, :, 1] < 1) * (mask[:, :, 2] < 1)
                talking_dict['mouth_mask'] = (mask[:, :, 0] == 100) * (mask[:, :, 1] == 100) * (mask[:, :, 2] == 100) + teeth_mask


            
            if audio_file == '':
                talking_dict['auds'] = get_audio_features(auds, 2, frame['img_id'])
                if frame['img_id'] > auds.shape[0]:
                    print("[warnining] audio feature is too short")
                    break
            else:
                talking_dict['auds'] = get_audio_features(auds, 2, idx)
                if idx >= auds.shape[0]:
                    break


            talking_dict['blink'] = torch.as_tensor(np.clip(au_blink[frame['img_id']], 0, 2) / 2)
            talking_dict['au25'] = [au25[frame['img_id']], au25_25, au25_50, au25_75, au25_100]

            talking_dict['au_exp'] = torch.as_tensor(au_exp[frame['img_id']])

            [xmin, xmax, ymin, ymax] = ldmks_lips[idx].tolist()
            # padding to H == W
            cx = (xmin + xmax) // 2
            cy = (ymin + ymax) // 2

            l = max(xmax - xmin, ymax - ymin) // 2
            xmin = cx - l
            xmax = cx + l
            ymin = cy - l
            ymax = cy + l

            talking_dict['lips_rect'] = [xmin, xmax, ymin, ymax]
            talking_dict['lhalf_rect'] = ldmks_lhalf[idx]
            talking_dict['mouth_bound'] = [mouth_lb, mouth_ub, ldmks_mouth[idx, 1] - ldmks_mouth[idx, 0]]
            talking_dict['img_id'] = frame['img_id']


            # norm_data = im_data / 255.0
            # arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            # image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            FovX = focal2fov(focal_len, w)
            FovY = focal2fov(focal_len, h)

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=w, height=h, background=bg, talking_dict=talking_dict))

            # if idx > 200: break
            # if idx > 6500: break
            
    return cam_infos
```

## CameraInfo 구조

```29:41:scene/dataset_readers.py
class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    background: np.array
    talking_dict: dict
```

### talking_dict 상세 구조

각 프레임의 `talking_dict`는 다음 정보를 포함:

```python
talking_dict = {
    'img_id': int,                    # 프레임 ID
    'auds': torch.Tensor,             # 오디오 특징 [29, 16] 또는 [1, 29, 16]
    'au_exp': torch.Tensor,           # AU 특징 [6] (AU1, AU4, AU5, AU6, AU7, AU45)
    'blink': torch.Tensor,            # 깜빡임 값 [0~1]
    'au25': List[float],              # 입 벌림 정도 [value, p25, p50, p75, max]
    'face_mask': np.ndarray,          # 얼굴 마스크 [H, W] (bool)
    'hair_mask': np.ndarray,          # 머리 마스크 [H, W] (bool)
    'mouth_mask': np.ndarray,         # 입 마스크 [H, W] (bool)
    'lips_rect': List[int],           # 입술 영역 [xmin, xmax, ymin, ymax]
    'lhalf_rect': List[int],          # 얼굴 하반부 영역 [xmin, xmax, ymin, ymax]
    'mouth_bound': List[float],       # 입 경계 정보 [min, max, current]
}
```

## 주요 처리 단계

### 1. 오디오 특징 로드

```python
if audio_file == '':
    aud_features = np.load(os.path.join(path, 'aud_{}.npy'.format(postfix_dict[audio_extractor])))
else:
    aud_features = np.load(audio_file)
aud_features = torch.from_numpy(aud_features)
aud_features = aud_features.float().permute(0, 2, 1)  # [N, 16, 29] → [N, 29, 16]
```

- DeepSpeech: `aud_ds.npy` → [N, 29, 16]
- HuBERT: `aud_hu.npy` → [N, 1024, 16]

### 2. Action Units 처리

```python
au_exp = []
for i in [1,4,5,6,7,45]:  # AU 종류
    _key = 'AU' + str(i).zfill(2) + '_r'
    au_exp_t = au_info[_key].values
    if i == 45:  # 눈 깜빡임
        au_exp_t = au_exp_t.clip(0, 2)
    au_exp.append(au_exp_t[:, None])
au_exp = np.concatenate(au_exp, axis=-1, dtype=np.float32)  # [N, 6]
```

- AU1, AU4, AU5, AU6, AU7: 얼굴 표정
- AU45: 눈 깜빡임 (0~2로 클리핑)

### 3. 랜드마크 처리

```python
lms = np.loadtxt(os.path.join(path, 'ori_imgs', str(frame['img_id']) + '.lms'))  # [68, 2]
lips = slice(48, 60)      # 입술 랜드마크
mouth = slice(60, 68)     # 입 랜드마크
```

- 68개 랜드마크 포인트 사용
- 입술/입 영역 추출

### 4. 카메라 포즈 변환

```python
c2w = np.array(frame["transform_matrix"])  # camera-to-world
c2w[:3, 1:3] *= -1  # OpenGL → COLMAP 좌표계 변환
w2c = np.linalg.inv(c2w)  # world-to-camera
R = np.transpose(w2c[:3,:3])  # 회전 행렬
T = w2c[:3, 3]  # 변환 벡터
```

- NeRF 형식의 transform_matrix 사용
- OpenGL 좌표계 → COLMAP 좌표계 변환

### 5. 오디오 특징 추출 (프레임별)

```python
talking_dict['auds'] = get_audio_features(auds, 2, frame['img_id'])
```

`get_audio_features()` 함수는 프레임 ID에 해당하는 오디오 특징을 추출합니다.
- 모드 2: 인덱스 기반 추출
- 윈도우 크기: 16 프레임

## Scene 클래스에서의 사용

```27:85:scene/__init__.py
def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
    """b
    :param path: Path to colmap scene main folder.
    """
    self.model_path = args.model_path
    self.loaded_iter = None
    self.gaussians = gaussians

    if load_iteration:
        if load_iteration == -1:
            self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
        else:
            self.loaded_iter = load_iteration
        print("Loading trained model at iteration {}".format(self.loaded_iter))

    self.train_cameras = {}
    self.test_cameras = {}

    if os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
        print("Found transforms_train.json file, assuming Blender data set!")
        scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, args=args)
    else:
        assert False, "Could not recognize scene type!"

    if not self.loaded_iter:
        with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
            dest_file.write(src_file.read())
        json_cams = []
        camlist = []
        if scene_info.test_cameras:
            camlist.extend(scene_info.test_cameras)
        if scene_info.train_cameras:
            camlist.extend(scene_info.train_cameras)
        for id, cam in enumerate(camlist):
            json_cams.append(camera_to_JSON(id, cam))
        with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
            json.dump(json_cams, file)

    if shuffle:
        random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
        random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

    self.cameras_extent = scene_info.nerf_normalization["radius"]

    for resolution_scale in resolution_scales:
        print("Loading Training Cameras")
        self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
        print("Loading Test Cameras")
        self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

    if self.loaded_iter:
        self.gaussians.load_ply(os.path.join(self.model_path,
                                                       "point_cloud",
                                                       "iteration_" + str(self.loaded_iter),
                                                       "point_cloud.ply"))
    else:
        self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    gc.collect()
```

## preload 옵션

```python
preload=False  # 기본값
```

- `preload=True`: 모든 이미지와 마스크를 메모리에 로드 (빠름, 메모리 많이 사용)
- `preload=False`: 필요할 때만 로드 (느림, 메모리 적게 사용)

**메모리 사용량**:
- 약 N × 32GB RAM (N × 5k 프레임, preload=True)
- 권장: `preload=False`로 설정하여 필요 시에만 로드

## 오디오 특징 추출 함수

`utils/audio_utils.py`의 `get_audio_features()` 함수:

```python
def get_audio_features(features, att_mode, index):
    # features: [N, 29, 16] 또는 [N, 1024, 16]
    # att_mode: 2 (인덱스 기반)
    # index: 프레임 ID
    
    if att_mode == 2:
        # 인덱스 기반: 해당 프레임의 오디오 특징 반환
        return features[index:index+1]  # [1, 29, 16]
```

## 데이터 정규화

### NeRF Normalization

```50:71:scene/dataset_readers.py
def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}
```

- 카메라 중심 계산
- 반경 계산 (대각선 × 1.1)
- Gaussian 모델 초기화에 사용

## 다음 단계

- `04_Gaussian_Model.md`: Gaussian 모델 구조
- `05_Motion_Network.md`: Motion Network가 talking_dict를 어떻게 사용하는지
- `01_Inference_Pipeline.md`: 추론 시 데이터 로딩


