# Gaussian 모델 상세 분석

## 개요

`scene/gaussian_model.py`의 `GaussianModel` 클래스는 3D Gaussian Splatting의 핵심입니다.
3D 공간의 각 포인트를 Gaussian으로 표현하며, 위치, 색상, 스케일, 회전, 투명도 등의 파라미터를 관리합니다.

## Gaussian 파라미터

각 Gaussian은 다음 파라미터로 표현됩니다:

1. **위치** (`_xyz`): [N, 3] - 3D 공간에서의 위치
2. **색상** (`_features_dc`, `_features_rest`): Spherical Harmonics (SH) 계수
3. **스케일** (`_scaling`): [N, 3] - 각 축의 스케일
4. **회전** (`_rotation`): [N, 4] - Quaternion 표현
5. **투명도** (`_opacity`): [N, 1] - 불투명도
6. **Identity** (`_identity`): [N, 1] - (추가 파라미터)

## 클래스 초기화

```24:60:scene/gaussian_model.py
class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)

        
        self.scaling_activation = torch.nn.functional.softplus # torch.exp
        self.scaling_inverse_activation = lambda x: x + torch.log(-torch.expm1(-x)) # torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid


        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._identity = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
```

### Activation Functions

- **scaling_activation**: `softplus` - 스케일을 양수로 제한
- **opacity_activation**: `sigmoid` - 투명도를 [0, 1] 범위로 제한
- **rotation_activation**: `normalize` - Quaternion 정규화

## 주요 Properties

```100:124:scene/gaussian_model.py
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_identity(self):
        return self._identity
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
```

모든 파라미터는 내부적으로 raw 값(`_xyz`, `_scaling` 등)으로 저장되고,
property를 통해 activation 함수를 적용한 값으로 반환됩니다.

## Point Cloud 생성

### 초기 생성

```133:159:scene/gaussian_model.py
    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        identity = torch.zeros((fused_point_cloud.shape[0], 1), device="cuda")

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._identity = nn.Parameter(identity.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
```

**초기화 과정**:
1. Point Cloud의 점들을 `_xyz`로 설정
2. 색상을 Spherical Harmonics로 변환
3. 스케일: 최근접 이웃 거리 기반 초기화
4. 회전: Identity quaternion [1, 0, 0, 0]
5. 투명도: 0.1로 초기화 (inverse_sigmoid 적용)

### PLY 파일 로드

```247:288:scene/gaussian_model.py
    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[idx, :] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[idx, :] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
```

학습된 모델을 PLY 파일에서 로드합니다.

## Spherical Harmonics (SH)

색상은 Spherical Harmonics로 표현됩니다:

- **DC (Direct Component)**: `_features_dc` - [N, 1, 3] - 기본 색상
- **Rest**: `_features_rest` - [N, (max_sh_degree+1)²-1, 3] - 방향성 색상

SH degree가 높을수록 더 세밀한 방향성 색상을 표현할 수 있습니다.
- degree 0: 1개 계수 (DC만)
- degree 1: 4개 계수
- degree 2: 9개 계수
- degree 3: 16개 계수

## Covariance 계산

```126:127:scene/gaussian_model.py
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
```

Covariance 행렬은 스케일과 회전으로부터 계산됩니다:
- Scaling 행렬 S 생성
- Rotation 행렬 R 적용
- Covariance = R S S^T R^T

## 체크포인트 관리

### 저장

```62:77:scene/gaussian_model.py
    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._identity,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
```

### 복원

```79:97:scene/gaussian_model.py
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._identity,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        if training_args is not None:
            self.training_setup(training_args)
            self.optimizer.load_state_dict(opt_dict)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
```

## Motion Network와의 관계

추론 시 Motion Network가 Gaussian 파라미터를 변형합니다:

```python
# render_motion에서
motion_preds = motion_net(pc.get_xyz, audio_feat, exp_feat, ind_code)

# 변형 적용
means3D = pc.get_xyz + motion_preds['d_xyz']  # 위치 변형
scales = pc.scaling_activation(pc._scaling + motion_preds['d_scale'])  # 스케일 변형
rotations = pc.rotation_activation(pc._rotation + motion_preds['d_rot'])  # 회전 변형
```

**중요**: 색상(`_features`)과 투명도(`_opacity`)는 변형되지 않습니다.
- Face Motion Network: 위치, 스케일, 회전 변형
- Mouth Motion Network: 위치만 변형

## PLY 파일 저장

```204:221:scene/gaussian_model.py
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
```

학습된 Gaussian 모델을 PLY 파일로 저장합니다.

## Densification (학습 시)

학습 과정에서 Gaussian 점들을 추가/제거합니다:

- **Densify**: 중요 영역에 점 추가
- **Prune**: 불필요한 점 제거

(자세한 내용은 학습 코드 참고)

## 다음 단계

- `05_Motion_Network.md`: Motion Network가 Gaussian을 어떻게 변형하는지
- `06_Rendering.md`: 변형된 Gaussian을 어떻게 렌더링하는지

