# 렌더링 과정 상세 분석

## 개요

`gaussian_renderer/__init__.py`의 렌더링 함수들은 변형된 Gaussian을 이미지로 렌더링합니다.
Gaussian Splatting을 사용하여 각 Gaussian을 2D로 투영하고 합성합니다.

## 주요 렌더링 함수

1. **`render_motion()`**: Face 모델 렌더링
2. **`render_motion_mouth()`**: Mouth 모델 렌더링

## render_motion (Face)

### 함수 시그니처

```106:195:gaussian_renderer/__init__.py
def render_motion(viewpoint_camera, pc : GaussianModel, motion_net : MotionNetwork, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, frame_idx = None, return_attn = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    audio_feat = viewpoint_camera.talking_dict["auds"].cuda()
    exp_feat = viewpoint_camera.talking_dict["au_exp"].cuda()

    # ind_code = motion_net.individual_codes[frame_idx if frame_idx is not None else viewpoint_camera.talking_dict["img_id"]]
    ind_code = None
    motion_preds = motion_net(pc.get_xyz, audio_feat, exp_feat, ind_code) #  
    means3D = pc.get_xyz + motion_preds['d_xyz']
    means2D = screenspace_points
    # opacity = pc.opacity_activation(pc._opacity + motion_preds['d_opa'])
    opacity = pc.get_opacity

    cov3D_precomp = None
    # scales = pc.get_scaling
    scales = pc.scaling_activation(pc._scaling + motion_preds['d_scale'])
    rotations = pc.rotation_activation(pc._rotation + motion_preds['d_rot'])

    colors_precomp = None
    shs = pc.get_features

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    # Attn
    rendered_attn = None
    if return_attn:
        attn_precomp = torch.cat([motion_preds['ambient_aud'], motion_preds['ambient_eye'], torch.zeros_like(motion_preds['ambient_eye'])], dim=-1)
        rendered_attn, _, _, _ = rasterizer(
            means3D = means3D.detach(),
            means2D = means2D,
            shs = None,
            colors_precomp = attn_precomp,
            opacities = opacity.detach(),
            scales = scales.detach(),
            rotations = rotations.detach(),
            cov3D_precomp = cov3D_precomp)


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "depth": rendered_depth, 
            "alpha": rendered_alpha,

            "motion": motion_preds,
            'attn': rendered_attn}
```

### 렌더링 과정

1. **Rasterization 설정**: 카메라 파라미터로 설정
2. **Motion Network 적용**: Gaussian 변형 계산
3. **변형 적용**: 위치, 스케일, 회전 적용
4. **Rasterization**: Gaussian을 이미지로 렌더링

### Motion Network 적용

```python
motion_preds = motion_net(pc.get_xyz, audio_feat, exp_feat, ind_code)

# 변형 적용
means3D = pc.get_xyz + motion_preds['d_xyz']  # 위치 변형
scales = pc.scaling_activation(pc._scaling + motion_preds['d_scale'])  # 스케일 변형
rotations = pc.rotation_activation(pc._rotation + motion_preds['d_rot'])  # 회전 변형
opacity = pc.get_opacity  # 투명도는 변형 안 함 (주석 처리됨)
```

**주목**: 
- 원래 코드에서 `d_opa` (투명도 변형)는 주석 처리되어 사용되지 않음
- 색상(`shs`)은 변형되지 않음

### Rasterization

```python
rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
    means3D = means3D,      # 변형된 3D 위치
    means2D = means2D,      # 2D 화면 좌표 (gradient용)
    shs = shs,              # Spherical Harmonics 색상
    colors_precomp = None,  # 미리 계산된 색상 (사용 안 함)
    opacities = opacity,    # 투명도
    scales = scales,        # 변형된 스케일
    rotations = rotations,  # 변형된 회전
    cov3D_precomp = None   # 미리 계산된 covariance (사용 안 함)
)
```

**출력**:
- `rendered_image`: [3, H, W] 렌더링된 이미지
- `radii`: [N] 각 Gaussian의 화면 반경
- `rendered_depth`: [H, W] 깊이 맵
- `rendered_alpha`: [H, W] 알파 채널

## render_motion_mouth (Mouth)

### 함수 시그니처

```201:270:gaussian_renderer/__init__.py
def render_motion_mouth(viewpoint_camera, pc : GaussianModel, motion_net : MouthMotionNetwork, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, frame_idx = None, return_attn = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    audio_feat = viewpoint_camera.talking_dict["auds"].cuda()

    motion_preds = motion_net(pc.get_xyz, audio_feat)
    means3D = pc.get_xyz + motion_preds['d_xyz']
    means2D = screenspace_points
    opacity = pc.get_opacity

    cov3D_precomp = None
    scales = pc.get_scaling
    rotations = pc.get_rotation

    colors_precomp = None
    shs = pc.get_features

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "depth": rendered_depth, 
            "alpha": rendered_alpha,
            "radii": radii,
            "motion": motion_preds}
```

### 차이점

1. **Motion Network**: `MouthMotionNetwork` 사용 (Action Units 입력 없음)
2. **변형**: 위치(`d_xyz`)만 변형, 스케일/회전은 원본 사용
3. **출력**: Attention 맵 없음

## 합성 과정 (synthesize_fuse.py)

```python
# Face 렌더링
render_pkg = render_motion(view, gaussians, motion_net, ...)

# Mouth 렌더링
render_pkg_mouth = render_motion_mouth(view, gaussians_mouth, motion_net_mouth, ...)

# Alpha Blending
mouth_image = render_pkg_mouth["render"] + view.background * (1.0 - alpha_mouth)
final_image = render_pkg["render"] + mouth_image * (1.0 - alpha)
```

**합성 순서**:
1. Mouth를 배경과 합성
2. Face 위에 Mouth 올리기 (Alpha Blending)

## Gaussian Rasterization

Gaussian Rasterization은 다음 과정을 거칩니다:

1. **3D → 2D 투영**: 각 Gaussian을 화면에 투영
2. **Covariance 계산**: 스케일과 회전으로부터 2D covariance 계산
3. **알파 블렌딩**: 각 Gaussian을 알파 블렌딩으로 합성
4. **색상 계산**: Spherical Harmonics로 뷰 방향에 따른 색상 계산

(자세한 구현은 CUDA 코드 참고)

## RasterizationSettings

```python
raster_settings = GaussianRasterizationSettings(
    image_height=int(viewpoint_camera.image_height),
    image_width=int(viewpoint_camera.image_width),
    tanfovx=tanfovx,                           # 시야각 (x)
    tanfovy=tanfovy,                           # 시야각 (y)
    bg=bg_color,                               # 배경색
    scale_modifier=scaling_modifier,           # 스케일 조정
    viewmatrix=viewpoint_camera.world_view_transform,  # World-to-view 변환
    projmatrix=viewpoint_camera.full_proj_transform,   # 투영 변환
    sh_degree=pc.active_sh_degree,            # SH degree
    campos=viewpoint_camera.camera_center,     # 카메라 위치
    prefiltered=False,                         # 필터링 여부
    debug=pipe.debug                           # 디버그 모드
)
```

## 출력 구조

렌더링 함수는 다음 딕셔너리를 반환:

```python
{
    "render": rendered_image,        # [3, H, W] 렌더링된 이미지
    "viewspace_points": screenspace_points,  # Gradient용 2D 좌표
    "visibility_filter": radii > 0,  # [N] 보이는 Gaussian 마스크
    "depth": rendered_depth,         # [H, W] 깊이 맵
    "alpha": rendered_alpha,         # [H, W] 알파 채널
    "motion": motion_preds,          # Motion Network 출력
    "radii": radii,                  # [N] 각 Gaussian의 화면 반경
    "attn": rendered_attn            # (선택) Attention 맵
}
```

## 다음 단계

- `01_Inference_Pipeline.md`: 추론 파이프라인에서 렌더링 사용
- `05_Motion_Network.md`: Motion Network 구조
- `04_Gaussian_Model.md`: Gaussian 모델 구조

