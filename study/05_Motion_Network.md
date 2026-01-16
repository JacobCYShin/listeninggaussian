# Motion Network 상세 분석

## 개요

`scene/motion_net.py`의 Motion Network는 오디오 특징을 입력받아 Gaussian 파라미터의 변형(delta)을 예측합니다.
두 가지 네트워크가 있습니다:
- **MotionNetwork**: 얼굴 전체 움직임 제어
- **MouthMotionNetwork**: 입 부분 움직임 제어

## MotionNetwork (Face)

### 구조

```92:146:scene/motion_net.py
class MotionNetwork(nn.Module):
    def __init__(self,
                 audio_dim = 32,
                 ind_dim = 0,
                 args = None,
                 ):
        super(MotionNetwork, self).__init__()

        if 'esperanto' in args.audio_extractor:
            self.audio_in_dim = 44
        elif 'deepspeech' in args.audio_extractor:
            self.audio_in_dim = 29
        elif 'hubert' in args.audio_extractor:
            self.audio_in_dim = 1024
        else:
            raise NotImplementedError
    
        self.bound = 0.15
        self.exp_eye = True

        
        self.individual_dim = ind_dim
        if self.individual_dim > 0:
            self.individual_codes = nn.Parameter(torch.randn(10000, self.individual_dim) * 0.1) 

        # audio network
        self.audio_dim = audio_dim
        self.audio_net = AudioNet(self.audio_in_dim, self.audio_dim)

        self.audio_att_net = AudioAttNet(self.audio_dim)

        # DYNAMIC PART
        self.num_levels = 12
        self.level_dim = 1
        self.encoder_xy, self.in_dim_xy = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17, desired_resolution=256 * self.bound)
        self.encoder_yz, self.in_dim_yz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17, desired_resolution=256 * self.bound)
        self.encoder_xz, self.in_dim_xz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17, desired_resolution=256 * self.bound)

        self.in_dim = self.in_dim_xy + self.in_dim_yz + self.in_dim_xz


        self.num_layers = 3       
        self.hidden_dim = 64

        self.exp_in_dim = 6 - 1
        self.eye_dim = 6 if self.exp_eye else 0
        self.exp_encode_net = MLP(self.exp_in_dim, self.eye_dim - 1, 16, 2)

        self.eye_att_net = MLP(self.in_dim, self.eye_dim, 16, 2)

        # rot: 4   xyz: 3   opac: 1  scale: 3
        self.out_dim = 11
        self.sigma_net = MLP(self.in_dim + self.audio_dim + self.eye_dim + self.individual_dim, self.out_dim, self.hidden_dim, self.num_layers)
        
        self.aud_ch_att_net = MLP(self.in_dim, self.audio_dim, 32, 2)
```

### 주요 컴포넌트

1. **AudioNet**: 오디오 특징 인코딩
2. **AudioAttNet**: 오디오 특징 Attention
3. **Hash Grid Encoders**: 공간 위치 인코딩 (xy, yz, xz 평면)
4. **Exp Encode Net**: Action Units 인코딩
5. **Sigma Net**: 최종 변형 예측

### Forward Pass

```180:212:scene/motion_net.py
    def forward(self, x, a, e=None, c=None):
        # x: [N, 3], in [-bound, bound]
        enc_x = self.encode_x(x, bound=self.bound)

        enc_a = self.encode_audio(a)
        enc_a = enc_a.repeat(enc_x.shape[0], 1)
        aud_ch_att = self.aud_ch_att_net(enc_x)
        enc_w = enc_a * aud_ch_att
        
        eye_att = torch.relu(self.eye_att_net(enc_x))
        enc_e = self.exp_encode_net(e[:-1])
        enc_e = torch.cat([enc_e, e[-1:]], dim=-1)
        enc_e = enc_e * eye_att
        if c is not None:
            c = c.repeat(enc_x.shape[0], 1)
            h = torch.cat([enc_x, enc_w, enc_e, c], dim=-1)
        else:
            h = torch.cat([enc_x, enc_w, enc_e], dim=-1)

        h = self.sigma_net(h)

        d_xyz = h[..., :3] * 1e-2
        d_rot = h[..., 3:7]
        d_opa = h[..., 7:8]
        d_scale = h[..., 8:11]
        return {
            'd_xyz': d_xyz,
            'd_rot': d_rot,
            'd_opa': d_opa,
            'd_scale': d_scale,
            'ambient_aud' : aud_ch_att.norm(dim=-1, keepdim=True),
            'ambient_eye' : eye_att.norm(dim=-1, keepdim=True),
        }
```

**입력**:
- `x`: Gaussian 위치 [N, 3] (bound [-0.15, 0.15] 내)
- `a`: 오디오 특징 [1, 29, 16] 또는 [1, 1024, 16]
- `e`: Action Units [6]
- `c`: Individual codes (옵션)

**출력**:
- `d_xyz`: 위치 변형 [N, 3]
- `d_rot`: 회전 변형 [N, 4] (quaternion)
- `d_opa`: 투명도 변형 [N, 1]
- `d_scale`: 스케일 변형 [N, 3]

### 공간 인코딩

```156:164:scene/motion_net.py
    def encode_x(self, xyz, bound):
        # x: [N, 3], in [-bound, bound]
        N, M = xyz.shape
        xy, yz, xz = self.split_xyz(xyz)
        feat_xy = self.encoder_xy(xy, bound=bound)
        feat_yz = self.encoder_yz(yz, bound=bound)
        feat_xz = self.encoder_xz(xz, bound=bound)
        
        return torch.cat([feat_xy, feat_yz, feat_xz], dim=-1)
```

**Hash Grid Encoding**: 
- 3D 공간을 xy, yz, xz 평면으로 분해
- 각 평면을 Hash Grid로 인코딩
- 빠른 위치 인코딩 (torch-ngp 스타일)

### 오디오 인코딩

```167:177:scene/motion_net.py
    def encode_audio(self, a):
        # a: [1, 29, 16] or [8, 29, 16], audio features from deepspeech
        # if emb, a should be: [1, 16] or [8, 16]

        # fix audio traininig
        if a is None: return None

        enc_a = self.audio_net(a) # [1/8, 64]
        enc_a = self.audio_att_net(enc_a.unsqueeze(0)) # [1, 64]
            
        return enc_a
```

**AudioNet**: 오디오 특징을 64차원으로 인코딩
**AudioAttNet**: 시간 축에 Attention 적용

### AudioNet 구조

```39:65:scene/motion_net.py
class AudioNet(nn.Module):
    def __init__(self, dim_in=29, dim_aud=64, win_size=16):
        super(AudioNet, self).__init__()
        self.win_size = win_size
        self.dim_aud = dim_aud
        self.encoder_conv = nn.Sequential(  # n x 29 x 16
            nn.Conv1d(dim_in, 32, kernel_size=3, stride=2, padding=1, bias=True),  # n x 32 x 8
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1, bias=True),  # n x 32 x 4
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1, bias=True),  # n x 64 x 2
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),  # n x 64 x 1
            nn.LeakyReLU(0.02, True),
        )
        self.encoder_fc1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(0.02, True),
            nn.Linear(64, dim_aud),
        )

    def forward(self, x):
        half_w = int(self.win_size/2)
        x = x[:, :, 8-half_w:8+half_w]
        x = self.encoder_conv(x).squeeze(-1)
        x = self.encoder_fc1(x)
        return x
```

Conv1D로 시간 축을 압축하고, FC 레이어로 최종 특징 생성.

### AudioAttNet 구조

```8:35:scene/motion_net.py
class AudioAttNet(nn.Module):
    def __init__(self, dim_aud=64, seq_len=8):
        super(AudioAttNet, self).__init__()
        self.seq_len = seq_len
        self.dim_aud = dim_aud
        self.attentionConvNet = nn.Sequential(  # b x subspace_dim x seq_len
            nn.Conv1d(self.dim_aud, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(4, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(2, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True)
        )
        self.attentionNet = nn.Sequential(
            nn.Linear(in_features=self.seq_len, out_features=self.seq_len, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x: [1, seq_len, dim_aud]
        y = x.permute(0, 2, 1)  # [1, dim_aud, seq_len]
        y = self.attentionConvNet(y) 
        y = self.attentionNet(y.view(1, self.seq_len)).view(1, self.seq_len, 1)
        return torch.sum(y * x, dim=1) # [1, dim_aud]
```

시간 축에 Attention을 적용하여 중요한 프레임에 가중치를 부여.

## MouthMotionNetwork

### 구조

```238:285:scene/motion_net.py
class MouthMotionNetwork(nn.Module):
    def __init__(self,
                 audio_dim = 32,
                 ind_dim = 0,
                 args = None,
                 ):
        super(MouthMotionNetwork, self).__init__()

        if 'esperanto' in args.audio_extractor:
            self.audio_in_dim = 44
        elif 'deepspeech' in args.audio_extractor:
            self.audio_in_dim = 29
        elif 'hubert' in args.audio_extractor:
            self.audio_in_dim = 1024
        else:
            raise NotImplementedError
        
        
        self.bound = 0.15

        
        self.individual_dim = ind_dim
        if self.individual_dim > 0:
            self.individual_codes = nn.Parameter(torch.randn(10000, self.individual_dim) * 0.1) 

        # audio network
        self.audio_dim = audio_dim
        self.audio_net = AudioNet(self.audio_in_dim, self.audio_dim)

        self.audio_att_net = AudioAttNet(self.audio_dim)

        # DYNAMIC PART
        self.num_levels = 12
        self.level_dim = 1
        self.encoder_xy, self.in_dim_xy = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=64, log2_hashmap_size=17, desired_resolution=384 * self.bound)
        self.encoder_yz, self.in_dim_yz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=64, log2_hashmap_size=17, desired_resolution=384 * self.bound)
        self.encoder_xz, self.in_dim_xz = get_encoder('hashgrid', input_dim=2, num_levels=self.level_dim, base_resolution=64, log2_hashmap_size=17, desired_resolution=384 * self.bound)

        self.in_dim = self.in_dim_xy + self.in_dim_yz + self.in_dim_xz

        ## sigma network
        self.num_layers = 3
        self.hidden_dim = 32

        self.out_dim = 3
        self.sigma_net = MLP(self.in_dim + self.audio_dim + self.individual_dim, self.out_dim, self.hidden_dim, self.num_layers)
        
        self.aud_ch_att_net = MLP(self.in_dim, self.audio_dim, 32, 2)
```

### Forward Pass

```319:338:scene/motion_net.py
    def forward(self, x, a):
        # x: [N, 3], in [-bound, bound]
        enc_x = self.encode_x(x, bound=self.bound)

        enc_a = self.encode_audio(a)
        enc_w = enc_a.repeat(enc_x.shape[0], 1)
        # aud_ch_att = self.aud_ch_att_net(enc_x)
        # enc_w = enc_a * aud_ch_att

        h = torch.cat([enc_x, enc_w], dim=-1)

        h = self.sigma_net(h)

        d_xyz = h * 1e-2
        d_xyz[..., 0] = d_xyz[..., 0] / 5
        d_xyz[..., 2] = d_xyz[..., 2] / 5
        return {
            'd_xyz': d_xyz,
            # 'ambient_aud' : aud_ch_att.norm(dim=-1, keepdim=True),
        }
```

**차이점**:
- Action Units 입력 없음
- 위치 변형(`d_xyz`)만 출력
- x, z 축 변형을 1/5로 스케일링 (입 움직임이 작게)

## 사용 예시 (렌더링 시)

```python
# render_motion에서
audio_feat = viewpoint_camera.talking_dict["auds"].cuda()  # [1, 29, 16]
exp_feat = viewpoint_camera.talking_dict["au_exp"].cuda()  # [6]

motion_preds = motion_net(pc.get_xyz, audio_feat, exp_feat, None)

# 변형 적용
means3D = pc.get_xyz + motion_preds['d_xyz']
scales = pc.scaling_activation(pc._scaling + motion_preds['d_scale'])
rotations = pc.rotation_activation(pc._rotation + motion_preds['d_rot'])
```

## 주요 특징

1. **공간 제한**: `bound = 0.15` - 변형이 제한된 영역 내에서만 발생
2. **Hash Grid**: 빠른 공간 인코딩
3. **Attention**: 오디오와 AU에 Attention 적용
4. **이중 구조**: Face와 Mouth를 분리하여 세밀한 제어

## 다음 단계

- `06_Rendering.md`: 변형된 Gaussian을 렌더링하는 과정
- `01_Inference_Pipeline.md`: 추론 파이프라인


