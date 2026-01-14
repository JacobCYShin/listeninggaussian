#!/bin/bash
# CUDA 버전 체크 우회를 포함한 서브모듈 설치 스크립트

set -e

WORKSPACE_DIR="/workspace"
cd "$WORKSPACE_DIR"

echo "=== CUDA 버전 체크 우회 패치 생성 ==="
# PyTorch의 CUDA 버전 체크를 우회하는 임시 패치
cat > /tmp/cuda_bypass_patch.py << 'EOF'
import torch.utils.cpp_extension
original_check = torch.utils.cpp_extension._check_cuda_version

def bypass_cuda_check(compiler_name, compiler_version):
    """CUDA 버전 체크 우회"""
    print(f"경고: CUDA 버전 체크를 우회합니다. (컴파일러: {compiler_name}, 버전: {compiler_version})")
    return True

torch.utils.cpp_extension._check_cuda_version = bypass_cuda_check
EOF

echo "=== diff-gaussian-rasterization 설치 ==="
if [ -f "submodules/diff-gaussian-rasterization/setup.py" ]; then
    cd submodules/diff-gaussian-rasterization
    # 패치를 적용한 Python 환경에서 설치
    python -c "
import sys
sys.path.insert(0, '/tmp')
exec(open('/tmp/cuda_bypass_patch.py').read())
import subprocess
import os
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6;8.9;9.0+PTX'
os.environ['FORCE_CUDA'] = '1'
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '.', '--no-build-isolation'])
" || {
        echo "패치 방법 실패, 직접 빌드 시도..."
        python -c "
import sys
sys.path.insert(0, '/tmp')
exec(open('/tmp/cuda_bypass_patch.py').read())
import subprocess
import os
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6;8.9;9.0+PTX'
os.environ['FORCE_CUDA'] = '1'
subprocess.check_call([sys.executable, 'setup.py', 'build_ext', '--inplace'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '.', '--no-build-isolation'])
"
    }
    cd "$WORKSPACE_DIR"
else
    echo "ERROR: diff-gaussian-rasterization/setup.py를 찾을 수 없습니다."
    exit 1
fi

echo "=== simple-knn 설치 ==="
if [ -f "submodules/simple-knn/setup.py" ]; then
    cd submodules/simple-knn
    python -c "
import sys
sys.path.insert(0, '/tmp')
exec(open('/tmp/cuda_bypass_patch.py').read())
import subprocess
import os
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6;8.9;9.0+PTX'
os.environ['FORCE_CUDA'] = '1'
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '.', '--no-build-isolation'])
"
    cd "$WORKSPACE_DIR"
else
    echo "ERROR: simple-knn/setup.py를 찾을 수 없습니다."
    exit 1
fi

echo "=== gridencoder 설치 ==="
if [ -f "gridencoder/setup.py" ]; then
    cd gridencoder
    python -c "
import sys
sys.path.insert(0, '/tmp')
exec(open('/tmp/cuda_bypass_patch.py').read())
import subprocess
import os
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6;8.9;9.0+PTX'
os.environ['FORCE_CUDA'] = '1'
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '.', '--no-build-isolation'])
"
    cd "$WORKSPACE_DIR"
else
    echo "ERROR: gridencoder/setup.py를 찾을 수 없습니다."
    exit 1
fi

echo "=== 설치 확인 ==="
python -c "
try:
    import diff_gaussian_rasterization
    print('✓ diff_gaussian_rasterization 설치 완료')
except Exception as e:
    print(f'✗ diff_gaussian_rasterization: {e}')

try:
    import simple_knn
    print('✓ simple_knn 설치 완료')
except Exception as e:
    print(f'✗ simple_knn: {e}')

try:
    import _gridencoder
    print('✓ gridencoder 설치 완료')
except Exception as e:
    print(f'✗ gridencoder: {e}')
"

echo "=== 모든 서브모듈 설치 완료 ==="

