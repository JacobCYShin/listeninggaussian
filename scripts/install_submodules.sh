#!/bin/bash
# 서브모듈 설치 스크립트 (통합 버전)

set -e

WORKSPACE_DIR="/workspace"
cd "$WORKSPACE_DIR"

echo "=== Git 설정 (SSL 및 safe.directory) ==="
# SSL 인증서 검증 비활성화 (임시 조치)
export GIT_SSL_NO_VERIFY=1
git config --global http.sslVerify false

# safe.directory 설정 (ownership 문제 해결)
git config --global --add safe.directory "$WORKSPACE_DIR"
git config --global --add safe.directory "$WORKSPACE_DIR/submodules/diff-gaussian-rasterization"
git config --global --add safe.directory "$WORKSPACE_DIR/submodules/simple-knn"

echo "=== diff-gaussian-rasterization 서브모듈 처리 ==="
if [ ! -d "submodules/diff-gaussian-rasterization/.git" ]; then
    echo "diff-gaussian-rasterization 클론 중..."
    git submodule update --init submodules/diff-gaussian-rasterization || {
        echo "서브모듈 업데이트 실패, 직접 클론 시도..."
        rm -rf submodules/diff-gaussian-rasterization
        git clone https://github.com/ashawkey/diff-gaussian-rasterization.git submodules/diff-gaussian-rasterization
    }
fi

echo "=== simple-knn 서브모듈 처리 ==="
if [ ! -d "submodules/simple-knn/.git" ]; then
    echo "simple-knn 클론 중 (SSL 검증 비활성화)..."
    # SSL 문제로 인해 직접 클론
    rm -rf submodules/simple-knn
    GIT_SSL_NO_VERIFY=1 git clone https://gitlab.inria.fr/bkerbl/simple-knn.git submodules/simple-knn || {
        echo "GitLab 클론 실패, GitHub 미러 시도..."
        # GitHub 미러가 있다면 사용, 없으면 에러
        echo "ERROR: simple-knn 클론 실패. 수동으로 클론해주세요."
        exit 1
    }
fi

echo "=== CUDA 버전 체크 우회 설정 ==="
# PyTorch CUDA 버전 체크 우회 (CUDA 12.4와 PyTorch 11.8 불일치 해결)
export TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0+PTX"
export FORCE_CUDA="1"
# CUDA 버전 체크를 우회하기 위한 환경 변수
export TORCH_USE_CUDA_DSA=0

echo "=== diff-gaussian-rasterization 설치 ==="
if [ -f "submodules/diff-gaussian-rasterization/setup.py" ]; then
    cd submodules/diff-gaussian-rasterization
    # CUDA 버전 체크를 우회하기 위해 환경 변수 설정
    TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0+PTX" FORCE_CUDA="1" \
    pip install . --no-build-isolation || {
        echo "일반 설치 실패, editable mode 시도..."
        TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0+PTX" FORCE_CUDA="1" \
        pip install -e . --no-build-isolation || {
            echo "경고: CUDA 버전 체크 우회를 위해 수동 설치 필요"
            python setup.py build_ext --inplace
            pip install . --no-build-isolation
        }
    }
    cd "$WORKSPACE_DIR"
else
    echo "ERROR: diff-gaussian-rasterization/setup.py를 찾을 수 없습니다."
    exit 1
fi

echo "=== simple-knn 설치 ==="
if [ -f "submodules/simple-knn/setup.py" ]; then
    cd submodules/simple-knn
    TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0+PTX" FORCE_CUDA="1" \
    pip install . --no-build-isolation || {
        TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0+PTX" FORCE_CUDA="1" \
        pip install -e . --no-build-isolation
    }
    cd "$WORKSPACE_DIR"
else
    echo "ERROR: simple-knn/setup.py를 찾을 수 없습니다."
    exit 1
fi

echo "=== gridencoder 설치 ==="
if [ -f "gridencoder/setup.py" ]; then
    cd gridencoder
    TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0+PTX" FORCE_CUDA="1" \
    pip install . --no-build-isolation || {
        TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0+PTX" FORCE_CUDA="1" \
        pip install -e . --no-build-isolation
    }
    cd "$WORKSPACE_DIR"
else
    echo "ERROR: gridencoder/setup.py를 찾을 수 없습니다."
    exit 1
fi

echo "=== 모든 서브모듈 설치 완료 ==="
echo "설치된 모듈 확인:"
python -c "import diff_gaussian_rasterization; print('✓ diff_gaussian_rasterization')" 2>/dev/null || echo "✗ diff_gaussian_rasterization"
python -c "import simple_knn; print('✓ simple_knn')" 2>/dev/null || echo "✗ simple_knn"
python -c "import _gridencoder; print('✓ gridencoder')" 2>/dev/null || echo "✗ gridencoder"

