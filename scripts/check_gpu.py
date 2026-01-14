#!/usr/bin/env python3
"""GPU 사용 여부 확인 스크립트"""

import torch
import sys

print("=" * 50)
print("GPU 사용 여부 확인")
print("=" * 50)

# PyTorch CUDA 사용 가능 여부
print(f"\n1. PyTorch CUDA 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   - CUDA 버전: {torch.version.cuda}")
    print(f"   - GPU 개수: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"   - GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"     메모리: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    
    # 현재 사용 중인 GPU 메모리
    print(f"\n2. 현재 GPU 메모리 사용량:")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"   GPU {i}: 할당됨 {allocated:.2f} GB, 예약됨 {reserved:.2f} GB")
    
    # 간단한 GPU 테스트
    print(f"\n3. GPU 연산 테스트:")
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print(f"   ✓ GPU 연산 성공!")
        del x, y, z
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"   ✗ GPU 연산 실패: {e}")
else:
    print("\n⚠️  CUDA를 사용할 수 없습니다!")
    print("   CPU 모드로 실행 중일 수 있습니다.")

print("\n" + "=" * 50)

