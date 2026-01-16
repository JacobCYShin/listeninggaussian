#!/usr/bin/env python3
"""
Markdown to PDF 변환 스크립트
pandoc 또는 markdown2pdf를 사용합니다.
"""

import os
import subprocess
import sys

def check_pandoc():
    """pandoc이 설치되어 있는지 확인"""
    try:
        result = subprocess.run(['pandoc', '--version'], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def convert_md_to_pdf_pandoc(md_file, pdf_file):
    """pandoc을 사용하여 MD를 PDF로 변환"""
    cmd = [
        'pandoc',
        md_file,
        '-o', pdf_file,
        '--pdf-engine=pdflatex',
        '--variable', 'geometry:margin=1in',
        '--variable', 'fontsize=11pt',
        '--variable', 'linestretch=1.2',
        '--highlight-style=tango'
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting {md_file}: {e.stderr.decode()}")
        return False
    except FileNotFoundError:
        return False

def convert_md_to_pdf_simple(md_file, pdf_file):
    """간단한 방법: xelatex 사용 (한국어 지원)"""
    cmd = [
        'pandoc',
        md_file,
        '-o', pdf_file,
        '--pdf-engine=xelatex',
        '--variable', 'geometry:margin=1in',
        '--variable', 'fontsize=11pt',
        '--variable', 'linestretch=1.2',
        '--variable', 'CJKmainfont="Noto Sans CJK KR"',
        '--highlight-style=tango'
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        # xelatex 실패 시 pdflatex로 재시도
        return convert_md_to_pdf_pandoc(md_file, pdf_file)
    except FileNotFoundError:
        return False

def main():
    study_dir = os.path.dirname(os.path.abspath(__file__))
    md_files = [
        '00_Overview.md',
        '01_Inference_Pipeline.md',
        '02_Data_Preprocessing.md',
        '03_Dataset_Loader.md',
        '04_Gaussian_Model.md',
        '05_Motion_Network.md',
        '06_Rendering.md'
    ]
    
    if not check_pandoc():
        print("Error: pandoc이 설치되어 있지 않습니다.")
        print("\n설치 방법:")
        print("  Ubuntu/Debian: sudo apt-get install pandoc texlive-xetex")
        print("  또는: sudo apt-get install pandoc texlive-latex-base")
        sys.exit(1)
    
    print("Markdown 파일을 PDF로 변환합니다...")
    print(f"작업 디렉토리: {study_dir}\n")
    
    success_count = 0
    for md_file in md_files:
        md_path = os.path.join(study_dir, md_file)
        pdf_file = md_file.replace('.md', '.pdf')
        pdf_path = os.path.join(study_dir, pdf_file)
        
        if not os.path.exists(md_path):
            print(f"⚠️  {md_file} 파일을 찾을 수 없습니다.")
            continue
        
        print(f"변환 중: {md_file} -> {pdf_file}...", end=' ', flush=True)
        
        # xelatex로 시도 (한국어 지원), 실패 시 pdflatex
        if convert_md_to_pdf_simple(md_path, pdf_path):
            print("✓ 완료")
            success_count += 1
        else:
            print("✗ 실패")
    
    print(f"\n총 {success_count}/{len(md_files)} 파일 변환 완료!")
    
    if success_count < len(md_files):
        print("\n참고: PDF 변환을 위해서는 다음이 필요합니다:")
        print("  - pandoc: sudo apt-get install pandoc")
        print("  - LaTeX: sudo apt-get install texlive-latex-base")
        print("  - 또는: sudo apt-get install texlive-xetex (한국어 지원)")

if __name__ == '__main__':
    main()


