#!/bin/bash
# Markdown to PDF 변환 스크립트
# 사용법: bash convert_to_pdf.sh

cd "$(dirname "$0")"

echo "Markdown 파일을 PDF로 변환합니다..."
echo ""

# pandoc 확인
if ! command -v pandoc &> /dev/null; then
    echo "Error: pandoc이 설치되어 있지 않습니다."
    echo ""
    echo "설치 방법:"
    echo "  sudo apt-get update"
    echo "  sudo apt-get install pandoc texlive-latex-base texlive-fonts-recommended"
    echo ""
    echo "한국어 지원을 위해서는:"
    echo "  sudo apt-get install texlive-xetex texlive-lang-korean"
    exit 1
fi

# PDF 변환
md_files=(
    "00_Overview.md"
    "01_Inference_Pipeline.md"
    "02_Data_Preprocessing.md"
    "03_Dataset_Loader.md"
    "04_Gaussian_Model.md"
    "05_Motion_Network.md"
    "06_Rendering.md"
)

success_count=0
for md_file in "${md_files[@]}"; do
    if [ ! -f "$md_file" ]; then
        echo "⚠️  $md_file 파일을 찾을 수 없습니다."
        continue
    fi
    
    pdf_file="${md_file%.md}.pdf"
    echo -n "변환 중: $md_file -> $pdf_file... "
    
    # pandoc로 변환 (한국어 지원을 위해 xelatex 사용 시도)
    if pandoc "$md_file" -o "$pdf_file" \
        --pdf-engine=xelatex \
        -V geometry:margin=1in \
        -V fontsize=11pt \
        -V linestretch=1.2 \
        --highlight-style=tango 2>/dev/null; then
        echo "✓ 완료"
        ((success_count++))
    elif pandoc "$md_file" -o "$pdf_file" \
        --pdf-engine=pdflatex \
        -V geometry:margin=1in \
        -V fontsize=11pt \
        -V linestretch=1.2 \
        --highlight-style=tango 2>/dev/null; then
        echo "✓ 완료 (pdflatex 사용)"
        ((success_count++))
    else
        echo "✗ 실패"
    fi
done

echo ""
echo "총 $success_count/${#md_files[@]} 파일 변환 완료!"


