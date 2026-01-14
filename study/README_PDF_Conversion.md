# PDF 변환 가이드

study 폴더의 Markdown 파일들을 PDF로 변환하는 방법입니다.

## 방법 1: pandoc 사용 (권장)

### 설치

```bash
# 기본 LaTeX 엔진 (영문용)
sudo apt-get update
sudo apt-get install pandoc texlive-latex-base texlive-fonts-recommended

# 한국어 지원 (선택사항, 더 좋은 한글 렌더링)
sudo apt-get install texlive-xetex texlive-lang-korean
```

### 변환

```bash
cd study
bash convert_to_pdf.sh
```

또는 개별 파일 변환:

```bash
# 기본 (pdflatex)
pandoc 00_Overview.md -o 00_Overview.pdf --pdf-engine=pdflatex

# 한국어 지원 (xelatex)
pandoc 00_Overview.md -o 00_Overview.pdf --pdf-engine=xelatex
```

## 방법 2: Python 스크립트 사용

```bash
cd study
python3 convert_to_pdf.py
```

(pandoc과 LaTeX가 설치되어 있어야 합니다)

## 방법 3: 온라인 변환기

다음 온라인 도구를 사용할 수 있습니다:
- [CloudConvert](https://cloudconvert.com/md-to-pdf)
- [Markdown to PDF](https://www.markdowntopdf.com/)

## 방법 4: VS Code 확장

VS Code를 사용하는 경우:
1. "Markdown PDF" 확장 설치
2. Markdown 파일 열기
3. `Ctrl+Shift+P` → "Markdown PDF: Export (pdf)" 선택

## 방법 5: 브라우저 인쇄

1. GitHub에 마크다운 파일 업로드
2. GitHub에서 미리보기
3. 브라우저에서 인쇄 (PDF로 저장)

## 문제 해결

### 한글이 깨질 때
- `xelatex` 엔진 사용
- `texlive-lang-korean` 패키지 설치

### 수식이 렌더링되지 않을 때
- `texlive-latex-extra` 패키지 설치

### 코드 블록이 제대로 표시되지 않을 때
- `--highlight-style` 옵션 확인
- `tango`, `pygments`, `kate` 등 스타일 시도

