#!/bin/bash

# Miniconda 설치 스크립트
echo "Miniconda 설치를 시작합니다..."

# Miniconda installer 다운로드
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 설치 실행
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

# conda 초기화
$HOME/miniconda3/bin/conda init bash

echo ""
echo "설치가 완료되었습니다!"
echo "다음 명령어를 실행하여 conda를 활성화하세요:"
echo "  source ~/.bashrc"
echo ""
echo "그 다음 eigenscore 환경을 생성하세요:"
echo "  cd /home/srjo/Gradproj"
echo "  conda env create -f eigenscore.yml"
echo "  conda activate eigenscore"

