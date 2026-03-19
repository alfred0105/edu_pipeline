#!/bin/bash
# ============================================================
# setup.sh — macOS Apple Silicon 설치 스크립트
# ============================================================

set -e

echo "[1/3] 가상 환경 생성..."
python3 -m venv .venv
source .venv/bin/activate

echo "[2/3] 패키지 설치..."
pip install --upgrade pip
pip install -r requirements_mac.txt

echo "[3/3] 설치 확인..."
python3 -c "
import platform, torch
print('OS      :', platform.system(), platform.machine())
print('PyTorch :', torch.__version__)
print('MPS OK  :', torch.backends.mps.is_available())
"
python3 config.py

echo ""
echo "============================================================"
echo "  설치 완료!"
echo "  activate: source .venv/bin/activate"
echo "  실행: python main.py -i data/input/video.mp4"
echo "============================================================"
