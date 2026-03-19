@echo off
REM ============================================================
REM setup.bat — Windows PC (RTX 4060 Ti 8GB) 설치 스크립트
REM 관리자 권한 없이 실행 가능
REM ============================================================

echo [1/4] Python 가상 환경 생성...
python -m venv .venv
call .venv\Scripts\activate.bat

echo [2/4] PyTorch (CUDA 12.4) 설치...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo [3/4] 나머지 패키지 설치...
pip install -r requirements_pc.txt

echo [4/4] 설치 확인...
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
python config.py

echo.
echo ============================================================
echo  설치 완료!
echo  사용법: python main.py -i data/input/video.mp4
echo ============================================================
pause
