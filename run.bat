@echo off
chcp 65001 >nul
title edu_pipeline

echo ========================================
echo   edu_pipeline 실행기
echo ========================================
echo.

:: 입력 파일 확인
if "%~1"=="" (
    echo [사용법]
    echo   방법 1: 영상 파일을 이 파일 위에 드래그 앤 드롭
    echo   방법 2: run.bat "영상파일경로"
    echo   방법 3: 아래에서 data\input 폴더의 파일 선택
    echo.

    :: data\input 폴더에 파일이 있으면 목록 표시
    setlocal enabledelayedexpansion
    set count=0
    for %%f in ("data\input\*.*") do (
        set /a count+=1
        set "file_!count!=%%f"
        echo   [!count!] %%~nxf
    )

    if !count!==0 (
        echo   data\input 폴더에 파일이 없습니다.
        echo   영상 파일을 data\input 폴더에 넣고 다시 실행하세요.
        echo.
        pause
        exit /b 1
    )

    echo.
    set /p choice="번호를 선택하세요 (1-!count!): "
    set "INPUT_FILE=!file_%choice%!"
    endlocal & set "INPUT_FILE=%INPUT_FILE%"
) else (
    set "INPUT_FILE=%~1"
)

echo.
echo  입력 파일: %INPUT_FILE%
echo ========================================
echo.

:: 파이프라인 실행
call .venv\Scripts\activate
python main.py -i "%INPUT_FILE%"

echo.
echo ========================================
echo   완료!
echo ========================================
pause
