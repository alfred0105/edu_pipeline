@echo off
chcp 65001 >nul
title edu_pipeline Web UI
call .venv\Scripts\activate
python web_ui.py
pause
