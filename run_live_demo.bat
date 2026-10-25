@echo off
chcp 65001 > nul
title Vietnamese KWS CNN - Live Microphone Demo
echo ========================================================================
echo   VIETNAMESE KEYWORD SPOTTING (KWS) — LAPTOP LIVE-MIC PROTOTYPE DEMO
echo   Governance: EXPERIMENTAL_LIVE_MIC_DEMO_NOT_VALIDATED
echo               NOT_VALIDATED_ON_NATURAL_SPEECH
echo               NO_DEPLOYABLE_SYSTEM_AT_CURRENT_SPECIFICATION
echo ========================================================================
echo.

cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXEC=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXEC=python"
)

echo Dang khoi dong ung dung demo bang %PYTHON_EXEC%...
%PYTHON_EXEC% experiments\vieneu_kws_research_v13_4\demo\live_mic_app.py %*

echo.
echo Phien lam viec ket thuc.
pause
