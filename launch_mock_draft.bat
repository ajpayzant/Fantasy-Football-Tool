@echo off
REM -- Fantasy Mock Draft launcher (local) ---------------------------------
REM Pin this (or the .vbs) to the taskbar. Serves the mock draft tool on
REM http://localhost:8502. If it is ALREADY running, this just opens the
REM browser to it instead of starting a second copy.
REM
REM Port 8502 is deliberate: the PLL BOSS Tool lives on 8501 and its launcher
REM treats any listener on 8501 as itself. Do not change this to 8501.

cd /d "%~dp0"
title Fantasy Mock Draft

REM If something is already listening on 8502, just open the browser to it.
netstat -ano | findstr /R /C:":8502 .*LISTENING" >nul 2>&1
if %errorlevel%==0 (
    start "" http://localhost:8502
    echo Fantasy Mock Draft is already running - opened it in your browser.
    echo Close this window; the tool keeps running.
    timeout /t 3 >nul
    exit /b 0
)

REM Not running yet -- open the browser shortly after Streamlit starts, then run.
start "" /b cmd /c "timeout /t 4 >nul & start http://localhost:8502"
python -m streamlit run app.py --server.headless true --server.port 8502

REM If Streamlit exits with an error, keep the window so you can read it.
if errorlevel 1 pause
