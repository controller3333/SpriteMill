@echo off
rem SpriteMill VideoLab をローカル起動する (Windows)
cd /d %~dp0
where python >nul 2>nul
if errorlevel 1 (
  echo Python 3.10 以上をインストールしてください: https://www.python.org/downloads/
  pause
  exit /b 1
)
python -m pip install -q -r requirements.txt
python videolab_server.py %*
pause
