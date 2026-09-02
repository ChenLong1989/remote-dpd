@echo off
rem Double-click launcher for the Remote DPD web console.
rem Starts the uvicorn service from the project venv, waits until the HTTP
rem health endpoint answers, then opens the default browser. Closing the
rem minimized "remote-dpd web console" window stops the service.
setlocal
pushd "%~dp0.."
set "PROJECT_ROOT=%CD%"
popd
set "PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe"
set "EXCHANGE_ROOT=%PROJECT_ROOT%\web-console"
set "PORT=8901"
set "URL=http://127.0.0.1:%PORT%/"

if not exist "%PYTHON%" (
  echo [remote-dpd] venv python not found: %PYTHON%
  echo [remote-dpd] create it with:
  echo [remote-dpd]   python -m venv .venv
  echo [remote-dpd]   .venv\Scripts\pip install -e ".[real-hardware]"
  pause
  exit /b 1
)

echo [remote-dpd] starting web console at %URL%
echo [remote-dpd] exchange root: %EXCHANGE_ROOT%
start "remote-dpd web console" /MIN "%PYTHON%" -m remote_dpd.cli ^
  --exchange-root "%EXCHANGE_ROOT%" --mode web --web-port %PORT%

rem Wait up to 30 s for the health endpoint before opening the browser.
rem ("ping -n 2" is a 1 s delay that works in every Windows shell.)
set /a TRIES=30
:wait_loop
ping -n 2 127.0.0.1 >nul
curl -s -o nul "%URL%api/v1/health"
if %errorlevel%==0 goto open
set /a TRIES-=1
if %TRIES% gtr 0 goto wait_loop
echo [remote-dpd] service did not answer within 30 s; opening the browser anyway.

:open
start "" "%URL%"
endlocal
