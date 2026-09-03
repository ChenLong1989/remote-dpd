@echo off
rem Double-click launcher for the Remote DPD web console.
rem - No service running: start it from the project venv, wait for the HTTP
rem   health endpoint, then open the default browser.
rem - Service already running: ask whether to restart it (picking up the
rem   latest local code changes) or keep it and just open the browser.
rem Closing the minimized "remote-dpd web console" window stops the service.
setlocal
pushd "%~dp0.."
set "PROJECT_ROOT=%CD%"
popd
set "PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe"
set "EXCHANGE_ROOT=%PROJECT_ROOT%\web-console"
set "PORT=8901"
set "URL=http://127.0.0.1:%PORT%/"
set "HEALTH=%URL%api/v1/health"

if not exist "%PYTHON%" (
  echo [remote-dpd] venv python not found: %PYTHON%
  echo [remote-dpd] create it with:
  echo [remote-dpd]   python -m venv .venv
  echo [remote-dpd]   .venv\Scripts\pip install -e ".[real-hardware]"
  pause
  exit /b 1
)

curl -s -o nul "%HEALTH%"
if not %errorlevel%==0 goto start_service

rem A console is already running; ask before restarting it.
rem MessageBox returns 6 for Yes and 7 for No as the exit code.
powershell -NoProfile -Command "Add-Type -AssemblyName System.Windows.Forms; exit [int][System.Windows.Forms.MessageBox]::Show('The Remote DPD web console is already running at %URL%. Restart it now to pick up the latest local code changes? Choose No to keep the current service (for example while a measurement is active).', 'Remote DPD Workbench', 'YesNo', 'Question')"
if %errorlevel%==6 goto restart_service
echo [remote-dpd] keeping the existing service; opening the browser.
start "" "%URL%"
exit /b 0

:restart_service
echo [remote-dpd] stopping the running web console...
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*remote_dpd.cli*' -and $_.CommandLine -like '*web-console*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
rem Wait up to 10 s for the port to be released. ("ping -n 2" is a 1 s
rem delay that works in every Windows shell.)
set /a TRIES=10
:wait_port_free
ping -n 2 127.0.0.1 >nul
netstat -ano | findstr /C:":%PORT% " | findstr /C:"LISTENING" >nul
if not %errorlevel%==0 goto start_service
set /a TRIES-=1
if %TRIES% gtr 0 goto wait_port_free
echo [remote-dpd] warning: port %PORT% is still occupied after stopping the service.

:start_service
echo [remote-dpd] starting web console at %URL%
echo [remote-dpd] exchange root: %EXCHANGE_ROOT%
start "remote-dpd web console" /MIN "%PYTHON%" -m remote_dpd.cli ^
  --exchange-root "%EXCHANGE_ROOT%" --mode web --web-port %PORT%

rem Wait up to 30 s for the health endpoint before opening the browser.
set /a TRIES=30
:wait_loop
ping -n 2 127.0.0.1 >nul
curl -s -o nul "%HEALTH%"
if %errorlevel%==0 goto open
set /a TRIES-=1
if %TRIES% gtr 0 goto wait_loop
echo [remote-dpd] service did not answer within 30 s; opening the browser anyway.

:open
start "" "%URL%"
endlocal
