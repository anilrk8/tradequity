@echo off
echo Stopping any running Streamlit / Python processes...
taskkill /F /IM python.exe /T >nul 2>&1
taskkill /F /IM streamlit.exe /T >nul 2>&1
timeout /t 1 /nobreak >nul

echo Clearing bytecode cache...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
del /s /q *.pyc >nul 2>&1

echo Starting TradeQuity...
cd /d "%~dp0"
c:\Users\KAN1PU\.conda\envs\SmartAnalytics\python.exe -m streamlit run app.py --server.port 8501
