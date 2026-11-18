@echo off
setlocal
where py >nul 2>nul
if errorlevel 1 (
  echo Python launcher not found. Install Python 3.11 or 3.12 first.
  pause
  exit /b 1
)
if not exist .venv py -3.11 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 goto failed
python -m src.generate_data
python -m src.features
python -m src.train
if errorlevel 1 goto failed
python -m pytest -q
if errorlevel 1 goto failed
echo.
echo Starting Mango Shelf-Life web application...
echo Open http://127.0.0.1:8000 in your browser.
uvicorn src.api:app --reload
goto end
:failed
echo Setup or execution failed. Read the error above.
pause
:end
endlocal
