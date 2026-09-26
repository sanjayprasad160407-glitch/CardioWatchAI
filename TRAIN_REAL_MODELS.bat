@echo off
call .\venv\Scripts\activate
python -m ml.train_model --epochs 15 --max-per-class 4000
pause
