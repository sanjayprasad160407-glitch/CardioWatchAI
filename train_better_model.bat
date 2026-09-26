@echo off
call venv\Scripts\activate
python -m ml.train_model --max-per-class 2500 --epochs 20
pause
