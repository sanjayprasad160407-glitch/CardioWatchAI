@echo off
py -3.12 -m venv venv
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python setup_demo.py
python create_demo_ecg.py
python app.py
