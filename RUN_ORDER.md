# CardioWatch AI — Recommended Run Order

1. Create Python 3.12 environment and install requirements.
2. Copy/verify `data/mitdb` (MIT-BIH).
3. Run `python diagnose_dataset.py`.
4. Train with `python -m ml.train_model --epochs 15 --max-per-class 4000`.
5. Create a real test CSV with `python make_real_csv.py --record 100 --seconds 20`.
6. Start the site with `python app.py`.
7. Upload the generated two-lead CSV in the browser.
8. Review balanced accuracy + per-class metrics before presenting the model.
9. For GitHub, keep `models/` artifacts out of the first push; use Git LFS or model storage for large binaries.
10. Deploy Flask on Render with `pip install -r requirements.txt` and `gunicorn --bind 0.0.0.0:$PORT app:app`.
