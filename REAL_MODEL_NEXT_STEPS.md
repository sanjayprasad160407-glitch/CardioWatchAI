# Next step after the improved model

1. Make sure `data/mitdb` contains all MIT-BIH files.
2. Train:
   `python -m ml.train_model --max-per-class 2500 --epochs 20`
3. Check:
   `models/real_model_report/metrics.json`
   `models/real_model_report/classification_report.txt`
4. Create a real two-lead test file:
   `python make_real_csv.py --record 100 --seconds 20`
5. Start the website:
   `python app.py`
6. Upload `real_ecg_100_two_lead.csv`.

Do not report model metrics as clinical performance. This remains an educational/research prototype.
