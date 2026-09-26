# Real ECG training — CardioWatch AI

The first training run showed very high training accuracy but very low validation/test accuracy. That means the model was overfitting and should not be used as the real model in the website.

This training version improves the experiment by using:

- fixed inter-patient DS1/DS2 records
- group-aware, stratified validation inside DS1
- per-class sampling limits
- light signal augmentation
- a residual 1D CNN
- early stopping on validation accuracy
- balanced accuracy and a full classification report

## Commands

From the project root:

```powershell
.\venv\Scripts\Activate.ps1
python diagnose_dataset.py
python -m ml.train_model --max-per-class 1500 --epochs 20
```

When training finishes, inspect:

```text
models\\real_model_report\\classification_report.txt
models\\real_model_report\\metrics.json
models\\real_model_report\\confusion_matrix.json
```

The website automatically uses `models/arrhythmia_model.keras` when that model and its metadata files exist.

The MIT-BIH database is a research dataset and this application remains an educational/research prototype; model performance is not clinical validation.

## TensorFlow/Keras compatibility note

This project intentionally uses `SparseCategoricalCrossentropy()` without the `label_smoothing` argument. Some installed Keras/TensorFlow combinations do not accept `label_smoothing` for the sparse loss constructor, which otherwise causes a `TypeError` before training starts.
