# CardioWatch AI

Real-Time Wearable Cardiac Arrhythmia Detection & Risk Alerting — educational software prototype.

## What is included

- Flask full-stack web application
- Top-center navigation and professional cardiac monitoring dashboard
- ECG CSV upload and preprocessing
- Demo ML fallback using a synthetic-data Random Forest classifier
- Real 1D CNN training on MIT-BIH Arrhythmia Database via WFDB/PhysioNet
- Record-level train/validation/test split
- Precision / recall / F1 / accuracy evaluation
- ECG waveform visualization with Plotly
- Heart-rate prototype estimate
- Signal-quality heuristic
- Educational alert engine
- Gradient saliency when the real CNN is active
- Recent-events SQLite history
- Review event endpoint
- FHIR Observation export
- Local event-summary chatbot and optional OpenAI Responses API chatbot
- GitHub + Render deployment files

## Safety

This is an educational/research software prototype. Model outputs, alerts and chatbot responses are not medical diagnoses and are not validated for clinical decision-making or emergency response.

## Windows / VS Code setup

Use Python 3.12.x. Do not use the Python 3.14 environment created earlier for this project.

```powershell
py -3.12 -m venv venv
.\\venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python setup_demo.py
python create_demo_ecg.py
python app.py
```

Open http://127.0.0.1:5000

Upload `test_ecg.csv`.

## Download real ECG data

```powershell
python download_mitdb.py
```

The download is stored in `data/mitdb`.

## Train the real CNN

```powershell
python -m ml.train_model
```

Optional larger run:

```powershell
python -m ml.train_model --max-per-class 1500 --epochs 20
```

When training completes, the real model replaces the demo model automatically because `ml/predict.py` prefers `models/arrhythmia_model.keras` when it exists.

## Test a real record

After downloading MIT-BIH, you can create a CSV from a local record with:

```powershell
python make_real_csv.py --record 100 --seconds 20
```

Then upload the generated CSV in the website.

## Optional OpenAI chatbot

Create `.env` from `.env.example` and set your API key:

```text
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5.6-luna
```

Never put an API key in JavaScript or commit `.env` to GitHub.

## GitHub

Create a new empty GitHub repository named `CardioWatchAI`, then in the project folder:

```powershell
git init
git add .
git commit -m "Initial CardioWatch AI project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/CardioWatchAI.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your real GitHub username. Do not copy a placeholder URL literally.

## Render

In Render, create **New → Web Service**, connect the GitHub repository, and use:

- Runtime: Python
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn --bind 0.0.0.0:$PORT app:app`

The repository also includes `render.yaml` and `.python-version`.

If you want the cloud chatbot, add `OPENAI_API_KEY` as a secret environment variable in Render. `OPENAI_MODEL` can stay as `gpt-5.6-luna`.

## Important Render note

Do not train the large MIT-BIH model during the Render build. Train locally first, verify it, and then commit the resulting model files only when their size is appropriate for your repository. Render web services have an ephemeral filesystem, so persistent uploaded patient data should not be treated as permanent storage in this prototype.


## Real model note
The real-model pipeline uses a two-stage design: a patient-disjoint Normal-vs-Abnormal screening CNN followed by a three-class subtype CNN (Normal/Ventricular/OtherAbnormal). This is deliberately used instead of presenting a misleading four-class headline accuracy. The MIT-BIH Arrhythmia Database is a research database of 48 two-channel ECG recordings with beat annotations; see PhysioNet for dataset details.


## If training stops before Epoch 1

Run:

```powershell
python -m ml.train_model --epochs 15 --max-per-class 4000
```

The included training code is compatible with the TensorFlow/Keras environment specified in `requirements.txt` and does not pass unsupported `label_smoothing` arguments to the sparse cross-entropy loss.

## UI Theme Update
The dashboard uses a dark clinical-tech "Midnight ECG" theme with cyan/mint accents, ECG grid styling, a heartbeat signal motif, project feature chips, and a centered top navigation. The visual changes are presentation-oriented and do not alter the ML data or medical-claim limitations of the application.
