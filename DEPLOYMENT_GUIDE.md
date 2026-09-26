# CardioWatch AI — VS Code + GitHub + Render guide

## A. Run in VS Code on Windows

1. Extract the ZIP and open the folder that contains `app.py`.
2. In VS Code: Terminal → New Terminal.
3. Run:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python setup_demo.py
python create_demo_ecg.py
python app.py
```

4. Open `http://127.0.0.1:5000`.
5. Select `test_ecg.csv`, sampling rate `360`, and click **Analyze ECG**.

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

## B. Download and train the real model

From the project root:

```powershell
python download_mitdb.py
python -m ml.train_model
```

The MIT-BIH database should be downloaded under `data\mitdb`. The training script then writes the real CNN artifacts into `models\`.

For a larger training run:

```powershell
python -m ml.train_model --max-per-class 1500 --epochs 20
```

After successful training, upload a real-record CSV created with:

```powershell
python make_real_csv.py --record 100 --seconds 20
```

## C. Push to GitHub

Create a **new empty repository** on GitHub named `CardioWatchAI`. Do not initialize it with a README if you want the commands below to be the only first commit.

Then in VS Code terminal:

```powershell
git init
git add .
git commit -m "Initial CardioWatch AI application"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/CardioWatchAI.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual GitHub username.

Do not copy the literal placeholder. A `Repository not found` error commonly means the repository URL or account name is wrong, or the account does not have access to that repository.

Check the remote with:

```powershell
git remote -v
```

If you need to correct it:

```powershell
git remote set-url origin https://github.com/YOUR_USERNAME/CardioWatchAI.git
```

### Commit the real CNN model

The `.gitignore` intentionally ignores large trained artifacts until you verify them. After training locally, inspect file sizes. If the model is suitable for your Git repository, add it explicitly:

```powershell
git add -f models/arrhythmia_model.keras models/label_encoder.joblib models/model_meta.json

git commit -m "Add trained ECG CNN model"
git push
```

If your Git provider rejects the file because it is too large, use Git LFS or another model-storage mechanism; do not remove the ignore rule blindly.

## D. Deploy on Render

1. Sign in to Render.
2. Select **New → Web Service**.
3. Connect your GitHub account.
4. Select the `CardioWatchAI` repository.
5. Set:

- Runtime: `Python`
- Build Command: `pip install -r requirements.txt`
- Start Command: `gunicorn --bind 0.0.0.0:$PORT app:app`
- Health Check Path: `/api/health`

The included `render.yaml` already contains these settings.

6. Deploy.
7. Open the Render-provided HTTPS URL.

## E. Enable the cloud AI chatbot

In Render → your service → Environment, add:

```text
OPENAI_API_KEY=<your API key>
OPENAI_MODEL=gpt-5.6-luna
```

Keep the API key in Render environment variables. Never put it in HTML/JavaScript and never commit `.env`.

## F. Understand which model is live

The top-right status badge calls `/api/health`.

- `Real 1D CNN` means the trained `arrhythmia_model.keras` is present.
- `Demo ML` means the site is running with the included synthetic-data Random Forest fallback.

The demo fallback makes deployment/testing possible, but it must not be presented as a clinically validated ECG model.

## G. Render filesystem reminder

The web service filesystem should not be treated as permanent patient-data storage. This project uses SQLite/uploads for demonstration only. For a real product, use managed persistent storage and appropriate security/privacy controls.
