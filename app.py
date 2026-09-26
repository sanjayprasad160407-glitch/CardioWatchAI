import os
import sqlite3
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

from ml.predict import analyze_ecg, assets_status
from ml.risk_engine import compute_alert

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
DB_PATH = BASE_DIR / "database" / "cardiowatch.db"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

load_dotenv(BASE_DIR / ".env")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {"csv"}


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def init_db():
    conn = get_db()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS patients (
            patient_id TEXT PRIMARY KEY,
            age INTEGER,
            sex TEXT,
            risk_profile TEXT NOT NULL DEFAULT 'Standard',
            notes TEXT,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            signal_quality REAL NOT NULL,
            alert_level TEXT NOT NULL,
            alert_reason TEXT NOT NULL,
            heart_rate INTEGER,
            subtype TEXT,
            subtype_confidence REAL,
            sampling_rate REAL NOT NULL,
            created_at TEXT NOT NULL,
            reviewed_at TEXT
        );
        """
    )
    conn.commit()
    # Migrate databases created by older project versions.
    for statement in (
        "ALTER TABLE events ADD COLUMN subtype TEXT",
        "ALTER TABLE events ADD COLUMN subtype_confidence REAL",
    ):
        try:
            conn.execute(statement)
            conn.commit()
        except sqlite3.OperationalError:
            pass
    conn.close()


init_db()


def row_to_dict(row):
    return dict(row) if row else None


def get_patient(patient_id):
    conn = get_db()
    row = conn.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,)).fetchone()
    conn.close()
    return row_to_dict(row)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def estimate_heart_rate(signal, fs):
    """Simple peak-based prototype estimate; not a clinical measurement."""
    x = np.asarray(signal, dtype=np.float32)
    if len(x) < int(fs * 2):
        return None

    x = np.nan_to_num(x)
    x = x - np.mean(x)
    std = float(np.std(x))
    if std < 1e-7:
        return None

    try:
        from scipy.signal import find_peaks
        distance = max(1, int(0.30 * fs))
        prominence = max(0.20 * std, 1e-6)
        peaks, _ = find_peaks(x, distance=distance, prominence=prominence)
    except Exception:
        return None

    if len(peaks) < 2:
        return None
    rr = np.diff(peaks) / float(fs)
    rr = rr[(rr >= 0.30) & (rr <= 2.0)]
    if len(rr) == 0:
        return None
    bpm = 60.0 / float(np.median(rr))
    return int(np.clip(round(bpm), 30, 220))


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "File too large. Maximum upload size is 10 MB."}), 413


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    status = assets_status()
    return jsonify({
        "status": "online",
        "model_ready": status["ready"],
        "model_type": status["type"],
        "message": status["message"],
        "api_ready": bool(os.getenv("OPENAI_API_KEY", "").strip()),
        "service": "CardioWatch AI",
    })


@app.route("/api/patient", methods=["POST"])
def save_patient():
    data = request.get_json(silent=True) or {}
    patient_id = str(data.get("patient_id") or "Demo-001").strip()
    if not patient_id:
        return jsonify({"error": "Patient ID is required."}), 400

    age = data.get("age")
    if age not in (None, ""):
        try:
            age = int(age)
        except (TypeError, ValueError):
            return jsonify({"error": "Age must be a whole number."}), 400
        if age < 1 or age > 120:
            return jsonify({"error": "Age must be between 1 and 120."}), 400
    else:
        age = None

    sex = str(data.get("sex") or "").strip()
    risk_profile = str(data.get("risk_profile") or "Standard").strip()
    notes = str(data.get("notes") or "").strip()
    updated_at = now_iso()

    conn = get_db()
    conn.execute(
        """
        INSERT INTO patients(patient_id, age, sex, risk_profile, notes, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(patient_id) DO UPDATE SET
            age=excluded.age,
            sex=excluded.sex,
            risk_profile=excluded.risk_profile,
            notes=excluded.notes,
            updated_at=excluded.updated_at
        """,
        (patient_id, age, sex, risk_profile, notes, updated_at),
    )
    conn.commit()
    conn.close()
    return jsonify({"ok": True, "patient": get_patient(patient_id)})


@app.route("/api/patient/<path:patient_id>")
def patient(patient_id):
    return jsonify(get_patient(patient_id) or {
        "patient_id": patient_id,
        "risk_profile": "Standard",
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    file = request.files.get("file")
    if file is None:
        return jsonify({"error": "Upload an ECG CSV file."}), 400
    if not file.filename:
        return jsonify({"error": "Choose an ECG CSV file."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Only CSV files are supported."}), 400

    patient_id = str(request.form.get("patient_id") or "Demo-001").strip() or "Demo-001"
    try:
        fs = float(request.form.get("sampling_rate") or 360)
        if not (10 <= fs <= 2000):
            raise ValueError
    except ValueError:
        return jsonify({"error": "Sampling rate must be between 10 and 2000 Hz."}), 400

    safe_original = secure_filename(file.filename) or "ecg.csv"
    stored_name = f"{uuid.uuid4().hex}_{safe_original}"
    path = UPLOAD_DIR / stored_name
    file.save(path)

    try:
        result = analyze_ecg(path, fs)
        alert, reason = compute_alert(
            prediction=result["prediction"],
            confidence=result["confidence"],
            signal_quality=result["signal_quality"],
            risk_profile=(get_patient(patient_id) or {}).get("risk_profile", "Standard"),
        )
        heart_rate = estimate_heart_rate(result["signal"], fs)

        created_at = now_iso()
        conn = get_db()
        cur = conn.execute(
            """
            INSERT INTO events(
                patient_id, filename, prediction, confidence, signal_quality,
                alert_level, alert_reason, heart_rate, subtype, subtype_confidence, sampling_rate, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                patient_id,
                safe_original,
                result["prediction"],
                result["confidence"],
                result["signal_quality"],
                alert,
                reason,
                heart_rate,
                result.get("subtype"),
                result.get("subtype_confidence"),
                fs,
                created_at,
            ),
        )
        event_id = cur.lastrowid
        conn.commit()
        conn.close()

        return jsonify({
            "ok": True,
            "event_id": event_id,
            "patient_id": patient_id,
            "prediction": result["prediction"],
            "primary_prediction": result.get("primary_prediction"),
            "subtype": result.get("subtype"),
            "subtype_confidence": round(result["subtype_confidence"], 4) if result.get("subtype_confidence") is not None else None,
            "confidence": round(result["confidence"], 4),
            "signal_quality": round(result["signal_quality"], 4),
            "alert_level": alert,
            "alert_reason": reason,
            "heart_rate": heart_rate,
            "sampling_rate": fs,
            "model_type": result["model_type"],
            "ecg": result["display_signal"],
            "saliency": result["saliency"],
            "salient_seconds": result["salient_seconds"],
            "waveform_offset_seconds": result["window_start"] / fs,
        })
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("ECG analysis failed")
        return jsonify({"error": f"ECG analysis failed: {exc}"}), 500
    finally:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


@app.route("/api/events/<path:patient_id>")
def events(patient_id):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM events WHERE patient_id=? ORDER BY id DESC LIMIT 50",
        (patient_id,),
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/events/<int:event_id>/review", methods=["POST"])
def review_event(event_id):
    reviewed_at = now_iso()
    conn = get_db()
    cur = conn.execute("UPDATE events SET reviewed_at=? WHERE id=?", (reviewed_at, event_id))
    conn.commit()
    conn.close()
    if cur.rowcount == 0:
        return jsonify({"error": "Event not found."}), 404
    return jsonify({"ok": True, "reviewed_at": reviewed_at})


@app.route("/api/stats/<path:patient_id>")
def stats(patient_id):
    conn = get_db()
    rows = conn.execute(
        "SELECT prediction, alert_level, confidence, signal_quality, heart_rate FROM events WHERE patient_id=? ORDER BY id DESC LIMIT 50",
        (patient_id,),
    ).fetchall()
    conn.close()
    data = [dict(r) for r in rows]
    return jsonify({
        "events": len(data),
        "abnormal": sum(r["prediction"] != "Normal" for r in data),
        "high_alerts": sum(r["alert_level"] == "HIGH" for r in data),
        "avg_confidence": round(float(np.mean([r["confidence"] for r in data])) if data else 0, 3),
    })


@app.route("/api/fhir/<int:event_id>")
def fhir(event_id):
    conn = get_db()
    row = conn.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
    conn.close()
    if row is None:
        return jsonify({"error": "Event not found."}), 404

    resource = {
        "resourceType": "Observation",
        "id": f"cardiowatch-{event_id}",
        "status": "final",
        "code": {"text": "CardioWatch AI ECG analysis (educational prototype)"},
        "subject": {"reference": f"Patient/{row['patient_id']}"},
        "effectiveDateTime": row["created_at"],
        "component": [
            {"code": {"text": "Prototype rhythm classification"}, "valueString": row["prediction"]},
            {"code": {"text": "Model confidence"}, "valueQuantity": {"value": round(row["confidence"] * 100, 1), "unit": "%"}},
            {"code": {"text": "Signal quality"}, "valueQuantity": {"value": round(row["signal_quality"] * 100, 1), "unit": "%"}},
            {"code": {"text": "Prototype alert level"}, "valueString": row["alert_level"]},
        ],
        "note": [{"text": "Educational software output; not a medical diagnosis or emergency alert service."}],
    }
    return jsonify(resource)
@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}

    message = str(data.get("message") or "").strip()
    patient_id = str(data.get("patient_id") or "Demo-001").strip()

    if not message:
        return jsonify({"error": "Enter a question."}), 400

    conn = get_db()

    rows = conn.execute(
        """
        SELECT *
        FROM events
        WHERE patient_id = ?
        ORDER BY id DESC
        LIMIT 5
        """,
        (patient_id,),
    ).fetchall()

    conn.close()

    context = "\n".join(
        (
            f"{r['created_at']}: "
            f"prediction={r['prediction']}, "
            f"confidence={r['confidence']:.2f}, "
            f"signal_quality={r['signal_quality']:.2f}, "
            f"alert={r['alert_level']}, "
            f"heart_rate={r['heart_rate']}"
        )
        for r in rows
    )

    if not context:
        context = "No recorded ECG events."

    # ---------------------------------------------------------
    # GEMINI API
    # ---------------------------------------------------------
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    gemini_model = os.getenv(
        "GEMINI_MODEL",
        "gemini-2.5-flash-lite"
    ).strip()

    if gemini_key:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=gemini_key)

            system_instruction = (
                "You are the CardioWatch AI software assistant for an "
                "educational ECG project. "
                "Use only the supplied event data when discussing patient events. "
                "Do not diagnose a medical condition, prescribe treatment, "
                "claim clinical validation, or invent measurements. "
                "For urgent medical concerns, advise the user to contact "
                "a qualified healthcare professional or local emergency service."
            )

            prompt = f"""
Patient ID: {patient_id}

Recent ECG events:
{context}

User question:
{message}
"""

            response = client.models.generate_content(
                model=gemini_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2,
                ),
            )

            return jsonify(
                {
                    "reply": response.text,
                    "mode": "Gemini",
                }
            )

        except Exception as exc:
            app.logger.warning(
                "Gemini chatbot request failed: %s",
                exc,
            )

    # ---------------------------------------------------------
    # LOCAL FALLBACK
    # ---------------------------------------------------------
    latest = rows[0] if rows else None

    if latest:
        reply = (
            f"The latest recorded event for {patient_id} was "
            f"{latest['prediction']} at {latest['created_at']}. "
            f"The model confidence was "
            f"{latest['confidence'] * 100:.1f}%, "
            f"signal quality was "
            f"{latest['signal_quality'] * 100:.1f}%, "
            f"and the prototype alert level was "
            f"{latest['alert_level']}. "
            "This is an educational software result and "
            "is not a medical diagnosis."
        )
    else:
        reply = (
            f"There are no ECG events recorded for {patient_id} yet. "
            "Upload an ECG CSV and run an analysis first."
        )

    return jsonify(
        {
            "reply": reply,
            "mode": "Local",
        }
    )


@app.route("/api/demo-download")
def demo_download():
    path = BASE_DIR / "test_ecg.csv"
    if not path.exists():
        return jsonify({"error": "Run python create_demo_ecg.py first."}), 404
    return send_file(path, as_attachment=True, download_name="test_ecg.csv")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=True)
