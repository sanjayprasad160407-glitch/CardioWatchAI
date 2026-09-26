NORMAL = {"Normal"}


def compute_alert(prediction, confidence, signal_quality, risk_profile="Standard"):
    """Educational alert logic only; not a clinically validated risk score."""
    confidence = float(confidence)
    signal_quality = float(signal_quality)
    risk = str(risk_profile or "Standard")

    if signal_quality < 0.50:
        return "DATA_REVIEW", "Signal quality is low; review the recording before interpreting the model output."
    if confidence < 0.60:
        return "DATA_REVIEW", "Model confidence is low; review the recording before interpreting the result."
    if prediction in NORMAL:
        return "LOW", "The prototype classified the selected window as Normal."
    if prediction == "Fusion":
        return "HIGH", "The prototype detected a non-Normal rhythm class with sufficient confidence."
    if prediction in {"PVC", "Supraventricular"}:
        if confidence >= 0.85 or risk.lower() == "high attention":
            return "HIGH", "The prototype classified a non-Normal rhythm class with high confidence."
        return "MEDIUM", "The prototype classified a non-Normal rhythm class; review the event."
    return "MEDIUM", "The prototype produced a non-Normal classification; review the event."
