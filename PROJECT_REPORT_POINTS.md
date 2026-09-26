# Project report points

## Title
CardioWatch AI — Real-Time Wearable Cardiac Arrhythmia Detection & Risk Alerting

## Problem
Serious arrhythmias can be intermittent, while noisy wearable signals can produce false alarms. The prototype demonstrates software methods for preprocessing ECG data, classifying selected rhythm classes, and routing results into an alert/dashboard workflow.

## Proposed solution
ECG CSV → preprocessing → signal quality check → ML classification → prototype alert engine → dashboard → explainable saliency → event history → chatbot/FHIR export.

## AI/ML
Real training uses a 1D CNN on beat-centered ECG windows from the MIT-BIH Arrhythmia Database. The script uses record-level train/validation/test groups to reduce leakage across records and reports precision, recall, F1-score, accuracy, and a confusion matrix.

## Classes in the educational model
- Normal
- Supraventricular
- PVC
- Fusion

These class mappings are an educational simplification of annotation symbols. They are not a substitute for a clinically validated labeling protocol.

## Website
- Patient profile
- ECG upload
- Sampling-rate input
- ECG waveform viewer
- Prediction/confidence/signal quality/alert cards
- Event history
- Event review
- Explainable saliency
- FHIR Observation export
- AI Assistant
- Live waveform demonstration

## Limitations
The system is an educational/research prototype. It is not clinically validated, does not diagnose disease, and is not an emergency-response service. Alert thresholds are software demonstration logic rather than clinical thresholds.
