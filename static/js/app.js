let currentPatient = 'Demo-001';
let currentEventId = null;
let currentECG = [];
let currentFs = 360;
let liveTimer = null;
let liveIndex = 0;

const $ = (id) => document.getElementById(id);
const pct = (value) => `${(Number(value) * 100).toFixed(1)}%`;
const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#039;' }[c]));

function drawECG(signal, fs, saliency = [], title = 'ECG Signal') {
  if (!signal?.length) {
    $('ecgChart').innerHTML = '<div class="chart-empty">Upload an ECG to display the waveform.</div>';
    return;
  }
  const time = signal.map((_, i) => i / Number(fs));
  const trace = {
    x: time,
    y: signal,
    mode: 'lines',
    line: { width: 1.5, color: '#25d4e6' },
    name: 'ECG'
  };
  const layout = {
    title,
    margin: { l: 55, r: 20, t: 42, b: 50 },
    paper_bgcolor: '#071828',
    plot_bgcolor: '#071828',
    xaxis: { title: 'Time (seconds)', gridcolor: 'rgba(120,170,195,.14)', zeroline: false, color: '#88a9bd' },
    yaxis: { title: 'Amplitude', gridcolor: 'rgba(120,170,195,.14)', zeroline: true, color: '#88a9bd' },
    showlegend: false,
    font: { family: 'Inter, Arial, sans-serif', color: '#cfe7ef' },
    shapes: []
  };
  const stride = saliency.length ? signal.length / saliency.length : 0;
  if (stride) {
    const sorted = saliency.map((v, i) => ({v: Number(v), i})).sort((a,b) => b.v - a.v).slice(0, 5);
    sorted.forEach((item) => {
      const x = item.i * stride / fs;
      layout.shapes.push({ type: 'line', x0: x, x1: x, y0: 0, y1: 1, yref: 'paper', line: { width: 1.2, dash: 'dot' } });
    });
  }
  Plotly.react('ecgChart', [trace], layout, { responsive: true, displaylogo: false });
}

async function checkHealth() {
  try {
    const response = await fetch('/api/health');
    const data = await response.json();
    $('systemStatus').textContent = data.message;
    $('statusDot').classList.toggle('offline', !data.model_ready);
  } catch {
    $('systemStatus').textContent = 'Server Offline';
    $('statusDot').classList.add('offline');
  }
}

async function loadEvents() {
  currentPatient = $('patientId').value.trim() || currentPatient;
  const response = await fetch(`/api/events/${encodeURIComponent(currentPatient)}`);
  const rows = await response.json();
  if (!rows.length) {
    $('eventsBody').innerHTML = '<tr><td colspan="7" class="empty-row">No events yet.</td></tr>';
    $('subtypeResult').textContent = '—';
    return;
  }
  $('eventsBody').innerHTML = rows.map((row) => `
    <tr>
      <td>${esc(row.created_at)}</td>
      <td>${esc(row.prediction)}</td>
      <td>${pct(row.confidence)}</td>
      <td>${pct(row.signal_quality)}</td>
      <td>${row.heart_rate ?? '—'}</td>
      <td>${esc(row.alert_level)}</td>
      <td class="${row.reviewed_at ? 'reviewed' : ''}">${row.reviewed_at ? 'Reviewed' : 'Not reviewed'}</td>
    </tr>`).join('');

  const latest = rows[0];
  $('summaryRhythm').textContent = latest.prediction;
  $('summaryAlert').textContent = latest.alert_level;
  $('heartRate').textContent = latest.heart_rate ?? '—';
  $('prediction').textContent = latest.prediction;
  $('subtypeResult').textContent = latest.subtype ? `Subtype confidence: ${pct(latest.subtype_confidence)}` : 'Primary screening result';
  $('confidence').textContent = pct(latest.confidence);
  $('quality').textContent = pct(latest.signal_quality);
}

async function savePatient() {
  const payload = {
    patient_id: $('patientId').value.trim() || 'Demo-001',
    age: $('age').value,
    sex: $('sex').value,
    risk_profile: $('riskProfile').value,
    notes: $('notes').value
  };
  const response = await fetch('/api/patient', {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)
  });
  const data = await response.json();
  if (!response.ok) { alert(data.error || 'Could not save patient.'); return; }
  currentPatient = payload.patient_id;
  $('summaryPatient').textContent = currentPatient;
  $('message').textContent = 'Patient profile saved.';
  await loadEvents();
}

function showSaliency(seconds) {
  if (!seconds?.length) {
    $('saliencyList').innerHTML = '<div class="muted">No highlighted regions.</div>';
    return;
  }
  $('saliencyList').innerHTML = seconds.map((s) => `<span class="saliency-tag">Highlighted region around <b>${Number(s).toFixed(2)} s</b></span>`).join('');
}

async function analyze(event) {
  event.preventDefault();
  const file = $('ecgFile').files[0];
  if (!file) return;
  currentPatient = $('patientId').value.trim() || 'Demo-001';
  $('analyzeButton').disabled = true;
  $('message').textContent = 'Running ECG preprocessing and ML analysis…';
  const formData = new FormData();
  formData.append('patient_id', currentPatient);
  formData.append('sampling_rate', $('samplingRate').value || '360');
  formData.append('file', file);

  try {
    const response = await fetch('/api/analyze', { method: 'POST', body: formData });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Analysis failed.');

    currentEventId = data.event_id;
    currentECG = data.ecg;
    currentFs = data.sampling_rate;
    $('summaryPatient').textContent = currentPatient;
    $('heartRate').textContent = data.heart_rate ?? '—';
    $('summaryRhythm').textContent = data.prediction;
    $('summaryAlert').textContent = data.alert_level;
    $('prediction').textContent = data.prediction;
    $('subtypeResult').textContent = data.subtype ? `Subtype confidence: ${pct(data.subtype_confidence)}` : 'Primary screening result';
    $('confidence').textContent = pct(data.confidence);
    $('quality').textContent = pct(data.signal_quality);
    $('alert').textContent = data.alert_level;
    $('waveformInfo').textContent = `${currentECG.length} displayed samples · ${data.sampling_rate} Hz · ${data.model_type}`;
    $('eventId').textContent = `Event #${data.event_id}`;
    $('explainText').textContent = `The ${data.model_type} predicted ${data.prediction} with ${pct(data.confidence)} confidence. The saliency view marks waveform locations that most influenced the selected class score.`;
    showSaliency(data.salient_seconds);
    $('fhirBtn').disabled = false;
    $('reviewStatus').textContent = 'Not reviewed';
    drawECG(currentECG, currentFs, data.saliency, 'Analyzed ECG');
    $('message').textContent = `Analysis completed successfully · ${data.alert_level} prototype alert.`;
    await loadEvents();
  } catch (error) {
    $('message').textContent = error.message;
  } finally {
    $('analyzeButton').disabled = false;
  }
}

async function exportFHIR() {
  if (!currentEventId) return;
  const response = await fetch(`/api/fhir/${currentEventId}`);
  const data = await response.json();
  if (!response.ok) { alert(data.error || 'FHIR export failed.'); return; }
  const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/fhir+json'});
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `cardiowatch-event-${currentEventId}.json`;
  link.click();
  URL.revokeObjectURL(url);
}

async function reviewEvent() {
  if (!currentEventId) return;
  const response = await fetch(`/api/events/${currentEventId}/review`, {method: 'POST'});
  const data = await response.json();
  if (!response.ok) { alert(data.error || 'Could not review event.'); return; }
  $('reviewStatus').textContent = `Reviewed locally at ${new Date(data.reviewed_at).toLocaleTimeString()}`;
  loadEvents();
}

function startLive() {
  if (!currentECG.length) { alert('Analyze an ECG first, then start the live demo.'); return; }
  stopLive();
  liveIndex = 0;
  $('liveStatus').textContent = 'Live demo running';
  liveTimer = setInterval(() => {
    const chunk = Math.max(90, Math.floor(currentFs * 0.5));
    const start = liveIndex;
    const end = Math.min(currentECG.length, start + chunk);
    drawECG(currentECG.slice(start, end), currentFs, [], 'Live ECG Demo');
    $('liveProgress').style.width = `${(end / currentECG.length) * 100}%`;
    $('liveClock').textContent = `${(end / currentFs).toFixed(1)} s`;
    liveIndex = end;
    if (end >= currentECG.length) stopLive();
  }, 350);
}

function stopLive() {
  if (liveTimer) clearInterval(liveTimer);
  liveTimer = null;
  $('liveStatus').textContent = 'Live demo stopped';
}

$('uploadForm').addEventListener('submit', analyze);
$('savePatientBtn').addEventListener('click', savePatient);
$('refreshEvents').addEventListener('click', loadEvents);
$('fhirBtn').addEventListener('click', exportFHIR);
$('reviewBtn').addEventListener('click', reviewEvent);
$('liveDemoBtn').addEventListener('click', () => liveTimer ? stopLive() : startLive());
$('patientId').addEventListener('change', loadEvents);

$('chatForm').addEventListener('submit', async (event) => {
  event.preventDefault();
  const input = $('chatInput');
  const message = input.value.trim();
  if (!message) return;
  const userBubble = document.createElement('div');
  userBubble.className = 'chat-bubble user';
  userBubble.textContent = message;
  $('chatLog').appendChild(userBubble);
  input.value = '';
  input.disabled = true;
  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message, patient_id: $('patientId').value.trim() || 'Demo-001'})
    });
    const data = await response.json();
    const botBubble = document.createElement('div');
    botBubble.className = 'chat-bubble bot';
    botBubble.textContent = data.reply || data.error || 'No response.';
    $('chatLog').appendChild(botBubble);
    $('chatMode').textContent = data.mode === 'OpenAI' ? 'OpenAI mode' : 'Local mode';
  } catch {
    const botBubble = document.createElement('div');
    botBubble.className = 'chat-bubble bot';
    botBubble.textContent = 'The assistant could not connect to the server.';
    $('chatLog').appendChild(botBubble);
  } finally {
    input.disabled = false;
    $('chatLog').scrollTop = $('chatLog').scrollHeight;
  }
});

checkHealth();
loadEvents();
