const form = document.getElementById("convert-form");
const fileInput = document.getElementById("file");
const fileName = document.getElementById("file-name");
const sourceAudio = document.getElementById("source-audio");
const sourceDuration = document.getElementById("source-duration");
const sourceWaveform = document.getElementById("source-waveform");
const outputAudio = document.getElementById("output-audio");
const outputWaveform = document.getElementById("output-waveform");
const trajectoryCanvas = document.getElementById("trajectory");
const azimuthLabel = document.getElementById("azimuth-label");
const progressCard = document.getElementById("progress-card");
const progressTitle = document.getElementById("progress-title");
const progressPercent = document.getElementById("progress-percent");
const progressBar = document.getElementById("progress-bar");
const progressStage = document.getElementById("progress-stage");
const queueDepth = document.getElementById("queue-depth");
const resultBox = document.getElementById("result");
const resultFormat = document.getElementById("result-format");
const metricsBox = document.getElementById("metrics");
const download = document.getElementById("download");
const batchList = document.getElementById("batch-list");
const preset = document.getElementById("preset");
const resetButton = document.getElementById("reset");

let sourceObjectUrl = null;
let trajectoryFrame = null;
let startedAt = 0;

function field(name) {
  return form.elements[name];
}

function showProgress(title = "Processing") {
  progressCard.hidden = false;
  progressTitle.textContent = title;
}

function updateProgress(progress, stage, depth = 0) {
  const value = Math.max(0, Math.min(100, Number(progress) || 0));
  progressPercent.textContent = `${Math.round(value)}%`;
  progressBar.style.width = `${value}%`;
  progressStage.textContent = stage || "Processing";
  queueDepth.textContent = `Queue depth: ${depth}`;
}

function drawEnvelope(canvas, values) {
  const ctx = canvas.getContext("2d");
  const width = canvas.clientWidth || canvas.width;
  const height = canvas.height;
  canvas.width = Math.max(320, Math.floor(width * window.devicePixelRatio));
  canvas.height = Math.max(120, Math.floor(180 * window.devicePixelRatio));
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!values.length) return;

  const scaleX = canvas.width / values.length;
  const mid = canvas.height / 2;
  ctx.beginPath();
  values.forEach((value, index) => {
    const x = index * scaleX;
    const amplitude = Math.min(1, Math.abs(value)) * (canvas.height * 0.43);
    ctx.lineTo(x, mid - amplitude);
  });
  for (let index = values.length - 1; index >= 0; index -= 1) {
    const x = index * scaleX;
    const amplitude = Math.min(1, Math.abs(values[index])) * (canvas.height * 0.43);
    ctx.lineTo(x, mid + amplitude);
  }
  ctx.closePath();
  ctx.fillStyle = "rgba(96, 165, 250, 0.22)";
  ctx.fill();

  ctx.beginPath();
  values.forEach((value, index) => {
    const x = index * scaleX;
    const amplitude = Math.min(1, Math.abs(value)) * (canvas.height * 0.43);
    const y = index % 2 === 0 ? mid - amplitude : mid + amplitude;
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = "#60a5fa";
  ctx.lineWidth = 1.5 * window.devicePixelRatio;
  ctx.stroke();
}

function drawTrajectory(timestamp) {
  const ctx = trajectoryCanvas.getContext("2d");
  const ratio = window.devicePixelRatio || 1;
  const logicalWidth = 500;
  const logicalHeight = 320;
  trajectoryCanvas.width = logicalWidth * ratio;
  trajectoryCanvas.height = logicalHeight * ratio;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);

  const elapsed = (timestamp - startedAt) / 1000;
  const speed = Number(field("pan_speed_hz").value || 0.5);
  const depth = Number(field("depth").value || 0);
  const azimuth = 90 * depth * Math.sin(2 * Math.PI * speed * elapsed);

  ctx.clearRect(0, 0, logicalWidth, logicalHeight);
  const cx = logicalWidth / 2;
  const cy = logicalHeight / 2 + 10;
  const radiusX = 175;
  const radiusY = 105;

  ctx.strokeStyle = "rgba(148,163,184,0.30)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.ellipse(cx, cy, radiusX, radiusY, 0, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(cx - radiusX, cy);
  ctx.lineTo(cx + radiusX, cy);
  ctx.moveTo(cx, cy - radiusY);
  ctx.lineTo(cx, cy + radiusY);
  ctx.stroke();

  const normalized = azimuth / 90;
  const angle = -normalized * (Math.PI / 2);
  const x = cx + Math.sin(angle) * radiusX;
  const y = cy - Math.cos(angle) * radiusY * 0.35;

  ctx.fillStyle = "rgba(96,165,250,0.12)";
  ctx.beginPath();
  ctx.arc(x, y, 26, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#60a5fa";
  ctx.beginPath();
  ctx.arc(x, y, 8, 0, Math.PI * 2);
  ctx.fill();

  azimuthLabel.textContent = `${azimuth >= 0 ? "+" : ""}${azimuth.toFixed(1)}°`;
  trajectoryFrame = requestAnimationFrame(drawTrajectory);
}

async function previewSource() {
  const [file] = fileInput.files;
  fileName.textContent = fileInput.files.length
    ? `${fileInput.files.length} file${fileInput.files.length === 1 ? "" : "s"} selected`
    : "No files selected";
  if (!file) return;

  if (sourceObjectUrl) URL.revokeObjectURL(sourceObjectUrl);
  sourceObjectUrl = URL.createObjectURL(file);
  sourceAudio.src = sourceObjectUrl;
  sourceAudio.hidden = false;

  try {
    const context = new AudioContext();
    const buffer = await context.decodeAudioData(await file.arrayBuffer());
    sourceDuration.textContent = `${buffer.duration.toFixed(2)} s`;
    const channel = buffer.getChannelData(0);
    const points = 512;
    const values = [];
    const step = Math.max(1, Math.floor(channel.length / points));
    for (let index = 0; index < channel.length; index += step) {
      values.push(channel[index]);
    }
    drawEnvelope(sourceWaveform, values.slice(0, points));
    await context.close();
  } catch (error) {
    sourceDuration.textContent = "Preview metadata unavailable";
    drawEnvelope(sourceWaveform, []);
  }
}

async function loadPresets() {
  try {
    const response = await fetch("/api/presets");
    const data = await response.json();
    preset.innerHTML = data.presets
      .map((item) => `<option value="${item.name}">${item.name[0].toUpperCase()}${item.name.slice(1)}</option>`)
      .join("");
    applyPreset(data.presets[0]);
  } catch (error) {
    preset.innerHTML = '<option value="balanced">Balanced</option>';
  }
}

function applyPreset(item) {
  if (!item?.config) return;
  const config = item.config;
  for (const name of [
    "pan_speed_hz", "depth", "reverb_delay_ms", "reverb_decay", "reverb_mix",
    "target_rms_db", "limiter_db", "output_format", "output_bitrate", "room_enabled",
    "hrtf_enabled",
  ]) {
    if (field(name) && config[name] !== undefined) field(name).value = String(config[name]);
  }
}

async function submitSingle(file) {
  const body = new FormData(form);
  body.delete("file");
  body.append("file", file);
  const response = await fetch("/api/jobs", { method: "POST", body });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Failed to queue job.");
  await pollJob(data.status_url);
}

async function submitBatch(files) {
  const body = new FormData(form);
  body.delete("file");
  for (const file of files) body.append("files", file);
  const response = await fetch("/api/batches", { method: "POST", body });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Failed to queue batch.");
  showProgress(`Batch · ${files.length} files`);
  await pollBatch(data.status_url);
}

async function pollJob(url) {
  for (;;) {
    const response = await fetch(url);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Unable to read job status.");
    updateProgress(data.progress, data.stage, data.queue_depth || 0);
    if (data.status === "failed") {
      progressStage.textContent = `Failed: ${data.error || "unknown error"}`;
      return;
    }
    if (data.status === "completed") {
      updateProgress(100, "Complete", 0);
      resultBox.hidden = false;
      resultFormat.textContent = data.output_format.toUpperCase();
      outputAudio.src = data.preview_url;
      download.href = data.download_url;
      metricsBox.textContent = JSON.stringify(data.metrics, null, 2);
      const waveformResponse = await fetch(data.waveform_url);
      const waveformData = await waveformResponse.json();
      drawEnvelope(outputWaveform, waveformData.waveform || []);
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
}

async function pollBatch(url) {
  batchList.hidden = false;
  for (;;) {
    const response = await fetch(url);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Unable to read batch status.");
    updateProgress(data.progress, `${data.completed}/${data.total} complete`, 0);
    batchList.innerHTML = data.jobs.map((job) => `
      <div class="batch-row">
        <span>${escapeHtml(job.filename)}</span>
        <span>${job.status}</span>
        <span>${job.status === "completed" ? `<a href="${job.download_url || `/api/jobs/${job.id}/download`}">Download</a>` : `${job.progress}%`}</span>
      </div>`).join("");
    if (data.status === "completed" || data.status === "failed") {
      progressStage.textContent = data.status === "completed" ? "Batch complete" : "Batch finished with failures";
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 650));
  }
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#039;", '"': "&quot;",
  }[character]));
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!fileInput.files.length) return;
  resultBox.hidden = true;
  batchList.hidden = true;
  showProgress(fileInput.files.length === 1 ? "Conversion queued" : `Batch · ${fileInput.files.length} files`);
  startedAt = performance.now();
  cancelAnimationFrame(trajectoryFrame);
  trajectoryFrame = requestAnimationFrame(drawTrajectory);
  try {
    if (fileInput.files.length === 1) await submitSingle(fileInput.files[0]);
    else await submitBatch([...fileInput.files]);
  } catch (error) {
    updateProgress(0, `Error: ${error.message}`, 0);
  }
});

preset.addEventListener("change", async () => {
  const response = await fetch("/api/presets");
  const data = await response.json();
  const item = data.presets.find((entry) => entry.name === preset.value);
  applyPreset(item);
});

fileInput.addEventListener("change", previewSource);

resetButton.addEventListener("click", () => {
  form.reset();
  fileName.textContent = "No files selected";
  sourceDuration.textContent = "—";
  sourceAudio.hidden = true;
  sourceAudio.removeAttribute("src");
  resultBox.hidden = true;
  progressCard.hidden = true;
  batchList.hidden = true;
  cancelAnimationFrame(trajectoryFrame);
});

window.addEventListener("beforeunload", () => {
  if (sourceObjectUrl) URL.revokeObjectURL(sourceObjectUrl);
});

loadPresets();
startedAt = performance.now();
trajectoryFrame = requestAnimationFrame(drawTrajectory);
