const state = {
  file: null,
  steps: [
    { id: "upload", label: "Video Upload", status: "Pending", detail: "Awaiting MP4 evidence." },
    { id: "preprocess", label: "Frame Sampling & Face Cropping", status: "Pending", detail: "Awaiting video preprocessing." },
    { id: "classify", label: "M2TR Artifact Classification", status: "Pending", detail: "Awaiting normalized facial regions." },
    { id: "report", label: "XAI Report Generation", status: "Pending", detail: "Awaiting diagnostic path." },
    { id: "eqaf", label: "EQAF Quality Assessment", status: "Pending", detail: "Awaiting generated explanation." }
  ]
};

const form = document.getElementById("uploadForm");
const input = document.getElementById("videoInput");
const fileName = document.getElementById("fileName");
const analyzeButton = document.getElementById("analyzeButton");
const dropZone = document.getElementById("dropZone");
const errorPanel = document.getElementById("errorPanel");

initialize();

function initialize() {
  renderStepper(state.steps);
  wireUpload();
  loadHealthStatus();
  lucide.createIcons();
}

function wireUpload() {
  input.addEventListener("change", () => setFile(input.files[0]));

  ["dragenter", "dragover"].forEach(eventName => {
    dropZone.addEventListener(eventName, event => {
      event.preventDefault();
      dropZone.classList.add("border-indigo-500", "bg-indigo-50");
    });
  });

  ["dragleave", "drop"].forEach(eventName => {
    dropZone.addEventListener(eventName, event => {
      event.preventDefault();
      dropZone.classList.remove("border-indigo-500", "bg-indigo-50");
    });
  });

  dropZone.addEventListener("drop", event => {
    setFile(event.dataTransfer.files[0]);
  });

  form.addEventListener("submit", async event => {
    event.preventDefault();
    if (!state.file) return;
    await analyzeVideo();
  });
}

function setFile(file) {
  if (!file) return;
  state.file = file;
  fileName.textContent = `${file.name} - ${formatBytes(file.size)}`;
  analyzeButton.disabled = false;
}

async function loadHealthStatus() {
  const healthStatus = document.getElementById("healthStatus");
  try {
    const response = await fetch("/api/health");
    const health = await response.json();
    healthStatus.textContent = health.model;
  } catch {
    healthStatus.textContent = "Backend health status is unavailable.";
  }
}

async function analyzeVideo() {
  clearError();
  setLoading(true);
  hideOutputs();

  const liveSteps = state.steps.map(step => ({ ...step, status: "Pending" }));
  liveSteps[0].status = "Processing";
  liveSteps[0].detail = `Uploading ${state.file.name}.`;
  renderStepper(liveSteps);

  const data = new FormData();
  data.append("video", state.file);

  try {
    const response = await fetch("/api/analyze", { method: "POST", body: data });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Analysis failed.");
    }

    await revealSequentially(payload);
  } catch (error) {
    showError(error.message);
    renderStepper(state.steps);
  } finally {
    setLoading(false);
  }
}

async function revealSequentially(payload) {
  const steps = payload.steps.map(step => ({ ...step, status: "Pending" }));
  renderStepper(steps);

  await completeStep(steps, payload, "upload");
  await completeStep(steps, payload, "preprocess", () => renderFaces(payload));
  await completeStep(steps, payload, "classify", () => renderClassification(payload));
  await completeStep(steps, payload, "report", () => renderReport(payload));
  await completeStep(steps, payload, "eqaf", () => renderEqaf(payload));
  renderSummary(payload);
}

async function completeStep(steps, payload, id, renderOutput) {
  const index = steps.findIndex(step => step.id === id);
  steps[index] = { ...payload.steps[index], status: "Processing" };
  renderStepper(steps);
  await delay(360);
  steps[index] = payload.steps[index];
  renderStepper(steps);
  if (renderOutput) {
    renderOutput();
  }
  await delay(220);
}

function renderStepper(steps) {
  const stepper = document.getElementById("stepper");
  stepper.innerHTML = steps.map((step, index) => `
    <div class="step-card" data-status="${step.status}">
      <div class="mb-4 flex items-center justify-between">
        <span class="text-xs font-bold uppercase tracking-wide text-slate-500">0${index + 1}</span>
        <span class="status-dot"></span>
      </div>
      <h3 class="min-h-10 text-sm font-semibold text-slate-950">${step.label}</h3>
      <p class="mt-3 text-xs font-semibold text-slate-600">${step.status}</p>
      <p class="mt-2 line-clamp-3 text-xs leading-5 text-slate-500">${step.detail}</p>
    </div>
  `).join("");
}

function renderSummary(payload) {
  const summaryBand = document.getElementById("summaryBand");
  const badge = document.getElementById("verdictBadge");
  const isFake = payload.verdict.isFake;

  badge.className = `mt-2 inline-flex items-center gap-2 rounded-lg px-4 py-2 text-lg font-bold ${isFake ? "bg-red-100 text-red-700" : "bg-emerald-100 text-emerald-700"}`;
  badge.innerHTML = `<i data-lucide="${isFake ? "shield-alert" : "shield-check"}" class="h-5 w-5"></i>${payload.verdict.status}`;

  document.getElementById("dominantSignal").textContent = payload.verdict.dominantSignal;
  document.getElementById("confidenceValue").textContent = formatPercent(payload.verdict.confidence);
  document.getElementById("eqafOverall").textContent = `${payload.eqaf.overall.toFixed(1)} / 10`;
  summaryBand.classList.remove("hidden");
  summaryBand.classList.add("fade-in");
  lucide.createIcons();
}

function renderFaces(payload) {
  const section = document.getElementById("facesSection");
  const grid = document.getElementById("facesGrid");
  document.getElementById("faceCount").textContent = `${payload.faces.length} crops saved`;

  grid.innerHTML = payload.faces.map(face => `
    <article class="overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm">
      <img src="${face.imageUrl}" alt="Detected face ${face.id}" class="aspect-square w-full object-cover">
      <div class="p-4">
        <div class="mb-3 flex items-center justify-between">
          <h3 class="font-semibold text-slate-950">${face.id}</h3>
          <span class="rounded-lg bg-slate-100 px-2 py-1 text-xs font-semibold text-slate-600">Frame ${face.frameIndex}</span>
        </div>
        <dl class="grid grid-cols-3 gap-2 text-xs">
          <div><dt class="text-slate-500">Sharp</dt><dd class="font-semibold">${face.sharpness.toFixed(2)}</dd></div>
          <div><dt class="text-slate-500">Light</dt><dd class="font-semibold">${face.illumination.toFixed(2)}</dd></div>
          <div><dt class="text-slate-500">Texture</dt><dd class="font-semibold">${face.textureEnergy.toFixed(2)}</dd></div>
        </dl>
      </div>
    </article>
  `).join("");

  section.classList.remove("hidden");
  section.classList.add("fade-in");
}

function renderClassification(payload) {
  const section = document.getElementById("classificationSection");
  const scoreBars = document.getElementById("scoreBars");
  const metrics = document.getElementById("modelMetrics");

  scoreBars.innerHTML = `
    <div class="mb-5 flex items-center justify-between">
      <div class="flex items-center gap-2">
        <i data-lucide="activity" class="h-5 w-5 text-indigo-500"></i>
        <h3 class="font-semibold text-slate-950">Artifact Probability Scores</h3>
      </div>
      <span class="rounded-lg bg-slate-100 px-3 py-1 text-sm font-semibold text-slate-600">${payload.classification.dominantClass}</span>
    </div>
    <div class="space-y-4">
      ${payload.classification.scores.map(score => {
        const isArtifact = score.className !== "Normal/Real";
        return `
          <div>
            <div class="mb-2 flex items-center justify-between gap-3 text-sm">
              <span class="font-medium text-slate-800">${score.className}</span>
              <span class="font-semibold text-slate-950">${formatPercent(score.probability)}</span>
            </div>
            <div class="probability-track">
              <div class="probability-fill ${isArtifact ? "fake" : ""}" style="width:${score.probability * 100}%"></div>
            </div>
          </div>
        `;
      }).join("")}
    </div>
  `;

  metrics.innerHTML = `
    ${metricRow("Model", payload.classification.modelName)}
    ${metricRow("Runtime", payload.classification.modelStatus)}
    ${metricRow("Accuracy", formatPercent(payload.classification.referenceAccuracy))}
    ${metricRow("Macro-F1", payload.classification.referenceMacroF1.toFixed(3))}
    ${metricRow("AUC", payload.classification.referenceAuc.toFixed(3))}
    ${metricRow("Frame Votes", String(payload.classification.stability.totalFrames))}
    ${metricRow("Suspicious Rate", formatPercent(payload.classification.stability.suspiciousFrameRate))}
    ${metricRow("Peak Anomaly", formatPercent(payload.classification.stability.peakArtifactConfidence))}
  `;

  section.classList.remove("hidden");
  section.classList.add("fade-in");
  lucide.createIcons();
}

function renderReport(payload) {
  const section = document.getElementById("reportSection");
  const report = payload.report;
  const isFake = payload.verdict.isFake;

  document.getElementById("reportStatus").className = `rounded-lg px-3 py-2 text-sm font-bold ${isFake ? "bg-red-100 text-red-700" : "bg-emerald-100 text-emerald-700"}`;
  document.getElementById("reportStatus").textContent = `STATUS: ${report.status}`;
  document.getElementById("reportTemplate").textContent = report.template;
  document.getElementById("verdictRule").textContent = payload.verdict.rule;
  document.getElementById("verificationMetrics").innerHTML = renderVerificationMetrics(payload);

  document.getElementById("reportBody").innerHTML = `
    ${reportBlock("Diagnostic Reasons", report.diagnosticReasons, "search-check")}
    ${report.potentialGenerativeMechanisms.length ? reportBlock("Potential Generative Mechanisms", report.potentialGenerativeMechanisms, "cpu") : ""}
    ${reportBlock("Forensic Verification Guidance", report.forensicVerificationGuidance, "clipboard-check")}
  `;

  section.classList.remove("hidden");
  section.classList.add("fade-in");
  lucide.createIcons();
}

function renderVerificationMetrics(payload) {
  const stability = payload.classification.stability;
  const hybridStatus = stability.majorityVotingPassed
    ? "35% Hybrid Vote Triggered"
    : "Below 35% Hybrid Bound";
  const frameThresholdStatus = stability.confidenceThresholdPassed ? "50% Frame Threshold Hit" : "No 50% Frame Hit";
  const peakStatus = stability.peakAnomalyTriggered ? "80% Peak Anomaly Triggered" : "No 80% Peak Anomaly";

  return `
    ${insightMetric("Suspicious Frame Rate", formatPercent(stability.suspiciousFrameRate), "activity")}
    ${insightMetric("Hybrid Voting Resolution", hybridStatus, stability.majorityVotingPassed ? "git-compare-arrows" : "shield-check")}
    ${insightMetric("Frame Confidence Bound", frameThresholdStatus, stability.confidenceThresholdPassed ? "lock-keyhole" : "lock")}
    ${insightMetric("Peak Artifact Detection", peakStatus, stability.peakAnomalyTriggered ? "siren" : "scan")}
    ${insightMetric("Peak Artifact", `${stability.peakArtifactClass} - ${formatPercent(stability.peakArtifactConfidence)}`, "scan-face")}
  `;
}

function insightMetric(label, value, icon) {
  return `
    <div class="rounded-lg border border-slate-700 bg-slate-900 px-3 py-3">
      <div class="mb-1 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
        <i data-lucide="${icon}" class="h-3.5 w-3.5 text-indigo-300"></i>
        <span>${label}</span>
      </div>
      <p class="text-sm font-semibold leading-5 text-white">${value}</p>
    </div>
  `;
}

function renderEqaf(payload) {
  const section = document.getElementById("eqafSection");
  const bars = document.getElementById("eqafBars");

  bars.innerHTML = payload.eqaf.dimensions.map(dimension => `
    <article class="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div class="mb-2 flex items-center justify-between gap-3">
        <h3 class="font-semibold text-slate-950">${dimension.name}</h3>
        <span class="rounded-lg bg-slate-100 px-2 py-1 text-sm font-bold text-slate-700">${dimension.score.toFixed(1)}</span>
      </div>
      <div class="probability-track">
        <div class="probability-fill" style="width:${dimension.score * 10}%"></div>
      </div>
      <p class="mt-2 text-sm leading-5 text-slate-500">${dimension.rationale}</p>
    </article>
  `).join("");

  drawRadar(payload.eqaf.dimensions);
  section.classList.remove("hidden");
  section.classList.add("fade-in");
}

function drawRadar(dimensions) {
  const canvas = document.getElementById("eqafRadar");
  const context = canvas.getContext("2d");
  const width = canvas.width;
  const center = width / 2;
  const radius = 112;

  context.clearRect(0, 0, width, width);
  context.font = "12px Segoe UI, sans-serif";
  context.textAlign = "center";
  context.textBaseline = "middle";

  for (let ring = 1; ring <= 5; ring++) {
    drawPolygon(context, dimensions.length, center, radius * ring / 5, "rgba(148, 163, 184, 0.35)", false);
  }

  dimensions.forEach((dimension, index) => {
    const angle = -Math.PI / 2 + index * 2 * Math.PI / dimensions.length;
    const x = center + Math.cos(angle) * radius;
    const y = center + Math.sin(angle) * radius;
    context.beginPath();
    context.moveTo(center, center);
    context.lineTo(x, y);
    context.strokeStyle = "rgba(148, 163, 184, 0.45)";
    context.stroke();

    const labelX = center + Math.cos(angle) * (radius + 34);
    const labelY = center + Math.sin(angle) * (radius + 26);
    context.fillStyle = "rgb(71, 85, 105)";
    context.fillText(dimension.name.split(" ")[0], labelX, labelY);
  });

  const points = dimensions.map((dimension, index) => {
    const angle = -Math.PI / 2 + index * 2 * Math.PI / dimensions.length;
    const distance = radius * dimension.score / 10;
    return [center + Math.cos(angle) * distance, center + Math.sin(angle) * distance];
  });

  context.beginPath();
  points.forEach(([x, y], index) => index === 0 ? context.moveTo(x, y) : context.lineTo(x, y));
  context.closePath();
  context.fillStyle = "rgba(79, 70, 229, 0.18)";
  context.fill();
  context.strokeStyle = "rgb(79, 70, 229)";
  context.lineWidth = 2;
  context.stroke();

  points.forEach(([x, y]) => {
    context.beginPath();
    context.arc(x, y, 4, 0, Math.PI * 2);
    context.fillStyle = "rgb(20, 184, 166)";
    context.fill();
  });
}

function drawPolygon(context, sides, center, radius, strokeStyle, fill) {
  context.beginPath();
  for (let i = 0; i < sides; i++) {
    const angle = -Math.PI / 2 + i * 2 * Math.PI / sides;
    const x = center + Math.cos(angle) * radius;
    const y = center + Math.sin(angle) * radius;
    i === 0 ? context.moveTo(x, y) : context.lineTo(x, y);
  }
  context.closePath();
  if (fill) context.fill();
  context.strokeStyle = strokeStyle;
  context.lineWidth = 1;
  context.stroke();
}

function reportBlock(title, items, icon) {
  return `
    <section>
      <div class="mb-3 flex items-center gap-2">
        <i data-lucide="${icon}" class="h-5 w-5 text-indigo-500"></i>
        <h3 class="font-semibold text-slate-950">${title}</h3>
      </div>
      <ul class="space-y-2">
        ${items.map(item => `<li class="rounded-lg border border-slate-200 bg-slate-50 px-3 py-3 text-sm leading-6 text-slate-700">${item}</li>`).join("")}
      </ul>
    </section>
  `;
}

function metricRow(label, value) {
  return `
    <div class="flex items-start justify-between gap-4 border-b border-slate-100 pb-3 last:border-b-0">
      <dt class="text-slate-500">${label}</dt>
      <dd class="max-w-52 text-right font-semibold text-slate-950">${value}</dd>
    </div>
  `;
}

function hideOutputs() {
  ["summaryBand", "facesSection", "classificationSection", "reportSection", "eqafSection"].forEach(id => {
    document.getElementById(id).classList.add("hidden");
  });
}

function setLoading(isLoading) {
  analyzeButton.disabled = isLoading || !state.file;
  analyzeButton.innerHTML = isLoading
    ? `<span class="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent"></span> Processing Evidence`
    : `<i data-lucide="scan-search" class="h-4 w-4"></i> Run Sequential Analysis`;
  lucide.createIcons();
}

function showError(message) {
  errorPanel.textContent = message;
  errorPanel.classList.remove("hidden");
}

function clearError() {
  errorPanel.classList.add("hidden");
  errorPanel.textContent = "";
}

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}
