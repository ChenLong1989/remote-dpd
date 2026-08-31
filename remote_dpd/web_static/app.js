"use strict";

const CONTROL_HEADERS = {
  "Content-Type": "application/json",
  "X-Remote-DPD-Request": "1",
};

const TRACE_COLORS = {
  baseline_z: "#ffd400",
  target_z: "#20d8d3",
  target_error: "#b58cff",
  reference_x: "#c1c7ca",
  target_y: "#e279d2",
};

const state = {
  devices: [],
  device: null,
  waveforms: [],
  selectedWaveform: "",
  session: null,
  runs: [],
  runDetails: new Map(),
  activeCommand: null,
  eventSource: null,
  analysis: null,
  analysisPending: false,
  analysisRefreshQueued: false,
  analysisKey: "",
  markerIndex: null,
  spectrumGeometry: null,
  displayInitializedFor: "",
  auxView: "convergence",
  initialized: false,
  selectedRunId: null,
  nextAction: null,
};

const byId = (id) => document.getElementById(id);
let requestSequence = 0;

function createRequestId(prefix = "") {
  requestSequence = (requestSequence + 1) % 0x100000;
  let randomToken;
  if (globalThis.crypto?.getRandomValues) {
    const bytes = new Uint8Array(6);
    globalThis.crypto.getRandomValues(bytes);
    randomToken = Array.from(bytes, (value) => value.toString(16).padStart(2, "0")).join("");
  } else {
    const timer = Math.floor((globalThis.performance?.now?.() || 0) * 1000).toString(36);
    randomToken = `${timer}${requestSequence.toString(36)}`;
  }
  return `${prefix}${Date.now().toString(36)}-${randomToken}`.slice(0, 40);
}

function formatNumber(value, digits = 2) {
  return Number.isFinite(value) ? Number(value).toFixed(digits) : "—";
}

function formatSigned(value, digits = 2, unit = "") {
  if (!Number.isFinite(value)) return `—${unit ? ` ${unit}` : ""}`;
  const sign = value > 0 ? "+" : "";
  return `${sign}${Number(value).toFixed(digits)}${unit ? ` ${unit}` : ""}`;
}

function formatEngineering(value, unit = "", digits = 3) {
  if (!Number.isFinite(value)) return "—";
  const magnitude = Math.abs(value);
  const scales = [
    [1e9, "G"],
    [1e6, "M"],
    [1e3, "k"],
    [1, ""],
    [1e-3, "m"],
    [1e-6, "µ"],
  ];
  const [scale, prefix] = scales.find(([candidate]) => magnitude >= candidate) || [1e-9, "n"];
  const normalized = value / scale;
  const fixedDigits = Math.abs(normalized) >= 100 ? 1 : Math.abs(normalized) >= 10 ? 2 : digits;
  return `${normalized.toFixed(fixedDigits)} ${prefix}${unit}`.trim();
}

function finiteValue(id, label) {
  const value = Number(byId(id).value);
  if (!Number.isFinite(value)) throw new Error(`${label} must be finite.`);
  return value;
}

function integerValue(id, label) {
  const value = finiteValue(id, label);
  if (!Number.isInteger(value)) throw new Error(`${label} must be an integer.`);
  return value;
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json") ? await response.json() : null;
  if (!response.ok) {
    const message = payload?.error?.message || `Request failed with status ${response.status}.`;
    const error = new Error(message);
    error.code = payload?.error?.code || "request_failed";
    error.status = response.status;
    throw error;
  }
  return payload;
}

async function postJSON(path, payload) {
  return api(path, {
    method: "POST",
    headers: CONTROL_HEADERS,
    body: JSON.stringify(payload),
  });
}

function setMessage(message, tone = "neutral") {
  const element = byId("command-message");
  element.textContent = message;
  element.classList.toggle("error", tone === "error");
  element.classList.toggle("success", tone === "success");
}

function setAnalysisStatus(message, tone = "neutral") {
  const element = byId("analysis-status");
  element.textContent = message;
  element.classList.toggle("busy", tone === "busy");
  element.classList.toggle("ready", tone === "ready");
  element.classList.toggle("error", tone === "error");
}

function prepareCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(280, Math.floor(canvas.clientWidth || 600));
  const declaredHeight = Number(canvas.getAttribute("height")) || 300;
  const height = Math.max(160, Math.floor(canvas.clientHeight || declaredHeight));
  canvas.width = Math.floor(width * ratio);
  canvas.height = Math.floor(height * ratio);
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#050607";
  context.fillRect(0, 0, width, height);
  return { context, width, height };
}

function drawEmptyPlot(canvas, message = "NO MEASUREMENT DATA") {
  const { context, width, height } = prepareCanvas(canvas);
  context.strokeStyle = "#25292c";
  context.lineWidth = 1;
  for (let index = 1; index < 10; index += 1) {
    const x = (width / 10) * index;
    const y = (height / 10) * index;
    context.beginPath();
    context.moveTo(x, 0);
    context.lineTo(x, height);
    context.moveTo(0, y);
    context.lineTo(width, y);
    context.stroke();
  }
  context.fillStyle = "#777e82";
  context.font = '10px "SFMono-Regular", Consolas, monospace';
  context.textAlign = "center";
  context.fillText(message, width / 2, height / 2);
}

function drawXYPlot(canvas, series, options = {}) {
  const { context, width, height } = prepareCanvas(canvas);
  const margins = options.margins || { left: 58, right: 16, top: 18, bottom: 35 };
  const plotWidth = Math.max(1, width - margins.left - margins.right);
  const plotHeight = Math.max(1, height - margins.top - margins.bottom);
  const allPoints = series.flatMap((item) => item.points).filter(
    ([x, y]) => Number.isFinite(x) && Number.isFinite(y),
  );
  if (!allPoints.length) {
    drawEmptyPlot(canvas, options.emptyMessage);
    return null;
  }
  const xValues = allPoints.map(([x]) => x);
  const yValues = allPoints.map(([, y]) => y);
  let xMin = Number.isFinite(options.xMin) ? options.xMin : Math.min(...xValues);
  let xMax = Number.isFinite(options.xMax) ? options.xMax : Math.max(...xValues);
  let yMin = Number.isFinite(options.yMin) ? options.yMin : Math.min(...yValues);
  let yMax = Number.isFinite(options.yMax) ? options.yMax : Math.max(...yValues);
  if (xMin === xMax) [xMin, xMax] = [xMin - 0.5, xMax + 0.5];
  if (yMin === yMax) [yMin, yMax] = [yMin - 0.5, yMax + 0.5];
  const xToPixel = (value) => margins.left + ((value - xMin) / (xMax - xMin)) * plotWidth;
  const yToPixel = (value) => margins.top + ((yMax - value) / (yMax - yMin)) * plotHeight;
  const xTicks = options.xTicks || 10;
  const yTicks = options.yTicks || 10;

  context.save();
  context.beginPath();
  context.rect(margins.left, margins.top, plotWidth, plotHeight);
  context.clip();
  (options.bands || []).forEach((band) => {
    const left = Math.max(margins.left, xToPixel(band.lower));
    const right = Math.min(margins.left + plotWidth, xToPixel(band.upper));
    if (right <= left) return;
    context.fillStyle = band.fill || "rgba(255, 212, 0, 0.08)";
    context.fillRect(left, margins.top, right - left, plotHeight);
    context.strokeStyle = band.stroke || "rgba(255, 212, 0, 0.42)";
    context.setLineDash([4, 4]);
    context.beginPath();
    context.moveTo(left, margins.top);
    context.lineTo(left, margins.top + plotHeight);
    context.moveTo(right, margins.top);
    context.lineTo(right, margins.top + plotHeight);
    context.stroke();
    context.setLineDash([]);
    if (band.label && right - left >= 18) {
      context.fillStyle = band.stroke || "rgba(255, 212, 0, 0.8)";
      context.font = '7px "SFMono-Regular", Consolas, monospace';
      context.textAlign = "center";
      context.fillText(band.label, (left + right) / 2, margins.top + 9);
    }
  });
  context.restore();

  context.strokeStyle = "#24282b";
  context.fillStyle = "#8c9397";
  context.font = '9px "SFMono-Regular", Consolas, monospace';
  context.lineWidth = 1;
  for (let index = 0; index <= xTicks; index += 1) {
    const ratio = index / xTicks;
    const x = margins.left + ratio * plotWidth;
    context.beginPath();
    context.moveTo(x, margins.top);
    context.lineTo(x, margins.top + plotHeight);
    context.stroke();
    const value = xMin + ratio * (xMax - xMin);
    context.textAlign = index === 0 ? "left" : index === xTicks ? "right" : "center";
    context.fillText(
      options.xFormatter ? options.xFormatter(value) : formatNumber(value, 2),
      x,
      height - 12,
    );
  }
  for (let index = 0; index <= yTicks; index += 1) {
    const ratio = index / yTicks;
    const y = margins.top + ratio * plotHeight;
    context.beginPath();
    context.moveTo(margins.left, y);
    context.lineTo(margins.left + plotWidth, y);
    context.stroke();
    const value = yMax - ratio * (yMax - yMin);
    context.textAlign = "right";
    context.fillText(
      options.yFormatter ? options.yFormatter(value) : formatNumber(value, 1),
      margins.left - 7,
      y + 3,
    );
  }
  context.strokeStyle = "#70777b";
  context.strokeRect(margins.left, margins.top, plotWidth, plotHeight);

  context.save();
  context.beginPath();
  context.rect(margins.left, margins.top, plotWidth, plotHeight);
  context.clip();
  series.forEach((item) => {
    context.strokeStyle = item.color;
    context.lineWidth = item.width || 1.4;
    context.setLineDash(item.dash || []);
    context.beginPath();
    let started = false;
    item.points.forEach(([xValue, yValue]) => {
      if (!Number.isFinite(xValue) || !Number.isFinite(yValue)) return;
      const x = xToPixel(xValue);
      const y = yToPixel(yValue);
      if (!started) {
        context.moveTo(x, y);
        started = true;
      } else {
        context.lineTo(x, y);
      }
    });
    context.stroke();
    context.setLineDash([]);
  });
  context.restore();

  if (options.xLabel) {
    context.fillStyle = "#a0a7aa";
    context.textAlign = "center";
    context.fillText(options.xLabel, margins.left + plotWidth / 2, height - 1);
  }
  if (options.yLabel) {
    context.save();
    context.translate(10, margins.top + plotHeight / 2);
    context.rotate(-Math.PI / 2);
    context.textAlign = "center";
    context.fillStyle = "#a0a7aa";
    context.fillText(options.yLabel, 0, 0);
    context.restore();
  }
  return { xMin, xMax, yMin, yMax, xToPixel, yToPixel, margins, plotWidth, plotHeight };
}

function selectedTraceKeys() {
  return Array.from(document.querySelectorAll("#operate-trace-bar input:checked")).map(
    (input) => input.value,
  );
}

function spectrumSeries(payload) {
  if (!payload) return [];
  return payload.traces.map((trace) => ({
    key: trace.key,
    label: trace.label,
    color: TRACE_COLORS[trace.key] || "#ffffff",
    points: payload.frequency_hz.map((frequency, index) => [
      frequency,
      trace.values_dbfs[index],
    ]),
  }));
}

function initializeSpectrumDisplay(payload) {
  if (!payload?.frequency_hz?.length) return;
  const sourceKey = `${payload.center_frequency_hz}:${payload.sample_rate_hz}:${payload.frequency_mode}`;
  if (state.displayInitializedFor === sourceKey) return;
  state.displayInitializedFor = sourceKey;
  const first = payload.frequency_hz[0];
  const last = payload.frequency_hz[payload.frequency_hz.length - 1];
  byId("display-center").value = ((first + last) / 2 / 1e6).toFixed(6).replace(/0+$/, "").replace(/\.$/, "");
  byId("display-span").value = ((last - first + payload.bin_width_hz) / 1e6)
    .toFixed(6)
    .replace(/0+$/, "")
    .replace(/\.$/, "");
  autoSetReference(payload);
}

function autoSetReference(payload = state.analysis) {
  if (!payload) return;
  const values = payload.traces.flatMap((trace) => trace.values_dbfs).filter(Number.isFinite);
  if (!values.length) return;
  const reference = Math.min(20, Math.ceil(Math.max(...values) / 10) * 10);
  byId("reference-level").value = reference;
  byId("db-per-division").value = 10;
}

function analysisBandsForPlot(payload) {
  if (!payload) return [];
  const frequencyOffset = payload.frequency_mode === "absolute" ? payload.center_frequency_hz : 0;
  return payload.bands
    .filter((band) => band.enabled)
    .map((band) => ({
      lower: frequencyOffset + band.center_offset_hz - band.integration_bandwidth_hz / 2,
      upper: frequencyOffset + band.center_offset_hz + band.integration_bandwidth_hz / 2,
      label: band.label,
      fill: band.role === "main" ? "rgba(32, 216, 211, 0.08)" : "rgba(255, 212, 0, 0.055)",
      stroke: band.role === "main" ? "rgba(32, 216, 211, 0.42)" : "rgba(255, 212, 0, 0.35)",
    }));
}

function drawSpectrum(canvasId, compact = false) {
  const canvas = byId(canvasId);
  const payload = state.analysis;
  const series = spectrumSeries(payload);
  if (!payload || !series.length) {
    drawEmptyPlot(canvas, "NO RF ANALYSIS DATA");
    if (canvasId === "spectrum-plot") state.spectrumGeometry = null;
    return;
  }
  initializeSpectrumDisplay(payload);
  const center = finiteValue("display-center", "Display center") * 1e6;
  const span = Math.max(payload.bin_width_hz, finiteValue("display-span", "Display span") * 1e6);
  const reference = finiteValue("reference-level", "Reference level");
  const dbPerDivision = Math.max(0.1, finiteValue("db-per-division", "Log scale"));
  const geometry = drawXYPlot(canvas, series, {
    xMin: compact ? payload.frequency_hz[0] : center - span / 2,
    xMax: compact ? payload.frequency_hz[payload.frequency_hz.length - 1] : center + span / 2,
    yMin: reference - dbPerDivision * 10,
    yMax: reference,
    xFormatter: (value) => formatEngineering(value, "Hz", 2),
    yFormatter: (value) => value.toFixed(0),
    xLabel: payload.frequency_mode === "absolute" ? "RF FREQUENCY" : "FREQUENCY OFFSET",
    yLabel: "dBFS / BIN",
    bands: analysisBandsForPlot(payload),
    xTicks: compact ? 6 : 8,
  });
  if (canvasId === "spectrum-plot") {
    state.spectrumGeometry = geometry;
    renderSpectrumMarker();
    byId("spectrum-start").textContent = `START ${formatEngineering(center - span / 2, "Hz", 2)}`;
    byId("spectrum-stop").textContent = `STOP ${formatEngineering(center + span / 2, "Hz", 2)}`;
    byId("spectrum-bin").textContent = `FFT ${payload.fft_size.toLocaleString()} · BIN BW ${formatEngineering(payload.bin_width_hz, "Hz", 2)}`;
  }
}

function renderSpectrumMarker() {
  const payload = state.analysis;
  const geometry = state.spectrumGeometry;
  const canvas = byId("spectrum-plot");
  if (!payload || !geometry || !Number.isInteger(state.markerIndex)) {
    byId("marker-readout").textContent = "M1 OFF";
    return;
  }
  const index = Math.max(0, Math.min(payload.frequency_hz.length - 1, state.markerIndex));
  const trace = payload.traces[0];
  if (!trace) return;
  const frequency = payload.frequency_hz[index];
  const level = trace.values_dbfs[index];
  const x = geometry.xToPixel(frequency);
  const y = geometry.yToPixel(level);
  const context = canvas.getContext("2d");
  const ratio = window.devicePixelRatio || 1;
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.save();
  context.beginPath();
  context.rect(
    geometry.margins.left,
    geometry.margins.top,
    geometry.plotWidth,
    geometry.plotHeight,
  );
  context.clip();
  context.strokeStyle = TRACE_COLORS[trace.key] || "#ffd400";
  context.setLineDash([3, 3]);
  context.beginPath();
  context.moveTo(x, geometry.margins.top);
  context.lineTo(x, geometry.margins.top + geometry.plotHeight);
  context.stroke();
  context.setLineDash([]);
  context.fillStyle = TRACE_COLORS[trace.key] || "#ffd400";
  context.beginPath();
  context.arc(x, y, 3.5, 0, Math.PI * 2);
  context.fill();
  context.restore();
  byId("marker-readout").textContent =
    `M1  ${trace.label}  ${formatEngineering(frequency, "Hz", 4)}  ${formatNumber(level, 2)} dBFS/bin`;
}

function setMarkerFromPointer(event) {
  const payload = state.analysis;
  const geometry = state.spectrumGeometry;
  if (!payload || !geometry) return;
  const rect = event.currentTarget.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const ratio = Math.max(0, Math.min(1, (x - geometry.margins.left) / geometry.plotWidth));
  const frequency = geometry.xMin + ratio * (geometry.xMax - geometry.xMin);
  let bestIndex = 0;
  let bestDistance = Infinity;
  payload.frequency_hz.forEach((value, index) => {
    const distance = Math.abs(value - frequency);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = index;
    }
  });
  state.markerIndex = bestIndex;
  drawSpectrum("spectrum-plot");
}

function peakSearch() {
  const payload = state.analysis;
  const geometry = state.spectrumGeometry;
  const trace = payload?.traces?.[0];
  if (!trace || !geometry) return;
  let bestIndex = null;
  let bestValue = -Infinity;
  payload.frequency_hz.forEach((frequency, index) => {
    const value = trace.values_dbfs[index];
    if (frequency >= geometry.xMin && frequency <= geometry.xMax && value > bestValue) {
      bestValue = value;
      bestIndex = index;
    }
  });
  state.markerIndex = bestIndex;
  drawSpectrum("spectrum-plot");
}

function drawTrend(canvas = byId("auxiliary-plot")) {
  const records = state.session?.controller?.records || [];
  const points = records.map((record) => [record.iteration, record.nmse_db]);
  drawXYPlot(
    canvas,
    [{ points, color: "#20d8d3", width: 1.8 }],
    {
      xFormatter: (value) => Math.round(value).toString(),
      yFormatter: (value) => value.toFixed(1),
      xLabel: "ITERATION",
      yLabel: "NMSE (dB)",
      emptyMessage: "NO EVALUATED ITERATIONS",
    },
  );
}

function drawPowerTrace(canvas = byId("auxiliary-plot")) {
  const trace = state.session?.controller?.power_trace || [];
  const points = trace.map((item) => [item.attenuation_db, item.power_dbm]);
  const series = [{ points, color: "#ffd400", width: 1.8 }];
  if (points.length) {
    const xValues = points.map(([x]) => x);
    const xMin = Math.min(...xValues);
    const xMax = Math.max(...xValues);
    const rf = state.session?.controller?.rf_config;
    if (Number.isFinite(rf?.target_power_dbm)) {
      series.push({
        points: [[xMin, rf.target_power_dbm], [xMax, rf.target_power_dbm]],
        color: "#5cc977",
        dash: [5, 4],
      });
    }
    if (Number.isFinite(rf?.safety_power_limit_dbm)) {
      series.push({
        points: [[xMin, rf.safety_power_limit_dbm], [xMax, rf.safety_power_limit_dbm]],
        color: "#e54747",
        dash: [5, 4],
      });
    }
  }
  drawXYPlot(canvas, series, {
    xFormatter: (value) => value.toFixed(1),
    yFormatter: (value) => value.toFixed(1),
    xLabel: "TX ATTENUATION (dB)",
    yLabel: "OUTPUT POWER (dBm)",
    emptyMessage: "NO POWER-TUNING TRACE",
  });
}

function drawAclrTemplate(canvas = byId("auxiliary-plot")) {
  const bands = (state.analysis?.bands || [])
    .filter((band) => band.enabled)
    .sort((left, right) => left.center_offset_hz - right.center_offset_hz);
  const measured = bands.filter((band) =>
    [band.traces?.baseline_z?.power_dbfs, band.traces?.target_z?.power_dbfs].some(
      Number.isFinite,
    ),
  );
  if (!measured.length) {
    drawEmptyPlot(canvas, "NO ACLR TEMPLATE DATA");
    return;
  }

  const { context, width, height } = prepareCanvas(canvas);
  const margins = { left: 58, right: 16, top: 22, bottom: 38 };
  const plotWidth = Math.max(1, width - margins.left - margins.right);
  const plotHeight = Math.max(1, height - margins.top - margins.bottom);
  const values = measured.flatMap((band) => [
    band.traces?.baseline_z?.power_dbfs,
    band.traces?.target_z?.power_dbfs,
  ]).filter(Number.isFinite);
  const yMax = Math.ceil(Math.max(...values) / 10) * 10;
  const yMin = Math.min(yMax - 20, Math.floor(Math.min(...values) / 10) * 10);
  const yToPixel = (value) =>
    margins.top + ((yMax - value) / Math.max(1, yMax - yMin)) * plotHeight;

  context.strokeStyle = "#25292c";
  context.fillStyle = "#8c9397";
  context.font = '8px "SFMono-Regular", Consolas, monospace';
  context.textAlign = "right";
  for (let tick = 0; tick <= 5; tick += 1) {
    const value = yMax - ((yMax - yMin) * tick) / 5;
    const y = yToPixel(value);
    context.beginPath();
    context.moveTo(margins.left, y);
    context.lineTo(margins.left + plotWidth, y);
    context.stroke();
    context.fillText(value.toFixed(0), margins.left - 6, y + 3);
  }

  const groupWidth = plotWidth / measured.length;
  const barWidth = Math.max(3, Math.min(13, groupWidth * 0.28));
  measured.forEach((band, index) => {
    const center = margins.left + groupWidth * (index + 0.5);
    const baseline = band.traces?.baseline_z?.power_dbfs;
    const target = band.traces?.target_z?.power_dbfs;
    [
      { value: baseline, color: TRACE_COLORS.baseline_z, offset: -barWidth },
      { value: target, color: TRACE_COLORS.target_z, offset: 0 },
    ].forEach((item) => {
      if (!Number.isFinite(item.value)) return;
      const top = yToPixel(item.value);
      context.fillStyle = item.color;
      context.fillRect(center + item.offset, top, barWidth, margins.top + plotHeight - top);
    });
    context.fillStyle = band.role === "adjacent" ? "#ffd400" : "#9ba2a6";
    context.font = '7px "SFMono-Regular", Consolas, monospace';
    context.textAlign = "center";
    context.fillText(band.label, center, height - 23);
  });

  context.fillStyle = TRACE_COLORS.baseline_z;
  context.fillRect(width - 145, 7, 10, 3);
  context.fillStyle = "#b9bec1";
  context.textAlign = "left";
  context.fillText("Z₀", width - 131, 11);
  context.fillStyle = TRACE_COLORS.target_z;
  context.fillRect(width - 101, 7, 10, 3);
  context.fillStyle = "#b9bec1";
  context.fillText("Zₙ", width - 87, 11);
  context.save();
  context.translate(12, margins.top + plotHeight / 2);
  context.rotate(-Math.PI / 2);
  context.textAlign = "center";
  context.fillText("CHANNEL POWER (dBFS)", 0, 0);
  context.restore();
}

function stimulusResponseSeries(kind) {
  const response = state.analysis?.stimulus_response;
  const baseline = response?.baseline?.points || [];
  const target = response?.target?.points || [];
  const amamSeries = [
    {
      points: baseline.map((point) => [point.input_amplitude, point.output_amplitude]),
      color: TRACE_COLORS.baseline_z,
    },
    {
      points: target.map((point) => [point.input_amplitude, point.output_amplitude]),
      color: TRACE_COLORS.target_z,
    },
  ];
  const ampmSeries = [
    {
      points: baseline.map((point) => [point.input_amplitude, point.phase_degrees]),
      color: TRACE_COLORS.baseline_z,
    },
    {
      points: target.map((point) => [point.input_amplitude, point.phase_degrees]),
      color: TRACE_COLORS.target_z,
    },
  ];
  return kind === "amam" ? amamSeries : ampmSeries;
}

function drawAuxiliary() {
  const canvas = byId("auxiliary-plot");
  const diagnostics = byId("alignment-diagnostics");
  diagnostics.hidden = state.auxView !== "alignment";
  canvas.hidden = state.auxView === "alignment";
  const titles = {
    convergence: "Iteration NMSE",
    aclr: "Channel Power / ACLR Template",
    amam: "AM / AM Stimulus Response",
    ampm: "AM / PM Stimulus Response",
    power: "Attenuation / Output Power",
    alignment: "Capture Alignment Diagnostics",
  };
  byId("auxiliary-title").textContent = titles[state.auxView];
  if (state.auxView === "alignment") {
    renderAlignmentDiagnostics();
    return;
  }
  if (state.auxView === "convergence") {
    drawTrend(canvas);
    return;
  }
  if (state.auxView === "power") {
    drawPowerTrace(canvas);
    return;
  }
  if (state.auxView === "aclr") {
    drawAclrTemplate(canvas);
    return;
  }
  const isAmAm = state.auxView === "amam";
  drawXYPlot(canvas, stimulusResponseSeries(state.auxView), {
    xFormatter: (value) => value.toFixed(2),
    yFormatter: (value) => value.toFixed(isAmAm ? 2 : 1),
    xLabel: "INPUT |Y|",
    yLabel: isAmAm ? "OUTPUT |Z|" : "PHASE (deg)",
    emptyMessage: "NO STIMULUS / RESPONSE DATA",
  });
}

function renderAnalysisResults() {
  const comparison = state.analysis?.comparison;
  const baseline = comparison?.baseline;
  const target = comparison?.target;
  const improvement = comparison?.improvement;
  byId("result-nmse-base").textContent = baseline ? `${formatNumber(baseline.nmse_db)} dB` : "—";
  byId("result-nmse-target").textContent = target ? `${formatNumber(target.nmse_db)} dB` : "—";
  byId("result-nmse-delta").textContent = improvement
    ? formatSigned(improvement.nmse_db, 2, "dB")
    : "— dB";
  byId("result-power-base").textContent = baseline ? `${formatNumber(baseline.power_dbm)} dBm` : "—";
  byId("result-power-target").textContent = target ? `${formatNumber(target.power_dbm)} dBm` : "—";
  byId("result-power-delta").textContent = improvement
    ? formatSigned(improvement.power_db, 2, "dB")
    : "— dB";
  byId("result-papr-base").textContent = baseline ? `${formatNumber(baseline.drive_papr_db)} dB` : "—";
  byId("result-papr-target").textContent = target ? `${formatNumber(target.drive_papr_db)} dB` : "—";
  byId("result-papr-delta").textContent = improvement
    ? formatSigned(improvement.drive_papr_db, 2, "dB")
    : "— dB";

  const container = byId("band-results");
  container.replaceChildren();
  const adjacent = (state.analysis?.bands || []).filter((band) => band.enabled && band.role === "adjacent");
  if (!adjacent.length) {
    const empty = document.createElement("p");
    empty.textContent = "ACLR not configured · define measurement bands in Configuration.";
    container.append(empty);
  } else {
    const header = document.createElement("div");
    header.className = "band-result band-result-head";
    ["ACLR dBc", "BASE", "TARGET", "IMPR"].forEach((value) => {
      const cell = document.createElement("span");
      cell.textContent = value;
      header.append(cell);
    });
    container.append(header);
    adjacent.forEach((band) => {
      const row = document.createElement("div");
      row.className = "band-result";
      const label = document.createElement("strong");
      label.textContent = band.label;
      const base = document.createElement("span");
      const targetValue = document.createElement("span");
      const improvement = document.createElement("em");
      const baselineDbc = band.traces?.baseline_z?.relative_power_dbc;
      const targetDbc = band.traces?.target_z?.relative_power_dbc;
      label.textContent = band.resolved_reference_label
        ? `${band.label} / ${band.resolved_reference_label}`
        : band.label;
      base.textContent = Number.isFinite(baselineDbc)
        ? `${formatNumber(baselineDbc)} dBc`
        : "—";
      targetValue.textContent = Number.isFinite(targetDbc)
        ? `${formatNumber(targetDbc)} dBc`
        : "—";
      improvement.textContent = Number.isFinite(baselineDbc) && Number.isFinite(targetDbc)
        ? formatSigned(baselineDbc - targetDbc, 2, "dB")
        : "—";
      row.append(label, base, targetValue, improvement);
      container.append(row);
    });
  }
  drawSpectrum("spectrum-plot");
  drawAuxiliary();
}

function renderAlignmentDiagnostics() {
  const container = byId("alignment-diagnostics");
  container.replaceChildren();
  const records = state.session?.controller?.records || [];
  const latest = records.length ? records[records.length - 1] : null;
  const segments = latest?.batches?.flatMap((batch) =>
    batch.segments.map((segment) => ({ ...segment, batch_index: batch.batch_index })),
  ) || [];
  if (!segments.length) {
    const empty = document.createElement("p");
    empty.textContent = "No evaluated capture is available.";
    container.append(empty);
    return;
  }
  const header = document.createElement("div");
  header.className = "diagnostic-row";
  ["BATCH / SEGMENT", "DELAY", "PHASE"].forEach((value) => {
    const cell = document.createElement("span");
    cell.textContent = value;
    header.append(cell);
  });
  container.append(header);
  segments.forEach((segment) => {
    const row = document.createElement("div");
    row.className = "diagnostic-row";
    const label = document.createElement("span");
    label.textContent = `B${segment.batch_index} / S${segment.segment_index}${segment.alignment_estimated ? " · EST" : " · REUSE"}`;
    const delay = document.createElement("span");
    delay.textContent = `${formatNumber(segment.delay_samples, 3)} smp`;
    const phase = document.createElement("span");
    phase.textContent = `${formatNumber(segment.phase_radians, 4)} rad`;
    row.append(label, delay, phase);
    container.append(row);
  });
}

function switchWorkspace(name) {
  if (name === "configuration") byId("configuration-dialog").showModal();
  if (name === "runs") {
    byId("runs-dialog").showModal();
    refreshRuns();
  }
  if (name === "analysis") {
    byId("runs-dialog").close();
    syncAnalysisIterationOptions();
    queueAnalysisRefresh(true);
  }
  window.setTimeout(renderAllPlots, 0);
}

function setConnectionState(connected) {
  ["tx-status-dot", "rx-status-dot", "power-status-dot"].forEach((id) => {
    byId(id).classList.toggle("online", connected);
  });
}

function setPathBlock(id, status) {
  const block = byId(id);
  block.classList.toggle("ready", status === "ready");
  block.classList.toggle("active", status === "active");
  block.classList.toggle("fault", status === "fault");
}

function renderSignalPath(controller) {
  const connected = Boolean(controller?.connected);
  const configured = Boolean(controller?.configured);
  const loaded = Boolean(controller?.reference_loaded);
  const transmitting = Boolean(controller?.transmitting);
  const calibrated = ["calibrated", "running", "completed"].includes(controller?.state);
  const failed = controller?.state === "failed";
  byId("path-waveform-state").textContent = loaded ? "Reference loaded" : "Not loaded";
  byId("path-dpd-state").textContent = configured ? "Basic ILC ready" : "Not configured";
  byId("path-tx-state").textContent = transmitting ? "RF active" : connected ? "Connected" : "Offline";
  byId("path-dut-state").textContent = configured ? controller.device_type || "Configured" : "Not configured";
  byId("path-rx-state").textContent = connected ? "Connected" : "Offline";
  byId("path-align-state").textContent = calibrated ? "Calibrated" : "Not calibrated";
  byId("path-analysis-state").textContent = controller?.record_count ? `${controller.record_count} rounds` : "Standby";
  setPathBlock("path-waveform", loaded ? "ready" : failed ? "fault" : "idle");
  setPathBlock("path-dpd", configured ? "ready" : failed ? "fault" : "idle");
  setPathBlock("path-tx", transmitting ? "active" : connected ? "ready" : failed ? "fault" : "idle");
  setPathBlock("path-dut", configured ? "ready" : failed ? "fault" : "idle");
  setPathBlock("path-rx", connected ? "ready" : failed ? "fault" : "idle");
  setPathBlock("path-align", calibrated ? "ready" : failed ? "fault" : "idle");
  setPathBlock("path-analysis", controller?.record_count ? "ready" : failed ? "fault" : "idle");
}

function renderSession(payload) {
  state.session = payload;
  const controller = payload.controller;
  const connected = Boolean(controller?.connected);
  const transmitting = Boolean(controller?.transmitting);
  byId("controller-state").textContent = (controller?.state || "IDLE").toUpperCase();
  byId("run-id").textContent = payload.run_id || "—";
  byId("bench-readout").textContent = controller?.device_type?.toUpperCase() || "—";
  byId("power-value").textContent = formatNumber(controller?.latest_power_dbm);
  byId("attenuation-value").textContent = formatNumber(controller?.locked_attenuation_db);
  byId("iteration-value").textContent = controller
    ? `${controller.iteration ?? "—"} / ${controller.max_iterations ?? "—"}`
    : "— / —";
  byId("gain-value").textContent = formatNumber(controller?.gain_correction, 4);
  byId("center-readout").textContent = Number.isFinite(controller?.rf_config?.center_frequency_hz)
    ? formatEngineering(controller.rf_config.center_frequency_hz, "Hz", 4)
    : "—";
  byId("sample-rate-readout").textContent = Number.isFinite(controller?.rf_config?.sample_rate_hz)
    ? formatEngineering(controller.rf_config.sample_rate_hz, "S/s", 4)
    : "—";
  byId("rf-output-state").textContent = transmitting ? "ON" : "OFF";
  byId("rf-state").classList.toggle("on", transmitting);
  byId("rf-state").classList.toggle("off", !transmitting);
  byId("rf-state").classList.toggle("fault", controller?.state === "failed");
  setConnectionState(connected);
  renderAlignmentDiagnostics();
  updateButtons();
  syncAnalysisIterationOptions();
  drawAuxiliary();
  byId("expert-controller-state").textContent = (controller?.state || "IDLE").toUpperCase();
  byId("expert-run-id").textContent = payload.run_id || "—";
  byId("service-status-dot").classList.add("online");
  byId("service-status-dot").classList.remove("warning");
  byId("service-status").textContent = "CONNECTED";

  const currentKey = `${payload.run_id || ""}:${controller?.iteration ?? ""}:${controller?.record_count ?? 0}`;
  if (byId("analysis-source").value === "session" && currentKey !== state.analysisKey.split("|")[0]) {
    queueAnalysisRefresh();
  }
}

function updateWorkflow(controller) {
  const busy = Boolean(state.session?.active_command_id || state.activeCommand);
  let label = "DEFAULT SIMULATION";
  let status = "Ready to run the complete closed loop.";
  let detail = "One click loads, configures, tunes, calibrates, iterates, stores, and stops RF.";
  if (!state.initialized) {
    label = "INITIALIZING";
    status = "Loading default bench and waveform…";
  } else if (!state.selectedWaveform) {
    label = "BLOCKED";
    status = "No safe MAT waveform is available.";
    detail = "Add a waveform containing vector x to the configured waveform root.";
  } else if (busy) {
    label = "AUTOMATIC LOOP ACTIVE";
    status = `${controller?.active_operation || "run"} · iteration ${controller?.iteration ?? 0}`;
    detail = "RF abort remains available while the command is active.";
  } else if (controller?.state === "completed") {
    label = "SIMULATION COMPLETED";
    status = "Final evaluated result is ready.";
    detail = "Run again with the current draft, export MAT, or inspect the auxiliary results.";
  } else if (controller?.state === "failed" || controller?.state === "stopped") {
    label = "CONTROLLER REQUIRES RESET";
    status = controller.last_error?.message || `Controller is ${controller.state}.`;
    detail = "Reset the controller before starting another default simulation.";
  }
  state.nextAction = { kind: "workspace", value: "configuration" };
  byId("next-action-label").textContent = label;
  byId("workflow-status").textContent = status;
  byId("next-action-detail").textContent = detail;
  byId("next-action").textContent = "Config";
}

function updateButtons() {
  const controller = state.session?.controller;
  const busy = Boolean(state.session?.active_command_id || state.activeCommand);
  const hasWaveform = Boolean(state.selectedWaveform);
  const configured = Boolean(controller?.configured);
  const referenceLoaded = Boolean(controller?.reference_loaded);
  document.querySelectorAll(".command-button").forEach((button) => {
    button.disabled = busy;
  });
  byId("action-load").disabled = busy || !hasWaveform;
  byId("action-configure").disabled = busy || !state.device;
  byId("action-connect").disabled = busy || !controller || controller.connected;
  byId("action-disconnect").disabled = busy || !controller?.connected;
  byId("action-start-tx").disabled =
    busy ||
    !configured ||
    !referenceLoaded ||
    controller?.transmitting ||
    !["ready", "power_ready"].includes(controller?.state);
  byId("action-stop-tx").disabled = busy || !controller?.transmitting;
  byId("action-run").disabled = !state.initialized || busy || !hasWaveform || !state.device;
  byId("config-run").disabled = byId("action-run").disabled;
  byId("config-apply").disabled = busy || !state.device;
  byId("action-power").disabled = busy || !configured || !referenceLoaded;
  byId("action-calibrate").disabled = busy || controller?.state !== "power_ready";
  byId("action-step").disabled = busy || controller?.state !== "calibrated";
  byId("action-export").disabled = busy || controller?.state !== "completed";
  byId("action-reset").disabled = busy || !controller;
  byId("action-stop").disabled = !busy && !controller?.transmitting;
  byId("next-action").disabled = busy;
  if (!state.initialized) byId("action-run").textContent = "Loading Default Bench";
  else if (busy) byId("action-run").textContent = `Running · Iteration ${controller?.iteration ?? 0}`;
  else if (["failed", "stopped"].includes(controller?.state)) byId("action-run").textContent = "Reset Controller";
  else if (controller?.state === "completed") byId("action-run").textContent = "Run Again";
  else byId("action-run").textContent = "Start Default Simulation";
  updateWorkflow(controller);
}

async function loadDevices() {
  const payload = await api("/api/v1/devices");
  state.devices = payload.devices;
  const select = byId("device-select");
  select.replaceChildren();
  state.devices.forEach((device) => {
    const option = document.createElement("option");
    option.value = device.device_type;
    option.textContent =
      device.device_type === "simulated" ? "Simulated RF bench" : device.device_type;
    select.append(option);
  });
  select.disabled = false;
  selectDevice(select.value || state.devices[0]?.device_type);
}

function selectDevice(deviceType) {
  state.device = state.devices.find((item) => item.device_type === deviceType) || null;
  if (!state.device) return;
  const configuration = state.device.default_configuration;
  const common = configuration.device_config;
  byId("center-frequency").value = common.center_frequency_hz / 1e6;
  byId("sample-rate").value = common.sample_rate_hz / 1e6;
  byId("target-power").value = common.target_power_dbm;
  byId("average-segments").value = common.average_segment_count;
  byId("safety-power").value = common.safety_power_limit_dbm;
  byId("initial-attenuation").value = common.initial_attenuation_db;
  byId("minimum-attenuation").value = common.min_attenuation_db;
  byId("maximum-attenuation").value = common.max_attenuation_db;
  byId("settle-seconds").value = common.settle_seconds;
  byId("max-adjustments").value = common.max_adjustments;
  byId("call-timeout").value = common.call_timeout_seconds;
  byId("tx-channel").value = common.tx_channel;
  byId("rx-channel").value = common.rx_channel;
  byId("trigger-source").value = common.trigger;
  byId("max-iterations").value = configuration.max_iterations;
  byId("ilc-mu").value = configuration.runtime_config.mu;
  byId("device-schema-label").textContent =
    `${deviceType.toUpperCase()} · V${state.device.schema.schema_version}`;
  renderDeviceOptions(state.device.schema.fields, common.device_options || {});
  updateButtons();
}

function renderDeviceOptions(fields, configuredOptions = {}) {
  const options = byId("device-option-fields");
  options.replaceChildren();
  const grid = document.createElement("div");
  grid.className = "option-grid";
  const coefficientField = fields.find((field) => field.name === "pa_coefficients");
  fields
    .filter((field) => field.name !== "pa_coefficients")
    .forEach((field) => {
      const hasConfiguredValue = Object.prototype.hasOwnProperty.call(
        configuredOptions,
        field.name,
      );
      const configuredValue = hasConfiguredValue ? configuredOptions[field.name] : field.default;
      const label = document.createElement("label");
      const title = document.createElement("span");
      title.textContent = field.name.replaceAll("_", " ");
      if (field.unit) {
        const unit = document.createElement("small");
        unit.textContent = field.unit;
        title.append(unit);
      }
      let control;
      if (field.enum?.length) {
        control = document.createElement("select");
        field.enum.forEach((value) => {
          const option = document.createElement("option");
          option.value = JSON.stringify(value);
          option.textContent = String(value);
          control.append(option);
        });
        control.value = JSON.stringify(configuredValue ?? field.enum[0]);
      } else if (field.type === "boolean") {
        control = document.createElement("input");
        control.type = "checkbox";
        control.checked = Boolean(configuredValue);
      } else if (field.type === "array" || field.type === "object") {
        control = document.createElement("textarea");
        control.value = JSON.stringify(
          configuredValue ?? (field.type === "array" ? [] : {}),
        );
        control.rows = 3;
      } else {
        control = document.createElement("input");
        control.value = configuredValue ?? "";
        if (field.minimum !== null) control.min = field.minimum;
        if (field.maximum !== null) control.max = field.maximum;
        if (field.step !== null) control.step = field.step;
      }
      control.dataset.option = field.name;
      control.dataset.type = field.type;
      control.title = field.description || field.name;
      label.append(title, control);
      grid.append(label);
  });
  options.append(grid);
  renderCoefficients(
    configuredOptions.pa_coefficients ?? coefficientField?.default ?? [],
  );
}

function renderCoefficients(coefficients) {
  const rows = byId("coefficient-rows");
  rows.replaceChildren();
  coefficients.forEach((coefficient) => addCoefficientRow(coefficient));
  const hidden = !state.device?.schema.fields.some((field) => field.name === "pa_coefficients");
  byId("coefficient-table").hidden = hidden;
  byId("add-coefficient").hidden = hidden;
}

function addCoefficientRow(coefficient = { p: 1, m: 0, real: 0, imag: 0 }) {
  const rows = byId("coefficient-rows");
  if (rows.children.length >= 256) {
    setMessage("The PA coefficient limit is 256 rows.", "error");
    return;
  }
  const row = document.createElement("div");
  row.className = "coefficient-row";
  row.dataset.coefficient = "1";
  ["p", "m", "real", "imag"].forEach((name) => {
    const input = document.createElement("input");
    input.dataset.coefficientField = name;
    input.setAttribute("aria-label", `${name} coefficient value`);
    input.value = coefficient[name];
    row.append(input);
  });
  const remove = document.createElement("button");
  remove.className = "row-remove";
  remove.type = "button";
  remove.setAttribute("aria-label", "Remove coefficient");
  remove.textContent = "×";
  remove.addEventListener("click", () => row.remove());
  row.append(remove);
  rows.append(row);
}

async function loadWaveforms() {
  const queue = [""];
  const waveforms = [];
  while (queue.length && waveforms.length < 500) {
    const directory = queue.shift();
    const payload = await api(
      `/api/v1/waveforms?directory=${encodeURIComponent(directory)}&limit=500`,
    );
    payload.entries.forEach((entry) => {
      if (entry.kind === "directory") queue.push(entry.path);
      else waveforms.push(entry);
    });
  }
  state.waveforms = waveforms;
  const select = byId("waveform-select");
  select.replaceChildren();
  if (!waveforms.length) {
    const option = document.createElement("option");
    option.textContent = "No MAT waveforms found";
    option.value = "";
    select.append(option);
  } else {
    waveforms.forEach((entry) => {
      const option = document.createElement("option");
      option.value = entry.path;
      option.textContent = entry.path;
      select.append(option);
    });
  }
  select.disabled = false;
  byId("refresh-waveforms").disabled = false;
  state.selectedWaveform = select.value;
  await previewSelectedWaveform();
  updateButtons();
}

async function previewSelectedWaveform() {
  state.selectedWaveform = byId("waveform-select").value;
  if (!state.selectedWaveform) {
    byId("waveform-meta").textContent = "Periodic MAT vector · variable x";
    return;
  }
  try {
    const payload = await api(
      `/api/v1/waveforms/preview?path=${encodeURIComponent(state.selectedWaveform)}&points=256`,
    );
    byId("waveform-meta").textContent =
      `${payload.sample_count.toLocaleString()} samples · peak ${formatNumber(payload.safety.reference_peak, 3)} · RMS ${formatNumber(payload.safety.reference_rms, 3)}`;
  } catch (error) {
    setMessage(error.message, "error");
  }
}

function collectConfiguration() {
  if (!state.device) throw new Error("Select an RF bench.");
  const options = {};
  document.querySelectorAll("[data-option]").forEach((input) => {
    const type = input.dataset.type;
    if (input.tagName === "SELECT") options[input.dataset.option] = JSON.parse(input.value);
    else if (type === "integer") options[input.dataset.option] = Number.parseInt(input.value, 10);
    else if (type === "number") options[input.dataset.option] = Number(input.value);
    else if (type === "boolean") options[input.dataset.option] = input.checked;
    else if (type === "array" || type === "object") options[input.dataset.option] = JSON.parse(input.value);
    else options[input.dataset.option] = input.value;
  });
  const coefficientRows = Array.from(document.querySelectorAll("[data-coefficient]"));
  if (coefficientRows.length) {
    options.pa_coefficients = coefficientRows.map((row) => {
      const value = {};
      row.querySelectorAll("[data-coefficient-field]").forEach((input) => {
        const name = input.dataset.coefficientField;
        value[name] =
          name === "p" || name === "m" ? Number.parseInt(input.value, 10) : Number(input.value);
      });
      return value;
    });
  }
  return {
    device_type: state.device.device_type,
    device_config: {
      center_frequency_hz: finiteValue("center-frequency", "Center frequency") * 1e6,
      sample_rate_hz: finiteValue("sample-rate", "Sample rate") * 1e6,
      tx_channel: byId("tx-channel").value,
      rx_channel: byId("rx-channel").value,
      trigger: byId("trigger-source").value,
      average_segment_count: integerValue("average-segments", "Average segments"),
      target_power_dbm: finiteValue("target-power", "Target power"),
      safety_power_limit_dbm: finiteValue("safety-power", "Safety power"),
      initial_attenuation_db: finiteValue("initial-attenuation", "Initial attenuation"),
      min_attenuation_db: finiteValue("minimum-attenuation", "Minimum attenuation"),
      max_attenuation_db: finiteValue("maximum-attenuation", "Maximum attenuation"),
      settle_seconds: finiteValue("settle-seconds", "Settle time"),
      max_adjustments: integerValue("max-adjustments", "Maximum adjustments"),
      call_timeout_seconds: finiteValue("call-timeout", "Call timeout"),
      device_options: options,
    },
    runtime_name: "basic_ilc",
    runtime_config: { mu: finiteValue("ilc-mu", "ILC mu") },
    max_iterations: integerValue("max-iterations", "ILC iterations"),
  };
}

function addMeasurementBandRow(
  band = {
    enabled: false,
    label: "Band",
    role: "other",
    reference_label: "",
    center_mhz: "",
    bandwidth_mhz: "",
  },
) {
  const container = byId("measurement-band-rows");
  if (container.children.length >= 32) {
    setMessage("The measurement-band limit is 32 rows.", "error");
    return;
  }
  const row = document.createElement("div");
  row.className = "measurement-band-row";
  row.dataset.measurementBand = "1";
  row.dataset.referenceLabel = band.reference_label || "";
  const enabled = document.createElement("input");
  enabled.type = "checkbox";
  enabled.checked = Boolean(band.enabled);
  enabled.dataset.bandField = "enabled";
  enabled.setAttribute("aria-label", "Enable measurement band");
  const label = document.createElement("input");
  label.value = band.label;
  label.dataset.bandField = "label";
  label.setAttribute("aria-label", "Measurement band label");
  const role = document.createElement("select");
  role.dataset.bandField = "role";
  ["main", "adjacent", "other"].forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value === "main" ? "TX" : value.toUpperCase();
    role.append(option);
  });
  role.value = band.role;
  const reference = document.createElement("select");
  reference.dataset.bandField = "reference";
  reference.setAttribute("aria-label", "Reference TX channel");
  const center = document.createElement("div");
  center.className = "unit-control";
  const centerInput = document.createElement("input");
  centerInput.value = band.center_mhz;
  centerInput.dataset.bandField = "center";
  centerInput.setAttribute("aria-label", "Band center offset in MHz");
  const centerUnit = document.createElement("b");
  centerUnit.textContent = "MHz";
  center.append(centerInput, centerUnit);
  const bandwidth = document.createElement("div");
  bandwidth.className = "unit-control";
  const bandwidthInput = document.createElement("input");
  bandwidthInput.value = band.bandwidth_mhz;
  bandwidthInput.dataset.bandField = "bandwidth";
  bandwidthInput.setAttribute("aria-label", "Band integration bandwidth in MHz");
  const bandwidthUnit = document.createElement("b");
  bandwidthUnit.textContent = "MHz";
  bandwidth.append(bandwidthInput, bandwidthUnit);
  const remove = document.createElement("button");
  remove.className = "row-remove";
  remove.type = "button";
  remove.setAttribute("aria-label", "Remove measurement band");
  remove.textContent = "×";
  remove.addEventListener("click", () => {
    row.remove();
    syncMeasurementBandReferences();
    queueAnalysisRefresh(true);
  });
  [enabled, label, role].forEach((control) => {
    control.addEventListener("change", () => {
      syncMeasurementBandReferences();
      queueAnalysisRefresh(true);
    });
  });
  [reference, centerInput, bandwidthInput].forEach((control) => {
    control.addEventListener("change", () => queueAnalysisRefresh(true));
  });
  row.append(enabled, label, role, reference, center, bandwidth, remove);
  container.append(row);
  syncMeasurementBandReferences();
}

function syncMeasurementBandReferences() {
  const rows = Array.from(document.querySelectorAll("[data-measurement-band]"));
  const mainLabels = rows
    .filter(
      (row) =>
        row.querySelector('[data-band-field="enabled"]').checked &&
        row.querySelector('[data-band-field="role"]').value === "main",
    )
    .map((row) => row.querySelector('[data-band-field="label"]').value.trim())
    .filter(Boolean);
  rows.forEach((row) => {
    const role = row.querySelector('[data-band-field="role"]').value;
    const reference = row.querySelector('[data-band-field="reference"]');
    const previous = reference.value || row.dataset.referenceLabel || "";
    reference.replaceChildren();
    const unavailable = document.createElement("option");
    unavailable.value = "";
    unavailable.textContent = role === "main" ? "—" : "Select TX";
    reference.append(unavailable);
    mainLabels.forEach((label) => {
      const option = document.createElement("option");
      option.value = label;
      option.textContent = label;
      reference.append(option);
    });
    reference.disabled = role === "main";
    reference.value = role === "main" ? "" : previous;
    row.dataset.referenceLabel = reference.value;
  });
}

function initializeMeasurementBands() {
  byId("measurement-band-rows").replaceChildren();
  Array.from({ length: 10 }, (_, index) => -90 + index * 20).forEach(
    (center, index) => {
      addMeasurementBandRow({
        enabled: true,
        label: `TX${index + 1}`,
        role: "main",
        reference_label: "",
        center_mhz: String(center),
        bandwidth_mhz: "20",
      });
    },
  );
  addMeasurementBandRow({
    enabled: true,
    label: "Adjacent L1",
    role: "adjacent",
    reference_label: "TX1",
    center_mhz: "-110",
    bandwidth_mhz: "20",
  });
  addMeasurementBandRow({
    enabled: true,
    label: "Adjacent R1",
    role: "adjacent",
    reference_label: "TX10",
    center_mhz: "110",
    bandwidth_mhz: "20",
  });
  syncMeasurementBandReferences();
}

function resetDefaults() {
  const deviceType = byId("device-select").value || state.devices[0]?.device_type;
  if (deviceType) selectDevice(deviceType);
  if (state.waveforms.length) {
    byId("waveform-select").value = state.waveforms[0].path;
    state.selectedWaveform = state.waveforms[0].path;
    previewSelectedWaveform();
  }
  initializeMeasurementBands();
  state.analysisKey = "";
  setMessage("Default simulated configuration restored.", "success");
  updateButtons();
}

function switchConfigTab(name) {
  document.querySelectorAll("[data-config-tab]").forEach((button) => {
    button.classList.toggle("active", button.dataset.configTab === name);
  });
  document.querySelectorAll("[data-config-panel]").forEach((panel) => {
    panel.classList.toggle("active", panel.dataset.configPanel === name);
  });
}

function applyDraft() {
  try {
    collectConfiguration();
    collectMeasurementBands();
    byId("configuration-dialog").close();
    state.analysisKey = "";
    setMessage("Configuration draft validated. It will apply to the next run.", "success");
    updateButtons();
  } catch (error) {
    setMessage(error.message, "error");
  }
}

function runPrimaryAction() {
  const controllerState = state.session?.controller?.state;
  if (["failed", "stopped"].includes(controllerState)) submitAction("reset");
  else submitAction("run");
}

function collectMeasurementBands() {
  const bands = [];
  document.querySelectorAll("[data-measurement-band]").forEach((row) => {
    const enabled = row.querySelector('[data-band-field="enabled"]').checked;
    const label = row.querySelector('[data-band-field="label"]').value.trim();
    const role = row.querySelector('[data-band-field="role"]').value;
    const referenceLabel = row.querySelector('[data-band-field="reference"]').value.trim();
    const centerRaw = row.querySelector('[data-band-field="center"]').value.trim();
    const bandwidthRaw = row.querySelector('[data-band-field="bandwidth"]').value.trim();
    if (!enabled && (!centerRaw || !bandwidthRaw)) return;
    const center = Number(centerRaw);
    const bandwidth = Number(bandwidthRaw);
    if (!label) throw new Error("Every enabled measurement band requires a label.");
    if (!Number.isFinite(center) || !Number.isFinite(bandwidth) || bandwidth <= 0) {
      if (enabled) throw new Error(`Measurement band ${label} requires a finite center and positive bandwidth.`);
      return;
    }
    const band = {
      label,
      role,
      center_offset_hz: center * 1e6,
      integration_bandwidth_hz: bandwidth * 1e6,
      enabled,
    };
    if (referenceLabel) band.reference_label = referenceLabel;
    bands.push(band);
  });
  return bands;
}

function analysisRequestPayload() {
  const payload = {
    schema_version: 1,
    points: 1600,
    frequency_mode: byId("frequency-mode").value,
    traces: selectedTraceKeys(),
    bands: collectMeasurementBands(),
    amplitude_floor_db: -50,
  };
  const baseline = byId("baseline-iteration").value;
  const target = byId("target-iteration").value;
  if (baseline !== "") payload.baseline_iteration = Number.parseInt(baseline, 10);
  if (target !== "") payload.target_iteration = Number.parseInt(target, 10);
  return payload;
}

function syncAnalysisIterationOptions() {
  const source = byId("analysis-source").value;
  let iterations = [];
  if (source === "session") {
    iterations = (state.session?.controller?.records || []).map((record) => record.iteration);
  } else {
    iterations = (state.runDetails.get(source)?.iterations || []).map((item) => item.iteration);
  }
  const baseline = byId("baseline-iteration");
  const target = byId("target-iteration");
  const previousBaseline = baseline.value;
  const previousTarget = target.value;
  baseline.replaceChildren();
  target.replaceChildren();
  const autoBaseline = document.createElement("option");
  autoBaseline.value = "";
  autoBaseline.textContent = "First evaluated";
  baseline.append(autoBaseline);
  const autoTarget = document.createElement("option");
  autoTarget.value = "";
  autoTarget.textContent = "Latest evaluated";
  target.append(autoTarget);
  iterations.forEach((iteration) => {
    const baseOption = document.createElement("option");
    baseOption.value = iteration;
    baseOption.textContent = `Iteration ${iteration}`;
    baseline.append(baseOption);
    const targetOption = baseOption.cloneNode(true);
    target.append(targetOption);
  });
  if (Array.from(baseline.options).some((option) => option.value === previousBaseline)) {
    baseline.value = previousBaseline;
  }
  if (Array.from(target.options).some((option) => option.value === previousTarget)) {
    target.value = previousTarget;
  }
}

function queueAnalysisRefresh(force = false) {
  window.clearTimeout(queueAnalysisRefresh.timer);
  queueAnalysisRefresh.timer = window.setTimeout(() => refreshAnalysis(force), 120);
}

async function refreshAnalysis(force = false) {
  if (state.analysisPending) {
    state.analysisRefreshQueued = true;
    return;
  }
  let payload;
  try {
    payload = analysisRequestPayload();
  } catch (error) {
    setAnalysisStatus(error.message, "error");
    return;
  }
  const source = byId("analysis-source").value;
  const sessionKey =
    source === "session"
      ? `${state.session?.run_id || ""}:${state.session?.controller?.iteration ?? ""}:${state.session?.controller?.record_count ?? 0}`
      : source;
  const key = `${sessionKey}|${JSON.stringify(payload)}`;
  if (!force && key === state.analysisKey) return;
  if (source === "session" && !state.session?.controller?.reference_loaded) {
    setAnalysisStatus("LOAD REFERENCE TO ENABLE ANALYSIS");
    return;
  }
  state.analysisPending = true;
  setAnalysisStatus("CALCULATING COMPLETE-PERIOD FFT…", "busy");
  try {
    const path =
      source === "session"
        ? "/api/v1/session/analysis"
        : `/api/v1/runs/${encodeURIComponent(source)}/analysis`;
    state.analysis = await postJSON(path, payload);
    state.analysisKey = key;
    state.markerIndex = null;
    setAnalysisStatus("ANALYSIS READY", "ready");
    renderAnalysisResults();
  } catch (error) {
    if (!["reference_missing", "analysis_configuration_missing"].includes(error.code)) {
      setAnalysisStatus(`${error.code}: ${error.message}`, "error");
    } else {
      setAnalysisStatus(error.message.toUpperCase());
    }
  } finally {
    state.analysisPending = false;
    if (state.analysisRefreshQueued) {
      state.analysisRefreshQueued = false;
      queueAnalysisRefresh(true);
    }
  }
}

async function submitAction(action) {
  try {
    const payload = {
      action,
      request_id: createRequestId(),
    };
    if (action === "load" || action === "run") {
      if (!state.selectedWaveform) throw new Error("Select a reference waveform.");
      payload.waveform_path = state.selectedWaveform;
    }
    if (action === "configure" || action === "run") payload.configuration = collectConfiguration();
    setMessage(`Submitting ${action.replaceAll("_", " ")}…`);
    const status = await postJSON("/api/v1/commands", payload);
    state.activeCommand = status.command_id;
    renderCommandStatus(status);
    updateButtons();
    pollCommand(status.command_id);
  } catch (error) {
    setMessage(error.message, "error");
    await refreshSession();
  }
}

async function pollCommand(commandId) {
  try {
    while (true) {
      const status = await api(`/api/v1/commands/${encodeURIComponent(commandId)}`);
      renderCommandStatus(status);
      if (["completed", "failed", "stopped", "rejected"].includes(status.phase)) {
        state.activeCommand = null;
        state.analysisKey = "";
        await Promise.all([refreshSession(), refreshRuns()]);
        updateButtons();
        queueAnalysisRefresh(true);
        if (status.result_url && status.action === "export") window.location.assign(status.result_url);
        return;
      }
      await new Promise((resolve) => window.setTimeout(resolve, 250));
    }
  } catch (error) {
    state.activeCommand = null;
    setMessage(error.message, "error");
    updateButtons();
  }
}

function renderCommandStatus(status) {
  if (status.error) setMessage(`${status.error.code}: ${status.error.message}`, "error");
  else if (status.phase === "completed") setMessage(status.message || "Command completed.", "success");
  else if (status.phase === "stopped") setMessage(status.message || "Command stopped.");
  else setMessage(`${status.action.replaceAll("_", " ")} · ${status.controller_state}`);
}

async function requestStop() {
  try {
    setMessage("Sending immediate RF-off request…");
    const status = await postJSON("/api/v1/stop", {
      request_id: createRequestId("stop-"),
    });
    renderCommandStatus(status);
    await refreshSession();
  } catch (error) {
    setMessage(error.message, "error");
  }
}

async function refreshRuns() {
  try {
    const payload = await api("/api/v1/runs?limit=50");
    state.runs = payload.runs;
    renderRunList();
    syncAnalysisSources();
  } catch (error) {
    console.warn(error);
  }
}

function syncAnalysisSources() {
  const select = byId("analysis-source");
  const previous = select.value;
  select.replaceChildren();
  const live = document.createElement("option");
  live.value = "session";
  live.textContent = "Live Session";
  select.append(live);
  state.runs.forEach((run) => {
    const option = document.createElement("option");
    option.value = run.run_id;
    option.textContent = `${run.run_id} · ${run.status}`;
    select.append(option);
  });
  if (Array.from(select.options).some((option) => option.value === previous)) {
    select.value = previous;
  }
  syncAnalysisIterationOptions();
}

function renderRunList() {
  const list = byId("run-list");
  list.replaceChildren();
  byId("run-placeholder").hidden = state.runs.length > 0;
  state.runs.forEach((run) => {
    const item = document.createElement("button");
    item.className = "run-item";
    item.type = "button";
    item.classList.toggle("selected", state.selectedRunId === run.run_id);
    const identity = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = run.run_id;
    const device = document.createElement("small");
    device.textContent = run.device_type || "unassigned";
    identity.append(title, device);
    const updated = document.createElement("span");
    updated.textContent = run.updated?.replace("T", " ").replace("Z", "") || "—";
    const iteration = document.createElement("span");
    iteration.textContent = run.latest_iteration ?? "—";
    const status = document.createElement("span");
    status.className = `run-status ${run.status}`;
    status.textContent = run.status;
    item.append(identity, updated, iteration, status);
    item.addEventListener("click", () => openRunDetail(run.run_id));
    list.append(item);
  });
}

async function openRunDetail(runId) {
  try {
    const payload = await api(`/api/v1/runs/${encodeURIComponent(runId)}`);
    const run = payload.run;
    state.runDetails.set(runId, run);
    state.selectedRunId = runId;
    renderRunList();
    byId("run-inspector-empty").hidden = true;
    byId("run-inspector-content").hidden = false;
    byId("run-detail-title").textContent = `${run.run_id} · ${run.status.toUpperCase()}`;
    byId("run-detail-config").textContent = JSON.stringify(run.config, null, 2);
    byId("run-detail-snapshot").textContent = JSON.stringify(run.snapshot, null, 2);
    const summary = byId("run-summary");
    summary.replaceChildren();
    [
      ["STATUS", run.status],
      ["DEVICE", run.device_type || "—"],
      ["ITERATIONS", run.iteration_count],
      ["UPDATED", run.updated],
    ].forEach(([label, value]) => {
      const cell = document.createElement("div");
      const name = document.createElement("span");
      name.textContent = label;
      const content = document.createElement("strong");
      content.textContent = value ?? "—";
      cell.append(name, content);
      summary.append(cell);
    });
    const events = byId("run-detail-events");
    events.replaceChildren();
    run.events
      .slice()
      .reverse()
      .forEach((event) => {
        const item = document.createElement("div");
        item.className = "event-item";
        const time = document.createElement("time");
        time.textContent = event.timestamp;
        const kind = document.createElement("strong");
        kind.textContent = event.kind;
        const message = document.createElement("span");
        message.textContent = event.message;
        item.append(time, kind, message);
        events.append(item);
      });
    const actions = byId("run-detail-actions");
    actions.replaceChildren();
    const analyze = document.createElement("button");
    analyze.className = "secondary-action";
    analyze.type = "button";
    analyze.textContent = "Analyze Iterations";
    analyze.addEventListener("click", () => {
      byId("analysis-source").value = runId;
      syncAnalysisIterationOptions();
      switchWorkspace("analysis");
      queueAnalysisRefresh(true);
    });
    actions.append(analyze);
    if (run.result_available) {
      const download = document.createElement("a");
      download.className = "secondary-action";
      download.href = `/api/v1/runs/${encodeURIComponent(runId)}/result.mat`;
      download.textContent = "Download MAT";
      actions.append(download);
    }
    syncAnalysisSources();
  } catch (error) {
    setMessage(error.message, "error");
  }
}

async function refreshSession() {
  try {
    renderSession(await api("/api/v1/session"));
  } catch (error) {
    byId("service-status-dot").classList.remove("online");
    byId("service-status").textContent = "DISCONNECTED";
    console.warn(error);
  }
}

function connectEvents() {
  if (!("EventSource" in window)) return;
  state.eventSource?.close();
  const source = new EventSource("/api/v1/events");
  state.eventSource = source;
  source.addEventListener("state", (event) => {
    renderSession(JSON.parse(event.data));
  });
  source.onerror = () => {
    byId("service-status-dot").classList.remove("online");
    byId("service-status-dot").classList.add("warning");
    byId("service-status").textContent = "RECONNECTING";
  };
}

function renderAllPlots() {
  try {
    drawSpectrum("spectrum-plot");
  } catch (error) {
    console.warn(error);
  }
  drawAuxiliary();
}

function bindControls() {
  byId("open-configuration").addEventListener("click", () => switchWorkspace("configuration"));
  byId("open-runs").addEventListener("click", () => switchWorkspace("runs"));
  byId("open-expert").addEventListener("click", () => byId("expert-dialog").showModal());
  document.querySelectorAll("[data-dialog-close]").forEach((button) => {
    button.addEventListener("click", () => byId(button.dataset.dialogClose).close());
  });
  document.querySelectorAll("[data-dialog-abort]").forEach((button) => {
    button.addEventListener("click", requestStop);
  });
  document.querySelectorAll("[data-config-tab]").forEach((button) => {
    button.addEventListener("click", () => switchConfigTab(button.dataset.configTab));
  });
  document.querySelectorAll("[data-aux-view]").forEach((button) => {
    button.addEventListener("click", () => {
      state.auxView = button.dataset.auxView;
      document.querySelectorAll("[data-aux-view]").forEach((item) => {
        item.classList.toggle("active", item === button);
      });
      drawAuxiliary();
    });
  });
  byId("device-select").addEventListener("change", (event) => selectDevice(event.target.value));
  byId("waveform-select").addEventListener("change", previewSelectedWaveform);
  byId("refresh-waveforms").addEventListener("click", loadWaveforms);
  byId("refresh-runs").addEventListener("click", refreshRuns);
  byId("add-coefficient").addEventListener("click", () => addCoefficientRow());
  byId("add-measurement-band").addEventListener("click", () => addMeasurementBandRow());
  byId("action-load").addEventListener("click", () => submitAction("load"));
  byId("action-configure").addEventListener("click", () => submitAction("configure"));
  byId("action-connect").addEventListener("click", () => submitAction("connect"));
  byId("action-disconnect").addEventListener("click", () => submitAction("disconnect"));
  byId("action-start-tx").addEventListener("click", () => submitAction("start_transmission"));
  byId("action-stop-tx").addEventListener("click", () => submitAction("stop_transmission"));
  byId("action-power").addEventListener("click", () => submitAction("power_tune"));
  byId("action-calibrate").addEventListener("click", () => submitAction("calibrate"));
  byId("action-step").addEventListener("click", () => submitAction("step"));
  byId("action-run").addEventListener("click", runPrimaryAction);
  byId("action-reset").addEventListener("click", () => submitAction("reset"));
  byId("action-export").addEventListener("click", () => submitAction("export"));
  byId("action-stop").addEventListener("click", requestStop);
  byId("config-reset-defaults").addEventListener("click", resetDefaults);
  byId("config-apply").addEventListener("click", applyDraft);
  byId("config-run").addEventListener("click", () => {
    byId("configuration-dialog").close();
    submitAction("run");
  });
  byId("next-action").addEventListener("click", () => switchWorkspace("configuration"));
  byId("refresh-analysis").addEventListener("click", () => refreshAnalysis(true));
  byId("analysis-refresh").addEventListener("click", () => refreshAnalysis(true));
  byId("frequency-mode").addEventListener("change", () => {
    state.displayInitializedFor = "";
    queueAnalysisRefresh(true);
  });
  ["display-center", "display-span", "reference-level", "db-per-division"].forEach((id) => {
    byId(id).addEventListener("change", () => renderAllPlots());
  });
  document.querySelectorAll("#operate-trace-bar input").forEach((input) => {
    input.addEventListener("change", () => queueAnalysisRefresh(true));
  });
  byId("spectrum-autoset").addEventListener("click", () => {
    state.displayInitializedFor = "";
    initializeSpectrumDisplay(state.analysis);
    renderAllPlots();
  });
  byId("marker-peak").addEventListener("click", peakSearch);
  byId("spectrum-plot").addEventListener("click", setMarkerFromPointer);
  byId("analysis-source").addEventListener("change", async (event) => {
    const runId = event.target.value;
    if (runId !== "session" && !state.runDetails.has(runId)) {
      try {
        const payload = await api(`/api/v1/runs/${encodeURIComponent(runId)}`);
        state.runDetails.set(runId, payload.run);
      } catch (error) {
        setAnalysisStatus(error.message, "error");
        return;
      }
    }
    syncAnalysisIterationOptions();
    state.analysisKey = "";
    queueAnalysisRefresh(true);
  });
  byId("baseline-iteration").addEventListener("change", () => queueAnalysisRefresh(true));
  byId("target-iteration").addEventListener("change", () => queueAnalysisRefresh(true));
  document.addEventListener("keydown", (event) => {
    const editing = event.target instanceof Element && event.target.matches("input, select, textarea");
    if (event.key.toLowerCase() !== "r" || editing) return;
    if (!byId("action-run").disabled) runPrimaryAction();
  });
  window.addEventListener("resize", renderAllPlots);
  window.addEventListener("beforeunload", () => state.eventSource?.close());
}

async function initialize() {
  ["spectrum-plot", "auxiliary-plot"].forEach((id) => drawEmptyPlot(byId(id)));
  initializeMeasurementBands();
  bindControls();
  try {
    await Promise.all([loadDevices(), loadWaveforms(), refreshSession(), refreshRuns()]);
    state.initialized = true;
    connectEvents();
    window.setInterval(refreshSession, 1000);
    setMessage("RF workbench ready.", "success");
    updateButtons();
    queueAnalysisRefresh();
  } catch (error) {
    setMessage(error.message, "error");
  }
}

initialize();
