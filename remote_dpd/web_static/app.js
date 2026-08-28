"use strict";

const CONTROL_HEADERS = {
  "Content-Type": "application/json",
  "X-Remote-DPD-Request": "1",
};

const state = {
  devices: [],
  device: null,
  waveforms: [],
  selectedWaveform: "",
  session: null,
  activeCommand: null,
  preview: null,
  eventSource: null,
  lastPreviewIteration: null,
};

const byId = (id) => document.getElementById(id);

function formatNumber(value, digits = 2) {
  return Number.isFinite(value) ? Number(value).toFixed(digits) : "—";
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

async function control(path, payload) {
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

function drawGrid(canvas, series, colors, fixedRange = null) {
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(280, canvas.clientWidth);
  const height = Number(canvas.getAttribute("height")) || 230;
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);
  context.clearRect(0, 0, width, height);
  context.strokeStyle = "rgba(127, 174, 180, 0.10)";
  context.lineWidth = 1;
  for (let index = 1; index < 5; index += 1) {
    const y = (height / 5) * index;
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(width, y);
    context.stroke();
  }

  const finiteSamples = series.flat().filter(Number.isFinite);
  const minimum = fixedRange ? fixedRange[0] : Math.min(...finiteSamples, 0);
  const maximum = fixedRange ? fixedRange[1] : Math.max(...finiteSamples, 1);
  const span = maximum - minimum || 1;
  series.forEach((values, seriesIndex) => {
    if (!values.length) return;
    context.strokeStyle = colors[seriesIndex];
    context.lineWidth = 1.8;
    context.beginPath();
    values.forEach((value, index) => {
      const x = (index / Math.max(1, values.length - 1)) * width;
      const y = height - 16 - ((value - minimum) / span) * (height - 32);
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    });
    context.stroke();
  });
}

function drawStandbyPlots() {
  const sampleCount = 180;
  const samples = Array.from({ length: sampleCount }, (_, index) => index / sampleCount);
  drawGrid(
    byId("waveform-plot"),
    [
      samples.map((value) => 0.52 + 0.17 * Math.sin(value * Math.PI * 12)),
      samples.map((value) => 0.49 + 0.15 * Math.sin(value * Math.PI * 12 + 0.08)),
      samples.map((value) => 0.51 + 0.16 * Math.sin(value * Math.PI * 12 - 0.05)),
    ],
    ["#39e5d4", "#f6b44c", "#9d8cff"],
    [0.25, 0.75],
  );
  drawGrid(byId("nmse-plot"), [[]], ["#9d8cff"]);
}

function renderSession(payload) {
  state.session = payload;
  const controller = payload.controller;
  byId("controller-state").textContent = controller?.state || "IDLE";
  byId("run-id").textContent = payload.run_id || "—";
  byId("status-led").classList.toggle("online", Boolean(controller?.connected));
  byId("iteration-value").textContent = controller
    ? `${controller.iteration ?? "—"} / ${controller.max_iterations ?? "—"}`
    : "— / —";
  byId("power-value").textContent = formatNumber(controller?.latest_power_dbm);
  byId("attenuation-value").textContent = formatNumber(controller?.locked_attenuation_db);
  byId("gain-value").textContent = formatNumber(controller?.gain_correction, 3);
  const records = controller?.records || [];
  const latest = records.length ? records[records.length - 1] : null;
  byId("nmse-value").textContent = formatNumber(latest?.nmse_db);
  renderLedger(records);
  renderPowerTrace(controller?.power_trace || []);
  drawGrid(
    byId("nmse-plot"),
    [records.map((record) => record.nmse_db)],
    ["#9d8cff"],
  );
  const connectionBanner = byId("connection-banner");
  connectionBanner.classList.add("connected");
  connectionBanner.lastChild.textContent = " Local control service connected.";
  updateButtons();

  const previewIteration = controller?.iteration ?? -1;
  if (controller?.reference_loaded && state.lastPreviewIteration !== previewIteration) {
    state.lastPreviewIteration = previewIteration;
    refreshCurrentPreview();
  }
}

function renderLedger(records) {
  const empty = byId("iteration-empty");
  const ledger = byId("iteration-ledger");
  empty.hidden = records.length > 0;
  ledger.hidden = records.length === 0;
  ledger.replaceChildren();
  if (records.length) {
    const header = document.createElement("div");
    header.className = "ledger-row ledger-head";
    ["Round", "NMSE", "Power", "RMS / peak", "Delay", "Phase", "Segments"].forEach((label) => {
      const cell = document.createElement("span");
      cell.textContent = label;
      header.append(cell);
    });
    ledger.append(header);
  }
  records.slice().reverse().forEach((record) => {
    const row = document.createElement("div");
    row.className = "ledger-row";
    const round = document.createElement("span");
    round.textContent = `#${record.iteration}`;
    const nmse = document.createElement("span");
    nmse.textContent = `${formatNumber(record.nmse_db)} dB`;
    const power = document.createElement("span");
    power.textContent = `${formatNumber(record.power_dbm)} dBm`;
    const digital = document.createElement("span");
    digital.textContent = `${formatNumber(record.digital_rms, 3)} / ${formatNumber(record.digital_peak, 3)}`;
    const firstSegment = record.batches?.[0]?.segments?.[0];
    const delay = document.createElement("span");
    delay.textContent = `${formatNumber(record.delay_samples ?? firstSegment?.delay_samples, 2)} smp`;
    const phase = document.createElement("span");
    phase.textContent = `${formatNumber(record.phase_radians ?? firstSegment?.phase_radians, 3)} rad`;
    const segments = document.createElement("small");
    segments.textContent = `${record.segment_count} seg`;
    row.append(round, nmse, power, digital, delay, phase, segments);
    ledger.append(row);
  });
}

function renderPowerTrace(trace) {
  const container = byId("power-trace");
  container.hidden = trace.length === 0;
  container.replaceChildren();
  trace.forEach((point) => {
    const element = document.createElement("span");
    element.className = "power-point";
    element.textContent = `${formatNumber(point.attenuation_db, 1)} dB → ${formatNumber(point.power_dbm, 2)} dBm`;
    element.title = `Target gap ${formatNumber(point.gap_db, 2)} dB`;
    container.append(element);
  });
}

function renderWaveformPreview(payload) {
  state.preview = payload;
  const series = [payload.x?.magnitude || []];
  if (payload.y?.magnitude) series.push(payload.y.magnitude);
  if (payload.z?.magnitude) series.push(payload.z.magnitude);
  drawGrid(byId("waveform-plot"), series, ["#39e5d4", "#f6b44c", "#9d8cff"]);
}

async function refreshCurrentPreview() {
  try {
    renderWaveformPreview(await api("/api/v1/session/preview?points=800"));
  } catch (error) {
    if (error.code !== "reference_missing") console.warn(error);
  }
}

async function loadDevices() {
  const payload = await api("/api/v1/devices");
  state.devices = payload.devices;
  const select = byId("device-select");
  select.replaceChildren();
  state.devices.forEach((device) => {
    const option = document.createElement("option");
    option.value = device.device_type;
    option.textContent = device.device_type === "simulated" ? "Simulated RF bench" : device.device_type;
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
  byId("device-schema-label").textContent = `${deviceType.toUpperCase()} · V${state.device.schema.schema_version}`;
  renderDeviceOptions(state.device.schema.fields);
  updateButtons();
}

function renderDeviceOptions(fields) {
  const options = byId("device-option-fields");
  options.replaceChildren();
  const grid = document.createElement("div");
  grid.className = "option-grid";
  const coefficientField = fields.find((field) => field.name === "pa_coefficients");
  fields.filter((field) => field.name !== "pa_coefficients").forEach((field) => {
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
      control.value = JSON.stringify(field.default ?? field.enum[0]);
    } else if (field.type === "boolean") {
      control = document.createElement("input");
      control.type = "checkbox";
      control.checked = Boolean(field.default);
    } else if (field.type === "array" || field.type === "object") {
      control = document.createElement("textarea");
      control.value = JSON.stringify(field.default ?? (field.type === "array" ? [] : {}));
      control.rows = 3;
    } else {
      control = document.createElement("input");
      control.value = field.default ?? "";
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
  renderCoefficients(coefficientField?.default || []);
}

function renderCoefficients(coefficients) {
  const rows = byId("coefficient-rows");
  rows.replaceChildren();
  coefficients.forEach((coefficient) => addCoefficientRow(coefficient));
  byId("coefficient-table").hidden = !state.device?.schema.fields.some(
    (field) => field.name === "pa_coefficients",
  );
  byId("add-coefficient").hidden = byId("coefficient-table").hidden;
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
  ["p", "m", "real"].forEach((name) => {
    const input = document.createElement("input");
    input.dataset.coefficientField = name;
    input.setAttribute("aria-label", `${name} coefficient value`);
    input.value = coefficient[name];
    row.append(input);
  });
  const cell = document.createElement("div");
  cell.className = "coefficient-cell";
  const imag = document.createElement("input");
  imag.className = "coefficient-imag";
  imag.dataset.coefficientField = "imag";
  imag.setAttribute("aria-label", "imaginary coefficient value");
  imag.value = coefficient.imag;
  const remove = document.createElement("button");
  remove.className = "coefficient-remove";
  remove.type = "button";
  remove.setAttribute("aria-label", "Remove coefficient");
  remove.textContent = "×";
  remove.addEventListener("click", () => row.remove());
  cell.append(imag, remove);
  row.append(cell);
  rows.append(row);
}

async function loadWaveforms() {
  const queue = [""];
  const waveforms = [];
  while (queue.length && waveforms.length < 500) {
    const directory = queue.shift();
    const payload = await api(`/api/v1/waveforms?directory=${encodeURIComponent(directory)}&limit=500`);
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
      `/api/v1/waveforms/preview?path=${encodeURIComponent(state.selectedWaveform)}&points=800`,
    );
    byId("waveform-meta").textContent = `${payload.sample_count.toLocaleString()} samples · peak ${formatNumber(payload.safety.reference_peak, 3)} · RMS ${formatNumber(payload.safety.reference_rms, 3)}`;
    drawGrid(byId("waveform-plot"), [payload.magnitude], ["#39e5d4"]);
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
        value[name] = name === "p" || name === "m" ? Number.parseInt(input.value, 10) : Number(input.value);
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

async function submitAction(action) {
  try {
    const payload = { action, request_id: `${Date.now().toString(36)}-${crypto.randomUUID().slice(0, 8)}` };
    if (action === "load" || action === "run") {
      if (!state.selectedWaveform) throw new Error("Select a reference waveform.");
      payload.waveform_path = state.selectedWaveform;
    }
    if (action === "configure" || action === "run") payload.configuration = collectConfiguration();
    setMessage(`Submitting ${action.replaceAll("_", " ")}…`);
    const status = await control("/api/v1/commands", payload);
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
        await Promise.all([refreshSession(), refreshRuns()]);
        updateButtons();
        if (status.result_url && status.action === "export") window.location.assign(status.result_url);
        return;
      }
      await new Promise((resolve) => setTimeout(resolve, 250));
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
    setMessage("Sending immediate safety stop…");
    const status = await control("/api/v1/stop", {
      request_id: `stop-${Date.now().toString(36)}`,
    });
    renderCommandStatus(status);
    await refreshSession();
  } catch (error) {
    setMessage(error.message, "error");
  }
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
  byId("action-run").disabled = busy || !hasWaveform || !state.device;
  byId("action-power").disabled = busy || !configured || !referenceLoaded;
  byId("action-calibrate").disabled = busy || controller?.state !== "power_ready";
  byId("action-step").disabled = busy || controller?.state !== "calibrated";
  byId("action-export").disabled = busy || controller?.state !== "completed";
  byId("action-reset").disabled = busy || !controller;
  byId("action-stop").disabled = !busy && !controller?.transmitting;
}

async function refreshRuns() {
  try {
    const payload = await api("/api/v1/runs?limit=12");
    const list = byId("run-list");
    const placeholder = byId("run-placeholder");
    list.replaceChildren();
    placeholder.hidden = payload.runs.length > 0;
    payload.runs.forEach((run) => {
      const item = document.createElement("div");
      item.className = "run-item";
      const main = document.createElement("div");
      const title = document.createElement("strong");
      title.textContent = run.run_id;
      const meta = document.createElement("small");
      meta.textContent = `${run.device_type || "unassigned"} · iteration ${run.latest_iteration ?? "—"}`;
      const actions = document.createElement("div");
      actions.className = "run-actions";
      const inspect = document.createElement("button");
      inspect.className = "text-button";
      inspect.type = "button";
      inspect.textContent = "Inspect";
      inspect.addEventListener("click", () => openRunDetail(run.run_id));
      actions.append(inspect);
      if (run.result_available) {
        const download = document.createElement("a");
        download.href = `/api/v1/runs/${encodeURIComponent(run.run_id)}/result.mat`;
        download.textContent = "Download MAT";
        actions.append(download);
      }
      main.append(title, meta, actions);
      const badge = document.createElement("span");
      badge.className = "run-status";
      badge.textContent = run.status;
      item.append(main, badge);
      list.append(item);
    });
  } catch (error) {
    console.warn(error);
  }
}

async function openRunDetail(runId) {
  try {
    const payload = await api(`/api/v1/runs/${encodeURIComponent(runId)}`);
    const run = payload.run;
    byId("run-dialog-title").textContent = `${run.run_id} · ${run.status}`;
    byId("run-dialog-snapshot").textContent = JSON.stringify(run.snapshot, null, 2);
    byId("run-dialog-config").textContent = JSON.stringify(run.config, null, 2);
    const events = byId("run-dialog-events");
    events.replaceChildren();
    run.events.slice().reverse().forEach((event) => {
      const item = document.createElement("div");
      item.className = "event-item";
      const time = document.createElement("time");
      time.textContent = event.timestamp;
      const message = document.createElement("span");
      message.textContent = `${event.kind}: ${event.message}`;
      item.append(time, message);
      events.append(item);
    });
    byId("run-dialog").showModal();
  } catch (error) {
    setMessage(error.message, "error");
  }
}

async function refreshSession() {
  try {
    renderSession(await api("/api/v1/session"));
  } catch (error) {
    byId("connection-banner").classList.remove("connected");
    byId("connection-banner").lastChild.textContent = ` ${error.message}`;
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
    const connectionBanner = byId("connection-banner");
    connectionBanner.classList.remove("connected");
    connectionBanner.lastChild.textContent = " Live updates reconnecting…";
  };
}

function bindControls() {
  byId("device-select").addEventListener("change", (event) => selectDevice(event.target.value));
  byId("waveform-select").addEventListener("change", previewSelectedWaveform);
  byId("refresh-waveforms").addEventListener("click", loadWaveforms);
  byId("refresh-runs").addEventListener("click", refreshRuns);
  byId("add-coefficient").addEventListener("click", () => addCoefficientRow());
  byId("action-load").addEventListener("click", () => submitAction("load"));
  byId("action-configure").addEventListener("click", () => submitAction("configure"));
  byId("action-connect").addEventListener("click", () => submitAction("connect"));
  byId("action-disconnect").addEventListener("click", () => submitAction("disconnect"));
  byId("action-start-tx").addEventListener("click", () => submitAction("start_transmission"));
  byId("action-stop-tx").addEventListener("click", () => submitAction("stop_transmission"));
  byId("action-power").addEventListener("click", () => submitAction("power_tune"));
  byId("action-calibrate").addEventListener("click", () => submitAction("calibrate"));
  byId("action-step").addEventListener("click", () => submitAction("step"));
  byId("action-run").addEventListener("click", () => submitAction("run"));
  byId("action-reset").addEventListener("click", () => submitAction("reset"));
  byId("action-export").addEventListener("click", () => submitAction("export"));
  byId("action-stop").addEventListener("click", requestStop);
  byId("run-dialog-close").addEventListener("click", () => byId("run-dialog").close());
  document.addEventListener("keydown", (event) => {
    const editing = event.target instanceof Element && event.target.matches("input, select, textarea");
    if (event.key.toLowerCase() !== "r" || editing) return;
    if (!byId("action-run").disabled) submitAction("run");
  });
  window.addEventListener("resize", () => {
    if (state.preview) renderWaveformPreview(state.preview);
    else drawStandbyPlots();
    const records = state.session?.controller?.records || [];
    drawGrid(byId("nmse-plot"), [records.map((record) => record.nmse_db)], ["#9d8cff"]);
  });
  window.addEventListener("beforeunload", () => state.eventSource?.close());
}

async function initialize() {
  drawStandbyPlots();
  bindControls();
  try {
    await Promise.all([loadDevices(), loadWaveforms(), refreshSession(), refreshRuns()]);
    connectEvents();
    window.setInterval(refreshSession, 1000);
    setMessage("Local console ready.", "success");
    updateButtons();
  } catch (error) {
    setMessage(error.message, "error");
  }
}

initialize();
