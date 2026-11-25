"use strict";

const N = 64;
const $ = (id) => document.getElementById(id);
const localInference = window.GRAVITY_LOCAL_INFERENCE === true;
// Jet-style spectrum used by both the heatmaps and their color bars.
const jetPalette = [
  [0, 0, 128], [0, 0, 255], [0, 128, 255], [0, 255, 255], [128, 255, 128],
  [255, 255, 0], [255, 128, 0], [255, 0, 0], [128, 0, 0]
];
const residualPalette = [
  [31, 65, 143], [133, 169, 224], [255, 255, 255], [229, 149, 140], [168, 35, 43]
];
const number = (x, decimals = 2) => Number.isFinite(x) ? x.toFixed(decimals) : "—";
const view = { active: null, fixture: null, density: null, prediction: null, error: null, pending: false, sequence: 0 };
if (localInference) {
  $("connection-mode").textContent = "本地推理服务";
  $("prediction-mode").textContent = "（实时推理）";
  $("latency-title").textContent = "推理耗时";
  document.querySelector(".method").textContent = "示例为合成数据，耗时不含模型加载与网络请求。";
  $("custom-tab").hidden = false;
  $("run").hidden = false;
}
const densityView = createDensityView($("volume-view"), paletteColor);
const MAX_PRISMS = 16;
const prismFields = [
  ["x", "中心 X (km)", 0, 20], ["y", "中心 Y (km)", 0, 20],
  ["depth", "中心埋深 (km)", 0, 7], ["height", "垂向高度 (km)", 0, 7],
  ["length", "X 向长度 (km)", 0, 20], ["width", "Y 向宽度 (km)", 0, 20],
  ["contrast", "密度差 (kg/m³)", -2000, 2500]
];
const firstPrism = {x: 10, y: 10, depth: 2.5, height: 1.5, length: 3, width: 2.5, contrast: 600};
let generatedParams = null;

async function request(path, options, binary = false) {
  const res = await fetch(path, options);
  if (!res.ok) {
    const info = await res.json().catch(() => ({}));
    throw new Error(info.error || (localInference ? "本地服务请求失败。" : "无法读取演示数据。"));
  }
  return binary ? res.arrayBuffer() : res.json();
}

function status(label, kind = "normal") {
  $("state").textContent = label;
  $("state").dataset.kind = kind;
}
function notice(message = "") {
  $("alert").textContent = message;
  $("alert").hidden = !message;
}
function limits(rows) {
  let low = Infinity, high = -Infinity;
  for (const row of rows) for (const v of row) {
    low = Math.min(low, v);
    high = Math.max(high, v);
  }
  return [low, high];
}
function paletteColor(value, [min, max], rgb, palette = jetPalette) {
  const last = palette.length - 1;
  const position = Math.min(1, Math.max(0, max === min ? .5 : (value - min) / (max - min))) * last;
  const left = Math.min(last - 1, Math.floor(position));
  const weight = position - left;
  for (let channel = 0; channel < 3; channel++) {
    rgb[channel] = palette[left][channel] * (1 - weight) + palette[left + 1][channel] * weight;
  }
}
function paint(canvas, get, span, palette = jetPalette) {
  const ctx = canvas.getContext("2d");
  const {width, height} = canvas;
  const img = ctx.createImageData(width, height);
  const rgb = [0, 0, 0];
  for (let row = 0; row < height; row++) for (let x = 0; x < width; x++) {
    paletteColor(get(x, height - 1 - row), span, rgb, palette);
    const off = (row * width + x) * 4;
    img.data.set(rgb, off);
    img.data[off+3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}
function drawDensity() {
  if (!view.fixture) return;
  const k = Number($("layer").value), scale = view.fixture.densityLimit;
  paint($("density"), (x,y) => view.density[(x*N+y)*N+k], [-scale,scale]);
  const km = (k+.5)/N*view.fixture.bounds[2]/1000;
  $("depth").textContent = number(km)+" km";
  $("layer").setAttribute("aria-valuetext", "深度 "+number(km)+" 千米，第 "+(k+1)+" 层");
  $("density-min").textContent = number(-scale,0);
  $("density-max").textContent = number(scale,0);
  densityView.setLayer(k, N);
}
function clearMetrics() {
  for (const id of ["rrmse","rmse","corr","latency","error-min","error-max"]) $(id).textContent = "—";
  $("device").textContent = "模型前向 · ms";
}
function drawResults() {
  if (!view.fixture) return;
  const {truth} = view.fixture;
  const fields = view.prediction ? limits([...truth, ...view.prediction]) : limits(truth);
  paint($("truth"), (x,y) => truth[x][y], fields);
  $("field-min").textContent = number(fields[0]);
  $("field-max").textContent = number(fields[1]);
  $("truth-placeholder").hidden = true;
  if (!view.prediction) return;
  const maxDelta = Math.max(...limits(view.error).map(Math.abs));
  paint($("prediction"), (x,y) => view.prediction[x][y], fields);
  paint($("error"), (x,y) => view.error[x][y], [-maxDelta,maxDelta], residualPalette);
  $("pred-placeholder").hidden = true;
  $("err-placeholder").hidden = true;
  $("error-min").textContent = number(-maxDelta);
  $("error-max").textContent = number(maxDelta);
}
function chartScale(low, high, target = 5) {
  const approximate = (high - low) / target;
  const magnitude = 10 ** Math.floor(Math.log10(approximate));
  const step = [1, 2, 2.5, 5, 10].find(n => n >= approximate / magnitude) * magnitude;
  const first = Math.floor(low / step);
  const last = Math.ceil(high / step);
  const decimals = Math.min(12, Math.max(0, -Math.floor(Math.log10(step))) +
    (Math.abs(step / 10 ** Math.floor(Math.log10(step)) - 2.5) < 1e-9 ? 1 : 0));
  return {
    low: first * step,
    high: last * step,
    ticks: Array.from({length: last - first + 1}, (_, i) => {
      const value = (first + i) * step;
      return {value, label: String(Number(value.toFixed(decimals)))};
    })
  };
}
function chartAxes(canvas, horizontal, vertical, xLabel, yLabel, square = false) {
  const width = canvas.clientWidth, height = canvas.clientHeight;
  if (width < 100 || height < 100) return null;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.font = '11px "Segoe UI", "Microsoft YaHei", sans-serif';
  const labelWidth = Math.max(...vertical.ticks.map(({label}) => ctx.measureText(label).width));
  const leftMargin = Math.ceil(28 + labelWidth), rightMargin = 18;
  const baseWidth = width - leftMargin - rightMargin;
  const baseHeight = height - 57;
  const w = square ? Math.min(baseWidth, baseHeight) : baseWidth;
  const h = square ? w : baseHeight;
  const left = leftMargin + (baseWidth - w) / 2;
  const top = 14, bottom = top + h, right = left + w;
  const x = value => left + (value - horizontal.low) / (horizontal.high - horizontal.low) * w;
  const y = value => bottom - (value - vertical.low) / (vertical.high - vertical.low) * h;
  ctx.fillStyle = "#f8faf9";
  ctx.fillRect(left, top, w, h);
  ctx.beginPath();
  ctx.strokeStyle = "#e2e8e4";
  ctx.fillStyle = "#465650";
  ctx.textAlign = "center";
  for (const {value, label} of horizontal.ticks) {
    const px = x(value);
    ctx.moveTo(px, top);
    ctx.lineTo(px, bottom);
    ctx.fillText(label, px, bottom + 14);
  }
  ctx.textAlign = "right";
  for (const {value, label} of vertical.ticks) {
    const py = y(value);
    ctx.moveTo(left, py);
    ctx.lineTo(right, py);
    ctx.fillText(label, left - 6, py + 4);
  }
  ctx.stroke();
  ctx.strokeStyle = "#aebbb4";
  ctx.strokeRect(left, top, w, h);
  ctx.textAlign = "center";
  ctx.fillText(xLabel, left + w / 2, bottom + 32);
  ctx.save();
  ctx.translate(13, top + h / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(yLabel, 0, 0);
  ctx.restore();
  return {ctx, x, y, bottom};
}
function drawHistogram(domain, truth, prediction) {
  const bins = 36;
  const step = (domain.high - domain.low) / bins;
  const count = values => {
    const frequencies = Array(bins).fill(0);
    for (const value of values) {
      const index = Math.max(0, Math.min(bins - 1, Math.floor((value - domain.low) / step)));
      frequencies[index]++;
    }
    return frequencies;
  };
  const referenceCounts = count(truth), predictionCounts = count(prediction);
  const maximum = Math.max(1, ...referenceCounts, ...predictionCounts);
  const chart = chartAxes($("histogram"), domain, chartScale(0, maximum), "重力异常 (mGal)", "像素数");
  if (!chart) return;
  const {ctx, x, y, bottom} = chart;
  for (const [counts, fill, outline] of [
    [referenceCounts, "rgba(106, 117, 124, .52)", "#626b73"],
    [predictionCounts, "rgba(223, 108, 81, .55)", "#a34c39"]
  ]) {
    ctx.fillStyle = fill;
    ctx.strokeStyle = outline;
    counts.forEach((frequency, index) => {
      if (!frequency) return;
      const left = x(domain.low + index * step);
      const width = x(domain.low + (index + 1) * step) - left;
      const height = bottom - y(frequency);
      ctx.fillRect(left, bottom - height, width, height);
      ctx.strokeRect(left, bottom - height, width, height);
    });
  }
}
function drawScatter(domain, truth, prediction) {
  const chart = chartAxes($("scatter"), domain, domain, "解析参考值 (mGal)", "FNO 预测值 (mGal)", true);
  if (!chart) return;
  const {ctx, x, y} = chart;
  ctx.fillStyle = "rgba(40, 88, 185, .36)";
  for (let i = 0; i < truth.length; i++) {
    ctx.fillRect(x(truth[i]) - 1, y(prediction[i]) - 1, 2, 2);
  }
  ctx.strokeStyle = "#35424b";
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(x(domain.low), y(domain.low));
  ctx.lineTo(x(domain.high), y(domain.high));
  ctx.stroke();
  ctx.setLineDash([]);
}
function drawDiagnostics() {
  if (!view.fixture || !view.prediction) return;
  const truth = view.fixture.truth.flat();
  const prediction = view.prediction.flat();
  const [low, high] = limits([truth, prediction]);
  const pad = (high - low) * .03 || Math.max(.01, Math.abs(low) * .03);
  const domain = chartScale(low - pad, high + pad, 4);
  $("pixel-count").textContent = truth.length.toLocaleString("zh-CN") + " 个像素";
  $("diagnostics").hidden = false;
  drawHistogram(domain, truth, prediction);
  drawScatter(domain, truth, prediction);
}
function resetOutput() {
  view.prediction = null;
  view.error = null;
  $("diagnostics").hidden = true;
  clearMetrics();
  for (const id of ["prediction","error"]) $(id).getContext("2d").clearRect(0,0,N,N);
  for (const id of ["pred-placeholder","err-placeholder"]) {
    $(id).hidden = false;
    $(id).textContent = localInference ? "待运行" : "加载中";
  }
  $("probe").textContent = "观测点 · —";
}
function clearSample(item, placeholder) {
  view.active = item;
  view.fixture = null;
  view.density = null;
  densityView.setModel(null);
  $("run").disabled = true;
  $("layer").disabled = true;
  $("depth").textContent = "—";
  for (const id of ["density-min","density-max","volume-min","volume-max","field-min","field-max","domain-label"]) $(id).textContent = "—";
  $("density").getContext("2d").clearRect(0,0,N,N);
  $("case-label").textContent = item.name + " · " + item.description;
  $("truth-placeholder").hidden = false;
  $("truth-placeholder").textContent = placeholder;
  resetOutput();
  notice();
}
function markSelected(id) {
  const custom = id === "custom";
  $("custom-form").hidden = !custom;
  $("custom-tab").setAttribute("aria-pressed", String(custom));
  for (const button of $("sample-buttons").children) {
    button.setAttribute("aria-pressed", String(button.dataset.id === id));
  }
}
function showSample(fixture, buffer, ready) {
  if (buffer.byteLength !== N*N*N*4) throw new Error("密度样本网格尺寸异常。");
  view.fixture = fixture;
  view.density = new Float32Array(buffer);
  $("layer").value = fixture.defaultLayer;
  $("layer").disabled = false;
  $("domain-label").textContent = "模型域 "+fixture.bounds.map(x => x/1000).join(" × ")+" km";
  $("volume-min").textContent = number(-fixture.densityLimit, 0);
  $("volume-max").textContent = number(fixture.densityLimit, 0);
  densityView.setModel(view.density, N, fixture.bounds, fixture.densityLimit);
  drawDensity();
  if (!localInference) return;
  drawResults();
  $("run").disabled = !ready;
  status(ready ? "待运行" : "缺少权重", ready ? "normal" : "error");
  if (!ready) notice("未找到对应权重；仍可查看密度模型、切片与解析参考值。");
}
async function selectSample(item) {
  const sequence = ++view.sequence;
  generatedParams = null;
  markSelected(item.id);
  clearSample(item, "加载中");
  status("正在加载样本", "busy");
  try {
    if (localInference) {
      const [fixture, buffer] = await Promise.all([
        request("/api/sample/" + item.id), request("/api/density/" + item.id, undefined, true)
      ]);
      if (sequence !== view.sequence) return;
      showSample(fixture, buffer, item.ready);
      return;
    }
    const base = "./demo-data/" + item.id;
    const [{fixture, result}, buffer] = await Promise.all([
      request(base + ".json"), request(base + ".bin", undefined, true)
    ]);
    if (sequence !== view.sequence) return;
    showSample(fixture, buffer);
    showOutput(result);
    status("结果已加载");
  } catch (err) {
    if (sequence !== view.sequence) return;
    $("truth-placeholder").textContent = "未能读取";
    if (!localInference) {
      $("pred-placeholder").textContent = "未能读取";
      $("err-placeholder").textContent = "未能读取";
    }
    status("载入失败", "error");
    notice(err.message);
  }
}
function customNotice(message = "", error = false) {
  const node = $("custom-notice");
  node.hidden = !message;
  node.textContent = message;
  node.dataset.error = String(error);
}
function openCustom(message = "") {
  view.sequence++;
  generatedParams = null;
  markSelected("custom");
  clearSample({id: "custom", name: "自定义样本", description: "参数建模", ready: false}, "请先生成密度模型");
  status("待生成");
  customNotice(message);
}
function refreshPrismList() {
  const cards = [...$("prism-list").children];
  cards.forEach((card, i) => {
    card.querySelector("h3 span").textContent = "棱柱 " + (i + 1);
    const button = card.querySelector("button");
    button.disabled = cards.length < 2;
    button.setAttribute("aria-label", "删除棱柱 " + (i + 1));
  });
  $("add-prism").disabled = cards.length >= MAX_PRISMS;
}
function addPrism(values, focus = false) {
  if ($("prism-list").children.length >= MAX_PRISMS) return;
  const card = document.createElement("section");
  card.className = "prism-card";
  const title = document.createElement("h3");
  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "remove-prism";
  remove.textContent = "删除";
  remove.addEventListener("click", () => {
    card.remove();
    refreshPrismList();
    openCustom("参数已修改，请重新生成。");
    $("add-prism").focus();
  });
  title.append(document.createElement("span"), remove);
  const fields = document.createElement("div");
  fields.className = "prism-inputs";
  for (const [key, name, min, max] of prismFields) {
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "number";
    input.name = key;
    input.required = true;
    input.step = "any";
    input.min = min;
    input.max = max;
    input.value = values[key];
    label.append(name, input);
    fields.append(label);
  }
  card.append(title, fields);
  $("prism-list").append(card);
  refreshPrismList();
  if (focus) card.querySelector("input").focus();
}
function setBusy(busy) {
  view.pending = busy;
  for (const button of $("sample-buttons").children) button.disabled = busy;
  $("custom-tab").disabled = busy;
  $("custom-fields").disabled = busy;
}
function jsonPost(payload) {
  return {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)};
}
async function generateCustom(event) {
  event.preventDefault();
  if (view.pending || view.active?.id !== "custom") return;
  const model = $("custom-model");
  const modelName = model.selectedOptions[0].textContent;
  const prisms = [...$("prism-list").children].map(card =>
    Object.fromEntries([...card.querySelectorAll("input")].map(input => [input.name, input.valueAsNumber])));
  const params = {model: model.value, prisms};
  openCustom();
  setBusy(true);
  status("正在生成", "busy");
  $("truth-placeholder").textContent = "正在生成";
  try {
    const fixture = await request("/api/custom/preview", jsonPost(params));
    const buffer = Uint8Array.from(atob(fixture.density), ch => ch.charCodeAt(0)).buffer;
    delete fixture.density;
    view.active.ready = fixture.ready;
    $("case-label").textContent = "自定义样本 · " + prisms.length + " 个棱柱 · " + modelName + "权重";
    showSample(fixture, buffer, fixture.ready);
    generatedParams = params;
    $("layer").focus();
    customNotice(fixture.outsideDensityRange
      ? "模型已生成。重叠后的密度差超出训练样本的单棱柱设定范围，预测仅供对比。"
      : "模型已生成。可先查看切片；自定义结构的预测精度以解析对比为准。");
    if (fixture.ready) status("模型已生成");
  } catch (err) {
    $("truth-placeholder").textContent = "未生成";
    status("生成失败", "error");
    customNotice(err.message, true);
  } finally {
    setBusy(false);
  }
}
function showOutput(result) {
  view.prediction = result.prediction;
  view.error = result.error;
  drawResults();
  drawDiagnostics();
  $("rrmse").textContent = result.metrics.rrmse == null ? "—" : number(result.metrics.rrmse) + "%";
  $("rmse").textContent = number(result.metrics.rmse, 3);
  $("corr").textContent = number(result.metrics.corr, 4);
  $("latency").textContent = number(result.metrics.ms, 0);
  $("device").textContent = result.device + " · ms" + (result.fallback ? "（回退）" : "");
}
async function run() {
  if (!view.fixture || view.pending || !view.active.ready) return;
  setBusy(true);
  $("run").disabled = true;
  $("run-label").textContent = "正在推理";
  status("正在推理", "busy");
  notice();
  resetOutput();
  drawResults();
  $("pred-placeholder").textContent = "计算中";
  $("err-placeholder").textContent = "待计算";
  try {
    const out = view.active.id === "custom"
      ? await request("/api/custom/predict", jsonPost(generatedParams))
      : await request("/api/predict/"+view.active.id, {method:"POST"});
    showOutput(out);
    status("推理完成");
  } catch (err) {
    status("推理失败", "error");
    resetOutput();
    notice(err.message);
  } finally {
    setBusy(false);
    $("run").disabled = false;
    $("run-label").textContent = "运行正演";
  }
}
function probe(event) {
  if (!view.fixture) return;
  const rect = event.currentTarget.getBoundingClientRect();
  const index = t => Math.min(N-1, Math.max(0, Math.floor(t*N)));
  const x = index((event.clientX-rect.left)/rect.width);
  const y = N-1-index((event.clientY-rect.top)/rect.height);
  const km = (i, v) => number((i+.5)*v/N/1000);
  const [lx,ly] = view.fixture.bounds;
  const from = (matrix) => matrix ? number(matrix[x][y], 3) : "—";
  $("probe").textContent = "观测点 · X "+km(x,lx)+" km / Y "+km(y,ly)+" km"
      + "　参考 "+from(view.fixture.truth)+"　预测 "+from(view.prediction)+"　残差 "+from(view.error)+" mGal";
}
function setupLocalControls(samples) {
  for (const item of samples) $("custom-model").add(new Option(item.name, item.id));
  $("custom-model").value = samples.find(item => item.id === "single" && item.ready)?.id
    || samples.find(item => item.ready)?.id || samples[0].id;
  $("custom-tab").disabled = false;
  $("run").addEventListener("click", run);
  $("custom-tab").addEventListener("click", () => {
    if (view.active?.id !== "custom") openCustom();
  });
  $("add-prism").addEventListener("click", () => {
    const i = $("prism-list").children.length;
    addPrism({...firstPrism, x: 4 + (i * 3) % 12, y: 7 + (i * 2) % 8, contrast: i % 2 ? -500 : 600}, true);
    openCustom("参数已修改，请重新生成。");
  });
  $("custom-form").addEventListener("input", () => {
    if (generatedParams || $("custom-notice").dataset.error === "true") {
      openCustom("参数已修改，请重新生成。");
    }
  });
  $("custom-form").addEventListener("submit", generateCustom);
  addPrism(firstPrism);
}
async function start() {
  for (const ramp of document.querySelectorAll(".ramp")) {
    const palette = ramp.id === "error-ramp" ? residualPalette : jetPalette;
    paint(ramp, x => x, [0, ramp.width - 1], palette);
  }
  const chartObserver = new ResizeObserver(drawDiagnostics);
  chartObserver.observe($("histogram"));
  chartObserver.observe($("scatter"));
  $("layer").addEventListener("input", drawDensity);
  for (const id of ["truth","prediction","error"]) {
    $(id).addEventListener("pointermove", probe);
    $(id).addEventListener("pointerleave", () => $("probe").textContent = "观测点 · —");
  }
  try {
    const samples = await request(localInference ? "/api/samples" : "./demo-data/samples.json");
    for (const item of samples) {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.id = item.id;
      button.textContent = item.name;
      button.title = item.description;
      button.addEventListener("click", () => { if (!view.pending) selectSample(item); });
      $("sample-buttons").append(button);
    }
    if (!samples.length) throw new Error("没有可用的内置样本。");
    if (localInference) setupLocalControls(samples);
    selectSample(samples[0]);
  } catch (err) {
    status("载入失败", "error");
    notice(location.protocol === "file:" ? "请通过 GitHub Pages 或本地 HTTP 地址打开演示。" : err.message);
  }
}
start();
