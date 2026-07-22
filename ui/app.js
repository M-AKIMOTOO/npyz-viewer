"use strict";

const invoke = window.__TAURI__.core.invoke;
const listen = window.__TAURI__.event.listen;

const state = {
  summary: null,
  array: null,
  derived: [],
  page: 1,
  pageSize: 100,
  searchCursor: -1,
  match: null,
  activeEntry: null,
  statsCache: null,
  fitResult: null
};

const dom = {};
let toastTimer = 0;

document.addEventListener("DOMContentLoaded", async () => {
  const initialOpenPromise = invoke("open_initial");
  [
    "open-button", "welcome-open", "entry-list", "entry-count", "file-name", "file-kind", "file-path",
    "welcome", "viewer", "array-name", "dtype", "shape", "elements", "search-input", "find-button",
    "search-result", "copy-button", "table-head", "table-body", "page-size", "row-range", "prev-page",
    "page-number", "page-total", "next-page", "column-reference", "calc-expression", "calc-name",
    "degrees", "add-column", "clear-calc", "calc-status", "plot-kind", "plot-x", "plot-y",
    "plot-y-label", "bins-label", "hist-bins", "axis-scale", "axis-scale-label", "plot-grid",
    "plot-canvas", "plot-message", "stats-summary", "stats-body", "copy-stats",
    "correlation-note", "correlation-matrix", "fit-x", "fit-y", "fit-model", "run-fit",
    "fit-empty", "fit-results", "fit-metrics", "fit-equation", "copy-fit", "fit-canvas",
    "residual-canvas", "fit-coefficients", "toast", "busy"
  ].forEach((id) => { dom[id] = document.getElementById(id); });

  dom["open-button"].addEventListener("click", chooseFile);
  dom["welcome-open"].addEventListener("click", chooseFile);
  dom["find-button"].addEventListener("click", findNext);
  dom["search-input"].addEventListener("keydown", (event) => {
    if (event.key === "Enter") findNext();
  });
  dom["search-input"].addEventListener("input", () => {
    state.searchCursor = -1;
    state.match = null;
    dom["search-result"].textContent = "";
    renderTable();
  });
  dom["copy-button"].addEventListener("click", copyCurrentPage);
  dom["page-size"].addEventListener("change", () => {
    state.pageSize = Number(dom["page-size"].value);
    state.page = 1;
    renderTable();
  });
  dom["prev-page"].addEventListener("click", () => changePage(state.page - 1));
  dom["next-page"].addEventListener("click", () => changePage(state.page + 1));
  dom["page-number"].addEventListener("change", () => changePage(Number(dom["page-number"].value)));
  dom["add-column"].addEventListener("click", addCalculatedColumn);
  dom["clear-calc"].addEventListener("click", clearCalculatedColumns);
  dom["copy-stats"].addEventListener("click", copyStatistics);
  dom["run-fit"].addEventListener("click", runFit);
  dom["copy-fit"].addEventListener("click", copyFitResult);
  ["plot-kind", "plot-x", "plot-y", "hist-bins", "axis-scale", "plot-grid"].forEach((id) => {
    dom[id].addEventListener("change", () => {
      updatePlotControls();
      drawPlot();
    });
  });
  ["fit-x", "fit-y", "fit-model"].forEach((id) => {
    dom[id].addEventListener("change", () => { state.fitResult = null; showFitEmpty(); });
  });

  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => activatePanel(button.dataset.panel));
  });

  window.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "o") {
      event.preventDefault();
      chooseFile();
    }
  });


  new ResizeObserver(() => {
    if (document.getElementById("plot-panel").classList.contains("active")) drawPlot();
    if (document.getElementById("fit-panel").classList.contains("active") && state.fitResult) {
      drawFitResult();
    }
  }).observe(document.querySelector(".viewer"));

  try {
    const opened = await initialOpenPromise;
    if (opened) applyOpenedFile(opened);
  } catch (error) {
    showToast(String(error), true);
  }
  listen("tauri://drag-enter", () => document.body.classList.add("dragging")).catch(console.error);
  listen("tauri://drag-leave", () => document.body.classList.remove("dragging")).catch(console.error);
  listen("tauri://drag-drop", (event) => {
    document.body.classList.remove("dragging");
    const paths = event.payload && event.payload.paths;
    if (Array.isArray(paths) && paths.length > 0) openPath(paths[0]);
  }).catch(console.error);
});

async function chooseFile() {
  try {
    const opened = await invoke("pick_and_open");
    if (opened) applyOpenedFile(opened);
  } catch (error) {
    showToast(String(error), true);
  }
}

async function openPath(path) {
  setBusy(true);
  try {
    const opened = await invoke("open_file", { path: String(path) });
    applyOpenedFile(opened);
  } catch (error) {
    showToast(cleanError(error), true);
  } finally {
    setBusy(false);
  }
}

function applyOpenedFile(opened) {
  state.summary = opened.summary;
  state.array = opened.array;
  state.activeEntry = opened.selectedEntry;
  state.derived = [];
  state.page = 1;
  state.searchCursor = -1;
  state.match = null;
  state.statsCache = null;
  state.fitResult = null;
  dom["search-input"].value = "";
  renderSummary();
  renderArray();
}

async function loadEntry(entry) {
  if (!state.summary) return;
  setBusy(true);
  try {
    const array = await invoke("load_array", {
      path: state.summary.path,
      entry: entry
    });
    state.array = array;
    state.activeEntry = entry;
    state.derived = [];
    state.page = 1;
    state.searchCursor = -1;
    state.match = null;
    state.statsCache = null;
    state.fitResult = null;
    dom["search-input"].value = "";
    renderSummary();
    renderArray();
  } catch (error) {
    showToast(cleanError(error), true);
  } finally {
    setBusy(false);
  }
}

function renderSummary() {
  const summary = state.summary;
  if (!summary) return;
  dom["file-name"].textContent = summary.fileName;
  dom["file-kind"].textContent = summary.kind.toUpperCase();
  dom["file-kind"].classList.remove("hidden");
  dom["file-path"].textContent = summary.path;
  dom["entry-count"].textContent = String(summary.entries.length);
  dom["entry-list"].replaceChildren();

  summary.entries.forEach((entry) => {
    const button = document.createElement("button");
    button.className = "entry-button";
    if ((summary.kind === "npy" && state.array) || entry === state.activeEntry) {
      button.classList.add("active");
    }
    button.textContent = entry;
    button.title = entry;
    button.addEventListener("click", () => loadEntry(summary.kind === "npy" ? null : entry));
    dom["entry-list"].appendChild(button);
  });
}

function renderArray() {
  if (!state.array) return;
  dom["welcome"].classList.add("hidden");
  dom["viewer"].classList.remove("hidden");
  dom["array-name"].textContent = state.array.name;
  dom["dtype"].textContent = state.array.dtype;
  dom["shape"].textContent = state.array.shape.length ? "(" + state.array.shape.join(", ") + ")" : "scalar";
  dom["elements"].textContent = Number(state.array.totalElements).toLocaleString();
  renderTable();
  renderColumnReferences();
  populatePlotColumns();
  drawPlot();
  populateFitColumns();
  showFitEmpty();
}

function layout() {
  if (!state.array) return { rows: 0, baseCols: 0, sourceCols: 0, componentCount: 1 };
  const array = state.array;
  let sourceCols = 1;
  if (array.fieldNames.length > 0) {
    sourceCols = array.fieldNames.length;
  } else if (array.shape.length >= 2) {
    sourceCols = array.shape[array.shape.length - 1] || 1;
  }
  const componentCount = Math.max(1, array.components.length);
  const baseCols = sourceCols * componentCount;
  const rows = baseCols ? Math.ceil(array.values.length / baseCols) : 0;
  return { rows, baseCols, sourceCols, componentCount };
}

function labels() {
  if (!state.array) return [];
  const info = layout();
  const output = [];
  for (let source = 0; source < info.sourceCols; source += 1) {
    const base = state.array.fieldNames[source] || excelColumn(source);
    if (info.componentCount > 1) {
      state.array.components.forEach((component) => output.push(base + "." + component));
    } else {
      output.push(base);
    }
  }
  state.derived.forEach((column) => output.push(column.name));
  return output;
}

function valueAt(row, column) {
  const info = layout();
  if (column < info.baseCols) {
    return state.array.values[row * info.baseCols + column] ?? "";
  }
  const derived = state.derived[column - info.baseCols];
  return derived ? derived.values[row] ?? "" : "";
}

function visibleColumns() {
  const info = layout();
  const baseVisible = Math.min(info.baseCols, 500);
  const columns = Array.from({ length: baseVisible }, (_, index) => index);
  for (let index = 0; index < state.derived.length; index += 1) {
    columns.push(info.baseCols + index);
  }
  return columns;
}

function effectivePageSize() {
  const columnCount = Math.max(1, visibleColumns().length);
  return Math.max(1, Math.min(state.pageSize, Math.floor(50000 / columnCount)));
}

function renderTable() {
  if (!state.array) return;
  const info = layout();
  const columnLabels = labels();
  const columns = visibleColumns();
  const size = effectivePageSize();
  const pages = Math.max(1, Math.ceil(info.rows / size));
  state.page = Math.max(1, Math.min(state.page, pages));
  const start = (state.page - 1) * size;
  const end = Math.min(info.rows, start + size);
  const query = dom["search-input"].value.trim().toLocaleLowerCase();

  let header = "<tr><th class=\"row-index\">#</th>";
  columns.forEach((column) => {
    header += "<th title=\"" + escapeHtml(columnLabels[column]) + "\">$" + (column + 1) + " · " + escapeHtml(columnLabels[column]) + "</th>";
  });
  if (columns.length < columnLabels.length) {
    header += "<th>… " + (columnLabels.length - columns.length).toLocaleString() + " more columns</th>";
  }
  header += "</tr>";
  dom["table-head"].innerHTML = header;

  const rows = [];
  for (let row = start; row < end; row += 1) {
    let html = "<tr><td class=\"row-index\">" + row.toLocaleString() + "</td>";
    columns.forEach((column) => {
      const raw = String(valueAt(row, column));
      const isMatch = query && raw.toLocaleLowerCase().includes(query);
      const current = state.match && state.match.row === row && state.match.column === column;
      html += "<td class=\"" + (isMatch ? "match" : "") + "\" title=\"" + escapeHtml(raw) + "\"" +
        (current ? " data-current-match=\"true\"" : "") + ">" + escapeHtml(raw) + "</td>";
    });
    if (columns.length < columnLabels.length) html += "<td>…</td>";
    html += "</tr>";
    rows.push(html);
  }
  dom["table-body"].innerHTML = rows.join("");

  dom["row-range"].textContent = info.rows ? (start + 1).toLocaleString() + "–" + end.toLocaleString() + " / " + info.rows.toLocaleString() : "0–0 / 0";
  dom["page-number"].value = String(state.page);
  dom["page-number"].max = String(pages);
  dom["page-total"].textContent = String(pages);
  dom["prev-page"].disabled = state.page <= 1;
  dom["next-page"].disabled = state.page >= pages;

  requestAnimationFrame(() => {
    const match = document.querySelector("[data-current-match=\"true\"]");
    if (match) match.scrollIntoView({ block: "center", inline: "center" });
  });
}

function changePage(page) {
  const pages = Math.max(1, Math.ceil(layout().rows / effectivePageSize()));
  state.page = Math.max(1, Math.min(Number.isFinite(page) ? page : 1, pages));
  renderTable();
}

function findNext() {
  if (!state.array) return;
  const query = dom["search-input"].value.trim().toLocaleLowerCase();
  if (!query) {
    dom["search-result"].textContent = "検索語を入力してください";
    return;
  }
  const info = layout();
  const totalCols = labels().length;
  const totalCells = info.rows * totalCols;
  if (!totalCells) return;

  for (let step = 1; step <= totalCells; step += 1) {
    const flat = (state.searchCursor + step + totalCells) % totalCells;
    const row = Math.floor(flat / totalCols);
    const column = flat % totalCols;
    if (String(valueAt(row, column)).toLocaleLowerCase().includes(query)) {
      state.searchCursor = flat;
      state.match = { row, column };
      state.page = Math.floor(row / effectivePageSize()) + 1;
      dom["search-result"].textContent = "row " + row + ", $" + (column + 1);
      renderTable();
      return;
    }
  }
  state.match = null;
  dom["search-result"].textContent = "見つかりません";
  renderTable();
}

function renderColumnReferences() {
  const columnLabels = labels();
  dom["column-reference"].replaceChildren();
  columnLabels.slice(0, 64).forEach((name, index) => {
    const chip = document.createElement("span");
    chip.className = "column-chip";
    chip.textContent = "$" + (index + 1) + " = " + name;
    dom["column-reference"].appendChild(chip);
  });
  if (columnLabels.length > 64) {
    const chip = document.createElement("span");
    chip.className = "column-chip";
    chip.textContent = "… +" + (columnLabels.length - 64) + " columns";
    dom["column-reference"].appendChild(chip);
  }
}

function addCalculatedColumn() {
  if (!state.array) return;
  const expression = dom["calc-expression"].value.trim();
  if (!expression) return setCalcStatus("式を入力してください", true);
  try {
    const parser = new ExpressionParser(expression);
    const tree = parser.parse();
    const info = layout();
    const totalColumns = labels().length;
    parser.references.forEach((column) => {
      if (column < 0 || column >= totalColumns) {
        throw new Error("$" + (column + 1) + " は列の範囲外です");
      }
    });

    const values = [];
    let valid = 0;
    for (let row = 0; row < info.rows; row += 1) {
      const result = evaluate(tree, row, dom["degrees"].checked);
      if (Number.isFinite(result)) {
        values.push(formatNumber(result));
        valid += 1;
      } else {
        values.push("");
      }
    }
    if (!valid) throw new Error("数値を計算できる行がありません");

    const name = dom["calc-name"].value.trim() || "Calc" + (state.derived.length + 1);
    state.derived.push({ name, values });
    dom["calc-name"].value = "";
    state.page = 1;
    state.statsCache = null;
    state.fitResult = null;
    setCalcStatus("'" + name + "' を追加しました", false);
    renderTable();
    renderColumnReferences();
    populatePlotColumns();
    populateFitColumns();
    showFitEmpty();
  } catch (error) {
    setCalcStatus(error.message || String(error), true);
  }
}

function clearCalculatedColumns() {
  state.derived = [];
  setCalcStatus("計算列をクリアしました", false);
  renderTable();
  renderColumnReferences();
  state.statsCache = null;
  state.fitResult = null;
  populatePlotColumns();
  populateFitColumns();
  showFitEmpty();
}

function setCalcStatus(message, error) {
  dom["calc-status"].textContent = message;
  dom["calc-status"].classList.toggle("error", error);
}

class ExpressionParser {
  constructor(input) {
    this.input = input;
    this.pos = 0;
    this.references = new Set();
  }
  parse() {
    const node = this.parseAddSub();
    this.skip();
    if (this.pos !== this.input.length) throw this.error("解釈できない文字があります");
    return node;
  }
  parseAddSub() {
    let node = this.parseMulDiv();
    while (true) {
      this.skip();
      if (this.take("+")) node = { type: "bin", op: "+", left: node, right: this.parseMulDiv() };
      else if (this.take("-")) node = { type: "bin", op: "-", left: node, right: this.parseMulDiv() };
      else return node;
    }
  }
  parseMulDiv() {
    let node = this.parsePower();
    while (true) {
      this.skip();
      if (this.take("*")) node = { type: "bin", op: "*", left: node, right: this.parsePower() };
      else if (this.take("/")) node = { type: "bin", op: "/", left: node, right: this.parsePower() };
      else return node;
    }
  }
  parsePower() {
    let node = this.parseUnary();
    this.skip();
    if (this.take("^")) node = { type: "bin", op: "^", left: node, right: this.parsePower() };
    return node;
  }
  parseUnary() {
    this.skip();
    if (this.take("+")) return this.parseUnary();
    if (this.take("-")) return { type: "neg", value: this.parseUnary() };
    return this.parsePrimary();
  }
  parsePrimary() {
    this.skip();
    if (this.take("(")) {
      const node = this.parseAddSub();
      this.skip();
      if (!this.take(")")) throw this.error("')' が必要です");
      return node;
    }
    if (this.take("$")) {
      const match = this.input.slice(this.pos).match(/^\d+/);
      if (!match || Number(match[0]) < 1) throw this.error("列は $1 の形式で指定します");
      this.pos += match[0].length;
      const column = Number(match[0]) - 1;
      this.references.add(column);
      return { type: "column", column };
    }
    const number = this.input.slice(this.pos).match(/^(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?/i);
    if (number) {
      this.pos += number[0].length;
      return { type: "number", value: Number(number[0]) };
    }
    const identifier = this.input.slice(this.pos).match(/^[a-z_][a-z0-9_]*/i);
    if (identifier) {
      this.pos += identifier[0].length;
      return this.parseFunction(identifier[0].toLowerCase());
    }
    throw this.error("値、列、または関数が必要です");
  }
  parseFunction(name) {
    const arity = { sin: 1, cos: 1, tan: 1, asin: 1, acos: 1, atan: 1, abs: 1, sqrt: 1, exp: 1, pow: 2, atan2: 2 };
    if (!(name in arity)) throw this.error("未知の関数: " + name);
    this.skip();
    if (!this.take("(")) throw this.error("関数の後に '(' が必要です");
    const args = [this.parseAddSub()];
    if (arity[name] === 2) {
      this.skip();
      if (!this.take(",")) throw this.error("引数の区切り ',' が必要です");
      args.push(this.parseAddSub());
    }
    this.skip();
    if (!this.take(")")) throw this.error("')' が必要です");
    return { type: "func", name, args };
  }
  skip() {
    while (/\s/.test(this.input[this.pos] || "")) this.pos += 1;
  }
  take(character) {
    if (this.input[this.pos] === character) {
      this.pos += 1;
      return true;
    }
    return false;
  }
  error(message) {
    return new Error(message + " (位置 " + (this.pos + 1) + ")");
  }
}

function evaluate(node, row, degrees) {
  if (node.type === "number") return node.value;
  if (node.type === "column") return numericValue(valueAt(row, node.column));
  if (node.type === "neg") return -evaluate(node.value, row, degrees);
  if (node.type === "bin") {
    const left = evaluate(node.left, row, degrees);
    const right = evaluate(node.right, row, degrees);
    if (node.op === "+") return left + right;
    if (node.op === "-") return left - right;
    if (node.op === "*") return left * right;
    if (node.op === "/") return right === 0 ? NaN : left / right;
    return Math.pow(left, right);
  }
  const args = node.args.map((arg) => evaluate(arg, row, degrees));
  const toRad = (value) => degrees ? value * Math.PI / 180 : value;
  const fromRad = (value) => degrees ? value * 180 / Math.PI : value;
  if (node.name === "sin") return Math.sin(toRad(args[0]));
  if (node.name === "cos") return Math.cos(toRad(args[0]));
  if (node.name === "tan") return Math.tan(toRad(args[0]));
  if (node.name === "asin") return fromRad(Math.asin(args[0]));
  if (node.name === "acos") return fromRad(Math.acos(args[0]));
  if (node.name === "atan") return fromRad(Math.atan(args[0]));
  if (node.name === "atan2") return fromRad(Math.atan2(args[0], args[1]));
  if (node.name === "abs") return Math.abs(args[0]);
  if (node.name === "sqrt") return Math.sqrt(args[0]);
  if (node.name === "exp") return Math.exp(args[0]);
  if (node.name === "pow") return Math.pow(args[0], args[1]);
  return NaN;
}

function numericValue(value) {
  if (typeof value === "number") return Number.isFinite(value) ? value : NaN;
  const text = String(value).trim();
  if (!text) return NaN;
  const number = Number(text);
  return Number.isFinite(number) ? number : NaN;
}

function populatePlotColumns() {
  const columnLabels = labels();
  const oldX = dom["plot-x"].value;
  const oldY = dom["plot-y"].value;
  dom["plot-x"].replaceChildren();
  const indexOption = document.createElement("option");
  indexOption.value = "-1";
  indexOption.textContent = "Row index";
  dom["plot-x"].appendChild(indexOption);
  columnLabels.forEach((name, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = "$" + (index + 1) + " · " + name;
    dom["plot-x"].appendChild(option);
  });
  dom["plot-y"].replaceChildren();
  columnLabels.forEach((name, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = "$" + (index + 1) + " · " + name;
    dom["plot-y"].appendChild(option);
  });
  dom["plot-x"].value = Array.from(dom["plot-x"].options).some((item) => item.value === oldX) ? oldX : "-1";
  dom["plot-y"].value = Array.from(dom["plot-y"].options).some((item) => item.value === oldY) ? oldY : "0";
  updatePlotControls();
  drawPlot();
}

function updatePlotControls() {
  const histogram = dom["plot-kind"].value === "hist";
  dom["plot-y-label"].classList.toggle("hidden", histogram);
  dom["bins-label"].classList.toggle("hidden", !histogram);
  const indexOption = dom["plot-x"].querySelector("option[value=\"-1\"]");
  if (indexOption) indexOption.disabled = histogram;
  if (histogram && dom["plot-x"].value === "-1") dom["plot-x"].value = "0";
}

function drawPlot() {
  if (!state.array || !dom["plot-canvas"]) return;
  const canvas = dom["plot-canvas"];
  const rect = canvas.getBoundingClientRect();
  if (rect.width < 10 || rect.height < 10) return;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.floor(rect.width * ratio);
  canvas.height = Math.floor(rect.height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  ctx.clearRect(0, 0, rect.width, rect.height);

  const kind = dom["plot-kind"].value;
  const xColumn = Number(dom["plot-x"].value);
  const yColumn = Number(dom["plot-y"].value);
  const info = layout();
  const stride = Math.max(1, Math.ceil(info.rows / 100000));

  if (kind === "hist") {
    const values = [];
    for (let row = 0; row < info.rows; row += stride) {
      const value = numericValue(valueAt(row, xColumn));
      if (Number.isFinite(value)) values.push(value);
    }
    drawHistogram(ctx, rect.width, rect.height, values, Number(dom["hist-bins"].value));
    return;
  }

  const points = [];
  for (let row = 0; row < info.rows; row += stride) {
    const x = xColumn === -1 ? row : numericValue(valueAt(row, xColumn));
    const y = numericValue(valueAt(row, yColumn));
    if (Number.isFinite(x) && Number.isFinite(y)) points.push([x, y]);
  }
  drawXY(ctx, rect.width, rect.height, points, kind);
}

function plotFrame(ctx, width, height) {
  const frame = { left: 62, top: 25, right: width - 22, bottom: height - 42 };
  ctx.fillStyle = "#11161e";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#394658";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(frame.left, frame.top);
  ctx.lineTo(frame.left, frame.bottom);
  ctx.lineTo(frame.right, frame.bottom);
  ctx.stroke();
  return frame;
}

function drawXY(ctx, width, height, points, kind) {
  const frame = plotFrame(ctx, width, height);
  if (!points.length) return showPlotMessage("数値としてプロットできるデータがありません");
  hidePlotMessage();
  let xMin = points[0][0], xMax = points[0][0], yMin = points[0][1], yMax = points[0][1];
  points.forEach(([x, y]) => {
    xMin = Math.min(xMin, x); xMax = Math.max(xMax, x);
    yMin = Math.min(yMin, y); yMax = Math.max(yMax, y);
  });
  [xMin, xMax] = normalizeRange(xMin, xMax);
  [yMin, yMax] = normalizeRange(yMin, yMax);
  const map = ([x, y]) => [
    frame.left + (x - xMin) / (xMax - xMin) * (frame.right - frame.left),
    frame.bottom - (y - yMin) / (yMax - yMin) * (frame.bottom - frame.top)
  ];

  if (kind === "line") {
    ctx.strokeStyle = "#67a5ff";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    points.forEach((point, index) => {
      const [x, y] = map(point);
      if (index) ctx.lineTo(x, y); else ctx.moveTo(x, y);
    });
    ctx.stroke();
  } else {
    ctx.fillStyle = "#5dd7b7";
    points.forEach((point) => {
      const [x, y] = map(point);
      ctx.beginPath();
      ctx.arc(x, y, 2.3, 0, Math.PI * 2);
      ctx.fill();
    });
  }
  drawAxisLabels(ctx, frame, xMin, xMax, yMin, yMax);
}

function drawHistogram(ctx, width, height, values, requestedBins) {
  const frame = plotFrame(ctx, width, height);
  if (!values.length) return showPlotMessage("数値として集計できるデータがありません");
  hidePlotMessage();
  let min = Math.min(...values), max = Math.max(...values);
  [min, max] = normalizeRange(min, max);
  const bins = Math.max(2, Math.min(200, requestedBins || 30));
  const counts = Array(bins).fill(0);
  const binWidth = (max - min) / bins;
  values.forEach((value) => {
    const index = Math.max(0, Math.min(bins - 1, Math.floor((value - min) / binWidth)));
    counts[index] += 1;
  });
  const peak = Math.max(...counts, 1);
  const widthPerBin = (frame.right - frame.left) / bins;
  ctx.fillStyle = "#67a5ff";
  counts.forEach((count, index) => {
    const height = count / peak * (frame.bottom - frame.top);
    ctx.fillRect(frame.left + index * widthPerBin + 1, frame.bottom - height, Math.max(1, widthPerBin - 2), height);
  });
  drawAxisLabels(ctx, frame, min, max, 0, peak);
}

function drawAxisLabels(ctx, frame, xMin, xMax, yMin, yMax) {
  ctx.fillStyle = "#7e8a9b";
  ctx.font = "11px ui-monospace, monospace";
  ctx.textBaseline = "top";
  ctx.textAlign = "left";
  ctx.fillText(formatNumber(xMin), frame.left, frame.bottom + 8);
  ctx.textAlign = "right";
  ctx.fillText(formatNumber(xMax), frame.right, frame.bottom + 8);
  ctx.textBaseline = "middle";
  ctx.fillText(formatNumber(yMax), frame.left - 8, frame.top);
  ctx.fillText(formatNumber(yMin), frame.left - 8, frame.bottom);
}

function normalizeRange(min, max) {
  if (min === max) {
    const pad = min === 0 ? 1 : Math.abs(min) * 0.05;
    return [min - pad, max + pad];
  }
  return [min, max];
}

function showPlotMessage(message) {
  dom["plot-message"].textContent = message;
  dom["plot-message"].classList.add("visible");
}
function hidePlotMessage() {
  dom["plot-message"].classList.remove("visible");
}

async function copyCurrentPage() {
  if (!state.array) return;
  const columnLabels = labels();
  const columns = visibleColumns();
  const info = layout();
  const size = effectivePageSize();
  const start = (state.page - 1) * size;
  const end = Math.min(info.rows, start + size);
  const lines = [columns.map((column) => csvEscape(columnLabels[column])).join(",")];
  for (let row = start; row < end; row += 1) {
    lines.push(columns.map((column) => csvEscape(valueAt(row, column))).join(","));
  }
  const text = lines.join("\n");
  try {
    await navigator.clipboard.writeText(text);
  } catch (_) {
    const area = document.createElement("textarea");
    area.value = text;
    document.body.appendChild(area);
    area.select();
    document.execCommand("copy");
    area.remove();
  }
  showToast((end - start).toLocaleString() + " 行を CSV としてコピーしました");
}

function csvEscape(value) {
  const text = String(value ?? "");
  return /[",\r\n]/.test(text) ? "\"" + text.replaceAll("\"", "\"\"") + "\"" : text;
}

function activatePanel(id) {
  document.querySelectorAll(".tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.panel === id));
  document.querySelectorAll(".panel").forEach((panel) => panel.classList.toggle("active", panel.id === id));
  if (id === "plot-panel") requestAnimationFrame(drawPlot);
  if (id === "stats-panel") requestAnimationFrame(renderStatistics);
  if (id === "fit-panel" && state.fitResult) requestAnimationFrame(drawFitResult);
}

function formatNumber(value) {
  if (!Number.isFinite(value)) return "";
  if (value === 0) return "0";
  const absolute = Math.abs(value);
  if (absolute >= 1e7 || absolute < 1e-5) return value.toExponential(7).replace(/\.?0+e/, "e");
  return Number(value.toPrecision(10)).toString();
}

function excelColumn(index) {
  let value = index + 1;
  let name = "";
  while (value > 0) {
    value -= 1;
    name = String.fromCharCode(65 + value % 26) + name;
    value = Math.floor(value / 26);
  }
  return name;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll("\"", "&quot;")
    .replaceAll("'", "&#039;");
}

function cleanError(error) {
  const text = typeof error === "string" ? error : (error && error.message) || String(error);
  return text.replace(/^Error:\s*/, "");
}

function setBusy(active) {
  dom["busy"].classList.toggle("hidden", !active);
}

function showToast(message, error = false) {
  clearTimeout(toastTimer);
  dom["toast"].textContent = message;
  dom["toast"].classList.toggle("error", error);
  dom["toast"].classList.add("show");
  toastTimer = setTimeout(() => dom["toast"].classList.remove("show"), 3200);
}


function numericColumnValues(column) {
  const rows = layout().rows;
  const values = [];
  for (let row = 0; row < rows; row += 1) {
    const value = numericValue(valueAt(row, column));
    if (Number.isFinite(value)) values.push(value);
  }
  return values;
}

function quantile(sorted, probability) {
  if (!sorted.length) return NaN;
  const position = (sorted.length - 1) * probability;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

function calculateColumnStatistics(column, name) {
  const rows = layout().rows;
  const values = numericColumnValues(column);
  if (!values.length) return null;
  let sum = 0;
  let sumSquares = 0;
  values.forEach((value) => {
    sum += value;
    sumSquares += value * value;
  });
  const mean = sum / values.length;
  let m2 = 0, m3 = 0, m4 = 0;
  values.forEach((value) => {
    const delta = value - mean;
    const square = delta * delta;
    m2 += square;
    m3 += square * delta;
    m4 += square * square;
  });
  const variance = values.length > 1 ? m2 / (values.length - 1) : 0;
  const populationVariance = m2 / values.length;
  const skew = populationVariance > 0 ? (m3 / values.length) / Math.pow(populationVariance, 1.5) : 0;
  const kurtosis = populationVariance > 0 ? (m4 / values.length) / (populationVariance * populationVariance) - 3 : 0;
  values.sort((a, b) => a - b);
  return {
    column, name, count: values.length, missing: rows - values.length,
    mean, std: Math.sqrt(variance), min: values[0], q1: quantile(values, 0.25),
    median: quantile(values, 0.5), q3: quantile(values, 0.75),
    max: values[values.length - 1], rms: Math.sqrt(sumSquares / values.length),
    skew, kurtosis
  };
}

function getStatistics() {
  if (state.statsCache) return state.statsCache;
  const columnLabels = labels();
  const limit = Math.min(columnLabels.length, 512);
  const stats = [];
  for (let column = 0; column < limit; column += 1) {
    const result = calculateColumnStatistics(column, columnLabels[column]);
    if (result) stats.push(result);
  }
  state.statsCache = { stats, examined: limit, totalColumns: columnLabels.length };
  return state.statsCache;
}

function renderStatistics() {
  if (!state.array) return;
  const result = getStatistics();
  const rows = layout().rows;
  dom["stats-summary"].innerHTML = [
    metricCard("Rows", rows.toLocaleString()),
    metricCard("Columns", labels().length.toLocaleString()),
    metricCard("Numeric columns", result.stats.length.toLocaleString()),
    metricCard("Elements", Number(state.array.totalElements).toLocaleString())
  ].join("");

  dom["stats-body"].innerHTML = result.stats.map((stat) => {
    const cells = [
      "$" + (stat.column + 1) + " · " + stat.name,
      stat.count.toLocaleString(), stat.missing.toLocaleString(),
      formatNumber(stat.mean), formatNumber(stat.std), formatNumber(stat.min),
      formatNumber(stat.q1), formatNumber(stat.median), formatNumber(stat.q3),
      formatNumber(stat.max), formatNumber(stat.rms), formatNumber(stat.skew),
      formatNumber(stat.kurtosis)
    ];
    return "<tr>" + cells.map((cell) => "<td title=\"" + escapeHtml(cell) + "\">" + escapeHtml(cell) + "</td>").join("") + "</tr>";
  }).join("");

  renderCorrelation(result.stats.slice(0, 12));
}

function metricCard(label, value) {
  return "<div class=\"metric-card\"><span>" + escapeHtml(label) + "</span><strong title=\"" +
    escapeHtml(value) + "\">" + escapeHtml(value) + "</strong></div>";
}

function renderCorrelation(stats) {
  if (stats.length < 2) {
    dom["correlation-note"].textContent = "数値列が2列以上必要です";
    dom["correlation-matrix"].innerHTML = "";
    return;
  }
  dom["correlation-note"].textContent = stats.length === 12 ? "先頭12数値列" : stats.length + " 数値列";
  const rows = layout().rows;
  let html = "<table class=\"correlation-table\"><thead><tr><th></th>";
  html += stats.map((stat) => "<th>$" + (stat.column + 1) + "</th>").join("") + "</tr></thead><tbody>";
  stats.forEach((left) => {
    html += "<tr><th>$" + (left.column + 1) + " · " + escapeHtml(left.name) + "</th>";
    stats.forEach((right) => {
      const value = left.column === right.column ? 1 : pearsonCorrelation(left.column, right.column, rows);
      const magnitude = Number.isFinite(value) ? Math.abs(value) : 0;
      const className = value < 0 ? "correlation-cell negative" : "correlation-cell";
      const text = Number.isFinite(value) ? value.toFixed(3) : "—";
      html += "<td class=\"" + className + "\" style=\"--corr:" + magnitude.toFixed(3) + "\" title=\"" +
        escapeHtml(left.name + " × " + right.name) + "\">" + text + "</td>";
    });
    html += "</tr>";
  });
  dom["correlation-matrix"].innerHTML = html + "</tbody></table>";
}

function pearsonCorrelation(leftColumn, rightColumn, rows) {
  let count = 0, sumX = 0, sumY = 0, sumXX = 0, sumYY = 0, sumXY = 0;
  const stride = Math.max(1, Math.ceil(rows / 50000));
  for (let row = 0; row < rows; row += stride) {
    const x = numericValue(valueAt(row, leftColumn));
    const y = numericValue(valueAt(row, rightColumn));
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    count += 1;
    sumX += x; sumY += y;
    sumXX += x * x; sumYY += y * y; sumXY += x * y;
  }
  if (count < 2) return NaN;
  const numerator = count * sumXY - sumX * sumY;
  const denominator = Math.sqrt((count * sumXX - sumX * sumX) * (count * sumYY - sumY * sumY));
  return denominator > 0 ? numerator / denominator : NaN;
}

async function copyStatistics() {
  const stats = getStatistics().stats;
  if (!stats.length) return showToast("数値列がありません", true);
  const headings = ["column", "n", "missing", "mean", "std", "min", "q1", "median", "q3", "max", "rms", "skew", "kurtosis"];
  const lines = [headings.join(",")];
  stats.forEach((stat) => {
    lines.push([
      stat.name, stat.count, stat.missing, stat.mean, stat.std, stat.min, stat.q1,
      stat.median, stat.q3, stat.max, stat.rms, stat.skew, stat.kurtosis
    ].map(csvEscape).join(","));
  });
  await writeClipboard(lines.join("\n"));
  showToast(stats.length + " 列の統計量をコピーしました");
}

async function writeClipboard(text) {
  try {
    await navigator.clipboard.writeText(text);
  } catch (_) {
    const area = document.createElement("textarea");
    area.value = text;
    document.body.appendChild(area);
    area.select();
    document.execCommand("copy");
    area.remove();
  }
}

function populateFitColumns() {
  if (!dom["fit-x"] || !state.array) return;
  const columnLabels = labels();
  const oldX = dom["fit-x"].value;
  const oldY = dom["fit-y"].value;
  dom["fit-x"].replaceChildren();
  const rowOption = document.createElement("option");
  rowOption.value = "-1";
  rowOption.textContent = "Row index";
  dom["fit-x"].appendChild(rowOption);
  columnLabels.forEach((name, index) => {
    const xOption = document.createElement("option");
    xOption.value = String(index);
    xOption.textContent = "$" + (index + 1) + " · " + name;
    dom["fit-x"].appendChild(xOption);
  });
  dom["fit-y"].replaceChildren();
  columnLabels.forEach((name, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = "$" + (index + 1) + " · " + name;
    dom["fit-y"].appendChild(option);
  });
  dom["fit-x"].value = Array.from(dom["fit-x"].options).some((option) => option.value === oldX) ? oldX : "-1";
  dom["fit-y"].value = Array.from(dom["fit-y"].options).some((option) => option.value === oldY) ? oldY : "0";
}

function showFitEmpty() {
  if (!dom["fit-empty"]) return;
  dom["fit-empty"].classList.remove("hidden");
  dom["fit-results"].classList.add("hidden");
}

function collectFitPoints(xColumn, yColumn, model) {
  const rows = layout().rows;
  const points = [];
  for (let row = 0; row < rows; row += 1) {
    const x = xColumn === -1 ? row : numericValue(valueAt(row, xColumn));
    const y = numericValue(valueAt(row, yColumn));
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    if ((model === "log" || model === "power") && x <= 0) continue;
    if ((model === "exp" || model === "power") && y <= 0) continue;
    points.push([x, y]);
  }
  return points;
}

function runFit() {
  if (!state.array) return;
  try {
    const xColumn = Number(dom["fit-x"].value);
    const yColumn = Number(dom["fit-y"].value);
    const model = dom["fit-model"].value;
    const points = collectFitPoints(xColumn, yColumn, model);
    const degree = model.startsWith("poly") ? Number(model.slice(4)) : 1;
    const parameterCount = model.startsWith("poly") ? degree + 1 : 2;
    if (points.length <= parameterCount) throw new Error("fitting に必要な有効点が足りません");

    let fit;
    if (model.startsWith("poly")) fit = polynomialFit(points, degree);
    else fit = transformedFit(points, model);

    const predictions = points.map(([x]) => fit.predict(x));
    const meanY = points.reduce((sum, point) => sum + point[1], 0) / points.length;
    let sse = 0, sst = 0, absoluteError = 0;
    const residuals = [];
    points.forEach((point, index) => {
      const residual = point[1] - predictions[index];
      residuals.push([point[0], residual]);
      sse += residual * residual;
      sst += (point[1] - meanY) * (point[1] - meanY);
      absoluteError += Math.abs(residual);
    });
    const r2 = sst > 0 ? 1 - sse / sst : (sse === 0 ? 1 : NaN);
    const adjustedR2 = points.length > parameterCount && Number.isFinite(r2)
      ? 1 - (1 - r2) * (points.length - 1) / (points.length - parameterCount)
      : NaN;
    state.fitResult = {
      ...fit, model, points, residuals, n: points.length, r2, adjustedR2,
      rmse: Math.sqrt(sse / points.length),
      mae: absoluteError / points.length,
      residualStd: points.length > parameterCount ? Math.sqrt(sse / (points.length - parameterCount)) : NaN,
      xName: xColumn === -1 ? "Row index" : labels()[xColumn],
      yName: labels()[yColumn]
    };
    renderFitMetrics();
    dom["fit-empty"].classList.add("hidden");
    dom["fit-results"].classList.remove("hidden");
    requestAnimationFrame(drawFitResult);
  } catch (error) {
    state.fitResult = null;
    showFitEmpty();
    showToast(error.message || String(error), true);
  }
}

function polynomialFit(points, degree) {
  const meanX = points.reduce((sum, point) => sum + point[0], 0) / points.length;
  const varianceX = points.reduce((sum, point) => sum + Math.pow(point[0] - meanX, 2), 0) / points.length;
  const scaleX = Math.sqrt(varianceX) || 1;
  const size = degree + 1;
  const matrix = Array.from({ length: size }, () => Array(size).fill(0));
  const vector = Array(size).fill(0);
  points.forEach(([x, y]) => {
    const z = (x - meanX) / scaleX;
    const powers = Array(2 * degree + 1).fill(1);
    for (let index = 1; index < powers.length; index += 1) powers[index] = powers[index - 1] * z;
    for (let row = 0; row < size; row += 1) {
      vector[row] += y * powers[row];
      for (let column = 0; column < size; column += 1) matrix[row][column] += powers[row + column];
    }
  });
  const scaled = solveLinearSystem(matrix, vector);
  const coefficients = Array(size).fill(0);
  for (let power = 0; power <= degree; power += 1) {
    const factor = scaled[power] / Math.pow(scaleX, power);
    for (let rawPower = 0; rawPower <= power; rawPower += 1) {
      coefficients[rawPower] += factor * binomial(power, rawPower) * Math.pow(-meanX, power - rawPower);
    }
  }
  const predict = (x) => horner(coefficients, x);
  const terms = coefficients.map((coefficient, index) => {
    if (index === 0) return formatNumber(coefficient);
    return (coefficient >= 0 ? "+ " : "− ") + formatNumber(Math.abs(coefficient)) + (index === 1 ? " x" : " x^" + index);
  });
  return {
    coefficients,
    coefficientNames: coefficients.map((_, index) => "a" + index),
    equation: "y = " + terms.join(" "),
    predict
  };
}

function transformedFit(points, model) {
  const transformed = points.map(([x, y]) => {
    if (model === "exp") return [x, Math.log(y)];
    if (model === "log") return [Math.log(x), y];
    return [Math.log(x), Math.log(y)];
  });
  const line = linearRegression(transformed);
  if (model === "exp") {
    const a = Math.exp(line.intercept), b = line.slope;
    return {
      coefficients: [a, b], coefficientNames: ["a", "b"],
      equation: "y = " + formatNumber(a) + " exp(" + formatNumber(b) + " x)",
      predict: (x) => a * Math.exp(b * x)
    };
  }
  if (model === "log") {
    const a = line.intercept, b = line.slope;
    return {
      coefficients: [a, b], coefficientNames: ["a", "b"],
      equation: "y = " + formatNumber(a) + (b >= 0 ? " + " : " − ") + formatNumber(Math.abs(b)) + " ln(x)",
      predict: (x) => a + b * Math.log(x)
    };
  }
  const a = Math.exp(line.intercept), b = line.slope;
  return {
    coefficients: [a, b], coefficientNames: ["a", "b"],
    equation: "y = " + formatNumber(a) + " x^" + formatNumber(b),
    predict: (x) => a * Math.pow(x, b)
  };
}

function linearRegression(points) {
  const n = points.length;
  let sumX = 0, sumY = 0, sumXX = 0, sumXY = 0;
  points.forEach(([x, y]) => {
    sumX += x; sumY += y; sumXX += x * x; sumXY += x * y;
  });
  const denominator = n * sumXX - sumX * sumX;
  if (Math.abs(denominator) < Number.EPSILON) throw new Error("X の変化がなく fitting できません");
  const slope = (n * sumXY - sumX * sumY) / denominator;
  return { slope, intercept: (sumY - slope * sumX) / n };
}

function solveLinearSystem(matrix, vector) {
  const size = vector.length;
  const augmented = matrix.map((row, index) => row.slice().concat(vector[index]));
  for (let pivot = 0; pivot < size; pivot += 1) {
    let best = pivot;
    for (let row = pivot + 1; row < size; row += 1) {
      if (Math.abs(augmented[row][pivot]) > Math.abs(augmented[best][pivot])) best = row;
    }
    if (Math.abs(augmented[best][pivot]) < 1e-14) throw new Error("行列が特異で fitting できません");
    [augmented[pivot], augmented[best]] = [augmented[best], augmented[pivot]];
    const divisor = augmented[pivot][pivot];
    for (let column = pivot; column <= size; column += 1) augmented[pivot][column] /= divisor;
    for (let row = 0; row < size; row += 1) {
      if (row === pivot) continue;
      const factor = augmented[row][pivot];
      for (let column = pivot; column <= size; column += 1) {
        augmented[row][column] -= factor * augmented[pivot][column];
      }
    }
  }
  return augmented.map((row) => row[size]);
}

function binomial(n, k) {
  let result = 1;
  for (let index = 1; index <= k; index += 1) result = result * (n - index + 1) / index;
  return result;
}

function horner(coefficients, x) {
  let result = 0;
  for (let index = coefficients.length - 1; index >= 0; index -= 1) result = result * x + coefficients[index];
  return result;
}

function renderFitMetrics() {
  const fit = state.fitResult;
  dom["fit-metrics"].innerHTML = [
    metricCard("Valid points", fit.n.toLocaleString()),
    metricCard("R²", formatNumber(fit.r2)),
    metricCard("Adjusted R²", formatNumber(fit.adjustedR2)),
    metricCard("RMSE", formatNumber(fit.rmse)),
    metricCard("MAE", formatNumber(fit.mae)),
    metricCard("Residual std", formatNumber(fit.residualStd))
  ].join("");
  dom["fit-equation"].textContent = fit.equation;
  dom["fit-coefficients"].innerHTML = fit.coefficients.map((coefficient, index) =>
    "<div class=\"coefficient-row\"><span>" + escapeHtml(fit.coefficientNames[index]) +
    "</span><strong>" + escapeHtml(formatNumber(coefficient)) + "</strong></div>"
  ).join("");
}

async function copyFitResult() {
  const fit = state.fitResult;
  if (!fit) return;
  const text = [
    "model\t" + fit.equation,
    "x\t" + fit.xName,
    "y\t" + fit.yName,
    "n\t" + fit.n,
    "r2\t" + fit.r2,
    "adjusted_r2\t" + fit.adjustedR2,
    "rmse\t" + fit.rmse,
    "mae\t" + fit.mae,
    "residual_std\t" + fit.residualStd
  ].join("\n");
  await writeClipboard(text);
  showToast("fitting 結果をコピーしました");
}

function prepareCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  if (rect.width < 10 || rect.height < 10) return null;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.floor(rect.width * ratio);
  canvas.height = Math.floor(rect.height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, width: rect.width, height: rect.height };
}

function drawFitResult() {
  const fit = state.fitResult;
  if (!fit) return;
  const main = prepareCanvas(dom["fit-canvas"]);
  const residual = prepareCanvas(dom["residual-canvas"]);
  if (!main || !residual) return;

  const sorted = fit.points.slice().sort((a, b) => a[0] - b[0]);
  const minX = sorted[0][0], maxX = sorted[sorted.length - 1][0];
  const curve = [];
  for (let index = 0; index < 300; index += 1) {
    const x = minX + (maxX - minX) * index / 299;
    const y = fit.predict(x);
    if (Number.isFinite(y)) curve.push([x, y]);
  }
  drawDataAndCurve(main.ctx, main.width, main.height, fit.points, curve);
  drawResiduals(residual.ctx, residual.width, residual.height, fit.residuals);
}

function drawDataAndCurve(ctx, width, height, points, curve) {
  const all = points.concat(curve);
  const bounds = pointBounds(all);
  const frame = drawFrame(ctx, width, height, true);
  const map = pointMapper(frame, bounds);
  const stride = Math.max(1, Math.ceil(points.length / 50000));
  ctx.fillStyle = "#5dd7b7";
  for (let index = 0; index < points.length; index += stride) {
    const [x, y] = map(points[index]);
    ctx.fillRect(x - 1.5, y - 1.5, 3, 3);
  }
  ctx.strokeStyle = "#ffb85c";
  ctx.lineWidth = 2;
  ctx.beginPath();
  curve.forEach((point, index) => {
    const [x, y] = map(point);
    if (index) ctx.lineTo(x, y); else ctx.moveTo(x, y);
  });
  ctx.stroke();
  drawAxisLabelsRich(ctx, frame, bounds.xMin, bounds.xMax, bounds.yMin, bounds.yMax);
}

function drawResiduals(ctx, width, height, points) {
  const bounds = pointBounds(points.concat([[points[0][0], 0]]));
  const frame = drawFrame(ctx, width, height, true);
  const map = pointMapper(frame, bounds);
  const zeroStart = map([bounds.xMin, 0]);
  const zeroEnd = map([bounds.xMax, 0]);
  ctx.strokeStyle = "#8995a7";
  ctx.beginPath();
  ctx.moveTo(zeroStart[0], zeroStart[1]);
  ctx.lineTo(zeroEnd[0], zeroEnd[1]);
  ctx.stroke();
  ctx.fillStyle = "#67a5ff";
  const stride = Math.max(1, Math.ceil(points.length / 50000));
  for (let index = 0; index < points.length; index += stride) {
    const [x, y] = map(points[index]);
    ctx.fillRect(x - 1.5, y - 1.5, 3, 3);
  }
  drawAxisLabelsRich(ctx, frame, bounds.xMin, bounds.xMax, bounds.yMin, bounds.yMax);
}

function pointBounds(points) {
  let xMin = points[0][0], xMax = points[0][0], yMin = points[0][1], yMax = points[0][1];
  points.forEach(([x, y]) => {
    xMin = Math.min(xMin, x); xMax = Math.max(xMax, x);
    yMin = Math.min(yMin, y); yMax = Math.max(yMax, y);
  });
  [xMin, xMax] = normalizeRange(xMin, xMax);
  [yMin, yMax] = normalizeRange(yMin, yMax);
  return { xMin, xMax, yMin, yMax };
}

function pointMapper(frame, bounds) {
  return ([x, y]) => [
    frame.left + (x - bounds.xMin) / (bounds.xMax - bounds.xMin) * (frame.right - frame.left),
    frame.bottom - (y - bounds.yMin) / (bounds.yMax - bounds.yMin) * (frame.bottom - frame.top)
  ];
}

function updatePlotControlsRich() {
  const kind = dom["plot-kind"].value;
  const singleColumn = kind === "hist" || kind === "box";
  dom["plot-y-label"].classList.toggle("hidden", singleColumn);
  dom["bins-label"].classList.toggle("hidden", kind !== "hist");
  dom["axis-scale-label"].classList.toggle("hidden", singleColumn);
  const indexOption = dom["plot-x"].querySelector("option[value=\"-1\"]");
  if (indexOption) indexOption.disabled = singleColumn;
  if (singleColumn && dom["plot-x"].value === "-1") dom["plot-x"].value = "0";
}

function drawPlotRich() {
  if (!state.array || !dom["plot-canvas"]) return;
  const canvas = prepareCanvas(dom["plot-canvas"]);
  if (!canvas) return;
  const kind = dom["plot-kind"].value;
  const xColumn = Number(dom["plot-x"].value);
  const yColumn = Number(dom["plot-y"].value);
  const rows = layout().rows;
  const stride = Math.max(1, Math.ceil(rows / 100000));

  if (kind === "hist" || kind === "box") {
    const values = [];
    for (let row = 0; row < rows; row += stride) {
      const value = numericValue(valueAt(row, xColumn));
      if (Number.isFinite(value)) values.push(value);
    }
    if (kind === "hist") drawHistogramRich(canvas.ctx, canvas.width, canvas.height, values, Number(dom["hist-bins"].value));
    else drawBoxPlot(canvas.ctx, canvas.width, canvas.height, values);
    return;
  }

  const points = [];
  for (let row = 0; row < rows; row += stride) {
    const x = xColumn === -1 ? row : numericValue(valueAt(row, xColumn));
    const y = numericValue(valueAt(row, yColumn));
    if (Number.isFinite(x) && Number.isFinite(y)) points.push([x, y]);
  }
  drawXYRich(canvas.ctx, canvas.width, canvas.height, points, kind, dom["axis-scale"].value);
}

function drawFrame(ctx, width, height, grid) {
  const frame = { left: 62, top: 20, right: Math.max(84, width - 20), bottom: Math.max(50, height - 40) };
  ctx.fillStyle = "#11161e";
  ctx.fillRect(0, 0, width, height);
  if (grid) {
    ctx.strokeStyle = "#25303e";
    ctx.lineWidth = 1;
    for (let index = 0; index <= 5; index += 1) {
      const x = frame.left + (frame.right - frame.left) * index / 5;
      const y = frame.top + (frame.bottom - frame.top) * index / 5;
      ctx.beginPath(); ctx.moveTo(x, frame.top); ctx.lineTo(x, frame.bottom); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(frame.left, y); ctx.lineTo(frame.right, y); ctx.stroke();
    }
  }
  ctx.strokeStyle = "#526074";
  ctx.beginPath();
  ctx.moveTo(frame.left, frame.top);
  ctx.lineTo(frame.left, frame.bottom);
  ctx.lineTo(frame.right, frame.bottom);
  ctx.stroke();
  return frame;
}

function drawXYRich(ctx, width, height, points, kind, scale) {
  const logX = scale === "log-x" || scale === "log-log";
  const logY = scale === "log-y" || scale === "log-log";
  const usable = points.filter(([x, y]) => (!logX || x > 0) && (!logY || y > 0));
  if (!usable.length) return showPlotMessage("選択した軸スケールで表示できる数値がありません");
  hidePlotMessage();
  const transformed = usable.map(([x, y]) => [logX ? Math.log10(x) : x, logY ? Math.log10(y) : y]);
  const bounds = pointBounds(transformed);
  const frame = drawFrame(ctx, width, height, dom["plot-grid"].checked);
  const map = pointMapper(frame, bounds);
  if (kind === "line") {
    ctx.strokeStyle = "#67a5ff";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    transformed.forEach((point, index) => {
      const [x, y] = map(point);
      if (index) ctx.lineTo(x, y); else ctx.moveTo(x, y);
    });
    ctx.stroke();
  } else {
    ctx.fillStyle = "#5dd7b7";
    transformed.forEach((point) => {
      const [x, y] = map(point);
      ctx.beginPath(); ctx.arc(x, y, 2.2, 0, Math.PI * 2); ctx.fill();
    });
  }
  drawAxisLabelsRich(
    ctx, frame,
    logX ? Math.pow(10, bounds.xMin) : bounds.xMin,
    logX ? Math.pow(10, bounds.xMax) : bounds.xMax,
    logY ? Math.pow(10, bounds.yMin) : bounds.yMin,
    logY ? Math.pow(10, bounds.yMax) : bounds.yMax
  );
}

function drawHistogramRich(ctx, width, height, values, requestedBins) {
  if (!values.length) return showPlotMessage("数値として集計できるデータがありません");
  hidePlotMessage();
  values.sort((a, b) => a - b);
  let min = values[0], max = values[values.length - 1];
  [min, max] = normalizeRange(min, max);
  const bins = Math.max(2, Math.min(200, requestedBins || 30));
  const counts = Array(bins).fill(0);
  const binWidth = (max - min) / bins;
  values.forEach((value) => {
    const index = Math.max(0, Math.min(bins - 1, Math.floor((value - min) / binWidth)));
    counts[index] += 1;
  });
  const peak = Math.max(...counts, 1);
  const frame = drawFrame(ctx, width, height, dom["plot-grid"].checked);
  const barWidth = (frame.right - frame.left) / bins;
  ctx.fillStyle = "#67a5ff";
  counts.forEach((count, index) => {
    const height = count / peak * (frame.bottom - frame.top);
    ctx.fillRect(frame.left + index * barWidth + 1, frame.bottom - height, Math.max(1, barWidth - 2), height);
  });
  drawAxisLabelsRich(ctx, frame, min, max, 0, peak);
}

function drawBoxPlot(ctx, width, height, values) {
  if (!values.length) return showPlotMessage("数値として集計できるデータがありません");
  hidePlotMessage();
  values.sort((a, b) => a - b);
  const q1 = quantile(values, 0.25), median = quantile(values, 0.5), q3 = quantile(values, 0.75);
  const iqr = q3 - q1;
  const lowLimit = q1 - 1.5 * iqr, highLimit = q3 + 1.5 * iqr;
  const low = values.find((value) => value >= lowLimit) ?? values[0];
  const high = values.slice().reverse().find((value) => value <= highLimit) ?? values[values.length - 1];
  let min = values[0], max = values[values.length - 1];
  [min, max] = normalizeRange(min, max);
  const frame = drawFrame(ctx, width, height, dom["plot-grid"].checked);
  const toX = (value) => frame.left + (value - min) / (max - min) * (frame.right - frame.left);
  const center = (frame.top + frame.bottom) / 2;
  const boxHeight = Math.min(90, (frame.bottom - frame.top) * 0.38);
  ctx.strokeStyle = "#5dd7b7";
  ctx.fillStyle = "#5dd7b72c";
  ctx.lineWidth = 2;
  ctx.fillRect(toX(q1), center - boxHeight / 2, toX(q3) - toX(q1), boxHeight);
  ctx.strokeRect(toX(q1), center - boxHeight / 2, toX(q3) - toX(q1), boxHeight);
  ctx.beginPath();
  ctx.moveTo(toX(low), center); ctx.lineTo(toX(q1), center);
  ctx.moveTo(toX(q3), center); ctx.lineTo(toX(high), center);
  ctx.moveTo(toX(low), center - boxHeight / 4); ctx.lineTo(toX(low), center + boxHeight / 4);
  ctx.moveTo(toX(high), center - boxHeight / 4); ctx.lineTo(toX(high), center + boxHeight / 4);
  ctx.moveTo(toX(median), center - boxHeight / 2); ctx.lineTo(toX(median), center + boxHeight / 2);
  ctx.stroke();
  ctx.fillStyle = "#ffb85c";
  const outliers = values.filter((value) => value < low || value > high);
  const stride = Math.max(1, Math.ceil(outliers.length / 2000));
  for (let index = 0; index < outliers.length; index += stride) {
    ctx.beginPath(); ctx.arc(toX(outliers[index]), center, 2, 0, Math.PI * 2); ctx.fill();
  }
  drawAxisLabelsRich(ctx, frame, min, max, 0, 1);
}

function drawAxisLabelsRich(ctx, frame, xMin, xMax, yMin, yMax) {
  ctx.fillStyle = "#8995a7";
  ctx.font = "11px ui-monospace, monospace";
  ctx.textBaseline = "top";
  ctx.textAlign = "left";
  ctx.fillText(formatNumber(xMin), frame.left, frame.bottom + 8);
  ctx.textAlign = "right";
  ctx.fillText(formatNumber(xMax), frame.right, frame.bottom + 8);
  ctx.textBaseline = "middle";
  ctx.fillText(formatNumber(yMax), frame.left - 8, frame.top);
  ctx.fillText(formatNumber(yMin), frame.left - 8, frame.bottom);
}

updatePlotControls = updatePlotControlsRich;
drawPlot = drawPlotRich;
