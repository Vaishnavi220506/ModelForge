// ModelForge website: live shrink-and-check in the browser.
// Author: Vaishnavi. Mirrors src/shrink.cpp (boundary-shift probing).
"use strict";
(() => {
const DATA = window.MODELFORGE_MODELS;
const RESULTS = window.MODELFORGE_DATA && window.MODELFORGE_DATA.real;
const NAMES = { iris: "Iris flowers", wine: "Wine", breast_cancer: "Breast cancer", digits: "Handwritten digits" };
const REALISTIC = 0.5;   // max RMS z-distance to a real sample
const BOX = 4;           // z-score box
const BUDGET = 200;      // extra smaller-model runs, as in the paper
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const frame = () => new Promise((resolve) => requestAnimationFrame(() => resolve()));
const el = (tag, attrs = {}, ...kids) => {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v; else if (k === "text") node.textContent = v;
    else if (k === "style") node.setAttribute("style", v); else node.setAttribute(k, v);
  }
  for (const kid of kids.flat()) if (kid != null) node.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  return node;
};

// ------------------------------------------------------------------ maths
function toModel(spec) {
  return { layers: spec.layers.map((l) => ({ in: l.in, out: l.out, w: Float32Array.from(l.w), b: Float32Array.from(l.b) })) };
}
function forward(model, x) {
  let h = x;
  const last = model.layers.length - 1;
  model.layers.forEach((layer, index) => {
    const z = new Float64Array(layer.out);
    for (let j = 0; j < layer.out; j++) z[j] = layer.b[j];
    for (let k = 0; k < layer.in; k++) {
      const v = h[k];
      if (v === 0) continue;
      const row = k * layer.out;
      for (let j = 0; j < layer.out; j++) z[j] += v * layer.w[row + j];
    }
    if (index < last) for (let j = 0; j < layer.out; j++) z[j] = Math.max(0, z[j]);
    h = z;
  });
  let max = -Infinity;
  for (const v of h) max = Math.max(max, v);
  let total = 0;
  const p = new Float64Array(h.length);
  for (let i = 0; i < h.length; i++) { p[i] = Math.exp(h[i] - max); total += p[i]; }
  for (let i = 0; i < p.length; i++) p[i] /= total;
  return p;
}
const argmax = (p) => { let best = 0; for (let i = 1; i < p.length; i++) if (p[i] > p[best]) best = i; return best; };
const margin = (p) => { let a = -1, b = -1; for (const v of p) { if (v > a) { b = a; a = v; } else if (v > b) b = v; } return p.length > 1 ? a - b : 1; };
const rms = (a, b) => { let t = 0; for (let i = 0; i < a.length; i++) t += (a[i] - b[i]) ** 2; return Math.sqrt(t / a.length); };
const lerp = (a, b, t) => a.map((v, i) => v + t * (b[i] - v));

function roundHalf(v) {
  if (v === 0 || !isFinite(v)) return v;
  if (Math.abs(v) > 65504) return Math.sign(v) * 65504;
  if (Math.abs(v) < 2 ** -14) return Math.round(v * 2 ** 24) / 2 ** 24;
  const e = Math.floor(Math.log2(Math.abs(v))) + 1;  // frexp exponent
  const m = v / 2 ** e;
  return Math.fround(Math.round(m * 2048) / 2048 * 2 ** e);
}
function compress(model, spec) {
  return { layers: model.layers.map((l) => {
    const w = Float32Array.from(l.w), b = Float32Array.from(l.b);
    if (spec.kind === "fp16") {
      for (let i = 0; i < w.length; i++) w[i] = roundHalf(w[i]);
      for (let i = 0; i < b.length; i++) b[i] = roundHalf(b[i]);
    } else if (spec.kind === "int") {
      let largest = 0;
      for (const v of w) largest = Math.max(largest, Math.abs(v));
      const levels = 2 ** (spec.bits - 1) - 1;
      const scale = largest > 0 ? largest / levels : 1;
      for (let i = 0; i < w.length; i++) w[i] = Math.fround(Math.round(w[i] / scale) * scale);
    } else {
      const cut = Math.floor(spec.fraction * w.length);
      const order = Array.from(w.keys()).sort((i, j) => Math.abs(w[i]) - Math.abs(w[j]));
      for (let i = 0; i < cut; i++) w[order[i]] = 0;
    }
    return { in: l.in, out: l.out, w, b };
  }) };
}
function specOf(kind, bits, prune) {
  if (kind === "fp16") return { kind: "fp16", label: "float16", size: 0.5 };
  if (kind === "int8") return { kind: "int", bits: 8, label: "int8", size: 8 / 32 };
  if (kind === "int4") return { kind: "int", bits: 4, label: "int4", size: 4 / 32 };
  if (kind === "int") return { kind: "int", bits, label: `int${bits}`, size: bits / 32 };
  return { kind: "prune", fraction: prune / 100, label: `prune${prune}`, size: 1 - prune / 100 };
}
function mulberry32(seed) {
  return () => { seed |= 0; seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const gaussian = (rand) => Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
const riskOf = (m) => m < 0.05 ? "negligible" : m < 0.2 ? "low" : m < 0.5 ? "moderate" : "high";

// ------------------------------------------------------------------ Guardian search
// Same three phases as src/shrink.cpp: screen lines, bisect the moved ones,
// then a severity-aware evolutionary refinement. Async so the page can animate.
async function guardianSearch(original, small, test, options = {}) {
  const budget = options.budget ?? BUDGET;
  const report = options.onEvent || (() => {});
  const animate = options.animate && !reduceMotion;
  const rows = test.rows, n = rows.length;
  const s = { evals: 0, extra: 0, lines: 0, moved: 0, flips: 0, worst: 0, worstInput: null, worstOrig: -1, worstComp: -1, worstDistance: 0, maxShift: 0, testFlips: 0 };
  const nearestDistance = (x) => { let best = Infinity; for (const r of rows) best = Math.min(best, rms(x, r)); return best; };
  const check = (x) => {
    const p = forward(original, x), q = forward(small, x);
    s.evals++;
    const c = argmax(p), d = argmax(q);
    let other = 0;
    for (let j = 0; j < q.length; j++) if (j !== c) other = Math.max(other, q[j]);
    if (c !== d) {
      const dist = nearestDistance(x);
      if (dist <= REALISTIC) {
        s.flips++;
        const m = margin(p);
        if (m > s.worst || !s.worstInput) {
          Object.assign(s, { worst: m, worstInput: x.slice(), worstOrig: c, worstComp: d, worstDistance: dist });
          report("worst", s);
        }
      }
    }
    return { disagree: c !== d, origMargin: margin(p), compMargin: q[c] - other };
  };
  const compClass = (x) => { s.evals++; return argmax(forward(small, x)); };
  const origClass = (x) => argmax(forward(original, x));

  // Standard practice first: the test set.
  const classes = [], margins = [];
  for (const r of rows) {
    const p = forward(original, r);
    classes.push(argmax(p));
    margins.push(margin(p));
    if (check(r).disagree) s.testFlips++;
  }
  const base = s.evals;
  const exhausted = () => s.evals - base >= budget;
  report("test", s);

  // Lines: each sample to its nearest samples of another class, least sure first.
  const order = Array.from(rows.keys()).sort((a, b) => margins[a] - margins[b]);
  const segments = [], seen = new Set();
  for (let k = 0; k < 3; k++) {
    for (const i of order) {
      const others = [];
      for (let j = 0; j < n; j++) if (classes[j] !== classes[i]) others.push([rms(rows[i], rows[j]), j]);
      others.sort((a, b) => a[0] - b[0]);
      if (k >= others.length) continue;
      const j = others[k][1], key = Math.min(i, j) + ":" + Math.max(i, j);
      if (!seen.has(key)) { seen.add(key); segments.push([i, j]); }
    }
  }
  const shifted = [];
  const screenEnd = base + Math.floor(budget * 0.3);
  for (const [i, j] of segments) {
    if (s.evals + 2 > screenEnd) break;
    const a = rows[i], b = rows[j], A = classes[i];
    let lo = 0, hi = 1;
    for (let step = 0; step < 30; step++) { const mid = (lo + hi) / 2; if (origClass(lerp(a, b, mid)) === A) lo = mid; else hi = mid; }
    const towardA = compClass(lerp(a, b, lo)) !== A;
    const towardB = compClass(lerp(a, b, hi)) === A;
    s.lines++;
    if (towardA || towardB) {
      s.moved++;
      shifted.push({ i, j, lo, hi, towardA });
      report("moved", s, { i, j, towardA });
    }
    if (animate && s.lines % 3 === 0) { report("progress", s); await frame(); }
  }
  const witnesses = [];
  const bisectEnd = screenEnd + Math.floor(budget * 0.3);
  for (const seg of shifted) {
    if (exhausted() || s.evals >= bisectEnd) break;
    const a = rows[seg.i], b = rows[seg.j], A = classes[seg.i];
    let left = seg.towardA ? 0 : seg.hi, right = seg.towardA ? seg.lo : 1;
    for (let step = 0; step < 12 && !exhausted(); step++) { const mid = (left + right) / 2; if (compClass(lerp(a, b, mid)) === A) left = mid; else right = mid; }
    const edge = seg.towardA ? right : left;
    const shift = Math.abs(edge - (seg.towardA ? seg.lo : seg.hi)) * rms(a, b);
    s.maxShift = Math.max(s.maxShift, shift);
    report("gap", s, { i: seg.i, j: seg.j, shift });
    if (exhausted()) break;
    const w = lerp(a, b, edge);
    check(w);
    witnesses.push(w);
    if (animate) { report("progress", s); await frame(); }
  }
  // Evolutionary refinement seeded with the gap edges and the least sure samples.
  const rand = mulberry32(7);
  const score = (x) => { const r = check(x); return r.disagree ? 1 + r.origMargin : -r.compMargin; };
  const population = [];
  for (const w of witnesses) { if (exhausted() || population.length >= 12) break; population.push({ x: w, f: score(w) }); }
  for (let k = 0; k < order.length && population.length < 12 && !exhausted(); k++) population.push({ x: rows[order[k]], f: score(rows[order[k]]) });
  let steps = 0;
  while (!exhausted() && population.length) {
    const pick = () => { const x = population[Math.floor(rand() * population.length)], y = population[Math.floor(rand() * population.length)]; return x.f > y.f ? x : y; };
    const p1 = pick(), p2 = pick();
    const child = p1.x.map((v, k) => {
      let c = rand() < 0.5 ? v : p2.x[k];
      if (rand() < 0.3) c += 0.1 * gaussian(rand);
      return Math.max(-BOX, Math.min(BOX, c));
    });
    const f = score(child);
    let weakest = 0;
    for (let k = 1; k < population.length; k++) if (population[k].f < population[weakest].f) weakest = k;
    if (f > population[weakest].f) population[weakest] = { x: child, f };
    if (animate && ++steps % 12 === 0) { report("progress", s); await frame(); }
  }
  s.extra = s.evals - base;
  report("done", s);
  return s;
}

// ------------------------------------------------------------------ state
const state = { dataset: "wine", arch: "mlp16", kind: "int4", bits: 4, prune: 30, original: null, small: null, test: null, last: null, running: false };
function load() {
  const d = DATA.datasets[state.dataset];
  state.test = { rows: d.test, labels: d.labels };
  state.original = toModel(d.models[state.arch]);
  const spec = specOf(state.kind, state.bits, state.prune);
  state.spec = spec;
  state.small = compress(state.original, spec);
  state.last = null;
}
const accuracy = (model) => state.test.rows.reduce((t, r, i) => t + (argmax(forward(model, r)) === state.test.labels[i]), 0) / state.test.rows.length;
const pct = (v, d = 1) => (v * 100).toFixed(d) + "%";

// ------------------------------------------------------------------ lab UI
function setupLab() {
  const select = $("dataset");
  for (const key of Object.keys(DATA.datasets)) select.append(el("option", { value: key, text: NAMES[key] || key }));
  select.value = state.dataset;
  select.addEventListener("change", () => { state.dataset = select.value; refresh(); });
  $("arch").addEventListener("change", (e) => { state.arch = e.target.value; refresh(); });
  document.querySelectorAll("#kinds button").forEach((b) => b.addEventListener("click", () => {
    state.kind = b.dataset.kind;
    if (state.kind === "int8") state.bits = 8;
    if (state.kind === "int4") state.bits = 4;
    $("bits").value = state.bits;
    refresh();
  }));
  $("bits").addEventListener("input", (e) => { state.bits = +e.target.value; state.kind = state.bits === 8 ? "int8" : state.bits === 4 ? "int4" : "int"; refresh(); });
  $("prune").addEventListener("input", (e) => { state.prune = +e.target.value; state.kind = "prune"; refresh(); });
  $("run").addEventListener("click", runCheck);
  $("compare-all").addEventListener("click", compareAll);
}
function refresh() {
  load();
  document.querySelectorAll("#kinds button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.kind === state.kind)));
  $("bits-field").hidden = state.kind === "fp16" || state.kind === "prune";
  $("prune-field").hidden = state.kind !== "prune";
  $("bits-value").textContent = state.bits;
  $("prune-value").textContent = state.prune + "%";
  $("s-size").textContent = pct(state.spec.size, 0);
  const a0 = accuracy(state.original), a1 = accuracy(state.small);
  $("s-acc").textContent = pct(a1);
  $("s-acc-note").textContent = `original ${pct(a0)}`;
  $("s-acc").style.color = a1 < a0 ? css("--warn") : "";
  const flips = state.test.rows.filter((r) => argmax(forward(state.original, r)) !== argmax(forward(state.small, r))).length;
  $("s-flips").textContent = `${flips}/${state.test.rows.length}`;
  $("s-flips").style.color = flips ? css("--warn") : "";
  resetGuardian();
  renderPlayground();
  // The slice view runs both models on 22,500 points; wait until the
  // controls settle so sliders stay smooth.
  clearTimeout(refresh.timer);
  refresh.timer = setTimeout(() => { renderExplorer(); renderBisect(); }, explorer.basis ? 220 : 0);
}
function resetGuardian() {
  $("risk").className = "small"; $("risk").textContent = "not run yet";
  for (const id of ["g-lines", "g-moved", "g-evals"]) $(id).textContent = "0";
  $("g-meter").style.width = "0"; $("g-worst").textContent = "none yet";
  $("log").replaceChildren();
  $("verdict").textContent = "Guardian walks between pairs of real samples that the original model classifies differently, finds where its decision changes, and checks whether the smaller model moved that boundary.";
}
function logLine(text, cls) {
  const log = $("log");
  log.append(el("div", { class: cls || "", text }));
  log.scrollTop = log.scrollHeight;
}
function className(i) {
  if (state.dataset === "wine") return `Cultivar ${i + 1}`;
  return DATA.datasets[state.dataset].classes[i] ?? String(i);
}
async function runCheck() {
  if (state.running) return;
  state.running = true;
  $("run").disabled = true;
  resetGuardian();
  logLine(`Checking ${state.spec.label} against the original on ${state.test.rows.length} test samples...`);
  const result = await guardianSearch(state.original, state.small, state.test, {
    animate: true,
    onEvent: (type, s, info) => {
      $("g-lines").textContent = s.lines; $("g-moved").textContent = s.moved;
      $("g-evals").textContent = Math.max(0, s.evals - state.test.rows.length);
      if (type === "test") logLine(`Test set: ${s.testFlips} prediction${s.testFlips === 1 ? "" : "s"} changed.`);
      if (type === "moved" && s.moved <= 6) logLine(`Line ${s.lines}: boundary moved toward ${info.towardA ? "A" : "B"}.`, "hit");
      if (type === "gap" && info.shift > 0.02) logLine(`Gap measured: boundary shifted ${info.shift.toFixed(3)} std.`, "hit");
      if (type === "worst") {
        $("g-meter").style.width = pct(Math.min(1, s.worst), 1);
        $("g-meter").style.background = css(s.worst < 0.2 ? "--good" : s.worst < 0.5 ? "--warn" : "--bad");
        $("g-worst").textContent = `${s.worst.toFixed(2)}: original says "${className(s.worstOrig)}", smaller model says "${className(s.worstComp)}", ${s.worstDistance.toFixed(2)} std from a real sample`;
        if (s.worst >= 0.05) logLine(`New worst flip: original was sure by ${s.worst.toFixed(2)}.`, "worst");
      }
    },
  });
  state.last = result;
  const risk = riskOf(result.worst);
  const badge = $("risk");
  badge.className = `risk ${risk} pop`; badge.textContent = risk;
  $("verdict").textContent = verdictText(result, risk);
  logLine(`Done: ${result.extra} extra runs, max boundary shift ${result.maxShift.toFixed(3)} std, risk ${risk}.`);
  $("run").disabled = false;
  state.running = false;
  renderExplorer();
  renderBisect();
}
function verdictText(r, risk) {
  if (!r.worstInput) return "No realistic input was found where the two models disagree.";
  const base = `Worst realistic flip: the original model was sure by ${r.worst.toFixed(2)} but the ${state.spec.label} model disagrees, ${r.worstDistance.toFixed(2)} std from a real sample.`;
  if (risk === "negligible" || risk === "low") return base + " Disagreements only happen at near ties, so this compression looks safe to ship.";
  return base + (r.testFlips === 0 ? " The test set showed no changed predictions, so accuracy alone would have hidden this." : " Treat this compression as risky.");
}
async function compareAll() {
  if (state.running) return;
  state.running = true;
  $("compare-all").disabled = true;
  const card = $("compare-card");
  card.hidden = false;
  const table = $("compare-table");
  table.replaceChildren(el("thead", {}, el("tr", {}, ["Level", "Weights", "Accuracy", "Test flips", "Worst flip", "Risk"].map((h, i) => el("th", { class: i && i < 5 ? "num" : "", text: h })))));
  const body = el("tbody");
  table.append(body);
  const a0 = accuracy(state.original);
  const rows = [];
  for (const [kind, prune] of [["fp16"], ["int8"], ["int4"], ["prune", 30], ["prune", 50]]) {
    const spec = specOf(kind, kind === "int8" ? 8 : 4, prune);
    const small = compress(state.original, spec);
    const r = await guardianSearch(state.original, small, state.test);
    const risk = riskOf(r.worst);
    const tr = el("tr", {}, el("td", { text: spec.label }), el("td", { class: "num", text: pct(spec.size, 0) }),
      el("td", { class: "num", text: `${pct(a0)} → ${pct(accuracy(small))}` }), el("td", { class: "num", text: `${r.testFlips}/${state.test.rows.length}` }),
      el("td", { class: "num", text: r.worst.toFixed(2) }), el("td", {}, el("span", { class: `risk ${risk}`, text: risk })));
    body.append(tr);
    rows.push({ tr, spec, risk });
    await frame();
  }
  const safe = rows.filter((r) => r.risk === "negligible" || r.risk === "low").sort((x, y) => x.spec.size - y.spec.size)[0];
  if (safe) {
    safe.tr.classList.add("recommended");
    table.append(el("caption", { style: "caption-side:bottom;text-align:left;padding-top:12px", class: "small", text: `Recommended: ${safe.spec.label}, the smallest level with low risk (weights at ${pct(safe.spec.size, 0)} of float32).` }));
  }
  $("compare-all").disabled = false;
  state.running = false;
}

// ------------------------------------------------------------------ explorer
const PLANE = 150;
const explorer = { sliceSeed: 0, view: "original", basis: null, zoom: 1 };
function classColour(index, present) {
  const slot = present.indexOf(index);
  return slot >= 0 && slot < 6 ? css(`--c${slot}`) : css("--text-muted");
}
function hexToRgb(hex) { const v = parseInt(hex.replace("#", ""), 16); return [(v >> 16) & 255, (v >> 8) & 255, v & 255]; }
function chooseSlice() {
  const rows = state.test.rows;
  const classes = rows.map((r) => argmax(forward(state.original, r)));
  let W = state.last && state.last.worstInput;
  const nearest = (x, pred) => { let best = -1, d = Infinity; rows.forEach((r, i) => { if (pred(i)) { const v = rms(x, r); if (v < d) { d = v; best = i; } } }); return best; };
  if (!W || explorer.sliceSeed > 0) {
    const rand = mulberry32(11 + explorer.sliceSeed);
    const i = Math.floor(rand() * rows.length);
    const j = nearest(rows[i], (k) => classes[k] !== classes[i]);
    const c = nearest(lerp(rows[i], rows[j], 0.5), (k) => k !== i && k !== j);
    return { A: rows[i], B: rows[j], W: rows[c], label: "C" };
  }
  const wc = argmax(forward(state.original, W));
  const ai = nearest(W, (k) => classes[k] === wc);
  const bi = nearest(W, (k) => classes[k] !== wc);
  return { A: rows[ai], B: rows[bi], W, label: "W" };
}
function renderExplorer() {
  const { A, B, W, label } = chooseSlice();
  const u = B.map((v, i) => v - A[i]);
  const L = Math.sqrt(u.reduce((t, v) => t + v * v, 0)) || 1;
  const w = W.map((v, i) => v - A[i]);
  const dot = w.reduce((t, v, i) => t + v * u[i], 0) / (L * L);
  let v = w.map((x, i) => x - dot * u[i]);
  let vn = Math.sqrt(v.reduce((t, x) => t + x * x, 0));
  if (vn < 1e-6) { v = u.map((_, i) => (i % 2 ? 1 : -1)); vn = Math.sqrt(v.length); }
  v = v.map((x) => (x / vn) * L);
  const wt = vn / L;
  const tMid = wt / 2;
  // Zoom centres on where the two models' boundaries split along A to B, so
  // thin disagreement bands become visible.
  let gapCentre = 0.5;
  {
    const cA = argmax(forward(state.original, A));
    let bo = null, bc = null;
    for (let i = 1; i <= 400 && (bo === null || bc === null); i++) {
      const x = lerp(A, B, i / 400);
      if (bo === null && argmax(forward(state.original, x)) !== cA) bo = i / 400;
      if (bc === null && argmax(forward(state.small, x)) !== cA) bc = i / 400;
    }
    if (bo !== null) gapCentre = bc !== null ? (bo + bc) / 2 : bo;
  }
  const span = Math.max(1.7, wt + 0.8) / explorer.zoom;
  const centreS = explorer.zoom > 1 ? gapCentre : 0.5, centreT = explorer.zoom > 1 ? 0 : tMid;
  const s0 = centreS - span / 2, t0 = centreT - span / 2;
  const point = (s, t) => A.map((a, i) => a + s * u[i] + t * v[i]);
  explorer.basis = { point, s0, t0, span, A, B, W, label, wt, dot };
  const canvas = $("plane");
  const N = PLANE;
  const orig = new Int16Array(N * N), comp = new Int16Array(N * N);
  const present = new Set();
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const p = point(s0 + (x / (N - 1)) * span, t0 + (1 - y / (N - 1)) * span);
    const c = argmax(forward(state.original, p)), d = argmax(forward(state.small, p));
    orig[y * N + x] = c; comp[y * N + x] = d; present.add(c);
  }
  const presentList = [...present].sort((a, b) => a - b);
  explorer.present = presentList;
  const off = document.createElement("canvas");
  off.width = N; off.height = N;
  const ctx = off.getContext("2d");
  const img = ctx.createImageData(N, N);
  const surface = hexToRgb(css("--surface-2"));
  const bad = hexToRgb(css("--bad"));
  const palette = presentList.map((c) => hexToRgb(classColour(c, presentList)));
  for (let i = 0; i < N * N; i++) {
    const cls = explorer.view === "original" ? orig[i] : comp[i];
    const slot = presentList.indexOf(cls);
    let rgb = slot >= 0 ? palette[slot].map((c, k) => Math.round(surface[k] * 0.55 + c * 0.45)) : surface;
    const x = i % N, y = Math.floor(i / N);
    const edge = (x + 1 < N && orig[i] !== orig[i + 1]) || (y + 1 < N && orig[i] !== orig[i + N]);
    if (orig[i] !== comp[i]) rgb = bad;
    else if (edge) rgb = rgb.map((c) => Math.round(c * 0.55));
    img.data.set([rgb[0], rgb[1], rgb[2], 255], i * 4);
  }
  ctx.putImageData(img, 0, 0);
  canvas.width = 480; canvas.height = 480;
  const g = canvas.getContext("2d");
  g.imageSmoothingEnabled = false;
  g.drawImage(off, 0, 0, 480, 480);
  const toPx = (s, t) => [((s - s0) / span) * 480, (1 - (t - t0) / span) * 480];
  const marker = (s, t, text, colour) => {
    const [x, y] = toPx(s, t);
    g.beginPath(); g.arc(x, y, 9, 0, Math.PI * 2); g.fillStyle = colour; g.fill();
    g.lineWidth = 2.5; g.strokeStyle = css("--surface-1"); g.stroke();
    g.fillStyle = "#fff"; g.font = "bold 11px system-ui, sans-serif"; g.textAlign = "center"; g.textBaseline = "middle"; g.fillText(text, x, y + 0.5);
  };
  marker(0, 0, "A", "#4b5563");
  marker(1, 0, "B", "#4b5563");
  marker(dot, wt, label, label === "W" ? css("--bad") : "#4b5563");
  const legend = $("plane-legend");
  legend.replaceChildren(...presentList.slice(0, 6).map((c) => el("span", {}, el("i", { class: "swatch", style: `background:${classColour(c, presentList)}` }), className(c))),
    el("span", {}, el("i", { class: "swatch", style: `background:${css("--bad")}` }), "models disagree"));
  renderSegment();
}
function showPoint(x, title, sub) {
  $("hover-title").textContent = title;
  $("hover-sub").textContent = sub;
  const p = forward(state.original, x), q = forward(state.small, x);
  const c = argmax(p), d = argmax(q);
  $("hover-orig").textContent = className(c);
  $("hover-comp").textContent = className(d);
  $("hover-comp").classList.toggle("disagree", c !== d);
  renderProbs($("hover-orig-probs"), p);
  renderProbs($("hover-comp-probs"), q);
}
function renderProbs(target, p, plain) {
  const top = Array.from(p.keys()).sort((a, b) => p[b] - p[a]).slice(0, 4);
  const present = explorer.present || [];
  const colourFor = (k) => (plain ? css("--accent") : classColour(k, present.length ? present : [k]));
  target.replaceChildren(...top.map((k) => el("div", { class: "prob" }, el("span", { text: className(k) }),
    el("div", { class: "track" }, el("div", { class: "fill", style: `width:${(p[k] * 100).toFixed(1)}%;background:${colourFor(k)}` })),
    el("b", { text: pct(p[k], 0) }))));
}
function setupExplorer() {
  const canvas = $("plane");
  const hover = (e) => {
    if (!explorer.basis) return;
    const box = canvas.getBoundingClientRect();
    const { point, s0, t0, span } = explorer.basis;
    const s = s0 + ((e.clientX - box.left) / box.width) * span;
    const t = t0 + (1 - (e.clientY - box.top) / box.height) * span;
    const x = point(s, t);
    let nearest = Infinity;
    for (const r of state.test.rows) nearest = Math.min(nearest, rms(x, r));
    showPoint(x, "Point on the slice", `${nearest.toFixed(2)} std from the nearest real sample${nearest <= REALISTIC ? " (realistic)" : ""}`);
  };
  canvas.addEventListener("pointermove", hover);
  canvas.addEventListener("pointerdown", hover);
  document.querySelectorAll("#view button").forEach((b) => b.addEventListener("click", () => {
    explorer.view = b.dataset.view;
    document.querySelectorAll("#view button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    renderExplorer();
  }));
  $("new-slice").addEventListener("click", () => { explorer.sliceSeed++; renderExplorer(); });
  document.querySelectorAll("#zoom button").forEach((b) => b.addEventListener("click", () => {
    explorer.zoom = +b.dataset.zoom;
    document.querySelectorAll("#zoom button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    renderExplorer();
  }));
  $("scrub").addEventListener("input", (e) => {
    const t = +e.target.value / 1000;
    const { A, B } = explorer.basis;
    showPoint(lerp(A, B, t), `${pct(t, 0)} of the way from A to B`, "Scrub the slider to move along the line.");
    renderSegment(t);
  });
}
function renderSegment(marker) {
  const { A, B } = explorer.basis;
  const cA = argmax(forward(state.original, A));
  const W = 600, H = 190, m = { l: 36, r: 10, t: 10, b: 24 };
  const xs = (t) => m.l + t * (W - m.l - m.r), ys = (v) => m.t + (1 - v) * (H - m.t - m.b);
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const add = (tag, attrs) => { const n = document.createElementNS(ns, tag); for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v); svg.append(n); return n; };
  const po = [], pc = [];
  let bo = null, bc = null;
  for (let i = 0; i <= 200; i++) {
    const t = i / 200, x = lerp(A, B, t);
    const p = forward(state.original, x)[cA], q = forward(state.small, x)[cA];
    if (i && bo === null && argmax(forward(state.original, x)) !== cA) bo = t;
    if (i && bc === null && argmax(forward(state.small, x)) !== cA) bc = t;
    po.push([xs(t), ys(p)]); pc.push([xs(t), ys(q)]);
  }
  if (bo !== null && bc !== null && Math.abs(bo - bc) > 0.001) add("rect", { x: xs(Math.min(bo, bc)), y: m.t, width: Math.max(2, Math.abs(xs(bo) - xs(bc))), height: H - m.t - m.b, fill: css("--bad"), opacity: 0.25 });
  for (const v of [0, 0.5, 1]) {
    add("line", { x1: m.l, x2: W - m.r, y1: ys(v), y2: ys(v), stroke: css("--grid") });
    const t = add("text", { x: m.l - 6, y: ys(v) + 4, "text-anchor": "end" }); t.textContent = pct(v, 0);
  }
  const label = (x, text) => { const t = add("text", { x, y: H - 6, "text-anchor": "middle" }); t.textContent = text; };
  label(xs(0), "A"); label(xs(1), "B");
  const path = (pts, colour) => add("path", { d: pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(""), fill: "none", stroke: colour, "stroke-width": 2.2, "stroke-linejoin": "round" });
  path(po, css("--c0"));
  path(pc, css("--c1"));
  if (marker !== undefined) add("line", { x1: xs(marker), x2: xs(marker), y1: m.t, y2: H - m.b, stroke: css("--text-primary"), "stroke-width": 1.5 });
  $("segment").replaceChildren(svg);
}

// ------------------------------------------------------------------ playground
const play = { raw: null };
function rawToZ(raw) { const d = DATA.datasets[state.dataset]; return raw.map((v, i) => (v - d.mean[i]) / d.std[i]); }
function zToRaw(z) { const d = DATA.datasets[state.dataset]; return z.map((v, i) => v * d.std[i] + d.mean[i]); }
function renderPlayground() {
  const d = DATA.datasets[state.dataset];
  const digits = state.dataset === "digits";
  $("play-title").textContent = digits ? "Draw a digit" : `Change a ${NAMES[state.dataset].toLowerCase()} sample`;
  $("play-lead").textContent = digits
    ? "Draw on the 8 by 8 grid (the size of the Digits dataset). Both models predict as you draw. Then ask Guardian to find the closest drawing where the smaller model changes its answer."
    : "Move the sliders (real units) and watch both models. Then ask Guardian to find the closest input where the smaller model changes its answer.";
  $("play-kind").textContent = state.spec.label;
  if (!play.raw || play.raw.length !== d.mean.length || play.dataset !== state.dataset) {
    play.dataset = state.dataset;
    play.raw = zToRaw(d.test[0]).map((v) => (digits ? Math.max(0, Math.min(16, Math.round(v))) : v));
  }
  const area = $("input-area");
  area.replaceChildren();
  if (digits) area.append(buildPad(play.raw, true));
  else area.append(buildSliders());
  $("nearby-result").textContent = "";
  $("nearby-area").replaceChildren();
  predictPlayground();
}
function padColour(v) {
  const bg = hexToRgb(css("--surface-3")), fg = hexToRgb(css("--text-primary"));
  const t = Math.max(0, Math.min(1, v / 16));
  return `rgb(${bg.map((c, k) => Math.round(c + (fg[k] - c) * t)).join(",")})`;
}
function buildPad(values, editable) {
  const pad = el("div", { class: "pad" + (editable ? "" : " mini-pad"), role: editable ? "application" : "img", "aria-label": editable ? "8 by 8 drawing grid" : "Digit found by Guardian" });
  values.forEach((v, i) => pad.append(el("div", { style: `background:${padColour(v)}`, "data-i": i })));
  if (!editable) return pad;
  let drawing = false;
  const paint = (e) => {
    const target = document.elementFromPoint(e.clientX, e.clientY);
    if (!target || target.parentElement !== pad) return;
    const i = +target.dataset.i, r = Math.floor(i / 8), c = i % 8;
    for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
      const rr = r + dr, cc = c + dc;
      if (rr < 0 || rr > 7 || cc < 0 || cc > 7) continue;
      const k = rr * 8 + cc;
      play.raw[k] = Math.min(16, play.raw[k] + (dr || dc ? 3 : 10));
      pad.children[k].style.background = padColour(play.raw[k]);
    }
    predictPlayground();
  };
  pad.addEventListener("pointerdown", (e) => { drawing = true; pad.setPointerCapture(e.pointerId); paint(e); });
  pad.addEventListener("pointermove", (e) => { if (drawing) paint(e); });
  pad.addEventListener("pointerup", () => { drawing = false; });
  return pad;
}
function buildSliders() {
  const d = DATA.datasets[state.dataset];
  const box = el("div", { class: "sliders" });
  d.features.forEach((name, i) => {
    const lo = d.mean[i] - 3 * d.std[i], hi = d.mean[i] + 3 * d.std[i];
    const input = el("input", { type: "range", min: lo, max: hi, step: (hi - lo) / 200, value: play.raw[i], "aria-label": name });
    const value = el("b", { text: Number(play.raw[i]).toPrecision(3) });
    input.addEventListener("input", () => { play.raw[i] = +input.value; value.textContent = Number(play.raw[i]).toPrecision(3); predictPlayground(); });
    box.append(el("div", { class: "slider-row" }, el("span", { text: name, title: name }), input, value));
  });
  return box;
}
function predictPlayground() {
  const z = rawToZ(play.raw);
  const p = forward(state.original, z), q = forward(state.small, z);
  const c = argmax(p), d = argmax(q);
  $("play-orig").textContent = className(c);
  $("play-comp").textContent = className(d);
  $("play-comp").classList.toggle("disagree", c !== d);
  renderProbs($("play-orig-probs"), p, true);
  renderProbs($("play-comp-probs"), q, true);
}
async function nearbyFlip() {
  const button = $("nearby");
  button.disabled = true;
  $("nearby-result").textContent = "Searching...";
  await frame();
  const start = rawToZ(play.raw);
  const rand = mulberry32(5);
  const p0 = argmax(forward(state.original, start));
  let best = null;
  // Evolutionary search inside a 0.5 std ball around your input: keep the
  // original's answer, push the smaller model to disagree, prefer confident originals.
  const fitness = (x) => {
    const p = forward(state.original, x), q = forward(state.small, x);
    const c = argmax(p), d = argmax(q);
    if (c !== d) return 1 + margin(p);
    let other = 0;
    for (let j = 0; j < q.length; j++) if (j !== c) other = Math.max(other, q[j]);
    return -(q[c] - other);
  };
  let pop = Array.from({ length: 16 }, (_, k) => { const x = k ? start.map((v) => v + 0.15 * gaussian(rand)) : start.slice(); return { x, f: fitness(x) }; });
  for (let gen = 0; gen < 800; gen++) {
    const pick = () => { const a = pop[Math.floor(rand() * pop.length)], b = pop[Math.floor(rand() * pop.length)]; return a.f > b.f ? a : b; };
    const a = pick(), b = pick();
    let child = a.x.map((v, k) => (rand() < 0.5 ? v : b.x[k]) + (rand() < 0.3 ? 0.1 * gaussian(rand) : 0));
    const dist = rms(child, start);
    if (dist > REALISTIC) child = child.map((v, k) => start[k] + (v - start[k]) * (REALISTIC / dist));
    const f = fitness(child);
    let weakest = 0;
    for (let k = 1; k < pop.length; k++) if (pop[k].f < pop[weakest].f) weakest = k;
    if (f > pop[weakest].f) pop[weakest] = { x: child, f };
    if (f >= 1 && (!best || f > best.f)) best = { x: child, f };
    if (gen % 100 === 99) await frame();
  }
  button.disabled = false;
  const area = $("nearby-area");
  area.replaceChildren();
  if (!best) {
    $("nearby-result").textContent = `No input within ${REALISTIC} std of yours makes the ${state.spec.label} model disagree. This region looks stable.`;
    return;
  }
  const p = forward(state.original, best.x), q = forward(state.small, best.x);
  $("nearby-result").textContent = `Found: ${rms(best.x, start).toFixed(2)} std from your input, the original says "${className(argmax(p))}" (sure by ${margin(p).toFixed(2)}) but the ${state.spec.label} model says "${className(argmax(q))}".${p0 !== argmax(p) ? " (The original's answer also changed here.)" : ""}`;
  const raw = zToRaw(best.x);
  if (state.dataset === "digits") {
    area.append(el("div", { class: "pads" }, el("div", {}, el("div", { class: "small", text: "Yours" }), buildPad(play.raw.map((v) => Math.max(0, Math.min(16, v))), false)),
      el("div", {}, el("div", { class: "small", text: "Flip found" }), buildPad(raw.map((v) => Math.max(0, Math.min(16, v))), false))));
  } else {
    const d = DATA.datasets[state.dataset];
    const changes = raw.map((v, i) => ({ i, dz: (v - play.raw[i]) / d.std[i], v })).sort((x, y) => Math.abs(y.dz) - Math.abs(x.dz)).slice(0, 4);
    area.append(el("div", { class: "small", text: "Biggest changes from your input:" }),
      el("ul", { class: "small" }, changes.map((c) => el("li", { text: `${d.features[c.i]}: ${Number(play.raw[c.i]).toPrecision(3)} → ${Number(c.v).toPrecision(3)}` }))));
  }
}
function setupPlayground() {
  $("sample").addEventListener("click", () => {
    const d = DATA.datasets[state.dataset];
    const row = d.test[Math.floor(Math.random() * d.test.length)];
    play.raw = zToRaw(row).map((v) => (state.dataset === "digits" ? Math.max(0, Math.min(16, Math.round(v))) : v));
    play.dataset = state.dataset;
    renderPlayground();
  });
  $("clear").addEventListener("click", () => {
    const d = DATA.datasets[state.dataset];
    play.raw = state.dataset === "digits" ? new Array(64).fill(0) : d.mean.slice();
    play.dataset = state.dataset;
    renderPlayground();
  });
  $("nearby").addEventListener("click", nearbyFlip);
}

// ------------------------------------------------------------------ bisection animation
let bisectTimer = null;
function renderBisect() {
  clearTimeout(bisectTimer);
  const { A, B } = explorer.basis || chooseSlice();
  const cA = argmax(forward(state.original, A));
  const events = [];
  let lo = 0, hi = 1;
  for (let k = 0; k < 14; k++) { const mid = (lo + hi) / 2; events.push({ t: mid, who: "o" }); if (argmax(forward(state.original, lerp(A, B, mid))) === cA) lo = mid; else hi = mid; }
  const towardA = argmax(forward(state.small, lerp(A, B, lo))) !== cA;
  events.push({ t: lo, who: "c" }, { t: hi, who: "c" });
  let gap = null;
  if (towardA || argmax(forward(state.small, lerp(A, B, hi))) === cA) {
    let left = towardA ? 0 : hi, right = towardA ? lo : 1;
    for (let k = 0; k < 12; k++) { const mid = (left + right) / 2; events.push({ t: mid, who: "c" }); if (argmax(forward(state.small, lerp(A, B, mid))) === cA) left = mid; else right = mid; }
    gap = towardA ? [right, lo] : [hi, left];
  }
  const W = 640, H = 90, xs = (t) => 30 + t * (W - 60);
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const add = (tag, attrs) => { const n = document.createElementNS(ns, tag); for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v); svg.append(n); return n; };
  add("line", { x1: xs(0), x2: xs(1), y1: 45, y2: 45, stroke: css("--border"), "stroke-width": 4, "stroke-linecap": "round" });
  for (const [t, text] of [[0, "A"], [1, "B"]]) { const n = add("text", { x: xs(t), y: 80, "text-anchor": "middle" }); n.textContent = text; }
  $("bisect").replaceChildren(svg);
  let k = 0;
  const step = () => {
    if (k < events.length) {
      const e = events[k++];
      add("circle", { cx: xs(e.t), cy: e.who === "o" ? 33 : 57, r: 4.5, fill: css(e.who === "o" ? "--c0" : "--c1"), class: "pop" });
      bisectTimer = setTimeout(step, reduceMotion ? 0 : 140);
    } else {
      const note = add("text", { x: W / 2, y: 14, "text-anchor": "middle" });
      if (gap && Math.abs(gap[1] - gap[0]) * (W - 60) > 0.5) {
        add("rect", { x: xs(Math.min(...gap)), y: 38, width: Math.max(3, Math.abs(xs(gap[1]) - xs(gap[0]))), height: 14, rx: 3, fill: css("--bad"), class: "pop" });
        note.textContent = `Boundary moved: ${Math.abs(gap[1] - gap[0]).toFixed(4)} of the line is now disputed`;
      } else note.textContent = "No visible boundary shift on this line";
    }
  };
  step();
}

// ------------------------------------------------------------------ hero animation
function hero() {
  const canvas = $("hero-canvas");
  const g = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  const rand = mulberry32(3);
  const pts = Array.from({ length: 70 }, () => ({ x: rand() * W, y: rand() * H }));
  const boundary = (x, phase, amp) => H * 0.5 + Math.sin(x / 70 + phase) * 40 + Math.sin(x / 31) * amp;
  let start = performance.now(), visible = true;
  new IntersectionObserver(([e]) => { visible = e.isIntersecting; }).observe(canvas);
  const draw = (now) => {
    const t = (now - start) / 1000;
    const amp = 10 + 8 * Math.sin(t * 0.9);
    g.fillStyle = css("--surface-1"); g.fillRect(0, 0, W, H);
    const c0 = css("--c0"), c1 = css("--c1"), bad = css("--bad");
    for (let x = 0; x <= W; x += 4) {
      const yo = boundary(x, 0, 6), yc = boundary(x, 0.25 * Math.sin(t * 0.7), amp);
      g.globalAlpha = 0.12; g.fillStyle = c0; g.fillRect(x, 0, 4, yo); g.fillStyle = c1; g.fillRect(x, yo, 4, H - yo);
      g.globalAlpha = 0.55; g.fillStyle = bad; g.fillRect(x, Math.min(yo, yc), 4, Math.abs(yc - yo));
    }
    g.globalAlpha = 1;
    for (const p of pts) {
      const above = p.y < boundary(p.x, 0, 6);
      const flips = above !== (p.y < boundary(p.x, 0.25 * Math.sin(t * 0.7), amp));
      g.beginPath(); g.arc(p.x, p.y, flips ? 5 : 3.5, 0, Math.PI * 2);
      g.fillStyle = flips ? bad : above ? c0 : c1; g.fill();
    }
    const line = (phase, a, colour, width) => { g.beginPath(); for (let x = 0; x <= W; x += 4) g[x ? "lineTo" : "moveTo"](x, boundary(x, phase, a)); g.strokeStyle = colour; g.lineWidth = width; g.stroke(); };
    line(0, 6, css("--text-primary"), 2);
    line(0.25 * Math.sin(t * 0.7), amp, bad, 2);
    g.font = "12px system-ui, sans-serif"; g.fillStyle = css("--text-secondary");
    g.fillText("original model's boundary vs the smaller model's (red)", 14, H - 14);
    if (!reduceMotion && visible) requestAnimationFrame(draw);
    else if (!reduceMotion) setTimeout(() => requestAnimationFrame(draw), 400);
  };
  requestAnimationFrame(draw);
}

// ------------------------------------------------------------------ results, reveal, theme
function results() {
  const box = $("result-tiles");
  const tile = (value, label, note) => el("div", { class: "card reveal stat" }, el("span", { class: "label", text: label }), el("span", { class: "value", text: value }), el("span", { class: "note", text: note }));
  if (!RESULTS) {
    box.append(tile("22/24", "Risky compressions caught", "test set alone: 14"));
    return;
  }
  const d = RESULTS.detected, h = RESULTS.detected_hidden, u = RESULTS.detected_unchanged;
  box.append(
    tile(`${Math.round(d.boundary_shift)}/${RESULTS.risky}`, "Risky compressions caught", `test set alone: ${Math.round(d.test_set)}, genetic search: ${d.genetic.toFixed(1)}`),
    tile(`${Math.round(h.boundary_shift)}/${RESULTS.hidden_from_test_set}`, "Caught where the test set saw nothing", `genetic search: ${h.genetic.toFixed(1)}, random noise: ${h.noise.toFixed(1)}`),
    tile(`${Math.round(u.boundary_shift)}/${RESULTS.accuracy_unchanged}`, "Flagged although accuracy did not drop", `test set alone: ${Math.round(u.test_set)}`));
}
function reveal() {
  const observer = new IntersectionObserver((entries) => entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("visible"); observer.unobserve(e.target); } }), { threshold: 0.12 });
  document.querySelectorAll(".reveal").forEach((n, i) => { n.style.animationDelay = `${(i % 4) * 60}ms`; observer.observe(n); });
  const links = [...document.querySelectorAll(".nav-links a")];
  const sections = links.map((a) => document.querySelector(a.getAttribute("href")));
  const spy = new IntersectionObserver((entries) => entries.forEach((e) => { if (e.isIntersecting) links.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === "#" + e.target.id)); }), { rootMargin: "-40% 0px -55% 0px" });
  sections.forEach((s) => s && spy.observe(s));
}
function theme() {
  try { const t = localStorage.getItem("mf-site-theme"); if (t) document.documentElement.dataset.theme = t; } catch (e) { /* storage unavailable */ }
  $("theme").addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme ? document.documentElement.dataset.theme === "dark" : !matchMedia("(prefers-color-scheme: light)").matches;
    document.documentElement.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("mf-site-theme", document.documentElement.dataset.theme); } catch (e) { /* storage unavailable */ }
    renderExplorer(); renderBisect(); predictPlayground();
  });
}

theme();
setupLab();
setupExplorer();
setupPlayground();
refresh();
results();
reveal();
hero();
})();
