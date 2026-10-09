// ModelForge website: Compiler section. Author: Vaishnavi.
"use strict";
(() => {
const MF = window.ModelForgeCompiler;
const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
const wait = (ms) => new Promise((r) => setTimeout(r, reduceMotion ? 0 : ms));
const el = (tag, attrs = {}, ...kids) => {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v; else if (k === "text") node.textContent = v;
    else if (k === "style") node.setAttribute("style", v); else node.setAttribute(k, v);
  }
  for (const kid of kids.flat()) if (kid != null) node.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  return node;
};
const OP_COLOURS = { INPUT: "#64748b", RETURN: "#64748b", GEMM: "#2563eb", MATMUL: "#2563eb", FUSED_GEMM_RELU: "#0d9488", FUSED_GEMM_RELU6: "#b91c1c", FUSED_GEMM_SIGMOID: "#b91c1c", ADD: "#0369a1", RELU: "#b45309", SIGMOID: "#be185d", SOFTMAX: "#0e7490" };
const PASS_NAMES = { constant_folding: "Constant folding", dense_relu_fusion: "Layer fusion (Gemm + ReLU)", dead_node_removal: "Dead-code removal" };
const state = { result: null, file: "source" };

// ------------------------------------------------------------------ samples
function wineSample() {
  const d = window.MODELFORGE_MODELS && window.MODELFORGE_MODELS.datasets.wine;
  if (!d) return null;
  const layers = d.models.mlp16.layers;
  const fmt = (v) => String(Number(v.toPrecision(7)));
  const lines = ["# Real Wine classifier trained with scikit-learn (author: Vaishnavi).", "# Input: 13 standardised measurements of a wine; output: 3 cultivars.",
    "model wine_mlp16", `input x float32 1,${layers[0].in}`, `output y float32 1,${layers[layers.length - 1].out}`];
  layers.forEach((l, i) => {
    lines.push(`tensor w${i} float32 ${l.in},${l.out} values=${l.w.map(fmt).join(",")}`);
    lines.push(`tensor b${i} float32 1,${l.out} values=${l.b.map(fmt).join(",")}`);
  });
  lines.push("node dense_0 Gemm x w0 b0 -> z0", "node relu_0 Relu z0 -> h0", "node classifier Gemm h0 w1 b1 -> logits", "node probabilities Softmax logits -> y axis=1");
  return { id: "wine", label: "Real Wine classifier (trained, 13 inputs)", text: lines.join("\n") + "\n" };
}
function setupSamples() {
  const samples = [...(window.MODELFORGE_SAMPLES || [])];
  const wine = wineSample();
  if (wine) samples.splice(2, 0, wine);
  const select = $("c-sample");
  samples.forEach((s, i) => select.append(el("option", { value: String(i), text: s.label })));
  const load = () => { $("c-source").value = samples[+select.value].text; };
  select.addEventListener("change", () => { load(); compile(); });
  load();
  $("c-file").addEventListener("change", async () => {
    const file = $("c-file").files[0];
    if (!file) return;
    $("c-source").value = await file.text();
    compile();
  });
}

// ------------------------------------------------------------------ code view
const KEYWORDS = /\b(static|const|constexpr|return|for|if|else|int|float|bool|auto|using|namespace|throw|try|catch|true|false|nullptr|std|Tensor|void|char|size_t|vector|string)\b/g;
function escapeHtml(s) { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
function highlight(line) {
  if (/^\s*#/.test(line)) return `<span class="tok-pre">${escapeHtml(line)}</span>`;
  const comment = line.indexOf("//");
  const code = comment >= 0 ? line.slice(0, comment) : line, rest = comment >= 0 ? line.slice(comment) : "";
  let html = "";
  code.split(/("(?:[^"\\]|\\.)*")/).forEach((part, i) => {
    if (i % 2) { html += `<span class="tok-str">${escapeHtml(part)}</span>`; return; }
    html += escapeHtml(part).replace(/\b(\d+(?:\.\d+)?(?:e[+-]?\d+)?f?)\b/g, '<span class="tok-num">$1</span>').replace(KEYWORDS, '<span class="tok-kw">$1</span>');
  });
  return html + (rest ? `<span class="tok-com">${escapeHtml(rest)}</span>` : "");
}
// Long constant lines are shortened on screen only; Copy keeps the full text.
function display(line) {
  if (line.length > 180 && line.startsWith("static const Tensor c_")) {
    const n = line.split(",").length;
    return line.slice(0, 150) + ` ... (${n} values)};`;
  }
  return line;
}
function renderCode(lines, target, marks = {}) {
  target.innerHTML = lines.map((l, i) => `<span class="ln${marks[i] ? " " + marks[i] : ""}">${highlight(display(l)) || " "}</span>`).join("");
}
function showFile() {
  const r = state.result;
  if (!r || !r.ok) { $("c-code").innerHTML = ""; return; }
  renderCode(r.cpp[state.file].split("\n"), $("c-code"));
  document.querySelectorAll("#c-files button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.file === state.file)));
}

// ------------------------------------------------------------------ pipeline
async function compile() {
  const run = (state.run = (state.run || 0) + 1);  // a newer compile cancels this one
  const button = $("c-compile");
  button.disabled = true;
  const stages = $("c-stages");
  stages.replaceChildren();
  $("c-errors").replaceChildren();
  $("c-summary").replaceChildren();
  await wait(30);
  if (run !== state.run) return;
  let result;
  try {
    result = MF.compile($("c-source").value, { guardian: $("c-guardian").checked, bug: $("c-bug").value || undefined });
  } catch (error) {
    result = { ok: false, stage: "Compiler", errors: [{ line: 0, message: String(error.message || error) }], timings: [] };
  }
  state.result = result;
  const details = {
    "Read model file": () => `model '${result.model.name}', ${result.model.nodes.length} nodes, ${result.model.tensors.size} tensors`,
    "Check shapes and operators": () => "every shape and operator is valid",
    "Build internal graph (IR)": () => `${result.metrics.before.instructions} steps`,
    "Optimise, checked by Guardian": () => {
      const acc = result.passes.filter((p) => !p.skipped && p.accepted).length, rej = result.passes.filter((p) => !p.accepted).length;
      return `${result.metrics.before.instructions} → ${result.metrics.after.instructions} steps, ${acc} rewrite group${acc === 1 ? "" : "s"} kept${rej ? `, ${rej} rejected` : ""}`;
    },
    "Generate C++": () => `${result.cpp.source.split("\n").length} lines of model.cpp`,
  };
  const names = ["Read model file", "Check shapes and operators", "Build internal graph (IR)", "Optimise, checked by Guardian", "Generate C++"];
  for (const name of names) {
    const timing = result.timings.find((t) => t.name === name);
    const failed = !result.ok && result.stage === name;
    if (!timing && !failed) break;
    const rejected = name.startsWith("Optimise") && result.ok && result.passes.some((p) => !p.accepted);
    const unchecked = name.startsWith("Optimise") && result.ok && $("c-guardian").checked === false;
    stages.append(el("li", {}, el("span", { class: "mark" + (failed ? " fail" : rejected || unchecked ? " warn" : ""), text: failed ? "✕" : rejected || unchecked ? "!" : "✓" }),
      el("div", {}, el("div", { text: name }), el("div", { class: "detail", text: failed ? `stopped here: ${result.errors.length} error${result.errors.length === 1 ? "" : "s"}` : !result.ok ? (name === "Read model file" ? "file read" : "done") : unchecked ? "Guardian is off: rewrites were not tested" : rejected ? details[name]() + " (rolled back)" : details[name]() })),
      el("span", { class: "ms", text: timing ? `${timing.ms.toFixed(1)} ms` : "" })));
    await wait(180);
    if (run !== state.run) return;
  }
  if (!result.ok) {
    $("c-errors").append(el("div", { class: "error-box" }, el("b", { text: "The model has errors:" }),
      el("ul", {}, result.errors.map((e) => el("li", { text: (e.line ? `line ${e.line}: ` : "") + e.message })))));
  } else {
    const m = result.metrics;
    const tile = (label, value, note) => el("div", { class: "tile stat" }, el("span", { class: "label", text: label }), el("span", { class: "value", text: value }), el("span", { class: "note", text: note }));
    const tested = result.passes.reduce((t, p) => t + (p.probes || 0), 0);
    $("c-summary").append(
      tile("Steps", `${m.before.instructions} → ${m.after.instructions}`, "in the generated infer()"),
      tile("Interpreter speed-up", `${(m.microsBefore / m.microsAfter).toFixed(2)}x`, "measured in this browser"),
      tile("Guardian", `${tested}`, $("c-guardian").checked ? "test inputs run" : "off"));
  }
  renderAll();
  button.disabled = false;
}

// ------------------------------------------------------------------ panels
function irList(ir, changed) {
  return el("div", { class: "ir" }, ir.instructions.map((ins) => el("div", { class: "node" + (changed && changed.has(ins.output + ins.op) ? " changed" : "") },
    el("span", { class: "op", style: `background:${OP_COLOURS[ins.op] || "#475569"}`, text: ins.op }),
    el("span", { class: "name", text: `${ins.node}  →  ${ins.output}` }), el("span", { class: "shape", text: ins.shape.join("×") }))));
}
function passCard(p) {
  const verdict = p.skipped ? ["skip", "nothing to change"] : p.unchecked ? ["warn", "kept without testing (Guardian off)"] : p.accepted ? ["ok", "accepted"] : ["no", "rejected and rolled back"];
  const card = el("div", { class: "decision" },
    el("div", { style: "display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap" }, el("b", { text: PASS_NAMES[p.name] || p.name }), el("span", { class: `verdict ${verdict[0]}`, text: verdict[1] })),
    p.events.length ? el("div", { class: "small", text: p.events.join(" · ") }) : null,
    !p.skipped && !p.unchecked ? el("div", { class: "small", text: `${p.probes} test input${p.probes === 1 ? "" : "s"} · largest output difference ${p.maxError.toExponential(1)}` }) : null);
  if (!p.accepted && p.witness) {
    const fmt = (a) => "[" + a.map((v) => Number(v.toPrecision(4))).join(", ") + "]";
    card.append(el("div", { class: "small" }, el("b", { text: "Counterexample: " }), `input ${fmt(p.witness.values)}`),
      el("div", { class: "small", text: `original output ${fmt(p.original)} vs rewritten ${fmt(p.candidate)}` }),
      p.divergence ? el("div", { class: "small" }, el("b", { text: "Bug located at: " }), `'${p.divergence.node}' (value '${p.divergence.value}' differs by ${p.divergence.difference.toPrecision(3)})`) : null,
      p.nearMiss ? el("div", { class: "small", text: "Found by near-miss search after the targeted inputs passed." }) : null);
  }
  return card;
}
function renderDiff() {
  const panel = document.querySelector('.tab-panel[data-panel="diff"]');
  const r = state.result;
  panel.replaceChildren();
  if (!r || !r.ok) return;
  const plain = r.plain.body, opt = r.cpp.body;
  const left = el("pre", { class: "code" }), right = el("pre", { class: "code" });
  const marksL = {}, marksR = {};
  plain.forEach((l, i) => { if (!opt.includes(l)) marksL[i] = "removed"; });
  opt.forEach((l, i) => { if (!plain.includes(l)) marksR[i] = "added"; });
  renderCode(plain, left, marksL);
  renderCode(opt, right, marksR);
  panel.append(
    el("p", { class: "sub", text: "The generated infer() function without optimisation (left) and with Guardian-checked optimisation (right). Red lines disappear, green lines are new." }),
    el("div", { class: "grid two" }, el("div", {}, el("h3", { text: "Without optimisation" }), left), el("div", {}, el("h3", { text: "With optimisation" }), right)),
    el("h3", { text: "What the optimiser did", style: "margin-top:18px" }),
    el("div", { class: "grid three", style: "margin-top:8px" }, r.passes.map(passCard)));
}
function renderInspector() {
  const panel = document.querySelector('.tab-panel[data-panel="inspect"]');
  const r = state.result;
  panel.replaceChildren();
  if (!r || !r.ok) return;
  const symbols = [...r.model.tensors.values()].map((t) => el("tr", {}, el("td", { class: "mono", text: t.name }), el("td", { text: t.kind }), el("td", { class: "num", text: t.shape.join(" × ") }), el("td", { class: "num", text: t.line ? String(t.line) : "" })));
  const beforeKeys = new Set(r.before.instructions.map((i) => i.output + i.op));
  const changed = new Set(r.optimized.instructions.map((i) => i.output + i.op).filter((k) => !beforeKeys.has(k)));
  const neuronColour = { active: css("--good"), inactive: css("--text-muted"), unstable: css("--warn") };
  panel.append(
    el("div", { class: "grid two" },
      el("div", {}, el("h3", { text: "Internal graph before optimisation" }), el("p", { class: "sub", text: "Each step the compiler will run, in order." }), irList(r.before)),
      el("div", {}, el("h3", { text: "After Guardian-checked optimisation" }), el("p", { class: "sub", text: "Green border: a step created by an optimisation." }), irList(r.optimized, changed))),
    el("h3", { text: "Guardian decisions", style: "margin-top:22px" }),
    el("div", { class: "grid three", style: "margin-top:8px" }, r.passes.map(passCard)),
    el("div", { class: "grid two", style: "margin-top:22px" },
      el("div", {}, el("h3", { text: "Neuron analysis (interval bounds)" }),
        el("p", { class: "sub", text: "Each square is a ReLU neuron. Amber: it can switch on and off for inputs in [-10, 10], so Guardian aims test inputs at its switching point. Green: always on. Grey: always off." }),
        r.sites.length ? el("div", { style: "display:grid;gap:10px" }, r.sites.map((s) => el("div", {}, el("div", { class: "small", text: `${s.node}: ${s.stability.filter((v) => v === "unstable").length} of ${s.width} can switch` }),
          el("div", { class: "neurons" }, s.stability.map((v, k) => el("span", { style: `background:${neuronColour[v]}`, title: `neuron ${k}: ${v}, range [${s.lower[k].toFixed(2)}, ${s.upper[k].toFixed(2)}]` })))))) : el("p", { class: "small", text: "This model has no ReLU layers." }),
        r.coverage !== null ? el("p", { class: "small", style: "margin-top:10px", text: `Guardian's ${r.probeCount} test inputs cover ${(r.coverage * 100).toFixed(0)}% of the reachable neuron states.` }) : null),
      el("div", {}, el("h3", { text: "First test inputs Guardian runs" }), el("p", { class: "sub", text: "Ordered so each one covers new neuron states; deep_boundary inputs sit exactly on a neuron's switching point." }),
        el("div", { class: "chips" }, r.probeSchedule.map((n) => el("span", { text: n }))))),
    el("h3", { text: "Symbol table", style: "margin-top:22px" }),
    el("div", { class: "scroll" }, el("table", {}, el("thead", {}, el("tr", {}, ["Tensor", "Kind", "Shape", "Line"].map((h, i) => el("th", { class: i > 1 ? "num" : "", text: h })))), el("tbody", {}, symbols))));
}
function cppNumber(v) { return String(Number(Math.fround(v).toPrecision(9))); }
function renderRun() {
  const r = state.result;
  const out = $("c-run-out");
  out.replaceChildren();
  if (!r || !r.ok) return;
  const values = $("c-input").value.split(",").map((v) => v.trim()).filter(Boolean).map(Number);
  if (values.length !== r.n || values.some((v) => !isFinite(v))) {
    out.append(el("div", { class: "error-box", text: `Expected ${r.n} numbers separated by commas.` }));
    return;
  }
  const opt = Array.from(MF.execute(r.optimized, values)), ref = Array.from(MF.execute(r.before, values));
  const best = (a) => a.indexOf(Math.max(...a));
  const diff = Math.max(...opt.map((v, i) => Math.abs(v - ref[i])));
  const same = best(opt) === best(ref) && diff <= 1e-5;
  out.append(
    el("div", { class: "small", text: "What the generated C++ program prints for this input:" }),
    el("div", { class: "terminal" }, el("span", { class: "prompt", text: `> generated_model ${values.join(" ")}\n` }),
      `Scores: ${opt.map(cppNumber).join(" ")}\nPrediction: ${best(opt)}\nConfidence: ${cppNumber(Math.max(...opt))}`),
    el("p", { class: "match " + (same ? "ok" : "no"), text: same
      ? `✓ Same answer as the original model before optimisation (largest difference ${diff.toExponential(1)}).`
      : `✕ Different from the original model (prediction ${best(ref)}, largest difference ${diff.toPrecision(3)}). The optimiser bug reached the generated code.` }));
}
function renderEffect() {
  const panel = document.querySelector('.tab-panel[data-panel="effect"]');
  const r = state.result;
  panel.replaceChildren();
  if (!r || !r.ok) return;
  const m = r.metrics;
  const row = (label, a, b, fmt = String, lowerIsBetter = true) => {
    const better = lowerIsBetter ? b < a : b > a;
    return el("tr", {}, el("td", { text: label }), el("td", { text: fmt(a) }), el("td", { class: better ? "better" : "", text: fmt(b) }),
      el("td", { text: a === b ? "same" : `${b < a ? "−" : "+"}${Math.abs(((b - a) / a) * 100).toFixed(0)}%` }));
  };
  const tested = r.passes.reduce((t, p) => t + (p.probes || 0), 0);
  panel.append(
    el("p", { class: "sub", text: "Without optimisation versus with Guardian-checked optimisation, for this model." }),
    el("div", { class: "scroll" }, el("table", { class: "metric-table" },
      el("thead", {}, el("tr", {}, ["Measure", "Without", "With", "Change"].map((h) => el("th", { text: h })))),
      el("tbody", {},
        row("Steps in infer()", m.before.instructions, m.after.instructions),
        row("Temporary tensors created per inference", m.before.buffers, m.after.buffers),
        row("Separate passes over data (ReLU, Sigmoid, Add)", m.before.elementwise, m.after.elementwise),
        row("Lines in infer()", m.lines.before, m.lines.after),
        row("Multiply-adds per inference", m.before.macs, m.after.macs),
        row("Measured time per inference (µs, this browser)", m.microsBefore, m.microsAfter, (v) => v.toFixed(2))))),
    el("p", { class: "small", style: "margin-top:12px", text: `Timing: ${m.runs.toLocaleString()} inferences of each version in this browser's JavaScript interpreter. The multiply-adds stay the same; the gain comes from fewer steps, fewer temporary tensors and fewer passes over the data. The native C++ program follows the same steps.` }),
    el("p", { class: "small", text: `Correctness: Guardian ran ${tested} test inputs across the optimisations and ${r.passes.some((p) => !p.accepted) ? "rejected at least one rewrite" : "accepted every rewrite"}. Toggle the planted bugs on the left to see it reject them.` }));
}
function renderAll() {
  showFile();
  renderDiff();
  renderInspector();
  const r = state.result;
  if (r && r.ok) {
    const current = $("c-input").value.split(",").filter((v) => v.trim()).length;
    if (current !== r.n) $("c-input").value = Array.from({ length: r.n }, (_, i) => ((i % 5) - 2) * 0.5 + 1).join(", ");
  }
  renderRun();
  renderEffect();
}
function setupTabs() {
  const tabs = document.querySelectorAll("#c-results .tabs button");
  tabs.forEach((t) => t.addEventListener("click", () => {
    tabs.forEach((x) => x.setAttribute("aria-selected", String(x === t)));
    document.querySelectorAll("#c-results .tab-panel").forEach((p) => { p.hidden = p.dataset.panel !== t.dataset.panel; });
  }));
  document.querySelectorAll("#c-files button").forEach((b) => b.addEventListener("click", () => { state.file = b.dataset.file; showFile(); }));
  $("c-copy").addEventListener("click", async () => {
    const r = state.result;
    if (!r || !r.ok) return;
    try { await navigator.clipboard.writeText(r.cpp[state.file]); $("c-copy-note").textContent = "Copied."; }
    catch (e) {
      const range = document.createRange(); range.selectNodeContents($("c-code"));
      const sel = getSelection(); sel.removeAllRanges(); sel.addRange(range);
      $("c-copy-note").textContent = "Selected: press Ctrl+C to copy (long constant lines are shortened on screen).";
    }
  });
  $("c-run").addEventListener("click", renderRun);
  $("c-input").addEventListener("keydown", (e) => { if (e.key === "Enter") renderRun(); });
  $("c-random").addEventListener("click", () => {
    const r = state.result;
    if (!r || !r.ok) return;
    $("c-input").value = Array.from({ length: r.n }, () => (Math.random() * 6 - 3).toFixed(2)).join(", ");
    renderRun();
  });
  $("c-compile").addEventListener("click", compile);
  $("c-guardian").addEventListener("change", compile);
  $("c-bug").addEventListener("change", compile);
}

setupSamples();
setupTabs();
compile();
})();
