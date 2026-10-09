// ModelForge compiler in the browser. Author: Vaishnavi.
// A JavaScript port of the C++ pipeline in ModelForge/review_2_core_implementation:
// loader -> validator -> IR -> guarded optimisation (Guardian) -> C++ code generator.
// The C++ text comes from the real generator's templates (cpp_template.js).
"use strict";
window.ModelForgeCompiler = (() => {
const OPS = ["Gemm", "MatMul", "Add", "Relu", "Sigmoid", "Softmax"];
const RADIUS = 10;
const TOL = 1e-5;
const count = (shape) => shape.reduce((a, b) => a * b, 1);
const shapeText = (s) => "[" + s.join(" x ") + "]";

// ------------------------------------------------------------------ loader
function parse(text) {
  const model = { name: "model", input: null, output: null, tensors: new Map(), nodes: [] };
  const errors = [];
  text.split(/\r?\n/).forEach((raw, index) => {
    const line = index + 1;
    const clean = raw.split("#")[0].trim();
    if (!clean) return;
    const t = clean.split(/\s+/);
    const shapeOf = (s) => { const dims = s.split(",").map(Number); return dims.every((d) => Number.isInteger(d) && d > 0) ? dims : null; };
    if (t[0] === "model") {
      if (t.length !== 2) errors.push({ line, message: "Expected: model <name>" }); else model.name = t[1];
    } else if (t[0] === "input" || t[0] === "output") {
      if (t.length !== 4) return errors.push({ line, message: `Expected: ${t[0]} <name> float32 <shape>` });
      if (t[2] !== "float32" && t[2] !== "float") return errors.push({ line, message: `Unsupported datatype '${t[2]}'` });
      const shape = shapeOf(t[3]);
      if (!shape) return errors.push({ line, message: `Invalid tensor shape '${t[3]}'` });
      if (model.tensors.has(t[1])) return errors.push({ line, message: `Duplicate tensor declaration '${t[1]}'` });
      model.tensors.set(t[1], { name: t[1], shape, kind: t[0] === "input" ? "input" : "declared", line });
      if (t[0] === "input") { if (model.input) errors.push({ line, message: "Only one input is supported" }); model.input = t[1]; }
      else { if (model.output) errors.push({ line, message: "Only one output is supported" }); model.output = t[1]; }
    } else if (t[0] === "tensor") {
      if (t.length < 5) return errors.push({ line, message: "Expected: tensor <name> float32 <shape> values=..." });
      const shape = shapeOf(t[3]);
      if (!shape) return errors.push({ line, message: "Invalid constant tensor declaration" });
      const valuesToken = t.find((x) => x.startsWith("values="));
      const data = valuesToken ? valuesToken.slice(7).split(",").map(Number) : [];
      if (!data.length || data.some((v) => !isFinite(v))) return errors.push({ line, message: `Invalid constant values for '${t[1]}'` });
      if (data.length !== count(shape)) return errors.push({ line, message: `Tensor '${t[1]}' has ${data.length} values but shape ${shapeText(shape)} needs ${count(shape)}` });
      if (model.tensors.has(t[1])) return errors.push({ line, message: `Duplicate tensor declaration '${t[1]}'` });
      model.tensors.set(t[1], { name: t[1], shape, kind: "constant", data: Float32Array.from(data), line });
    } else if (t[0] === "node") {
      const arrow = t.indexOf("->");
      if (t.length < 5 || arrow < 4 || arrow + 1 >= t.length) return errors.push({ line, message: "Expected: node <name> <op> <inputs...> -> <output> [axis=N]" });
      const node = { name: t[1], op: t[2], inputs: t.slice(3, arrow), output: t[arrow + 1], axis: -1, transA: false, transB: false, line };
      for (const attr of t.slice(arrow + 2)) {
        const [k, v] = attr.split("=");
        if (k === "axis") node.axis = Number(v);
        else if (k === "transA") node.transA = v !== "0";
        else if (k === "transB") node.transB = v !== "0";
      }
      model.nodes.push(node);
    } else {
      errors.push({ line, message: `Unknown statement '${t[0]}'` });
    }
  });
  return { model, errors };
}

// ------------------------------------------------------------------ validator
function validate(model) {
  const errors = [];
  if (!model.input) errors.push({ line: 0, message: "Declared model input is missing" });
  if (!model.output) errors.push({ line: 0, message: "Declared model output is missing" });
  const available = new Set();
  for (const [name, t] of model.tensors) if (t.kind === "constant" || t.kind === "input") available.add(name);
  for (const node of model.nodes) {
    const fail = (message) => errors.push({ line: node.line, message });
    if (!OPS.includes(node.op)) { fail(`Unsupported operator '${node.op}' in node '${node.name}'`); continue; }
    let ok = true;
    for (const i of node.inputs) if (!available.has(i)) { fail(`Input tensor '${i}' is not available before node '${node.name}'`); ok = false; }
    const existing = model.tensors.get(node.output);
    if (existing && (existing.kind === "input" || existing.kind === "constant" || existing.producer)) { fail(`Duplicate or read-only output tensor '${node.output}'`); continue; }
    if (!ok) continue;
    const a = model.tensors.get(node.inputs[0]);
    let shape = null;
    if (node.op === "Gemm" || node.op === "MatMul") {
      if (node.op === "Gemm" ? !(node.inputs.length === 2 || node.inputs.length === 3) : node.inputs.length !== 2) { fail(node.op === "Gemm" ? "Gemm expects two inputs and an optional bias" : "MatMul expects exactly two inputs"); continue; }
      const b = model.tensors.get(node.inputs[1]);
      if (a.shape.length !== 2 || b.shape.length !== 2) { fail(`${node.op} requires two rank-2 tensors`); continue; }
      const rows = node.transA ? a.shape[1] : a.shape[0], inner = node.transA ? a.shape[0] : a.shape[1];
      const bRows = node.transB ? b.shape[1] : b.shape[0], cols = node.transB ? b.shape[0] : b.shape[1];
      if (inner !== bRows) { fail(`${node.op} dimensions are incompatible: ${shapeText(a.shape)} cannot multiply ${shapeText(b.shape)}`); continue; }
      shape = [rows, cols];
      if (node.inputs.length === 3) {
        const bias = count(model.tensors.get(node.inputs[2]).shape);
        if (bias !== cols && bias !== rows * cols) { fail("Gemm bias must contain one value per output element"); continue; }
      }
    } else if (node.op === "Add") {
      const b = model.tensors.get(node.inputs[1]);
      if (node.inputs.length !== 2 || !b || a.shape.join() !== b.shape.join()) { fail("Add inputs must have identical shapes"); continue; }
      shape = a.shape.slice();
    } else {
      if (node.inputs.length !== 1) { fail(`${node.op} expects exactly one input`); continue; }
      shape = a.shape.slice();
      if (node.op === "Softmax") {
        const axis = node.axis < 0 ? shape.length - 1 : node.axis;
        if (axis < 0 || axis >= shape.length) { fail("Softmax axis is outside the tensor rank"); continue; }
        if (axis !== shape.length - 1) { fail("Only the final Softmax axis is supported by the first code generator"); continue; }
        node.axis = axis;
      }
    }
    if (existing && existing.kind === "declared" && existing.shape.join() !== shape.join()) { fail(`Declared shape for '${node.output}' is ${shapeText(existing.shape)}, but the operation produces ${shapeText(shape)}`); continue; }
    model.tensors.set(node.output, { name: node.output, shape, kind: node.output === model.output ? "output" : "value", producer: node.name, line: node.line });
    available.add(node.output);
  }
  if (model.output && !available.has(model.output)) errors.push({ line: 0, message: `Declared output tensor '${model.output}' is never produced` });
  return errors;
}

// ------------------------------------------------------------------ IR
function buildIR(model) {
  const values = new Map(), constants = new Map();
  for (const [name, t] of model.tensors) {
    values.set(name, { shape: t.shape });
    if (t.kind === "constant") constants.set(name, { shape: t.shape, data: Float32Array.from(t.data) });
  }
  const instructions = [{ op: "INPUT", node: "input", inputs: [], output: model.input, shape: values.get(model.input).shape }];
  for (const n of model.nodes) instructions.push({ op: n.op.toUpperCase(), node: n.name, inputs: n.inputs.slice(), output: n.output, shape: values.get(n.output).shape, axis: n.axis, transA: n.transA, transB: n.transB });
  instructions.push({ op: "RETURN", node: "return", inputs: [model.output], output: model.output, shape: values.get(model.output).shape });
  return { name: model.name, input: model.input, output: model.output, values, constants, instructions };
}
function cloneIR(ir) {
  return { ...ir, values: new Map([...ir.values].map(([k, v]) => [k, { shape: v.shape.slice() }])),
    constants: new Map([...ir.constants].map(([k, v]) => [k, { shape: v.shape.slice(), data: Float32Array.from(v.data) }])),
    instructions: ir.instructions.map((i) => ({ ...i, inputs: i.inputs.slice() })) };
}
const GEMM_LIKE = new Set(["GEMM", "MATMUL", "FUSED_GEMM_RELU", "FUSED_GEMM_RELU6", "FUSED_GEMM_SIGMOID"]);
function gemmDims(ir, ins) {
  const a = ir.values.get(ins.inputs[0]).shape, b = ir.values.get(ins.inputs[1]).shape;
  const tA = ins.op !== "MATMUL" && ins.transA, tB = ins.op !== "MATMUL" && ins.transB;
  return { rows: tA ? a[1] : a[0], inner: tA ? a[0] : a[1], cols: tB ? b[0] : b[1], tA, tB };
}
function execute(ir, input, trace) {
  const vals = new Map();
  const get = (name) => vals.get(name) || (ir.constants.get(name) && ir.constants.get(name).data);
  for (const ins of ir.instructions) {
    if (ins.op === "INPUT") { vals.set(ins.output, Float64Array.from(input)); continue; }
    if (ins.op === "RETURN") { if (trace) for (const [k, v] of vals) trace.set(k, v); return get(ins.inputs[0]); }
    const a = get(ins.inputs[0]);
    let r;
    if (GEMM_LIKE.has(ins.op)) {
      const b = get(ins.inputs[1]), bias = ins.op !== "MATMUL" && ins.inputs[2] ? get(ins.inputs[2]) : null;
      const { rows, inner, cols, tA, tB } = gemmDims(ir, ins);
      r = new Float64Array(rows * cols);
      for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
        let v = bias ? bias[bias.length === cols ? j : i * cols + j] : 0;
        for (let k = 0; k < inner; k++) v += a[tA ? k * rows + i : i * inner + k] * b[tB ? j * inner + k : k * cols + j];
        if (ins.op === "FUSED_GEMM_RELU") v = Math.max(0, v);
        if (ins.op === "FUSED_GEMM_RELU6") v = Math.min(6, Math.max(0, v));
        if (ins.op === "FUSED_GEMM_SIGMOID") v = 1 / (1 + Math.exp(-v));
        r[i * cols + j] = v;
      }
    } else if (ins.op === "ADD") { const b = get(ins.inputs[1]); r = a.map((v, i) => v + b[i]); }
    else if (ins.op === "RELU") r = Float64Array.from(a, (v) => Math.max(0, v));
    else if (ins.op === "SIGMOID") r = Float64Array.from(a, (v) => 1 / (1 + Math.exp(-v)));
    else if (ins.op === "SOFTMAX") { const m = Math.max(...a); const e = Float64Array.from(a, (v) => Math.exp(v - m)); const s = e.reduce((x, y) => x + y, 0); r = e.map((v) => v / s); }
    else throw new Error("Cannot execute " + ins.op);
    vals.set(ins.output, r);
  }
  throw new Error("IR has no return instruction");
}

// ------------------------------------------------------------------ optimisation passes
function liveValues(ir) {
  const live = new Set([ir.output]);
  for (let i = ir.instructions.length - 1; i >= 0; i--) { const ins = ir.instructions[i]; if (ins.op === "RETURN" || live.has(ins.output)) ins.inputs.forEach((x) => live.add(x)); }
  return live;
}
function constantFolding(ir, bug) {
  const events = [];
  ir.instructions = ir.instructions.filter((ins) => {
    if (ins.op === "INPUT" || ins.op === "RETURN" || !ins.inputs.every((x) => ir.constants.has(x))) return true;
    if (!["ADD", "RELU", "SIGMOID", "SOFTMAX"].includes(ins.op)) return true;
    const tmp = { ...ir, instructions: [{ op: "INPUT", inputs: [], output: "__none" }, ins, { op: "RETURN", inputs: [ins.output] }], values: new Map([...ir.values, ["__none", { shape: [1] }]]) };
    let data = Float32Array.from(execute(tmp, [0]));
    if (bug === "fold_round") data = data.map((v) => Math.round(v * 100) / 100);
    ir.constants.set(ins.output, { shape: ins.shape.slice(), data });
    events.push(`Constant-folded ${ins.op} '${ins.node}' into '${ins.output}'`);
    return false;
  });
  return events;
}
function fusion(ir, bug) {
  const events = [], out = [];
  const uses = (v) => ir.instructions.reduce((t, i) => t + i.inputs.filter((x) => x === v).length, 0);
  for (let i = 0; i < ir.instructions.length; i++) {
    const g = ir.instructions[i], r = ir.instructions[i + 1];
    if (g.op === "GEMM" && r && r.op === "RELU" && r.inputs[0] === g.output && uses(g.output) === 1) {
      const fused = { ...g, inputs: g.inputs.slice(), op: "FUSED_GEMM_RELU", node: `${g.node}+${r.node}`, output: r.output, shape: r.shape };
      if (bug === "drop_bias" && fused.inputs.length === 3) fused.inputs.pop();
      if (bug === "relu6") fused.op = "FUSED_GEMM_RELU6";
      if (bug === "sigmoid") fused.op = "FUSED_GEMM_SIGMOID";
      out.push(fused);
      events.push(`Fused GEMM '${g.node}' with ReLU '${r.node}'`);
      i++;
    } else out.push(g);
  }
  ir.instructions = out;
  return events;
}
function deadNodes(ir) {
  const events = [], live = liveValues(ir);
  ir.instructions = ir.instructions.filter((ins) => {
    if (ins.op === "INPUT" || ins.op === "RETURN" || live.has(ins.output)) return true;
    events.push(`Removed dead instruction '${ins.node}' producing '${ins.output}'`);
    return false;
  });
  return events;
}

// ------------------------------------------------------------------ analysis (IBP + boundary probes + coverage)
function unfuse(ir) {
  const copy = cloneIR(ir);
  copy.instructions = [];
  for (const ins of ir.instructions) {
    if (!ins.op.startsWith("FUSED_")) { copy.instructions.push({ ...ins, inputs: ins.inputs.slice() }); continue; }
    const pre = ins.output + "__preact";
    copy.values.set(pre, { shape: ins.shape.slice() });
    copy.instructions.push({ ...ins, op: "GEMM", inputs: ins.inputs.slice(), output: pre });
    copy.instructions.push({ op: ins.op === "FUSED_GEMM_SIGMOID" ? "SIGMOID" : "RELU", node: ins.node, inputs: [pre], output: ins.output, shape: ins.shape });
  }
  return copy;
}
function intervals(ir) {
  const box = new Map();
  const get = (n) => box.get(n) || (ir.constants.has(n) ? { lo: ir.constants.get(n).data, hi: ir.constants.get(n).data } : null);
  for (const ins of ir.instructions) {
    if (ins.op === "INPUT") { const c = count(ins.shape); box.set(ins.output, { lo: new Float64Array(c).fill(-RADIUS), hi: new Float64Array(c).fill(RADIUS) }); continue; }
    if (ins.op === "RETURN") continue;
    const a = get(ins.inputs[0]);
    if (!a) return box;
    let r;
    if (GEMM_LIKE.has(ins.op)) {
      const b = get(ins.inputs[1]), bias = ins.op !== "MATMUL" && ins.inputs[2] ? get(ins.inputs[2]) : null;
      const { rows, inner, cols, tA, tB } = gemmDims(ir, ins);
      r = { lo: new Float64Array(rows * cols), hi: new Float64Array(rows * cols) };
      for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
        let lo = bias ? bias.lo[bias.lo.length === cols ? j : i * cols + j] : 0, hi = bias ? bias.hi[bias.hi.length === cols ? j : i * cols + j] : 0;
        for (let k = 0; k < inner; k++) {
          const ai = tA ? k * rows + i : i * inner + k, bi = tB ? j * inner + k : k * cols + j;
          const p = [a.lo[ai] * b.lo[bi], a.lo[ai] * b.hi[bi], a.hi[ai] * b.lo[bi], a.hi[ai] * b.hi[bi]];
          lo += Math.min(...p); hi += Math.max(...p);
        }
        if (ins.op.startsWith("FUSED_GEMM_RELU")) { lo = Math.max(0, lo); hi = Math.max(0, hi); }
        r.lo[i * cols + j] = lo; r.hi[i * cols + j] = hi;
      }
    } else if (ins.op === "ADD") { const b = get(ins.inputs[1]); r = { lo: a.lo.map((v, i) => v + b.lo[i]), hi: a.hi.map((v, i) => v + b.hi[i]) }; }
    else if (ins.op === "RELU") r = { lo: a.lo.map((v) => Math.max(0, v)), hi: a.hi.map((v) => Math.max(0, v)) };
    else if (ins.op === "SIGMOID") r = { lo: a.lo.map((v) => 1 / (1 + Math.exp(-v))), hi: a.hi.map((v) => 1 / (1 + Math.exp(-v))) };
    else r = { lo: a.lo.map(() => 0), hi: a.hi.map(() => 1) };
    box.set(ins.output, r);
  }
  return box;
}
function analyzeSites(ir) {
  const u = unfuse(ir), live = liveValues(u), box = intervals(u), sites = [];
  for (const ins of u.instructions) {
    if (ins.op !== "RELU" || !live.has(ins.output)) continue;
    const b = box.get(ins.inputs[0]);
    if (!b) continue;
    sites.push({ node: ins.node, preact: ins.inputs[0], width: b.lo.length, lower: Array.from(b.lo), upper: Array.from(b.hi),
      stability: Array.from(b.lo, (lo, k) => (lo > 0 ? "active" : b.hi[k] <= 0 ? "inactive" : "unstable")) });
  }
  return { sites, unfused: u };
}
function mulberry32(seed) { return () => { seed |= 0; seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function genericProbes(n, seed = 20260917) {
  const rand = mulberry32(seed), probes = [];
  const add = (name, values) => probes.push({ name, values });
  add("zero", new Array(n).fill(0)); add("ones", new Array(n).fill(1));
  add("relu_boundary", Array.from({ length: n }, (_, i) => (i % 2 ? 1 : -1)));
  add("near_zero_positive", new Array(n).fill(1e-4)); add("near_zero_negative", new Array(n).fill(-1e-4));
  add("ramp", Array.from({ length: n }, (_, i) => -1 + (2 * i) / Math.max(1, n - 1)));
  for (const m of [0.1, 1, 10]) { add(`all_positive_${m}`, new Array(n).fill(m)); add(`all_negative_${m}`, new Array(n).fill(-m)); }
  for (let i = 0; i < Math.min(n, 16); i++) for (const v of [-10, -1, 1, 10]) { const x = new Array(n).fill(0); x[i] = v; add(`coordinate_${i}_${v}`, x); }
  for (let s = 0; s < 32; s++) add(`random_${s}`, Array.from({ length: n }, () => (rand() * 2 - 1) * (s % 2 ? 10 : 1)));
  return probes;
}
function preact(u, site, x) { const trace = new Map(); execute(u, x, trace); return trace.get(site.preact); }
function boundaryProbes(u, sites, n, limit = 48) {
  const targets = [];
  for (let round = 0; targets.length < limit; round++) {
    let any = false;
    sites.forEach((site, s) => { const units = site.stability.map((v, k) => (v === "unstable" ? k : -1)).filter((k) => k >= 0); if (round < units.length && targets.length < limit) { targets.push([s, units[round]]); any = true; } });
    if (!any) break;
  }
  const probes = [], h = 1e-4;
  for (const [s, unit] of targets) {
    const site = sites[s];
    let x = new Array(n).fill(0), z = preact(u, site, x)[unit], g = null, ok = false;
    for (let it = 0; it < 10; it++) {
      g = Array.from({ length: n }, (_, k) => { const p = x.slice(), m = x.slice(); p[k] += h; m[k] -= h; return (preact(u, site, p)[unit] - preact(u, site, m)[unit]) / (2 * h); });
      const norm = g.reduce((t, v) => t + v * v, 0);
      if (norm < 1e-12) break;
      if (Math.abs(z) < 1e-6) { ok = true; break; }
      x = x.map((v, k) => Math.max(-RADIUS, Math.min(RADIUS, v - (z * g[k]) / norm)));
      z = preact(u, site, x)[unit];
    }
    if (!ok) continue;
    const norm = g.reduce((t, v) => t + v * v, 0);
    for (const [tag, target] of [["neg", -0.05], ["p5e-4", 5e-4], ["p5e-3", 5e-3], ["p5e-2", 5e-2]]) {
      probes.push({ name: `deep_boundary_L${s}_n${unit}_${tag}`, values: x.map((v, k) => Math.max(-RADIUS, Math.min(RADIUS, v + ((target - z) * g[k]) / norm))) });
    }
  }
  return probes;
}
const BIN_EDGES = [-Infinity, -0.1, 0, 1e-3, 1e-2, 0.1, Infinity];
const binOf = (z) => (z < -0.1 ? 0 : z <= 0 ? 1 : z <= 1e-3 ? 2 : z <= 1e-2 ? 3 : z <= 0.1 ? 4 : 5);
function statesOf(u, sites, probe) {
  const trace = new Map(); execute(u, probe.values, trace);
  const states = []; let offset = 0;
  for (const site of sites) { const z = trace.get(site.preact); if (z) z.forEach((v, k) => states.push(offset + k * 6 + binOf(v))); offset += site.width * 6; }
  return states;
}
function prioritize(u, sites, probes) {
  const states = probes.map((p) => statesOf(u, sites, p)), covered = new Set(), used = new Array(probes.length).fill(false), ordered = [];
  for (;;) {
    let best = -1, gain = 0;
    states.forEach((s, i) => { if (used[i]) return; const g = s.filter((x) => !covered.has(x)).length; if (g > gain) { gain = g; best = i; } });
    if (best < 0) break;
    used[best] = true; states[best].forEach((x) => covered.add(x)); ordered.push(probes[best]);
  }
  probes.forEach((p, i) => { if (!used[i]) ordered.push(p); });
  return ordered;
}
function coverage(u, sites, probes) {
  const covered = new Set();
  probes.forEach((p) => statesOf(u, sites, p).forEach((s) => covered.add(s)));
  let feasible = 0, hit = 0, offset = 0;
  for (const site of sites) {
    for (let k = 0; k < site.width; k++) for (let b = 0; b < 6; b++) {
      const lo = site.lower[k], hi = site.upper[k];
      const ok = hi >= BIN_EDGES[b] && lo <= BIN_EDGES[b + 1] && !(b >= 2 && hi <= BIN_EDGES[b]);
      const h = covered.has(offset + k * 6 + b);
      if (ok || h) { feasible++; if (h) hit++; }
    }
    offset += site.width * 6;
  }
  return feasible ? hit / feasible : 0;
}

// ------------------------------------------------------------------ Guardian
const argmax = (p) => { let b = 0; for (let i = 1; i < p.length; i++) if (p[i] > p[b]) b = i; return b; };
function exposes(before, after, x) {
  let a, b;
  try { a = execute(before, x); b = execute(after, x); } catch (e) { return { bad: true, a: [], b: [], err: Infinity }; }
  let err = 0;
  for (let i = 0; i < a.length; i++) err = Math.max(err, Math.abs(a[i] - b[i]));
  return { bad: a.length !== b.length || !(err <= TOL) || argmax(a) !== argmax(b), a: Array.from(a), b: Array.from(b), err };
}
function localize(before, after, x) {
  const t1 = new Map(), t2 = new Map();
  try { execute(before, x, t1); execute(after, x, t2); } catch (e) { return null; }
  for (const ins of before.instructions) {
    if (ins.op === "INPUT" || ins.op === "RETURN") continue;
    const a = t1.get(ins.output), b = t2.get(ins.output);
    if (!a || !b) continue;
    let d = 0; for (let i = 0; i < a.length; i++) d = Math.max(d, Math.abs(a[i] - b[i]));
    if (d > TOL) return { node: ins.node, value: ins.output, difference: d };
  }
  return null;
}
function guardianSuite(before) {
  const n = count(before.values.get(before.input).shape);
  const { sites, unfused } = analyzeSites(before);
  let probes = genericProbes(n);
  if (sites.length) probes = prioritize(unfused, sites, probes.concat(boundaryProbes(unfused, sites, n)));
  return { probes, sites, unfused, n };
}
function checkRewrite(before, after) {
  const { probes, n } = guardianSuite(before);
  let maxError = 0, evaluations = 0;
  for (const p of probes) {
    evaluations++;
    const r = exposes(before, after, p.values);
    if (isFinite(r.err)) maxError = Math.max(maxError, r.err);
    if (r.bad) return finishReject(before, after, p, r, evaluations, maxError, n, false);
  }
  // Near-miss search: climb sub-tolerance output differences.
  const rand = mulberry32(7);
  let best = probes.slice(0, 4).map((p) => ({ x: p.values, s: exposes(before, after, p.values).err })).sort((a, b) => b.s - a.s)[0];
  let step = 1;
  for (let i = 0; i < 32 && best; i++) {
    evaluations++;
    const x = best.x.map((v) => Math.max(-RADIUS, Math.min(RADIUS, v + step * (rand() * 2 - 1))));
    const r = exposes(before, after, x);
    if (r.bad) return finishReject(before, after, { name: `near_miss_${i + 1}`, values: x }, r, evaluations, maxError, n, true);
    if (r.err > best.s) { best = { x, s: r.err }; step *= 1.5; } else step *= 0.7;
  }
  return { accepted: true, probes: evaluations, maxError };
}
function finishReject(before, after, probe, r, evaluations, maxError, n, nearMiss) {
  // Shrink the witness toward zero while it still exposes the fault.
  let x = probe.values.slice();
  for (let i = 0; i < n; i++) for (const trial of [0, x[i] / 2]) { const y = x.slice(); y[i] = trial; if (exposes(before, after, y).bad) x = y; }
  const final = exposes(before, after, x);
  return { accepted: false, probes: evaluations, maxError, witness: { name: "minimized_from_" + probe.name, values: x }, original: final.a, candidate: final.b, divergence: localize(before, after, x), nearMiss };
}

// ------------------------------------------------------------------ C++ code generator
const sanitize = (v) => { let r = v.replace(/[^A-Za-z0-9_]/g, "_"); if (!r || /^[0-9]/.test(r)) r = "_" + r; return r; };
const variable = (n) => "v_" + sanitize(n), constant = (n) => "c_" + sanitize(n);
function floatLiteral(v) {
  const [m, e] = Math.fround(v).toExponential(9).split("e");
  const exp = Number(e);
  return `${m}e${exp < 0 ? "-" : "+"}${String(Math.abs(exp)).padStart(2, "0")}f`;
}
function inferBody(ir) {
  const ref = (n) => (ir.constants.has(n) ? constant(n) : variable(n));
  const lines = [`std::vector<float> infer(const std::vector<float>& input) {`,
    `    if (input.size() != ${count(ir.values.get(ir.input).shape)}) {`, `        throw std::runtime_error("Unexpected input size");`, `    }`];
  for (const ins of ir.instructions) {
    const out = `    Tensor ${variable(ins.output)} = `;
    if (ins.op === "INPUT") lines.push(`    Tensor ${variable(ins.output)} = input;`);
    else if (ins.op === "RETURN") lines.push(`    return ${ref(ins.inputs[0])};`);
    else if (GEMM_LIKE.has(ins.op) && ins.op !== "MATMUL") {
      const { tA, tB } = gemmDims(ir, ins), a = ir.values.get(ins.inputs[0]).shape, b = ir.values.get(ins.inputs[1]).shape;
      const fn = ins.op === "GEMM" ? "gemm" : "fused_gemm_relu";
      const call = `${fn}(${ref(ins.inputs[0])}, ${ref(ins.inputs[1])}, ${ins.inputs[2] ? "&" + ref(ins.inputs[2]) : "nullptr"}, ${tA ? a[1] : a[0]}, ${tA ? a[0] : a[1]}, ${tB ? b[0] : b[1]}, ${tA}, ${tB})`;
      if (ins.op === "FUSED_GEMM_RELU6") lines.push(`${out}${call};  // optimiser bug: clamps at 6`, `    for (float& value : ${variable(ins.output)}) value = std::min(value, 6.0f);`);
      else if (ins.op === "FUSED_GEMM_SIGMOID") lines.push(`${out}sigmoid(gemm(${ref(ins.inputs[0])}, ${ref(ins.inputs[1])}, ${ins.inputs[2] ? "&" + ref(ins.inputs[2]) : "nullptr"}, ${tA ? a[1] : a[0]}, ${tA ? a[0] : a[1]}, ${tB ? b[0] : b[1]}, ${tA}, ${tB}));  // optimiser bug`);
      else lines.push(`${out}${call};`);
    } else if (ins.op === "MATMUL") { const a = ir.values.get(ins.inputs[0]).shape, b = ir.values.get(ins.inputs[1]).shape; lines.push(`${out}matmul(${ref(ins.inputs[0])}, ${ref(ins.inputs[1])}, ${a[0]}, ${a[1]}, ${b[1]});`); }
    else if (ins.op === "ADD") lines.push(`${out}add(${ref(ins.inputs[0])}, ${ref(ins.inputs[1])});`);
    else lines.push(`${out}${ins.op.toLowerCase()}(${ref(ins.inputs[0])});`);
  }
  lines.push("}");
  return lines;
}
function codegen(ir) {
  const T = window.MODELFORGE_CPP;
  const used = new Set(ir.instructions.flatMap((i) => i.inputs));
  const constants = [...ir.constants].filter(([n]) => used.has(n)).map(([n, c]) => `static const Tensor ${constant(n)} = {${Array.from(c.data, floatLiteral).join(", ")}};`);
  const body = inferBody(ir);
  const source = T.prologue + constants.join("\n") + "\n\n" + body.join("\n") + "\n" +
    T.epilogue.replace("{{INPUTS}}", count(ir.values.get(ir.input).shape)).replace("{{OUTPUTS}}", count(ir.values.get(ir.output).shape));
  return { header: T.header, source, cmake: T.cmake, body };
}

// ------------------------------------------------------------------ metrics and pipeline
function metrics(ir) {
  const work = ir.instructions.filter((i) => i.op !== "INPUT" && i.op !== "RETURN");
  let macs = 0;
  for (const i of work) if (GEMM_LIKE.has(i.op)) { const d = gemmDims(ir, i); macs += d.rows * d.inner * d.cols; }
  // Counts the input and return steps too, like the C++ compiler's report.
  return { instructions: ir.instructions.length, buffers: work.length, elementwise: work.filter((i) => ["RELU", "SIGMOID", "ADD"].includes(i.op)).length, macs };
}
function timeRuns(ir, n, runs) {
  const rand = mulberry32(1), inputs = Array.from({ length: 64 }, () => Array.from({ length: n }, () => rand() * 2 - 1));
  const start = performance.now();
  for (let r = 0; r < runs; r++) execute(ir, inputs[r & 63]);
  return (performance.now() - start) / runs * 1000;  // microseconds per inference
}
function compile(text, options = {}) {
  const timings = [], stage = (name, fn) => { const t = performance.now(); const r = fn(); timings.push({ name, ms: performance.now() - t }); return r; };
  const { model, errors: parseErrors } = stage("Read model file", () => parse(text));
  if (parseErrors.length) return { ok: false, stage: "Read model file", errors: parseErrors, timings };
  const errors = stage("Check shapes and operators", () => validate(model));
  if (errors.length) return { ok: false, stage: "Check shapes and operators", errors, timings, model };
  const before = stage("Build internal graph (IR)", () => buildIR(model));
  const passes = [];
  const optimized = stage("Optimise, checked by Guardian", () => {
    let current = before;
    for (const [name, run] of [["constant_folding", constantFolding], ["dense_relu_fusion", fusion], ["dead_node_removal", deadNodes]]) {
      const candidate = cloneIR(current);
      const events = run(candidate, options.bug);
      if (!events.length) { passes.push({ name, events, accepted: true, probes: 0, maxError: 0, skipped: true }); continue; }
      const verdict = options.guardian === false ? { accepted: true, probes: 0, maxError: 0, unchecked: true } : checkRewrite(current, candidate);
      passes.push({ name, events, ...verdict });
      if (verdict.accepted) current = candidate;
    }
    return current;
  });
  const cpp = stage("Generate C++", () => codegen(optimized));
  const plain = codegen(before);
  const analysis = analyzeSites(before);
  const suite = guardianSuite(before);
  const n = count(before.values.get(before.input).shape);
  const runs = Math.max(200, Math.min(20000, Math.floor(2e6 / Math.max(1, metrics(before).macs))));
  return {
    ok: true, model, before, optimized, passes, cpp, plain, timings, n,
    sites: analysis.sites,
    probeSchedule: suite.probes.slice(0, 14).map((p) => p.name), probeCount: suite.probes.length,
    coverage: analysis.sites.length ? coverage(suite.unfused, suite.sites, suite.probes) : null,
    metrics: { before: metrics(before), after: metrics(optimized), lines: { before: plain.body.length, after: cpp.body.length },
      microsBefore: timeRuns(before, n, runs), microsAfter: timeRuns(optimized, n, runs), runs },
  };
}
return { compile, execute, parse, floatLiteral };
})();
