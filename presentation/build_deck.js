const pptxgen = require("pptxgenjs");
const { applyTheme } = require("/root/.claude/skills/synced/bec45f37-43da-4b3b-87a3-2860656f640f_cf0f4cb7-338e-4925-846c-909a9f238104/pptx/scripts/apply_theme.js");

const OUT = process.argv[2] || "Temper_Faculty_Review.pptx";
const THEME = {
  name: "Temper Hexagon",
  headFontFace: "Calibri",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "333F4F", lt1: "FFFFFF", dk2: "3A6EB5", lt2: "EEF4FA",
    accent1: "6F9BD8", accent2: "86C9D9", accent3: "C3E3EE", accent4: "2E9AB0",
    accent5: "D3D6DB", accent6: "D9614C", hlink: "3A6EB5", folHlink: "2E9AB0",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625
pres.author = "Vaishnavi";
pres.title = "Temper: falsify-or-certify trust for neural networks";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const BLUE = C.accent1, TEAL = C.accent2, PALE = C.accent3, DEEP = C.accent4, GREY = C.accent5, RED = C.accent6;

// Flat-top hexagon; w is the corner-to-corner width.
function hex(slide, x, y, w, color, opts = {}) {
  const o = { x, y, w, h: w * 0.866, fill: { color, transparency: opts.t || 0 }, line: { type: "none" } };
  if (opts.name) o.objectName = opts.name;
  if (opts.text !== undefined) {
    slide.addText(opts.text, { ...o, shape: pres.shapes.HEXAGON, align: "center", valign: "middle",
      color: opts.color || C.background1, fontSize: opts.size || 14, bold: true, margin: 0 });
  } else slide.addShape(pres.shapes.HEXAGON, o);
}

// Small honeycomb in the top-right and bottom-right corners of content slides.
function corner(slide) {
  hex(slide, 9.05, -0.35, 0.9, PALE);
  hex(slide, 9.55, 0.05, 0.6, TEAL, { t: 20 });
  hex(slide, 9.3, 4.9, 1.0, GREY, { t: 30 });
}

pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: HEX.lt1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 8.4, h: 0.75,
        fontSize: 30, bold: true, color: C.text2, valign: "middle", margin: 0 }, text: "" } },
    { text: { text: "Temper · ModelForge", options: { x: 0.5, y: 5.22, w: 4, h: 0.25, fontSize: 10, color: C.accent4, margin: 0 } } },
  ],
  slideNumber: { x: 8.7, y: 5.2, w: 0.5, h: 0.25, fontSize: 10, color: C.accent4 },
});
pres.defineSlideMaster({ title: "TITLE", background: { color: HEX.lt1 }, objects: [] });

const body = (o) => ({ isTextBox: true, fontSize: 14, color: C.text1, valign: "top", margin: 0, ...o });
function card(slide, x, y, w, h, fill = C.background2) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { type: "none" } });
}

// ---------------------------------------------------------------- 1. title
pres.addSection({ title: "Introduction" });
let s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Introduction" });
hex(s, -0.55, -0.45, 2.1, TEAL, { t: 10 });
hex(s, 1.05, -0.9, 2.1, PALE);
hex(s, -1.35, 1.35, 2.1, GREY, { t: 40 });
hex(s, 0.25, 0.45, 2.4, BLUE);
hex(s, 2.05, 1.05, 1.6, TEAL, { t: 25 });
hex(s, -0.25, 2.75, 1.7, PALE);
hex(s, 1.45, 2.55, 1.2, GREY, { t: 40 });
hex(s, 3.15, 0.25, 0.55, TEAL);
hex(s, 8.7, 2.9, 1.5, BLUE, { t: 10 });
hex(s, 8.05, 4.1, 0.6, TEAL, { t: 20 });
hex(s, 8.55, 4.55, 1.6, GREY, { t: 40 });
s.addText("TEMPER", body({ x: 4.0, y: 1.25, w: 5, h: 0.95, fontSize: 54, bold: true, color: C.text2, charSpacing: 4 }));
s.addText("Falsify-or-Certify Trust for Neural Networks", body({ x: 4.0, y: 2.2, w: 5.6, h: 0.45, fontSize: 20, bold: true, color: C.accent4 }));
s.addText("From trained model to shipped binary — built on the ModelForge C++17 model compiler",
  body({ x: 4.0, y: 2.8, w: 4.8, h: 0.6, fontSize: 14, color: C.text1 }));
s.addText([{ text: "Vaishnavi", options: { bold: true, breakLine: true } }, { text: "Project Review · October 2026" }],
  body({ x: 4.0, y: 3.75, w: 4, h: 0.6, fontSize: 14, color: C.text1 }));
s.addNotes("Temper is a trust layer: for every transformation a model goes through on its way to a device, it gives a counterexample, a proof, or the share already proven. ModelForge is the compiler underneath.");

// ---------------------------------------------------------------- 2. contents
s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Introduction" });
hex(s, -0.6, -0.3, 1.6, TEAL, { t: 20 });
hex(s, 0.6, 0.4, 1.3, PALE);
hex(s, -0.2, 1.55, 2.6, BLUE, { text: "CONTENTS", size: 22 });
hex(s, 1.95, 2.15, 1.2, PALE);
hex(s, -0.5, 3.75, 1.5, TEAL, { t: 20 });
hex(s, 1.0, 3.55, 0.9, GREY, { t: 30 });
hex(s, 9.3, 4.9, 1.0, GREY, { t: 30 });
const items = [
  ["The problem", "Every step to the device can silently change a prediction"],
  ["Our idea", "Three verdicts: refuted, certified, undecided"],
  ["How Temper works", "Falsifier + float32-sound certifier in one loop"],
  ["Headline metrics", "2,352 transformations of 96 networks"],
  ["Real models, real bug classes", "Compression, binaries, compiler-bug study"],
  ["After the verdict", "Repair, culprit localisation, reusable proofs"],
  ["Novelty and next steps", "What is new, honest limits, roadmap"],
];
const colors = [BLUE, TEAL, DEEP, BLUE, TEAL, DEEP, BLUE];
items.forEach(([h, d], i) => {
  const y = 0.42 + i * 0.67;
  hex(s, 4.0, y, 0.55, colors[i], { text: String(i + 1).padStart(2, "0"), size: 12 });
  s.addText(h, body({ x: 4.75, y: y - 0.04, w: 4.6, h: 0.3, fontSize: 15, bold: true, color: C.text2 }));
  s.addText(d, body({ x: 4.75, y: y + 0.24, w: 4.6, h: 0.25, fontSize: 11, color: C.text1 }));
});

// ---------------------------------------------------------------- 3. problem
pres.addSection({ title: "Approach" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Approach" });
corner(s);
s.addText("Every step to the device can change a prediction", { placeholder: "title" });
const steps = ["Trained model", "Compiler rewrites", "Quantise / prune", "Code generation", "C++ flags", "Device"];
steps.forEach((t, i) => {
  const x = 0.5 + i * 1.47;
  hex(s, x, 1.3, 1.2, i === 5 ? DEEP : [BLUE, TEAL][i % 2], { text: t, size: 12 });
  if (i < 5) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + 1.22, y: 1.7, w: 0.22, h: 0.22, fill: { color: GREY }, line: { type: "none" } });
});
const probs = [
  ["Testing", "says “no bug found” — never a proof that none exists."],
  ["Verifiers", "say “unknown” — even when a real bug is present, with no input to show it."],
  ["The gap", "no tool both finds the bug and proves its absence, soundly for float32 arithmetic."],
];
probs.forEach(([h, d], i) => {
  const x = 0.5 + i * 2.95;
  card(s, x, 2.75, 2.7, 1.95, i === 2 ? C.accent3 : C.background2);
  s.addText(h, body({ x: x + 0.2, y: 2.92, w: 2.3, h: 0.35, fontSize: 18, bold: true, color: i === 2 ? C.accent4 : C.text2 }));
  s.addText(d, body({ x: x + 0.2, y: 3.35, w: 2.3, h: 1.2, fontSize: 14 }));
});
s.addNotes("A neural network reaches a device through several transformations. Each can silently change outputs. Tests can only fail to find bugs; verifiers often return unknown. We want both answers from one tool.");

// ---------------------------------------------------------------- 4. three verdicts
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Approach" });
corner(s);
s.addText("One question, three honest answers", { placeholder: "title" });
const verdicts = [
  ["✕", RED, "REFUTED", "A replayable input where the outputs differ, localised to the first IR value that diverges."],
  ["✓", DEEP, "CERTIFIED", "A proof: outputs bit-identical (or within ε) for every input in the deployment domain — sound for float32."],
  ["?", BLUE, "UNDECIDED", "The share of the domain already proven, so you know exactly what is left."],
];
verdicts.forEach(([sym, col, h, d], i) => {
  const x = 0.5 + i * 2.95;
  card(s, x, 1.3, 2.7, 2.55);
  hex(s, x + 0.95, 1.5, 0.8, col, { text: sym, size: 22 });
  s.addText(h, body({ x: x + 0.2, y: 2.3, w: 2.3, h: 0.35, fontSize: 18, bold: true, color: C.text2, align: "center" }));
  s.addText(d, body({ x: x + 0.2, y: 2.7, w: 2.3, h: 1.1, fontSize: 13 }));
});
card(s, 0.5, 4.1, 8.6, 0.85, C.accent3);
s.addText([
  { text: "One command: ", options: { bold: true, color: C.accent4 } },
  { text: "temper model.mforge --data test.csv", options: { fontFace: "Courier New", bold: true } },
  { text: "  →  dossier.json + CI exit codes (0 ship, 3 blocked)" },
], body({ x: 0.7, y: 4.25, w: 8.2, h: 0.55, fontSize: 14, valign: "middle" }));

// ---------------------------------------------------------------- 5. how it works
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Approach" });
corner(s);
s.addText("How Temper works: falsify, then certify", { placeholder: "title" });
const stages = [
  ["1", "Static proof", "Canonical form + exactness bit: many rewrites proven with 0 evaluations"],
  ["2", "Guardian-APC+", "Falsifier: inputs on every ReLU kink, ordered by activation-pattern coverage"],
  ["3", "Branch & bound", "Certificate-guided: undecided boxes produce witnesses (Newton steps)"],
];
stages.forEach(([n, h, d], i) => {
  const x = 0.5 + i * 2.95;
  hex(s, x, 1.25, 0.75, [BLUE, TEAL, DEEP][i], { text: n, size: 18 });
  s.addText(h, body({ x: x + 0.85, y: 1.3, w: 1.8, h: 0.55, fontSize: 16, bold: true, color: C.text2, valign: "middle" }));
  s.addText(d, body({ x, y: 1.98, w: 2.6, h: 0.85, fontSize: 13 }));
  if (i < 2) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + 2.65, y: 1.48, w: 0.25, h: 0.25, fill: { color: GREY }, line: { type: "none" } });
});
const cols = [
  ["Falsifier — finds bugs", [
    "Newton on the exact gradient places inputs on ReLU kinks at any depth",
    "Focuses on what the rewrite changed, plus near-miss search",
    "99.2% of observable faults with 128 inputs vs 75.0% random (p < 1e-4)",
  ]],
  ["Certifier — proves absence", [
    "Float32-sound symbolic bounds (dot-product rounding γₙ)",
    "Lockstep Δ = candidate − original with an exactness bit per value",
    "Divergence sources tell the falsifier where bugs can live",
  ]],
];
cols.forEach(([h, list], i) => {
  const x = 0.5 + i * 4.4;
  card(s, x, 3.0, 4.2, 2.0);
  s.addText(h, body({ x: x + 0.2, y: 3.12, w: 3.8, h: 0.35, fontSize: 15, bold: true, color: C.accent4 }));
  s.addText(list.map((t, k) => ({ text: t, options: { bullet: true, breakLine: k < list.length - 1 } })),
    body({ x: x + 0.2, y: 3.5, w: 3.85, h: 1.45, fontSize: 12, paraSpaceAfter: 4 }));
});

// ---------------------------------------------------------------- 6. headline metrics
pres.addSection({ title: "Results" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
corner(s);
s.addText("Headline metrics: 2,352 transformations", { placeholder: "title" });
const stats = [
  ["2,057 / 2,058", "observable faults refuted"],
  ["100%", "benign rewrites certified bit-exact"],
  ["99.7%", "of all cases decided"],
  ["0", "unsound verdicts"],
];
stats.forEach(([n, l], i) => {
  const x = 0.5 + i * 2.17;
  card(s, x, 1.2, 2.0, 1.15);
  s.addText(n, body({ x: x + 0.1, y: 1.27, w: 1.8, h: 0.6, fontSize: i === 0 ? 22 : 30, bold: true, color: C.accent4, align: "center", valign: "middle" }));
  s.addText(l, body({ x: x + 0.1, y: 1.88, w: 1.8, h: 0.4, fontSize: 11, align: "center" }));
});
s.addChart(pres.charts.BAR, [
  { name: "Faults refuted (%)", labels: ["Falsifier only", "Certifier only", "Temper"], values: [96.7, 82.3, 99.95] },
  { name: "All decided (%)", labels: ["Falsifier only", "Certifier only", "Temper"], values: [84.6, 84.2, 99.7] },
], {
  x: 0.5, y: 2.55, w: 5.6, h: 2.55, barDir: "col", barGrouping: "clustered",
  chartColors: [HEX.accent1, HEX.accent2], showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 9,
  dataLabelFormatCode: "0.0#", dataLabelColor: HEX.dk1, dataLabelFontFace: "+mn-lt",
  valAxisMinVal: 0, valAxisMaxVal: 110, valAxisHidden: true, valGridLine: { style: "none" },
  catAxisLabelColor: HEX.dk1, catAxisLabelFontSize: 11, catAxisLabelFontFace: "+mn-lt",
  showLegend: true, legendPos: "t", legendFontSize: 10, legendFontFace: "+mn-lt", legendColor: HEX.dk1,
});
card(s, 6.35, 2.55, 2.75, 2.5, C.accent3);
s.addText("Evaluations per case", body({ x: 6.55, y: 2.68, w: 2.4, h: 0.3, fontSize: 14, bold: true, color: C.accent4 }));
[["Falsifier only", "23.5"], ["Certifier only", "101.8"], ["Temper", "11.3"]].forEach(([k, v], i) => {
  s.addText(k, body({ x: 6.55, y: 3.08 + i * 0.4, w: 1.6, h: 0.3, fontSize: 13, bold: i === 2 }));
  s.addText(v, body({ x: 8.1, y: 3.08 + i * 0.4, w: 0.8, h: 0.3, fontSize: 13, bold: true, align: "right", color: i === 2 ? C.accent4 : C.text1 }));
});
s.addText("54 faults no random input exposed were found only by certificate-guided witnesses.",
  body({ x: 6.55, y: 4.32, w: 2.4, h: 0.65, fontSize: 11 }));
s.addNotes("2,064 injected faults, 96 negative controls, 192 real optimisation passes. Temper refutes all but one observable fault, certifies every benign rewrite bit-exact, and is the cheapest of the three.");

// ---------------------------------------------------------------- 7. real models
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
corner(s);
s.addText("Real models and real compiler-bug classes", { placeholder: "title" });
s.addChart(pres.charts.BAR, [
  { name: "Faults detected (of 71)", labels: ["Test-set check", "1,000 random inputs", "Temper"], values: [62, 66, 71] },
], {
  x: 0.5, y: 1.15, w: 4.5, h: 2.75, barDir: "bar", chartColors: [HEX.accent2, HEX.accent2, HEX.accent4],
  showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11, dataLabelColor: HEX.dk1, dataLabelFontFace: "+mn-lt",
  valAxisMinVal: 0, valAxisMaxVal: 80, valAxisHidden: true, valGridLine: { style: "none" },
  catAxisLabelColor: HEX.dk1, catAxisLabelFontSize: 11, catAxisLabelFontFace: "+mn-lt", showLegend: false,
  showTitle: true, title: "Observable faults detected (of 71)", titleFontSize: 13, titleColor: HEX.dk2, titleFontFace: "+mn-lt",
});
s.addText("80 faults from 10 bug classes in the Shen et al. (FSE’21) compiler-bug study — layout, fusion, broadcast, loop bounds, activation lowering, quantisation rounding/overflow — on 8 trained classifiers.",
  body({ x: 0.5, y: 4.0, w: 4.5, h: 0.95, fontSize: 12 }));
const facts = [
  ["9 bugs", "proven harmless on the deployment domain, 0 wrong proofs"],
  ["40 compression cases", "12 certified low-risk, 25 refuted (7 kept full test accuracy), 3 undecided"],
  ["8 / 8 binaries", "bit-exact at -O0/-O2; FMA and fast-math move scores < 1e-6"],
  ["1 real bug", "found in the generated parser (subnormal inputs rejected), fixed"],
];
facts.forEach(([h, d], i) => {
  const y = 1.2 + i * 0.95;
  hex(s, 5.35, y + 0.08, 0.5, [DEEP, BLUE, TEAL, RED][i]);
  s.addText(h, body({ x: 6.0, y, w: 3.0, h: 0.3, fontSize: 15, bold: true, color: C.text2 }));
  s.addText(d, body({ x: 6.0, y: y + 0.3, w: 3.0, h: 0.55, fontSize: 12 }));
});
s.addNotes("Datasets: Iris, Wine, Breast Cancer, Digits; two MLP sizes each. Temper catches all 71 observable faults where standard test-set checking catches 62.");

// ---------------------------------------------------------------- 8. lifecycle
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
corner(s);
s.addText("After the verdict: what to do next", { placeholder: "title" });
const life = [
  ["Certified repair", "--repair", "35.7%", "of weights kept (Wine 32×16) vs 50% for the best certified uniform format", "8 / 8 mixed-precision plans certified low-risk"],
  ["Culprit localisation", "--compare", "8 / 8", "models: the exact harmful weight found among 21–24 edits", "8–10 decisions each; revert certified on 5 / 8"],
  ["Incremental proofs", "--reuse-cert", "≈ 2×", "less work: 761 → 384 boxes when re-certifying a new version", "Old bug back? Caught in 1 evaluation (vs 49–2,487 boxes)"],
];
life.forEach(([h, flag, n, l, extra], i) => {
  const x = 0.5 + i * 2.95;
  card(s, x, 1.2, 2.7, 3.8);
  hex(s, x + 0.2, 1.38, 0.5, [BLUE, TEAL, DEEP][i]);
  s.addText(h, body({ x: x + 0.8, y: 1.38, w: 1.8, h: 0.45, fontSize: 15, bold: true, color: C.text2, valign: "middle" }));
  s.addText(flag, body({ x: x + 0.2, y: 1.95, w: 2.3, h: 0.25, fontSize: 11, fontFace: "Courier New", color: C.accent4 }));
  s.addText(n, body({ x: x + 0.2, y: 2.25, w: 2.3, h: 0.7, fontSize: 36, bold: true, color: C.accent4 }));
  s.addText(l, body({ x: x + 0.2, y: 2.95, w: 2.3, h: 0.95, fontSize: 12 }));
  s.addText(extra, body({ x: x + 0.2, y: 3.95, w: 2.3, h: 0.9, fontSize: 12, italic: true, color: C.text2 }));
});
s.addNotes("Inspired by HAWQ (mixed precision), delta debugging and DeepFault (localisation), and IVAN (incremental verification) — but every result here comes with a proof.");

// ---------------------------------------------------------------- 9. novelty & next steps
pres.addSection({ title: "Conclusion" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Conclusion" });
corner(s);
s.addText("Novelty, honest limits and next steps", { placeholder: "title" });
const concl = [
  ["What is new", DEEP, [
    "One falsify-or-certify loop with three verdicts",
    "Float32-sound differential proofs with an exactness bit",
    "Certificate-guided witnesses",
    "Proof-backed repair, localisation and reuse",
  ]],
  ["Honest limits", RED, [
    "Small MLPs (Iris, Wine, Breast Cancer, Digits)",
    "No head-to-head yet with ReluDiff or α,β-CROWN",
    "Bug classes modelled on a study, not reproduced issues",
  ]],
  ["Next steps", BLUE, [
    "Scale to CNNs and larger models",
    "Baseline comparison on the same benchmark",
    "Reproduce real TVM / ONNX bug reports",
    "Target a workshop, then a main-track venue",
  ]],
];
concl.forEach(([h, col, list], i) => {
  const x = 0.5 + i * 2.95;
  card(s, x, 1.2, 2.7, 2.85);
  hex(s, x + 0.2, 1.36, 0.45, col);
  s.addText(h, body({ x: x + 0.78, y: 1.36, w: 1.8, h: 0.4, fontSize: 16, bold: true, color: C.text2, valign: "middle" }));
  s.addText(list.map((t, k) => ({ text: t, options: { bullet: true, breakLine: k < list.length - 1 } })),
    body({ x: x + 0.2, y: 1.95, w: 2.35, h: 2.0, fontSize: 12, paraSpaceAfter: 5 }));
});
card(s, 0.5, 4.25, 8.6, 0.8, C.accent3);
s.addText([
  { text: "Live demo: ", options: { bold: true, color: C.accent4 } },
  { text: "./samples/run_demo.sh", options: { fontFace: "Courier New", bold: true } },
  { text: "  — compile, catch a broken rewrite, audit, compare, repair.  Thank you!" },
], body({ x: 0.7, y: 4.38, w: 8.2, h: 0.55, fontSize: 14, valign: "middle" }));

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
