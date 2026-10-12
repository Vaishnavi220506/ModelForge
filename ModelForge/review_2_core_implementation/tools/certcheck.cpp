// modelforge_certcheck: independent checker for ForgeCert certificates.
// Author: Vaishnavi
//
// Proof-carrying code for neural networks: the compiler (untrusted, large:
// search, refinement, splitting) ships integer-only C plus a certificate; this
// small program re-establishes every claim from two inputs only:
//   * the source model (.mforge text), and
//   * the emitted C file (model_q.c), whose data arrays it parses and whose
//     code it compares against its own copy of the code template.
// It shares no code with the compiler (no ModelForge headers are included).
//
// Checked claims
//   1. the C code is exactly the expected template for the metadata
//   2. the input quantiser never clamps and no accumulator partial sum leaves
//      its declared type, anywhere in the input domain
//   3. no requantised value saturates, anywhere in the input domain
//   4. the claimed bound on |integer logit - float logit| holds on the domain
//   5. for every region in the certificate, following its split tree, the
//      float model is constant on every leaf and the integer program returns
//      the same class there
// Optional: --expect <raw.csv> <inputs.csv> compares the checker's own
// integer semantics with the output of the compiled binary (generated_model_q
// --raw), closing the loop between the proof's semantics and the real C
// compiler.
//
// Usage: modelforge_certcheck <artifact-dir> <model.mforge> [--expect raw.csv inputs.csv]

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------- JSON ----
struct Json {
    enum Kind { Null, Bool, Number, String, Array, Object } kind = Null;
    double number = 0;
    bool boolean = false;
    std::string text;
    std::vector<Json> items;
    std::map<std::string, Json> fields;
    const Json& operator[](const std::string& key) const {
        static const Json none;
        const auto it = fields.find(key);
        return it == fields.end() ? none : it->second;
    }
    std::vector<double> numbers() const {
        std::vector<double> out;
        for (const auto& item : items) out.push_back(item.number);
        return out;
    }
};

struct JsonParser {
    const std::string& s;
    std::size_t p = 0;
    bool ok = true;
    void space() {
        while (p < s.size() && std::isspace(static_cast<unsigned char>(s[p]))) ++p;
    }
    Json parse() {
        space();
        Json v;
        if (p >= s.size()) {
            ok = false;
            return v;
        }
        const char c = s[p];
        if (c == '{') {
            v.kind = Json::Object;
            ++p;
            space();
            if (s[p] == '}') {
                ++p;
                return v;
            }
            while (ok) {
                space();
                const Json key = parse();
                space();
                if (key.kind != Json::String || s[p] != ':') {
                    ok = false;
                    break;
                }
                ++p;
                v.fields[key.text] = parse();
                space();
                if (s[p] == ',') {
                    ++p;
                } else if (s[p] == '}') {
                    ++p;
                    break;
                } else {
                    ok = false;
                }
            }
        } else if (c == '[') {
            v.kind = Json::Array;
            ++p;
            space();
            if (s[p] == ']') {
                ++p;
                return v;
            }
            while (ok) {
                v.items.push_back(parse());
                space();
                if (s[p] == ',') {
                    ++p;
                } else if (s[p] == ']') {
                    ++p;
                    break;
                } else {
                    ok = false;
                }
            }
        } else if (c == '"') {
            v.kind = Json::String;
            ++p;
            while (p < s.size() && s[p] != '"') {
                if (s[p] == '\\') ++p;
                v.text += s[p++];
            }
            ++p;
        } else if (s.compare(p, 4, "true") == 0 || s.compare(p, 5, "false") == 0) {
            v.kind = Json::Bool;
            v.boolean = s[p] == 't';
            p += v.boolean ? 4 : 5;
        } else if (s.compare(p, 4, "null") == 0) {
            p += 4;
        } else {
            v.kind = Json::Number;
            char* end = nullptr;
            v.number = std::strtod(s.c_str() + p, &end);
            if (end == s.c_str() + p) ok = false;
            p = static_cast<std::size_t>(end - s.c_str());
        }
        return v;
    }
};

std::string readFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

// -------------------------------------------------- source model (.mforge)
struct Layer {
    std::size_t in = 0, out = 0;
    std::vector<double> w;  // out x in
    std::vector<double> b;
    bool relu = false;
};

bool loadReference(const std::string& path, std::vector<Layer>& layers, std::string& error) {
    std::ifstream in(path);
    if (!in) {
        error = "cannot open " + path;
        return false;
    }
    struct Node {
        std::string op;
        std::vector<std::string> inputs;
        bool transB = false;
        bool unit = true;  // alpha = beta = 1
    };
    std::map<std::string, std::pair<std::vector<long long>, std::vector<double>>> tensors;
    std::map<std::string, Node> producer;
    std::string inputName, outputName, line;
    while (std::getline(in, line)) {
        std::istringstream words(line);
        std::vector<std::string> t;
        std::string word;
        while (words >> word) t.push_back(word);
        if (t.empty() || t[0][0] == '#') continue;
        if (t[0] == "input" && t.size() > 1) inputName = t[1];
        if (t[0] == "output" && t.size() > 1) outputName = t[1];
        if (t[0] == "tensor" && t.size() >= 5) {
            std::vector<long long> shape;
            std::stringstream dims(t[3]);
            std::string d;
            while (std::getline(dims, d, ',')) shape.push_back(std::atoll(d.c_str()));
            std::vector<double> values;
            std::stringstream list(t[4].substr(t[4].find('=') + 1));
            while (std::getline(list, d, ',')) values.push_back(static_cast<double>(std::strtof(d.c_str(), nullptr)));
            tensors[t[1]] = {shape, values};
        }
        if (t[0] == "node") {
            const auto arrow = std::find(t.begin(), t.end(), "->");
            if (arrow == t.end() || arrow + 1 == t.end() || t.size() < 4) continue;
            Node node;
            node.op = t[2];
            node.inputs.assign(t.begin() + 3, arrow);
            for (auto it = arrow + 2; it != t.end(); ++it) {
                if (*it == "transB=1") node.transB = true;
                if ((it->rfind("alpha=", 0) == 0 || it->rfind("beta=", 0) == 0) &&
                    std::strtod(it->c_str() + it->find('=') + 1, nullptr) != 1.0) {
                    node.unit = false;
                }
            }
            producer[*(arrow + 1)] = node;
        }
    }
    std::vector<Layer> reversed;
    std::string current = outputName;
    bool pendingRelu = false;
    bool head = true;
    const auto weightFrom = [&](const std::string& name, bool transposed, Layer& layer) {
        const auto it = tensors.find(name);
        if (it == tensors.end() || it->second.first.size() != 2) return false;
        const auto rows = static_cast<std::size_t>(it->second.first[0]);
        const auto cols = static_cast<std::size_t>(it->second.first[1]);
        layer.in = transposed ? cols : rows;
        layer.out = transposed ? rows : cols;
        layer.w.assign(layer.in * layer.out, 0.0);
        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < cols; ++c) {
                const std::size_t o = transposed ? r : c, i = transposed ? c : r;
                layer.w[o * layer.in + i] = it->second.second[r * cols + c];
            }
        }
        layer.b.assign(layer.out, 0.0);
        return true;
    };
    const auto biasFrom = [&](const std::string& name, Layer& layer) {
        const auto it = tensors.find(name);
        if (it == tensors.end() || it->second.second.size() != layer.out) return false;
        for (std::size_t o = 0; o < layer.out; ++o) layer.b[o] += it->second.second[o];
        return true;
    };
    while (current != inputName) {
        const auto found = producer.find(current);
        if (found == producer.end() || found->second.inputs.empty()) {
            error = "value '" + current + "' has no producer";
            return false;
        }
        const Node& node = found->second;
        if ((node.op == "Softmax" || node.op == "Sigmoid") && head && reversed.empty() && !pendingRelu) {
            head = false;
            current = node.inputs[0];
        } else if (node.op == "Relu" && !reversed.empty() && !pendingRelu) {
            pendingRelu = true;
            current = node.inputs[0];
        } else if (node.op == "Gemm" && node.unit && node.inputs.size() >= 2) {
            Layer layer;
            if (!weightFrom(node.inputs[1], node.transB, layer) ||
                (node.inputs.size() == 3 && !biasFrom(node.inputs[2], layer))) {
                error = "bad Gemm operands";
                return false;
            }
            layer.relu = pendingRelu;
            pendingRelu = false;
            head = false;
            reversed.push_back(layer);
            current = node.inputs[0];
        } else if (node.op == "Add" && node.inputs.size() == 2) {
            const bool firstConst = tensors.count(node.inputs[0]) > 0;
            const std::string other = node.inputs[firstConst ? 1 : 0];
            const auto mm = producer.find(other);
            if (mm == producer.end() || mm->second.op != "MatMul" || mm->second.inputs.size() != 2) {
                error = "unsupported Add";
                return false;
            }
            Layer layer;
            if (!weightFrom(mm->second.inputs[1], false, layer) || !biasFrom(node.inputs[firstConst ? 0 : 1], layer)) {
                error = "bad MatMul/Add operands";
                return false;
            }
            layer.relu = pendingRelu;
            pendingRelu = false;
            head = false;
            reversed.push_back(layer);
            current = mm->second.inputs[0];
        } else if (node.op == "MatMul" && node.inputs.size() == 2) {
            Layer layer;
            if (!weightFrom(node.inputs[1], false, layer)) {
                error = "bad MatMul operands";
                return false;
            }
            layer.relu = pendingRelu;
            pendingRelu = false;
            head = false;
            reversed.push_back(layer);
            current = node.inputs[0];
        } else {
            error = "unsupported operator " + node.op;
            return false;
        }
    }
    if (reversed.empty() || pendingRelu || reversed.front().relu) {
        error = "unsupported network shape";
        return false;
    }
    layers.assign(reversed.rbegin(), reversed.rend());
    Layer& last = layers.back();
    if (last.out == 1) {  // a single logit z decides z > 0
        std::vector<double> w(2 * last.in, 0.0);
        std::copy(last.w.begin(), last.w.end(), w.begin() + static_cast<long>(last.in));
        last.w = w;
        last.b = {0.0, last.b[0]};
        last.out = 2;
    }
    return true;
}

// ---------------------------------------------------- artifact (model_q.c)
struct Array {
    std::string type;
    std::vector<double> values;  // integers are exact up to 2^53
};

std::map<std::string, Array> parseArrays(const std::string& source) {
    std::map<std::string, Array> arrays;
    std::size_t p = 0;
    const std::string key = "static const ";
    while ((p = source.find(key, p)) != std::string::npos) {
        p += key.size();
        std::istringstream head(source.substr(p, 80));
        std::string type, declarator;
        head >> type >> declarator;
        const auto bracket = declarator.find('[');
        const auto open = source.find('{', p);
        const auto semicolon = source.find(';', p);
        if (bracket == std::string::npos || open == std::string::npos || open > semicolon) continue;
        const auto close = source.find('}', open);
        Array array;
        array.type = type;
        const char* cursor = source.c_str() + open + 1;
        const char* end = source.c_str() + close;
        while (cursor < end) {
            char* next = nullptr;
            const double v = std::strtod(cursor, &next);
            if (next == cursor) {
                ++cursor;
                continue;
            }
            array.values.push_back(v);
            cursor = next;
        }
        arrays[declarator.substr(0, bracket)] = array;
        p = close;
    }
    return arrays;
}

const char* cType(int bits, bool isUnsigned) {
    if (bits <= 8) return isUnsigned ? "uint8_t" : "int8_t";
    if (bits <= 16) return isUnsigned ? "uint16_t" : "int16_t";
    if (bits <= 32) return isUnsigned ? "uint32_t" : "int32_t";
    return isUnsigned ? "uint64_t" : "int64_t";
}

bool fitsType(const std::string& type, double v) {
    const std::map<std::string, std::pair<double, double>> ranges = {
        {"int8_t", {-128, 127}},          {"uint8_t", {0, 255}},
        {"int16_t", {-32768, 32767}},     {"uint16_t", {0, 65535}},
        {"int32_t", {-2147483648.0, 2147483647.0}}, {"uint32_t", {0, 4294967295.0}},
        {"int64_t", {-9.2e18, 9.2e18}}};
    const auto it = ranges.find(type);
    return it != ranges.end() && v >= it->second.first && v <= it->second.second && v == std::floor(v);
}

struct QLayer {
    std::size_t in = 0, out = 0;
    bool relu = false, outUnsigned = false;
    int wbits = 8, accBits = 32, outBits = 8;
    std::vector<long long> w, b, m;
    std::vector<int> s;
    std::vector<double> outScale;
    long long outMax = 0, outMin = 0;
};

struct Program {
    std::size_t inputs = 0, classes = 0;
    int inputBits = 8;
    long long inMax = 0;
    std::vector<float> inv;  // input inverse scales as the C code uses them
    std::vector<QLayer> layers;
};

// The code template, as the checker expects it to be emitted.
std::string expectedLayer(const Program& p, std::size_t l) {
    const QLayer& q = p.layers[l];
    const char* inT = l == 0 ? cType(p.inputBits, false) : cType(p.layers[l - 1].outBits, p.layers[l - 1].outUnsigned);
    const char* outT = cType(q.outBits, q.outUnsigned);
    const char* accT = cType(q.accBits, false);
    std::ostringstream t;
    t << "static void mf_layer" << l << "(const " << inT << " *x, " << outT << " *y) {\n"
      << "    for (int o = 0; o < " << q.out << "; ++o) {\n"
      << "        " << accT << " acc = (" << accT << ")MF_B" << l << "[o];\n"
      << "        for (int i = 0; i < " << q.in << "; ++i) {\n";
    if (q.wbits <= 4) {
        t << "            acc = (" << accT << ")(acc + (" << accT << ")mf_nibble(MF_W" << l << ", o * " << q.in
          << " + i) * (" << accT << ")x[i]);\n";
    } else {
        t << "            acc = (" << accT << ")(acc + (" << accT << ")MF_W" << l << "[o * " << q.in << " + i] * ("
          << accT << ")x[i]);\n";
    }
    t << "        }\n";
    if (q.relu) t << "        if (acc < 0) acc = 0;\n";
    t << "        int64_t q = mf_round_shift((int64_t)acc * MF_M" << l << "[o], MF_S" << l << "[o]);\n"
      << "        if (q > " << q.outMax << ") q = " << q.outMax << ";  /* saturation: proved unreachable on the domain */\n"
      << "        if (q < " << q.outMin << ") q = " << q.outMin << ";\n"
      << "        y[o] = (" << outT << ")q;\n    }\n}\n";
    return t.str();
}

std::vector<std::string> expectedFragments(const Program& p) {
    std::vector<std::string> parts{
        "static int64_t mf_round_shift(int64_t t, int s) {\n"
        "    int64_t u = t + ((int64_t)1 << (s - 1));\n"
        "    return u >= 0 ? (u >> s) : -((-u + (((int64_t)1 << s) - 1)) >> s);\n}\n",
        "static int8_t mf_nibble(const uint8_t *packed, int k) {\n"
        "    int v = (k & 1) ? (packed[k >> 1] >> 4) : (packed[k >> 1] & 15);\n"
        "    return (int8_t)(v >= 8 ? v - 16 : v);\n}\n"};
    for (std::size_t l = 0; l < p.layers.size(); ++l) parts.push_back(expectedLayer(p, l));
    std::ostringstream quant;
    quant << "void mf_quantize_input(const float *x, mf_input_t *q) {\n"
          << "    for (int i = 0; i < MF_INPUTS; ++i) {\n"
          << "        float t = x[i] * MF_INPUT_INV_SCALE[i];\n"
          << "        float r = t >= 0.0f ? t + 0.5f : t - 0.5f;\n"
          << "        if (r > 1.0e9f) r = 1.0e9f;\n        if (r < -1.0e9f) r = -1.0e9f;\n"
          << "        long long v = (long long)r;\n"
          << "        if (v > " << p.inMax << ") v = " << p.inMax << ";\n"
          << "        if (v < " << -p.inMax << ") v = " << -p.inMax << ";\n"
          << "        q[i] = (mf_input_t)v;\n    }\n}\n";
    parts.push_back(quant.str());
    std::ostringstream infer;
    infer << "int mf_infer_q(const mf_input_t *q, mf_logit_t *logits) {\n";
    for (std::size_t l = 0; l + 1 < p.layers.size(); ++l) {
        infer << "    " << cType(p.layers[l].outBits, p.layers[l].outUnsigned) << " t" << l << "["
              << p.layers[l].out << "];\n";
    }
    infer << "    mf_logit_t out[MF_CLASSES];\n";
    for (std::size_t l = 0; l < p.layers.size(); ++l) {
        infer << "    mf_layer" << l << "(" << (l == 0 ? std::string("q") : "t" + std::to_string(l - 1)) << ", "
              << (l + 1 == p.layers.size() ? std::string("out") : "t" + std::to_string(l)) << ");\n";
    }
    infer << "    int best = 0;\n    for (int c = 0; c < MF_CLASSES; ++c) {\n"
          << "        if (logits) logits[c] = out[c];\n        if (out[c] > out[best]) best = c;\n    }\n"
          << "    return best;\n}\n";
    parts.push_back(infer.str());
    return parts;
}

std::string loadProgram(const std::string& source, Program& p) {
    const auto arrays = parseArrays(source);
    const auto get = [&](const std::string& name) -> const Array* {
        const auto it = arrays.find(name);
        return it == arrays.end() ? nullptr : &it->second;
    };
    const Array* meta = get("MF_META");
    if (!meta || meta->values.size() < 5 || meta->values[0] != 1) return "missing or unknown MF_META";
    p.inputs = static_cast<std::size_t>(meta->values[1]);
    p.classes = static_cast<std::size_t>(meta->values[2]);
    p.inputBits = static_cast<int>(meta->values[3]);
    p.inMax = (1LL << (p.inputBits - 1)) - 1;
    const auto L = static_cast<std::size_t>(meta->values[4]);
    if (meta->values.size() != 5 + 7 * L) return "MF_META has the wrong length";
    const Array* inv = get("MF_INPUT_INV_SCALE");
    if (!inv || inv->values.size() != p.inputs || inv->type != "float") return "bad MF_INPUT_INV_SCALE";
    for (double v : inv->values) {
        if (!(v > 0)) return "input scales must be positive";
        p.inv.push_back(static_cast<float>(v));
    }
    std::size_t width = p.inputs;
    for (std::size_t l = 0; l < L; ++l) {
        const double* m = &meta->values[5 + 7 * l];
        QLayer q;
        q.in = static_cast<std::size_t>(m[0]);
        q.out = static_cast<std::size_t>(m[1]);
        q.relu = m[2] != 0;
        q.wbits = static_cast<int>(m[3]);
        q.accBits = static_cast<int>(m[4]);
        q.outBits = static_cast<int>(m[5]);
        q.outUnsigned = m[6] != 0;
        if (q.in != width || (q.accBits != 16 && q.accBits != 32 && q.accBits != 64) || q.wbits < 2 ||
            q.wbits > 16 || q.outBits < 2 || q.outBits > 16 || (l + 1 == L && (q.relu || q.outUnsigned))) {
            return "inconsistent layer metadata";
        }
        width = q.out;
        q.outMax = q.outUnsigned ? (1LL << q.outBits) - 1 : (1LL << (q.outBits - 1)) - 1;
        q.outMin = q.outUnsigned ? 0 : -q.outMax;
        const std::string id = std::to_string(l);
        const Array* w = get("MF_W" + id);
        const Array* b = get("MF_B" + id);
        const Array* mm = get("MF_M" + id);
        const Array* s = get("MF_S" + id);
        const Array* sc = get("MF_SCALE_" + id);
        if (!w || !b || !mm || !s || !sc) return "missing arrays for layer " + id;
        const long long wmax = (1LL << (q.wbits - 1)) - 1;
        if (q.wbits <= 4) {
            if (w->type != "uint8_t" || w->values.size() != (q.in * q.out + 1) / 2) return "bad packed MF_W" + id;
            for (std::size_t k = 0; k < q.in * q.out; ++k) {
                const int byte = static_cast<int>(w->values[k / 2]);
                const int v = (k % 2) ? (byte >> 4) : (byte & 15);
                q.w.push_back(v >= 8 ? v - 16 : v);
            }
        } else {
            if (w->type != cType(q.wbits, false) || w->values.size() != q.in * q.out) return "bad MF_W" + id;
            for (double v : w->values) q.w.push_back(static_cast<long long>(v));
        }
        for (std::size_t k = 0; k < w->values.size(); ++k) {
            if (!fitsType(w->type, w->values[k])) return "MF_W" + id + " value does not fit its type";
        }
        for (long long v : q.w) {
            if (std::llabs(v) > wmax) return "MF_W" + id + " exceeds its bit width";
        }
        if (b->values.size() != q.out || mm->values.size() != q.out || s->values.size() != q.out ||
            sc->values.size() != q.out || mm->type != "int32_t" || s->type != "uint8_t" ||
            b->type != cType(q.accBits, false)) {
            return "bad bias/multiplier/shift/scale arrays for layer " + id;
        }
        for (std::size_t o = 0; o < q.out; ++o) {
            if (!fitsType(b->type, b->values[o]) || !fitsType("int32_t", mm->values[o]) || mm->values[o] < 0 ||
                s->values[o] < 1 || s->values[o] > 62 || !(sc->values[o] > 0)) {
                return "out-of-range constant in layer " + id;
            }
            q.b.push_back(static_cast<long long>(b->values[o]));
            q.m.push_back(static_cast<long long>(mm->values[o]));
            q.s.push_back(static_cast<int>(s->values[o]));
            q.outScale.push_back(sc->values[o]);
        }
        p.layers.push_back(q);
    }
    if (width != p.classes) return "class count does not match the last layer";
    for (double v : p.layers.back().outScale) {
        if (v != p.layers.back().outScale[0]) return "logits must share one scale";
    }
    return "";
}

// ------------------------------------------------- affine forms (checker)
struct F {
    double c = 0;
    std::vector<double> a;
};
double rad(const F& f) {
    double r = 0;
    for (double v : f.a) r += std::fabs(v);
    return r;
}
double lo(const F& f) {
    const double r = rad(f);
    return f.c - r - 1e-12 * (std::fabs(f.c) + r) - 1e-300;
}
double hi(const F& f) {
    const double r = rad(f);
    return f.c + r + 1e-12 * (std::fabs(f.c) + r) + 1e-300;
}
void add(F& d, double k, const F& s) {
    if (k == 0) return;
    d.c += k * s.c;
    if (d.a.size() < s.a.size()) d.a.resize(s.a.size(), 0);
    for (std::size_t i = 0; i < s.a.size(); ++i) d.a[i] += k * s.a[i];
}
void sym(F& f, std::size_t k, double v) {
    if (f.a.size() <= k) f.a.resize(k + 1, 0);
    f.a[k] += v;
}

struct Pair {
    F r, e;  // float model value, integer program error
};

struct Analysis {
    std::vector<Pair> logits;
    std::vector<std::vector<long long>> qLo, qHi;  // per layer input ranges (layer 0 = input)
    std::string failure;
};

// Same mathematics as the compiler, written independently: affine forms for
// the float model (DeepZ ReLU) and for the integer program's error.
Analysis analyse(const std::vector<Layer>& net, const Program& p, const std::vector<double>& blo,
                 const std::vector<double>& bhi) {
    Analysis out;
    std::size_t symbols = 0;
    std::vector<Pair> v(p.inputs);
    std::vector<long long> qlo(p.inputs), qhi(p.inputs);
    std::vector<double> inScale(p.inputs);
    for (std::size_t i = 0; i < p.inputs; ++i) {
        v[i].r.c = 0.5 * (blo[i] + bhi[i]);
        sym(v[i].r, symbols++, 0.5 * (bhi[i] - blo[i]));
    }
    for (std::size_t i = 0; i < p.inputs; ++i) {
        const double inv = static_cast<double>(p.inv[i]);
        const double T = std::max(std::fabs(blo[i]), std::fabs(bhi[i])) * inv;
        const double delta = 0.5 + 2.0 * (T + 1.0) * std::ldexp(1.0, -24);  // float32 multiply and +-0.5
        inScale[i] = 1.0 / inv;
        sym(v[i].e, symbols++, delta / inv);
        qlo[i] = static_cast<long long>(std::ceil(blo[i] * inv - delta));
        qhi[i] = static_cast<long long>(std::floor(bhi[i] * inv + delta));
        if (qlo[i] < -p.inMax || qhi[i] > p.inMax) out.failure = "input quantiser may clamp";
    }
    out.qLo.push_back(qlo);
    out.qHi.push_back(qhi);
    for (std::size_t l = 0; l < p.layers.size(); ++l) {
        const QLayer& q = p.layers[l];
        const Layer& f = net[l];
        // Overflow: every partial sum, starting from the bias.
        const long long accMax = (1LL << (q.accBits - 1)) - 1;
        for (std::size_t o = 0; o < q.out; ++o) {
            double bound = std::fabs(static_cast<double>(q.b[o]));
            for (std::size_t i = 0; i < q.in; ++i) {
                bound += std::fabs(static_cast<double>(q.w[o * q.in + i])) *
                         std::max(std::fabs(static_cast<double>(qlo[i])), std::fabs(static_cast<double>(qhi[i])));
            }
            if (bound > static_cast<double>(accMax)) out.failure = "accumulator may overflow in layer " + std::to_string(l);
            if (bound * static_cast<double>(q.m[o]) + std::ldexp(1.0, q.s[o] - 1) >= std::ldexp(1.0, 63)) {
                out.failure = "64-bit requantisation product may overflow in layer " + std::to_string(l);
            }
        }
        std::vector<Pair> next(q.out);
        for (std::size_t o = 0; o < q.out; ++o) {
            const double k = q.outScale[o] * std::ldexp(static_cast<double>(q.m[o]), -q.s[o]);
            Pair& n = next[o];
            n.r.c = f.b[o];
            n.e.c = k * static_cast<double>(q.b[o]) - f.b[o];
            for (std::size_t i = 0; i < q.in; ++i) {
                const double wf = f.w[o * q.in + i];
                const double wq = k * static_cast<double>(q.w[o * q.in + i]) / inScale[i];
                add(n.r, wf, v[i].r);
                add(n.e, wq - wf, v[i].r);
                add(n.e, wq, v[i].e);
            }
        }
        std::vector<long long> nlo(q.out), nhi(q.out);
        for (std::size_t o = 0; o < q.out; ++o) {
            Pair& n = next[o];
            F qv = n.r;
            add(qv, 1, n.e);
            double ql = lo(qv), qu = hi(qv);
            if (q.relu) {
                const double l0 = lo(n.r), u0 = hi(n.r);
                const auto relax = [&](const F& x, double a, double b) {
                    F y;
                    if (b <= 0) return y;
                    if (a >= 0) return x;
                    const double lambda = b / (b - a), mu = -lambda * a / 2;
                    add(y, lambda, x);
                    y.c += mu;
                    sym(y, symbols++, mu);
                    return y;
                };
                const F rr = relax(n.r, l0, u0);
                const F rq = relax(qv, ql, qu);
                F affine = rq;  // ReLU~(q) - ReLU~(r)
                add(affine, -1, rr);
                const double el = lo(n.e), eu = hi(n.e);
                double a = std::min(0.0, el), b = std::max(0.0, eu);
                if (l0 >= 0) { a = std::max(a, std::max(el, -u0)); b = std::min(b, std::max(eu, -l0)); }
                if (u0 <= 0) { a = std::max(a, std::max(0.0, ql)); b = std::min(b, std::max(0.0, qu)); }
                if (qu <= 0) { a = std::max(a, -std::max(0.0, u0)); b = std::min(b, -std::max(0.0, l0)); }
                if (ql >= 0) { a = std::max(a, el + std::min(l0, 0.0)); b = std::min(b, eu + std::min(u0, 0.0)); }
                if (b < a) a = b = 0.5 * (a + b);
                if (u0 <= 0 && qu <= 0) {
                    n.e = F{};  // both stay zero
                } else if (l0 >= 0 && ql >= 0) {
                    // both pass through: the error is unchanged
                } else if (hi(affine) - lo(affine) <= (b - a) * (1 + 1e-9)) {
                    n.e = affine;
                } else {
                    n.e = F{};
                    n.e.c = 0.5 * (a + b);
                    sym(n.e, symbols++, 0.5 * (b - a));
                }
                n.r = rr;
                ql = std::max(0.0, ql);
                qu = std::max(0.0, qu);
            }
            const double tl = ql / q.outScale[o], tu = qu / q.outScale[o];
            nlo[o] = static_cast<long long>(std::floor(tl + 0.5));
            nhi[o] = static_cast<long long>(std::floor(tu + 0.5));
            if (nhi[o] > q.outMax || nlo[o] < q.outMin) out.failure = "saturation possible in layer " + std::to_string(l);
            sym(n.e, symbols++, 0.5 * q.outScale[o] * (1 + 1e-12));
        }
        v = std::move(next);
        qlo = nlo;
        qhi = nhi;
        inScale = q.outScale;
        out.qLo.push_back(qlo);
        out.qHi.push_back(qhi);
    }
    out.logits = v;
    return out;
}

// The integer program, exactly as the C code computes it.
std::vector<long long> runProgram(const Program& p, const std::vector<float>& x) {
    std::vector<long long> cur(p.inputs);
    for (std::size_t i = 0; i < p.inputs; ++i) {
        const float t = x[i] * p.inv[i];
        float r = t >= 0.0f ? t + 0.5f : t - 0.5f;
        r = std::max(-1.0e9f, std::min(1.0e9f, r));
        long long v = static_cast<long long>(r);
        cur[i] = std::max(-p.inMax, std::min(p.inMax, v));
    }
    for (const auto& q : p.layers) {
        std::vector<long long> next(q.out);
        for (std::size_t o = 0; o < q.out; ++o) {
            long long acc = q.b[o];
            for (std::size_t i = 0; i < q.in; ++i) acc += q.w[o * q.in + i] * cur[i];
            if (q.relu && acc < 0) acc = 0;
            const long long t = acc * q.m[o];
            const long long u = t + (1LL << (q.s[o] - 1));
            long long r = u >= 0 ? (u >> q.s[o]) : -((-u + ((1LL << q.s[o]) - 1)) >> q.s[o]);
            next[o] = std::max(q.outMin, std::min(q.outMax, r));
        }
        cur = next;
    }
    return cur;
}

int verdicts = 0, failures = 0;
void report(bool ok, const std::string& what) {
    ++verdicts;
    if (!ok) ++failures;
    std::cout << "  [" << (ok ? "PASS" : "FAIL") << "] " << what << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3 && argc != 6) {
        std::cerr << "usage: modelforge_certcheck <artifact-dir> <model.mforge> [--expect raw.csv inputs.csv]\n";
        return 2;
    }
    const std::string dir = argv[1];
    std::cout << "ForgeCert independent checker (by Vaishnavi)\n  artifact: " << dir << "/model_q.c\n  model:    "
              << argv[2] << "\n";
    std::vector<Layer> net;
    std::string error;
    if (!loadReference(argv[2], net, error)) {
        std::cout << "  [FAIL] source model: " << error << "\n";
        return 1;
    }
    const std::string source = readFile(dir + "/model_q.c");
    const std::string header = readFile(dir + "/model_q.h");
    const std::string certificateText = readFile(dir + "/certificate.json");
    JsonParser parser{certificateText};
    const Json certificate = parser.parse();
    if (source.empty() || !parser.ok || certificate["format"].text != "forgecert-1") {
        std::cout << "  [FAIL] cannot read model_q.c or certificate.json\n";
        return 1;
    }
    Program program;
    error = loadProgram(source, program);
    report(error.empty(), "artifact arrays parsed and within their declared types" + (error.empty() ? "" : ": " + error));
    if (!error.empty()) return 1;

    // 1. Code template.
    bool templateOk = header.find("typedef " + std::string(cType(program.inputBits, false)) + " mf_input_t;") !=
                          std::string::npos &&
                      header.find("typedef " + std::string(cType(program.layers.back().outBits, false)) +
                                  " mf_logit_t;") != std::string::npos;
    for (const auto& fragment : expectedFragments(program)) templateOk = templateOk && source.find(fragment) != std::string::npos;
    report(templateOk, "C code matches the checker's template for the metadata");

    // Shapes of the source model.
    bool shapes = net.size() == program.layers.size();
    for (std::size_t l = 0; shapes && l < net.size(); ++l) {
        shapes = net[l].in == program.layers[l].in && net[l].out == program.layers[l].out &&
                 net[l].relu == program.layers[l].relu;
    }
    report(shapes, "integer program has the source model's layer structure");
    if (!shapes || !templateOk) return 1;

    // 2-4. Domain-wide claims.
    const auto domainLo = certificate["domain"]["lo"].numbers();
    const auto domainHi = certificate["domain"]["hi"].numbers();
    if (domainLo.size() != program.inputs || domainHi.size() != program.inputs) {
        report(false, "certificate domain has the input dimension");
        return 1;
    }
    const auto global = analyse(net, program, domainLo, domainHi);
    report(global.failure.empty(), "no input clamp, accumulator overflow or saturation on the domain" +
                                       (global.failure.empty() ? "" : ": " + global.failure));
    const auto claimedBits = certificate["claims"]["accumulator_bits"].numbers();
    bool bitsOk = claimedBits.size() == program.layers.size();
    for (std::size_t l = 0; bitsOk && l < claimedBits.size(); ++l) bitsOk = claimedBits[l] == program.layers[l].accBits;
    report(bitsOk, "claimed accumulator widths match the code");
    const auto claimedError = certificate["claims"]["max_abs_logit_error_on_domain"].numbers();
    bool errorOk = claimedError.size() == program.classes;
    double worst = 0;
    for (std::size_t c = 0; errorOk && c < program.classes; ++c) {
        const double bound = std::max(std::fabs(lo(global.logits[c].e)), std::fabs(hi(global.logits[c].e)));
        worst = std::max(worst, bound);
        errorOk = bound <= claimedError[c];
    }
    std::ostringstream errorText;
    errorText << "claimed logit error bound holds on the domain (checker bound " << worst << ")";
    report(errorOk, errorText.str());

    // 5. Regions.
    const double logitScale = program.layers.back().outScale[0];
    std::size_t regions = 0, leaves = 0, regionFailures = 0;
    std::string firstFailure;
    for (const auto& region : certificate["regions"].items) {
        ++regions;
        const int cls = static_cast<int>(region["class"].number);
        auto rlo = region["lo"].numbers();
        auto rhi = region["hi"].numbers();
        bool ok = cls >= 0 && static_cast<std::size_t>(cls) < program.classes && rlo.size() == program.inputs &&
                  rhi.size() == program.inputs;
        for (std::size_t i = 0; ok && i < program.inputs; ++i) {
            ok = rlo[i] >= domainLo[i] && rhi[i] <= domainHi[i] && rlo[i] <= rhi[i];
        }
        std::istringstream tree(region["tree"].text);
        std::vector<std::pair<std::vector<double>, std::vector<double>>> stack{{rlo, rhi}};
        std::string token;
        while (ok && tree >> token) {
            if (stack.empty()) {
                ok = false;
                break;
            }
            auto box = stack.back();
            stack.pop_back();
            if (token == "L") {
                ++leaves;
                const auto a = analyse(net, program, box.first, box.second);
                if (!a.failure.empty()) ok = false;
                for (std::size_t j = 0; ok && j < program.classes; ++j) {
                    if (static_cast<int>(j) == cls) continue;
                    F ref = a.logits[cls].r;
                    add(ref, -1, a.logits[j].r);
                    F quant = ref;
                    add(quant, 1, a.logits[cls].e);
                    add(quant, -1, a.logits[j].e);
                    const double needed = static_cast<int>(j) < cls ? 0.5 * logitScale : 0.0;
                    ok = lo(ref) > 0 && lo(quant) > needed;
                }
            } else if (token[0] == 'S') {
                const auto colon = token.find(':');
                const auto dim = static_cast<std::size_t>(std::atoi(token.c_str() + 1));
                const double at = std::strtod(token.c_str() + colon + 1, nullptr);
                if (colon == std::string::npos || dim >= program.inputs || at < box.first[dim] || at > box.second[dim]) {
                    ok = false;
                    break;
                }
                auto low = box, high = box;
                low.second[dim] = at;
                high.first[dim] = at;
                stack.push_back(high);  // pre-order: the lower half is next
                stack.push_back(low);
            } else {
                ok = false;
            }
        }
        ok = ok && stack.empty();
        if (!ok) {
            ++regionFailures;
            if (firstFailure.empty()) firstFailure = region["name"].text;
        }
    }
    std::ostringstream regionText;
    regionText << regions << " regions (" << leaves << " leaves): integer decision = float decision everywhere";
    if (regionFailures) regionText << " -- " << regionFailures << " failed, first " << firstFailure;
    report(regionFailures == 0, regionText.str());

    // Optional: the compiled binary agrees with the checker's semantics.
    if (argc == 6 && std::string(argv[3]) == "--expect") {
        std::ifstream raw(argv[4]), inputs(argv[5]);
        std::string rawLine, inputLine;
        std::size_t rows = 0, mismatches = 0;
        while (std::getline(inputs, inputLine)) {
            std::vector<float> x;
            const char* cursor = inputLine.c_str();
            while (x.size() < program.inputs) {
                char* end = nullptr;
                const float v = std::strtof(cursor, &end);
                if (end == cursor) break;
                x.push_back(v);
                cursor = end;
                while (*cursor == ',' || *cursor == ' ') ++cursor;
            }
            if (x.empty()) continue;
            if (!std::getline(raw, rawLine)) {
                ++mismatches;
                break;
            }
            const auto logits = runProgram(program, x);
            const int best = static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
            std::ostringstream expected;
            expected << rows << "," << best;
            for (long long v : logits) expected << "," << v;
            if (expected.str() != rawLine) ++mismatches;
            ++rows;
        }
        std::ostringstream text;
        text << "compiled binary matches the checker's integer semantics on " << rows << " inputs";
        if (mismatches) text << " (" << mismatches << " mismatches)";
        report(mismatches == 0 && rows > 0, text.str());
    }
    std::cout << (failures == 0 ? "CERTIFICATE VALID" : "CERTIFICATE REJECTED") << " (" << verdicts - failures << "/"
              << verdicts << " checks passed)\n";
    return failures == 0 ? 0 : 1;
}
