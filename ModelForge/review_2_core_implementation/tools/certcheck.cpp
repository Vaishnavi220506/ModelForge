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
//   2. the input quantiser never clamps, no product leaves its multiply type
//      and no accumulator leaves its type anywhere in the input domain: for a
//      classical accumulator every partial sum must fit, for a modular
//      (wrap-around) accumulator the final sum must fit
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
struct Tap {
    std::size_t input = 0, weight = 0;
};

// Layer kinds, numbered as in MF_META.
enum Kind { kDense = 0, kConv = 1, kAdd = 2, kMaxPool = 3 };

struct Shape3 {
    long c = 0, h = 1, w = 1;
    bool spatial = false;
    std::size_t size() const { return static_cast<std::size_t>(c * h * w); }
};

struct Geometry {
    long inC = 0, inH = 0, inW = 0, outC = 0, outH = 0, outW = 0, kH = 0, kW = 0, stride = 1, pad = 0, groups = 1;
    bool operator==(const Geometry& o) const {
        return inC == o.inC && inH == o.inH && inW == o.inW && outC == o.outC && outH == o.outH && outW == o.outW &&
               kH == o.kH && kW == o.kW && stride == o.stride && pad == o.pad && groups == o.groups;
    }
};

// A layer of the float model. Tensor 0 is the input, tensor l + 1 the output
// of layer l; `src` are the tensors the layer reads.
struct Layer {
    int kind = kDense;
    std::vector<std::size_t> src;
    Shape3 shape;  // output
    std::size_t in = 0, out = 0, perChannel = 1;
    std::vector<double> w;  // dense: out x in; conv: outC x inC/groups x kH x kW; add: C x 2
    std::vector<double> b;  // per output channel
    bool relu = false;
    Geometry g;
    std::vector<std::vector<Tap>> taps;
    std::size_t channels() const { return shape.spatial ? static_cast<std::size_t>(shape.c) : out; }
    std::size_t channelOf(std::size_t o) const {
        return shape.spatial ? o / static_cast<std::size_t>(shape.h * shape.w) : o;
    }
};

struct Net {
    std::vector<Shape3> tensors;
    std::vector<Layer> layers;
};

void makeTaps(Layer& layer, const std::vector<Shape3>& tensors) {
    layer.taps.assign(layer.out, {});
    if (layer.kind == kDense) {
        for (std::size_t o = 0; o < layer.out; ++o) {
            for (std::size_t i = 0; i < layer.in; ++i) layer.taps[o].push_back({i, o * layer.in + i});
        }
        return;
    }
    if (layer.kind == kAdd) {
        const std::size_t n = tensors[layer.src[0]].size();
        for (std::size_t o = 0; o < layer.out; ++o) {
            const std::size_t c = layer.channelOf(o);
            layer.taps[o] = {{o, 2 * c}, {n + o, 2 * c + 1}};
        }
        return;
    }
    const Geometry& g = layer.g;
    const bool pool = layer.kind == kMaxPool;
    const long cpg = pool ? 1 : g.inC / g.groups, opg = pool ? 1 : g.outC / g.groups;
    for (long c = 0; c < g.outC; ++c)
        for (long oy = 0; oy < g.outH; ++oy)
            for (long ox = 0; ox < g.outW; ++ox)
                for (long ci = 0; ci < cpg; ++ci)
                    for (long ky = 0; ky < g.kH; ++ky) {
                        const long iy = oy * g.stride + ky - g.pad;
                        if (iy < 0 || iy >= g.inH) continue;
                        for (long kx = 0; kx < g.kW; ++kx) {
                            const long ix = ox * g.stride + kx - g.pad;
                            if (ix < 0 || ix >= g.inW) continue;
                            const long channel = c / opg * cpg + ci;
                            layer.taps[static_cast<std::size_t>((c * g.outH + oy) * g.outW + ox)].push_back(
                                {static_cast<std::size_t>((channel * g.inH + iy) * g.inW + ix),
                                 pool ? 0 : static_cast<std::size_t>(((c * cpg + ci) * g.kH + ky) * g.kW + kx)});
                        }
                    }
}

// Reads the .mforge text and rebuilds the layer DAG on its own: biases and
// batch normalisations are folded into the preceding linear layer when no
// other node reads it, average pools become depthwise convolutions with
// weights 1/k^2, a batch normalisation that cannot be folded a 1x1 depthwise
// convolution, a tensor Add a residual layer, Flatten and the output
// activation disappear.
bool loadReference(const std::string& path, Net& net, std::string& error) {
    std::ifstream in(path);
    if (!in) {
        error = "cannot open " + path;
        return false;
    }
    struct Node {
        std::string op, output;
        std::vector<std::string> inputs;
        bool transB = false;
        bool unit = true;  // alpha = beta = 1
        long stride = 1, pad = 0, group = 1, kernel = 0;
        float epsilon = 1e-5f;
    };
    std::map<std::string, std::pair<std::vector<long long>, std::vector<double>>> constants;
    std::vector<Node> nodes;
    std::string inputName, outputName, line;
    std::vector<long long> inputShape;
    while (std::getline(in, line)) {
        std::istringstream words(line);
        std::vector<std::string> t;
        std::string word;
        while (words >> word) t.push_back(word);
        if (t.empty() || t[0][0] == '#') continue;
        const auto dims = [](const std::string& text) {
            std::vector<long long> shape;
            std::stringstream list(text);
            std::string d;
            while (std::getline(list, d, ',')) shape.push_back(std::atoll(d.c_str()));
            return shape;
        };
        if (t[0] == "input" && t.size() > 3) {
            inputName = t[1];
            inputShape = dims(t[3]);
        }
        if (t[0] == "output" && t.size() > 1) outputName = t[1];
        if (t[0] == "tensor" && t.size() >= 5) {
            std::vector<double> values;
            std::stringstream list(t[4].substr(t[4].find('=') + 1));
            std::string d;
            while (std::getline(list, d, ',')) values.push_back(static_cast<double>(std::strtof(d.c_str(), nullptr)));
            constants[t[1]] = {dims(t[3]), values};
        }
        if (t[0] == "node") {
            const auto arrow = std::find(t.begin(), t.end(), "->");
            if (arrow == t.end() || arrow + 1 == t.end() || t.size() < 4) continue;
            Node node;
            node.op = t[2];
            node.output = *(arrow + 1);
            node.inputs.assign(t.begin() + 3, arrow);
            for (auto it = arrow + 2; it != t.end(); ++it) {
                const auto value = [&](const char* key, long& field) {
                    const std::string k = std::string(key) + "=";
                    if (it->rfind(k, 0) == 0) field = std::atol(it->c_str() + k.size());
                };
                if (*it == "transB=1") node.transB = true;
                value("strides", node.stride);
                value("pads", node.pad);
                value("group", node.group);
                value("kernel", node.kernel);
                if (it->rfind("epsilon=", 0) == 0) node.epsilon = std::strtof(it->c_str() + 8, nullptr);
                if ((it->rfind("alpha=", 0) == 0 || it->rfind("beta=", 0) == 0) &&
                    std::strtod(it->c_str() + it->find('=') + 1, nullptr) != 1.0) {
                    node.unit = false;
                }
            }
            nodes.push_back(node);
        }
    }
    Shape3 input;
    if (inputShape.size() == 4) {
        input = {static_cast<long>(inputShape[1]), static_cast<long>(inputShape[2]), static_cast<long>(inputShape[3]), true};
    } else {
        long n = 1;
        for (std::size_t k = inputShape.empty() ? 0 : 1; k < inputShape.size(); ++k) n *= static_cast<long>(inputShape[k]);
        input = {n, 1, 1, false};
    }
    net = Net{};
    net.tensors.push_back(input);
    std::map<std::string, std::size_t> uses, tensorOf{{inputName, 0}};
    for (const auto& node : nodes) {
        for (const auto& name : node.inputs) ++uses[name];
    }
    std::vector<std::size_t> pending{uses[inputName]};
    std::vector<bool> read{false};
    std::string activation;
    const auto alias = [&](const std::string& value, std::size_t t) {
        tensorOf[value] = t;
        pending[t] += uses[value];
    };
    const auto take = [&](const std::string& value, std::size_t& t) {
        const auto found = tensorOf.find(value);
        if (value == activation || found == tensorOf.end()) return false;
        t = found->second;
        --pending[t];
        return true;
    };
    const auto changeable = [&](std::size_t t) -> Layer* {
        return t == 0 || pending[t] != 0 || read[t] ? nullptr : &net.layers[t - 1];
    };
    const auto push = [&](Layer layer, const std::string& value) {
        layer.in = 0;
        for (std::size_t s : layer.src) {
            layer.in += net.tensors[s].size();
            read[s] = true;
        }
        layer.out = layer.shape.size();
        net.tensors.push_back(layer.shape);
        pending.push_back(0);
        read.push_back(false);
        net.layers.push_back(layer);
        alias(value, net.tensors.size() - 1);
    };
    const auto depthwise = [&](std::size_t s, long k, long stride, const std::vector<double>& weight,
                               const std::vector<double>& bias) {
        const Shape3 x = net.tensors[s];
        Layer layer;
        layer.kind = kConv;
        layer.src = {s};
        layer.g = {x.c, x.h, x.w, x.c, (x.h - k) / stride + 1, (x.w - k) / stride + 1, k, k, stride, 0, x.c};
        layer.shape = {x.c, layer.g.outH, layer.g.outW, true};
        layer.perChannel = static_cast<std::size_t>(k * k);
        for (long c = 0; c < x.c; ++c) layer.w.insert(layer.w.end(), static_cast<std::size_t>(k * k), weight[c]);
        layer.b = bias;
        return layer;
    };
    const auto matrix = [&](const std::string& name, bool transposed, Layer& layer) {
        const auto it = constants.find(name);
        if (it == constants.end() || it->second.first.size() != 2) return false;
        const auto rows = static_cast<std::size_t>(it->second.first[0]);
        const auto cols = static_cast<std::size_t>(it->second.first[1]);
        const std::size_t in = transposed ? cols : rows, out = transposed ? rows : cols;
        layer.w.assign(in * out, 0.0);
        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < cols; ++c) {
                const std::size_t o = transposed ? r : c, i = transposed ? c : r;
                layer.w[o * in + i] = it->second.second[r * cols + c];
            }
        }
        layer.shape = {static_cast<long>(out), 1, 1, false};
        layer.perChannel = in;
        layer.b.assign(out, 0.0);
        return true;
    };
    const auto addBias = [&](const std::string& name, Layer& layer) {
        const auto it = constants.find(name);
        if (it == constants.end() || it->second.second.size() != layer.b.size()) return false;
        for (std::size_t o = 0; o < layer.b.size(); ++o) layer.b[o] += it->second.second[o];
        return true;
    };
    for (const auto& node : nodes) {
        std::size_t x = 0;
        const auto fail = [&](const std::string& what) {
            error = what + " at " + node.op + " -> " + node.output;
            return false;
        };
        if (node.op == "Softmax" || node.op == "Sigmoid") {
            if (!take(node.inputs[0], x)) return fail("bad operand");
            activation = node.output;
            alias(node.output, x);
        } else if (node.op == "Flatten") {
            if (!take(node.inputs[0], x)) return fail("bad operand");
            alias(node.output, x);
        } else if (node.op == "Relu") {
            if (!take(node.inputs[0], x)) return fail("bad operand");
            Layer* p = changeable(x);
            if (!p || p->relu || p->kind == kMaxPool) return fail("unsupported ReLU");
            p->relu = true;
            alias(node.output, x);
        } else if (node.op == "BatchNormalization" && node.inputs.size() == 5) {
            if (!take(node.inputs[0], x)) return fail("bad operand");
            const long channels = net.tensors[x].c;
            std::vector<double> a(static_cast<std::size_t>(channels)), b(a.size());
            for (std::size_t k = 1; k < 5; ++k) {
                const auto it = constants.find(node.inputs[k]);
                if (it == constants.end() || it->second.second.size() != a.size()) return fail("bad batch norm");
            }
            const auto& gamma = constants[node.inputs[1]].second;
            const auto& beta = constants[node.inputs[2]].second;
            const auto& mean = constants[node.inputs[3]].second;
            const auto& var = constants[node.inputs[4]].second;
            for (std::size_t c = 0; c < a.size(); ++c) {
                a[c] = gamma[c] / std::sqrt(var[c] + static_cast<double>(node.epsilon));
                b[c] = beta[c] - mean[c] * a[c];
            }
            Layer* p = changeable(x);
            if (p && p->kind != kMaxPool && !p->relu && p->channels() == a.size()) {
                for (std::size_t k = 0; k < p->w.size(); ++k) p->w[k] *= a[k / p->perChannel];
                for (std::size_t c = 0; c < a.size(); ++c) p->b[c] = a[c] * p->b[c] + b[c];
                alias(node.output, x);
            } else {
                push(depthwise(x, 1, 1, a, b), node.output);  // a vector is read as C x 1 x 1
            }
        } else if (node.op == "AveragePool" || node.op == "GlobalAveragePool") {
            if (!take(node.inputs[0], x)) return fail("bad operand");
            const Shape3 s = net.tensors[x];
            const bool global = node.op == "GlobalAveragePool";
            if (!s.spatial || node.pad != 0 || (global && s.h != s.w) || (!global && node.kernel < 1)) {
                return fail("unsupported pooling");
            }
            const long k = global ? s.h : node.kernel;
            push(depthwise(x, k, global ? 1 : node.stride, std::vector<double>(static_cast<std::size_t>(s.c), 1.0 / static_cast<double>(k * k)),
                           std::vector<double>(static_cast<std::size_t>(s.c), 0.0)),
                 node.output);
        } else if (node.op == "MaxPool") {
            if (!take(node.inputs[0], x)) return fail("bad operand");
            const Shape3 s = net.tensors[x];
            if (!s.spatial || node.pad != 0 || node.kernel < 1 || node.kernel > s.h || node.kernel > s.w) {
                return fail("unsupported max-pool");
            }
            Layer layer;
            layer.kind = kMaxPool;
            layer.src = {x};
            layer.g = {s.c, s.h, s.w, s.c, (s.h - node.kernel) / node.stride + 1, (s.w - node.kernel) / node.stride + 1,
                       node.kernel, node.kernel, node.stride, 0, s.c};
            layer.shape = {s.c, layer.g.outH, layer.g.outW, true};
            push(layer, node.output);
        } else if (node.op == "Conv" && node.inputs.size() >= 2) {
            const auto weight = constants.find(node.inputs[1]);
            if (!take(node.inputs[0], x) || weight == constants.end() || weight->second.first.size() != 4) {
                return fail("bad Conv operands");
            }
            const Shape3 s = net.tensors[x];
            const auto& shape = weight->second.first;
            Layer layer;
            layer.kind = kConv;
            layer.src = {x};
            Geometry& g = layer.g;
            g = {s.c, s.h, s.w, static_cast<long>(shape[0]), 0, 0, static_cast<long>(shape[2]), static_cast<long>(shape[3]),
                 node.stride, node.pad, node.group};
            if (!s.spatial || g.groups < 1 || g.inC % g.groups || g.outC % g.groups || shape[1] * g.groups != g.inC ||
                g.stride < 1 || g.pad < 0) {
                return fail("bad Conv shape");
            }
            g.outH = (g.inH + 2 * g.pad - g.kH) / g.stride + 1;
            g.outW = (g.inW + 2 * g.pad - g.kW) / g.stride + 1;
            layer.shape = {g.outC, g.outH, g.outW, true};
            layer.perChannel = static_cast<std::size_t>(shape[1] * shape[2] * shape[3]);
            layer.w = weight->second.second;
            layer.b.assign(static_cast<std::size_t>(g.outC), 0.0);
            if (node.inputs.size() == 3 && !addBias(node.inputs[2], layer)) return fail("bad Conv bias");
            push(layer, node.output);
        } else if ((node.op == "Gemm" && node.unit) || node.op == "MatMul") {
            Layer layer;
            const bool gemm = node.op == "Gemm";
            if (node.inputs.size() < 2 || !take(node.inputs[0], x) || !matrix(node.inputs[1], gemm && node.transB, layer) ||
                layer.perChannel != net.tensors[x].size() ||
                (gemm && node.inputs.size() == 3 && !addBias(node.inputs[2], layer))) {
                return fail("bad Gemm/MatMul operands");
            }
            layer.src = {x};
            push(layer, node.output);
        } else if (node.op == "Add" && node.inputs.size() == 2) {
            const bool c0 = constants.count(node.inputs[0]) > 0, c1 = constants.count(node.inputs[1]) > 0;
            if (c0 != c1) {
                if (!take(node.inputs[c0 ? 1 : 0], x)) return fail("bad operand");
                Layer* p = changeable(x);
                if (!p || p->kind == kMaxPool || p->relu || !addBias(node.inputs[c0 ? 0 : 1], *p)) return fail("unsupported bias Add");
                alias(node.output, x);
            } else {
                std::size_t y = 0;
                if (c0 || !take(node.inputs[0], x) || !take(node.inputs[1], y) ||
                    net.tensors[x].size() != net.tensors[y].size()) {
                    return fail("unsupported Add");
                }
                Layer layer;
                layer.kind = kAdd;
                layer.src = {x, y};
                layer.shape = net.tensors[x];
                const long channels = layer.shape.spatial ? layer.shape.c : static_cast<long>(layer.shape.size());
                layer.g = {channels, layer.shape.h, layer.shape.w, channels, layer.shape.h, layer.shape.w, 1, 1, 1, 0, channels};
                layer.perChannel = 2;
                layer.w.assign(static_cast<std::size_t>(2 * channels), 1.0);
                layer.b.assign(static_cast<std::size_t>(channels), 0.0);
                push(layer, node.output);
            }
        } else {
            return fail("unsupported operator");
        }
    }
    const auto out = tensorOf.find(outputName);
    if (net.layers.empty() || out == tensorOf.end() || out->second != net.layers.size() ||
        net.layers.back().kind != kDense || net.layers.back().relu) {
        error = "unsupported network shape (the output must be a dense layer without ReLU)";
        return false;
    }
    Layer& last = net.layers.back();
    if (last.out == 1) {  // a single logit z decides z > 0
        std::vector<double> wide(2 * last.in, 0.0);
        std::copy(last.w.begin(), last.w.end(), wide.begin() + static_cast<long>(last.in));
        last.w = wide;
        last.b = {0.0, last.b[0]};
        last.out = 2;
        last.shape = {2, 1, 1, false};
        net.tensors.back() = last.shape;
    }
    for (auto& layer : net.layers) makeTaps(layer, net.tensors);
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
        std::istringstream head(source.substr(p, 100));
        std::string type, declarator;
        head >> type;
        if (type == "MF_ROM") head >> type;  // flash qualifier (empty or __flash)
        head >> declarator;
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
            const double v = type == "float" ? static_cast<double>(std::strtof(cursor, &next))
                                             : std::strtod(cursor, &next);
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
    int kind = kDense;
    Geometry g;
    std::vector<std::size_t> src;
    std::size_t in = 0, out = 0, channels = 0;
    bool relu = false, outUnsigned = false, modular = false;
    int wbits = 8, accBits = 32, mulBits = 32, outBits = 8, buffer = -1;
    std::vector<long long> w, b, m;  // b: the exact bias (decoded modulo 2^accBits if modular)
    std::vector<int> s;
    std::vector<double> outScale;  // per output neuron
    long long outMax = 0, outMin = 0;
    std::size_t hw() const { return out / channels; }
    std::size_t channelOf(std::size_t o) const { return o / hw(); }
};

struct Program {
    std::size_t inputs = 0, classes = 0;
    int inputBits = 8;
    bool inputUnsigned = false;
    long long inMax = 0, inMin = 0;
    std::vector<float> inv;  // input inverse scales as the C code uses them
    std::vector<QLayer> layers;
    std::string type(std::size_t tensor) const {
        if (tensor == 0) return cType(inputBits, inputUnsigned);
        return cType(layers[tensor - 1].outBits, layers[tensor - 1].outUnsigned);
    }
    // Last layer reading each tensor (a tensor nobody reads dies at once).
    std::vector<std::size_t> lastUse() const {
        std::vector<std::size_t> last(layers.size() + 1, 0);
        for (std::size_t t = 1; t <= layers.size(); ++t) last[t] = t - 1;
        for (std::size_t l = 0; l < layers.size(); ++l) {
            for (std::size_t s : layers[l].src) last[s] = std::max(last[s], l);
        }
        return last;
    }
    std::size_t buffers() const {
        int count = 0;
        for (const auto& q : layers) count = std::max(count, q.buffer + 1);
        return static_cast<std::size_t>(count);
    }
};

// The code templates, as the checker expects them to be emitted.
std::string expectedLayer(const Program& p, std::size_t l) {
    const QLayer& q = p.layers[l];
    const std::string inT = p.type(q.src[0]);
    const char* outT = cType(q.outBits, q.outUnsigned);
    const std::string id = std::to_string(l);
    std::ostringstream t;
    if (q.kind == kMaxPool) {
        const Geometry& g = q.g;
        const std::string s = std::to_string(g.stride);
        t << "static void mf_layer" << l << "(const " << inT << " *x, " << outT << " *y) {\n"
          << "    for (int c = 0; c < " << g.outC << "; ++c) {\n"
          << "        for (int oy = 0; oy < " << g.outH << "; ++oy) {\n"
          << "            for (int ox = 0; ox < " << g.outW << "; ++ox) {\n"
          << "                " << outT << " m = x[(c * " << g.inH << " + oy * " << s << ") * " << g.inW << " + ox * " << s
          << "];\n"
          << "                for (int ky = 0; ky < " << g.kH << "; ++ky) {\n"
          << "                    for (int kx = 0; kx < " << g.kW << "; ++kx) {\n"
          << "                        const " << outT << " v = x[(c * " << g.inH << " + oy * " << s << " + ky) * " << g.inW
          << " + ox * " << s << " + kx];\n"
          << "                        if (v > m) m = v;\n"
          << "                    }\n                }\n"
          << "                y[(c * " << g.outH << " + oy) * " << g.outW << " + ox] = m;\n"
          << "            }\n        }\n    }\n}\n";
        return t.str();
    }
    const char* accT = cType(q.accBits, false);
    const char* uaccT = cType(q.accBits, true);
    const char* mulT = q.mulBits == 16 ? "int16_t" : "int32_t";
    const auto weightAt = [&](const std::string& index) {
        return q.wbits <= 4 ? "mf_nibble(MF_W" + id + ", " + index + ")" : "MF_W" + id + "[" + index + "]";
    };
    const auto step = [&](const std::string& in, const std::string& wi, const std::string& x) {
        const std::string product = "(" + std::string(mulT) + ")" + weightAt(wi) + " * (" + mulT + ")" + x;
        return q.modular ? in + "u = (" + uaccT + ")(u + (" + uaccT + ")(" + product + "));\n"
                         : in + "acc = (" + accT + ")(acc + (" + accT + ")(" + product + "));\n";
    };
    const auto init = [&](const std::string& in, const std::string& ch) {
        return q.modular ? in + uaccT + " u = MF_B" + id + "[" + ch + "];\n"
                         : in + accT + " acc = (" + accT + ")MF_B" + id + "[" + ch + "];\n";
    };
    const auto tail = [&](const std::string& in, const std::string& ch) {
        std::ostringstream r;
        if (q.modular) {
            r << in << "int64_t acc = u >= " << (q.accBits == 16 ? "0x8000u" : "0x80000000u") << " ? (int64_t)u - "
              << (q.accBits == 16 ? "65536" : "4294967296LL") << " : (int64_t)u;\n";
        }
        if (q.relu) r << in << "if (acc < 0) acc = 0;\n";
        r << in << "int64_t q = mf_round_shift((int64_t)acc * MF_M" << l << "[" << ch << "], MF_S" << l << "[" << ch
          << "]);\n"
          << in << "if (q > " << q.outMax << ") q = " << q.outMax << ";  /* saturate (modelled by the certificate) */\n"
          << in << "if (q < " << q.outMin << ") q = " << q.outMin << ";\n";
        return r.str();
    };
    if (q.kind == kAdd) {
        const std::string in = "            ", at = "[c * " + std::to_string(q.hw()) + " + k]";
        t << "static void mf_layer" << l << "(const " << inT << " *a, const " << p.type(q.src[1]) << " *b, " << outT
          << " *y) {\n"
          << "    for (int c = 0; c < " << q.channels << "; ++c) {\n"
          << "        for (int k = 0; k < " << q.hw() << "; ++k) {\n"
          << init(in, "c") << step(in, "c * 2", "a" + at) << step(in, "c * 2 + 1", "b" + at) << tail(in, "c") << in
          << "y" << at << " = (" << outT << ")q;\n"
          << "        }\n    }\n}\n";
        return t.str();
    }
    t << "static void mf_layer" << l << "(const " << inT << " *x, " << outT << " *y) {\n";
    if (q.kind == kConv) {
        const Geometry& g = q.g;
        const long cpg = g.inC / g.groups, opg = g.outC / g.groups;
        const std::string in = "                ";
        const std::string ch = g.groups == 1 ? "ci" : "(c / " + std::to_string(opg) + " * " + std::to_string(cpg) + " + ci)";
        t << "    for (int c = 0; c < " << g.outC << "; ++c) {\n"
          << "        for (int oy = 0; oy < " << g.outH << "; ++oy) {\n"
          << "            for (int ox = 0; ox < " << g.outW << "; ++ox) {\n"
          << init(in, "c") << in << "for (int ci = 0; ci < " << cpg << "; ++ci) {\n"
          << in << "    for (int ky = 0; ky < " << g.kH << "; ++ky) {\n"
          << in << "        const int iy = oy * " << g.stride << " + ky - " << g.pad << ";\n"
          << in << "        if (iy < 0 || iy >= " << g.inH << ") continue;\n"
          << in << "        for (int kx = 0; kx < " << g.kW << "; ++kx) {\n"
          << in << "            const int ix = ox * " << g.stride << " + kx - " << g.pad << ";\n"
          << in << "            if (ix < 0 || ix >= " << g.inW << ") continue;\n"
          << step(in + "            ",
                  "((c * " + std::to_string(cpg) + " + ci) * " + std::to_string(g.kH) + " + ky) * " +
                      std::to_string(g.kW) + " + kx",
                  "x[(" + ch + " * " + std::to_string(g.inH) + " + iy) * " + std::to_string(g.inW) + " + ix]")
          << in << "        }\n" << in << "    }\n" << in << "}\n"
          << tail(in, "c") << in << "y[(c * " << g.outH << " + oy) * " << g.outW << " + ox] = (" << outT << ")q;\n"
          << "            }\n        }\n    }\n}\n";
        return t.str();
    }
    const std::string in = "        ";
    t << "    for (int o = 0; o < " << q.out << "; ++o) {\n"
      << init(in, "o") << in << "for (int i = 0; i < " << q.in << "; ++i) {\n"
      << step(in + "    ", "o * " + std::to_string(q.in) + " + i", "x[i]") << in << "}\n"
      << tail(in, "o") << in << "y[o] = (" << outT << ")q;\n    }\n}\n";
    return t.str();
}

std::vector<std::string> expectedFragments(const Program& p) {
    std::vector<std::string> parts{
        "static int64_t mf_round_shift(int64_t t, int s) {\n"
        "    int64_t u = t + ((int64_t)1 << (s - 1));\n"
        "    return u >= 0 ? (u >> s) : -((-u + (((int64_t)1 << s) - 1)) >> s);\n}\n"};
    if (std::any_of(p.layers.begin(), p.layers.end(), [](const QLayer& q) { return q.kind != kMaxPool && q.wbits <= 4; })) {
        parts.push_back("static int8_t mf_nibble(const MF_ROM uint8_t *packed, int k) {\n"
                        "    int v = (k & 1) ? (packed[k >> 1] >> 4) : (packed[k >> 1] & 15);\n"
                        "    return (int8_t)(v >= 8 ? v - 16 : v);\n}\n");
    }
    for (std::size_t l = 0; l < p.layers.size(); ++l) parts.push_back(expectedLayer(p, l));
    std::ostringstream quant;
    quant << "void mf_quantize_input(const float *x, mf_input_t *q) {\n"
          << "    for (int i = 0; i < MF_INPUTS; ++i) {\n"
          << "        float t = x[i] * MF_INPUT_INV_SCALE[i];\n"
          << "        float r = t >= 0.0f ? t + 0.5f : t - 0.5f;\n"
          << "        if (r > 1.0e9f) r = 1.0e9f;\n        if (r < -1.0e9f) r = -1.0e9f;\n"
          << "        long long v = (long long)r;\n"
          << "        if (v > " << p.inMax << ") v = " << p.inMax << ";\n"
          << "        if (v < " << p.inMin << ") v = " << p.inMin << ";\n"
          << "        q[i] = (mf_input_t)v;\n    }\n}\n";
    parts.push_back(quant.str());
    // Buffers: one declaration each, sized for the largest tensor it holds.
    std::ostringstream infer;
    infer << "int mf_infer_q(const mf_input_t *q, mf_logit_t *logits) {\n";
    for (std::size_t k = 0; k < p.buffers(); ++k) {
        std::size_t size = 0;
        std::string type;
        for (const auto& q : p.layers) {
            if (q.buffer != static_cast<int>(k)) continue;
            size = std::max(size, q.out);
            if (type.empty()) type = cType(q.outBits, q.outUnsigned);
        }
        infer << "    static " << type << " mf_buf" << k << "[" << size << "];\n";
    }
    infer << "    mf_logit_t out[MF_CLASSES];\n";
    const auto name = [&](std::size_t t) {
        if (t == 0) return std::string("q");
        const int b = p.layers[t - 1].buffer;
        return b < 0 ? std::string("out") : "mf_buf" + std::to_string(b);
    };
    for (std::size_t l = 0; l < p.layers.size(); ++l) {
        infer << "    mf_layer" << l << "(";
        for (std::size_t s : p.layers[l].src) infer << name(s) << ", ";
        infer << name(l + 1) << ");\n";
    }
    infer << "    int best = 0;\n    for (int c = 0; c < MF_CLASSES; ++c) {\n"
          << "        if (logits) logits[c] = out[c];\n        if (out[c] > out[best]) best = c;\n    }\n"
          << "    return best;\n}\n";
    parts.push_back(infer.str());
    return parts;
}

// Buffer safety: a buffer holds tensors of one C type, and a tensor is never
// overwritten (or written by a layer that reads the same buffer) before its
// last reader has run.
std::string checkBuffers(const Program& p) {
    const auto last = p.lastUse();
    for (std::size_t a = 0; a < p.layers.size(); ++a) {
        const QLayer& qa = p.layers[a];
        if ((a + 1 == p.layers.size()) != (qa.buffer < 0)) return "only the logits may bypass the buffers";
        for (std::size_t b = a + 1; b < p.layers.size(); ++b) {
            const QLayer& qb = p.layers[b];
            if (qa.buffer < 0 || qb.buffer != qa.buffer) continue;
            if (std::string(cType(qa.outBits, qa.outUnsigned)) != cType(qb.outBits, qb.outUnsigned)) {
                return "buffer " + std::to_string(qa.buffer) + " holds tensors of different types";
            }
            if (last[a + 1] >= b) {
                return "layer " + std::to_string(b) + " overwrites the output of layer " + std::to_string(a) +
                       " before its last reader";
            }
        }
    }
    return "";
}

std::string loadProgram(const std::string& source, Program& p) {
    const auto arrays = parseArrays(source);
    const auto get = [&](const std::string& name) -> const Array* {
        const auto it = arrays.find(name);
        return it == arrays.end() ? nullptr : &it->second;
    };
    const Array* meta = get("MF_META");
    constexpr std::size_t kPerLayer = 24;
    if (!meta || meta->values.size() < 6 || meta->values[0] != 4) return "missing or unknown MF_META (need v4)";
    p.inputs = static_cast<std::size_t>(meta->values[1]);
    p.classes = static_cast<std::size_t>(meta->values[2]);
    p.inputBits = static_cast<int>(meta->values[3]);
    p.inputUnsigned = meta->values[4] != 0;
    if (p.inputBits < 2 || p.inputBits > 16) return "bad input bit width";
    p.inMax = p.inputUnsigned ? (1LL << p.inputBits) - 1 : (1LL << (p.inputBits - 1)) - 1;
    p.inMin = p.inputUnsigned ? 0 : -p.inMax;
    const auto L = static_cast<std::size_t>(meta->values[5]);
    if (L == 0 || meta->values.size() != 6 + kPerLayer * L) return "MF_META has the wrong length";
    const Array* inv = get("MF_INPUT_INV_SCALE");
    if (!inv || inv->values.size() != p.inputs || inv->type != "float") return "bad MF_INPUT_INV_SCALE";
    for (double v : inv->values) {
        if (!(v > 0)) return "input scales must be positive";
        p.inv.push_back(static_cast<float>(v));
    }
    for (std::size_t l = 0; l < L; ++l) {
        const double* m = &meta->values[6 + kPerLayer * l];
        QLayer q;
        q.kind = static_cast<int>(m[0]);
        q.in = static_cast<std::size_t>(m[1]);
        q.out = static_cast<std::size_t>(m[2]);
        q.relu = m[3] != 0;
        q.wbits = static_cast<int>(m[4]);
        q.accBits = static_cast<int>(m[5]);
        q.modular = m[6] != 0;
        q.mulBits = static_cast<int>(m[7]);
        q.outBits = static_cast<int>(m[8]);
        q.outUnsigned = m[9] != 0;
        q.g = {static_cast<long>(m[10]), static_cast<long>(m[11]), static_cast<long>(m[12]), static_cast<long>(m[13]),
               static_cast<long>(m[14]), static_cast<long>(m[15]), static_cast<long>(m[16]), static_cast<long>(m[17]),
               static_cast<long>(m[18]), static_cast<long>(m[19]), static_cast<long>(m[20])};
        for (int k = 21; k <= 22; ++k) {
            if (m[k] < 0) continue;
            if (m[k] > static_cast<double>(l)) return "layer " + std::to_string(l) + " reads a later tensor";
            q.src.push_back(static_cast<std::size_t>(m[k]));
        }
        q.buffer = static_cast<int>(m[23]);
        if (q.src.empty() || q.kind < kDense || q.kind > kMaxPool || q.out == 0) return "inconsistent layer metadata";
        q.channels = q.kind == kDense ? q.out : static_cast<std::size_t>(q.g.outC);
        if (q.channels == 0 || q.out % q.channels) return "inconsistent layer metadata";
        if (q.kind == kMaxPool) {
            // The output keeps the input's codes, type and scale.
            const bool fromInput = q.src[0] == 0;
            const QLayer* producer = fromInput ? nullptr : &p.layers[q.src[0] - 1];
            if (q.outBits != (fromInput ? p.inputBits : producer->outBits) ||
                q.outUnsigned != (fromInput ? p.inputUnsigned : producer->outUnsigned) || q.relu || l + 1 == L) {
                return "max-pool must keep its input type";
            }
            q.outMax = fromInput ? p.inMax : producer->outMax;
            q.outMin = fromInput ? p.inMin : producer->outMin;
            const Array* sc = get("MF_SCALE_" + std::to_string(l));
            if (!sc || sc->values.size() != q.out) return "missing scales for layer " + std::to_string(l);
            q.outScale = sc->values;
            p.layers.push_back(q);
            continue;
        }
        if ((q.accBits != 16 && q.accBits != 32 && q.accBits != 64) || (q.modular && q.accBits == 64) ||
            (q.mulBits != 16 && q.mulBits != 32) || q.wbits < 2 || q.wbits > 16 || q.outBits < 2 || q.outBits > 16 ||
            (l + 1 == L && (q.relu || q.outUnsigned || q.kind != kDense)) || (q.kind == kAdd) != (q.src.size() == 2)) {
            return "inconsistent layer metadata";
        }
        q.outMax = q.outUnsigned ? (1LL << q.outBits) - 1 : (1LL << (q.outBits - 1)) - 1;
        q.outMin = q.outUnsigned ? 0 : -q.outMax;
        const std::string id = std::to_string(l);
        const Array* w = get("MF_W" + id);
        const Array* b = get("MF_B" + id);
        const Array* mm = get("MF_M" + id);
        const Array* s = get("MF_S" + id);
        const Array* sc = get("MF_SCALE_" + id);
        if (!w || !b || !mm || !s || !sc) return "missing arrays for layer " + id;
        const std::size_t params = q.kind == kConv  ? static_cast<std::size_t>(q.g.outC * (q.g.inC / std::max(1L, q.g.groups)) * q.g.kH * q.g.kW)
                                   : q.kind == kAdd ? 2 * q.channels
                                                    : q.in * q.out;
        const long long wmax = (1LL << (q.wbits - 1)) - 1;
        if (q.wbits <= 4) {
            if (w->type != "uint8_t" || w->values.size() != (params + 1) / 2) return "bad packed MF_W" + id;
            for (std::size_t k = 0; k < params; ++k) {
                const int byte = static_cast<int>(w->values[k / 2]);
                const int v = (k % 2) ? (byte >> 4) : (byte & 15);
                q.w.push_back(v >= 8 ? v - 16 : v);
            }
        } else {
            if (w->type != cType(q.wbits, false) || w->values.size() != params) return "bad MF_W" + id;
            for (double v : w->values) q.w.push_back(static_cast<long long>(v));
        }
        for (double v : w->values) {
            if (!fitsType(w->type, v)) return "MF_W" + id + " value does not fit its type";
        }
        for (long long v : q.w) {
            if (std::llabs(v) > wmax) return "MF_W" + id + " exceeds its bit width";
        }
        if (b->values.size() != q.channels || mm->values.size() != q.channels || s->values.size() != q.channels ||
            sc->values.size() != q.out || mm->type != "int32_t" || s->type != "uint8_t" ||
            b->type != cType(q.accBits, q.modular)) {
            return "bad bias/multiplier/shift/scale arrays for layer " + id;
        }
        for (std::size_t c = 0; c < q.channels; ++c) {
            if (!fitsType(b->type, b->values[c]) || !fitsType("int32_t", mm->values[c]) || mm->values[c] < 0 ||
                s->values[c] < 1 || s->values[c] > 62) {
                return "out-of-range constant in layer " + id;
            }
            long long bias = static_cast<long long>(b->values[c]);
            if (q.modular && bias >= (1LL << (q.accBits - 1))) bias -= 1LL << q.accBits;  // two's complement view
            q.b.push_back(bias);
            q.m.push_back(static_cast<long long>(mm->values[c]));
            q.s.push_back(static_cast<int>(s->values[c]));
        }
        for (std::size_t o = 0; o < q.out; ++o) {
            if (!(sc->values[o] > 0)) return "scales must be positive";
            q.outScale.push_back(sc->values[o]);
        }
        p.layers.push_back(q);
    }
    if (p.layers.back().out != p.classes) return "class count does not match the last layer";
    for (double v : p.layers.back().outScale) {
        if (v != p.layers.back().outScale[0]) return "logits must share one scale";
    }
    return checkBuffers(p);
}

// ------------------------------------------------- affine forms (checker)
// Sparse affine form: constant plus (symbol, coefficient) pairs sorted by symbol.
struct F {
    double c = 0;
    std::vector<std::pair<std::size_t, double>> a;
};
double rad(const F& f) {
    double r = 0;
    for (const auto& t : f.a) r += std::fabs(t.second);
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
    std::vector<std::pair<std::size_t, double>> out;
    out.reserve(d.a.size() + s.a.size());
    std::size_t i = 0, j = 0;
    while (i < d.a.size() || j < s.a.size()) {
        if (j == s.a.size() || (i < d.a.size() && d.a[i].first < s.a[j].first)) {
            out.push_back(d.a[i++]);
        } else if (i == d.a.size() || s.a[j].first < d.a[i].first) {
            out.push_back({s.a[j].first, k * s.a[j].second});
            ++j;
        } else {
            out.push_back({d.a[i].first, d.a[i].second + k * s.a[j].second});
            ++i;
            ++j;
        }
    }
    d.a = std::move(out);
}
void sym(F& f, std::size_t k, double v) {
    F single;
    single.a.push_back({k, v});
    if (f.a.empty() || f.a.back().first < k) {
        f.a.push_back({k, v});
    } else {
        add(f, 1, single);
    }
}
// Dense scratch for long sums (one output neuron of a layer).
struct Sum {
    std::vector<double> value;
    std::vector<char> used;
    std::vector<std::size_t> touched;
    void add(double k, const F& f, double& constant) {
        constant += k * f.c;
        for (const auto& t : f.a) {
            if (t.first >= value.size()) {
                value.resize(t.first + 1, 0);
                used.resize(t.first + 1, 0);
            }
            if (!used[t.first]) {
                used[t.first] = 1;
                touched.push_back(t.first);
            }
            value[t.first] += k * t.second;
        }
    }
    F take(double constant) {
        std::sort(touched.begin(), touched.end());
        F f;
        f.c = constant;
        for (auto k : touched) {
            if (value[k] != 0) f.a.push_back({k, value[k]});
            value[k] = 0;
            used[k] = 0;
        }
        touched.clear();
        return f;
    }
};

struct Pair {
    F r, e;  // float model value, integer program error
};

struct Analysis {
    std::vector<Pair> logits;
    std::string failure;     // overflow or clamp of the input quantiser: never allowed
    bool saturates = false;  // a requantisation clamp can trigger (modelled, allowed if not claimed otherwise)
};

// Same mathematics as the compiler, written independently: affine forms for
// the float model (DeepZ ReLU) and for the integer program's error, with
// effective real weights rebuilt from the parsed integers.
Analysis analyse(const Net& net, const Program& p, const std::vector<double>& blo, const std::vector<double>& bhi) {
    Analysis out;
    std::size_t symbols = 0;
    const std::size_t L = p.layers.size();
    // Per tensor: forms, scales and the proved range of the integer codes.
    std::vector<std::vector<Pair>> value(L + 1);
    std::vector<std::vector<long long>> qlo(L + 1), qhi(L + 1);
    std::vector<std::vector<double>> scale(L + 1);
    value[0].resize(p.inputs);
    for (std::size_t i = 0; i < p.inputs; ++i) {
        value[0][i].r.c = 0.5 * (blo[i] + bhi[i]);
        if (bhi[i] > blo[i]) sym(value[0][i].r, symbols, 0.5 * (bhi[i] - blo[i]));
        ++symbols;
    }
    for (std::size_t i = 0; i < p.inputs; ++i) {
        const double inv = static_cast<double>(p.inv[i]);
        const double T = std::max(std::fabs(blo[i]), std::fabs(bhi[i])) * inv;
        const double delta = 0.5 + 2.0 * (T + 1.0) * std::ldexp(1.0, -24);  // float32 multiply and +-0.5
        scale[0].push_back(1.0 / inv);
        sym(value[0][i].e, symbols++, delta / inv);
        qlo[0].push_back(static_cast<long long>(std::ceil(blo[i] * inv - delta)));
        qhi[0].push_back(static_cast<long long>(std::floor(bhi[i] * inv + delta)));
        if (qlo[0][i] < p.inMin || qhi[0][i] > p.inMax) out.failure = "input quantiser may clamp";
    }
    const auto last = p.lastUse();
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
    for (std::size_t l = 0; l < L; ++l) {
        const QLayer& q = p.layers[l];
        const Layer& f = net.layers[l];
        // The layer reads the concatenation of its sources.
        std::vector<const Pair*> v;
        std::vector<long long> lo_, hi_;
        std::vector<double> inScale;
        for (std::size_t s : q.src) {
            for (const auto& n : value[s]) v.push_back(&n);
            lo_.insert(lo_.end(), qlo[s].begin(), qlo[s].end());
            hi_.insert(hi_.end(), qhi[s].begin(), qhi[s].end());
            inScale.insert(inScale.end(), scale[s].begin(), scale[s].end());
        }
        std::vector<Pair> next(q.out);
        std::vector<long long> nlo(q.out), nhi(q.out);
        if (q.kind == kMaxPool) {
            // Codes are compared and copied, so a window must share one scale.
            // A candidate that can never be the maximum is dropped; a unique
            // maximum is copied; otherwise the maximum is an interval and the
            // error is the tighter of Q' - R' and [min E_k, max E_k].
            for (std::size_t o = 0; o < q.out; ++o) {
                const auto& taps = f.taps[o];
                const std::size_t n = taps.size();
                std::vector<F> qv(n);
                std::vector<double> rl(n), ru(n), ql(n), qu(n);
                double rFloor = -INFINITY, qFloor = -INFINITY;
                nlo[o] = lo_[taps[0].input];
                nhi[o] = hi_[taps[0].input];
                for (std::size_t k = 0; k < n; ++k) {
                    const Pair& x = *v[taps[k].input];
                    if (inScale[taps[k].input] != q.outScale[o]) out.failure = "max-pool window mixes scales";
                    qv[k] = x.r;
                    add(qv[k], 1, x.e);
                    rl[k] = lo(x.r);
                    ru[k] = hi(x.r);
                    ql[k] = lo(qv[k]);
                    qu[k] = hi(qv[k]);
                    rFloor = std::max(rFloor, rl[k]);
                    qFloor = std::max(qFloor, ql[k]);
                    nlo[o] = std::max(nlo[o], lo_[taps[k].input]);
                    nhi[o] = std::max(nhi[o], hi_[taps[k].input]);
                }
                std::vector<std::size_t> rs, qs, live;
                for (std::size_t k = 0; k < n; ++k) {
                    if (ru[k] >= rFloor) rs.push_back(k);
                    if (qu[k] >= qFloor) qs.push_back(k);
                    if (ru[k] >= rFloor || qu[k] >= qFloor) live.push_back(k);
                }
                const auto unique = [](const std::vector<std::size_t>& set, const std::vector<double>& a,
                                       const std::vector<double>& b) -> long {
                    for (std::size_t i : set) {
                        bool all = true;
                        for (std::size_t j : set) all = all && (i == j || a[i] >= b[j]);
                        if (all) return static_cast<long>(i);
                    }
                    return -1;
                };
                const long kr = unique(rs, rl, ru), kq = unique(qs, ql, qu);
                Pair& y = next[o];
                double rHi = -INFINITY, qHi = -INFINITY, eLo = INFINITY, eHi = -INFINITY;
                for (std::size_t k : live) {
                    rHi = std::max(rHi, ru[k]);
                    qHi = std::max(qHi, qu[k]);
                    eLo = std::min(eLo, lo(v[taps[k].input]->e));
                    eHi = std::max(eHi, hi(v[taps[k].input]->e));
                }
                if (kr >= 0) {
                    y.r = v[taps[static_cast<std::size_t>(kr)].input]->r;
                } else {
                    y.r.c = 0.5 * (rFloor + rHi);
                    sym(y.r, symbols++, 0.5 * (rHi - rFloor));
                }
                if (kr >= 0 && kq == kr) {
                    y.e = v[taps[static_cast<std::size_t>(kq)].input]->e;
                    continue;
                }
                F qm;
                if (kq >= 0) {
                    qm = qv[static_cast<std::size_t>(kq)];
                } else {
                    qm.c = 0.5 * (qFloor + qHi);
                    sym(qm, symbols++, 0.5 * (qHi - qFloor));
                }
                F d = qm;
                add(d, -1, y.r);
                if (hi(d) - lo(d) <= (eHi - eLo) * (1 + 1e-9)) {
                    y.e = d;
                } else {
                    y.e.c = 0.5 * (eLo + eHi);
                    sym(y.e, symbols++, 0.5 * (eHi - eLo));
                }
            }
        } else {
            const long long accMax = (1LL << (q.accBits - 1)) - 1;
            const long long mulMax = (1LL << (q.mulBits - 1)) - 1;
            std::vector<double> factor(q.out);
            Sum sum;
            for (std::size_t o = 0; o < q.out; ++o) {
                const std::size_t c = q.channelOf(o);
                // Products and (classical) partial sums.
                double bound = std::fabs(static_cast<double>(q.b[c]));
                for (const auto& tap : f.taps[o]) {
                    const double product = std::fabs(static_cast<double>(q.w[tap.weight])) *
                                           std::max(std::fabs(static_cast<double>(lo_[tap.input])),
                                                    std::fabs(static_cast<double>(hi_[tap.input])));
                    if (product > static_cast<double>(mulMax)) out.failure = "product may overflow in layer " + std::to_string(l);
                    bound += product;
                }
                if (!q.modular && bound > static_cast<double>(accMax)) {
                    out.failure = "accumulator may overflow in layer " + std::to_string(l);
                }
                const double k = q.outScale[o] * std::ldexp(static_cast<double>(q.m[c]), -q.s[c]);
                factor[o] = k;
                Pair& n = next[o];
                double rc = f.b[c], ec = k * static_cast<double>(q.b[c]) - f.b[c];
                for (const auto& tap : f.taps[o]) sum.add(f.w[tap.weight], v[tap.input]->r, rc);
                n.r = sum.take(rc);
                for (const auto& tap : f.taps[o]) {
                    const double wf = f.w[tap.weight];
                    const double wq = k * static_cast<double>(q.w[tap.weight]) / inScale[tap.input];
                    sum.add(wq - wf, v[tap.input]->r, ec);
                    sum.add(wq, v[tap.input]->e, ec);
                }
                n.e = sum.take(ec);
            }
            for (std::size_t o = 0; o < q.out; ++o) {
                const std::size_t c = q.channelOf(o);
                Pair& n = next[o];
                F qv = n.r;
                add(qv, 1, n.e);
                double ql = lo(qv), qu = hi(qv);
                // The final sum is exactly (pre-ReLU value) / factor.
                const double finalSum = std::max(std::fabs(ql), std::fabs(qu)) / factor[o];
                if (q.modular && finalSum > static_cast<double>(accMax)) {
                    out.failure = "modular accumulator: final sum may not fit in layer " + std::to_string(l);
                }
                if (finalSum * static_cast<double>(q.m[c]) + std::ldexp(1.0, q.s[c] - 1) >= std::ldexp(1.0, 63)) {
                    out.failure = "64-bit requantisation product may overflow in layer " + std::to_string(l);
                }
                if (q.relu) {
                    const double l0 = lo(n.r), u0 = hi(n.r);
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
                bool exact = q.relu && qu <= 0;  // exactly 0
                // Clamp to the type range: clamp(Y) - Y relaxed like a ReLU.
                const double U = static_cast<double>(q.outMax) * q.outScale[o];
                const auto clampPart = [&](double sign, double a, double b) {
                    if (b <= 0) return;
                    out.saturates = true;
                    F y = n.r;
                    add(y, 1, n.e);
                    if (a >= 0) {  // always at the limit
                        add(n.e, -1, y);
                        n.e.c += sign * U;
                        exact = true;
                        return;
                    }
                    const double lambda = b / (b - a), mu = -lambda * a / 2;
                    add(n.e, -lambda, y);
                    n.e.c += lambda * sign * U - sign * mu;
                    sym(n.e, symbols++, mu);
                };
                clampPart(1, ql - U, qu - U);
                ql = std::min(ql, U);
                qu = std::min(qu, U);
                if (!q.relu) {
                    clampPart(-1, -U - qu, -U - ql);
                    ql = std::max(ql, -U);
                    qu = std::max(qu, -U);
                }
                nlo[o] = std::max(q.outMin, static_cast<long long>(std::floor(ql / q.outScale[o] + 0.5)));
                nhi[o] = std::min(q.outMax, static_cast<long long>(std::floor(qu / q.outScale[o] + 0.5)));
                if (!exact) sym(n.e, symbols++, 0.5 * q.outScale[o] * (1 + 1e-12));
            }
        }
        value[l + 1] = std::move(next);
        qlo[l + 1] = nlo;
        qhi[l + 1] = nhi;
        scale[l + 1] = q.outScale;
        for (std::size_t s : q.src) {
            if (last[s] == l) std::vector<Pair>().swap(value[s]);
        }
    }
    out.logits = value[L];
    return out;
}

// The integer program, exactly as the C code computes it (modular
// accumulators wrap; classical ones are exact when no overflow is possible).
std::vector<long long> runProgram(const Net& net, const Program& p, const std::vector<float>& x) {
    std::vector<std::vector<long long>> t(p.layers.size() + 1);
    for (std::size_t i = 0; i < p.inputs; ++i) {
        const float s = x[i] * p.inv[i];
        float r = s >= 0.0f ? s + 0.5f : s - 0.5f;
        r = std::max(-1.0e9f, std::min(1.0e9f, r));
        long long v = static_cast<long long>(r);
        t[0].push_back(std::max(p.inMin, std::min(p.inMax, v)));
    }
    for (std::size_t l = 0; l < p.layers.size(); ++l) {
        const QLayer& q = p.layers[l];
        std::vector<long long> cur;
        for (std::size_t s : q.src) cur.insert(cur.end(), t[s].begin(), t[s].end());
        std::vector<long long> next(q.out);
        for (std::size_t o = 0; o < q.out; ++o) {
            if (q.kind == kMaxPool) {
                long long m = cur[net.layers[l].taps[o][0].input];
                for (const auto& tap : net.layers[l].taps[o]) m = std::max(m, cur[tap.input]);
                next[o] = m;
                continue;
            }
            const std::size_t c = q.channelOf(o);
            long long acc = q.b[c];
            for (const auto& tap : net.layers[l].taps[o]) acc += q.w[tap.weight] * cur[tap.input];
            if (q.modular) {
                const unsigned long long mask = (1ULL << q.accBits) - 1;
                const unsigned long long u = static_cast<unsigned long long>(acc) & mask;
                acc = u >= (1ULL << (q.accBits - 1)) ? static_cast<long long>(u) - (1LL << q.accBits)
                                                      : static_cast<long long>(u);
            }
            if (q.relu && acc < 0) acc = 0;
            const long long product = acc * q.m[c];
            const long long u = product + (1LL << (q.s[c] - 1));
            long long r = u >= 0 ? (u >> q.s[c]) : -((-u + ((1LL << q.s[c]) - 1)) >> q.s[c]);
            next[o] = std::max(q.outMin, std::min(q.outMax, r));
        }
        t[l + 1] = next;
    }
    return t.back();
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
    Net net;
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
    bool templateOk = header.find("typedef " + std::string(cType(program.inputBits, program.inputUnsigned)) +
                                  " mf_input_t;") !=
                          std::string::npos &&
                      header.find("typedef " + std::string(cType(program.layers.back().outBits, false)) +
                                  " mf_logit_t;") != std::string::npos;
    for (const auto& fragment : expectedFragments(program)) templateOk = templateOk && source.find(fragment) != std::string::npos;
    report(templateOk, "C code matches the checker's template for the metadata");

    // Shapes of the source model.
    bool shapes = net.layers.size() == program.layers.size() && net.tensors[0].size() == program.inputs;
    for (std::size_t l = 0; shapes && l < net.layers.size(); ++l) {
        const auto& f = net.layers[l];
        const auto& q = program.layers[l];
        shapes = f.kind == q.kind && f.src == q.src && f.in == q.in && f.out == q.out && f.relu == q.relu &&
                 f.channels() == q.channels && (f.kind == kDense || f.g == q.g);
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
    report(global.failure.empty(), "no input clamp or product/accumulator overflow on the domain" +
                                       (global.failure.empty() ? "" : ": " + global.failure));
    if (certificate["claims"]["no_saturation_on_domain"].boolean) {
        report(!global.saturates, "no requantisation clamp can trigger on the domain (as claimed)");
    } else {
        std::cout << "  [INFO] saturation is possible on the domain; the clamps are modelled in every proof\n";
    }
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
            const auto logits = runProgram(net, program, x);
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
