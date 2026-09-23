#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

struct Node {
    std::string name;
    std::string op;
    std::string input;
    std::string output;
};

struct Model {
    std::string name;
    std::string input;
    std::string output;
    std::vector<Node> nodes;
};

const std::unordered_set<std::string> supported_ops = {
    "Gemm", "MatMul", "Add", "Relu", "Sigmoid", "Softmax"};

bool load_manifest(const std::string& path, Model& model, std::string& error) {
    std::ifstream file(path);
    if (!file) {
        error = "Cannot open manifest: " + path;
        return false;
    }

    std::string line;
    int line_number = 0;
    while (std::getline(file, line)) {
        ++line_number;
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::istringstream stream(line);
        std::string keyword;
        stream >> keyword;

        if (keyword == "model") {
            stream >> model.name;
        } else if (keyword == "input") {
            stream >> model.input;
        } else if (keyword == "output") {
            stream >> model.output;
        } else if (keyword == "node") {
            Node node;
            stream >> node.name >> node.op >> node.input >> node.output;
            if (node.name.empty() || node.op.empty() || node.input.empty() ||
                node.output.empty()) {
                error = "Invalid node declaration at line " +
                        std::to_string(line_number);
                return false;
            }
            model.nodes.push_back(node);
        } else {
            error = "Unknown declaration '" + keyword + "' at line " +
                    std::to_string(line_number);
            return false;
        }
    }

    if (model.name.empty() || model.input.empty() || model.output.empty() ||
        model.nodes.empty()) {
        error = "Manifest must define a model, input, output, and at least one node";
        return false;
    }
    return true;
}

bool validate(const Model& model, std::string& error) {
    std::unordered_set<std::string> produced{model.input};
    for (const Node& node : model.nodes) {
        if (!supported_ops.count(node.op)) {
            error = "Unsupported operator '" + node.op + "' in node '" +
                    node.name + "'";
            return false;
        }
        if (!produced.count(node.input)) {
            error = "Input tensor '" + node.input + "' is not available before node '" +
                    node.name + "'";
            return false;
        }
        if (produced.count(node.output)) {
            error = "Duplicate tensor name '" + node.output + "'";
            return false;
        }
        produced.insert(node.output);
    }
    if (!produced.count(model.output)) {
        error = "Declared output tensor '" + model.output + "' is never produced";
        return false;
    }
    return true;
}

void print_ir(const Model& model) {
    std::cout << "\nMODEL FORGE PHASE 1 PROTOTYPE\n";
    std::cout << "=============================\n";
    std::cout << "Model: " << model.name << "\n";
    std::cout << "Input: " << model.input << "\n";
    std::cout << "Output: " << model.output << "\n";
    std::cout << "\nSUPPORTED OPERATORS\n";
    for (const Node& node : model.nodes) {
        std::cout << "  " << node.op << " ................ OK\n";
    }
    std::cout << "\nMODEL FORGE IR\n";
    std::cout << "  INPUT  " << model.input << "\n";
    for (const Node& node : model.nodes) {
        std::cout << "  " << node.op << "  input=" << node.input
                  << "  output=" << node.output << "\n";
    }
    std::cout << "  RETURN " << model.output << "\n";
    std::cout << "\nSTATUS: PROTOTYPE VALIDATION PASSED\n";
}

int main(int argc, char* argv[]) {
    const std::string path = argc > 1 ? argv[1] : "sample_model.mforge";
    Model model;
    std::string error;
    if (!load_manifest(path, model, error) || !validate(model, error)) {
        std::cerr << "COMPILATION ERROR\n" << error << "\n";
        return 1;
    }
    print_ir(model);
    return 0;
}
