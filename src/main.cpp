#include <SEALTorch/sealtorch.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Model artifacts are JSON, but the inference library deliberately has no
// JSON dependency.  The small reader stays at the application boundary.
namespace json {
struct Value {
    enum class Kind { null_value, number, string, array, object } kind = Kind::null_value;
    double number = 0.0;
    std::string string;
    std::vector<Value> array;
    std::map<std::string, Value> object;
    const Value &at(const std::string &key) const { return object.at(key); }
    bool has(const std::string &key) const { return object.find(key) != object.end(); }
};

class Parser {
public:
    explicit Parser(std::string text) : text_(std::move(text)) {}
    Value parse() { skip(); return value(); }
private:
    std::string text_; std::size_t position_ = 0;
    void skip() { while (position_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[position_]))) ++position_; }
    void expect(char expected) { skip(); if (position_ >= text_.size() || text_[position_++] != expected) throw std::runtime_error("invalid JSON"); }
    std::string string_value() { expect('"'); std::string result; while (position_ < text_.size() && text_[position_] != '"') { if (text_[position_] == '\\' && position_ + 1 < text_.size()) ++position_; result += text_[position_++]; } expect('"'); return result; }
    Value value() {
        skip(); if (position_ >= text_.size()) throw std::runtime_error("unexpected end of JSON");
        if (text_[position_] == '{') return object_value(); if (text_[position_] == '[') return array_value();
        if (text_[position_] == '"') { Value result; result.kind = Value::Kind::string; result.string = string_value(); return result; }
        const std::size_t begin = position_; while (position_ < text_.size() && std::string(",]} \t\r\n").find(text_[position_]) == std::string::npos) ++position_;
        const std::string token = text_.substr(begin, position_ - begin); if (token == "null" || token == "true" || token == "false") return {};
        Value result; result.kind = Value::Kind::number; result.number = std::stod(token); return result;
    }
    Value array_value() { expect('['); Value result; result.kind = Value::Kind::array; skip(); if (position_ < text_.size() && text_[position_] == ']') { ++position_; return result; } for (;;) { result.array.push_back(value()); skip(); if (position_ < text_.size() && text_[position_] == ']') { ++position_; return result; } expect(','); } }
    Value object_value() { expect('{'); Value result; result.kind = Value::Kind::object; skip(); if (position_ < text_.size() && text_[position_] == '}') { ++position_; return result; } for (;;) { const std::string key = string_value(); expect(':'); result.object.emplace(key, value()); skip(); if (position_ < text_.size() && text_[position_] == '}') { ++position_; return result; } expect(','); } }
};
}

struct ModelArtifact { sealtorch::Sequential encrypted; };

static ModelArtifact load_model(const std::string &path) {
    std::ifstream file(path); if (!file) throw std::runtime_error("cannot open model: " + path);
    std::stringstream contents; contents << file.rdbuf(); const auto root = json::Parser(contents.str()).parse();
    const auto &tensors = root.at("tensors").object; const auto &layers = root.at("model").at("layers").array;
    ModelArtifact result; const std::vector<std::pair<std::string, std::string>> names = {{"network.1.weight", "network.1.bias"}, {"network.3.weight", "network.3.bias"}, {"network.5.weight", "network.5.bias"}};
    std::size_t layer_position = 0;
    for (const auto &name : names) {
        while (layers.at(layer_position).at("type").string != "Linear") ++layer_position;
        const auto &weights = tensors.at(name.first).at("data").array; const auto &biases = tensors.at(name.second).at("data").array;
        sealtorch::DenseLayer layer{static_cast<int>(weights.front().array.size()), static_cast<int>(weights.size()), {}, {}};
        layer.weights.resize(weights.size()); layer.biases.resize(biases.size());
        for (std::size_t row = 0; row < weights.size(); ++row) { for (const auto &weight : weights[row].array) layer.weights[row].push_back(weight.number); layer.biases[row] = biases[row].number; }
        result.encrypted.add(sealtorch::Linear(std::move(layer)));
        ++layer_position;
        if (layer_position < layers.size() && layers[layer_position].at("type").string != "Linear") {
            const std::string type = layers[layer_position].at("type").string;
            const auto activation = type == "ReLU" ? sealtorch::ActivationType::Relu : sealtorch::ActivationType::Gelu;
            result.encrypted.add(activation == sealtorch::ActivationType::Relu ? sealtorch::Activation::relu() : sealtorch::Activation::gelu());
        }
    }
    return result;
}

struct RunConfig {
    bool packed = true; std::size_t threads = 4; std::string device = "auto";
    int ring_dim = 16384; int depth = 15; int scaling_mod_bits = 40; int first_mod_bits = 50; int scale_bits = 25;
    bool operator==(const RunConfig &) const = default;
};

static std::string string_or(const json::Value &object, const std::string &name, const std::string &fallback) { return object.has(name) ? object.at(name).string : fallback; }
static int int_or(const json::Value &object, const std::string &name, int fallback) {
    if (!object.has(name)) return fallback;
    const auto &value = object.at(name);
    return value.kind == json::Value::Kind::string ? std::stoi(value.string) : static_cast<int>(value.number);
}
static RunConfig parse_config(const json::Value &request) {
    const json::Value &value = request.has("config") ? request.at("config") : request;
    RunConfig config; config.packed = string_or(value, "backend", "packed") != "scalar"; config.device = string_or(value, "device", string_or(value, "plaintext_device", "auto"));
    config.threads = static_cast<std::size_t>(int_or(value, "threads", 4)); config.ring_dim = int_or(value, "ring_dim", 16384); config.depth = int_or(value, "depth", 15);
    config.scaling_mod_bits = int_or(value, "scaling_mod_bits", 40); config.first_mod_bits = int_or(value, "first_mod_bits", 50); config.scale_bits = int_or(value, "scale_bits", 25);
    if (config.threads == 0 || config.ring_dim < 1024 || (config.ring_dim & (config.ring_dim - 1)) || config.depth < 1 || config.scale_bits < 1) throw std::runtime_error("invalid inference configuration");
    if (config.device != "auto" && config.device != "cpu" && config.device != "cuda") throw std::runtime_error("plaintext_device must be auto, cpu, or cuda");
    return config;
}

static sealtorch::EncryptedInferenceOptions inference_options(const RunConfig &config) {
    return {
        config.device == "cpu" ? sealtorch::InferenceProvider::CPU :
        config.device == "cuda" ? sealtorch::InferenceProvider::CUDA : sealtorch::InferenceProvider::Auto,
        config.packed ? sealtorch::CiphertextLayout::Packed : sealtorch::CiphertextLayout::Scalar,
        config.threads, static_cast<std::size_t>(config.ring_dim), static_cast<std::size_t>(config.depth),
        static_cast<std::size_t>(config.scaling_mod_bits), static_cast<std::size_t>(config.first_mod_bits),
        static_cast<std::size_t>(config.scale_bits)};
}

static double memory_mb() { std::ifstream file("/proc/self/status"); std::string line; while (std::getline(file, line)) if (line.rfind("VmRSS:", 0) == 0) { std::istringstream values(line.substr(6)); double kilobytes = 0; values >> kilobytes; return kilobytes / 1024.0; } return 0; }
static void print_numbers(const std::vector<double> &values) { std::cout << '['; for (std::size_t i = 0; i < values.size(); ++i) { if (i) std::cout << ','; std::cout << std::setprecision(12) << values[i]; } std::cout << ']'; }
static const std::string &model_path(const json::Value &request) { static const std::string relu = "src/mnist_mlp.json", gelu = "src/mnist_mlp_gelu.json"; const std::string selected = request.has("model") ? request.at("model").string : "relu"; if (selected == "relu") return relu; if (selected == "gelu") return gelu; throw std::runtime_error("model must be relu or gelu"); }

static int run_web_worker() {
    std::unique_ptr<sealtorch::EncryptedInference> inference;
    std::string active_model; RunConfig active_config; std::string line;
    while (std::getline(std::cin, line)) try {
        const auto request = json::Parser(line).parse(); const auto &pixels = request.at("pixels"); if (pixels.array.size() != 784) throw std::runtime_error("pixels must contain 784 values"); std::vector<double> input; for (const auto &item : pixels.array) input.push_back(item.number);
        const RunConfig config = parse_config(request); const std::string &selected_model = model_path(request);
        const bool rebuild = !inference || config != active_config || selected_model != active_model;
        double setup_ms = 0.0;
        if (rebuild) {
            const auto setup_start = std::chrono::steady_clock::now();
            const ModelArtifact artifact = load_model(selected_model);
            inference = std::make_unique<sealtorch::EncryptedInference>(artifact.encrypted, inference_options(config));
            inference->predict(input);
            active_config = config;
            active_model = selected_model;
            setup_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - setup_start).count();
        }
        const double memory_before = memory_mb();
        const sealtorch::EncryptedInferenceResult encrypted = inference->predict(input);
        const bool use_cuda = inference->provider() == sealtorch::InferenceProvider::CUDA;
        std::cout << "{\"encrypted\":"; print_numbers(encrypted.values); std::cout << ",\"setup_ms\":" << setup_ms << ",\"encrypt_ms\":" << encrypted.encrypt_ms << ",\"evaluate_ms\":" << encrypted.evaluate_ms << ",\"decrypt_ms\":" << encrypted.decrypt_ms << ",\"encrypted_ms\":" << encrypted.encrypt_ms + encrypted.evaluate_ms + encrypted.decrypt_ms << ",\"memory_before_mb\":" << memory_before << ",\"memory_after_mb\":" << memory_mb() << ",\"input_ciphertext_bytes\":" << encrypted.input_ciphertext_bytes << ",\"output_ciphertext_bytes\":" << encrypted.output_ciphertext_bytes << ",\"backend\":\"" << (config.packed ? "packed" : "scalar") << "\",\"device\":\"" << (use_cuda ? "cuda" : "cpu") << "\"}\n";
    } catch (const std::exception &error) { std::cout << "{\"error\":\"" << error.what() << "\"}\n"; }
    std::cout.flush();
    return 0;
}

int main(int argc, char **argv) {
    try { if (argc == 2 && std::string(argv[1]) == "--web-worker") return run_web_worker(); throw std::runtime_error("SEALTorch is served through webui/server.py"); }
    catch (const std::exception &error) { std::cerr << "SEALTorch error: " << error.what() << '\n'; return 1; }
}
