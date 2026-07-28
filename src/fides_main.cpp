// CUDA worker.  This file intentionally speaks the FIDESlib API directly.
// It is separate from main.cpp because FIDESlib is not a Microsoft SEAL API.
#include <fideslib.hpp>
#include <cuda_runtime_api.h>

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

namespace json {
struct Value { enum class Kind { null_value, number, string, array, object } kind = Kind::null_value; double number{}; std::string string; std::vector<Value> array; std::map<std::string, Value> object; const Value &at(const std::string &key) const { return object.at(key); } bool has(const std::string &key) const { return object.contains(key); } };
class Parser {
public: explicit Parser(std::string text) : text_(std::move(text)) {} Value parse() { skip(); return value(); }
private:
    std::string text_; std::size_t position_{};
    void skip() { while (position_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[position_]))) ++position_; }
    void expect(char expected) { skip(); if (position_ == text_.size() || text_[position_++] != expected) throw std::runtime_error("invalid JSON"); }
    std::string string_value() { expect('"'); std::string result; while (position_ < text_.size() && text_[position_] != '"') { if (text_[position_] == '\\' && position_ + 1 < text_.size()) ++position_; result += text_[position_++]; } expect('"'); return result; }
    Value value() { skip(); if (position_ == text_.size()) throw std::runtime_error("unexpected end of JSON"); if (text_[position_] == '{') return object_value(); if (text_[position_] == '[') return array_value(); if (text_[position_] == '"') return {.kind = Value::Kind::string, .string = string_value()}; const auto begin = position_; while (position_ < text_.size() && std::string(",]} \t\r\n").find(text_[position_]) == std::string::npos) ++position_; const auto token = text_.substr(begin, position_ - begin); if (token == "null" || token == "true" || token == "false") return {}; return {.kind = Value::Kind::number, .number = std::stod(token)}; }
    Value array_value() { expect('['); Value result{.kind = Value::Kind::array}; skip(); if (position_ < text_.size() && text_[position_] == ']') { ++position_; return result; } for (;;) { result.array.push_back(value()); skip(); if (position_ < text_.size() && text_[position_] == ']') { ++position_; return result; } expect(','); } }
    Value object_value() { expect('{'); Value result{.kind = Value::Kind::object}; skip(); if (position_ < text_.size() && text_[position_] == '}') { ++position_; return result; } for (;;) { auto key = string_value(); expect(':'); result.object.emplace(std::move(key), value()); skip(); if (position_ < text_.size() && text_[position_] == '}') { ++position_; return result; } expect(','); } }
};
} // namespace json

enum class Activation { None, Relu, Gelu };
struct DenseLayer { std::vector<std::vector<double>> weights; std::vector<double> bias; Activation activation = Activation::None; };
struct Model { std::vector<DenseLayer> layers; };

static Model load_model(const std::string &path) {
    std::ifstream file(path); if (!file) throw std::runtime_error("cannot open model: " + path);
    std::stringstream contents; contents << file.rdbuf(); const auto root = json::Parser(contents.str()).parse();
    const auto &tensors = root.at("tensors").object; const auto &layers = root.at("model").at("layers").array;
    Model model; const std::vector<std::pair<std::string, std::string>> names = {{"network.1.weight", "network.1.bias"}, {"network.3.weight", "network.3.bias"}, {"network.5.weight", "network.5.bias"}};
    std::size_t position = 0;
    for (const auto &[weight_name, bias_name] : names) {
        while (layers.at(position).at("type").string != "Linear") ++position;
        const auto &weights = tensors.at(weight_name).at("data").array; const auto &bias = tensors.at(bias_name).at("data").array;
        DenseLayer layer; layer.weights.resize(weights.size()); layer.bias.resize(bias.size());
        for (std::size_t row = 0; row < weights.size(); ++row) { for (const auto &value : weights[row].array) layer.weights[row].push_back(value.number); layer.bias[row] = bias[row].number; }
        ++position;
        if (position < layers.size()) {
            const auto &type = layers[position].at("type").string;
            layer.activation = type == "ReLU" ? Activation::Relu : type == "GELU" ? Activation::Gelu : Activation::None;
        }
        model.layers.push_back(std::move(layer));
    }
    return model;
}

struct RunConfig { bool packed = true; std::size_t threads = 4; std::string device = "auto"; int ring_dim = 16384; int depth = 15; int scaling_mod_bits = 40; int first_mod_bits = 50; int scale_bits = 25; bool operator==(const RunConfig &) const = default; };
static std::string string_or(const json::Value &object, const std::string &name, const std::string &fallback) { return object.has(name) ? object.at(name).string : fallback; }
static int int_or(const json::Value &object, const std::string &name, int fallback) { if (!object.has(name)) return fallback; const auto &value = object.at(name); return value.kind == json::Value::Kind::string ? std::stoi(value.string) : static_cast<int>(value.number); }
static RunConfig parse_config(const json::Value &request) { const auto &value = request.has("config") ? request.at("config") : request; RunConfig config; config.packed = string_or(value, "backend", "packed") != "scalar"; config.device = string_or(value, "plaintext_device", "auto"); config.threads = int_or(value, "threads", 4); config.ring_dim = int_or(value, "ring_dim", 16384); config.depth = int_or(value, "depth", 15); config.scaling_mod_bits = int_or(value, "scaling_mod_bits", 40); config.first_mod_bits = int_or(value, "first_mod_bits", 50); config.scale_bits = int_or(value, "scale_bits", 25); if (config.device == "cpu") throw std::runtime_error("the CUDA worker requires plaintext_device cuda or auto"); if (config.threads == 0 || config.ring_dim < 1024 || (config.ring_dim & (config.ring_dim - 1))) throw std::runtime_error("invalid inference configuration"); return config; }

using Context = fideslib::CryptoContext<fideslib::DCRTPoly>;
using Ciphertext = fideslib::Ciphertext<fideslib::DCRTPoly>;
using Plaintext = fideslib::Plaintext;

// Weight plaintexts are immutable model data.  Build and upload them once
// during setup, not once per image.  Otherwise the host dominates a GPU run.
struct PackedLayerCache {
    std::vector<Plaintext> diagonals;
    Plaintext bias;
};

static PackedLayerCache cache_packed_layer(
    const Context &context, const DenseLayer &layer, std::size_t slots) {
    PackedLayerCache cache;
    const std::size_t diagonal_count = layer.weights.front().size();
    cache.diagonals.reserve(diagonal_count);
    for (std::size_t diagonal = 0; diagonal < diagonal_count; ++diagonal) {
        std::vector<double> values(slots, 0.0);
        for (std::size_t row = 0; row < layer.weights.size(); ++row) {
            const auto column = (row + diagonal) % slots;
            if (column < layer.weights[row].size()) values[row] = layer.weights[row][column];
        }
        cache.diagonals.push_back(context->MakeCKKSPackedPlaintext(values));
        context->LoadPlaintext(cache.diagonals.back());
    }
    std::vector<double> padded_bias(slots, 0.0);
    std::copy(layer.bias.begin(), layer.bias.end(), padded_bias.begin());
    cache.bias = context->MakeCKKSPackedPlaintext(padded_bias);
    context->LoadPlaintext(cache.bias);
    return cache;
}

static Ciphertext packed_linear(
    const Context &context, const Ciphertext &input, PackedLayerCache &cache) {
    Ciphertext result; bool first = true;
    // Diagonal matrix-vector multiplication keeps all output neurons in one
    // ciphertext.  This is the useful packed baseline for new HE layers.
    for (std::size_t diagonal = 0; diagonal < cache.diagonals.size(); ++diagonal) {
        auto rotated = diagonal == 0 ? input : context->EvalRotate(input, static_cast<int32_t>(diagonal));
        auto term = context->EvalMult(rotated, cache.diagonals[diagonal]);
        if (first) { result = std::move(term); first = false; } else context->EvalAddInPlace(result, term);
    }
    context->RescaleInPlace(result);
    context->EvalAddInPlace(result, cache.bias);
    return result;
}

static Ciphertext polynomial_activation(const Context &context, const Ciphertext &input, Activation type, std::size_t slots) {
    // Low-degree HE-friendly fits. Native FIDESlib methods make the level
    // transitions explicit and make this the single place to add a new fit.
    auto squared = context->EvalMult(input, input); context->RescaleInPlace(squared);
    auto linear = context->EvalMult(input, 0.5); context->RescaleInPlace(linear);
    const double constant = type == Activation::Relu ? 0.33810450 : 0.0;
    const double quadratic_coefficient = type == Activation::Relu ? 0.096514968 : 0.3989422804014327;
    auto quadratic = context->EvalMult(squared, quadratic_coefficient); context->RescaleInPlace(quadratic);
    context->EvalAddInPlace(quadratic, linear);
    auto constant_plain = context->MakeCKKSPackedPlaintext(std::vector<double>(slots, constant)); context->EvalAddInPlace(quadratic, constant_plain); return quadratic;
}

class FidesInference {
public:
    FidesInference(Model model, RunConfig config) : model_(std::move(model)), config_(std::move(config)) {
        if (!config_.packed)
            throw std::runtime_error("native FIDESlib worker currently supports packed inference only; use the CPU worker's scalar mode for the non-packed benchmark baseline");
        int devices = 0; if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) throw std::runtime_error("FIDESlib CUDA worker requires a CUDA device");
        fideslib::CCParams<fideslib::CryptoContextCKKSRNS> parameters; parameters.SetRingDim(config_.ring_dim); parameters.SetBatchSize(config_.ring_dim / 2); parameters.SetMultiplicativeDepth(config_.depth); parameters.SetScalingModSize(config_.scaling_mod_bits); parameters.SetFirstModSize(config_.first_mod_bits); parameters.SetSecurityLevel(fideslib::HEStd_NotSet); parameters.SetScalingTechnique(fideslib::FLEXIBLEAUTO); parameters.SetKeySwitchTechnique(fideslib::HYBRID); parameters.SetDevices(std::vector<int>{0}); parameters.SetPlaintextAutoload(false); parameters.SetCiphertextAutoload(true);
        context_ = fideslib::GenCryptoContext(parameters); context_->Enable(fideslib::PKE); context_->Enable(fideslib::KEYSWITCH); context_->Enable(fideslib::LEVELEDSHE); keys_ = context_->KeyGen(); context_->EvalMultKeyGen(keys_.secretKey);
        std::vector<int32_t> rotations; for (const auto &layer : model_.layers) for (std::size_t step = 1; step < layer.weights.front().size(); ++step) rotations.push_back(static_cast<int32_t>(step)); context_->EvalRotateKeyGen(keys_.secretKey, rotations); context_->LoadContext(keys_.publicKey);
        for (const auto &layer : model_.layers) packed_layers_.push_back(cache_packed_layer(context_, layer, slots()));
    }
    std::vector<double> predict(const std::vector<double> &values) {
        auto plain = context_->MakeCKKSPackedPlaintext(values); auto ciphertext = context_->Encrypt(keys_.secretKey, plain);
        for (std::size_t index = 0; index < model_.layers.size(); ++index) {
            const auto &layer = model_.layers[index];
            ciphertext = packed_linear(context_, ciphertext, packed_layers_[index]);
            if (layer.activation != Activation::None) ciphertext = polynomial_activation(context_, ciphertext, layer.activation, slots());
        }
        fideslib::Plaintext output; context_->Decrypt(ciphertext, keys_.secretKey, &output); auto result = output->GetRealPackedValue(); result.resize(model_.layers.back().bias.size()); return result;
    }
private: std::size_t slots() const { return config_.ring_dim / 2; } Model model_; RunConfig config_; Context context_; fideslib::KeyPair<fideslib::DCRTPoly> keys_; std::vector<PackedLayerCache> packed_layers_;
};

static void print_numbers(const std::vector<double> &values) { std::cout << '['; for (std::size_t i = 0; i < values.size(); ++i) { if (i) std::cout << ','; std::cout << std::setprecision(12) << values[i]; } std::cout << ']'; }
static const std::string &model_path(const json::Value &request) { static const std::string relu = "src/mnist_mlp.json", gelu = "src/mnist_mlp_gelu.json"; const auto selected = request.has("model") ? request.at("model").string : "relu"; if (selected == "relu") return relu; if (selected == "gelu") return gelu; throw std::runtime_error("model must be relu or gelu"); }
int main(int argc, char **argv) {
    try { if (argc != 2 || std::string(argv[1]) != "--web-worker") throw std::runtime_error("SEALTorch is served through webui/server.py"); std::unique_ptr<FidesInference> inference; RunConfig active; std::string model_name, line; while (std::getline(std::cin, line)) try { const auto request = json::Parser(line).parse(); std::vector<double> pixels; for (const auto &value : request.at("pixels").array) pixels.push_back(value.number); if (pixels.size() != 784) throw std::runtime_error("pixels must contain 784 values"); const auto config = parse_config(request); const auto &selected = model_path(request); double setup_ms = 0; if (!inference || active != config || model_name != selected) { const auto started = std::chrono::steady_clock::now(); inference = std::make_unique<FidesInference>(load_model(selected), config); inference->predict(pixels); active = config; model_name = selected; setup_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count(); } const auto started = std::chrono::steady_clock::now(); const auto output = inference->predict(pixels); const auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count(); std::cout << "{\"encrypted\":"; print_numbers(output); std::cout << ",\"setup_ms\":" << setup_ms << ",\"encrypt_ms\":0,\"evaluate_ms\":" << elapsed << ",\"decrypt_ms\":0,\"encrypted_ms\":" << elapsed << ",\"input_ciphertext_bytes\":0,\"output_ciphertext_bytes\":0,\"backend\":\"packed\",\"device\":\"cuda\"}\n"; } catch (const std::exception &error) { std::cout << "{\"error\":\"" << error.what() << "\"}\n"; } std::cout.flush(); return 0; } catch (const std::exception &error) { std::cerr << "SEALTorch error: " << error.what() << '\n'; return 1; }
}
