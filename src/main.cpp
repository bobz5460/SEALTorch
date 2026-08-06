#include "inference.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace json {
struct Value {
    enum class Kind { null_value, number, string, array, object } kind = Kind::null_value;
    double number = 0.0;
    std::string string;
    std::vector<Value> array;
    std::map<std::string, Value> object;
    const Value& at(const std::string& key) const { return object.at(key); }
    bool has(const std::string& key) const { return object.contains(key); }
};

class Parser {
  public:
    explicit Parser(std::string text) : text_(std::move(text)) {}
    Value parse() { skip(); auto result = value(); skip(); if (position_ != text_.size()) fail(); return result; }
  private:
    std::string text_; std::size_t position_ = 0;
    [[noreturn]] void fail() const { throw std::runtime_error("invalid JSON"); }
    void skip() { while (position_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[position_]))) ++position_; }
    void expect(char character) { skip(); if (position_ >= text_.size() || text_[position_] != character) fail(); ++position_; }
    std::string string_value() {
        expect('"'); std::string result;
        while (position_ < text_.size() && text_[position_] != '"') {
            char character = text_[position_++];
            if (character == '\\') {
                if (position_ >= text_.size()) fail();
                const char escaped = text_[position_++];
                if (escaped == 'n') character = '\n'; else if (escaped == 'r') character = '\r';
                else if (escaped == 't') character = '\t'; else character = escaped;
            }
            result += character;
        }
        expect('"'); return result;
    }
    Value value() {
        skip(); if (position_ >= text_.size()) fail();
        if (text_[position_] == '{') return object_value();
        if (text_[position_] == '[') return array_value();
        if (text_[position_] == '"') { Value result; result.kind = Value::Kind::string; result.string = string_value(); return result; }
        const std::size_t begin = position_;
        while (position_ < text_.size() && std::string(",]} \t\r\n").find(text_[position_]) == std::string::npos) ++position_;
        const std::string token = text_.substr(begin, position_ - begin);
        if (token == "null" || token == "true" || token == "false") return {};
        Value result; result.kind = Value::Kind::number;
        try { result.number = std::stod(token); } catch (...) { fail(); }
        return result;
    }
    Value array_value() {
        expect('['); Value result; result.kind = Value::Kind::array; skip();
        if (position_ < text_.size() && text_[position_] == ']') { ++position_; return result; }
        for (;;) { result.array.push_back(value()); skip(); if (position_ < text_.size() && text_[position_] == ']') { ++position_; return result; } expect(','); }
    }
    Value object_value() {
        expect('{'); Value result; result.kind = Value::Kind::object; skip();
        if (position_ < text_.size() && text_[position_] == '}') { ++position_; return result; }
        for (;;) { const auto key = string_value(); expect(':'); result.object.emplace(key, value()); skip(); if (position_ < text_.size() && text_[position_] == '}') { ++position_; return result; } expect(','); }
    }
};
} // namespace json

namespace {

int checked_int(std::size_t value, const std::string& name) {
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error(name + " is too large");
    return static_cast<int>(value);
}

std::size_t positive(const json::Value& value, const std::string& name) {
    if (value.kind != json::Value::Kind::number || value.number <= 0 || std::floor(value.number) != value.number)
        throw std::runtime_error("invalid " + name);
    return static_cast<std::size_t>(value.number);
}

std::vector<double> numbers(const json::Value& value) {
    std::vector<double> result; result.reserve(value.array.size());
    for (const auto& item : value.array) {
        if (item.kind != json::Value::Kind::number || !std::isfinite(item.number))
            throw std::runtime_error("expected finite number array");
        result.push_back(item.number);
    }
    return result;
}

std::size_t image_index(std::size_t channel, std::size_t row, std::size_t column,
                        std::size_t height, std::size_t width) {
    return (channel * height + row) * width + column;
}

sealtorch::DenseLayer lower_convolution(const std::vector<double>& weights,
    const std::vector<double>& biases, std::size_t channels, std::size_t height,
    std::size_t width, std::size_t outputs, std::size_t kernel) {
    const std::size_t out_height = height - kernel + 1, out_width = width - kernel + 1;
    if (kernel == 0 || kernel > height || kernel > width ||
        weights.size() != outputs * channels * kernel * kernel || biases.size() != outputs)
        throw std::runtime_error("invalid convolution tensors");
    sealtorch::DenseLayer result{ checked_int(channels * height * width, "convolution input"),
        checked_int(outputs * out_height * out_width, "convolution output"), {}, {} };
    result.weights.assign(result.output_size, std::vector<double>(result.input_size, 0.0));
    result.biases.resize(result.output_size);
    for (std::size_t output = 0; output < outputs; ++output)
        for (std::size_t row = 0; row < out_height; ++row)
            for (std::size_t column = 0; column < out_width; ++column) {
                const auto destination = image_index(output, row, column, out_height, out_width);
                result.biases[destination] = biases[output];
                for (std::size_t channel = 0; channel < channels; ++channel)
                    for (std::size_t y = 0; y < kernel; ++y)
                        for (std::size_t x = 0; x < kernel; ++x)
                            result.weights[destination][image_index(channel, row + y, column + x, height, width)] =
                                weights[(((output * channels + channel) * kernel + y) * kernel + x)];
            }
    return result;
}

sealtorch::DenseLayer lower_pool(std::size_t channels, std::size_t height, std::size_t width,
                                 std::size_t kernel, std::size_t stride) {
    const std::size_t out_height = (height - kernel) / stride + 1, out_width = (width - kernel) / stride + 1;
    sealtorch::DenseLayer result{ checked_int(channels * height * width, "pool input"),
        checked_int(channels * out_height * out_width, "pool output"), {}, {} };
    result.weights.assign(result.output_size, std::vector<double>(result.input_size, 0.0));
    result.biases.assign(result.output_size, 0.0);
    const double factor = 1.0 / static_cast<double>(kernel * kernel);
    for (std::size_t channel = 0; channel < channels; ++channel)
        for (std::size_t row = 0; row < out_height; ++row)
            for (std::size_t column = 0; column < out_width; ++column) {
                const auto destination = image_index(channel, row, column, out_height, out_width);
                for (std::size_t y = 0; y < kernel; ++y)
                    for (std::size_t x = 0; x < kernel; ++x)
                        result.weights[destination][image_index(channel, row * stride + y, column * stride + x, height, width)] = factor;
            }
    return result;
}

struct LoadedModel { sealtorch::Model model; std::size_t minimum_ring = 0; std::string activation; };

LoadedModel load_model(const std::string& path) {
    std::ifstream file(path); if (!file) throw std::runtime_error("cannot open model: " + path);
    std::stringstream contents; contents << file.rdbuf(); const auto root = json::Parser(contents.str()).parse();
    if (!root.has("format") || root.at("format").string != "sealtorch-model-v1")
        throw std::runtime_error("unsupported model artifact");
    const auto& state = root.at("state_dict");
    auto tensor = [&](const std::string& name) { return numbers(state.at(name).at("data")); };
    const auto& architecture = root.at("architecture");
    const auto& shape = architecture.at("input_shape").array;
    if (shape.size() != 3) throw std::runtime_error("input_shape must be CHW");
    std::size_t channels = positive(shape[0], "channels"), height = positive(shape[1], "height"), width = positive(shape[2], "width");
    std::size_t widest = channels * height * width;
    LoadedModel loaded; loaded.activation = root.at("activation").string;
    for (const auto& layer : architecture.at("layers").array) {
        const std::string op = layer.at("op").string;
        if (op == "flatten") continue;
        if (op == "conv2d") {
            const auto outputs = positive(layer.at("out_channels"), "out_channels");
            const auto kernel = positive(layer.at("kernel"), "kernel");
            loaded.model.add(lower_convolution(tensor(layer.at("weight").string), tensor(layer.at("bias").string),
                                               channels, height, width, outputs, kernel));
            height -= kernel - 1; width -= kernel - 1; channels = outputs;
        } else if (op == "avg_pool2d") {
            const auto kernel = positive(layer.at("kernel"), "pool kernel");
            const auto stride = positive(layer.at("stride"), "pool stride");
            loaded.model.add(lower_pool(channels, height, width, kernel, stride));
            height = (height - kernel) / stride + 1; width = (width - kernel) / stride + 1;
        } else if (op == "linear") {
            const auto input = positive(layer.at("in"), "linear input"), output = positive(layer.at("out"), "linear output");
            auto weights = tensor(layer.at("weight").string); auto biases = tensor(layer.at("bias").string);
            if (weights.size() != input * output || biases.size() != output) throw std::runtime_error("invalid linear tensors");
            sealtorch::DenseLayer dense{ checked_int(input, "linear input"), checked_int(output, "linear output"), {}, std::move(biases) };
            dense.weights.resize(output, std::vector<double>(input));
            for (std::size_t row = 0; row < output; ++row)
                std::copy_n(weights.begin() + row * input, input, dense.weights[row].begin());
            loaded.model.add(std::move(dense));
        } else if (op == "relu" || op == "gelu" || op == "tanh") {
            if (op != loaded.activation) throw std::runtime_error("mixed activations are not supported");
            loaded.model.activation();
        } else throw std::runtime_error("unsupported operation: " + op);
        widest = std::max(widest, channels * height * width);
    }
    loaded.minimum_ring = 2;
    while (loaded.minimum_ring / 2 < widest) loaded.minimum_ring *= 2;
    return loaded;
}

struct Config {
    std::string path; std::size_t ring = 32768, depth = 12, scale = 40, first = 50;
    double lower = -1.0, upper = 1.0; int device = 0; bool stream_weights = true;
    std::vector<double> coefficients;
    bool operator==(const Config&) const = default;
};

int integer_or(const json::Value& value, const std::string& key, int fallback) {
    return value.has(key) ? static_cast<int>(value.at(key).number) : fallback;
}
double number_or(const json::Value& value, const std::string& key, double fallback) {
    return value.has(key) ? value.at(key).number : fallback;
}

Config parse_config(const json::Value& request) {
    Config config;
    config.path = request.at("model_path").string;
    config.ring = static_cast<std::size_t>(integer_or(request, "ring_dim", 32768));
    config.depth = static_cast<std::size_t>(integer_or(request, "depth", 12));
    config.scale = static_cast<std::size_t>(integer_or(request, "scaling_mod_bits", 40));
    config.first = static_cast<std::size_t>(integer_or(request, "first_mod_bits", 50));
    config.device = integer_or(request, "device", 0);
    config.stream_weights = integer_or(request, "stream_weights", 1) != 0;
    config.lower = number_or(request, "lower_bound", -1.0);
    config.upper = number_or(request, "upper_bound", 1.0);
    config.coefficients = numbers(request.at("coefficients"));
    if (!(config.lower < config.upper)) throw std::runtime_error("invalid polynomial interval");
    return config;
}

void print_string(const std::string& value) {
    std::cout << '"'; for (const char character : value) { if (character == '"' || character == '\\') std::cout << '\\'; if (character == '\n') std::cout << "\\n"; else std::cout << character; } std::cout << '"';
}
void print_values(const std::vector<double>& values) {
    std::cout << '['; for (std::size_t index = 0; index < values.size(); ++index) { if (index) std::cout << ','; std::cout << std::setprecision(12) << values[index]; } std::cout << ']';
}

int worker() {
    std::unique_ptr<sealtorch::InferenceEngine> engine; Config active; std::string line;
    while (std::getline(std::cin, line)) {
        try {
            const auto request = json::Parser(line).parse(); const auto config = parse_config(request);
            const auto input = numbers(request.at("pixels")); double setup_ms = 0.0;
            if (!engine || !(config == active)) {
                const auto start = std::chrono::steady_clock::now(); auto loaded = load_model(config.path);
                if (config.ring < loaded.minimum_ring) throw std::runtime_error("ring dimension is too small for this model");
                sealtorch::InferenceOptions options{ config.ring, config.depth, config.scale, config.first,
                    config.lower, config.upper, config.coefficients, config.device, config.stream_weights };
                // Release the previous context before allocating the next one.
                // Constructing first and assigning afterwards temporarily kept
                // two complete CKKS contexts on the same GPU during sweeps.
                engine.reset();
                engine = std::make_unique<sealtorch::InferenceEngine>(std::move(loaded.model), std::move(options));
                engine->predict(input); active = config;
                setup_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            }
            const auto result = engine->predict(input);
            std::cout << "{\"logits\":"; print_values(result.values);
            std::cout << ",\"setup_ms\":" << setup_ms << ",\"encrypt_ms\":" << result.encrypt_ms
                      << ",\"evaluate_ms\":" << result.evaluate_ms << ",\"decrypt_ms\":" << result.decrypt_ms
                      << ",\"total_ms\":" << result.encrypt_ms + result.evaluate_ms + result.decrypt_ms
                      << ",\"idle_ram_bytes\":" << result.idle_ram_bytes
                      << ",\"ram_bytes\":" << result.ram_bytes
                      << ",\"idle_vram_bytes\":" << result.idle_vram_bytes
                      << ",\"vram_bytes\":" << result.vram_bytes << "}\n";
        } catch (const std::exception& error) { std::cout << "{\"error\":"; print_string(error.what()); std::cout << "}\n"; }
        std::cout.flush();
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--worker") return worker();
        if (argc == 2 && std::string(argv[1]) == "--capabilities") {
            std::cout << "{\"fideslib\":true,\"cuda\":" << (sealtorch::InferenceEngine::available() ? "true" : "false") << "}\n"; return 0;
        }
        if (argc == 3 && std::string(argv[1]) == "--validate-model") {
            const auto model = load_model(argv[2]);
            std::cout << "{\"input_size\":" << model.model.input_size() << ",\"output_size\":" << model.model.output_size()
                      << ",\"activations\":" << model.model.activation_count() << ",\"minimum_ring_dimension\":" << model.minimum_ring << "}\n"; return 0;
        }
        throw std::runtime_error("use --worker, --capabilities, or --validate-model <artifact>");
    } catch (const std::exception& error) { std::cerr << "SEALTorch error: " << error.what() << '\n'; return 1; }
}
