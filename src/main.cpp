#include <SEALTorch/sealtorch.h>

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
#include <unistd.h>
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

    Value parse() {
        skip();
        Value result = value();
        skip();
        if (position_ != text_.size())
            throw std::runtime_error("unexpected text after JSON value");
        return result;
    }

private:
    std::string text_;
    std::size_t position_ = 0;

    void skip() {
        while (position_ < text_.size() &&
               std::isspace(static_cast<unsigned char>(text_[position_])))
            ++position_;
    }

    void expect(char expected) {
        skip();
        if (position_ >= text_.size() || text_[position_] != expected)
            throw std::runtime_error("invalid JSON");
        ++position_;
    }

    std::string string_value() {
        expect('"');
        std::string result;
        while (position_ < text_.size() && text_[position_] != '"') {
            char character = text_[position_++];
            if (character == '\\') {
                if (position_ >= text_.size())
                    throw std::runtime_error("invalid JSON escape");
                const char escaped = text_[position_++];
                if (escaped == 'n') character = '\n';
                else if (escaped == 'r') character = '\r';
                else if (escaped == 't') character = '\t';
                else character = escaped;
            }
            result += character;
        }
        expect('"');
        return result;
    }

    Value value() {
        skip();
        if (position_ >= text_.size())
            throw std::runtime_error("unexpected end of JSON");
        if (text_[position_] == '{')
            return object_value();
        if (text_[position_] == '[')
            return array_value();
        if (text_[position_] == '"') {
            Value result;
            result.kind = Value::Kind::string;
            result.string = string_value();
            return result;
        }

        const std::size_t begin = position_;
        while (position_ < text_.size() &&
               std::string(",]} \t\r\n").find(text_[position_]) ==
                   std::string::npos)
            ++position_;
        const std::string token = text_.substr(begin, position_ - begin);
        if (token == "null" || token == "true" || token == "false")
            return {};

        Value result;
        result.kind = Value::Kind::number;
        result.number = std::stod(token);
        return result;
    }

    Value array_value() {
        expect('[');
        Value result;
        result.kind = Value::Kind::array;
        skip();
        if (position_ < text_.size() && text_[position_] == ']') {
            ++position_;
            return result;
        }
        for (;;) {
            result.array.push_back(value());
            skip();
            if (position_ < text_.size() && text_[position_] == ']') {
                ++position_;
                return result;
            }
            expect(',');
        }
    }

    Value object_value() {
        expect('{');
        Value result;
        result.kind = Value::Kind::object;
        skip();
        if (position_ < text_.size() && text_[position_] == '}') {
            ++position_;
            return result;
        }
        for (;;) {
            const std::string key = string_value();
            expect(':');
            result.object.emplace(key, value());
            skip();
            if (position_ < text_.size() && text_[position_] == '}') {
                ++position_;
                return result;
            }
            expect(',');
        }
    }
};
}

struct ModelArtifact {
    sealtorch::Sequential encrypted;
    bool cnn = false;
    std::size_t minimum_ring_dimension = 0;
};

static void flatten_tensor(const json::Value &value, std::vector<double> &output) {
    if (value.kind == json::Value::Kind::array)
        for (const auto &item : value.array) flatten_tensor(item, output);
    else output.push_back(value.number);
}

static int checked_size(std::size_t value, const std::string &name)
{
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error(name + " is too large");
    return static_cast<int>(value);
}

static std::vector<double> read_tensor(const std::map<std::string, json::Value> &tensors,
    const std::string &key) {
    std::vector<double> result;
    const json::Value &value = tensors.at(key);
    flatten_tensor(value.has("data") ? value.at("data") : value, result);
    return result;
}

static std::size_t positive_size(const json::Value &value, const std::string &field) {
    if (value.kind != json::Value::Kind::number || value.number <= 0 ||
        std::floor(value.number) != value.number)
        throw std::runtime_error("invalid " + field);
    return static_cast<std::size_t>(value.number);
}

static std::size_t pair_value(const json::Value &layer, const std::string &field) {
    const json::Value &value = layer.at(field);
    if (value.kind == json::Value::Kind::number) return positive_size(value, field);
    if (value.kind != json::Value::Kind::array || value.array.size() != 2 ||
        positive_size(value.array[0], field) != positive_size(value.array[1], field))
        throw std::runtime_error(field + " must be a square positive pair");
    return positive_size(value.array[0], field);
}

static bool zero_pair(const json::Value &layer, const std::string &field) {
    const json::Value &value = layer.at(field);
    if (value.kind == json::Value::Kind::number) return value.number == 0;
    return value.kind == json::Value::Kind::array && value.array.size() == 2 &&
        value.array[0].kind == json::Value::Kind::number && value.array[0].number == 0 &&
        value.array[1].kind == json::Value::Kind::number && value.array[1].number == 0;
}

static sealtorch::Activation activation_for(const std::string &op) {
    if (op == "relu") return sealtorch::Activation::relu();
    if (op == "gelu") return sealtorch::Activation::gelu();
    if (op == "tanh") return sealtorch::Activation::tanh();
    throw std::runtime_error("encrypted inference does not yet approximate trainer activation: " + op);
}

static std::size_t image_index(std::size_t channel, std::size_t row, std::size_t column,
                               std::size_t height, std::size_t width) {
    return (channel * height + row) * width + column;
}

// The encrypted providers currently share a dense packed interface. Lower CNN
// operations once at model load, then keep only their non-zero CKKS diagonals.
static sealtorch::DenseLayer lower_convolution(const std::vector<double> &weights,
    const std::vector<double> &biases, std::size_t channels, std::size_t height, std::size_t width,
    std::size_t outputs, std::size_t kernel_height, std::size_t kernel_width, std::size_t stride) {
    const std::size_t expected_weights =
        outputs * channels * kernel_height * kernel_width;
    if (channels == 0 || outputs == 0 || kernel_height == 0 ||
        kernel_width == 0 || stride == 0 ||
        kernel_height > height || kernel_width > width ||
        weights.size() != expected_weights || biases.size() != outputs)
        throw std::runtime_error("invalid convolution tensor dimensions");
    const std::size_t out_height = (height - kernel_height) / stride + 1;
    const std::size_t out_width = (width - kernel_width) / stride + 1;
    sealtorch::DenseLayer result{
        checked_size(channels * height * width, "convolution input"),
        checked_size(outputs * out_height * out_width, "convolution output"),
        {},
        {},
    };
    result.weights.assign(static_cast<std::size_t>(result.output_size), std::vector<double>(result.input_size, 0.0));
    result.biases.resize(static_cast<std::size_t>(result.output_size));
    for (std::size_t output = 0; output < outputs; ++output)
        for (std::size_t row = 0; row < out_height; ++row)
            for (std::size_t column = 0; column < out_width; ++column) {
                const std::size_t destination = image_index(output, row, column, out_height, out_width);
                result.biases[destination] = biases[output];
                for (std::size_t channel = 0; channel < channels; ++channel)
                    for (std::size_t ky = 0; ky < kernel_height; ++ky)
                        for (std::size_t kx = 0; kx < kernel_width; ++kx)
                            result.weights[destination][image_index(
                                channel,
                                row * stride + ky,
                                column * stride + kx,
                                height,
                                width)] = weights[
                                    (((output * channels + channel) *
                                      kernel_height + ky) *
                                     kernel_width + kx)];
            }
    return result;
}

static sealtorch::DenseLayer lower_average_pool(std::size_t channels, std::size_t height, std::size_t width,
    std::size_t pool_height, std::size_t pool_width, std::size_t stride) {
    if (channels == 0 || pool_height == 0 || pool_width == 0 || stride == 0 ||
        pool_height > height || pool_width > width)
        throw std::runtime_error("invalid average-pooling dimensions");
    const std::size_t out_height = (height - pool_height) / stride + 1;
    const std::size_t out_width = (width - pool_width) / stride + 1;
    sealtorch::DenseLayer result{
        checked_size(channels * height * width, "pooling input"),
        checked_size(channels * out_height * out_width, "pooling output"),
        {},
        {},
    };
    result.weights.assign(static_cast<std::size_t>(result.output_size), std::vector<double>(result.input_size, 0.0));
    result.biases.assign(static_cast<std::size_t>(result.output_size), 0.0);
    const double coefficient = 1.0 / static_cast<double>(pool_height * pool_width);
    for (std::size_t channel = 0; channel < channels; ++channel)
        for (std::size_t row = 0; row < out_height; ++row)
            for (std::size_t column = 0; column < out_width; ++column) {
                const std::size_t destination = image_index(channel, row, column, out_height, out_width);
                for (std::size_t py = 0; py < pool_height; ++py)
                    for (std::size_t px = 0; px < pool_width; ++px)
                        result.weights[destination][image_index(
                            channel,
                            row * stride + py,
                            column * stride + px,
                            height,
                            width)] = coefficient;
            }
    return result;
}

static ModelArtifact load_model(const std::string &path) {
    std::ifstream file(path); if (!file) throw std::runtime_error("cannot open model: " + path);
    std::stringstream contents; contents << file.rdbuf(); const auto root = json::Parser(contents.str()).parse();
    const auto &tensors = root.at("tensors").object;
    ModelArtifact result;
    if (root.has("format") && root.at("format").string == "sealtorch-lenet-trainer-v1") {
        result.cnn = true;
        const auto &architecture = root.at("architecture");
        const auto &shape = architecture.at("input").at("shape").array;
        if (shape.size() != 4 || positive_size(shape[0], "batch") != 1)
            throw std::runtime_error("trainer export must have a single NCHW input");
        std::size_t channels = positive_size(shape[1], "input channels");
        std::size_t height = positive_size(shape[2], "input height");
        std::size_t width = positive_size(shape[3], "input width");
        std::size_t widest = channels * height * width;
        for (const auto &layer : architecture.at("layers").array) {
            const std::string op = layer.at("op").string;
            if (op == "conv2d") {
                if (!zero_pair(layer, "padding") || pair_value(layer, "dilation") != 1 ||
                    positive_size(layer.at("groups"), "groups") != 1)
                    throw std::runtime_error("encrypted trainer convolution requires zero padding, dilation 1, and groups 1");
                const std::size_t input_channels = positive_size(layer.at("in_channels"), "in_channels");
                const std::size_t outputs = positive_size(layer.at("out_channels"), "out_channels");
                if (input_channels != channels) throw std::runtime_error("trainer convolution channel mismatch");
                const std::size_t kernel = pair_value(layer, "kernel");
                const std::size_t stride = pair_value(layer, "stride");
                result.encrypted.add(sealtorch::Linear(lower_convolution(read_tensor(tensors, layer.at("weight_key").string), read_tensor(tensors, layer.at("bias_key").string), channels, height, width, outputs, kernel, kernel, stride)));
                height = (height - kernel) / stride + 1; width = (width - kernel) / stride + 1; channels = outputs;
                widest = std::max(widest, channels * height * width);
            } else if (op == "avg_pool2d") {
                if (!zero_pair(layer, "padding"))
                    throw std::runtime_error("encrypted trainer average pooling requires zero padding");
                const std::size_t kernel = pair_value(layer, "kernel"), stride = pair_value(layer, "stride");
                result.encrypted.add(sealtorch::Linear(lower_average_pool(channels, height, width, kernel, kernel, stride)));
                height = (height - kernel) / stride + 1; width = (width - kernel) / stride + 1;
                widest = std::max(widest, channels * height * width);
            } else if (op == "max_pool2d") {
                throw std::runtime_error("encrypted inference does not support trainer max_pool2d; use --pooling avg");
            } else if (op == "flatten") {
                // Lowered convolution/pooling tensors are already flat NCHW vectors.
            } else if (op == "linear") {
                const std::vector<double> weights = read_tensor(tensors, layer.at("weight_key").string);
                const std::vector<double> biases = read_tensor(tensors, layer.at("bias_key").string);
                const std::size_t output = positive_size(layer.at("out_features"), "out_features");
                const std::size_t input = positive_size(layer.at("in_features"), "in_features");
                if (weights.size() != input * output || biases.size() != output)
                    throw std::runtime_error("invalid trainer linear tensor");
                sealtorch::DenseLayer dense{checked_size(input, "linear input"), checked_size(output, "linear output"), {}, biases};
                dense.weights.resize(output, std::vector<double>(input));
                for (std::size_t row = 0; row < output; ++row)
                    std::copy_n(weights.begin() + row * input, input, dense.weights[row].begin());
                result.encrypted.add(sealtorch::Linear(std::move(dense)));
            } else if (op == "tanh" || op == "relu" || op == "gelu" || op == "sigmoid" || op == "leaky_relu" || op == "elu" || op == "silu") {
                result.encrypted.add(activation_for(op));
            } else {
                throw std::runtime_error("unsupported trainer operation: " + op);
            }
        }
        result.minimum_ring_dimension = 2;
        while (result.minimum_ring_dimension / 2 < widest)
            result.minimum_ring_dimension *= 2;
        return result;
    }
    throw std::runtime_error("only translated LeNet trainer exports are supported");
}

struct RunConfig {
    bool packed = true;
    std::size_t threads = 4;
    std::string device = "auto";
    int ring_dim = 16384;
    int depth = 15;
    int scaling_mod_bits = 40;
    int first_mod_bits = 50;
    int scale_bits = 40;

    bool operator==(const RunConfig &) const = default;
};

static std::string string_or(
    const json::Value &object,
    const std::string &name,
    const std::string &fallback)
{
    return object.has(name) ? object.at(name).string : fallback;
}

static int int_or(
    const json::Value &object,
    const std::string &name,
    int fallback)
{
    if (!object.has(name))
        return fallback;
    const auto &value = object.at(name);
    if (value.kind == json::Value::Kind::string)
        return std::stoi(value.string);
    return static_cast<int>(value.number);
}

static RunConfig parse_config(const json::Value &request)
{
    const json::Value &value = request.has("config") ? request.at("config") : request;
    RunConfig config;
    config.packed = string_or(value, "backend", "packed") != "scalar";
    config.device = string_or(
        value, "device", string_or(value, "plaintext_device", "auto"));

    const int threads = int_or(value, "threads", 4);
    config.ring_dim = int_or(value, "ring_dim", 16384);
    config.depth = int_or(value, "depth", 15);
    config.scaling_mod_bits = int_or(value, "scaling_mod_bits", 40);
    config.first_mod_bits = int_or(value, "first_mod_bits", 50);
    config.scale_bits = int_or(value, "scale_bits", 40);

    if (threads < 1)
        throw std::runtime_error("threads must be greater than zero");
    config.threads = static_cast<std::size_t>(threads);
    if (config.ring_dim < 1024 ||
        (config.ring_dim & (config.ring_dim - 1)) != 0)
        throw std::runtime_error(
            "ring_dim must be a power of two of at least 1024");
    if (config.depth < 1 || config.scaling_mod_bits < 1 ||
        config.first_mod_bits < 1 || config.scale_bits < 1)
        throw std::runtime_error("ciphertext bit sizes must be positive");
    if (config.device != "auto" &&
        config.device != "cpu" &&
        config.device != "cuda")
        throw std::runtime_error(
            "ciphertext inference device must be auto, cpu, or cuda");
    return config;
}

static sealtorch::CiphertextInferenceOptions ciphertext_inference_options(
    const RunConfig &config)
{
    return {
        config.device == "cpu"
            ? sealtorch::ExecutionTarget::Cpu
            : config.device == "cuda"
                ? sealtorch::ExecutionTarget::Cuda
                : sealtorch::ExecutionTarget::Auto,
        config.packed
            ? sealtorch::CiphertextLayout::Packed
            : sealtorch::CiphertextLayout::Scalar,
        config.threads,
        static_cast<std::size_t>(config.ring_dim),
        static_cast<std::size_t>(config.depth),
        static_cast<std::size_t>(config.scaling_mod_bits),
        static_cast<std::size_t>(config.first_mod_bits),
        static_cast<std::size_t>(config.scale_bits)
    };
}

static std::size_t resident_memory_bytes()
{
    std::ifstream file("/proc/self/statm");
    std::size_t total_pages = 0;
    std::size_t resident_pages = 0;
    if (!(file >> total_pages >> resident_pages))
        return 0;
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0)
        return 0;
    return resident_pages * static_cast<std::size_t>(page_size);
}

static void print_numbers(const std::vector<double> &values)
{
    std::cout << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0)
            std::cout << ',';
        std::cout << std::setprecision(12) << values[index];
    }
    std::cout << ']';
}

static void print_json_string(const std::string &value)
{
    std::cout << '"';
    for (char character : value) {
        if (character == '"' || character == '\\')
            std::cout << '\\';
        if (character == '\n') {
            std::cout << "\\n";
            continue;
        }
        std::cout << character;
    }
    std::cout << '"';
}

static std::string model_path(const json::Value &request)
{
    if (request.has("model_path"))
        return request.at("model_path").string;
    throw std::runtime_error("a translated LeNet trainer model_path is required");
}

static std::vector<double> request_pixels(const json::Value &request)
{
    const json::Value &pixels = request.at("pixels");
    if (pixels.array.size() != 784 && pixels.array.size() != 1024)
        throw std::runtime_error("pixels must contain 784 or 1024 values");

    std::vector<double> input;
    input.reserve(pixels.array.size());
    for (const json::Value &pixel : pixels.array)
        input.push_back(pixel.number);
    return input;
}

static void print_result(
    const sealtorch::CiphertextInferenceResult &result,
    double setup_ms,
    std::size_t memory_before,
    std::size_t memory_after,
    const RunConfig &config,
    bool used_cuda)
{
    std::cout << "{\"encrypted\":";
    print_numbers(result.values);
    std::cout
        << ",\"setup_ms\":" << setup_ms
        << ",\"encrypt_ms\":" << result.encrypt_ms
        << ",\"evaluate_ms\":" << result.evaluate_ms
        << ",\"decrypt_ms\":" << result.decrypt_ms
        << ",\"encrypted_ms\":"
        << result.encrypt_ms + result.evaluate_ms + result.decrypt_ms
        << ",\"memory_before_bytes\":" << memory_before
        << ",\"memory_after_bytes\":" << memory_after
        << ",\"memory_delta_bytes\":"
        << static_cast<long long>(memory_after) -
               static_cast<long long>(memory_before)
        << ",\"input_ciphertext_bytes\":" << result.input_ciphertext_bytes
        << ",\"output_ciphertext_bytes\":" << result.output_ciphertext_bytes
        << ",\"input_ciphertext_memory_bytes\":"
        << result.input_ciphertext_memory_bytes
        << ",\"output_ciphertext_memory_bytes\":"
        << result.output_ciphertext_memory_bytes
        << ",\"secret_key_bytes\":" << result.secret_key_bytes
        << ",\"relin_keys_bytes\":" << result.relin_keys_bytes
        << ",\"galois_keys_bytes\":" << result.galois_keys_bytes
        << ",\"backend\":\"" << (config.packed ? "packed" : "scalar")
        << "\",\"device\":\"" << (used_cuda ? "cuda" : "cpu")
        << "\"}\n";
}

static int run_web_worker()
{
    std::unique_ptr<sealtorch::CiphertextInference> ciphertext_inference;
    std::string active_model;
    RunConfig active_config;
    std::string line;

    while (std::getline(std::cin, line)) {
        try {
            const json::Value request = json::Parser(line).parse();
            const RunConfig config = parse_config(request);
            const std::string selected_model = model_path(request);
            const std::vector<double> input = request_pixels(request);
            const bool rebuild =
                !ciphertext_inference ||
                config != active_config ||
                selected_model != active_model;
            double setup_ms = 0.0;

            if (rebuild) {
                const auto setup_start = std::chrono::steady_clock::now();
                ModelArtifact artifact = load_model(selected_model);
                sealtorch::CiphertextInferenceOptions options =
                    ciphertext_inference_options(config);
                if (artifact.cnn) {
                    // Seven linear transforms and four cubic tanh operations
                    // consume 19 levels. Keep one additional level as margin.
                    options.multiplicative_depth =
                        std::max(options.multiplicative_depth, std::size_t{20});

                    const std::size_t required_ring = artifact.minimum_ring_dimension ?
                        artifact.minimum_ring_dimension : std::size_t{8192};
                    if (options.ring_dimension < required_ring)
                        throw std::runtime_error(
                            "LeNet ciphertext inference needs a larger ring_dim for its widest feature map");
                    options.ring_dimension = required_ring;
                }
                ciphertext_inference =
                    std::make_unique<sealtorch::CiphertextInference>(
                        std::move(artifact.encrypted), options);

                // Warm provider allocations once. This cost is reported as
                // setup and excluded from steady-state inference latency.
                ciphertext_inference->predict(input);
                active_config = config;
                active_model = selected_model;
                setup_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - setup_start).count();
            }

            const std::size_t memory_before = resident_memory_bytes();
            const sealtorch::CiphertextInferenceResult encrypted =
                ciphertext_inference->predict(input);
            const bool used_cuda =
                ciphertext_inference->target() ==
                sealtorch::ExecutionTarget::Cuda;
            const std::size_t memory_after = resident_memory_bytes();
            print_result(
                encrypted, setup_ms, memory_before, memory_after,
                config, used_cuda);
        } catch (const std::exception &error) {
            std::cout << "{\"error\":";
            print_json_string(error.what());
            std::cout << "}\n";
        }
        std::cout.flush();
    }
    return 0;
}

int main(int argc, char **argv)
{
    try {
        if (argc == 2 && std::string(argv[1]) == "--web-worker")
            return run_web_worker();
        if (argc == 2 && std::string(argv[1]) == "--capabilities") {
            std::cout
                << "{\"cpu\":true,\"cuda\":"
                << (sealtorch::CiphertextInference::cuda_available()
                        ? "true"
                        : "false")
                << "}\n";
            return 0;
        }
        throw std::runtime_error(
            "run webui/server.py, or use --web-worker");
    } catch (const std::exception &error) {
        std::cerr << "SEALTorch error: " << error.what() << '\n';
        return 1;
    }
}
