#include <SEALTorch/evaluator.h>
#include <SEALTorch/math.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <algorithm>
#include <utility>
#include <vector>

namespace json {
struct Value {
    enum class Kind { null_value, number, string, array, object } kind = Kind::null_value;
    double number = 0.0;
    std::string string;
    std::vector<Value> array;
    std::map<std::string, Value> object;
    const Value &at(const std::string &key) const { return object.at(key); }
};

class Parser {
public:
    explicit Parser(std::string text) : text_(std::move(text)) {}
    Value parse() { skip(); return value(); }
private:
    std::string text_; std::size_t position_ = 0;
    void skip() { while (position_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[position_]))) ++position_; }
    void expect(char c) { skip(); if (position_ >= text_.size() || text_[position_++] != c) throw std::runtime_error("invalid JSON"); }
    std::string string_value() {
        expect('"'); std::string result;
        while (position_ < text_.size() && text_[position_] != '"') {
            if (text_[position_] == '\\' && position_ + 1 < text_.size()) { ++position_; result += text_[position_++]; }
            else result += text_[position_++];
        }
        expect('"'); return result;
    }
    Value value() {
        skip(); if (position_ >= text_.size()) throw std::runtime_error("unexpected end of JSON");
        if (text_[position_] == '{') return object_value();
        if (text_[position_] == '[') return array_value();
        if (text_[position_] == '"') { Value result; result.kind = Value::Kind::string; result.string = string_value(); return result; }
        const std::size_t start = position_;
        while (position_ < text_.size() && std::string(",]} \t\r\n").find(text_[position_]) == std::string::npos) ++position_;
        const std::string token = text_.substr(start, position_ - start);
        if (token == "null" || token == "true" || token == "false") return {};
        Value result; result.kind = Value::Kind::number; result.number = std::stod(token); return result;
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
}

struct TraceLayer {
    std::vector<double> plaintext_pre;
    std::vector<double> ckks_pre;
    std::vector<double> plaintext_post;
    std::vector<double> ckks_post;
};

static NeuralNetwork load_model(const std::string &path)
{
    std::ifstream file(path); if (!file) throw std::runtime_error("cannot open model: " + path);
    std::stringstream contents; contents << file.rdbuf();
    const auto root = json::Parser(contents.str()).parse();
    const auto &tensors = root.at("tensors").object;
    NeuralNetwork model{784, 10, {}};
    for (const auto &names : {std::pair<std::string, std::string>{"network.1.weight", "network.1.bias"},
                              {"network.3.weight", "network.3.bias"}, {"network.5.weight", "network.5.bias"}}) {
        const auto &weights = tensors.at(names.first).at("data").array;
        const auto &biases = tensors.at(names.second).at("data").array;
        DenseLayer layer{static_cast<int>(weights.front().array.size()), static_cast<int>(weights.size()), {}, {}};
        layer.weights.resize(weights.size()); layer.biases.resize(biases.size());
        for (std::size_t output = 0; output < weights.size(); ++output) {
            for (const auto &weight : weights[output].array) layer.weights[output].push_back(weight.number);
            layer.biases[output] = biases[output].number;
        }
        model.layers.push_back(std::move(layer));
    }
    return model;
}

static std::vector<double> plaintext_layer(const DenseLayer &layer, const std::vector<double> &input)
{
    std::vector<double> result(layer.output_size, 0.0);
    for (int output = 0; output < layer.output_size; ++output) {
        result[output] = layer.biases[output];
        for (int index = 0; index < layer.input_size; ++index) result[output] += layer.weights[output][index] * input[index];
    }
    return result;
}

static double approximate_gelu_plain(double value)
{
    return 0.06762090 + 0.5 * value + 0.48257484 * value * value
        - 0.06659632 * value * value * value * value;
}

static std::vector<double> decrypt(const std::vector<seal::Ciphertext> &values,
                                   seal::Decryptor &decryptor, seal::CKKSEncoder &encoder)
{
    std::vector<double> result; result.reserve(values.size());
    for (const auto &value : values) {
        seal::Plaintext plain; decryptor.decrypt(value, plain);
        std::vector<double> decoded; encoder.decode(plain, decoded);
        if (decoded.empty()) throw std::runtime_error("empty decrypted trace value");
        result.push_back(decoded.front());
    }
    return result;
}

static std::vector<double> read_input()
{
    std::vector<double> input(784);
    for (double &value : input) if (!(std::cin >> value)) throw std::runtime_error("expected 784 input values on stdin");
    return input;
}

static void write_array(const std::vector<double> &values)
{
    std::cout << '[' << std::setprecision(17);
    for (std::size_t index = 0; index < values.size(); ++index) { if (index) std::cout << ','; std::cout << values[index]; }
    std::cout << ']';
}

int main(int argc, char **argv)
{
    try {
        std::string model_path = "src/mnist_mlp_gelu_improved.json";
        std::size_t max_concurrency = 4;
        for (int arg = 1; arg < argc; ++arg) {
            const std::string value = argv[arg];
            if (value == "--threads" && arg + 1 < argc) max_concurrency = std::stoull(argv[++arg]);
            else if (value == "--help") { std::cout << "usage: sealtorch_trace [model.json] [--threads N]\n"; return 0; }
            else if (!value.empty() && value[0] != '-') model_path = value;
            else throw std::runtime_error("unknown argument: " + value);
        }
        const auto model = load_model(model_path);
        const auto input = read_input();
        std::vector<double> plaintext = input;

        seal::EncryptionParameters parameters(seal::scheme_type::ckks);
        parameters.set_poly_modulus_degree(16384);
        parameters.set_coeff_modulus(seal::CoeffModulus::Create(16384, {60,30,30,30,30,30,30,30,30,30,30,60}));
        seal::SEALContext context(parameters); seal::KeyGenerator key_generator(context);
        seal::Encryptor encryptor(context, key_generator.secret_key()); seal::Decryptor decryptor(context, key_generator.secret_key());
        seal::Evaluator evaluator(context); seal::CKKSEncoder encoder(context);
        seal::RelinKeys relin_keys; seal::GaloisKeys galois_keys;
        key_generator.create_relin_keys(relin_keys); key_generator.create_galois_keys(galois_keys);
        constexpr double scale = 1073741824.0;
        seal::Plaintext encoded; encoder.encode(input, scale, encoded); seal::Ciphertext encrypted; encryptor.encrypt_symmetric(encoded, encrypted);
        std::vector<seal::Ciphertext> encrypted_values{std::move(encrypted)};
        sealtorch::Evaluator model_evaluator(model, max_concurrency);
        std::vector<TraceLayer> traces(model.layers.size());
        double plaintext_ms = 0.0;
        const auto plaintext_total_start = std::chrono::steady_clock::now();
        for (std::size_t index = 0; index < model.layers.size(); ++index) {
            const auto &layer = model.layers[index];
            traces[index].plaintext_pre = plaintext_layer(layer, plaintext);
            traces[index].plaintext_post = traces[index].plaintext_pre;
            if (index + 1 != model.layers.size()) {
                for (double &value : traces[index].plaintext_post) value = approximate_gelu_plain(value);
            }
            plaintext = traces[index].plaintext_post;
        }
        const auto plaintext_total_end = std::chrono::steady_clock::now();
        plaintext_ms = std::chrono::duration<double, std::milli>(plaintext_total_end - plaintext_total_start).count();

        const auto encrypted_start = std::chrono::steady_clock::now();
        model_evaluator.predict(encrypted_values, evaluator, relin_keys, galois_keys, encoder, scale, {},
            [&](std::size_t layer, bool after_activation, const std::vector<seal::Ciphertext> &values) {
                if (after_activation) traces[layer].ckks_post = decrypt(values, decryptor, encoder);
                else traces[layer].ckks_pre = decrypt(values, decryptor, encoder);
            });
        const auto encrypted_end = std::chrono::steady_clock::now();
        const double encrypted_ms = std::chrono::duration<double, std::milli>(encrypted_end - encrypted_start).count();

        struct rusage usage{};
        getrusage(RUSAGE_SELF, &usage);
        const double peak_memory_mb = static_cast<double>(usage.ru_maxrss) / 1024.0;
        std::cout << "{\"metrics\":{";
        std::cout << "\"plaintext_ms\":" << plaintext_ms;
        std::cout << ",\"encrypted_ms\":" << encrypted_ms;
        std::cout << ",\"peak_memory_mb\":" << peak_memory_mb << "},\"layers\":[";
        for (std::size_t index = 0; index < traces.size(); ++index) {
            if (index) std::cout << ',';
            const auto &trace = traces[index];
            std::cout << "{\"plaintext_pre\":"; write_array(trace.plaintext_pre);
            std::cout << ",\"ckks_pre\":"; write_array(trace.ckks_pre);
            std::cout << ",\"plaintext_post\":"; write_array(trace.plaintext_post);
            std::cout << ",\"ckks_post\":"; write_array(trace.ckks_post); std::cout << '}';
        }
        std::cout << "]}\n";
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
