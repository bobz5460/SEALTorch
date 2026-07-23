#include <SEALTorch/sealtorch.h>

#include <X11/Xlib.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <cstdio>

// The application deliberately keeps its JSON reader local.  The JSON files
// are model artifacts, while SEALTorch remains an independent library.
namespace json
{
struct Value
{
    enum class Kind { null_value, number, string, array, object } kind = Kind::null_value;
    double number = 0.0;
    std::string string;
    std::vector<Value> array;
    std::map<std::string, Value> object;

    const Value &at(const std::string &key) const { return object.at(key); }
};

class Parser
{
public:
    explicit Parser(std::string text) : text_(std::move(text)) {}
    Value parse() { skip(); return value(); }

private:
    std::string text_;
    std::size_t position_ = 0;

    void skip() { while (position_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[position_]))) ++position_; }
    void expect(char expected)
    {
        skip();
        if (position_ >= text_.size() || text_[position_++] != expected) throw std::runtime_error("invalid JSON");
    }
    std::string string_value()
    {
        expect('"');
        std::string result;
        while (position_ < text_.size() && text_[position_] != '"') {
            if (text_[position_] == '\\' && position_ + 1 < text_.size()) ++position_;
            result += text_[position_++];
        }
        expect('"');
        return result;
    }
    Value value()
    {
        skip();
        if (position_ >= text_.size()) throw std::runtime_error("unexpected end of JSON");
        if (text_[position_] == '{') return object_value();
        if (text_[position_] == '[') return array_value();
        if (text_[position_] == '"') { Value result; result.kind = Value::Kind::string; result.string = string_value(); return result; }
        const std::size_t begin = position_;
        while (position_ < text_.size() && std::string(",]} \t\r\n").find(text_[position_]) == std::string::npos) ++position_;
        const std::string token = text_.substr(begin, position_ - begin);
        if (token == "null" || token == "true" || token == "false") return {};
        Value result; result.kind = Value::Kind::number; result.number = std::stod(token); return result;
    }
    Value array_value()
    {
        expect('['); Value result; result.kind = Value::Kind::array; skip();
        if (position_ < text_.size() && text_[position_] == ']') { ++position_; return result; }
        for (;;) { result.array.push_back(value()); skip(); if (position_ < text_.size() && text_[position_] == ']') { ++position_; return result; } expect(','); }
    }
    Value object_value()
    {
        expect('{'); Value result; result.kind = Value::Kind::object; skip();
        if (position_ < text_.size() && text_[position_] == '}') { ++position_; return result; }
        for (;;) { const std::string key = string_value(); expect(':'); result.object.emplace(key, value()); skip(); if (position_ < text_.size() && text_[position_] == '}') { ++position_; return result; } expect(','); }
    }
};
}

static sealtorch::Sequential load_model(const std::string &path)
{
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot open model: " + path);
    std::stringstream contents; contents << file.rdbuf();
    const auto root = json::Parser(contents.str()).parse();
    const auto &tensors = root.at("tensors").object;
    sealtorch::Sequential model;
    const std::vector<std::pair<std::string, std::string>> names_list = {
        {"network.1.weight", "network.1.bias"},
        {"network.3.weight", "network.3.bias"},
        {"network.5.weight", "network.5.bias"}};
    for (std::size_t layer_index = 0; layer_index < names_list.size(); ++layer_index) {
        const auto &names = names_list[layer_index];
        const auto &weights = tensors.at(names.first).at("data").array;
        const auto &biases = tensors.at(names.second).at("data").array;
        sealtorch::DenseLayer layer{static_cast<int>(weights.front().array.size()), static_cast<int>(weights.size()), {}, {}};
        layer.weights.resize(weights.size()); layer.biases.resize(biases.size());
        for (std::size_t output = 0; output < weights.size(); ++output) {
            for (const auto &weight : weights[output].array) layer.weights[output].push_back(weight.number);
            layer.biases[output] = biases[output].number;
        }
        model.add(sealtorch::Linear(std::move(layer)));
        if (layer_index + 1 != names_list.size()) model.add(sealtorch::Activation::relu());
    }
    return model;
}

constexpr int kCanvasSize = 280;
constexpr int kOrigin = 30;
constexpr int kDisplaySize = 504;
constexpr int kButtonLeft = 580;

static std::vector<double> preprocess(const std::vector<unsigned char> &canvas)
{
    int min_x = kCanvasSize, min_y = kCanvasSize, max_x = -1, max_y = -1;
    for (int y = 0; y < kCanvasSize; ++y) for (int x = 0; x < kCanvasSize; ++x)
        if (canvas[y * kCanvasSize + x] > 20) { min_x = std::min(min_x, x); max_x = std::max(max_x, x); min_y = std::min(min_y, y); max_y = std::max(max_y, y); }
    std::vector<double> result(784, 0.0);
    if (max_x < 0) return result;
    const int width = max_x - min_x + 1, height = max_y - min_y + 1;
    const double factor = std::min(20.0 / width, 20.0 / height);
    const int target_width = std::max(1, static_cast<int>(std::lround(width * factor)));
    const int target_height = std::max(1, static_cast<int>(std::lround(height * factor)));
    const int offset_x = (28 - target_width) / 2, offset_y = (28 - target_height) / 2;
    for (int y = 0; y < target_height; ++y) for (int x = 0; x < target_width; ++x) {
        const int sx = min_x + std::min(width - 1, static_cast<int>(x * width / static_cast<double>(target_width)));
        const int sy = min_y + std::min(height - 1, static_cast<int>(y * height / static_cast<double>(target_height)));
        result[(offset_y + y) * 28 + offset_x + x] = canvas[sy * kCanvasSize + sx] / 255.0;
    }
    return result;
}

static std::vector<double> softmax(const std::vector<double> &scores)
{
    if (scores.empty()) return {};
    const double maximum = *std::max_element(scores.begin(), scores.end());
    std::vector<double> result(scores.size()); double total = 0.0;
    for (std::size_t i = 0; i < scores.size(); ++i) { result[i] = std::exp(scores[i] - maximum); total += result[i]; }
    for (double &value : result) value /= total;
    return result;
}

// The demo application owns encryption and decryption. The library only sees
// the ciphertexts passed to CiphertextModel.
class WebInference
{
public:
    WebInference(const sealtorch::Sequential &model, std::size_t thread_count, sealtorch::Backend backend)
        : context_(make_context()), keys_(context_), encryptor_(context_, keys_.secret_key()), evaluator_(context_), encoder_(context_), model_(model), thread_count_(thread_count), backend_(backend), scale_(33554432.0), ciphertext_size_(0)
    {
        keys_.create_relin_keys(relin_keys_);
        keys_.create_galois_keys(galois_keys_);
        seal::Plaintext plain;
        encoder_.encode(std::vector<double>(model.input_size(), 0.0), scale_, plain);
        seal::Ciphertext encrypted;
        encryptor_.encrypt_symmetric(plain, encrypted);
        std::ostringstream output;
        encrypted.save(output, seal::compr_mode_type::none);
        ciphertext_size_ = output.str().size();
    }

    std::vector<double> predict(const std::vector<double> &input)
    {
        seal::Plaintext plain;
        encoder_.encode(input, scale_, plain);
        seal::Ciphertext encrypted;
        encryptor_.encrypt_symmetric(plain, encrypted);
        sealtorch::PredictionConfig config(
            context_, evaluator_, relin_keys_, galois_keys_, encoder_,
            scale_, backend_object(), thread_count_);
        const std::vector<seal::Ciphertext> output = model_.predict(
            std::vector<seal::Ciphertext>(1, encrypted), config);
        std::vector<double> values;
        seal::Decryptor decryptor(context_, keys_.secret_key());

        if (backend_ == sealtorch::Backend::Packed)
        {
            seal::Plaintext decoded;
            decryptor.decrypt(output.front(), decoded);
            encoder_.decode(decoded, values);
        }
        else
        {
            for (const seal::Ciphertext &value : output)
            {
                seal::Plaintext decoded_plain;
                std::vector<double> decoded;
                decryptor.decrypt(value, decoded_plain);
                encoder_.decode(decoded_plain, decoded);
                values.push_back(decoded.front());
            }
        }
        values.resize(static_cast<std::size_t>(model_.model().output_size()));
        return values;
    }

    std::vector<double> predict_plain(const std::vector<double> &input) const
    {
        std::vector<double> values = input;
        const std::vector<sealtorch::DenseLayer> &layers = model_.model().layers();
        for (std::size_t layer = 0; layer < layers.size(); ++layer)
        {
            const sealtorch::DenseLayer &current = layers[layer];
            std::vector<double> next(current.output_size, 0.0);
            for (int row = 0; row < current.output_size; ++row)
            {
                next[row] = current.biases[row];
                for (int column = 0; column < current.input_size; ++column)
                    next[row] += current.weights[row][column] * values[column];
                if (model_.model().has_activation(layer))
                {
                    const double x = next[row];
                    next[row] = 0.06762090 + 0.5 * x + 0.48257484 * x * x - 0.06659632 * x * x * x * x;
                }
            }
            values = std::move(next);
        }
        return values;
    }

    void set_backend(sealtorch::Backend backend) { backend_ = backend; }
    std::string backend_name() const { return backend_ == sealtorch::Backend::Packed ? "packed" : "scalar"; }
    std::size_t ciphertext_size() const { return ciphertext_size_; }

private:
    const sealtorch::CiphertextBackend &backend_object() const
    {
        if (backend_ == sealtorch::Backend::Packed) return packed_backend_;
        return scalar_backend_;
    }

    static seal::SEALContext make_context()
    {
        seal::EncryptionParameters parameters(seal::scheme_type::ckks);
        parameters.set_poly_modulus_degree(16384);
        parameters.set_coeff_modulus(seal::CoeffModulus::Create(
            16384, {40, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 40}));
        return seal::SEALContext(parameters);
    }

    seal::SEALContext context_;
    seal::KeyGenerator keys_;
    seal::RelinKeys relin_keys_;
    seal::GaloisKeys galois_keys_;
    seal::Encryptor encryptor_;
    seal::Evaluator evaluator_;
    seal::CKKSEncoder encoder_;
    sealtorch::ScalarBackend scalar_backend_;
    sealtorch::PackedBackend packed_backend_;
    sealtorch::CiphertextModel model_;
    std::size_t thread_count_;
    sealtorch::Backend backend_;
    double scale_;
    std::size_t ciphertext_size_;
};

static double memory_mb() {
    std::ifstream file("/proc/self/status");
    std::string line;
    while (std::getline(file, line)) {
        if (line.find("VmRSS:") == 0) {
            std::istringstream values(line.substr(6));
            double kilobytes = 0.0;
            values >> kilobytes;
            return kilobytes / 1024.0;
        }
    }
    return 0.0;
}

static void print_numbers(const std::vector<double> &values) {
    std::cout << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) std::cout << ',';
        std::cout << std::setprecision(12) << values[i];
    }
    std::cout << ']';
}

static int run_web_worker(const std::string &model_path, std::size_t thread_count, sealtorch::Backend backend) {
    WebInference inference(load_model(model_path), thread_count, backend);
    std::string line;
    while (std::getline(std::cin, line)) {
        try {
            const auto value = json::Parser(line).parse();
            std::vector<double> input;
            const json::Value *pixels = &value;
            if (value.kind == json::Value::Kind::object)
            {
                pixels = &value.at("pixels");
                if (value.object.find("backend") != value.object.end())
                    inference.set_backend(value.at("backend").string == "scalar" ? sealtorch::Backend::Scalar : sealtorch::Backend::Packed);
            }
            for (const auto &item : pixels->array) input.push_back(item.number);
            if (input.size() != 784) throw std::runtime_error("expected 784 pixels");

            const auto memory_before = memory_mb();
            const auto plain_start = std::chrono::steady_clock::now();
            const auto plain = inference.predict_plain(input);
            const auto plain_end = std::chrono::steady_clock::now();
            const auto encrypted_start = std::chrono::steady_clock::now();
            const auto encrypted = inference.predict(input);
            const auto encrypted_end = std::chrono::steady_clock::now();
            const double plain_ms = std::chrono::duration<double, std::milli>(plain_end - plain_start).count();
            const double encrypted_ms = std::chrono::duration<double, std::milli>(encrypted_end - encrypted_start).count();
            const sealtorch::OutputComparison comparison = sealtorch::compare_outputs(plain, encrypted);

            std::cout << "{\"encrypted\":"; print_numbers(encrypted);
            std::cout << ",\"plain\":"; print_numbers(plain);
            std::cout << ",\"encrypted_ms\":" << encrypted_ms;
            std::cout << ",\"plain_ms\":" << plain_ms;
            std::cout << ",\"memory_before_mb\":" << memory_before;
            std::cout << ",\"memory_after_mb\":" << memory_mb();
            std::cout << ",\"ciphertext_bytes\":" << inference.ciphertext_size();
            std::cout << ",\"plaintext_bytes\":" << input.size() * sizeof(double);
            std::cout << ",\"backend\":\"" << inference.backend_name() << "\"";
            std::cout << ",\"max_abs_error\":" << comparison.maximum_absolute_error;
            std::cout << ",\"mean_abs_error\":" << comparison.mean_absolute_error;
            std::cout << ",\"different_values\":" << comparison.different_values << "}\n";
        } catch (const std::exception &error) {
            std::cout << "{\"error\":\"" << error.what() << "\"}\n";
        }
        std::cout.flush();
    }
    return 0;
}

static void draw(Display *display, Window window, GC gc, const std::vector<unsigned char> &canvas, const std::string &status, const std::vector<double> &probabilities, const std::string &model_name)
{
    XSetForeground(display, gc, 0x202124); XFillRectangle(display, window, gc, 0, 0, 900, 600);
    XSetForeground(display, gc, 0x000000); XFillRectangle(display, window, gc, kOrigin, kOrigin, kDisplaySize, kDisplaySize);
    for (int y = 0; y < kCanvasSize; ++y) for (int x = 0; x < kCanvasSize; ++x) if (canvas[y * kCanvasSize + x]) {
        const unsigned char value = canvas[y * kCanvasSize + x]; XSetForeground(display, gc, (value << 16) | (value << 8) | value);
        XFillRectangle(display, window, gc, kOrigin + x * kDisplaySize / kCanvasSize, kOrigin + y * kDisplaySize / kCanvasSize, 2, 2);
    }
    XSetForeground(display, gc, 0x4b8bff);
    XFillRectangle(display, window, gc, kButtonLeft, 60, 250, 40); XFillRectangle(display, window, gc, kButtonLeft, 115, 250, 40); XFillRectangle(display, window, gc, kButtonLeft, 170, 250, 40);
    XSetForeground(display, gc, 0xffffff);
    auto label = [&](const char *text, int x, int y) { XDrawString(display, window, gc, x, y, const_cast<char *>(text), static_cast<int>(std::strlen(text))); };
    label("Predict (encrypted)", kButtonLeft + 60, 86); label("Clear", kButtonLeft + 105, 141); label("Switch JSON model", kButtonLeft + 55, 196);
    std::string model_line = "Model: " + model_name; label(model_line.c_str(), kButtonLeft, 240);
    label(status.c_str(), kButtonLeft, 270); label("Probabilities", kButtonLeft, 315);
    for (std::size_t digit = 0; digit < probabilities.size() && digit < 10; ++digit) {
        const int y = 340 + static_cast<int>(digit) * 20; const int bar = static_cast<int>(std::clamp(probabilities[digit], 0.0, 1.0) * 180);
        XSetForeground(display, gc, 0x555555); XFillRectangle(display, window, gc, kButtonLeft, y - 12, 180, 12);
        XSetForeground(display, gc, digit == static_cast<std::size_t>(std::distance(probabilities.begin(), std::max_element(probabilities.begin(), probabilities.end()))) ? 0x35d07f : 0x4b8bff);
        XFillRectangle(display, window, gc, kButtonLeft, y - 12, bar, 12); XSetForeground(display, gc, 0xffffff);
        std::ostringstream text; text << digit << ": " << std::fixed << std::setprecision(1) << probabilities[digit] * 100.0 << "%";
        label(text.str().c_str(), kButtonLeft + 190, y);
    }
    XFlush(display);
}

int main(int argc, char **argv)
{
    try {
        if (argc > 1 && std::string(argv[1]) == "--web-worker") {
            const std::string model_path = argc > 2 ? argv[2] : "src/mnist_mlp.json";
            const std::size_t thread_count = argc > 3 ? std::stoul(argv[3]) : 4;
            if (thread_count == 0) throw std::runtime_error("thread count must be greater than zero");
            const std::string backend = argc > 4 ? argv[4] : "packed";
            return run_web_worker(model_path, thread_count, backend == "scalar" ? sealtorch::Backend::Scalar : sealtorch::Backend::Packed);
        }
        const std::string first_path = argc > 1 ? argv[1] : "src/mnist_mlp.json";
        const std::size_t thread_count = argc > 2 ? std::stoul(argv[2]) : 4;
        if (thread_count == 0) throw std::runtime_error("thread count must be greater than zero");
        const std::string second_path = first_path.find("gelu") == std::string::npos ? "src/mnist_mlp_gelu.json" : "src/mnist_mlp.json";
        sealtorch::Sequential first_model = load_model(first_path);
        std::unique_ptr<WebInference> inference(new WebInference(first_model, thread_count, sealtorch::Backend::Packed));
        Display *display = XOpenDisplay(nullptr); if (!display) throw std::runtime_error("could not open X11 display");
        const int screen = DefaultScreen(display); Window window = XCreateSimpleWindow(display, RootWindow(display, screen), 100, 100, 900, 600, 1, BlackPixel(display, screen), 0x202124);
        XStoreName(display, window, "SEALTorch - MNIST JSON demo"); XSelectInput(display, window, ExposureMask | ButtonPressMask | ButtonReleaseMask | PointerMotionMask); XMapWindow(display, window);
        GC gc = XCreateGC(display, window, 0, nullptr); std::vector<unsigned char> canvas(kCanvasSize * kCanvasSize); std::vector<double> probabilities; bool drawing = false, running = false, first_active = true;
        std::string status = "Draw a digit, then click Predict"; std::future<std::vector<double>> task;
        auto model_name = [&]() { return first_active ? first_path : second_path; };
        auto paint = [&](int px, int py) { const int x = (px - kOrigin) * kCanvasSize / kDisplaySize, y = (py - kOrigin) * kCanvasSize / kDisplaySize; if (x < 0 || x >= kCanvasSize || y < 0 || y >= kCanvasSize) return; for (int dy = -8; dy <= 8; ++dy) for (int dx = -8; dx <= 8; ++dx) if (dx * dx + dy * dy <= 64 && x + dx >= 0 && x + dx < kCanvasSize && y + dy >= 0 && y + dy < kCanvasSize) canvas[(y + dy) * kCanvasSize + x + dx] = 255; draw(display, window, gc, canvas, status, probabilities, model_name()); };
        for (;;) {
            if (running && task.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) { try { probabilities = softmax(task.get()); status = "Prediction: " + std::to_string(std::distance(probabilities.begin(), std::max_element(probabilities.begin(), probabilities.end()))); } catch (const std::exception &error) { status = std::string("Inference failed: ") + error.what(); } running = false; draw(display, window, gc, canvas, status, probabilities, model_name()); }
            if (XPending(display) == 0) { std::this_thread::sleep_for(std::chrono::milliseconds(25)); continue; }
            XEvent event; XNextEvent(display, &event);
            if (event.type == Expose) draw(display, window, gc, canvas, status, probabilities, model_name());
            else if (event.type == ButtonPress) {
                if (event.xbutton.x >= kButtonLeft && event.xbutton.x < kButtonLeft + 250 && event.xbutton.y >= 115 && event.xbutton.y < 155) { std::fill(canvas.begin(), canvas.end(), 0); probabilities.clear(); status = "Canvas cleared"; draw(display, window, gc, canvas, status, probabilities, model_name()); }
                else if (!running && event.xbutton.x >= kButtonLeft && event.xbutton.x < kButtonLeft + 250 && event.xbutton.y >= 170 && event.xbutton.y < 210) { first_active = !first_active; status = "Loading selected JSON model..."; draw(display, window, gc, canvas, status, probabilities, model_name()); inference.reset(new WebInference(load_model(model_name()), thread_count, sealtorch::Backend::Packed)); probabilities.clear(); status = "Model loaded; draw a digit"; draw(display, window, gc, canvas, status, probabilities, model_name()); }
                else if (!running && event.xbutton.x >= kButtonLeft && event.xbutton.x < kButtonLeft + 250 && event.xbutton.y >= 60 && event.xbutton.y < 100) { running = true; probabilities.clear(); status = "Encrypting and evaluating..."; const auto input = preprocess(canvas); task = std::async(std::launch::async, [&inference, input] { return inference->predict(input); }); draw(display, window, gc, canvas, status, probabilities, model_name()); }
                else { drawing = true; paint(event.xbutton.x, event.xbutton.y); }
            } else if (event.type == ButtonRelease) drawing = false; else if (event.type == MotionNotify && drawing) paint(event.xmotion.x, event.xmotion.y);
        }
    } catch (const std::exception &error) { std::cerr << "SEALTorch GUI error: " << error.what() << '\n'; return 1; }
}
