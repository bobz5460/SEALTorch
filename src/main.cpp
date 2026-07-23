#include <SEALTorch/evaluator.h>

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

static NeuralNetwork load_model(const std::string &path)
{
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot open model: " + path);
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

class EncryptedInference
{
public:
    EncryptedInference(const NeuralNetwork &model, std::size_t thread_count)
        : context_(make_context()), keys_(context_), encryptor_(context_, keys_.secret_key()), evaluator_(context_), encoder_(context_), model_(model), thread_count_(thread_count)
    {
        keys_.create_relin_keys(relin_keys_); keys_.create_galois_keys(galois_keys_);
    }

    std::vector<double> predict(const std::vector<double> &input) {
        seal::Plaintext plain; encoder_.encode(input, scale_, plain);
        seal::Ciphertext encrypted; encryptor_.encrypt_symmetric(plain, encrypted);
        const auto output = model_.predict(context_, encrypted, evaluator_, relin_keys_, galois_keys_, encoder_, scale_, thread_count_);
        seal::Decryptor decryptor(context_, keys_.secret_key()); seal::Plaintext decoded_plain;
        decryptor.decrypt(output, decoded_plain); std::vector<double> decoded; encoder_.decode(decoded_plain, decoded);
        if (decoded.size() < 10) throw std::runtime_error("SEALTorch returned too few output values");
        return {decoded.begin(), decoded.begin() + 10};
    }

private:
    static seal::SEALContext make_context() {
        seal::EncryptionParameters parameters(seal::scheme_type::ckks);
        parameters.set_poly_modulus_degree(16384);
        parameters.set_coeff_modulus(seal::CoeffModulus::Create(16384, {60, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 60}));
        return seal::SEALContext(parameters);
    }
    seal::SEALContext context_; seal::KeyGenerator keys_; seal::RelinKeys relin_keys_; seal::GaloisKeys galois_keys_;
    seal::Encryptor encryptor_; seal::Evaluator evaluator_; seal::CKKSEncoder encoder_; sealtorch::Evaluator model_; std::size_t thread_count_; const double scale_ = std::pow(2.0, 30);
};

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
        const std::string first_path = argc > 1 ? argv[1] : "src/mnist_mlp.json";
        const std::size_t thread_count = argc > 2 ? std::stoul(argv[2]) : 4;
        if (thread_count == 0) throw std::runtime_error("thread count must be greater than zero");
        const std::string second_path = first_path.find("gelu") == std::string::npos ? "src/mnist_mlp_gelu.json" : "src/mnist_mlp.json";
        NeuralNetwork first_model = load_model(first_path);
        std::unique_ptr<EncryptedInference> inference(new EncryptedInference(first_model, thread_count));
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
                else if (!running && event.xbutton.x >= kButtonLeft && event.xbutton.x < kButtonLeft + 250 && event.xbutton.y >= 170 && event.xbutton.y < 210) { first_active = !first_active; status = "Loading selected JSON model..."; draw(display, window, gc, canvas, status, probabilities, model_name()); inference.reset(new EncryptedInference(load_model(model_name()), thread_count)); probabilities.clear(); status = "Model loaded; draw a digit"; draw(display, window, gc, canvas, status, probabilities, model_name()); }
                else if (!running && event.xbutton.x >= kButtonLeft && event.xbutton.x < kButtonLeft + 250 && event.xbutton.y >= 60 && event.xbutton.y < 100) { running = true; probabilities.clear(); status = "Encrypting and evaluating..."; const auto input = preprocess(canvas); task = std::async(std::launch::async, [&inference, input] { return inference->predict(input); }); draw(display, window, gc, canvas, status, probabilities, model_name()); }
                else { drawing = true; paint(event.xbutton.x, event.xbutton.y); }
            } else if (event.type == ButtonRelease) drawing = false; else if (event.type == MotionNotify && drawing) paint(event.xmotion.x, event.xmotion.y);
        }
    } catch (const std::exception &error) { std::cerr << "SEALTorch GUI error: " << error.what() << '\n'; return 1; }
}
