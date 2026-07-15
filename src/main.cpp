#include <SEALTorch/evaluator.h>

#include <X11/Xlib.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <fstream>
#include <future>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
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
        skip();
        if (position_ >= text_.size()) throw std::runtime_error("unexpected end of JSON");
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

constexpr int kCanvasSize = 280;
constexpr int kCanvasOrigin = 30;
constexpr int kCanvasDisplaySize = 560;
constexpr int kCanvasPixelDisplaySize = 2;

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

static std::vector<double> preprocess(const std::vector<unsigned char> &canvas)
{
    int min_x = kCanvasSize, min_y = kCanvasSize, max_x = -1, max_y = -1;
    for (int y = 0; y < kCanvasSize; ++y) for (int x = 0; x < kCanvasSize; ++x) if (canvas[y * kCanvasSize + x] > 20) {
        min_x = std::min(min_x, x); max_x = std::max(max_x, x); min_y = std::min(min_y, y); max_y = std::max(max_y, y);
    }
    std::vector<double> result(784, 0.0);
    if (max_x < 0) return result;
    const int width = max_x - min_x + 1, height = max_y - min_y + 1;
    const double resize = std::min(20.0 / width, 20.0 / height);
    const int target_width = std::max(1, static_cast<int>(std::lround(width * resize)));
    const int target_height = std::max(1, static_cast<int>(std::lround(height * resize)));
    const int offset_x = (28 - target_width) / 2, offset_y = (28 - target_height) / 2;
    for (int y = 0; y < target_height; ++y) for (int x = 0; x < target_width; ++x) {
        // Area-average the high-resolution crop into the MNIST pixel. This
        // preserves antialiasing instead of selecting one source pixel.
        const int source_x0 = min_x + static_cast<int>(x * width / static_cast<double>(target_width));
        const int source_x1 = min_x + std::max(source_x0 - min_x + 1, static_cast<int>((x + 1) * width / static_cast<double>(target_width)));
        const int source_y0 = min_y + static_cast<int>(y * height / static_cast<double>(target_height));
        const int source_y1 = min_y + std::max(source_y0 - min_y + 1, static_cast<int>((y + 1) * height / static_cast<double>(target_height)));
        double sum = 0.0; int count = 0;
        for (int source_y = source_y0; source_y < std::min(max_y + 1, source_y1); ++source_y)
            for (int source_x = source_x0; source_x < std::min(max_x + 1, source_x1); ++source_x) {
                sum += canvas[source_y * kCanvasSize + source_x]; ++count;
            }
        result[(offset_y + y) * 28 + offset_x + x] = count == 0 ? 0.0 : (sum / count) / 255.0;
    }
    return result;
}

static std::vector<double> plaintext_predict(const NeuralNetwork &model, const std::vector<double> &input)
{
    std::vector<double> values = input;
    for (std::size_t layer_index = 0; layer_index < model.layers.size(); ++layer_index) {
        const auto &layer = model.layers[layer_index];
        std::vector<double> next(layer.output_size, 0.0);
        for (int output = 0; output < layer.output_size; ++output) {
            next[output] = layer.biases[output];
            for (int input_index = 0; input_index < layer.input_size; ++input_index)
                next[output] += layer.weights[output][input_index] * values[input_index];
        }
        if (layer_index + 1 != model.layers.size())
            for (double &value : next) value = std::max(0.0, value);
        values = std::move(next);
    }
    return values;
}

static std::vector<double> softmax(const std::vector<double> &scores)
{
    if (scores.empty()) return {};
    const double maximum = *std::max_element(scores.begin(), scores.end());
    std::vector<double> probabilities(scores.size());
    double total = 0.0;
    for (std::size_t index = 0; index < scores.size(); ++index) {
        probabilities[index] = std::exp(scores[index] - maximum);
        total += probabilities[index];
    }
    for (double &probability : probabilities) probability /= total;
    return probabilities;
}

class EncryptedInference {
public:
    explicit EncryptedInference(const NeuralNetwork &model)
        : context_(make_context()), key_generator_(context_), encryptor_(context_, key_generator_.secret_key()),
          evaluator_(context_), encoder_(context_), model_evaluator_(model)
    {
        key_generator_.create_galois_keys(galois_keys_);
        key_generator_.create_relin_keys(relin_keys_);
    }

    std::vector<double> predict(const std::vector<double> &input,
                                const std::function<void(int, const std::string &)> &progress) {
        if (progress) progress(0, "Encoding 784 input pixels...");
        seal::Plaintext plain;
        encoder_.encode(input, scale_, plain);
        seal::Ciphertext encrypted;
        encryptor_.encrypt_symmetric(plain, encrypted);
        if (progress) progress(5, "Input encrypted; evaluating 3 encrypted layers...");
        const auto outputs = model_evaluator_.predict(
            {encrypted}, evaluator_, relin_keys_, galois_keys_, encoder_, scale_,
            [&](const sealtorch::ProgressInfo &info) {
                if (!progress) return;
                if (info.total == 0) { progress(5, info.phase); return; }
                const int percent = 5 + static_cast<int>(90 * info.completed / info.total);
                std::string detail = info.phase + " | network " + std::to_string(info.completed) + "/" + std::to_string(info.total);
                if (info.layer_total > 0) {
                    detail += " | layer " + std::to_string(info.layer) + "/" + std::to_string(info.total_layers);
                    if (info.layer_completed > 0) detail += " | neuron " + std::to_string(info.layer_completed) + "/" + std::to_string(info.layer_total);
                }
                progress(percent, detail);
            });
        seal::Decryptor decryptor(context_, key_generator_.secret_key());
        std::vector<double> result;
        for (std::size_t index = 0; index < outputs.size(); ++index) {
            if (progress) progress(95 + static_cast<int>(5 * index / outputs.size()), "Decrypting output " + std::to_string(index + 1) + "/" + std::to_string(outputs.size()) + "...");
            const auto &output = outputs[index];
            seal::Plaintext decoded_plain; decryptor.decrypt(output, decoded_plain);
            std::vector<double> decoded; encoder_.decode(decoded_plain, decoded);
            if (decoded.empty()) throw std::runtime_error("empty decrypted output");
            result.push_back(decoded.front());
        }
        if (progress) progress(100, "Decryption complete; preparing prediction");
        return result;
    }
private:
    static seal::SEALContext make_context() {
        seal::EncryptionParameters parameters(seal::scheme_type::ckks);
        parameters.set_poly_modulus_degree(16384);
        parameters.set_coeff_modulus(seal::CoeffModulus::Create(
            16384, {60, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 60}));
        return seal::SEALContext(parameters);
    }
    seal::SEALContext context_; seal::KeyGenerator key_generator_; seal::RelinKeys relin_keys_; seal::GaloisKeys galois_keys_;
    seal::Encryptor encryptor_; seal::Evaluator evaluator_; seal::CKKSEncoder encoder_; sealtorch::Evaluator model_evaluator_;
    const double scale_ = std::pow(2.0, 30);
};

static void draw(Display *display, Window window, GC gc, const std::vector<unsigned char> &canvas,
                 const std::string &status, int progress_percent,
                 const std::vector<double> &probabilities)
{
    XClearWindow(display, window);
    XSetForeground(display, gc, 0x202124); XFillRectangle(display, window, gc, 0, 0, 980, 560);
    XSetForeground(display, gc, 0x000000); XFillRectangle(display, window, gc, kCanvasOrigin, kCanvasOrigin, kCanvasDisplaySize, kCanvasDisplaySize);
    for (int y = 0; y < kCanvasSize; ++y) for (int x = 0; x < kCanvasSize; ++x) if (canvas[y * kCanvasSize + x]) {
        const unsigned value = canvas[y * kCanvasSize + x]; XSetForeground(display, gc, (value << 16) | (value << 8) | value);
        XFillRectangle(display, window, gc, kCanvasOrigin + x * kCanvasPixelDisplaySize, kCanvasOrigin + y * kCanvasPixelDisplaySize, kCanvasPixelDisplaySize, kCanvasPixelDisplaySize);
    }
    XSetForeground(display, gc, 0x4b8bff); XFillRectangle(display, window, gc, 640, 80, 180, 42); XFillRectangle(display, window, gc, 640, 140, 180, 42); XFillRectangle(display, window, gc, 640, 200, 180, 42);
    XSetForeground(display, gc, 0xffffff); const std::string predict = "Predict", clear = "Clear";
    XDrawString(display, window, gc, 695, 107, const_cast<char *>(predict.c_str()), predict.size());
    XDrawString(display, window, gc, 700, 167, const_cast<char *>(clear.c_str()), clear.size());
    const std::string mode = "Toggle mode";
    XDrawString(display, window, gc, 685, 227, const_cast<char *>(mode.c_str()), mode.size());
    XSetForeground(display, gc, 0x202124); XFillRectangle(display, window, gc, 620, 260, 400, 65);
    XSetForeground(display, gc, 0xffffff);
    XRectangle clip{620, 260, 400, 65}; XSetClipRectangles(display, gc, 0, 0, &clip, 1, Unsorted);
    XDrawString(display, window, gc, 630, 295, const_cast<char *>(status.c_str()), static_cast<int>(status.size()));
    XSetClipMask(display, gc, None);
    XSetForeground(display, gc, 0x555555); XFillRectangle(display, window, gc, 630, 345, 370, 22);
    XSetForeground(display, gc, 0x35d07f); XFillRectangle(display, window, gc, 630, 345, 370 * std::clamp(progress_percent, 0, 100) / 100, 22);
    XSetForeground(display, gc, 0xffffff); XDrawRectangle(display, window, gc, 630, 345, 370, 22);
    const std::string percent = std::to_string(std::clamp(progress_percent, 0, 100)) + "%";
    XDrawString(display, window, gc, 790, 363, const_cast<char *>(percent.c_str()), static_cast<int>(percent.size())); XFlush(display);

    XSetForeground(display, gc, 0xffffff);
    XDrawString(display, window, gc, 630, 395, const_cast<char *>("Probabilities"), 12);
    for (std::size_t digit = 0; digit < probabilities.size() && digit < 10; ++digit) {
        const int y = 418 + static_cast<int>(digit) * 14;
        const int bar_width = static_cast<int>(std::clamp(probabilities[digit], 0.0, 1.0) * 210.0);
        XSetForeground(display, gc, 0x555555);
        XFillRectangle(display, window, gc, 655, y - 10, 210, 10);
        XSetForeground(display, gc, digit == static_cast<std::size_t>(
            std::distance(probabilities.begin(), std::max_element(probabilities.begin(), probabilities.end())))
            ? 0x35d07f : 0x4b8bff);
        XFillRectangle(display, window, gc, 655, y - 10, bar_width, 10);
        XSetForeground(display, gc, 0xffffff);
        std::ostringstream label;
        label << digit << ": " << std::fixed << std::setprecision(2) << probabilities[digit] * 100.0 << "%";
        const std::string text = label.str();
        XDrawString(display, window, gc, 875, y, const_cast<char *>(text.c_str()), static_cast<int>(text.size()));
    }
    XFlush(display);
}

int main(int argc, char **argv)
{
    try {
        const std::string model_path = argc > 1 ? argv[1] : "src/mnist_mlp.json";
        sealtorch::Evaluator model_evaluator(load_model(model_path));
        EncryptedInference inference(model_evaluator.model());
        Display *display = XOpenDisplay(nullptr); if (!display) throw std::runtime_error("could not open X11 display");
        const int screen = DefaultScreen(display);
        Window window = XCreateSimpleWindow(display, RootWindow(display, screen), 100, 100, 980, 560, 1, BlackPixel(display, screen), 0x202124);
        XStoreName(display, window, "SEALTorch encrypted MNIST inference");
        XSelectInput(display, window, ExposureMask | ButtonPressMask | ButtonReleaseMask | PointerMotionMask); XMapWindow(display, window);
        GC gc = XCreateGC(display, window, 0, nullptr); std::vector<unsigned char> canvas(kCanvasSize * kCanvasSize);
        std::vector<double> probabilities;
        std::string status = "Draw a digit, then click Predict"; bool drawing = false; bool inference_running = false; bool encrypted_mode = true; bool running_encrypted = true;
        std::atomic<int> progress_percent{0}; std::mutex progress_mutex; std::string progress_text = status;
        std::future<std::vector<double>> inference_task;
        auto paint = [&](int px, int py) {
            const int x = (px - kCanvasOrigin) / kCanvasPixelDisplaySize, y = (py - kCanvasOrigin) / kCanvasPixelDisplaySize;
            if (x < 0 || x >= kCanvasSize || y < 0 || y >= kCanvasSize) return;
            constexpr int brush_radius = 10;
            for (int dy = -brush_radius; dy <= brush_radius; ++dy) for (int dx = -brush_radius; dx <= brush_radius; ++dx)
                if (dx * dx + dy * dy <= brush_radius * brush_radius && x + dx >= 0 && x + dx < kCanvasSize && y + dy >= 0 && y + dy < kCanvasSize)
                    canvas[(y + dy) * kCanvasSize + x + dx] = 255;
            draw(display, window, gc, canvas, status, progress_percent.load(), probabilities);
        };
        for (;;) {
            if (inference_running && inference_task.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                try {
                    const auto scores = inference_task.get();
                    probabilities = softmax(scores);
                    const int digit = static_cast<int>(std::max_element(probabilities.begin(), probabilities.end()) - probabilities.begin());
                    status = "Prediction: " + std::to_string(digit) + (running_encrypted ? " (encrypted inference complete)" : " (plaintext inference complete)");
                } catch (const std::exception &error) {
                    status = std::string("Inference failed: ") + error.what();
                }
                inference_running = false; progress_percent.store(100);
                draw(display, window, gc, canvas, status, 100, probabilities);
            }
            if (inference_running) {
                std::lock_guard<std::mutex> lock(progress_mutex);
                draw(display, window, gc, canvas, progress_text, progress_percent.load(), probabilities);
            }
            if (XPending(display) == 0) { std::this_thread::sleep_for(std::chrono::milliseconds(30)); continue; }
            XEvent event; XNextEvent(display, &event);
            if (event.type == Expose) draw(display, window, gc, canvas, status, progress_percent.load(), probabilities);
            else if (event.type == ButtonPress) {
                if (event.xbutton.x >= 640 && event.xbutton.x <= 820 && event.xbutton.y >= 140 && event.xbutton.y <= 182) { std::fill(canvas.begin(), canvas.end(), 0); probabilities.clear(); status = "Canvas cleared"; progress_percent.store(0); draw(display, window, gc, canvas, status, 0, probabilities); }
                else if (!inference_running && event.xbutton.x >= 640 && event.xbutton.x <= 820 && event.xbutton.y >= 200 && event.xbutton.y <= 242) {
                    encrypted_mode = !encrypted_mode;
                    probabilities.clear();
                    status = encrypted_mode ? "Mode: homomorphically encrypted CKKS" : "Mode: regular plaintext inference";
                    draw(display, window, gc, canvas, status, 0, probabilities);
                }
                else if (!inference_running && event.xbutton.x >= 640 && event.xbutton.x <= 820 && event.xbutton.y >= 80 && event.xbutton.y <= 122) {
                    const auto input = preprocess(canvas); inference_running = true; running_encrypted = encrypted_mode; progress_percent.store(0);
                    { std::lock_guard<std::mutex> lock(progress_mutex); progress_text = running_encrypted ? "Starting encrypted inference..." : "Starting plaintext inference..."; }
                    const bool use_encryption = running_encrypted;
                    inference_task = std::async(std::launch::async, [&inference, &model_evaluator, input, use_encryption, &progress_percent, &progress_mutex, &progress_text]() {
                        if (!use_encryption) {
                            progress_percent.store(20); { std::lock_guard<std::mutex> lock(progress_mutex); progress_text = "Running regular dense-layer inference..."; }
                            auto result = plaintext_predict(model_evaluator.model(), input);
                            progress_percent.store(100); { std::lock_guard<std::mutex> lock(progress_mutex); progress_text = "Plaintext inference complete"; }
                            return result;
                        }
                        return inference.predict(input, [&](int percent, const std::string &detail) {
                            progress_percent.store(percent); std::lock_guard<std::mutex> lock(progress_mutex); progress_text = detail;
                        });
                    });
                    draw(display, window, gc, canvas, running_encrypted ? "Starting encrypted inference..." : "Starting plaintext inference...", 0, probabilities);
                }
                else { drawing = true; paint(event.xbutton.x, event.xbutton.y); }
            } else if (event.type == ButtonRelease) drawing = false;
            else if (event.type == MotionNotify && drawing) paint(event.xmotion.x, event.xmotion.y);
        }
    } catch (const std::exception &error) { std::cerr << "SEALTorch GUI error: " << error.what() << '\n'; return 1; }
}
