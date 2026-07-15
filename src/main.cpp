#include <SEALTorch/evaluator.h>

#include <X11/Xlib.h>
#include <X11/Xutil.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <future>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <string>
#include <utility>
#include <vector>

// This file intentionally contains the small JSON reader and the UI.  The
// SEALTorch library is not modified: its NeuralNetwork type is used as the
// model contract and its Evaluator is used to own the loaded model.
namespace json
{
struct Value
{
    enum class Kind { Null, Number, String, Array, Object } kind = Kind::Null;
    double number = 0.0;
    std::string string;
    std::vector<Value> array;
    std::map<std::string, Value> object;

    const Value &at(const std::string &key) const { return object.at(key); }
    double number_at(std::size_t i) const { return array.at(i).number; }
};

class Parser
{
public:
    explicit Parser(std::string text) : text_(std::move(text)) {}
    Value parse() { skip(); return value(); }

private:
    std::string text_; std::size_t p_ = 0;
    void skip() { while (p_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[p_]))) ++p_; }
    void expect(char c) { skip(); if (p_ >= text_.size() || text_[p_++] != c) throw std::runtime_error("invalid JSON"); }
    Value value()
    {
        skip(); if (p_ >= text_.size()) throw std::runtime_error("unexpected end of JSON");
        if (text_[p_] == '{') return object(); if (text_[p_] == '[') return array();
        if (text_[p_] == '"') return {Value::Kind::String, 0.0, string_value(), {}, {}};
        std::size_t end = p_;
        while (end < text_.size() && std::string(",]} \t\r\n").find(text_[end]) == std::string::npos) ++end;
        std::string token = text_.substr(p_, end - p_); p_ = end;
        if (token == "null" || token == "true" || token == "false") return {};
        return {Value::Kind::Number, std::stod(token), {}, {}, {}};
    }
    std::string string_value()
    {
        expect('"'); std::string out;
        while (p_ < text_.size() && text_[p_] != '"') {
            if (text_[p_] == '\\' && p_ + 1 < text_.size()) { ++p_; out += text_[p_++]; }
            else out += text_[p_++];
        }
        expect('"'); return out;
    }
    Value array()
    {
        expect('['); Value out; out.kind = Value::Kind::Array; skip();
        if (p_ < text_.size() && text_[p_] == ']') { ++p_; return out; }
        for (;;) { out.array.push_back(value()); skip(); if (p_ < text_.size() && text_[p_] == ']') { ++p_; return out; } expect(','); }
    }
    Value object()
    {
        expect('{'); Value out; out.kind = Value::Kind::Object; skip();
        if (p_ < text_.size() && text_[p_] == '}') { ++p_; return out; }
        for (;;) { std::string key = string_value(); expect(':'); out.object.emplace(std::move(key), value()); skip(); if (p_ < text_.size() && text_[p_] == '}') { ++p_; return out; } expect(','); }
    }
};
}

static NeuralNetwork load_model(const std::string &path)
{
    std::ifstream file(path); if (!file) throw std::runtime_error("cannot open model: " + path);
    std::stringstream buffer; buffer << file.rdbuf();
    const auto root = json::Parser(buffer.str()).parse();
    NeuralNetwork model; model.input_size = 784; model.output_size = 10;
    const auto &tensors = root.at("tensors").object;
    const std::vector<std::pair<std::string, std::string>> names = {
        {"network.1.weight", "network.1.bias"}, {"network.3.weight", "network.3.bias"}, {"network.5.weight", "network.5.bias"}};
    for (const auto &name : names) {
        const auto &w = tensors.at(name.first).at("data").array;
        const auto &b = tensors.at(name.second).at("data").array;
        DenseLayer layer{static_cast<int>(w.front().array.size()), static_cast<int>(w.size()), {}, {}};
        layer.weights.resize(w.size()); layer.biases.resize(b.size());
        for (std::size_t o = 0; o < w.size(); ++o) { for (const auto &n : w[o].array) layer.weights[o].push_back(n.number); layer.biases[o] = b[o].number; }
        model.layers.push_back(std::move(layer));
    }
    return model;
}

class EncryptedInference
{
public:
    explicit EncryptedInference(const NeuralNetwork &model)
        : context_(make_context()),
          key_generator_(context_),
          encryptor_(context_, key_generator_.secret_key()),
          evaluator_(context_),
          encoder_(context_),
          model_evaluator_(model)
    {
        key_generator_.create_relin_keys(relin_keys_);
    }

    std::vector<double> predict(const std::vector<double> &input, const std::function<void(int, const std::string &)> &progress)
    {
        const std::size_t network_work = [&]() {
            std::size_t work = 0;
            for (const auto &layer : model_evaluator_.model().layers) work += static_cast<std::size_t>(layer.output_size);
            return work;
        }();
        const std::size_t total_work = input.size() + network_work + model_evaluator_.model().output_size;
        std::atomic<std::size_t> completed{0};
        auto report = [&](std::size_t amount) {
            const auto value = completed.fetch_add(amount) + amount;
            if (progress) progress(static_cast<int>(5 * value / std::max<std::size_t>(1, input.size())), "Encrypting input: " + std::to_string(value) + "/" + std::to_string(input.size()) + " pixels");
        };
        if (progress) progress(0, "Encrypting input...");

        // Encryption is independent for every input pixel. Use one task per
        // worker and a private encoder/encryptor in each task.
        std::vector<seal::Ciphertext> encrypted_input(input.size());
        const unsigned workers = std::max(1u, std::thread::hardware_concurrency());
        const std::size_t chunk = (input.size() + workers - 1) / workers;
        std::vector<std::future<void>> encryption_tasks;
        for (std::size_t begin = 0; begin < input.size(); begin += chunk) {
            const std::size_t end = std::min(input.size(), begin + chunk);
            encryption_tasks.push_back(std::async(std::launch::async, [&, begin, end]() {
                seal::CKKSEncoder encoder(context_);
                seal::Encryptor encryptor(context_, key_generator_.secret_key());
                for (std::size_t i = begin; i < end; ++i) {
                    seal::Plaintext plain;
                    encoder.encode(input[i], scale_, plain);
                    encryptor.encrypt_symmetric(plain, encrypted_input[i]);
                    report(1);
                }
            }));
        }
        for (auto &task : encryption_tasks) task.get();

        const auto encrypted_output = model_evaluator_.predict(
            encrypted_input, evaluator_, relin_keys_, encoder_, scale_,
            [&](const sealtorch::ProgressInfo &info) {
                completed.store(input.size() + info.completed);
                if (info.layer_total == 0) {
                    if (progress) progress(5, "Starting encrypted network evaluation...");
                    return;
                }
                const int percent = 5 + static_cast<int>(90 * info.completed / std::max<std::size_t>(1, info.total));
                const std::size_t left_in_layer = info.layer_total - info.layer_completed;
                const std::string text = "Encrypted layer " + std::to_string(info.layer + 1) + ": neuron " +
                    std::to_string(info.layer_completed) + "/" + std::to_string(info.layer_total) +
                    " completed (" + std::to_string(left_in_layer) + " left); network " +
                    std::to_string(info.completed) + "/" + std::to_string(info.total) +
                    " (" + std::to_string(info.total - info.completed) + " left)";
                if (progress) progress(percent, text);
            });

        std::vector<double> output(encrypted_output.size());
        std::vector<std::future<void>> decryption_tasks;
        std::atomic<std::size_t> decrypted{0};
        for (std::size_t i = 0; i < encrypted_output.size(); ++i) {
            decryption_tasks.push_back(std::async(std::launch::async, [&, i]() {
                seal::Decryptor decryptor(context_, key_generator_.secret_key());
                seal::CKKSEncoder encoder(context_);
                seal::Plaintext plain;
                decryptor.decrypt(encrypted_output[i], plain);
                std::vector<double> decoded;
                encoder.decode(plain, decoded);
                if (decoded.empty()) throw std::runtime_error("SEALTorch returned an empty prediction");
                output[i] = decoded.front();
                const auto done = decrypted.fetch_add(1) + 1;
                if (progress) progress(95 + static_cast<int>(5 * done / std::max<std::size_t>(1, encrypted_output.size())),
                    "Decrypting output: " + std::to_string(done) + "/" + std::to_string(encrypted_output.size()) + " scores");
            }));
        }
        for (auto &task : decryption_tasks) task.get();
        return output;
    }

private:
    static seal::SEALContext make_context()
    {
        seal::EncryptionParameters parameters(seal::scheme_type::ckks);
        // Keep the chain within SEAL's security limits. The previous
        // configuration used 600 total modulus bits at degree 8192, while
        // that degree permits only about 218 bits and SEAL rejected the
        // context with "encryption parameters are not set correctly".
        // Keep each prime comfortably above the 2^30 CKKS scale. This also
        // prevents repeated rescaling from rounding away the encrypted part
        // of the ciphertext and producing a transparent result.
        parameters.set_poly_modulus_degree(16384);
        parameters.set_coeff_modulus(seal::CoeffModulus::Create(
            16384, {40, 40, 40, 40, 40, 40, 40, 40, 40, 40}));
        return seal::SEALContext(parameters);
    }

    seal::SEALContext context_;
    seal::KeyGenerator key_generator_;
    seal::RelinKeys relin_keys_;
    seal::Encryptor encryptor_;
    seal::Evaluator evaluator_;
    seal::CKKSEncoder encoder_;
    sealtorch::Evaluator model_evaluator_;
    const double scale_ = std::pow(2.0, 30);
};

static std::vector<double> preprocess(const std::vector<unsigned char> &canvas)
{
    int min_x = 28, min_y = 28, max_x = -1, max_y = -1;
    for (int y = 0; y < 28; ++y) for (int x = 0; x < 28; ++x) if (canvas[y * 28 + x] > 20) { min_x = std::min(min_x, x); max_x = std::max(max_x, x); min_y = std::min(min_y, y); max_y = std::max(max_y, y); }
    std::vector<double> out(784, 0.0); if (max_x < 0) return out;
    const int width = max_x - min_x + 1, height = max_y - min_y + 1;
    const double resize = std::min(20.0 / width, 20.0 / height);
    const int target_width = std::max(1, static_cast<int>(std::lround(width * resize)));
    const int target_height = std::max(1, static_cast<int>(std::lround(height * resize)));
    const int offset_x = (28 - target_width) / 2, offset_y = (28 - target_height) / 2;
    for (int y = 0; y < target_height; ++y) for (int x = 0; x < target_width; ++x) {
        const int sx = min_x + std::min(width - 1, static_cast<int>(x * width / static_cast<double>(target_width)));
        const int sy = min_y + std::min(height - 1, static_cast<int>(y * height / static_cast<double>(target_height)));
        out[(offset_y + y) * 28 + offset_x + x] = canvas[sy * 28 + sx] / 255.0;
    }
    return out;
}

static void draw(Display *display, Window window, GC gc, const std::vector<unsigned char> &canvas, const std::string &status, int progress_percent = 0)
{
    XClearWindow(display, window); XSetForeground(display, gc, 0x202124); XFillRectangle(display, window, gc, 0, 0, 720, 560);
    // The model was trained on MNIST's white-ink-on-black convention.
    // Keep the visible canvas exactly 28 x 28 logical pixels (18 px each).
    XSetForeground(display, gc, 0x000000); XFillRectangle(display, window, gc, 30, 30, 504, 504);
    for (int y = 0; y < 28; ++y) for (int x = 0; x < 28; ++x) if (canvas[y * 28 + x]) { XSetForeground(display, gc, (canvas[y * 28 + x] << 16) | (canvas[y * 28 + x] << 8) | canvas[y * 28 + x]); XFillRectangle(display, window, gc, 30 + x * 18, 30 + y * 18, 18, 18); }
    XSetForeground(display, gc, 0x4b8bff); XFillRectangle(display, window, gc, 560, 80, 120, 42); XFillRectangle(display, window, gc, 560, 140, 120, 42);
    XSetForeground(display, gc, 0x606060); XFillRectangle(display, window, gc, 550, 300, 150, 20);
    XSetForeground(display, gc, 0x46d369); XFillRectangle(display, window, gc, 550, 300, 150 * std::clamp(progress_percent, 0, 100) / 100, 20);
    XSetForeground(display, gc, 0xffffff); XDrawString(display, window, gc, 585, 107, const_cast<char*>("Predict"), 7); XDrawString(display, window, gc, 590, 167, const_cast<char*>("Clear"), 5); XDrawString(display, window, gc, 550, 250, const_cast<char*>(status.c_str()), status.size());
    XFlush(display);
}

int main(int argc, char **argv)
{
    try {
        const std::string model_path = argc > 1 ? argv[1] : "src/mnist_mlp.json";
        sealtorch::Evaluator sealtorch_model(load_model(model_path));
        const NeuralNetwork &model = sealtorch_model.model();
        EncryptedInference encrypted_inference(model);
        Display *display = XOpenDisplay(nullptr); if (!display) throw std::runtime_error("could not open X11 display");
        const int screen = DefaultScreen(display); Window window = XCreateSimpleWindow(display, RootWindow(display, screen), 100, 100, 720, 560, 1, BlackPixel(display, screen), 0x202124);
        XStoreName(display, window, "SEALTorch MNIST inference"); XSelectInput(display, window, ExposureMask | ButtonPressMask | ButtonReleaseMask | PointerMotionMask); XMapWindow(display, window); GC gc = XCreateGC(display, window, 0, nullptr);
        std::vector<unsigned char> canvas(784); std::string status = "Draw a digit, then click Predict"; bool drawing = false; XEvent event;
        std::future<std::vector<double>> inference_task;
        std::atomic<int> progress_percent{0};
        std::mutex progress_mutex;
        std::string progress_text = "Ready";
        bool inference_running = false;
        auto paint = [&](int pixel_x, int pixel_y) {
            const int x = (pixel_x - 30) / 18, y = (pixel_y - 30) / 18;
            if (x < 0 || x >= 28 || y < 0 || y >= 28) return;
            for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx)
                if (x + dx >= 0 && x + dx < 28 && y + dy >= 0 && y + dy < 28)
                    canvas[(y + dy) * 28 + x + dx] = 255;
            draw(display, window, gc, canvas, status);
        };
        while (true) {
            if (inference_running && inference_task.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                try {
                    const auto probabilities = inference_task.get();
                    const auto best = static_cast<int>(std::max_element(probabilities.begin(), probabilities.end()) - probabilities.begin());
                    status = "Prediction: " + std::to_string(best) + " (" + std::to_string(static_cast<int>(probabilities[best] * 100)) + "%)";
                } catch (const std::exception &error) {
                    status = std::string("Inference failed: ") + error.what();
                }
                inference_running = false;
                draw(display, window, gc, canvas, status, 100);
            }
            if (inference_running) {
                std::string visible_progress;
                { std::lock_guard<std::mutex> lock(progress_mutex); visible_progress = progress_text; }
                draw(display, window, gc, canvas, visible_progress, progress_percent.load());
            }
            if (XPending(display) == 0) { std::this_thread::sleep_for(std::chrono::milliseconds(30)); continue; }
            XNextEvent(display, &event);
            if (event.type == Expose) {
                draw(display, window, gc, canvas, status);
            } else if (event.type == ButtonPress) {
                if (event.xbutton.x >= 560 && event.xbutton.x <= 680 && event.xbutton.y >= 140 && event.xbutton.y <= 182) {
                    std::fill(canvas.begin(), canvas.end(), 0); status = "Canvas cleared"; draw(display, window, gc, canvas, status);
                } else if (!inference_running && event.xbutton.x >= 560 && event.xbutton.x <= 680 && event.xbutton.y >= 80 && event.xbutton.y <= 122) {
                    status = "Running encrypted inference...";
                    const auto input = preprocess(canvas);
                    progress_percent.store(0); { std::lock_guard<std::mutex> lock(progress_mutex); progress_text = "Starting encrypted inference..."; }
                    inference_running = true;
                    inference_task = std::async(std::launch::async, [&encrypted_inference, input, &progress_percent, &progress_mutex, &progress_text]() {
                        return encrypted_inference.predict(input, [&](int percent, const std::string &text) {
                            progress_percent.store(percent);
                            std::lock_guard<std::mutex> lock(progress_mutex);
                            progress_text = text;
                        });
                    });
                    draw(display, window, gc, canvas, status, 0);
                } else {
                    drawing = true; paint(event.xbutton.x, event.xbutton.y);
                }
            } else if (event.type == ButtonRelease) {
                drawing = false;
            } else if (event.type == MotionNotify && drawing) {
                paint(event.xmotion.x, event.xmotion.y);
            }
        }
        XFreeGC(display, gc); XCloseDisplay(display);
    } catch (const std::exception &error) { std::cerr << "SEALTorch GUI error: " << error.what() << '\n'; return 1; }
}
