#include "inference.h"
#include "packing.h"

#include <cuda_runtime_api.h>
#include <fideslib.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unistd.h>
#include <utility>

namespace sealtorch {
namespace {

using Context = fideslib::CryptoContext<fideslib::DCRTPoly>;
using Ciphertext = fideslib::Ciphertext<fideslib::DCRTPoly>;
using Plaintext = fideslib::Plaintext;

struct PackedLayer {
    struct Diagonal { std::size_t input_index = 0; Plaintext weights; };
    struct Group { std::int32_t rotation = 0; std::vector<Diagonal> diagonals; };
    int input_size = 0;
    int output_size = 0;
    std::vector<std::int32_t> input_rotations;
    std::vector<Group> groups;
    Plaintext bias;
};

std::size_t peak_resident_memory_bytes() {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (!line.starts_with("VmHWM:")) continue;
        std::istringstream value(line.substr(6));
        std::size_t kibibytes = 0;
        if (value >> kibibytes) return kibibytes * 1024;
    }
    return 0;
}

std::size_t current_resident_memory_bytes() {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (!line.starts_with("VmRSS:")) continue;
        std::istringstream value(line.substr(6));
        std::size_t kibibytes = 0;
        if (value >> kibibytes) return kibibytes * 1024;
    }
    return 0;
}

std::size_t used_vram_bytes(int device) {
    cudaSetDevice(device);
    std::size_t free = 0, total = 0;
    if (cudaMemGetInfo(&free, &total) != cudaSuccess) return 0;
    return total - free;
}

void require_device(int device) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || device < 0 || device >= count)
        throw std::runtime_error("FIDESlib requires an available CUDA device");
}

PackedLayer pack_layer(const Context& context, const DenseLayer& layer, std::size_t slots,
                       std::size_t level) {
    PackedLayer packed;
    packed.input_size = layer.input_size;
    packed.output_size = layer.output_size;
    std::map<std::size_t, std::size_t> baby_indexes;
    std::map<std::size_t, std::size_t> group_indexes;
    const auto diagonals = active_diagonals(layer, slots);
    const std::size_t baby_step = baby_step_size(slots, diagonals.size());

    for (const std::size_t diagonal : diagonals) {
        const auto split = split_diagonal(diagonal, baby_step);
        if (!baby_indexes.contains(split.baby)) {
            baby_indexes[split.baby] = packed.input_rotations.size();
            packed.input_rotations.push_back(signed_rotation(split.baby, slots));
        }
        if (!group_indexes.contains(split.giant)) {
            group_indexes[split.giant] = packed.groups.size();
            packed.groups.push_back({ signed_rotation(split.giant, slots), {} });
        }
        std::vector<double> values(slots, 0.0);
        for (std::size_t row = 0; row < layer.weights.size(); ++row) {
            const std::size_t column = (row + diagonal) % slots;
            if (column < layer.weights[row].size())
                values[(row + split.giant) % slots] = layer.weights[row][column];
        }
        packed.groups[group_indexes.at(split.giant)].diagonals.push_back(
            { baby_indexes.at(split.baby),
              context->MakeCKKSPackedPlaintext(values, 1, static_cast<std::uint32_t>(level)) });
    }
    std::vector<double> bias(slots, 0.0);
    std::copy(layer.biases.begin(), layer.biases.end(), bias.begin());
    packed.bias = context->MakeCKKSPackedPlaintext(
        bias, 1, static_cast<std::uint32_t>(level + 1));
    return packed;
}

void load_layer(const Context& context, PackedLayer& layer) {
    for (auto& group : layer.groups)
        for (auto& diagonal : group.diagonals)
            context->LoadPlaintext(diagonal.weights);
    context->LoadPlaintext(layer.bias);
}

void unload_plaintext(const Context& context, Plaintext& plaintext) {
    if (!plaintext || !plaintext->loaded) return;
    if (!context->EvictDevicePlaintext(plaintext->gpu))
        throw std::runtime_error("FIDESlib could not evict streamed plaintext weights");
    plaintext->gpu = 0;
    plaintext->loaded = false;
}

void unload_layer(const Context& context, PackedLayer& layer) {
    for (auto& group : layer.groups)
        for (auto& diagonal : group.diagonals)
            unload_plaintext(context, diagonal.weights);
    unload_plaintext(context, layer.bias);
}

Ciphertext linear(const Context& context, const Ciphertext& input, PackedLayer& layer) {
    std::vector<std::int32_t> nonzero_steps;
    for (const auto step : layer.input_rotations)
        if (step != 0) nonzero_steps.push_back(step);
    std::vector<Ciphertext> hoisted;
    if (!nonzero_steps.empty()) {
        const auto precomputation = context->EvalFastRotationPrecompute(input);
        hoisted = context->EvalFastRotation(
            input, nonzero_steps, context->GetCyclotomicOrder(), precomputation);
    }
    std::vector<Ciphertext> rotations;
    rotations.reserve(layer.input_rotations.size());
    std::size_t nonzero_index = 0;
    for (const auto step : layer.input_rotations)
        rotations.push_back(step == 0 ? input->Clone() : std::move(hoisted[nonzero_index++]));

    Ciphertext result;
    bool first_group = true;
    for (auto& group : layer.groups) {
        Ciphertext subtotal;
        bool first_diagonal = true;
        for (auto& diagonal : group.diagonals) {
            auto term = context->EvalMult(rotations[diagonal.input_index], diagonal.weights);
            if (first_diagonal) {
                subtotal = std::move(term);
                first_diagonal = false;
            } else {
                context->EvalAddInPlace(subtotal, term);
            }
        }
        if (group.rotation != 0) context->EvalRotateInPlace(subtotal, group.rotation);
        if (first_group) {
            result = std::move(subtotal);
            first_group = false;
        } else {
            context->EvalAddInPlace(result, subtotal);
        }
    }
    context->RescaleInPlace(result);
    context->EvalAddInPlace(result, layer.bias);
    return result;
}

} // namespace

struct InferenceEngine::Implementation {
    Model model;
    InferenceOptions options;
    Context context;
    fideslib::KeyPair<fideslib::DCRTPoly> keys;
    std::vector<std::unique_ptr<PackedLayer>> layers;
    std::size_t ram_baseline = 0;
    std::size_t vram_baseline = 0;

    Implementation(Model model_value, InferenceOptions options_value)
        : model(std::move(model_value)), options(std::move(options_value)) {
        require_device(options.device);
        if (options.ring_dimension < 1024 ||
            (options.ring_dimension & (options.ring_dimension - 1)) != 0)
            throw std::runtime_error("ring dimension must be a power of two of at least 1024");
        if (static_cast<std::size_t>(model.input_size()) > slot_count())
            throw std::runtime_error("model input does not fit in one CKKS ciphertext");
        if (options.chebyshev_coefficients.size() < 2)
            throw std::runtime_error("at least two polynomial coefficients are required");

        cudaSetDevice(options.device);
        ram_baseline = current_resident_memory_bytes();
        vram_baseline = used_vram_bytes(options.device);
        fideslib::CCParams<fideslib::CryptoContextCKKSRNS> parameters;
        parameters.SetRingDim(static_cast<std::uint32_t>(options.ring_dimension));
        parameters.SetBatchSize(static_cast<std::uint32_t>(slot_count()));
        parameters.SetMultiplicativeDepth(static_cast<std::uint32_t>(options.multiplicative_depth));
        parameters.SetScalingModSize(static_cast<std::uint32_t>(options.scaling_modulus_bits));
        parameters.SetFirstModSize(static_cast<std::uint32_t>(options.first_modulus_bits));
        parameters.SetSecurityLevel(fideslib::HEStd_128_classic);
        parameters.SetScalingTechnique(fideslib::FLEXIBLEAUTO);
        parameters.SetKeySwitchTechnique(fideslib::HYBRID);
        parameters.SetDevices(std::vector<int>{ options.device });
        parameters.SetPlaintextAutoload(false);
        parameters.SetCiphertextAutoload(true);
        context = fideslib::GenCryptoContext(parameters);
        context->Enable(fideslib::PKE);
        context->Enable(fideslib::KEYSWITCH);
        context->Enable(fideslib::LEVELEDSHE);
        context->Enable(fideslib::ADVANCEDSHE);

        layers.resize(model.operations().size());
        std::set<std::int32_t> rotations;
        for (std::size_t index = 0; index < model.operations().size(); ++index) {
            const auto& operation = model.operations()[index];
            if (operation.kind != OperationKind::Linear) continue;
            const auto diagonals = active_diagonals(operation.linear, slot_count());
            const auto baby_step = baby_step_size(slot_count(), diagonals.size());
            for (const auto diagonal : diagonals) {
                const auto split = split_diagonal(diagonal, baby_step);
                const auto baby = signed_rotation(split.baby, slot_count());
                const auto giant = signed_rotation(split.giant, slot_count());
                if (baby != 0) rotations.insert(baby);
                if (giant != 0) rotations.insert(giant);
            }
        }

        keys = context->KeyGen();
        context->EvalMultKeyGen(keys.secretKey);
        context->EvalRotateKeyGen(keys.secretKey, { rotations.begin(), rotations.end() });
        context->LoadContext(keys.publicKey);
        context->Synchronize();
    }

    std::size_t slot_count() const { return options.ring_dimension / 2; }

    InferenceResult predict(const std::vector<double>& input) {
        if (input.size() != static_cast<std::size_t>(model.input_size()))
            throw std::runtime_error("input size does not match model input");
        InferenceResult result;
        const auto encrypt_start = std::chrono::steady_clock::now();
        auto plain = context->MakeCKKSPackedPlaintext(input);
        auto encrypted = context->Encrypt(keys.publicKey, plain);
        context->Synchronize();
        const auto encrypt_end = std::chrono::steady_clock::now();

        for (std::size_t index = 0; index < model.operations().size(); ++index) {
            if (model.operations()[index].kind == OperationKind::Linear) {
                // Packed LeNet layers contain many CKKS plaintext diagonals.
                // Build each layer lazily at the ciphertext level where it is
                // consumed. In low-VRAM mode it returns to CPU memory after
                // use; resident mode retains it on the GPU for later images.
                if (!layers[index]) {
                    layers[index] = std::make_unique<PackedLayer>(pack_layer(
                        context, model.operations()[index].linear, slot_count(), encrypted->GetLevel()));
                }
                load_layer(context, *layers[index]);
                encrypted = linear(context, encrypted, *layers[index]);
                if (options.stream_weights) {
                    context->Synchronize();
                    unload_layer(context, *layers[index]);
                }
            } else {
                auto coefficients = options.chebyshev_coefficients;
                while (coefficients.size() > 1 && std::abs(coefficients.back()) < 1e-15)
                    coefficients.pop_back();
                if (coefficients.size() == 1) {
                    context->EvalMultInPlace(encrypted, 0.0);
                    context->RescaleInPlace(encrypted);
                    context->EvalAddInPlace(encrypted, coefficients.front());
                } else if (coefficients.size() == 2) {
                    const double width = options.upper_bound - options.lower_bound;
                    const double slope = coefficients[1] * 2.0 / width;
                    const double intercept = coefficients[0] -
                        coefficients[1] * (options.upper_bound + options.lower_bound) / width;
                    context->EvalMultInPlace(encrypted, slope);
                    context->RescaleInPlace(encrypted);
                    context->EvalAddInPlace(encrypted, intercept);
                } else if (coefficients.size() == 3) {
                    // FIDESlib's degree-2 Chebyshev-series path is unstable
                    // after the first trained MLP layer. Evaluate the exact
                    // same quadratic explicitly in normalized coordinates.
                    // Wire coefficients use OpenFHE's c0/2 convention.
                    const double width = options.upper_bound - options.lower_bound;
                    const double slope = 2.0 / width;
                    const double intercept = -(options.upper_bound + options.lower_bound) / width;
                    auto normalized = context->EvalMult(encrypted, slope);
                    if (intercept != 0.0) context->EvalAddInPlace(normalized, intercept);
                    auto squared = context->EvalSquare(normalized);
                    auto result = context->EvalMult(squared, 2.0 * coefficients[2]);
                    auto linear = context->EvalMult(normalized, coefficients[1]);
                    context->EvalAddInPlace(result, linear);
                    context->EvalAddInPlace(result, coefficients[0] / 2.0 - coefficients[2]);
                    encrypted = std::move(result);
                } else {
                    context->EvalChebyshevSeriesInPlace(
                        encrypted, coefficients, options.lower_bound, options.upper_bound);
                }
            }
        }
        context->Synchronize();
        const auto evaluate_end = std::chrono::steady_clock::now();

        Plaintext decoded;
        context->Decrypt(keys.secretKey, encrypted, &decoded);
        context->Synchronize();
        result.values = decoded->GetRealPackedValue();
        result.values.resize(static_cast<std::size_t>(model.output_size()));
        const auto decrypt_end = std::chrono::steady_clock::now();
        result.encrypt_ms = std::chrono::duration<double, std::milli>(encrypt_end - encrypt_start).count();
        result.evaluate_ms = std::chrono::duration<double, std::milli>(evaluate_end - encrypt_end).count();
        result.decrypt_ms = std::chrono::duration<double, std::milli>(decrypt_end - evaluate_end).count();
        result.idle_ram_bytes = ram_baseline;
        result.ram_bytes = peak_resident_memory_bytes();
        const auto used = used_vram_bytes(options.device);
        result.idle_vram_bytes = vram_baseline;
        result.vram_bytes = used > vram_baseline ? used - vram_baseline : 0;
        if (!std::all_of(result.values.begin(), result.values.end(), [](double value) { return std::isfinite(value); }))
            throw std::runtime_error("encrypted inference produced non-finite logits");
        return result;
    }
};

InferenceEngine::InferenceEngine(Model model, InferenceOptions options)
    : implementation_(std::make_unique<Implementation>(std::move(model), std::move(options))) {}
InferenceEngine::~InferenceEngine() = default;
InferenceEngine::InferenceEngine(InferenceEngine&&) noexcept = default;
InferenceEngine& InferenceEngine::operator=(InferenceEngine&&) noexcept = default;
InferenceResult InferenceEngine::predict(const std::vector<double>& input) { return implementation_->predict(input); }
std::size_t InferenceEngine::slot_count() const { return implementation_->slot_count(); }
bool InferenceEngine::available() { int count = 0; return cudaGetDeviceCount(&count) == cudaSuccess && count > 0; }

} // namespace sealtorch
