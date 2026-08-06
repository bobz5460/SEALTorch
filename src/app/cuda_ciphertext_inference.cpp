#include "cuda_ciphertext_inference.h"
#include <SEALTorch/math.h>
#include <SEALTorch/packing.h>

#include <cuda_runtime_api.h>
#include <fideslib.hpp>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <map>
#include <limits>
#include <stdexcept>
#include <utility>

namespace sealtorch::cuda
{
    namespace
    {
        using Context = fideslib::CryptoContext<fideslib::DCRTPoly>;
        using Ciphertext = fideslib::Ciphertext<fideslib::DCRTPoly>;
        using Plaintext = fideslib::Plaintext;

        struct PackedLayer
        {
            struct Diagonal
            {
                std::size_t input_index = 0;
                Plaintext weights;
            };

            struct Group
            {
                std::int32_t rotation = 0;
                std::vector<Diagonal> diagonals;
            };

            int input_size = 0;
            int output_size = 0;
            std::vector<std::int32_t> input_rotations;
            std::vector<Group> groups;
            Plaintext bias;
            ActivationType activation = ActivationType::Relu;
            bool has_activation = false;
        };

        void load_layer(const Context &context, PackedLayer &layer)
        {
            for (auto &group : layer.groups)
                for (auto &diagonal : group.diagonals)
                    context->LoadPlaintext(diagonal.weights);
            context->LoadPlaintext(layer.bias);
        }

        void evict_plaintext(const Context &context, Plaintext &plaintext)
        {
            if (!plaintext->loaded) return;
            context->EvictDevicePlaintext(plaintext->gpu);
            plaintext->loaded = false;
            plaintext->gpu = 0;
        }

        void evict_layer(const Context &context, PackedLayer &layer)
        {
            cudaDeviceSynchronize();
            for (auto &group : layer.groups)
                for (auto &diagonal : group.diagonals)
                    evict_plaintext(context, diagonal.weights);
            evict_plaintext(context, layer.bias);
        }

        void require_cuda_device(int device)
        {
            int device_count = 0;
            if (cudaGetDeviceCount(&device_count) != cudaSuccess || device < 0 || device >= device_count)
                throw std::runtime_error("FIDESlib inference needs an available CUDA device");
        }

        std::vector<std::int32_t> rotation_steps(
            std::size_t slot_count)
        {
            std::vector<std::int32_t> steps;
            for (std::size_t power = 1; power <= slot_count / 2; power <<= 1) {
                steps.push_back(static_cast<std::int32_t>(power));
                steps.push_back(-static_cast<std::int32_t>(power));
            }
            std::sort(steps.begin(), steps.end());
            steps.erase(std::unique(steps.begin(), steps.end()), steps.end());
            return steps;
        }

        Ciphertext rotate(
            const Context &context, const Ciphertext &input,
            std::int32_t rotation)
        {
            if (rotation == 0) return input;
            Ciphertext result = input;
            std::uint32_t remaining = static_cast<std::uint32_t>(
                rotation < 0 ? -static_cast<std::int64_t>(rotation) : rotation);
            std::int32_t power = rotation < 0 ? -1 : 1;
            while (remaining != 0) {
                if (remaining & 1U)
                    result = context->EvalRotate(result, power);
                remaining >>= 1U;
                power *= 2;
            }
            return result;
        }

        PackedLayer prepare_layer(
            const Context &context,
            const DenseLayer &layer,
            std::size_t slot_count,
            bool has_activation, ActivationType activation)
        {
            PackedLayer prepared;
            prepared.input_size = layer.input_size;
            prepared.output_size = layer.output_size;
            prepared.activation = activation;
            prepared.has_activation = has_activation;
            std::map<std::size_t, std::size_t> baby_indices;
            std::map<std::size_t, std::size_t> group_indices;

            const std::vector<std::size_t> diagonals =
                active_diagonals(layer, slot_count);
            const std::size_t baby_step =
                baby_step_size(slot_count, diagonals.size());
            for (std::size_t diagonal : diagonals)
            {
                const DiagonalSplit split =
                    split_diagonal(diagonal, baby_step);
                if (baby_indices.find(split.baby) == baby_indices.end())
                {
                    const std::size_t index =
                        prepared.input_rotations.size();
                    baby_indices[split.baby] = index;
                    prepared.input_rotations.push_back(
                        signed_rotation(split.baby, slot_count));
                }
                if (group_indices.find(split.giant) == group_indices.end())
                {
                    const std::size_t index = prepared.groups.size();
                    group_indices[split.giant] = index;
                    PackedLayer::Group group;
                    group.rotation =
                        signed_rotation(split.giant, slot_count);
                    prepared.groups.push_back(std::move(group));
                }

                std::vector<double> values(slot_count, 0.0);
                for (std::size_t row = 0; row < layer.weights.size(); ++row)
                {
                    const std::size_t column =
                        (row + diagonal) % slot_count;
                    if (column < layer.weights[row].size())
                    {
                        const std::size_t shifted_row =
                            (row + split.giant) % slot_count;
                        values[shifted_row] =
                            layer.weights[row][column];
                    }
                }

                PackedLayer::Diagonal encoded;
                encoded.input_index = baby_indices.at(split.baby);
                encoded.weights = context->MakeCKKSPackedPlaintext(values);
                prepared.groups[group_indices.at(split.giant)]
                    .diagonals.push_back(std::move(encoded));
            }

            std::vector<double> bias(slot_count, 0.0);
            std::copy(layer.biases.begin(), layer.biases.end(), bias.begin());
            prepared.bias = context->MakeCKKSPackedPlaintext(bias);
            return prepared;
        }

        Ciphertext linear(const Context &context, const Ciphertext &input, PackedLayer &layer)
        {
            std::vector<Ciphertext> rotated_inputs;
            rotated_inputs.reserve(layer.input_rotations.size());
            for (std::int32_t rotation : layer.input_rotations)
            {
                rotated_inputs.push_back(
                    rotation == 0
                        ? input
                        : rotate(context, input, rotation));
            }

            Ciphertext result;
            bool first_group = true;
            for (PackedLayer::Group &group : layer.groups)
            {
                Ciphertext subtotal;
                bool first_diagonal = true;
                for (PackedLayer::Diagonal &diagonal : group.diagonals)
                {
                    Ciphertext term = context->EvalMult(
                        rotated_inputs[diagonal.input_index],
                        diagonal.weights);
                    if (first_diagonal) {
                        subtotal = std::move(term);
                        first_diagonal = false;
                    } else {
                        context->EvalAddInPlace(subtotal, term);
                    }
                }
                if (group.rotation != 0)
                    subtotal = rotate(context, subtotal, group.rotation);

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

        Ciphertext activate(
            const Context &context,
            const Ciphertext &input,
            ActivationType type,
            std::size_t degree,
            double range,
            ActivationApproximation method)
        {
            const auto coefficients = activation_polynomial_coefficients(
                type, degree, range, method);
            const std::size_t highest = coefficients.size() - 1;

            const bool odd_polynomial = coefficients.size() >= 4 &&
                [&coefficients]() {
                    for (std::size_t order = 0;
                         order < coefficients.size(); order += 2)
                        if (coefficients[order] != 0.0) return false;
                    return true;
                }();
            if (odd_polynomial) {
                Ciphertext square = context->EvalMult(input, input);
                context->RescaleInPlace(square);
                Ciphertext result = square->Clone();
                context->EvalMultInPlace(result, coefficients[highest]);
                context->RescaleInPlace(result);
                for (std::size_t order = highest - 2;
                     order > 1; order -= 2) {
                    if (coefficients[order] != 0.0)
                        context->EvalAddInPlace(result, coefficients[order]);
                    Ciphertext factor = square->Clone();
                    factor->SetLevel(result->GetLevel());
                    result = context->EvalMult(result, factor);
                    context->RescaleInPlace(result);
                }
                if (coefficients[1] != 0.0)
                    context->EvalAddInPlace(result, coefficients[1]);
                Ciphertext factor = input->Clone();
                factor->SetLevel(result->GetLevel());
                result = context->EvalMult(result, factor);
                context->RescaleInPlace(result);
                return result;
            }

            Ciphertext result = input->Clone();
            if (coefficients[highest] != 1.0) {
                context->EvalMultInPlace(result, coefficients[highest]);
                context->RescaleInPlace(result);
            }
            for (std::size_t order = highest; order-- > 1;) {
                if (coefficients[order] != 0.0)
                    context->EvalAddInPlace(result, coefficients[order]);
                Ciphertext factor = input->Clone();
                factor->SetLevel(result->GetLevel());
                result = context->EvalMult(result, factor);
                context->RescaleInPlace(result);
            }
            if (coefficients[0] != 0.0)
                context->EvalAddInPlace(result, coefficients[0]);
            return result;
        }
    }

    bool cuda_available()
    {
        int device_count = 0;
        return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
    }

    struct CiphertextInferenceEngine::Implementation
    {
        CiphertextInferenceOptions options;
        Context context;
        fideslib::KeyPair<fideslib::DCRTPoly> keys;
        Sequential model;
        std::vector<std::unique_ptr<PackedLayer>> cached_layers;
        std::vector<bool> resident_layers;

        Implementation(Sequential model_value, CiphertextInferenceOptions options_value)
            : options(std::move(options_value)), model(std::move(model_value))
        {
            if (options.thread_count == 0 ||
                options.thread_count >
                    static_cast<std::size_t>(std::numeric_limits<int>::max()))
                throw std::runtime_error(
                    "CUDA thread count must be between 1 and INT_MAX");

            // FIDESlib dispatches CPU-side work through OpenMP. Its default is
            // often all host cores, including for CUDA-backed operations.
            omp_set_dynamic(0);
            omp_set_num_threads(static_cast<int>(options.thread_count));

            const DenseLayer *first_layer = nullptr;
            for (const Operation &operation : model.operations())
            {
                if (operation.kind == OperationKind::Linear)
                {
                    if (first_layer == nullptr)
                        first_layer = &operation.linear_layer;
                }
            }
            if (first_layer == nullptr)
                throw std::runtime_error("FIDESlib inference needs at least one dense layer");
            if (options.ring_dimension < 1024 || (options.ring_dimension & (options.ring_dimension - 1)))
                throw std::runtime_error("ring_dimension must be a power of two of at least 1024");
            const std::size_t maximum_parameter =
                std::numeric_limits<std::uint32_t>::max();
            if (options.ring_dimension > maximum_parameter ||
                options.multiplicative_depth > maximum_parameter ||
                options.scaling_modulus_bits > maximum_parameter ||
                options.first_modulus_bits > maximum_parameter)
                throw std::runtime_error("FIDESlib parameters are too large");
            if (static_cast<std::size_t>(first_layer->input_size) > slot_count())
                throw std::runtime_error("model input does not fit in one CKKS ciphertext");

            require_cuda_device(options.device);
            fideslib::CCParams<fideslib::CryptoContextCKKSRNS> parameters;
            parameters.SetRingDim(
                static_cast<std::uint32_t>(options.ring_dimension));
            parameters.SetBatchSize(
                static_cast<std::uint32_t>(options.ring_dimension / 2));
            parameters.SetMultiplicativeDepth(
                static_cast<std::uint32_t>(options.multiplicative_depth));
            parameters.SetScalingModSize(
                static_cast<std::uint32_t>(options.scaling_modulus_bits));
            parameters.SetFirstModSize(
                static_cast<std::uint32_t>(options.first_modulus_bits));
            parameters.SetSecurityLevel(fideslib::HEStd_NotSet);
            parameters.SetScalingTechnique(fideslib::FLEXIBLEAUTO);
            parameters.SetKeySwitchTechnique(fideslib::HYBRID);
            parameters.SetDevices(std::vector<int>{options.device});
            // CNN lowering has thousands of sparse diagonals. Eagerly
            // loading all of them permanently occupies GPU memory; let
            // FIDES transfer a diagonal when its multiplication is issued.
            // Keep packed CPU encodings for every model constant. Selected
            // layers are loaded permanently below; oversized layers reuse the
            // CPU encoding while their GPU copy is loaded/evicted per run.
            parameters.SetPlaintextAutoload(false);
            parameters.SetCiphertextAutoload(true);
            context = fideslib::GenCryptoContext(parameters);
            context->Enable(fideslib::PKE);
            context->Enable(fideslib::KEYSWITCH);
            context->Enable(fideslib::LEVELEDSHE);

            keys = context->KeyGen();
            context->EvalMultKeyGen(keys.secretKey);
            context->EvalRotateKeyGen(
                keys.secretKey, rotation_steps(slot_count()));
            context->LoadContext(keys.publicKey);

            cached_layers.resize(model.operations().size());
            resident_layers.resize(model.operations().size(), false);
            for (std::size_t index = 0;
                 index < model.operations().size(); ++index) {
                const Operation &operation = model.operations()[index];
                if (operation.kind != OperationKind::Linear) continue;
                const std::size_t diagonal_count = active_diagonals(
                    operation.linear_layer, slot_count()).size();
                // Cache normal layers, but stream unusually large lowered
                // convolutions. This avoids both all-model OOM and all-layer
                // repacking on every prediction.
                cached_layers[index] = std::make_unique<PackedLayer>(
                    prepare_layer(
                        context, operation.linear_layer, slot_count(),
                        false, ActivationType::Relu));
                if (diagonal_count <= 2048) {
                    load_layer(context, *cached_layers[index]);
                    resident_layers[index] = true;
                }
            }
        }

        std::size_t slot_count() const { return options.ring_dimension / 2; }

        CiphertextInferenceResult predict(const std::vector<double> &input)
        {
            if (input.size() != static_cast<std::size_t>(model.input_size()))
                throw std::runtime_error("input size does not match the model");

            CiphertextInferenceResult result;
            const auto encrypt_start = std::chrono::steady_clock::now();
            Plaintext plain = context->MakeCKKSPackedPlaintext(input);
            Ciphertext encrypted = context->Encrypt(keys.secretKey, plain);
            cudaDeviceSynchronize();
            const auto encrypt_end = std::chrono::steady_clock::now();
            const auto evaluate_start = encrypt_end;
            for (std::size_t index = 0;
                 index < model.operations().size(); ++index) {
                const Operation &operation = model.operations()[index];
                if (operation.kind == OperationKind::Linear) {
                    PackedLayer &layer = *cached_layers[index];
                    encrypted = linear(context, encrypted, layer);
                    if (!resident_layers[index])
                        evict_layer(context, layer);
                } else {
                    encrypted = activate(
                        context, encrypted, operation.activation_type,
                        options.activation_degree, options.activation_range,
                        options.approximation_method);
                }
            }
            cudaDeviceSynchronize();
            const auto evaluate_end = std::chrono::steady_clock::now();

            const auto decrypt_start = evaluate_end;
            Plaintext decoded;
            context->Decrypt(encrypted, keys.secretKey, &decoded);
            cudaDeviceSynchronize();
            result.values = decoded->GetRealPackedValue();
            result.values.resize(static_cast<std::size_t>(model.output_size()));
            if (!std::all_of(
                    result.values.begin(), result.values.end(),
                    [](double value) { return std::isfinite(value); }))
                throw std::runtime_error(
                    "ciphertext inference produced non-finite values; "
                    "increase CKKS modulus precision");
            const auto decrypt_end = std::chrono::steady_clock::now();
            result.encrypt_ms = std::chrono::duration<double, std::milli>(encrypt_end - encrypt_start).count();
            result.evaluate_ms = std::chrono::duration<double, std::milli>(evaluate_end - evaluate_start).count();
            result.decrypt_ms = std::chrono::duration<double, std::milli>(decrypt_end - decrypt_start).count();
            return result;
        }
    };

    CiphertextInferenceEngine::CiphertextInferenceEngine(Sequential model, CiphertextInferenceOptions options)
        : implementation_(std::make_unique<Implementation>(std::move(model), std::move(options))) {}
    CiphertextInferenceEngine::~CiphertextInferenceEngine() = default;
    CiphertextInferenceEngine::CiphertextInferenceEngine(CiphertextInferenceEngine &&) noexcept = default;
    CiphertextInferenceEngine &CiphertextInferenceEngine::operator=(CiphertextInferenceEngine &&) noexcept = default;
    CiphertextInferenceResult CiphertextInferenceEngine::predict(
        const std::vector<double> &input)
    {
        return implementation_->predict(input);
    }
    std::size_t CiphertextInferenceEngine::slot_count() const { return implementation_->slot_count(); }
}
