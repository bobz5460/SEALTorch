#include <SEALTorch/cuda_ciphertext_inference.h>
#include <SEALTorch/packing.h>

#include <cuda_runtime_api.h>
#include <fideslib.hpp>

#include <algorithm>
#include <chrono>
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

        void require_cuda_device(int device)
        {
            int device_count = 0;
            if (cudaGetDeviceCount(&device_count) != cudaSuccess || device < 0 || device >= device_count)
                throw std::runtime_error("FIDESlib inference needs an available CUDA device");
        }

        std::vector<std::int32_t> rotation_steps(
            const Sequential &model,
            std::size_t slot_count)
        {
            std::vector<std::int32_t> steps;
            for (const Operation &operation : model.operations())
            {
                if (operation.kind != OperationKind::Linear)
                    continue;

                for (std::size_t diagonal :
                     active_diagonals(operation.linear_layer, slot_count))
                {
                    const DiagonalSplit split =
                        split_diagonal(diagonal, slot_count);
                    const std::int32_t baby_rotation =
                        signed_rotation(split.baby, slot_count);
                    const std::int32_t giant_rotation =
                        signed_rotation(split.giant, slot_count);
                    if (baby_rotation != 0)
                        steps.push_back(baby_rotation);
                    if (giant_rotation != 0)
                        steps.push_back(giant_rotation);
                }
            }
            std::sort(steps.begin(), steps.end());
            steps.erase(std::unique(steps.begin(), steps.end()), steps.end());
            return steps;
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

            for (std::size_t diagonal : active_diagonals(layer, slot_count))
            {
                const DiagonalSplit split =
                    split_diagonal(diagonal, slot_count);
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
                        : context->EvalRotate(input, rotation));
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
                    subtotal = context->EvalRotate(
                        subtotal, group.rotation);

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

        Ciphertext activate_polynomial(
            const Context &context,
            const Ciphertext &input,
            double constant,
            double linear_coefficient,
            double quadratic_coefficient,
            double quartic_coefficient,
            std::size_t slot_count)
        {
            Ciphertext squared = context->EvalMult(input, input);
            context->RescaleInPlace(squared);

            Ciphertext fourth = context->EvalMult(squared, squared);
            context->RescaleInPlace(fourth);

            Ciphertext linear_term = context->EvalMult(input, linear_coefficient);
            context->RescaleInPlace(linear_term);
            Ciphertext quadratic_term = context->EvalMult(squared, quadratic_coefficient);
            context->RescaleInPlace(quadratic_term);
            Ciphertext result = context->EvalMult(fourth, quartic_coefficient);
            context->RescaleInPlace(result);

            context->EvalAddInPlace(result, quadratic_term);
            context->EvalAddInPlace(result, linear_term);
            Plaintext constant_term = context->MakeCKKSPackedPlaintext(
                std::vector<double>(slot_count, constant));
            context->EvalAddInPlace(result, constant_term);
            return result;
        }

        Ciphertext activate_tanh(
            const Context &context,
            const Ciphertext &input)
        {
            Ciphertext squared = context->EvalMult(input, input);
            context->RescaleInPlace(squared);
            Ciphertext cubic = context->EvalMult(
                squared, -0.003956717265710641);
            context->RescaleInPlace(cubic);
            cubic = context->EvalMult(cubic, input);
            context->RescaleInPlace(cubic);

            Ciphertext linear = context->EvalMult(
                input, 0.3370407386009496);
            context->RescaleInPlace(linear);
            context->EvalAddInPlace(cubic, linear);
            return cubic;
        }

        Ciphertext activate(
            const Context &context,
            const Ciphertext &input,
            ActivationType type,
            std::size_t slot_count)
        {
            if (type == ActivationType::Relu)
            {
                return activate_polynomial(
                    context, input,
                    0.33810450, 0.5, 0.096514968, -0.00053277056,
                    slot_count);
            }
            if (type == ActivationType::Gelu)
            {
                return activate_polynomial(
                    context, input,
                    0.0, 0.5, 0.3989422804014327, -0.0664903800669054,
                    slot_count);
            }
            return activate_tanh(context, input);
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
        std::vector<PackedLayer> layers;

        Implementation(Sequential model_value, CiphertextInferenceOptions options_value)
            : options(std::move(options_value))
        {
            std::size_t dense_layer_count = 0;
            const DenseLayer *first_layer = nullptr;
            for (const Operation &operation : model_value.operations())
            {
                if (operation.kind == OperationKind::Linear)
                {
                    if (first_layer == nullptr)
                        first_layer = &operation.linear_layer;
                    ++dense_layer_count;
                }
                else if (operation.kind != OperationKind::Activation)
                {
                    throw std::runtime_error(
                        "FIDESlib inference currently supports dense models only");
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
            parameters.SetPlaintextAutoload(true);
            parameters.SetCiphertextAutoload(true);
            context = fideslib::GenCryptoContext(parameters);
            context->Enable(fideslib::PKE);
            context->Enable(fideslib::KEYSWITCH);
            context->Enable(fideslib::LEVELEDSHE);

            keys = context->KeyGen();
            context->EvalMultKeyGen(keys.secretKey);
            context->EvalRotateKeyGen(
                keys.secretKey, rotation_steps(model_value, slot_count()));
            context->LoadContext(keys.publicKey);

            layers.reserve(dense_layer_count);
            std::size_t dense_index = 0;
            for (const Operation &operation : model_value.operations())
            {
                if (operation.kind != OperationKind::Linear)
                    continue;
                const bool has_activation =
                    model_value.has_activation(dense_index);
                layers.push_back(prepare_layer(
                    context,
                    operation.linear_layer,
                    slot_count(),
                    has_activation,
                    has_activation
                        ? model_value.activation(dense_index)
                        : ActivationType::Relu));
                ++dense_index;
            }
        }

        std::size_t slot_count() const { return options.ring_dimension / 2; }

        CiphertextInferenceResult predict(const std::vector<double> &input)
        {
            if (input.size() != static_cast<std::size_t>(layers.front().input_size))
                throw std::runtime_error("input size does not match the model");

            CiphertextInferenceResult result;
            const auto encrypt_start = std::chrono::steady_clock::now();
            Plaintext plain = context->MakeCKKSPackedPlaintext(input);
            Ciphertext encrypted = context->Encrypt(keys.secretKey, plain);
            cudaDeviceSynchronize();
            const auto encrypt_end = std::chrono::steady_clock::now();
            const auto evaluate_start = encrypt_end;
            for (PackedLayer &layer : layers)
            {
                encrypted = linear(context, encrypted, layer);
                if (layer.has_activation)
                    encrypted = activate(context, encrypted, layer.activation, slot_count());
            }
            cudaDeviceSynchronize();
            const auto evaluate_end = std::chrono::steady_clock::now();

            const auto decrypt_start = evaluate_end;
            Plaintext decoded;
            context->Decrypt(encrypted, keys.secretKey, &decoded);
            cudaDeviceSynchronize();
            result.values = decoded->GetRealPackedValue();
            result.values.resize(static_cast<std::size_t>(layers.back().output_size));
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
