#include <SEALTorch/cuda_ciphertext_inference.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
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
            DenseLayer parameters;
            std::vector<Plaintext> diagonals;
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

        std::vector<int32_t> rotation_steps(const std::vector<DenseLayer> &layers)
        {
            std::vector<int32_t> steps;
            for (const DenseLayer &layer : layers)
                for (int step = 1; step < layer.input_size; ++step)
                    steps.push_back(static_cast<int32_t>(step));
            std::sort(steps.begin(), steps.end());
            steps.erase(std::unique(steps.begin(), steps.end()), steps.end());
            return steps;
        }

        PackedLayer prepare_layer(
            const Context &context, DenseLayer layer, std::size_t slots,
            bool has_activation, ActivationType activation)
        {
            PackedLayer prepared{.parameters = std::move(layer), .activation = activation, .has_activation = has_activation};
            prepared.diagonals.reserve(static_cast<std::size_t>(prepared.parameters.input_size));

            for (int diagonal = 0; diagonal < prepared.parameters.input_size; ++diagonal)
            {
                std::vector<double> values(slots, 0.0);
                for (std::size_t row = 0; row < prepared.parameters.weights.size(); ++row)
                {
                    const std::size_t column = (row + static_cast<std::size_t>(diagonal)) % slots;
                    if (column < prepared.parameters.weights[row].size())
                        values[row] = prepared.parameters.weights[row][column];
                }
                prepared.diagonals.push_back(context->MakeCKKSPackedPlaintext(values));
                context->LoadPlaintext(prepared.diagonals.back());
            }

            std::vector<double> bias(slots, 0.0);
            std::copy(prepared.parameters.biases.begin(), prepared.parameters.biases.end(), bias.begin());
            prepared.bias = context->MakeCKKSPackedPlaintext(bias);
            context->LoadPlaintext(prepared.bias);
            return prepared;
        }

        Ciphertext linear(const Context &context, const Ciphertext &input, PackedLayer &layer)
        {
            Ciphertext result;
            for (std::size_t diagonal = 0; diagonal < layer.diagonals.size(); ++diagonal)
            {
                const Ciphertext rotated = diagonal == 0
                    ? input
                    : context->EvalRotate(input, static_cast<int32_t>(diagonal));
                Ciphertext term = context->EvalMult(rotated, layer.diagonals[diagonal]);
                if (diagonal == 0) result = std::move(term);
                else context->EvalAddInPlace(result, term);
            }
            context->RescaleInPlace(result);
            context->EvalAddInPlace(result, layer.bias);
            return result;
        }

        Ciphertext activate(const Context &context, const Ciphertext &input,
                            ActivationType type, std::size_t slots)
        {
            // Small polynomial fits keep activation evaluation practical for
            // CKKS. Add a new activation here without changing model code.
            const double constant = type == ActivationType::Relu ? 0.33810450 : 0.0;
            const double quadratic_coefficient = type == ActivationType::Relu
                ? 0.096514968 : 0.3989422804014327;

            Ciphertext squared = context->EvalMult(input, input);
            context->RescaleInPlace(squared);
            Ciphertext linear_term = context->EvalMult(input, 0.5);
            context->RescaleInPlace(linear_term);
            Ciphertext quadratic_term = context->EvalMult(squared, quadratic_coefficient);
            context->RescaleInPlace(quadratic_term);
            context->EvalAddInPlace(quadratic_term, linear_term);

            Plaintext constant_term = context->MakeCKKSPackedPlaintext(std::vector<double>(slots, constant));
            context->EvalAddInPlace(quadratic_term, constant_term);
            return quadratic_term;
        }
    }

    bool cuda_available()
    {
        int device_count = 0;
        return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
    }

    struct CiphertextInferenceEngine::Implementation
    {
        Sequential model;
        CiphertextInferenceOptions options;
        Context context;
        fideslib::KeyPair<fideslib::DCRTPoly> keys;
        std::vector<PackedLayer> layers;

        Implementation(Sequential model_value, CiphertextInferenceOptions options_value)
            : model(std::move(model_value)), options(std::move(options_value))
        {
            const std::vector<DenseLayer> &dense_layers = model.layers();
            if (dense_layers.empty()) throw std::runtime_error("FIDESlib inference needs at least one dense layer");
            if (options.ring_dimension < 1024 || (options.ring_dimension & (options.ring_dimension - 1)))
                throw std::runtime_error("ring_dimension must be a power of two of at least 1024");
            if (static_cast<std::size_t>(dense_layers.front().input_size) > slot_count())
                throw std::runtime_error("model input does not fit in one CKKS ciphertext");

            require_cuda_device(options.device);
            fideslib::CCParams<fideslib::CryptoContextCKKSRNS> parameters;
            parameters.SetRingDim(options.ring_dimension);
            parameters.SetBatchSize(options.ring_dimension / 2);
            parameters.SetMultiplicativeDepth(options.multiplicative_depth);
            parameters.SetScalingModSize(options.scaling_modulus_bits);
            parameters.SetFirstModSize(options.first_modulus_bits);
            parameters.SetSecurityLevel(fideslib::HEStd_NotSet);
            parameters.SetScalingTechnique(fideslib::FLEXIBLEAUTO);
            parameters.SetKeySwitchTechnique(fideslib::HYBRID);
            parameters.SetDevices(std::vector<int>{options.device});
            parameters.SetPlaintextAutoload(false);
            parameters.SetCiphertextAutoload(true);
            context = fideslib::GenCryptoContext(parameters);
            context->Enable(fideslib::PKE);
            context->Enable(fideslib::KEYSWITCH);
            context->Enable(fideslib::LEVELEDSHE);

            keys = context->KeyGen();
            context->EvalMultKeyGen(keys.secretKey);
            context->EvalRotateKeyGen(keys.secretKey, rotation_steps(dense_layers));
            context->LoadContext(keys.publicKey);

            for (std::size_t index = 0; index < dense_layers.size(); ++index)
                layers.push_back(prepare_layer(
                    context, dense_layers[index], slot_count(),
                    model.has_activation(index),
                    model.has_activation(index) ? model.activation(index) : ActivationType::Relu));
        }

        std::size_t slot_count() const { return options.ring_dimension / 2; }

        std::vector<double> predict(const std::vector<double> &input)
        {
            if (input.size() != static_cast<std::size_t>(layers.front().parameters.input_size))
                throw std::runtime_error("input size does not match the model");

            Plaintext plain = context->MakeCKKSPackedPlaintext(input);
            Ciphertext encrypted = context->Encrypt(keys.secretKey, plain);
            for (PackedLayer &layer : layers)
            {
                encrypted = linear(context, encrypted, layer);
                if (layer.has_activation)
                    encrypted = activate(context, encrypted, layer.activation, slot_count());
            }

            Plaintext decoded;
            context->Decrypt(encrypted, keys.secretKey, &decoded);
            std::vector<double> output = decoded->GetRealPackedValue();
            output.resize(static_cast<std::size_t>(layers.back().parameters.output_size));
            return output;
        }
    };

    CiphertextInferenceEngine::CiphertextInferenceEngine(Sequential model, CiphertextInferenceOptions options)
        : implementation_(std::make_unique<Implementation>(std::move(model), std::move(options))) {}
    CiphertextInferenceEngine::~CiphertextInferenceEngine() = default;
    CiphertextInferenceEngine::CiphertextInferenceEngine(CiphertextInferenceEngine &&) noexcept = default;
    CiphertextInferenceEngine &CiphertextInferenceEngine::operator=(CiphertextInferenceEngine &&) noexcept = default;
    std::vector<double> CiphertextInferenceEngine::predict(const std::vector<double> &input) { return implementation_->predict(input); }
    std::size_t CiphertextInferenceEngine::slot_count() const { return implementation_->slot_count(); }
}
