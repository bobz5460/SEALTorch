#pragma once

#include <SEALTorch/evaluator.h>

#include <cstddef>
#include <utility>
#include <vector>

namespace sealtorch
{
    class SealCiphertextBackend;

    // A tensor represented by Microsoft SEAL ciphertexts.  The provider name
    // is intentional: CUDA/FIDES tensors are not interchangeable with these.
    struct SealCiphertextTensor
    {
        std::vector<seal::Ciphertext> values;
        std::vector<std::size_t> shape;

        SealCiphertextTensor() = default;
        SealCiphertextTensor(
            std::vector<seal::Ciphertext> values_value,
            std::vector<std::size_t> shape_value)
            : values(std::move(values_value)), shape(std::move(shape_value))
        {
        }
    };

    // Dependencies required to evaluate a model with Microsoft SEAL.
    struct SealInferenceConfig
    {
        std::size_t thread_count = 4;
        double scale = 33554432.0;
        const SealCiphertextBackend *backend = nullptr;
        const seal::SEALContext *context = nullptr;
        const seal::Evaluator *evaluator = nullptr;
        const seal::RelinKeys *relin_keys = nullptr;
        const seal::GaloisKeys *galois_keys = nullptr;
        seal::CKKSEncoder *encoder = nullptr;

        SealInferenceConfig() = default;

        SealInferenceConfig(
            const seal::SEALContext &context_value,
            const seal::Evaluator &evaluator_value,
            const seal::RelinKeys &relin_keys_value,
            const seal::GaloisKeys &galois_keys_value,
            seal::CKKSEncoder &encoder_value,
            double scale_value,
            const SealCiphertextBackend &backend_value,
            std::size_t thread_count_value = 4)
            : thread_count(thread_count_value),
              scale(scale_value),
              backend(&backend_value),
              context(&context_value),
              evaluator(&evaluator_value),
              relin_keys(&relin_keys_value),
              galois_keys(&galois_keys_value),
              encoder(&encoder_value)
        {
        }
    };

    class SealCiphertextBackend
    {
    public:
        virtual ~SealCiphertextBackend() = default;

        virtual std::vector<seal::Ciphertext> run(
            const Sequential &model,
            Evaluator &evaluator,
            const std::vector<seal::Ciphertext> &input,
            const SealInferenceConfig &config) const;

        // Backends can add operations without changing the model API.
        virtual SealCiphertextTensor linear(
            const SealCiphertextTensor &input,
            const DenseLayer &layer,
            std::size_t layer_index,
            Evaluator &evaluator,
            const SealInferenceConfig &config) const;

        virtual SealCiphertextTensor activation(
            const SealCiphertextTensor &input,
            ActivationType type,
            Evaluator &evaluator,
            const SealInferenceConfig &config) const;

        virtual SealCiphertextTensor convolution2d(
            const SealCiphertextTensor &input,
            const Convolution2D &layer,
            Evaluator &evaluator,
            const SealInferenceConfig &config) const;

        virtual SealCiphertextTensor pool2d(
            const SealCiphertextTensor &input,
            const Pooling2D &layer,
            Evaluator &evaluator,
            const SealInferenceConfig &config) const;
    };

    class SealScalarBackend : public SealCiphertextBackend
    {
    public:
        SealCiphertextTensor linear(
            const SealCiphertextTensor &input, const DenseLayer &layer, std::size_t layer_index,
            Evaluator &evaluator, const SealInferenceConfig &config) const override;
        SealCiphertextTensor activation(
            const SealCiphertextTensor &input, ActivationType type,
            Evaluator &evaluator, const SealInferenceConfig &config) const override;
    };

    class SealPackedBackend : public SealCiphertextBackend
    {
    public:
        SealCiphertextTensor linear(
            const SealCiphertextTensor &input, const DenseLayer &layer, std::size_t layer_index,
            Evaluator &evaluator, const SealInferenceConfig &config) const override;
        SealCiphertextTensor activation(
            const SealCiphertextTensor &input, ActivationType type,
            Evaluator &evaluator, const SealInferenceConfig &config) const override;
    };

    // A custom model type can implement this small interface and keep the
    // same ciphertext-only predict call as the built-in sequential model.
    class SealCiphertextProgram
    {
    public:
        virtual ~SealCiphertextProgram() = default;
        virtual std::vector<seal::Ciphertext> predict(
            const std::vector<seal::Ciphertext> &input,
            const SealInferenceConfig &config) const = 0;
    };

    class SealTensorProgram
    {
    public:
        virtual ~SealTensorProgram() = default;
        virtual SealCiphertextTensor predict(
            const SealCiphertextTensor &input,
            const SealInferenceConfig &config) const = 0;
    };

    // The library only receives ciphertexts. The caller owns encryption,
    // decryption, keys, context, evaluator, and encoder.
    class SealCiphertextModel : public SealCiphertextProgram, public SealTensorProgram
    {
    public:
        explicit SealCiphertextModel(Sequential model);

        // Both backends use the same input and output type.
        // Packed mode expects one input ciphertext. Scalar mode returns one
        // ciphertext for each output value.
        std::vector<seal::Ciphertext> predict(
            const std::vector<seal::Ciphertext> &input,
            const SealInferenceConfig &config) const override;

        SealCiphertextTensor predict(
            const SealCiphertextTensor &input,
            const SealInferenceConfig &config) const override;

        const Sequential &model() const;
    private:
        mutable Evaluator evaluator_;
    };

    // Compatibility aliases for callers that used the original generic
    // names. New code should use the provider-specific names above.
    using CiphertextTensor = SealCiphertextTensor;
    using PredictionConfig = SealInferenceConfig;
    using CiphertextBackend = SealCiphertextBackend;
    using ScalarBackend = SealScalarBackend;
    using PackedBackend = SealPackedBackend;
    using CiphertextProgram = SealCiphertextProgram;
    using TensorProgram = SealTensorProgram;
    using CiphertextModel = SealCiphertextModel;
}
