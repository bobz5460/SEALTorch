#pragma once

#include <SEALTorch/evaluator.h>

#include <cstddef>
#include <utility>
#include <vector>

namespace sealtorch
{
    class CiphertextBackend;

    struct CiphertextTensor
    {
        std::vector<seal::Ciphertext> values;
        std::vector<std::size_t> shape;

        CiphertextTensor() = default;
        CiphertextTensor(
            std::vector<seal::Ciphertext> values_value,
            std::vector<std::size_t> shape_value)
            : values(std::move(values_value)), shape(std::move(shape_value))
        {
        }
    };

    struct Convolution2D
    {
        std::vector<double> weights;
        std::vector<double> biases;
        std::size_t input_channels = 0;
        std::size_t output_channels = 0;
        std::size_t kernel_height = 0;
        std::size_t kernel_width = 0;
        std::size_t stride = 1;
        std::size_t padding = 0;
    };

    enum class PoolingType
    {
        Average,
        Maximum
    };

    struct Pooling2D
    {
        PoolingType type = PoolingType::Average;
        std::size_t height = 2;
        std::size_t width = 2;
        std::size_t stride = 2;
    };

    struct PredictionConfig
    {
        std::size_t thread_count = 4;
        double scale = 33554432.0;
        const CiphertextBackend *backend = nullptr;
        const seal::SEALContext *context = nullptr;
        const seal::Evaluator *evaluator = nullptr;
        const seal::RelinKeys *relin_keys = nullptr;
        const seal::GaloisKeys *galois_keys = nullptr;
        seal::CKKSEncoder *encoder = nullptr;

        PredictionConfig() = default;

        PredictionConfig(
            const seal::SEALContext &context_value,
            const seal::Evaluator &evaluator_value,
            const seal::RelinKeys &relin_keys_value,
            const seal::GaloisKeys &galois_keys_value,
            seal::CKKSEncoder &encoder_value,
            double scale_value,
            const CiphertextBackend &backend_value,
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

    class CiphertextBackend
    {
    public:
        virtual ~CiphertextBackend() = default;

        virtual std::vector<seal::Ciphertext> run(
            const Sequential &model,
            Evaluator &evaluator,
            const std::vector<seal::Ciphertext> &input,
            const PredictionConfig &config) const = 0;

        // Backends can add operations without changing the model API.
        virtual CiphertextTensor linear(
            const CiphertextTensor &input,
            const DenseLayer &layer,
            const PredictionConfig &config) const;

        virtual CiphertextTensor activation(
            const CiphertextTensor &input,
            ActivationType type,
            const PredictionConfig &config) const;

        virtual CiphertextTensor convolution2d(
            const CiphertextTensor &input,
            const Convolution2D &layer,
            const PredictionConfig &config) const;

        virtual CiphertextTensor pool2d(
            const CiphertextTensor &input,
            const Pooling2D &layer,
            const PredictionConfig &config) const;
    };

    class ScalarBackend : public CiphertextBackend
    {
    public:
        std::vector<seal::Ciphertext> run(
            const Sequential &model,
            Evaluator &evaluator,
            const std::vector<seal::Ciphertext> &input,
            const PredictionConfig &config) const override;
    };

    class PackedBackend : public CiphertextBackend
    {
    public:
        std::vector<seal::Ciphertext> run(
            const Sequential &model,
            Evaluator &evaluator,
            const std::vector<seal::Ciphertext> &input,
            const PredictionConfig &config) const override;
    };

    // A custom model type can implement this small interface and keep the
    // same ciphertext-only predict call as the built-in sequential model.
    class CiphertextProgram
    {
    public:
        virtual ~CiphertextProgram() = default;
        virtual std::vector<seal::Ciphertext> predict(
            const std::vector<seal::Ciphertext> &input,
            const PredictionConfig &config) const = 0;
    };

    class TensorProgram
    {
    public:
        virtual ~TensorProgram() = default;
        virtual CiphertextTensor predict(
            const CiphertextTensor &input,
            const PredictionConfig &config) const = 0;
    };

    // The library only receives ciphertexts. The caller owns encryption,
    // decryption, keys, context, evaluator, and encoder.
    class CiphertextModel : public CiphertextProgram, public TensorProgram
    {
    public:
        explicit CiphertextModel(Sequential model);

        // Both backends use the same input and output type.
        // Packed mode expects one input ciphertext. Scalar mode returns one
        // ciphertext for each output value.
        std::vector<seal::Ciphertext> predict(
            const std::vector<seal::Ciphertext> &input,
            const PredictionConfig &config) const override;

        CiphertextTensor predict(
            const CiphertextTensor &input,
            const PredictionConfig &config) const override;

        const Sequential &model() const;
    private:
        mutable Evaluator evaluator_;
    };
}
