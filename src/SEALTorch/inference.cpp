#include "inference.h"

#include <stdexcept>
#include <utility>

namespace sealtorch
{
    static void unsupported(const char *name)
    {
        throw std::runtime_error(std::string("backend does not implement ") + name);
    }

    CiphertextTensor CiphertextBackend::linear(
        const CiphertextTensor &, const DenseLayer &, const PredictionConfig &) const
    {
        unsupported("linear");
        return {};
    }

    CiphertextTensor CiphertextBackend::activation(
        const CiphertextTensor &, ActivationType, const PredictionConfig &) const
    {
        unsupported("activation");
        return {};
    }

    CiphertextTensor CiphertextBackend::convolution2d(
        const CiphertextTensor &, const Convolution2D &, const PredictionConfig &) const
    {
        unsupported("convolution2d");
        return {};
    }

    CiphertextTensor CiphertextBackend::pool2d(
        const CiphertextTensor &, const Pooling2D &, const PredictionConfig &) const
    {
        unsupported("pool2d");
        return {};
    }

    std::vector<seal::Ciphertext> ScalarBackend::run(
        const Sequential &model,
        Evaluator &evaluator,
        const std::vector<seal::Ciphertext> &input,
        const PredictionConfig &config) const
    {
        return evaluator.predict_scalar(
            input, *config.evaluator, *config.relin_keys,
            *config.galois_keys, *config.encoder, config.scale);
    }

    std::vector<seal::Ciphertext> PackedBackend::run(
        const Sequential &model,
        Evaluator &evaluator,
        const std::vector<seal::Ciphertext> &input,
        const PredictionConfig &config) const
    {
        if (config.context == nullptr)
            throw std::runtime_error("packed prediction needs a SEAL context");
        if (input.size() != 1)
            throw std::runtime_error("packed prediction needs one input ciphertext");
        const seal::Ciphertext output = evaluator.predict_packed(
            *config.context, input.front(), *config.evaluator,
            *config.relin_keys, *config.galois_keys, *config.encoder,
            config.scale, config.thread_count);
        return std::vector<seal::Ciphertext>(1, output);
    }

    CiphertextModel::CiphertextModel(Sequential model)
        : evaluator_(std::move(model))
    {
    }

    std::vector<seal::Ciphertext> CiphertextModel::predict(
        const std::vector<seal::Ciphertext> &input,
        const PredictionConfig &config) const
    {
        if (input.empty()) throw std::runtime_error("prediction input is empty");
        if (config.thread_count == 0) throw std::runtime_error("thread count must be greater than zero");
        if (config.scale <= 0.0) throw std::runtime_error("scale must be greater than zero");
        if (config.backend == nullptr)
            throw std::runtime_error("prediction config is missing a backend");
        if (config.evaluator == nullptr || config.relin_keys == nullptr ||
            config.galois_keys == nullptr || config.encoder == nullptr)
            throw std::runtime_error("prediction config is missing SEAL objects");
        return config.backend->run(model(), evaluator_, input, config);
    }

    const Sequential &CiphertextModel::model() const { return evaluator_.model(); }

    CiphertextTensor CiphertextModel::predict(
        const CiphertextTensor &input,
        const PredictionConfig &config) const
    {
        const std::vector<seal::Ciphertext> output = predict(input.values, config);
        return CiphertextTensor(
            output,
            std::vector<std::size_t>(1, output.size()));
    }
}
