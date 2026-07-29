#include "seal_ciphertext_backend.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace sealtorch
{
    static void unsupported(const char *name)
    {
        throw std::runtime_error(std::string("backend does not implement ") + name);
    }

    std::vector<seal::Ciphertext> SealCiphertextBackend::run(
        const Sequential &model,
        Evaluator &evaluator,
        const std::vector<seal::Ciphertext> &input,
        const SealInferenceConfig &config) const
    {
        SealCiphertextTensor values(input, std::vector<std::size_t>(1, input.size()));
        std::size_t dense_index = 0;
        for (const Operation &operation : model.operations())
        {
            switch (operation.kind)
            {
            case OperationKind::Linear:
                values = linear(values, operation.linear_layer,
                                dense_index++, evaluator, config);
                break;
            case OperationKind::Activation:
                values = activation(values, operation.activation_type, evaluator, config);
                break;
            case OperationKind::Convolution2D:
                values = convolution2d(values, operation.convolution_layer, evaluator, config);
                break;
            case OperationKind::Pooling2D:
                values = pool2d(values, operation.pooling_layer, evaluator, config);
                break;
            case OperationKind::Flatten:
                // Flatten changes interpretation, not ciphertext contents.
                values.shape = operation.output_shape.empty() ?
                    std::vector<std::size_t>{values.values.size()} : operation.output_shape;
                break;
            case OperationKind::Custom:
                throw std::runtime_error("backend does not implement custom operation: " + operation.name);
            }
        }
        return values.values;
    }

    SealCiphertextTensor SealCiphertextBackend::linear(
        const SealCiphertextTensor &, const DenseLayer &, std::size_t,
        Evaluator &, const SealInferenceConfig &) const
    {
        unsupported("linear");
        return {};
    }

    SealCiphertextTensor SealCiphertextBackend::activation(
        const SealCiphertextTensor &, ActivationType, Evaluator &,
        const SealInferenceConfig &) const
    {
        unsupported("activation");
        return {};
    }

    SealCiphertextTensor SealCiphertextBackend::convolution2d(
        const SealCiphertextTensor &, const Convolution2D &, Evaluator &,
        const SealInferenceConfig &) const
    {
        unsupported("convolution2d");
        return {};
    }

    SealCiphertextTensor SealCiphertextBackend::pool2d(
        const SealCiphertextTensor &, const Pooling2D &, Evaluator &,
        const SealInferenceConfig &) const
    {
        unsupported("pool2d");
        return {};
    }

    SealCiphertextTensor SealScalarBackend::linear(
        const SealCiphertextTensor &input,
        const DenseLayer &layer,
        std::size_t,
        Evaluator &evaluator,
        const SealInferenceConfig &config) const
    {
        return SealCiphertextTensor(
            evaluator.linear_scalar(
                input.values, layer, *config.evaluator,
                *config.galois_keys, *config.encoder, config.scale),
            std::vector<std::size_t>(1, layer.output_size));
    }

    SealCiphertextTensor SealScalarBackend::activation(
        const SealCiphertextTensor &input,
        ActivationType type,
        Evaluator &evaluator,
        const SealInferenceConfig &config) const
    {
        return SealCiphertextTensor(
            evaluator.activation(
                input.values, type, *config.context, *config.evaluator,
                *config.relin_keys, *config.encoder, config.scale,
                config.thread_count),
            input.shape);
    }

    SealCiphertextTensor SealPackedBackend::linear(
        const SealCiphertextTensor &input,
        const DenseLayer &layer,
        std::size_t layer_index,
        Evaluator &evaluator,
        const SealInferenceConfig &config) const
    {
        if (config.context == nullptr)
            throw std::runtime_error("packed prediction needs a SEAL context");
        if (input.values.size() != 1)
            throw std::runtime_error("packed prediction needs one input ciphertext");

        std::vector<seal::Ciphertext> output(1);
        output[0] = evaluator.linear_packed(
            *config.context, input.values.front(), layer, layer_index,
            *config.evaluator, *config.galois_keys, *config.encoder,
            config.scale, config.thread_count);
        return SealCiphertextTensor(
            std::move(output), std::vector<std::size_t>(1, layer.output_size));
    }

    SealCiphertextTensor SealPackedBackend::activation(
        const SealCiphertextTensor &input,
        ActivationType type,
        Evaluator &evaluator,
        const SealInferenceConfig &config) const
    {
        return SealCiphertextTensor(
            evaluator.activation(
                input.values, type, *config.context, *config.evaluator,
                *config.relin_keys, *config.encoder, config.scale,
                config.thread_count),
            input.shape);
    }

    SealCiphertextModel::SealCiphertextModel(Sequential model)
        : evaluator_(std::move(model))
    {
    }

    std::vector<seal::Ciphertext> SealCiphertextModel::predict(
        const std::vector<seal::Ciphertext> &input,
        const SealInferenceConfig &config) const
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

    SealCiphertextTensor SealCiphertextModel::predict(
        const SealCiphertextTensor &input,
        const SealInferenceConfig &config) const
    {
        const std::vector<seal::Ciphertext> output = predict(input.values, config);
        return SealCiphertextTensor(output, std::vector<std::size_t>(1, output.size()));
    }

    const Sequential &SealCiphertextModel::model() const { return evaluator_.model(); }
}
