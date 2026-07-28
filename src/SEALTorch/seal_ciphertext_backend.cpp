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

    std::vector<seal::Ciphertext> CiphertextBackend::run(
        const Sequential &model,
        Evaluator &evaluator,
        const std::vector<seal::Ciphertext> &input,
        const PredictionConfig &config) const
    {
        CiphertextTensor values(input, std::vector<std::size_t>(1, input.size()));
        std::size_t dense_index = 0;
        for (const Operation &operation : model.operations())
        {
            switch (operation.kind)
            {
            case OperationKind::Linear:
                values = linear(values, std::any_cast<const DenseLayer &>(operation.payload),
                                dense_index++, evaluator, config);
                break;
            case OperationKind::Activation:
                values = activation(values, std::any_cast<ActivationType>(operation.payload), evaluator, config);
                break;
            case OperationKind::Convolution2D:
                values = convolution2d(values, std::any_cast<const Convolution2D &>(operation.payload), evaluator, config);
                break;
            case OperationKind::Pooling2D:
                values = pool2d(values, std::any_cast<const Pooling2D &>(operation.payload), evaluator, config);
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

    CiphertextTensor CiphertextBackend::linear(
        const CiphertextTensor &, const DenseLayer &, std::size_t,
        Evaluator &, const PredictionConfig &) const
    {
        unsupported("linear");
        return {};
    }

    CiphertextTensor CiphertextBackend::activation(
        const CiphertextTensor &, ActivationType, Evaluator &,
        const PredictionConfig &) const
    {
        unsupported("activation");
        return {};
    }

    CiphertextTensor CiphertextBackend::convolution2d(
        const CiphertextTensor &, const Convolution2D &, Evaluator &,
        const PredictionConfig &) const
    {
        unsupported("convolution2d");
        return {};
    }

    CiphertextTensor CiphertextBackend::pool2d(
        const CiphertextTensor &, const Pooling2D &, Evaluator &,
        const PredictionConfig &) const
    {
        unsupported("pool2d");
        return {};
    }

    CiphertextTensor ScalarBackend::linear(
        const CiphertextTensor &input,
        const DenseLayer &layer,
        std::size_t,
        Evaluator &evaluator,
        const PredictionConfig &config) const
    {
        return CiphertextTensor(
            evaluator.linear_scalar(
                input.values, layer, *config.evaluator,
                *config.galois_keys, *config.encoder, config.scale),
            std::vector<std::size_t>(1, layer.output_size));
    }

    CiphertextTensor ScalarBackend::activation(
        const CiphertextTensor &input,
        ActivationType type,
        Evaluator &evaluator,
        const PredictionConfig &config) const
    {
        return CiphertextTensor(
            evaluator.activation(
                input.values, type, *config.context, *config.evaluator,
                *config.relin_keys, *config.encoder, config.scale,
                config.thread_count),
            input.shape);
    }

    CiphertextTensor PackedBackend::linear(
        const CiphertextTensor &input,
        const DenseLayer &layer,
        std::size_t layer_index,
        Evaluator &evaluator,
        const PredictionConfig &config) const
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
        return CiphertextTensor(
            std::move(output), std::vector<std::size_t>(1, layer.output_size));
    }

    CiphertextTensor PackedBackend::activation(
        const CiphertextTensor &input,
        ActivationType type,
        Evaluator &evaluator,
        const PredictionConfig &config) const
    {
        return CiphertextTensor(
            evaluator.activation(
                input.values, type, *config.context, *config.evaluator,
                *config.relin_keys, *config.encoder, config.scale,
                config.thread_count),
            input.shape);
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

    CiphertextTensor CiphertextModel::predict(
        const CiphertextTensor &input,
        const PredictionConfig &config) const
    {
        const std::vector<seal::Ciphertext> output = predict(input.values, config);
        return CiphertextTensor(output, std::vector<std::size_t>(1, output.size()));
    }

    const Sequential &CiphertextModel::model() const { return evaluator_.model(); }
}
