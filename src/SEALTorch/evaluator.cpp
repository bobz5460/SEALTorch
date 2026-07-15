#include "evaluator.h"

#include "math.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace
{

    seal::Ciphertext evaluate_neuron(
        const std::vector<double> &weights,
        const std::vector<seal::Ciphertext> &input,
        double bias,
        const seal::Evaluator &evaluator,
        seal::CKKSEncoder &encoder,
        double scale)
    {
        std::vector<seal::Plaintext> encoded_weights(weights.size());
        for (std::size_t index = 0; index < weights.size(); ++index)
        {
            encoder.encode(weights[index], scale, encoded_weights[index]);
        }

        auto result = sealtorch::encrypted_dot_product(evaluator, encoded_weights, input);
        evaluator.rescale_to_next_inplace(result);

        seal::Plaintext encoded_bias;
        encoder.encode(bias, scale, encoded_bias);
        evaluator.mod_switch_to_inplace(encoded_bias, result.parms_id());
        evaluator.add_plain_inplace(result, encoded_bias);
        return result;
    }
}

namespace sealtorch
{
    Evaluator::Evaluator(SklearnMLPModel model)
    {
        set_model(std::move(model));
    }

    void Evaluator::set_model(SklearnMLPModel model)
    {
        validate_model(model);
        model_ = std::move(model);
    }

    const SklearnMLPModel &Evaluator::model() const noexcept
    {
        return model_;
    }

    bool Evaluator::is_classifier() const noexcept
    {
        return model_.estimator_type.find("MLPClassifier") != std::string::npos;
    }

    bool Evaluator::is_regressor() const noexcept
    {
        return model_.estimator_type.find("MLPRegressor") != std::string::npos;
    }

    std::vector<seal::Ciphertext> Evaluator::predict(
        const std::vector<seal::Ciphertext> &input,
        const seal::Evaluator &evaluator,
        const seal::RelinKeys &relin_keys,
        seal::CKKSEncoder &encoder,
        double scale) const
    {
        std::vector<seal::Ciphertext> values = input;
        for (std::size_t layer = 0; layer < model_.weights.size(); ++layer)
        {
            const auto output_width = model_.layer_sizes[layer + 1];
            std::vector<seal::Ciphertext> next;
            next.reserve(output_width);
            for (std::size_t output = 0; output < output_width; ++output)
            {
                std::vector<double> neuron_weights;
                neuron_weights.reserve(values.size());
                for (std::size_t input_index = 0; input_index < values.size(); ++input_index)
                {
                    neuron_weights.push_back(model_.weights[layer][input_index * output_width + output]);
                }
                next.push_back(evaluate_neuron(
                    neuron_weights, values, model_.biases[layer][output], evaluator, encoder, scale));
            }

            // Keep final-layer values as logits. Softmax and argmax happen
            // after decryption on the client.
            if (layer + 1 != model_.weights.size())
            {
                for (auto &ciphertext : next)
                {
                    ciphertext = approximate_gelu(evaluator, relin_keys, encoder, ciphertext, scale);
                }
            }
            values = std::move(next);
        }
        return values;
    }

    void Evaluator::validate_model(const SklearnMLPModel &model)
    {
        if (model.estimator_type.empty() || model.activation.empty() || model.output_activation.empty())
        {
            throw std::invalid_argument("MLP model is missing estimator or activation metadata");
        }
        if (model.layer_sizes.size() < 2 || model.input_features == 0 || model.output_features == 0 ||
            model.layer_sizes.front() != model.input_features || model.layer_sizes.back() != model.output_features)
        {
            throw std::invalid_argument("MLP model has inconsistent layer metadata");
        }
        if (model.weights.size() != model.layer_sizes.size() - 1 || model.biases.size() != model.weights.size())
        {
            throw std::invalid_argument("MLP model has inconsistent weight or bias layer counts");
        }
        for (std::size_t layer = 0; layer < model.weights.size(); ++layer)
        {
            const auto expected_weights = checked_product(model.layer_sizes[layer], model.layer_sizes[layer + 1]);
            if (model.weights[layer].size() != expected_weights ||
                model.biases[layer].size() != model.layer_sizes[layer + 1])
            {
                throw std::invalid_argument("MLP model has invalid weight or bias dimensions");
            }
        }
    }
}
