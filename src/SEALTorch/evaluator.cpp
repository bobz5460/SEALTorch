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
        encoder.encode(bias, result.scale(), encoded_bias);
        evaluator.mod_switch_to_inplace(encoded_bias, result.parms_id());
        evaluator.add_plain_inplace(result, encoded_bias);
        return result;
    }
}

namespace sealtorch
{
    Evaluator::Evaluator(NeuralNetwork model)
    {
        set_model(std::move(model));
    }

    void Evaluator::set_model(NeuralNetwork model)
    {
        model_ = std::move(model);
    }

    const NeuralNetwork &Evaluator::model() const
    {
        return model_;
    }

    std::vector<seal::Ciphertext> Evaluator::predict(
        const std::vector<seal::Ciphertext> &input,
        const seal::Evaluator &evaluator,
        const seal::RelinKeys &relin_keys,
        seal::CKKSEncoder &encoder,
        double scale) const
    {
        std::vector<seal::Ciphertext> values = input;
        for (int layer = 0; layer < model_.layers.size(); layer++)
        {
            const auto output_width = model_.layers[layer].output_size;
            std::vector<seal::Ciphertext> next;
            next.reserve(output_width);

            for (int output = 0; output < output_width; output++)
            {
                next.push_back(evaluate_neuron(
                    model_.layers[layer].weights[output], values, model_.layers[layer].biases[output], evaluator, encoder, scale));
            }

            if (layer + 1 != model_.layers.size())
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
}
