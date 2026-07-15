#include "evaluator.h"

#include "math.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace
{

    seal::Ciphertext evaluate_neuron(
        const std::vector<double> &weights,
        const seal::Ciphertext &input,
        double bias,
        const seal::Evaluator &evaluator,
        const seal::GaloisKeys &galois_keys,
        seal::CKKSEncoder &encoder,
        double scale)
    {
        seal::Plaintext encoded_weights;
        encoder.encode(weights, scale, encoded_weights);

        auto result = sealtorch::encrypted_dot_product(
            evaluator, galois_keys, encoded_weights, input, weights.size());
        evaluator.rescale_to_next_inplace(result);

        seal::Plaintext encoded_bias;
        encoder.encode(bias, result.scale(), encoded_bias);
        evaluator.mod_switch_to_inplace(encoded_bias, result.parms_id());
        evaluator.add_plain_inplace(result, encoded_bias);
        return result;
    }

    seal::Ciphertext evaluate_scalar_neuron(
        const std::vector<double> &weights,
        const std::vector<seal::Ciphertext> &input,
        double bias,
        const seal::Evaluator &evaluator,
        seal::CKKSEncoder &encoder,
        double scale)
    {

        seal::Ciphertext result;
        for (std::size_t index = 0; index < weights.size(); ++index)
        {
            seal::Plaintext encoded_weight;
            encoder.encode(weights[index], scale, encoded_weight);
            // Each activation has already been rescaled. Align the encoded
            // plaintext with that ciphertext before multiply_plain; otherwise
            // SEAL rejects the NTT parameters as belonging to different
            // modulus levels.
            evaluator.mod_switch_to_inplace(encoded_weight, input[index].parms_id());

            seal::Ciphertext term;
            evaluator.multiply_plain(input[index], encoded_weight, term);

            if (index == 0)
            {
                result = std::move(term);
            }
            else
            {
                evaluator.add_inplace(result, term);
            }
        }

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
        const seal::GaloisKeys &galois_keys,
        seal::CKKSEncoder &encoder,
        double scale,
        const ProgressCallback &progress) const
    {
        std::vector<seal::Ciphertext> values;
        std::size_t total = 0;
        for (const auto &layer : model_.layers) total += static_cast<std::size_t>(layer.output_size);
        std::size_t completed = 0;
        if (progress) progress({"starting encrypted network", 0, model_.layers.size(), 0, total, 0, 0});

        for (int layer = 0; layer < model_.layers.size(); layer++)
        {
            const auto output_width = model_.layers[layer].output_size;
            std::vector<seal::Ciphertext> next;
            next.reserve(output_width);

            for (int output = 0; output < output_width; output++)
            {
                if (layer == 0)
                {
                    next.push_back(evaluate_neuron(
                        model_.layers[layer].weights[output], input.front(),
                        model_.layers[layer].biases[output], evaluator,
                        galois_keys, encoder, scale));
                }
                else
                {
                    next.push_back(evaluate_scalar_neuron(
                        model_.layers[layer].weights[output], values,
                        model_.layers[layer].biases[output], evaluator,
                        encoder, scale));
                }

                ++completed;
                if (progress) {
                    progress({"evaluating encrypted layer", static_cast<std::size_t>(layer + 1), model_.layers.size(),
                              completed, total, static_cast<std::size_t>(output + 1),
                              static_cast<std::size_t>(output_width)});
                }
            }

            if (layer + 1 != model_.layers.size())
            {
                if (progress) progress({"applying encrypted activation", static_cast<std::size_t>(layer + 1),
                                         model_.layers.size(), completed, total, 0,
                                         static_cast<std::size_t>(next.size())});
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
