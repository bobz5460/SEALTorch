#include "evaluator.h"

#include "math.h"

#include <algorithm>
#include <future>
#include <stdexcept>
#include <utility>

namespace
{
    template <typename Function>

    auto parallel_map(std::size_t count, std::size_t max_concurrency, Function function)
        -> std::vector<std::invoke_result_t<Function, std::size_t>>
    {
        using Result = std::invoke_result_t<Function, std::size_t>;
        std::vector<Result> results(count);
        if (count == 0) return results;

        const std::size_t workers = std::max<std::size_t>(
            1, std::min(count, max_concurrency == 0 ? 1 : max_concurrency));
        for (std::size_t start = 0; start < count; start += workers)
        {
            const std::size_t end = std::min(count, start + workers);
            std::vector<std::future<Result>> futures;
            futures.reserve(end - start);
            for (std::size_t index = start; index < end; ++index)
                futures.emplace_back(std::async(std::launch::async, function, index));
            for (std::size_t offset = 0; offset < futures.size(); ++offset)
                results[start + offset] = futures[offset].get();
        }
        return results;
    }

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
    Evaluator::Evaluator(NeuralNetwork model, std::size_t max_concurrency)
        : max_concurrency_(max_concurrency == 0 ? 1 : max_concurrency)
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

    void Evaluator::set_max_concurrency(std::size_t max_concurrency)
    {
        max_concurrency_ = max_concurrency == 0 ? 1 : max_concurrency;
    }

    std::size_t Evaluator::max_concurrency() const
    {
        return max_concurrency_;
    }

    std::vector<seal::Ciphertext> Evaluator::predict(
        const std::vector<seal::Ciphertext> &input,
        const seal::Evaluator &evaluator,
        const seal::RelinKeys &relin_keys,
        const seal::GaloisKeys &galois_keys,
        seal::CKKSEncoder &encoder,
        double scale,
        const ProgressCallback &progress,
        const LayerCallback &layer_callback) const
    {
        std::vector<seal::Ciphertext> values;
        std::size_t total = 0;
        for (const auto &layer : model_.layers)
            total += static_cast<std::size_t>(layer.output_size);
        std::size_t completed = 0;
        if (progress) progress({"starting encrypted network", 0, model_.layers.size(), 0, total, 0, 0});

        for (std::size_t layer = 0; layer < model_.layers.size(); ++layer)
        {
            const auto &current_layer = model_.layers[layer];
            const std::size_t output_width = static_cast<std::size_t>(current_layer.output_size);

            auto next = parallel_map(output_width, max_concurrency_, [&](std::size_t output) {
                if (layer == 0)
                    return evaluate_neuron(current_layer.weights[output], input.front(),
                        current_layer.biases[output], evaluator, galois_keys, encoder, scale);
                return evaluate_scalar_neuron(current_layer.weights[output], values,
                    current_layer.biases[output], evaluator, encoder, scale);
            });

            for (std::size_t output = 0; output < output_width; ++output)
            {
                ++completed;
                if (progress) {
                    progress({"evaluating encrypted layer", layer + 1, model_.layers.size(),
                              completed, total, output + 1, output_width});
                }
            }

            if (layer_callback) layer_callback(layer, false, next);
            if (layer + 1 < model_.layers.size())
            {
                if (progress) progress({"applying encrypted activation", layer + 1,
                                         model_.layers.size(), completed, total, 0, next.size()});

                next = parallel_map(next.size(), max_concurrency_, [&](std::size_t neuron) {
                    return approximate_gelu(evaluator, relin_keys, encoder, next[neuron], scale);
                });
            }
            if (layer_callback) layer_callback(layer, true, next);
            values = std::move(next);
        }
        return values;
    }
}
