#include "evaluator.h"

#include "math.h"

#include <utility>

namespace
{
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
            if (index == 0) result = std::move(term);
            else evaluator.add_inplace(result, term);
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
    Evaluator::Evaluator(NeuralNetwork model, ExecutionMode mode) : mode_(mode)
    {
        set_model(std::move(model));
    }

    void Evaluator::set_model(NeuralNetwork model)
    {
        model_ = std::move(model);
        cached_weights_.clear();
        cached_parms_.clear();
    }

    const NeuralNetwork &Evaluator::model() const
    {
        return model_;
    }

    void Evaluator::set_mode(ExecutionMode mode) { mode_ = mode; }

    ExecutionMode Evaluator::mode() const { return mode_; }

    std::vector<seal::Ciphertext> Evaluator::predict_scalar(
        const std::vector<seal::Ciphertext> &input,
        const seal::Evaluator &evaluator,
        const seal::RelinKeys &relin_keys,
        const seal::GaloisKeys &galois_keys,
        seal::CKKSEncoder &encoder,
        double scale) const
    {
        std::vector<seal::Ciphertext> values = input;
        for (std::size_t layer = 0; layer < model_.layers.size(); ++layer)
        {
            const DenseLayer &current = model_.layers[layer];
            std::vector<seal::Ciphertext> next;
            next.reserve(current.output_size);
            for (std::size_t output = 0; output < current.output_size; ++output)
            {
                if (layer == 0)
                {
                    seal::Plaintext weights;
                    encoder.encode(current.weights[output], scale, weights);
                    seal::Ciphertext value = encrypted_dot_product(
                        evaluator, galois_keys, weights, values.front(), current.input_size);
                    evaluator.rescale_to_next_inplace(value);
                    seal::Plaintext bias;
                    encoder.encode(current.biases[output], value.scale(), bias);
                    evaluator.mod_switch_to_inplace(bias, value.parms_id());
                    evaluator.add_plain_inplace(value, bias);
                    next.push_back(std::move(value));
                }
                else
                {
                    next.push_back(evaluate_scalar_neuron(
                        current.weights[output], values, current.biases[output],
                        evaluator, encoder, scale));
                }
            }
            if (layer + 1 != model_.layers.size())
                for (seal::Ciphertext &value : next)
                    value = approximate_gelu(evaluator, relin_keys, encoder, value, scale);
            values = std::move(next);
        }
        return values;
    }

    seal::Ciphertext Evaluator::predict_packed(
        const seal::SEALContext &context,
        const seal::Ciphertext &input,
        const seal::Evaluator &evaluator,
        const seal::RelinKeys &relin_keys,
        const seal::GaloisKeys &galois_keys,
        seal::CKKSEncoder &encoder,
        double scale,
        std::size_t thread_count) const
    {
        seal::Ciphertext values = input;
        std::size_t input_width = model_.input_size;

        if (cached_weights_.size() != model_.layers.size())
        {
            cached_weights_.resize(model_.layers.size());
            cached_parms_.resize(model_.layers.size());
        }

        for (std::size_t layer = 0; layer < model_.layers.size(); ++layer)
        {
            const auto &current = model_.layers[layer];
            std::size_t output_width = current.output_size;

            if (cached_weights_[layer].empty() || cached_parms_[layer] != values.parms_id())
            {
                std::size_t size = encoder.slot_count();
                cached_weights_[layer].resize(size);

                for (std::size_t diagonal = 0; diagonal < size; ++diagonal)
                {
                    std::vector<double> diagonal_values(size, 0.0);
                    bool used = false;

                    for (std::size_t row = 0; row < output_width; ++row)
                    {
                        std::size_t column = (row + diagonal) % size;
                        if (column < input_width)
                        {
                            diagonal_values[row] = current.weights[row][column];
                            used = true;
                        }
                    }

                    if (used)
                    {
                        encoder.encode(diagonal_values, scale, cached_weights_[layer][diagonal]);
                        evaluator.mod_switch_to_inplace(cached_weights_[layer][diagonal], values.parms_id());
                    }
                }
                cached_parms_[layer] = values.parms_id();
            }

            seal::Ciphertext next = encrypted_matrix_vector_product(
                context, evaluator, galois_keys, encoder, values,
                current.weights, input_width, output_width, scale, thread_count,
                &cached_weights_[layer]);

            evaluator.rescale_to_next_inplace(next);

            seal::Plaintext bias;
            encoder.encode(current.biases, next.scale(), bias);
            evaluator.mod_switch_to_inplace(bias, next.parms_id());
            evaluator.add_plain_inplace(next, bias);

            if (layer + 1 != model_.layers.size())
            {
                next = approximate_gelu(
                    evaluator, relin_keys, encoder, next, scale);
            }

            values = std::move(next);
            input_width = output_width;
        }
        return values;
    }
}
