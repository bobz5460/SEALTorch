#include "evaluator.h"

#include "math.h"

#include <stdexcept>
#include <algorithm>
#include <thread>
#include <utility>

namespace
{
    seal::Ciphertext scalar_neuron(
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
            seal::Plaintext weight;
            encoder.encode(weights[index], scale, weight);
            evaluator.mod_switch_to_inplace(weight, input[index].parms_id());
            seal::Ciphertext term;
            evaluator.multiply_plain(input[index], weight, term);
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

    seal::Ciphertext packed_neuron(
        const sealtorch::DenseLayer &layer,
        const std::vector<seal::Ciphertext> &input,
        std::size_t output,
        const seal::Evaluator &evaluator,
        const seal::GaloisKeys &galois_keys,
        seal::CKKSEncoder &encoder,
        double scale)
    {
        seal::Plaintext weights;
        encoder.encode(layer.weights[output], scale, weights);
        seal::Ciphertext result = sealtorch::encrypted_dot_product(
            evaluator, galois_keys, weights, input.front(), layer.input_size);
        evaluator.rescale_to_next_inplace(result);
        seal::Plaintext bias;
        encoder.encode(layer.biases[output], result.scale(), bias);
        evaluator.mod_switch_to_inplace(bias, result.parms_id());
        evaluator.add_plain_inplace(result, bias);
        return result;
    }
}

namespace sealtorch
{
    Evaluator::Evaluator(Sequential model) : model_(std::move(model)) {}

    const Sequential &Evaluator::model() const { return model_; }

    std::vector<seal::Ciphertext> Evaluator::linear_scalar(
        const std::vector<seal::Ciphertext> &input,
        const DenseLayer &layer,
        const seal::Evaluator &evaluator,
        const seal::GaloisKeys &galois_keys,
        seal::CKKSEncoder &encoder,
        double scale) const
    {
        if (input.size() == 1 && layer.input_size > 1)
        {
            std::vector<seal::Ciphertext> output;
            for (std::size_t row = 0; row < layer.weights.size(); ++row)
                output.push_back(packed_neuron(layer, input, row, evaluator, galois_keys, encoder, scale));
            return output;
        }

        std::vector<seal::Ciphertext> output;
        for (std::size_t row = 0; row < layer.weights.size(); ++row)
            output.push_back(scalar_neuron(
                layer.weights[row], input, layer.biases[row], evaluator, encoder, scale));
        return output;
    }

    seal::Ciphertext Evaluator::linear_packed(
        const seal::SEALContext &context,
        const seal::Ciphertext &input,
        const DenseLayer &layer,
        std::size_t layer_index,
        const seal::Evaluator &evaluator,
        const seal::GaloisKeys &galois_keys,
        seal::CKKSEncoder &encoder,
        double scale,
        std::size_t thread_count) const
    {
        if (cached_weights_.size() <= layer_index)
        {
            cached_weights_.resize(layer_index + 1);
            cached_parms_.resize(layer_index + 1);
        }

        const std::size_t input_width = layer.input_size;
        const std::size_t output_width = layer.output_size;
        if (cached_weights_[layer_index].empty() ||
            cached_parms_[layer_index] != input.parms_id())
        {
            const std::size_t size = encoder.slot_count();
            cached_weights_[layer_index].resize(size);
            for (std::size_t diagonal = 0; diagonal < size; ++diagonal)
            {
                std::vector<double> values(size, 0.0);
                bool used = false;
                for (std::size_t row = 0; row < output_width; ++row)
                {
                    const std::size_t column = (row + diagonal) % size;
                    if (column < input_width)
                    {
                        values[row] = layer.weights[row][column];
                        used = true;
                    }
                }
                if (used)
                {
                    encoder.encode(values, scale, cached_weights_[layer_index][diagonal]);
                    evaluator.mod_switch_to_inplace(
                        cached_weights_[layer_index][diagonal], input.parms_id());
                }
            }
            cached_parms_[layer_index] = input.parms_id();
        }

        seal::Ciphertext result = encrypted_matrix_vector_product(
            context, evaluator, galois_keys, encoder, input,
            layer.weights, input_width, output_width, scale, thread_count,
            &cached_weights_[layer_index]);
        evaluator.rescale_to_next_inplace(result);
        seal::Plaintext bias;
        encoder.encode(layer.biases, result.scale(), bias);
        evaluator.mod_switch_to_inplace(bias, result.parms_id());
        evaluator.add_plain_inplace(result, bias);
        return result;
    }

    std::vector<seal::Ciphertext> Evaluator::activation(
        const std::vector<seal::Ciphertext> &input,
        ActivationType type,
        const seal::SEALContext &context,
        const seal::Evaluator &evaluator,
        const seal::RelinKeys &relin_keys,
        seal::CKKSEncoder &encoder,
        double scale,
        std::size_t thread_count) const
    {
        (void)type;
        std::vector<seal::Ciphertext> output(input.size());
        if (input.empty()) return output;
        thread_count = std::max<std::size_t>(1, std::min(thread_count, input.size()));
        std::vector<std::thread> workers;
        for (std::size_t worker = 0; worker < thread_count; ++worker)
        {
            workers.emplace_back([&, worker]() {
                seal::Evaluator local_evaluator(context);
                seal::CKKSEncoder local_encoder(context);
                for (std::size_t index = worker; index < input.size(); index += thread_count)
                    output[index] = approximate_gelu(
                        local_evaluator, relin_keys, local_encoder, input[index], scale);
            });
        }
        for (std::thread &worker : workers) worker.join();
        return output;
    }
}
