#include "evaluator.h"

#include "math.h"
#include "packing.h"

#include <stdexcept>
#include <algorithm>
#include <map>
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
        if (cached_weights_[layer_index].groups.empty() ||
            cached_parms_[layer_index] != input.parms_id())
        {
            const std::size_t slot_count = encoder.slot_count();
            const std::vector<std::size_t> diagonals =
                active_diagonals(layer, slot_count);
            EncodedMatrix encoded_matrix;
            std::map<std::size_t, std::size_t> baby_indices;
            std::map<std::size_t, std::size_t> group_indices;

            for (std::size_t diagonal : diagonals)
            {
                const DiagonalSplit split =
                    split_diagonal(diagonal, slot_count);

                if (baby_indices.find(split.baby) == baby_indices.end())
                {
                    const std::size_t index =
                        encoded_matrix.input_rotations.size();
                    baby_indices[split.baby] = index;
                    encoded_matrix.input_rotations.push_back(
                        signed_rotation(split.baby, slot_count));
                }

                if (group_indices.find(split.giant) == group_indices.end())
                {
                    const std::size_t index = encoded_matrix.groups.size();
                    group_indices[split.giant] = index;
                    EncodedDiagonalGroup group;
                    group.rotation =
                        signed_rotation(split.giant, slot_count);
                    encoded_matrix.groups.push_back(std::move(group));
                }

                std::vector<double> values(slot_count, 0.0);
                for (std::size_t row = 0; row < output_width; ++row)
                {
                    const std::size_t column = (row + diagonal) % slot_count;
                    if (column < input_width)
                    {
                        const std::size_t shifted_row =
                            (row + split.giant) % slot_count;
                        values[shifted_row] = layer.weights[row][column];
                    }
                }

                EncodedDiagonal encoded;
                encoded.input_index = baby_indices.at(split.baby);
                encoder.encode(values, scale, encoded.weights);
                evaluator.mod_switch_to_inplace(
                    encoded.weights, input.parms_id());
                encoded_matrix.groups[group_indices.at(split.giant)]
                    .diagonals.push_back(std::move(encoded));
            }
            cached_weights_[layer_index] = std::move(encoded_matrix);
            cached_parms_[layer_index] = input.parms_id();
        }

        seal::Ciphertext result = encrypted_matrix_vector_product(
            context, evaluator, galois_keys, input,
            cached_weights_[layer_index], thread_count, thread_pool_);
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
        const seal::RelinKeys &relin_keys,
        double scale,
        std::size_t thread_count) const
    {
        std::vector<seal::Ciphertext> output(input.size());
        if (input.empty()) return output;
        thread_pool_.parallel_for_workers(input.size(), thread_count, [&](std::size_t worker, std::size_t jobs) {
                seal::Evaluator local_evaluator(context);
                seal::CKKSEncoder local_encoder(context);
                for (std::size_t index = worker; index < input.size(); index += jobs)
                    output[index] = type == ActivationType::Relu
                        ? approximate_relu(local_evaluator, relin_keys, local_encoder, input[index], scale)
                        : type == ActivationType::Gelu
                            ? approximate_gelu(local_evaluator, relin_keys, local_encoder, input[index], scale)
                            : approximate_tanh(local_evaluator, relin_keys, local_encoder, input[index], scale);
            });
        return output;
    }
}
