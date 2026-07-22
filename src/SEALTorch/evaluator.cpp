#include "evaluator.h"

#include "math.h"

#include <utility>

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

    seal::Ciphertext Evaluator::predict(
        const seal::Ciphertext &input,
        const seal::Evaluator &evaluator,
        const seal::RelinKeys &relin_keys,
        const seal::GaloisKeys &galois_keys,
        seal::CKKSEncoder &encoder,
        double scale) const
    {
        seal::Ciphertext values = input;
        std::size_t input_width = model_.input_size;

        for (std::size_t layer = 0; layer < model_.layers.size(); ++layer)
        {
            const auto &current = model_.layers[layer];
            std::size_t output_width = current.output_size;

            seal::Ciphertext next = encrypted_matrix_vector_product(
                evaluator, galois_keys, encoder, values,
                current.weights, input_width, output_width, scale);

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
