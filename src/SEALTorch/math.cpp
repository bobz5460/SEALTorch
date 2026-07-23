#include "math.h"
#include <cmath>
#include <stdexcept>
#include <thread>
#include <algorithm>

namespace sealtorch
{
    seal::Ciphertext encrypted_matrix_vector_product(
        const seal::SEALContext& context,
        const seal::Evaluator& evaluator,
        const seal::GaloisKeys& galois_keys,
        seal::CKKSEncoder& encoder,
        const seal::Ciphertext& input,
        const std::vector<std::vector<double>>& weights,
        std::size_t input_width,
        std::size_t output_width,
        double scale,
        std::size_t thread_count,
        const std::vector<seal::Plaintext> *cached_weights)
    {
        // Use the actual number of CKKS slots for the wraparound. This is
        // important when the layer is smaller than the ciphertext.
        std::size_t size = encoder.slot_count();

        // Only these rotations can contain a real input/output pair.
        // This avoids rotating through every CKKS slot.
        std::vector<bool> used(size, false);
        for (std::size_t row = 0; row < output_width; ++row)
        {
            for (std::size_t column = 0; column < input_width; ++column)
            {
                used[(column + size - row) % size] = true;
            }
        }

        // Each diagonal is independent. Compute the terms in parallel, then
        // add them together below in one thread.
        std::vector<seal::Ciphertext> terms(size);
        std::size_t diagonal_count = 0;
        for (bool value : used)
            if (value) ++diagonal_count;

        if (thread_count == 0) thread_count = 1;
        thread_count = std::min(thread_count, diagonal_count);

        std::vector<std::thread> workers;
        for (std::size_t worker = 0; worker < thread_count; ++worker)
        {
            workers.emplace_back([&, worker]() {
                seal::Evaluator local_evaluator(context);
                seal::CKKSEncoder local_encoder(context);
                seal::MemoryPoolHandle pool = seal::MemoryPoolHandle::ThreadLocal();

                std::size_t current = worker;
                while (current < size)
                {
                    if (used[current])
                    {
                        seal::Plaintext encoded;
                        const seal::Plaintext *encoded_value = &encoded;
                        if (cached_weights)
                        {
                            encoded_value = &(*cached_weights)[current];
                        }
                        else
                        {
                            std::vector<double> values(size, 0.0);
                            for (std::size_t row = 0; row < output_width; ++row)
                            {
                                std::size_t column = (row + current) % size;
                                if (column < input_width)
                                    values[row] = weights[row][column];
                            }
                            local_encoder.encode(values, scale, encoded, pool);
                        }

                        seal::Ciphertext rotated;
                        if (current == 0)
                        {
                            rotated = input;
                        }
                        else
                        {
                            local_evaluator.rotate_vector(
                                input,
                                static_cast<int>(current),
                                galois_keys,
                                rotated,
                                pool);
                        }

                        if (!cached_weights)
                            local_evaluator.mod_switch_to_inplace(encoded, rotated.parms_id());

                        local_evaluator.multiply_plain(rotated, *encoded_value, terms[current], pool);
                    }
                    current += thread_count;
                }
            });
        }

        for (auto& worker : workers) worker.join();

        seal::Ciphertext result;
        bool first = true;
        for (std::size_t diagonal = 0; diagonal < size; ++diagonal)
        {
            if (!used[diagonal]) continue;
            if (first)
            {
                result = std::move(terms[diagonal]);
                first = false;
            }
            else
            {
                evaluator.add_inplace(result, terms[diagonal]);
            }
        }

        return result;
    }

    seal::Ciphertext approximate_gelu(
        const seal::Evaluator& evaluator,
        const seal::RelinKeys& relin_keys,
        seal::CKKSEncoder& encoder,
        const seal::Ciphertext& input,
        double scale)
    {

        // Minimax degree-four ReLU fit on [-2, 2]. The constant term lowers
        // the worst-case approximation error without adding multiplicative
        // depth; it is added at the deepest ciphertext level below.
        constexpr double constant = 0.06762090;
        constexpr double linear = 0.5;
        constexpr double quadratic = 0.48257484;
        constexpr double quartic = -0.06659632;

        // x^2 and x^4 are the only ciphertext-ciphertext products. Each is
        // relinearized and rescaled before it is used in another operation.
        seal::Ciphertext squared;
        evaluator.square(input, squared);
        evaluator.relinearize_inplace(squared, relin_keys);
        evaluator.rescale_to_next_inplace(squared);

        seal::Ciphertext fourth;
        evaluator.square(squared, fourth);
        evaluator.relinearize_inplace(fourth, relin_keys);
        evaluator.rescale_to_next_inplace(fourth);

        seal::Plaintext linear_plain;
        seal::Plaintext quadratic_plain;
        seal::Plaintext quartic_plain;
        seal::Plaintext constant_plain;
        encoder.encode(linear, scale, linear_plain);
        encoder.encode(quadratic, scale, quadratic_plain);
        encoder.encode(quartic, scale, quartic_plain);

        // The three branches have different levels because x, x^2, and x^4
        // have different multiplicative depths. Rescale each branch and then
        // switch it to the level of the deepest branch before adding.
        evaluator.mod_switch_to_inplace(linear_plain, input.parms_id());
        seal::Ciphertext linear_term;
        evaluator.multiply_plain(input, linear_plain, linear_term);
        evaluator.rescale_to_next_inplace(linear_term);

        evaluator.mod_switch_to_inplace(quadratic_plain, squared.parms_id());
        seal::Ciphertext quadratic_term;
        evaluator.multiply_plain(squared, quadratic_plain, quadratic_term);
        evaluator.rescale_to_next_inplace(quadratic_term);

        evaluator.mod_switch_to_inplace(quartic_plain, fourth.parms_id());
        seal::Ciphertext quartic_term;
        evaluator.multiply_plain(fourth, quartic_plain, quartic_term);
        evaluator.rescale_to_next_inplace(quartic_term);

        // Encode the constant at the exact scale produced by the final
        // rescale. Encoding it at the caller's input scale can leave a small
        // metadata mismatch that SEAL rejects during add_plain_inplace.
        encoder.encode(constant, quartic_term.scale(), constant_plain);

        evaluator.mod_switch_to_inplace(linear_term, quartic_term.parms_id());
        evaluator.mod_switch_to_inplace(quadratic_term, quartic_term.parms_id());
        evaluator.mod_switch_to_inplace(constant_plain, quartic_term.parms_id());

        // Rescaling produces very close, but not bit-identical, scales.
        // CKKS addition requires exact metadata equality.
        linear_term.scale() = quartic_term.scale();
        quadratic_term.scale() = quartic_term.scale();
        evaluator.add_plain_inplace(quartic_term, constant_plain);
        evaluator.add_inplace(quartic_term, quadratic_term);
        evaluator.add_inplace(quartic_term, linear_term);
        return quartic_term;
    }

}
