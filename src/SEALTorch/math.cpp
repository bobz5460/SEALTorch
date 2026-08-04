#include "math.h"
#include "thread_pool.h"

namespace sealtorch
{
    std::vector<double> activation_taylor_coefficients(ActivationType type)
    {
        if (type == ActivationType::Relu)
            return {0.0, 1.0};
        if (type == ActivationType::Tanh)
            return {0.0, 1.0, 0.0, -1.0 / 3.0};
        return {0.0, 0.5, 0.3989422804014327, 0.0,
                -0.0664903800669054};
    }

    seal::Ciphertext approximate_activation(
        const seal::Evaluator &evaluator, const seal::RelinKeys &relin_keys,
        seal::CKKSEncoder &encoder, const seal::Ciphertext &input, double scale,
        ActivationType type)
    {
        const std::vector<double> coefficients =
            activation_taylor_coefficients(type);
        const std::size_t highest = coefficients.size() - 1;
        if (highest == 1) return input;

        seal::Ciphertext result = input;
        if (coefficients[highest] != 1.0) {
            seal::Plaintext coefficient;
            encoder.encode(coefficients[highest], scale, coefficient);
            evaluator.mod_switch_to_inplace(coefficient, result.parms_id());
            evaluator.multiply_plain_inplace(result, coefficient);
            evaluator.rescale_to_next_inplace(result);
        }

        for (std::size_t order = highest; order-- > 1;) {
            if (coefficients[order] != 0.0) {
                seal::Plaintext coefficient;
                encoder.encode(coefficients[order], result.scale(), coefficient);
                evaluator.mod_switch_to_inplace(coefficient, result.parms_id());
                evaluator.add_plain_inplace(result, coefficient);
            }
            seal::Ciphertext factor = input;
            evaluator.mod_switch_to_inplace(factor, result.parms_id());
            evaluator.multiply_inplace(result, factor);
            evaluator.relinearize_inplace(result, relin_keys);
            evaluator.rescale_to_next_inplace(result);
        }
        if (coefficients[0] != 0.0) {
            seal::Plaintext constant;
            encoder.encode(coefficients[0], result.scale(), constant);
            evaluator.mod_switch_to_inplace(constant, result.parms_id());
            evaluator.add_plain_inplace(result, constant);
        }
        return result;
    }

    seal::Ciphertext encrypted_matrix_vector_product(
        const seal::SEALContext& context,
        const seal::Evaluator& evaluator,
        const seal::GaloisKeys& galois_keys,
        const seal::Ciphertext& input,
        const EncodedMatrix& matrix,
        std::size_t thread_count,
        ThreadPool &thread_pool)
    {
        // Baby-step/giant-step evaluates many diagonals while using only about
        // 2*sqrt(slot_count) rotations. Rotation keys are much larger than
        // plaintext weights, so this substantially reduces memory use.
        std::vector<seal::Ciphertext> rotated_inputs(
            matrix.input_rotations.size());
        thread_pool.parallel_for_workers(
            matrix.input_rotations.size(),
            thread_count,
            [&](std::size_t worker, std::size_t jobs)
            {
                seal::Evaluator local_evaluator(context);
                seal::MemoryPoolHandle pool = seal::MemoryPoolHandle::ThreadLocal();
                for (std::size_t index = worker;
                     index < matrix.input_rotations.size();
                     index += jobs)
                {
                    const int rotation = matrix.input_rotations[index];
                    if (rotation == 0)
                        rotated_inputs[index] = input;
                    else
                        local_evaluator.rotate_vector(
                            input,
                            rotation,
                            galois_keys,
                            rotated_inputs[index],
                            pool);
                }
            });

        std::vector<seal::Ciphertext> group_results(matrix.groups.size());
        thread_pool.parallel_for_workers(
            matrix.groups.size(),
            thread_count,
            [&](std::size_t worker, std::size_t jobs)
            {
                seal::Evaluator local_evaluator(context);
                seal::MemoryPoolHandle pool = seal::MemoryPoolHandle::ThreadLocal();
                for (std::size_t group_index = worker;
                     group_index < matrix.groups.size();
                     group_index += jobs)
                {
                    const EncodedDiagonalGroup &group =
                        matrix.groups[group_index];
                    seal::Ciphertext subtotal;
                    bool first = true;
                    for (const EncodedDiagonal &diagonal : group.diagonals)
                    {
                        seal::Ciphertext term;
                        local_evaluator.multiply_plain(
                            rotated_inputs[diagonal.input_index],
                            diagonal.weights,
                            term,
                            pool);
                        if (first) {
                            subtotal = std::move(term);
                            first = false;
                        } else {
                            local_evaluator.add_inplace(subtotal, term);
                        }
                    }

                    if (group.rotation == 0)
                        group_results[group_index] = std::move(subtotal);
                    else
                        local_evaluator.rotate_vector(
                            subtotal,
                            group.rotation,
                            galois_keys,
                            group_results[group_index],
                            pool);
                }
            });

        seal::Ciphertext result = std::move(group_results.front());
        for (std::size_t index = 1; index < group_results.size(); ++index)
        {
            evaluator.add_inplace(result, group_results[index]);
        }
        return result;
    }

}
