#include "math.h"
#include "thread_pool.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace sealtorch
{
    static seal::Ciphertext sum_slots(
        const seal::Ciphertext &input,
        const seal::Evaluator &evaluator,
        const seal::GaloisKeys &galois_keys,
        std::size_t count)
    {
        seal::Ciphertext result = input;
        for (std::size_t step = 1; step < count; step *= 2)
        {
            seal::Ciphertext rotated;
            evaluator.rotate_vector(result, static_cast<int>(step), galois_keys, rotated);
            evaluator.add_inplace(result, rotated);
        }
        return result;
    }

    seal::Ciphertext encrypted_dot_product(
        const seal::Evaluator &evaluator,
        const seal::GaloisKeys &galois_keys,
        const seal::Plaintext &weights,
        const seal::Ciphertext &input,
        std::size_t input_width)
    {
        seal::Ciphertext result;
        evaluator.multiply_plain(input, weights, result);
        return sum_slots(result, evaluator, galois_keys, input_width);
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

    static seal::Ciphertext approximate_polynomial(
        const seal::Evaluator& evaluator,
        const seal::RelinKeys& relin_keys,
        seal::CKKSEncoder& encoder,
        const seal::Ciphertext& input,
        double scale,
        double constant,
        double linear,
        double quadratic,
        double quartic)
    {

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

    seal::Ciphertext approximate_relu(
        const seal::Evaluator& evaluator,
        const seal::RelinKeys& relin_keys,
        seal::CKKSEncoder& encoder,
        const seal::Ciphertext& input,
        double scale)
    {
        // The base fit is defined on [-2, 2], while the exported MNIST MLP
        // has activations up to about 10.  Apply P(x / 5) * 5 so the same
        // degree and multiplicative depth cover [-10, 10].  Evaluating the
        // unscaled polynomial at those values makes its negative quartic
        // term dominate and collapses ciphertext predictions.
        return approximate_polynomial(
            evaluator, relin_keys, encoder, input, scale,
            0.33810450, 0.5, 0.096514968, -0.00053277056);
    }

    seal::Ciphertext approximate_gelu(
        const seal::Evaluator& evaluator,
        const seal::RelinKeys& relin_keys,
        seal::CKKSEncoder& encoder,
        const seal::Ciphertext& input,
        double scale)
    {
        return approximate_polynomial(
            evaluator, relin_keys, encoder, input, scale,
            0.0, 0.5, 0.3989422804014327, -0.0664903800669054);
    }

    seal::Ciphertext approximate_tanh(
        const seal::Evaluator& evaluator,
        const seal::RelinKeys& relin_keys,
        seal::CKKSEncoder& encoder,
        const seal::Ciphertext& input,
        double scale)
    {
        // Least-squares degree-three fit on [-8, 8]. The wider range avoids
        // the sign flip a narrow local fit produces in LeNet's second block.
        // It is deliberately shallow enough for practical CKKS evaluation.
        seal::Ciphertext squared;
        evaluator.square(input, squared);
        evaluator.relinearize_inplace(squared, relin_keys);
        evaluator.rescale_to_next_inplace(squared);
        seal::Ciphertext result;
        seal::Plaintext cubic_coefficient;
        encoder.encode(-0.003956717265710641, scale, cubic_coefficient);
        evaluator.mod_switch_to_inplace(cubic_coefficient, squared.parms_id());
        evaluator.multiply_plain_inplace(squared, cubic_coefficient);
        evaluator.rescale_to_next_inplace(squared);

        seal::Ciphertext cubic_input = input;
        evaluator.mod_switch_to_inplace(cubic_input, squared.parms_id());
        evaluator.multiply(squared, cubic_input, result);
        evaluator.relinearize_inplace(result, relin_keys);
        evaluator.rescale_to_next_inplace(result);
        seal::Plaintext linear_coefficient;
        encoder.encode(0.3370407386009496, scale, linear_coefficient);
        evaluator.mod_switch_to_inplace(linear_coefficient, input.parms_id());
        seal::Ciphertext linear;
        evaluator.multiply_plain(input, linear_coefficient, linear);
        evaluator.rescale_to_next_inplace(linear);
        evaluator.mod_switch_to_inplace(linear, result.parms_id());
        linear.scale() = result.scale();
        evaluator.add_inplace(result, linear);
        return result;
    }

}
