#include "math.h"
#include <cmath>
#include <stdexcept>

namespace sealtorch
{
    seal::Ciphertext sum_slots(
        const seal::Ciphertext &input,
        const seal::Evaluator &evaluator,
        const seal::GaloisKeys &galois_keys,
        std::size_t count)
    {
        seal::Ciphertext result = input;

        for (std::size_t step = 1; step < count; step *= 2)
        {
            seal::Ciphertext rotated;

            evaluator.rotate_vector(
                result,
                static_cast<int>(step),
                galois_keys,
                rotated);

            evaluator.add_inplace(result, rotated);
        }

        return result;
    }

    seal::Ciphertext encrypted_dot_product(
        const seal::Evaluator& evaluator,
        const seal::GaloisKeys& galois_keys,
        const seal::Plaintext& weights,
        const seal::Ciphertext& input,
        std::size_t input_width)
    {
        seal::Ciphertext result;
        evaluator.multiply_plain(input, weights, result);
        return sum_slots(result, evaluator, galois_keys, input_width);
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
