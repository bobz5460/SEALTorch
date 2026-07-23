#pragma once

#include <seal/seal.h>

#include <cstddef>
#include <vector>

namespace sealtorch
{
    // Computes all rows of weights * input in one packed ciphertext.
    // Input and output values use the first slots.
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
        const std::vector<seal::Plaintext> *cached_weights);

    // Degree-four polynomial approximation of ReLU, exposed under the legacy
    // approximate_gelu name used by the evaluator. The coefficients are a
    // minimax fit to max(0, x) on the bounded interval [-2, 2]:
    //
    //   ReLU(x) ~= 0.06762090 + 0.5*x + 0.48257484*x^2 - 0.06659632*x^4
    //
    // This minimax fit reduces the worst-case error on [-2, 2] while using
    // the same multiplicative depth as the old GELU approximation.

    // Evaluate the polynomial on a CKKS ciphertext. Accuracy is best when
    // values entering the activation are mostly in [-2, 2]. The ciphertext
    // must have at least three rescaling levels available, and the caller's
    // scale should match the scale used to encrypt input.
    seal::Ciphertext approximate_gelu(
        const seal::Evaluator& evaluator,
        const seal::RelinKeys& relin_keys,
        seal::CKKSEncoder& encoder,
        const seal::Ciphertext& input,
        double scale);

}
