#pragma once

#include <SEALTorch/fides_compat.h>

#include <cstddef>
#include <vector>

namespace sealtorch
{
    class ThreadPool;

    seal::Ciphertext encrypted_dot_product(
        const seal::Evaluator &evaluator,
        const seal::GaloisKeys &galois_keys,
        const seal::Plaintext &weights,
        const seal::Ciphertext &input,
        std::size_t input_width);

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
        ThreadPool &thread_pool,
        const std::vector<seal::Plaintext> *cached_weights);

    seal::Ciphertext approximate_relu(
        const seal::Evaluator& evaluator,
        const seal::RelinKeys& relin_keys,
        seal::CKKSEncoder& encoder,
        const seal::Ciphertext& input,
        double scale);

    // Degree-four polynomial approximation of GELU. Accuracy is best when
    // values entering the activation are mostly in [-2, 2].
    //
    //   GELU(x) ~= 0.5*x + 0.39894228*x^2 - 0.06649038*x^4
    //
    // The ciphertext
    // must have at least three rescaling levels available, and the caller's
    // scale should match the scale used to encrypt input.
    seal::Ciphertext approximate_gelu(
        const seal::Evaluator& evaluator,
        const seal::RelinKeys& relin_keys,
        seal::CKKSEncoder& encoder,
        const seal::Ciphertext& input,
        double scale);

}
