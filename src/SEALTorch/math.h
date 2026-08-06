#pragma once

#include <seal/seal.h>
#include <SEALTorch/model.h>

#include <cstddef>
#include <vector>

namespace sealtorch
{
    class ThreadPool;

    // Selected power-basis approximation over [-range, range].
    std::vector<double> activation_polynomial_coefficients(
        ActivationType type, std::size_t degree, double range,
        ActivationApproximation method = ActivationApproximation::LeastSquares);

    // Multiplicative levels used by the evaluator. Odd polynomials use an
    // x * q(x^2) circuit instead of depth-linear Horner evaluation.
    std::size_t activation_polynomial_depth(
        ActivationType type, std::size_t degree, double range,
        ActivationApproximation method = ActivationApproximation::LeastSquares);

    seal::Ciphertext approximate_activation(
        const seal::Evaluator& evaluator, const seal::RelinKeys& relin_keys,
        seal::CKKSEncoder& encoder, const seal::Ciphertext& input,
        double scale, ActivationType type, std::size_t degree, double range,
        ActivationApproximation method = ActivationApproximation::LeastSquares);

    struct EncodedDiagonal
    {
        std::size_t input_index = 0;
        seal::Plaintext weights;
    };

    struct EncodedDiagonalGroup
    {
        int rotation = 0;
        std::vector<EncodedDiagonal> diagonals;
    };

    struct EncodedMatrix
    {
        std::vector<int> input_rotations;
        std::vector<EncodedDiagonalGroup> groups;
    };

    // Computes all rows of weights * input in one packed ciphertext.
    // Input and output values use the first slots.
    seal::Ciphertext encrypted_matrix_vector_product(
        const seal::SEALContext& context,
        const seal::Evaluator& evaluator,
        const seal::GaloisKeys& galois_keys,
        const seal::Ciphertext& input,
        const EncodedMatrix& matrix,
        std::size_t thread_count,
        ThreadPool &thread_pool);

}
