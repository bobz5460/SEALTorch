#pragma once

#include <seal/seal.h>

#include <vector>

namespace sealtorch
{
    seal::Ciphertext encrypted_dot_product(
        const seal::Evaluator& evaluator,
        std::vector<seal::Plaintext>& weights,
        const std::vector<seal::Ciphertext>& input);

    // Polynomial approximation of GELU. This is the degree-four Maclaurin
    // approximation of x * Phi(x), and is suitable for bounded CKKS inputs:
    //
    //   GELU(x) ~= 0.5*x + 0.3989422804*x^2 - 0.0664903801*x^4
    //
    // Keeping this helper free of erf/tanh makes it usable in encrypted
    // inference code as well as in ordinary host-side code.

    // Evaluate the same polynomial on a CKKS ciphertext. The ciphertext must
    // have at least three rescaling levels available, and the caller's scale
    // should match the scale used to encrypt input.
    seal::Ciphertext approximate_gelu(
        const seal::Evaluator& evaluator,
        const seal::RelinKeys& relin_keys,
        seal::CKKSEncoder& encoder,
        const seal::Ciphertext& input,
        double scale);

}
