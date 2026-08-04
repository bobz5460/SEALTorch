#pragma once

#include <SEALTorch/evaluator.h>

#include <cstddef>
#include <vector>

namespace sealtorch
{
    struct SealInferenceConfig
    {
        const seal::SEALContext &context;
        const seal::Evaluator &evaluator;
        const seal::RelinKeys &relin_keys;
        const seal::GaloisKeys &galois_keys;
        seal::CKKSEncoder &encoder;
        double scale;
        std::size_t thread_count = 4;
    };

    // SEALTorch receives encrypted user data and returns encrypted results.
    // The application owns the context, keys, encoding, encryption, and
    // decryption boundary.
    class SealCiphertextModel
    {
    public:
        explicit SealCiphertextModel(Sequential model);

        seal::Ciphertext predict(
            const seal::Ciphertext &input,
            const SealInferenceConfig &config) const;

        const Sequential &model() const;

    private:
        mutable Evaluator evaluator_;
    };
}
