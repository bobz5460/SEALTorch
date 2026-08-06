#pragma once

#include <seal/seal.h>
#include <SEALTorch/model.h>
#include <SEALTorch/math.h>
#include <SEALTorch/thread_pool.h>

#include <cstddef>
#include <vector>

namespace sealtorch
{
    class Evaluator
    {
    public:
        explicit Evaluator(Sequential model);

        const Sequential &model() const;

        seal::Ciphertext linear_packed(
            const seal::SEALContext &context,
            const seal::Ciphertext &input,
            const DenseLayer &layer,
            std::size_t layer_index,
            const seal::Evaluator &evaluator,
            const seal::GaloisKeys &galois_keys,
            seal::CKKSEncoder &encoder,
            double scale,
            std::size_t thread_count) const;

        seal::Ciphertext activation(
            const seal::Ciphertext &input,
            ActivationType type,
            const seal::SEALContext &context,
            const seal::RelinKeys &relin_keys,
            double scale,
            std::size_t degree,
            double range,
            ActivationApproximation method) const;

    private:
        Sequential model_;
        mutable std::vector<EncodedMatrix> cached_weights_;
        mutable std::vector<seal::parms_id_type> cached_parms_;
        mutable ThreadPool thread_pool_{1};
    };
}
