#pragma once

#include <SEALTorch/fides_compat.h>
#include <SEALTorch/model.h>
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

        std::vector<seal::Ciphertext> linear_scalar(
            const std::vector<seal::Ciphertext> &input,
            const DenseLayer &layer,
            const seal::Evaluator &evaluator,
            const seal::GaloisKeys &galois_keys,
            seal::CKKSEncoder &encoder,
            double scale) const;

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

        std::vector<seal::Ciphertext> activation(
            const std::vector<seal::Ciphertext> &input,
            ActivationType type,
            const seal::SEALContext &context,
            const seal::Evaluator &evaluator,
            const seal::RelinKeys &relin_keys,
            seal::CKKSEncoder &encoder,
            double scale,
            std::size_t thread_count) const;

    private:
        Sequential model_;
        mutable std::vector<std::vector<seal::Plaintext>> cached_weights_;
        mutable std::vector<seal::parms_id_type> cached_parms_;
        mutable ThreadPool thread_pool_{1};
    };
}
