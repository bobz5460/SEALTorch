#pragma once
#include <seal/seal.h>
#include <SEALTorch/model.h>
#include <functional>
#include <string>
#include <vector>

namespace sealtorch
{
    class Evaluator
    {
    public:
        explicit Evaluator(NeuralNetwork model);

        void set_model(NeuralNetwork model);

        const NeuralNetwork &model() const;

        seal::Ciphertext predict(
            const seal::SEALContext &context,
            const seal::Ciphertext &input,
            const seal::Evaluator &evaluator,
            const seal::RelinKeys &relin_keys,
            const seal::GaloisKeys &galois_keys,
            seal::CKKSEncoder &encoder,
            double scale,
            std::size_t thread_count) const;

    private:
        NeuralNetwork model_;
        mutable std::vector<std::vector<seal::Plaintext>> cached_weights_;
        mutable std::vector<seal::parms_id_type> cached_parms_;
    };
}
