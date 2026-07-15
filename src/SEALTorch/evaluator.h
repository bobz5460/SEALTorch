#pragma once
#include <seal/seal.h>
#include <SEALTorch/model.h>
#include <vector>

namespace sealtorch
{
    class Evaluator
    {
    public:
        explicit Evaluator(NeuralNetwork model);

        void set_model(NeuralNetwork model);

        const NeuralNetwork &model() const;

        std::vector<seal::Ciphertext> predict(
            const std::vector<seal::Ciphertext> &input,
            const seal::Evaluator &evaluator,
            const seal::RelinKeys &relin_keys,
            seal::CKKSEncoder &encoder,
            double scale) const;

    private:
        NeuralNetwork model_;
    };
}
