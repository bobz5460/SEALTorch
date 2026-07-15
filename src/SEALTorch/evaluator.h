#pragma once
#include <seal/seal.h>
#include <SEALTorch/model.h>
#include <functional>
#include <vector>

namespace sealtorch
{
    struct ProgressInfo
    {
        std::size_t completed;
        std::size_t total;
        std::size_t layer;
        std::size_t neuron;
        std::size_t layer_completed;
        std::size_t layer_total;
    };

    using ProgressCallback = std::function<void(const ProgressInfo &progress)>;

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
            double scale,
            ProgressCallback progress = {}) const;

    private:
        NeuralNetwork model_;
    };
}
