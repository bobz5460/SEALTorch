#pragma once
#include <seal/seal.h>
#include <SEALTorch/model.h>
#include <functional>
#include <string>
#include <vector>

namespace sealtorch
{
    struct ProgressInfo
    {
        std::string phase;
        std::size_t layer = 0;
        std::size_t total_layers = 0;
        std::size_t completed = 0;
        std::size_t total = 0;
        std::size_t layer_completed = 0;
        std::size_t layer_total = 0;
    };

    using ProgressCallback = std::function<void(const ProgressInfo &)>;

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
            const seal::GaloisKeys &galois_keys,
            seal::CKKSEncoder &encoder,
            double scale,
            const ProgressCallback &progress = {}) const;

    private:
        NeuralNetwork model_;
    };
}
