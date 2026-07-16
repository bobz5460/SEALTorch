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
    using LayerCallback = std::function<void(std::size_t layer, bool after_activation,
                                             const std::vector<seal::Ciphertext> &values)>;

    class Evaluator
    {
    public:
        explicit Evaluator(NeuralNetwork model, std::size_t max_concurrency = 4);

        void set_model(NeuralNetwork model);

        const NeuralNetwork &model() const;

        void set_max_concurrency(std::size_t max_concurrency);
        std::size_t max_concurrency() const;

        std::vector<seal::Ciphertext> predict(
            const std::vector<seal::Ciphertext> &input,
            const seal::Evaluator &evaluator,
            const seal::RelinKeys &relin_keys,
            const seal::GaloisKeys &galois_keys,
            seal::CKKSEncoder &encoder,
            double scale,
            const ProgressCallback &progress = {},
            const LayerCallback &layer_callback = {}) const;

    private:
        NeuralNetwork model_;
        std::size_t max_concurrency_;
    };
}
