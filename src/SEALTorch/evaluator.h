#pragma once

#include <seal/seal.h>

#include "sklearn_joblib_loader.h"

#include <vector>

namespace sealtorch
{
    class Evaluator
    {
    public:
        explicit Evaluator(SklearnMLPModel model);

        void set_model(SklearnMLPModel model);

        [[nodiscard]] const SklearnMLPModel &model() const noexcept;
        [[nodiscard]] bool is_classifier() const noexcept;
        [[nodiscard]] bool is_regressor() const noexcept;

        // Evaluates the model with CKKS ciphertext inputs. The returned
        // ciphertexts are final-layer logits; this intentionally does not
        // apply softmax or select/decrypt a class label.
        [[nodiscard]] std::vector<seal::Ciphertext> predict(
            const std::vector<seal::Ciphertext> &input,
            const seal::Evaluator &evaluator,
            const seal::RelinKeys &relin_keys,
            seal::CKKSEncoder &encoder,
            double scale) const;

    private:
        static void validate_model(const SklearnMLPModel &model);

        SklearnMLPModel model_;
    };
}
