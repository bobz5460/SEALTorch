#include "seal_ciphertext_backend.h"

#include <utility>

namespace sealtorch
{
    SealCiphertextModel::SealCiphertextModel(Sequential model)
        : evaluator_(std::move(model))
    {
    }

    seal::Ciphertext SealCiphertextModel::predict(
        const seal::Ciphertext &input,
        const SealInferenceConfig &config) const
    {
        seal::Ciphertext value = input;
        std::size_t layer_index = 0;
        for (const Operation &operation : model().operations()) {
            if (operation.kind == OperationKind::Linear) {
                value = evaluator_.linear_packed(
                    config.context, value, operation.linear_layer, layer_index++,
                    config.evaluator, config.galois_keys, config.encoder,
                    config.scale, config.thread_count);
            } else {
                value = evaluator_.activation(
                    value, operation.activation_type, config.context,
                    config.relin_keys, config.scale,
                    config.activation_degree);
            }
        }
        return value;
    }

    const Sequential &SealCiphertextModel::model() const
    {
        return evaluator_.model();
    }
}
