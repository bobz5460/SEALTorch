#pragma once
#include <seal/seal.h>
#include <SEALTorch/model.h>
#include <functional>
#include <string>
#include <vector>

namespace sealtorch
{
    enum class Backend
    {
        Scalar,
        Packed
    };

    class Evaluator
    {
    public:
        explicit Evaluator(Sequential model, Backend backend = Backend::Packed);

        void set_model(Sequential model);

        const Sequential &model() const;

        void set_backend(Backend backend);
        Backend backend() const;

        // Old names kept as small compatibility helpers.
        void set_mode(Backend backend) { set_backend(backend); }
        Backend mode() const { return backend(); }

        std::vector<seal::Ciphertext> predict_scalar(
            const std::vector<seal::Ciphertext> &input,
            const seal::Evaluator &evaluator,
            const seal::RelinKeys &relin_keys,
            const seal::GaloisKeys &galois_keys,
            seal::CKKSEncoder &encoder,
            double scale) const;

        seal::Ciphertext predict_packed(
            const seal::SEALContext &context,
            const seal::Ciphertext &input,
            const seal::Evaluator &evaluator,
            const seal::RelinKeys &relin_keys,
            const seal::GaloisKeys &galois_keys,
            seal::CKKSEncoder &encoder,
            double scale,
            std::size_t thread_count) const;

    private:
        Sequential model_;
        Backend backend_;
        mutable std::vector<std::vector<seal::Plaintext>> cached_weights_;
        mutable std::vector<seal::parms_id_type> cached_parms_;
    };
}
