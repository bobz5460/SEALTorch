#pragma once

#include <SEALTorch/model.h>

#include <cstddef>
#include <memory>
#include <vector>

namespace sealtorch
{
    // Native ciphertexts are provider-specific. This API deliberately owns
    // them, so callers can switch providers without changing their inference
    // flow or accidentally mixing incompatible ciphertext formats.
    // Selects where native homomorphic-encryption operations run. Use normal
    // C++ type casing for new code; the all-caps spellings remain aliases.
    enum class ExecutionTarget { Auto, Cpu, Cuda };

    struct CiphertextInferenceOptions
    {
        ExecutionTarget target = ExecutionTarget::Auto;
        std::size_t thread_count = 4;
        std::size_t ring_dimension = 16384;
        std::size_t multiplicative_depth = 15;
        std::size_t scaling_modulus_bits = 40;
        std::size_t first_modulus_bits = 50;
        std::size_t scale_bits = 40;
        std::size_t activation_degree = 3;
        double activation_range = 4.0;
        ActivationApproximation approximation_method =
            ActivationApproximation::LeastSquares;
        int cuda_device = 0;
    };

    struct CiphertextInferenceResult
    {
        std::vector<double> values;
        double encrypt_ms = 0.0;
        double evaluate_ms = 0.0;
        double decrypt_ms = 0.0;
        // Exact bytes emitted by the provider's native serialization format.
        // This is the relevant size for storage or transport.
        std::size_t input_ciphertext_bytes = 0;
        std::size_t output_ciphertext_bytes = 0;
        // Uncompressed coefficient buffers currently occupied by ciphertexts.
        std::size_t input_ciphertext_memory_bytes = 0;
        std::size_t output_ciphertext_memory_bytes = 0;
        // Serialized context key material. These are setup costs, not bytes
        // transferred for an individual prediction.
        std::size_t secret_key_bytes = 0;
        std::size_t relin_keys_bytes = 0;
        std::size_t galois_keys_bytes = 0;
    };

    // Application-side encrypt -> evaluate -> decrypt adapter used by the
    // dashboard. The SEALTorch library itself exposes SealCiphertextModel.
    class CiphertextInference
    {
    public:
        CiphertextInference(Sequential model, CiphertextInferenceOptions options = {});
        ~CiphertextInference();
        CiphertextInference(CiphertextInference &&) noexcept;
        CiphertextInference &operator=(CiphertextInference &&) noexcept;
        CiphertextInference(const CiphertextInference &) = delete;
        CiphertextInference &operator=(const CiphertextInference &) = delete;

        CiphertextInferenceResult predict(const std::vector<double> &input);
        ExecutionTarget target() const;
        static bool cuda_available();

    private:
        struct Implementation;
        std::unique_ptr<Implementation> implementation_;
    };
}
