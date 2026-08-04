#pragma once

#include <SEALTorch/model.h>

#include <cstddef>
#include <memory>
#include <vector>

namespace sealtorch::cuda
{
    // True when FIDESlib can use at least one CUDA device in this process.
    bool cuda_available();

    // Parameters needed to create a native FIDESlib CKKS context. Keep this
    // separate from a model so experiments can vary HE parameters clearly.
    struct CiphertextInferenceOptions
    {
        // FIDESlib uses OpenMP for host-side work even when the HE kernels
        // execute on CUDA. Keep this bounded so a GPU inference does not
        // default to every logical CPU on the machine.
        std::size_t thread_count = 4;
        std::size_t ring_dimension = 16384;
        std::size_t multiplicative_depth = 15;
        std::size_t scaling_modulus_bits = 40;
        std::size_t first_modulus_bits = 50;
        std::size_t activation_degree = 3;
        int device = 0;
    };

    struct CiphertextInferenceResult
    {
        std::vector<double> values;
        double encrypt_ms = 0.0;
        double evaluate_ms = 0.0;
        double decrypt_ms = 0.0;
    };

    // Packed ciphertext inference backed directly by FIDESlib. The engine
    // owns its context, keys, and cached model constants. It is an
    // application adapter; the core SEALTorch target remains ciphertext-only.
    class CiphertextInferenceEngine
    {
    public:
        CiphertextInferenceEngine(Sequential model, CiphertextInferenceOptions options = {});
        ~CiphertextInferenceEngine();
        CiphertextInferenceEngine(CiphertextInferenceEngine &&) noexcept;
        CiphertextInferenceEngine &operator=(CiphertextInferenceEngine &&) noexcept;
        CiphertextInferenceEngine(const CiphertextInferenceEngine &) = delete;
        CiphertextInferenceEngine &operator=(const CiphertextInferenceEngine &) = delete;

        CiphertextInferenceResult predict(const std::vector<double> &input);
        std::size_t slot_count() const;

    private:
        struct Implementation;
        std::unique_ptr<Implementation> implementation_;
    };
}
