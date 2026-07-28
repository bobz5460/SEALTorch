#pragma once

// Backward-compatible FIDESlib names. New code should use the CUDA-specific
// API in cuda_ciphertext_inference.h.
#include <SEALTorch/cuda_ciphertext_inference.h>

namespace sealtorch::fides
{
    using Options = cuda::CiphertextInferenceOptions;
    using InferenceEngine = cuda::CiphertextInferenceEngine;
    inline bool cuda_available() { return cuda::cuda_available(); }
}
