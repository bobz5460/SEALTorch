# SEALTorch

SEALTorch is a small research project for comparing encrypted MNIST inference
on a CPU and an NVIDIA GPU. It uses:

- Microsoft SEAL for CPU homomorphic encryption
- FIDESlib for CUDA homomorphic encryption
- PyTorch as the plaintext reference
- A local browser dashboard for predictions and benchmarks

The public C++ entry point is `#include <SEALTorch/sealtorch.h>`. The dashboard
talks to one long-running native worker, so encryption keys and prepared model
weights are reused between predictions.

## Build

You need CMake 3.22 or newer, a C++20 compiler, Microsoft SEAL 4.4, Python 3,
and PyTorch. FIDESlib, OpenMP, and the CUDA runtime are optional; when CMake
cannot find FIDESlib, it builds a CPU-only version.

```bash
cmake -S . -B build \
  -DSEAL_DIR="$HOME/.local/lib/cmake/SEAL-4.4" \
  -DFIDESLIB_DIR="$HOME/.local/fideslib/share/fideslib/cmake"
cmake --build build -j2
```

To explicitly build without CUDA, add `-DSEALTORCH_ENABLE_CUDA=OFF`.

Run the tests:

```bash
ctest --test-dir build --output-on-failure
python -m unittest discover -s tests -p "test_*.py" -v
```

## Run the dashboard

```bash
python webui/server.py
```

Then open <http://127.0.0.1:8080>. The server only listens on the local
computer.

To use a build directory other than `build`, set `SEALTORCH_BUILD_DIR`.
To select one exact worker binary, set `SEALTORCH_HE_BINARY`.

## Code map

- `src/SEALTorch/model.h` defines layers and sequential models.
- `src/SEALTorch/ciphertext_inference.*` chooses the CPU or CUDA provider.
- `src/SEALTorch/evaluator.*` runs Microsoft SEAL operations.
- `src/SEALTorch/cuda_ciphertext_inference.*` runs FIDESlib operations.
- `src/SEALTorch/math.*` contains packed linear algebra and activation fits.
- `src/SEALTorch/packing.h` finds the sparse diagonals needed by packed CKKS.
- `src/main.cpp` translates JSON model artifacts and implements the native worker protocol.
- `webui/model_translator.py` accepts the LeNet trainer's self-describing `.pt`
  bundles or `.json` + `.weights.npz` manifests, folds inference batch norm,
  and caches native worker artifacts.
- `webui/server.py` provides the local HTTP API and benchmark runner.
- `webui/index.html` is the dependency-free dashboard.

## Performance notes

Encrypted inference is inherently much slower and larger than plaintext
inference. SEALTorch reduces avoidable overhead by caching encoded weights,
generating rotation keys only for non-zero diagonals, reusing warm workers, and
limiting the PyTorch model cache.

The LeNet artifact is currently lowered to sparse linear transforms because the
native providers share a dense packed interface. This keeps the implementation
consistent, but it is still more expensive than a provider-specific encrypted
convolution kernel.

## LeNet trainer exports

When the sibling `../LeNet-5` checkout is present (or `SEALTORCH_LENET_ROOT`
points to it), the dashboard discovers its `exports/**/*.pt` models. The
paired `.json` manifests are also supported and use their referenced
`.weights.npz` tensors. Plaintext runs support every inference method emitted
by the trainer, including batch norm, dropout, all activations, and both pool
types. The HE path currently approximates tanh, ReLU, and GELU and supports
average pooling; it reports a precise error for non-polynomial activations or
max pooling rather than silently changing the trained network.
