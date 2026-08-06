# SEALTorch

SEALTorch is a small research project for encrypted MNIST inference on a CPU
or NVIDIA GPU. It uses:

- Microsoft SEAL for CPU homomorphic encryption
- FIDESlib for CUDA homomorphic encryption
- PyTorch as an application-side reference
- A local browser dashboard for predictions and benchmarks

The public C++ entry point is `#include <SEALTorch/sealtorch.h>`. Its inference
API accepts a Microsoft SEAL ciphertext and returns a ciphertext. Encoding,
encryption, decryption, and decoded user data stay at the application boundary.
The dashboard keeps one native worker alive so keys and prepared model weights
are reused between predictions.

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

Validate that an artifact can be translated without creating encryption keys:

```bash
build/sealtorch_gui --validate-model src/lenet.json
```

## Code map

- `src/SEALTorch/model.h` defines layers and sequential models.
- `src/app/ciphertext_inference.*` is the dashboard's provider adapter.
- `src/SEALTorch/evaluator.*` runs Microsoft SEAL operations.
- `src/app/cuda_ciphertext_inference.*` runs FIDESlib operations.
- `src/SEALTorch/math.*` contains packed linear algebra and fitted activation
  evaluation.
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
average pooling; it reports a precise error for unsupported activations or max
pooling.

Trainer exports are identified by their path below `exports/`, so models with
the same filename in different experiment folders all appear in the dashboard.
Encrypted inference rejects clamp layers because silently replacing a clamp
with the identity changes the trained graph. Export a real supported
activation and let SEALTorch approximate it instead.

Encrypted activations use a power-basis polynomial selected by
`approximation_method`: `least_squares` fits 257 uniformly spaced samples,
`chebyshev` interpolates at Chebyshev roots, and `taylor` uses the historical
zero-centered series. `activation_range` controls the fitting interval for
least-squares and Chebyshev, while `activation_degree` selects the degree. The
same coefficients drive CPU, CUDA, depth validation, and the dashboard graph.
Activations outside the selected interval remain extrapolation; choose the
range from measured model activations. Least-squares is the default because it
preserved the best accuracy in the current LeNet regression checks.

The dashboard accepts polynomial degrees 1 through 15. Its approximation
advisor calibrates the selected model on a deterministic validation subset,
scans candidate ranges, and reports validation accuracy, activation
percentiles, effective polynomial degrees, required multiplicative depth, and
the slot/security-driven ring recommendation before an expensive HE run. The
Apply button sets the complete profile: range, depth, ring dimension, first
modulus, scaling modulus, and CKKS scale. Deep lowered CNNs stream one layer of
packed CUDA constants at a time instead of retaining every convolution
plaintext on the GPU. CUDA retains all CPU-packed constants, keeps ordinary
layers GPU-resident, and transfers only oversized lowered convolutions. Compact
power-of-two rotation keys replace a separate key for every convolution
rotation. Odd Tanh polynomials use the lower-depth `x * q(x²)` circuit; for
example, degree 5 uses four multiplicative levels rather than five.
