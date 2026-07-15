## Build

```sh
cmake -S . -B build
cmake --build build
```

## Web trace UI

Build the trace executable, then start the local dashboard:

```sh
cmake --build build
python3 webui/server.py
```

Open <http://127.0.0.1:8000>. Draw a digit and click **Run comparison**. The
 dashboard uses `src/mnist_mlp_gelu.json` and shows plaintext and CKKS
 pre-activation values, post-activation values, and absolute error for every
 neuron in every layer.

If SEAL is installed somewhere else, provide its package directory:

```sh
cmake -S . -B build \
  -DSEAL_DIR="$HOME/.local/seal/lib/cmake/SEAL-4.3"
cmake --build build
```
