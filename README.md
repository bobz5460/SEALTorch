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

The dashboard evaluates neurons in each layer concurrently, with four worker
threads by default. Configure the cap when starting it:

```sh
python3 webui/server.py --threads 2
```

Open <http://127.0.0.1:8000>. Draw a digit and click **Run comparison**. The
 dashboard uses `src/mnist_mlp_gelu.json` and shows plaintext and CKKS
 pre-activation values, post-activation values, and absolute error for every
 neuron in every layer.

The **MNIST validation** tab runs a configurable number of images from the
raw MNIST test set through each selected model. It reports plaintext GELU and
ReLU accuracy, encrypted approximate-GELU accuracy, prediction agreement,
per-image runtime, and mean encrypted/plain output error. The server looks for
the raw files under `data/MNIST/raw` inside this repository by default. Set the
portable `MNIST_DIR` environment variable or use `--mnist-dir PATH` when they
are elsewhere. The batch validator can also
be run directly after building:

```sh
./build/sealtorch_validate \
  --images /path/to/t10k-images-idx3-ubyte \
  --labels /path/to/t10k-labels-idx1-ubyte \
  --count 10 --model src/mnist_mlp_gelu.json
```

To access a UI running on another machine through SSH, keep the server bound
to localhost and run this on your local machine:

```sh
ssh -N -L 8000:127.0.0.1:8000 user@remote-host
```

Then open <http://127.0.0.1:8000> locally. For a cloud/container platform
that forwards a port directly, bind to all interfaces instead:

```sh
python3 webui/server.py --host 0.0.0.0 --port 8000
```

The trace endpoint runs a local executable and has no authentication, so use
SSH forwarding or put it behind an authenticated reverse proxy rather than
exposing port 8000 directly to the public internet.

If SEAL is installed somewhere else, provide its package directory:

```sh
cmake -S . -B build \
  -DSEAL_DIR="$HOME/.local/seal/lib/cmake/SEAL-4.3"
cmake --build build
```
