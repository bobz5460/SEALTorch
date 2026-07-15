## Build

```sh
cmake -S . -B build
cmake --build build
```

Run the drawing GUI with the bundled JSON model:

```sh
./build/sealtorch_gui src/mnist_mlp.json
```

Run the tiny encrypted inference smoke test:

```sh
./build/sealtorch_tiny_test
```

The test uses a 2-input, 2-hidden-neuron, 1-output model and exercises the
rescaling boundary between layers.

Draw a white digit on the canvas, click `Predict`, or click `Clear` to start
again. The JSON is loaded into SEALTorch's `NeuralNetwork`/`Evaluator` model
contract; `Predict` encrypts the 784 inputs, calls SEALTorch's CKKS evaluator,
then decrypts the ten output scores for display. The JSON preprocessing
contract is applied before encryption.

If SEAL is installed somewhere else, provide its package directory:

```sh
cmake -S . -B build \
  -DSEAL_DIR="$HOME/.local/seal/lib/cmake/SEAL-4.3"
cmake --build build
```
