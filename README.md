# SEALTorch

SEALTorch is a small research pipeline for comparing real PyTorch activations,
polynomial substitutes, and CKKS encrypted inference. It supports fixed MLP and
classic LeNet-5 baselines on MNIST and EMNIST ByClass with ReLU, GELU, or Tanh.

Encrypted inference is single-GPU and FIDESlib-only. There is no Microsoft SEAL
CPU provider, multi-GPU sharding, bootstrapping, or model-specific training in
this repository.

## 1. Train baselines

In the sibling trainer:

```bash
cd ../LeNet-5
python train.py --dataset all --model all --activation all --output-dir exports
```

The dashboard discovers the resulting `exports/<dataset>/<model>-<activation>.json`
artifacts automatically.

## 2. Build encrypted inference

```bash
cmake -S . -B build -DFIDESLIB_DIR="$HOME/.local/fideslib/share/fideslib/cmake"
cmake --build build -j2
build/sealtorch_he --capabilities
```

FIDESlib supplies the CKKS context, GPU operations, rotation/accumulation
primitives, and native Chebyshev-series evaluator. Automatic profiles target
128-bit classic security and skip circuits that would require bootstrapping.
Packed weights stay on the CPU between operations by default and stream through FIDESlib's
GPU memory pool one layer at a time, rather than keeping every LeNet diagonal
resident simultaneously. The publication MLP uses measured minimum profiles:
depths 9/11/13/13 for degrees 2/4/6/8, 16K rings for degrees 2/4, 32K rings for
degrees 6/8, 25-bit scaling, and a 35-bit first modulus. OpenFHE still enforces
128-bit classic security for every generated context.

The refactor was grounded in the complete local FIDESlib example set: `simple`
and `advanced` for the public context API, `serial` for persistence,
`bootstrap` for the deliberately excluded bootstrapping path, `hpca` for
polynomial/SIMD patterns, `logreg` for packed inference, `resnet` for
convolution transforms, `bert-tiny` for linear transforms and polynomial
evaluation, plus the matrix-vector benchmarks. SEALTorch uses the public
high-level API, native polynomial evaluation, and hoisted rotations rather
than importing example internals.

## 3. Run experiments

```bash
python -m pip install -r requirements.txt
python webui/server.py
```

Open <http://127.0.0.1:8080>. Experiments are built as explicit manual jobs.
Each job fixes its model artifact, approximation method, degree and interval,
GPU, plaintext/encrypted modes, sample selection, timing counts, activation
sampling, weight-cache policy, and all four CKKS parameters. Optional coefficient
arrays and exact dataset indices can also be supplied. The runner never calibrates
the interval, changes an HE parameter, or retries a failed context with a larger
depth or ring.

Add jobs in the form and inspect or edit the JSON batch contract before launch.
Jobs assigned to different GPU indices execute in parallel. Jobs assigned to the
same GPU execute sequentially in submission order, which prevents two contexts
from unexpectedly competing for that GPU's VRAM.

**Auto-select minimums + interval** fills the smallest architecture/degree CKKS
profile currently verified by SEALTorch. For Chebyshev it records deterministic
training activations and searches symmetric interval candidates for the lowest
activation-weighted RMSE. For Taylor it uses the measured absolute 99.9th
percentile because changing the basis interval does not change the polynomial.
The selected interval, HE parameters, and generated coefficients remain editable
and are not applied to jobs already present in the batch JSON.

The same contract can be submitted without the browser:

```bash
curl -X POST http://127.0.0.1:8080/api/manual-batches \
  -H 'Content-Type: application/json' \
  --data '{
    "label": "GELU degree study",
    "jobs": [{
      "name": "Chebyshev d2",
      "model": "mnist/mlp-gelu.json",
      "method": "chebyshev",
      "degree": 2,
      "range": 4.0,
      "gpu": 0,
      "run_plaintext": true,
      "run_encrypted": true,
      "samples_per_class": 100,
      "timed_runs": 1000,
      "warmup_runs": 5,
      "seed": 42,
      "weight_cache": "cpu",
      "activation_split": "train",
      "activation_samples_per_class": 500,
      "activation_sample_limit": 960000,
      "he_parameters": {
        "ring_dim": 16384,
        "depth": 9,
        "scaling_mod_bits": 25,
        "first_mod_bits": 35
      }
    }]
  }'
```

Encrypted jobs require every HE parameter and a valid GPU assignment. Plaintext
is also required when encrypted mode is enabled because the runner checks the
first encrypted logits against the matched polynomial model. A rejected or
under-provisioned CKKS configuration is saved as an explicit failed point.

Each run and the combined batch directory export `accuracy_data.csv`,
`approximation_error_data.csv`, `inference_time_data.csv`, and `memory_data.csv`.
These contain the graph dimensions and flattened model/GPU/CKKS metadata. The
accuracy table includes correct and total predictions plus percent and Wilson
intervals. Approximation error includes activation-weighted RMSE, observed
maximum absolute error, input counts, and range coverage. Timing includes
warmups, timed trials, batch size, quartiles, and encrypt/evaluate/decrypt
components. Memory includes idle, peak, and incremental RAM and VRAM in bytes
and GiB.

Runs also retain their exact manifest, environment, restart-safe point files,
raw sample timings and predictions in `samples.csv`, activation input/output
samples in `activation_samples.csv`, per-class logits, native FIDESlib responses,
coefficients, confusion matrices, and progress events. Combined outputs live in
`results/_sweeps/<batch-id>/`; child runs live directly under `results/<run-id>/`.

Long runs additionally checkpoint `status.json` throughout every phase and
append every progress update to `events.jsonl` and `events.csv`. The run records
its software/GPU environment, raw plaintext and ciphertext logits, native
FIDESlib responses, warmups, setup measurements, coefficients, confusion
matrices, per-sample timings, and per-point wall time. `logits.csv` exposes
per-class plaintext/ciphertext differences without requiring JSON parsing.

## Graph Lab

The **Graph Lab** dashboard tab can use a single run or combine selected runs.
Choose point summaries, long-form figure values, raw sample timings, per-class
logits, or the progress/ETA event timeline. Any numeric columns can be assigned
to the left or right Y axis, while categorical fields can control color, style,
filters, and facets. Matplotlib and Seaborn render line, scatter, bar, point,
box, violin, strip, swarm, histogram, KDE, ECDF, regression, area, and heatmap
plots with configurable themes, palettes, aggregation, error bars, logarithmic
scales, bounds, dimensions, DPI, and typography.

Every Graph Lab render is saved under `results/_graph_lab/<graph-id>/` with PNG
and SVG output, the exact filtered `data.csv`, a reusable `config.json`, library
version metadata, and a small `reproduce.py` script. Edit either the data or
configuration and rerun the script to reproduce a customized figure.

## Tests

```bash
python -m unittest discover -s tests -v
ctest --test-dir build --output-on-failure
```
