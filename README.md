## Build

```sh
cmake -S . -B build
cmake --build build
```

If SEAL is installed somewhere else, provide its package directory:

```sh
cmake -S . -B build \
  -DSEAL_DIR="$HOME/.local/seal/lib/cmake/SEAL-4.3"
cmake --build build
```
