# MLA absorption trick


This repository contains code for benchmarking different MLA implementations. The code accompanies the blog post: [Something](https://something).

The core benchmarking code was taken from [here](https://github.com/madsys-dev/deepseekv2-profile/tree/924174cb5dc11fad24bdaad3fd820ebf87506368) but we simplified the model classes.

## Overview

This repo demonstrates three different variants of MLA computation:

1. **SimpleAttention**: MLA with uncompressed KV cache.
2. **SimpleCompressedAttention**: MLA with compressed KV cache.
3. **SimpleAbsorbedAttention**: MLA with absorption trick.

To run a model, use the following command:

```bash
python3 mla/benchmark.py --bench SimpleAttention --kv_len 1000 --bsz=32 --config=mla/config.json --min_run_time=2.0
```

To run all models, use the following command:

```bash
python3 benchmark_run.py
```

To plot the results, use the following command:

```bash
python3 plot_results.py
```

