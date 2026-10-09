Auxiliary Falcata reproduction — 2026-10-09

All nine required GPUQ jobs are terminal. Recorded 65/65 planned endpoints: seven graph pairs (56), six ingestion draws and three nightly cases.
Every recorded endpoint, failure, missing cell, actual parameter/config identity and GPUQ state is retained in [auxiliary-timing-evidence.json](auxiliary-timing-evidence.json).

Graph times are seconds: median [minimum, maximum] over three timed draws per arm, excluding warmups.
OFF/ON ratios require complete, correctly identified, sanity-passing timed models, a done strict job with no recorded contention, and observed ON-warmup graph capture.
They contrast the full graph and host routes within one selected build; they do not isolate these new ports or establish a quality improvement.

| Historical cell | OFF seconds | ON seconds | OFF/ON | Valid timed OFF/ON | ON capture observed | GPUQ state / contention |
|---|---:|---:|---:|---:|---|---|
| covtype-deep-quant | 0.718066 [0.708133, 0.718073] | 1.10606 [1.09599, 1.10943] | 0.6492× | 3/3, 3/3 | True | done / 0 |
| covtype-shallow-quant | 0.465946 [0.463871, 0.481836] | 0.55988 [0.537631, 0.571307] | 0.8322× | 3/3, 3/3 | True | done / 0 |
| year-shallow-quant | 0.240923 [0.237672, 0.248194] | 0.29081 [0.279893, 0.293632] | 0.8285× | 3/3, 3/3 | True | done / 0 |
| fraud-deep-quant | 0.245947 [0.236017, 0.246568] | 0.263157 [0.254302, 0.280155] | 0.9346× | 3/3, 3/3 | True | done / 0 |
| higgs-shallow-quant | 0.653181 [0.637889, 0.673651] | 0.680942 [0.677688, 0.684578] | 0.9592× | 3/3, 3/3 | True | done / 0 |
| epsilon-shallow-quant | 0.911617 [0.890473, 0.92173] | 1.30415 [1.28962, 1.30864] | 0.6990× | 3/3, 3/3 | True | done / 0 |
| numerai-example-quant | 1.33861 [1.3341, 1.34493] | 2.07833 [2.07202, 2.11302] | 0.6441× | 3/3, 3/3 | True | done / 0 |

Finite held-out endpoint metrics below include warmups and sanity-failed models. They are descriptive; seeds and configs are fixed reproductions, not fresh selection evidence.

| Cell / arm | Metric | Median [minimum, maximum] | Finite endpoints | Sanity failures | Other failures / missing |
|---|---|---:|---:|---:|---|
| covtype-deep-quant / off | accuracy | 0.918832 [0.918832, 0.918832] | 4 | 0 | {}, missing=0 |
| covtype-deep-quant / off | multiclass log loss | 0.226731 [0.226731, 0.226731] | 4 | 0 | {}, missing=0 |
| covtype-deep-quant / on | accuracy | 0.918832 [0.918832, 0.918832] | 4 | 0 | {}, missing=0 |
| covtype-deep-quant / on | multiclass log loss | 0.226731 [0.226731, 0.226731] | 4 | 0 | {}, missing=0 |
| covtype-shallow-quant / off | accuracy | 0.834479 [0.834479, 0.834479] | 4 | 0 | {}, missing=0 |
| covtype-shallow-quant / off | multiclass log loss | 0.396475 [0.396475, 0.396475] | 4 | 0 | {}, missing=0 |
| covtype-shallow-quant / on | accuracy | 0.834479 [0.834479, 0.834479] | 4 | 0 | {}, missing=0 |
| covtype-shallow-quant / on | multiclass log loss | 0.396475 [0.396475, 0.396475] | 4 | 0 | {}, missing=0 |
| year-shallow-quant / off | RMSE | 9.08857 [9.08857, 9.08857] | 4 | 0 | {}, missing=0 |
| year-shallow-quant / on | RMSE | 9.08857 [9.08857, 9.08857] | 4 | 0 | {}, missing=0 |
| fraud-deep-quant / off | AUC | 0.939115 [0.939115, 0.939115] | 4 | 0 | {}, missing=0 |
| fraud-deep-quant / on | AUC | 0.939115 [0.939115, 0.939115] | 4 | 0 | {}, missing=0 |
| higgs-shallow-quant / off | AUC | 0.812977 [0.812977, 0.812977] | 4 | 0 | {}, missing=0 |
| higgs-shallow-quant / on | AUC | 0.812977 [0.812977, 0.812977] | 4 | 0 | {}, missing=0 |
| epsilon-shallow-quant / off | AUC | 0.918857 [0.918857, 0.918857] | 4 | 0 | {}, missing=0 |
| epsilon-shallow-quant / on | AUC | 0.918857 [0.918857, 0.918857] | 4 | 0 | {}, missing=0 |
| numerai-example-quant / off | RMSE | 0.223651 [0.223651, 0.223651] | 4 | 0 | {}, missing=0 |
| numerai-example-quant / off | Numerai CORR mean (NumeraiEvaluator) | 0.0144002 [0.0144002, 0.0144002] | 4 | 0 | {}, missing=0 |
| numerai-example-quant / off | Numerai CORR std (NumeraiEvaluator) | 0.0149317 [0.0149317, 0.0149317] | 4 | 0 | {}, missing=0 |
| numerai-example-quant / off | Numerai CORR Sharpe (NumeraiEvaluator) | 0.9644 [0.9644, 0.9644] | 4 | 0 | {}, missing=0 |
| numerai-example-quant / off | Numerai max drawdown (NumeraiEvaluator) | -0.070085 [-0.070085, -0.070085] | 4 | 0 | {}, missing=0 |
| numerai-example-quant / on | RMSE | 0.223651 [0.223651, 0.223651] | 4 | 0 | {}, missing=0 |
| numerai-example-quant / on | Numerai CORR mean (NumeraiEvaluator) | 0.0144002 [0.0144002, 0.0144002] | 4 | 0 | {}, missing=0 |
| numerai-example-quant / on | Numerai CORR std (NumeraiEvaluator) | 0.0149317 [0.0149317, 0.0149317] | 4 | 0 | {}, missing=0 |
| numerai-example-quant / on | Numerai CORR Sharpe (NumeraiEvaluator) | 0.9644 [0.9644, 0.9644] | 4 | 0 | {}, missing=0 |
| numerai-example-quant / on | Numerai max drawdown (NumeraiEvaluator) | -0.070085 [-0.070085, -0.070085] | 4 | 0 | {}, missing=0 |

Ingestion reuses the original unquantized first-five-tree recipe: float32 then int8, three fresh processes per dtype. Native resolved mode/precision, full params and tree hashes are in the evidence.
These absolute readings have no held-out quality metric and no before/after optimization claim. Historical construction/RSS comparisons are harness/dtype-confounded.

| Input | Construct seconds | Create seconds | First five trees seconds | Peak RSS GiB | Valid draws | Failures / missing |
|---|---:|---:|---:|---:|---:|---|
| f32 | 31.34 [14.76, 34.74] | 1.34 [1.23, 1.39] | 0.17 [0.16, 0.2] | 64.9 [64.3, 88.2] | 3/3 | 0 / 0 |
| i8 | 3.89 [3.87, 14.7] | 0.76 [0.76, 0.76] | 0.16 [0.15, 0.16] | 32.3 [32, 36.2] | 3/3 | 0 / 0 |

The three nightly cells reuse the original adaptive repeats (3–15); throughput only, with no holdout scoring. Raw samples and recorded recipe identities are retained. No causal comparison with older nightly baselines is made.

| Nightly case | Construct seconds | Train seconds | Repeats | Strict timing eligible | Status |
|---|---:|---:|---:|---|---|
| bench/covtype-shallow-nonquant | 0.0635 | 0.454 | 4 | True | ok |
| bench/year-shallow-fp32 | 0.1526 | 0.1273 | 10 | True | ok |
| bench/numerai-nonquant-fp32 | 3.9175 | 3.2002 | 3 | True | ok |

Numerai's added graph holdout score uses NumeraiEvaluator(cpu); the historical graph ablation had no such score. The fixed build1226 cache has an unidentified label, so it supports implementation reproduction, not current-target, production or prequential model selection.
Capture is observed only where diagnostics were enabled in warmups; timed-run diagnostics were disabled. Missing or differing tree/prediction fingerprints are exposed in the evidence and do not become an identity claim.

Runtime: `d0bb8c47789e062acbf6b770918a0589a74ae9b2`; installed library SHA256: `e67a7ad314a75cec7414ecdcfaa05895040191f50c86c960c432bce919d0a90c`.
A missing endpoint remains missing. No failure was replaced or GPU job launched by this report generator.
