This evidence records the completed 2026-10-09 correctness/quality verdict. It does not contain speed
evidence. JSON summaries retain recorded metrics/statuses and source-summary digests; discarded fields
are timing and resource readings, not failed draws. Do not rewrite these files when strict timings finish.

Baseline runtime: `09b6be9724f330f947c86de6e819654b077f70f2`, library SHA256
`cb6de00388a1d85e32c8a8d1144310777d73767e1e56a8b32ab810e47d636e08`.
Selected runtime: `d0bb8c47789e062acbf6b770918a0589a74ae9b2`, library SHA256
`e67a7ad314a75cec7414ecdcfaa05895040191f50c86c960c432bce919d0a90c`.
The later `63a1fc93` source commit changes documentation only. Earlier seeds 1–8 used candidate
`f38de5c49d6adb4d80491ab7388807676767f59a`, with explicit precision recorded per draw.

Both final arms use the same CMake/Ninja recipe with `USE_CUDA=ON`, CUDA 13.0.88, architecture `120-real`,
`/usr/bin/c++` (GCC 15.2.0), Python 3.12.13, and the same bundled dependencies. Invocations clear
`PYTHONPATH`/`LD_LIBRARY_PATH`, use the built artifact's interpreter, and record its library digest and
resolved precision. The single RTX 5090 runs through GPUQ. Quality/functionality uses relaxed admission;
future strict timings require quiet admission and their own report.

The fixed benchmark cache is `/home/felixjk/Documents/exaboost-bench/data/cache`. Covtype training is
464,809 rows × 54 features; fraud is 227,845 × 29. Inputs are float32, while both fresh arms use FP64
histogram/gain arithmetic. The native benchmark's deep regime uses 500 rounds, learning rate 0.1,
1,023 leaves, depth 10, max bin 255, `quant_mode=none`, CUDA, 32 threads and runtime defaults for other
parameters; there is no aligned-L2 override. Seeds 9–16 were declared before scoring. Predictions use CPU.
Finite sanity-failed endpoints remain in mean/SD and paired summaries; missing/infra failures are reported
separately. Saved endpoints allow independent checking of the descriptive statistics.

Numerai's legacy cache is build1226: 6,753,184 rows × 3,555 features, training/test boundary 5,463,797,
200 test eras and metadata embargo 10 eras. Its label is unidentified. Numerai quality uses
`evaluation.evaluator.NumeraiEvaluator(cpu)` through the benchmark wrapper, never hand-computed corr or
Sharpe. This is fixed-cache implementation evidence, not a current target/production/prequential verdict.
Harness/evaluator source digests and full provenance are in [reproduction.json](reproduction.json).
