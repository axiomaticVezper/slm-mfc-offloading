# DA2_MFC Progress Report

**Project:** Multi-Agent SLM-Based Adaptive Task Offloading and Resource Negotiation in a Mist–Fog–Cloud Environment  
**Report date:** 9 October 2026  
**Repository:** [axiomaticVezper/slm-mfc-offloading](https://github.com/axiomaticVezper/slm-mfc-offloading)  
**Status:** Prototype implemented; latency optimization underway; final multi-seed evaluation pending.

## 1. Project objective

The project investigates whether small language models (SLMs) operating at different computing tiers can negotiate task placement in a Mist–Fog–Cloud environment. A SimPy discrete-event simulator models task execution, queues, network transfer, energy use, deadlines, and resource constraints. A deterministic guard is used to validate or override SLM recommendations when resource feasibility or fault conditions require it.

## 2. Current architecture

| Tier | Model | Intended role |
|---|---|---|
| Mist | `qwen2.5:0.5b` | Initial task-placement proposal at the constrained end device |
| Fog | `llama3.2:1b` | Evaluate the task at the edge and accept or forward it |
| Cloud | `llama3.1:8b` | Handle decisions requiring the largest model / centralized tier |

The implementation includes:

- `src/llm_agent.py` — Ollama calls, prompts, structured decision handling, and decision caching.
- `src/negotiation.py` — multi-tier negotiation flow.
- `src/guard.py` — deterministic resource-feasibility checks and overrides.
- `src/models.py` — task, node-state, and decision data models.
- `src/simulation.py` — SimPy task and resource simulation.
- `src/network.py` — network latency, bandwidth, and energy assumptions.
- `src/main.py` — policy/scenario runs and result generation.
- `src/profile_latency.py` — local profiling of live Ollama calls, including optional Cloud-model tests.
- `results/` — CSV outputs from the current 20-task, single-seed experiment.

## 3. Work completed so far

1. Implemented the Mist–Fog–Cloud SLM agents and hierarchical negotiation path.
2. Added a deterministic guard and fallback behavior so malformed or unavailable model responses do not automatically stop the simulation.
3. Implemented baseline policies for comparison, including local-only, greedy, static-threshold, and always-Fog placement.
4. Added normal, heavy-load, and fault-injection experiment variants.
5. Added latency profiling to separate model loading, prompt evaluation, generation, and total wall time.
6. Shortened the agent prompts and limited generated output. The updated agent configuration uses a 1,024-token context, an 80-token output cap, and a 30-minute keep-alive request.
7. Pushed the current code and experiment checkpoint to the repository's `main` branch. The latest push reported `c64347f..c846417 main -> main`.

## 4. Latency optimization results

The following values are from two local profiler runs shared on 9 October 2026. Warm wall-time averages below exclude the first call in each consecutive-call section where possible. They are measurements from this machine, not universal model benchmarks.

| Model / tier | Prompt tokens, before → after | Warm wall time, before → after | Approx. change |
|---|---:|---:|---:|
| Mist — Qwen 2.5 0.5B | 221 → 142 | 4.40 s → 2.49 s | 43% faster |
| Fog — Llama 3.2 1B | 206 → 131 | 5.00 s → 3.18 s | 36% faster |
| Cloud — Llama 3.1 8B | 191 → 110 | 9.25 s → 6.88 s | 26% faster |

Generation time also fell in the profiler:

- Mist: approximately 1.12 s to 0.56 s per warm call.
- Fog: approximately 1.97 s to 1.16 s per warm call.
- Cloud: approximately 6.72 s to 4.58 s per warm call.

The results support the conclusion that the shorter prompts and output cap reduced warm-call latency. They do **not** establish that the complete simulator's deadline or energy results have improved; that requires a new simulation run using the updated code and fresh decision caches.

## 5. Model-loading issue still under investigation

The updated profiler still shows loading penalties after switching from Cloud back to smaller models:

| Test | Previous profile | Updated profile |
|---|---:|---:|
| Cold Cloud call | 43.83 s | 32.10 s |
| Mist immediately after Cloud | 3.92 s load | 4.70 s load |
| Fog after Cloud | 7.27 s load | 6.10 s load |

The computer reports 16,167 MB of physical RAM, and all models in these tests run on CPU. A 30-minute keep-alive request alone has not prevented the observed model-loading penalties. Concurrent model residency should be tested before attributing the issue solely to prompt processing or CPU generation speed. `OLLAMA_MAX_LOADED_MODELS=3` is a possible experiment, not yet a verified fix for this setup.

## 6. Existing simulation results and interpretation

The current checked-in CSVs describe **20 tasks per configuration, one seed (`101`), and normal/heavy-load scenarios**. They are preliminary results and should not be presented as final multi-seed findings.

Selected values from `results/run_metrics.csv`:

| Scenario | Policy | Deadline hit rate | Average turnaround | Average energy | Memory violations |
|---|---|---:|---:|---:|---:|
| Normal | `local_only` | 15% | 19.03 s | 2.96 J | 15% |
| Normal | `greedy` | 100% | 1.38 s | 12.99 J | 0% |
| Normal | `static_threshold` | 100% | 1.36 s | 13.03 J | 0% |
| Normal | `always_fog` | 100% | 1.36 s | 8.48 J | 0% |
| Normal | `slm_noguard` | 100% | 1.39 s | 8.41 J | 0% |
| Normal | `slm_guard` | 100% | 1.31 s | 10.69 J | 0% |
| Normal | `slm_guard_charged` | 0% | 15.65 s | 8.41 J | 0% |
| Heavy load | `always_fog` | 100% | 1.47 s | 8.48 J | 5% |
| Heavy load | `slm_guard` | 100% | 1.41 s | 8.65 J | 5% |

### What these results suggest

- Local-only execution struggles with queueing and deadlines in the current workload.
- `slm_noguard` behaves similarly to `always_fog` in these runs, so the SLM policy has not yet demonstrated a clear advantage over that simple baseline.
- The guard changes tier selection and can help under injected faults, but Cloud overrides may increase energy when faults are absent.
- The guard does not eliminate heavy-load memory violations in the current implementation. Checking free memory at decision time may be insufficient when multiple tasks subsequently compete for the same capacity.
- In the `slm_guard_charged` experiment, accounting for live inference latency causes the deadline hit rate to fall to 0%. This highlights the importance of reporting both cache-assisted simulation outcomes and inference-aware outcomes.

Because these are small single-seed runs, the observations are preliminary rather than statistically conclusive.

## 7. Current limitations

1. Live SLM decisions take seconds, and Cloud calls can be much slower than the simulated task deadlines.
2. Switching between model tiers can cause additional model-load delays.
3. Decision caching can make repeated experiments faster, but cache-assisted timings should not be presented as if every decision were a fresh live inference.
4. The current guard checks resource availability but does not yet reserve memory atomically for tasks that are in transit or queued.
5. The current 20-task workload did not meaningfully stress Fog queueing; a larger workload is needed to assess saturation and placement behavior.
6. Results use one seed and need multi-seed validation.

## 8. Next steps

1. Test whether Ollama can keep all three models resident and rerun `src/profile_latency.py --cloud`.
2. Run a small end-to-end smoke test with fresh caches to validate the updated prompts and decisions.
3. Implement or evaluate memory reservation at decision time to prevent overcommitment under concurrent task arrivals.
4. Add a saturation/heavy-contention experiment with a larger workload, such as 60 tasks, to exercise Fog queues and resource pressure.
5. Compare cache-assisted and inference-charged results explicitly; document assumptions for how decision latency is included in simulation time.
6. Run multiple seeds, regenerate plots and CSVs, and update the report only after verifying the new results.
7. Replace the README's older 40-task benchmark claims with current, clearly labelled results.

## 9. Overall status

The prototype, baseline policies, guard, result collection, and latency profiler are in place. Prompt optimization has produced measurable reductions in warm local inference latency. The main outstanding work is to resolve or characterize model-loading overhead, make resource guarding robust under concurrency, and rerun a larger multi-seed benchmark before drawing final conclusions about the proposed approach.
