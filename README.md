# GSllama

A llama.cpp fork tuned for **two GPUs from different vendors in one process**:
an NVIDIA RTX 4080 Super (CUDA) and an AMD RX 7900 XTX (Vulkan), driven
together by a single `llama-server`.

Most of the work here came out of making that combination fast on a
memory-starved desktop, and measuring every step instead of guessing.

## Why this exists

Running one model across a CUDA card and a Vulkan card at the same time works
in llama.cpp, but almost nothing is written down about *how it behaves*. The
useful surprises were not in the model code, they were in memory accounting,
graph reuse, and PCIe topology.

The machine, for context, because every number below is tied to it:

| | |
|---|---|
| CPU / RAM | Ryzen 7 5800X3D, 32 GB DDR4-3600 (2 DIMM slots, hard cap 64 GB) |
| GPU 0 | RTX 4080 Super 16 GB, PCIe 4.0 x16 from the CPU |
| GPU 1 | RX 7900 XTX 24 GB, 8 GT/s x4 behind the chipset |
| Board | Gigabyte B550I AORUS PRO AX, Mini-ITX, one x16 slot |
| Model | Qwen3.8-Flash-Next (MoE) at UD-Q2_K_XL, 78.9 GB, 200k context |

A 79 GB model on 40 GB of VRAM and 32 GB of RAM. That constraint is what
produced most of the patches.

## Results

All figures below are the **same model, same quantization, same 200k context**,
measured with a 9109-token prompt on the machine in the table above. Median of
three warm runs; the first run is always discarded as warm-up. Run-to-run
dispersion on the swap measurement was 0.50% and 0.62%.

| stage | prefill | decode |
|---|---|---|
| early working config (`-ub 1024`, `-ts 27/21`) | 132 t/s | 25.2 t/s |
| after parameter tuning and the patches here | 306 t/s | 28.4 t/s |
| **after swapping the two cards** (NVIDIA into the CPU x16 slot) | 900 t/s | 28.1 t/s |
| **delivered configuration** | **920 t/s** | **29 t/s** |

**Prefill ended up 7x the starting point. Be clear about where that came from:**
about **2.3x was software** (tuning plus the patches in this repository) and
about **2.9x was one screwdriver**, moving the NVIDIA card out of the chipset
x4 link and into the CPU x16 slot. Decode barely moved throughout, +15% total,
because at batch 1 the link is not the constraint.

That split is the single most useful thing in this repository. On a machine
with an MoE model streaming experts from RAM, **prefill is bounded by the PCIe
link of whichever card computes the offloaded experts**, and no amount of code
will buy back a x4 slot.

Two findings that transfer to stock llama.cpp, no patches needed:

- **`-ub 2048` instead of the default `-ub 512` was worth 2.28x prefill**
  (108.75 to 247.52 t/s, measured at a 32k context) at no cost in layer
  placement: the larger ubatch needed no extra layers moved to CPU. The
  per-ubatch fixed cost dominates, so this is much larger than it looks.
- **`-ts` does not decide who computes CPU-resident experts.** The first
  backend named in `-dev` does. Getting this backwards silently sends all
  expert traffic across the wrong link.

The delivered configuration is validated at **198,168 tokens of context**,
5 needles out of 5 retrieved at every depth, with a sabotage cell that
correctly loses all of them.

## What is in here

Grouped by how general it is. **The first group is not specific to any model**
and is the part intended for upstream.

### Generic

- **`llama`: size the compute buffers for `-ckv` tokens, not the full context.**
  llama.cpp reserves a worst-case graph sized on the *configured* context. On
  this machine that buffer was worth **five times the KV cache**, and it is
  what decides whether a configuration loads at all. Adding a separate
  compute-window argument made a 200k-context setup fit where it previously
  did not.
- **`ggml-cuda`: keep the parallel-fork events per split.** Upstream reset the
  event map on every split, and on a two-backend machine the decode graph has 51
  alternating splits, so the fork survived only in the last CUDA split. With
  counters and an execution signature, so it is visible whether forks happen at
  all. In three and a half hours of production this fork reports **72,817 graph
  reuses and zero reallocations**.
- **`ggml-backend`: count graph reallocations instead of aborting.** Sixteen
  lines. An abort tells you it happened once; a counter tells you how often,
  which is the number you actually need.
- **`ggml-backend`: keep big single-row sources on one backend.** A scheduler
  gate (`GGML_SCHED_BIGSRC*`) that stops large single-row tensors from being
  copied across the PCIe boundary every step.
- **`ggml-cuda`: wait for VRAM instead of failing.** `GGML_CUDA_WAIT_MEM_MS`.
  A process that has just exited does not return its VRAM immediately, and the
  resulting OOM blames the wrong thing. See the notes repo, this one cost a
  full session to diagnose.

### Vulkan, on AMD

- **Fused `sigmoid` + `mul`**, and **`hc_combine`** for hyper-connections, with
  the two shaders. On the 7900 XTX the cost is dominated by *dispatches*, not
  arithmetic. Profiling showed 827 GPU pipeline flushes per token at roughly
  4.8 us each, while the host side accounted for only 9.7% of the time.
  Folding operations together removed 104 ops in one change and 68 barriers in
  another, for a few percent of decode each.

### Qwen3.8-Flash-Next specific

- Per-block top-k for the sparse indexer, recurrent conv state rollback, and
  MTP draft loading. These only apply to the `qwen4exp` architecture.

## Things that cost us time, so they may save you some

- **`lspci` at idle tells you the opposite of the truth** on Navi 31. The XTX
  reports `16GT/s x16` because the card presents itself as an internal switch,
  and an idle GeForce reports `2.5GT/s` because of power saving. Read the
  **upstream bridge**, not the endpoint.
- **Expert offload goes to the first backend named in `-dev`**, regardless of
  what `-ts` says. `-ts` splits layers; it does not decide who computes the
  CPU-resident experts.
- **Deliverability is decided by the startup graph reserve, not by
  steady-state VRAM.** A configuration that looks lightest in `nvidia-smi`
  during generation can have a reserve that does not fit. It is only visible
  with `-lv 5`.
- **Read GTT alongside VRAM.** On AMD, memory that "fits" may have quietly
  spilled to system memory over PCIe. VRAM alone looks fine while performance
  is halved.
- **PCIe topology dominated everything.** Moving the NVIDIA card from the
  chipset x4 link into the CPU x16 slot **nearly tripled prefill** on an
  otherwise identical configuration. No software change came close.

## Attribution, and what is *not* ours

- Base: [`unslothai/llama.cpp`](https://github.com/unslothai/llama.cpp), which
  tracks [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp).
- **`src/models/qwen4exp.cpp` is not our work.** It was written by **Daniel
  Han** (Unsloth) and added upstream on 27 August 2026, 1728 lines. Everything
  we did to that file is optimization on top of an architecture that already
  worked. Several commits in this branch are his and not ours, including the
  CUDA graph cache keyed by shape and the MTP draft-head loading. `git log`
  shows the author of every commit; where it does not say `therattto`, it is
  not ours.
- Everything else in llama.cpp belongs to its authors. This fork adds patches,
  it does not claim the project.

## Honest warnings

- **This is a fork, and forks go stale.** The base here is from late August
  2026. llama.cpp merges dozens of pull requests a week. If you are reading
  this much later, prefer upstream and treat this repository as a source of
  patches and notes, not as something to run.
- **The numbers are tied to the machine in the table above.** The *methods*
  generalize; the figures do not. Two GPUs on different links behave nothing
  like two identical cards.
- The generic patches are intended to be proposed upstream. If they land
  there, that is the version you want.

## License

MIT, same as llama.cpp upstream. See [LICENSE](LICENSE).

---

The upstream llama.cpp README is preserved here as
[README.llama.cpp.md](README.llama.cpp.md). For build instructions, supported
models, and everything not specific to this fork, read that one.
