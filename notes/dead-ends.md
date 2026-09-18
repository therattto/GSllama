# Dead ends

Things that were tried and closed, each with the number that closed it. This is
the file we would have wanted before starting, and the one that is almost never
written, because a negative result is nobody's favourite thing to publish.

Several of these are ideas we were confident about. Two of them are patches that
exist in this repository, work correctly, do exactly what they claim, and are
**off by default** because measuring them said no.

Unless marked otherwise these are on the sparse MoE model. Where the PCIe
topology matters, the note says pre-swap or post-swap (see
[README.md](README.md)).

## Patches that work and are switched off

**Fusing the hyper-connection recombination chain.** Replaces five consecutive
graph nodes with one shader. It removes **132 dispatches and 68 barriers per
token, and 12.5% of the AMD card's GPU time**. The token rate does not move:
decode **+0.3%**, inside the noise, while prefill **loses 2.5%** in three paired
comparisons out of three.

This is the most instructive result in the repository. Taking measurable work
away from the GPU bought nothing, because that work was already overlapping with
something else. A dispatch count is not a time.

**Pulling the gate multiply next to its sigmoid so the pair fuses.** Worth
**1700 fewer dispatches per 100 graphs and zero time**, because the sigmoids that
the reordering separates cost 1.44 us each instead of 5.98. They were already
free, for the same overlap reason.

The one in this family that did pay is smaller and duller: folding three
identical scale constants into the norm weight, which is a parameter tensor
rather than an activation. That removes **104 operations** and is worth **+2.8%
decode**. It is on by default.

## Vulkan and the AMD card

**Narrowing the barrier access masks.** 0.4% across three variants, one of which
is not even correct per spec, with the barrier count unchanged. Detail in
[two-gpus.md](two-gpus.md).

**Building the Vulkan equivalent of CUDA graphs.** Host time on that card is
9.7% of the split, so the entire ceiling is a quarter of what the barriers cost.
Not worth building.

**`RADV_PERFTEST=nogttspill`.** Was worth up to **+86% prefill**, and is now
harmful. It was the remedy for a bad VRAM placement produced by the per-cell
indexer graph. With the per-block top-k the placement comes out right on its own,
and all that is left of the variable is the damage: it removes the host-memory
escape valve. A fix for a problem you have since fixed can become a bug.

**The native AMD backend (ROCm) on this card.** Does not start at the production
configuration, and where both run it is 2.1x slower on the operation that matters
at batch 1. Detail in [two-gpus.md](two-gpus.md).

## CPU side

**Repacking expert weights for the CPU, including writing the missing path for
our quantization.** Not merely unprofitable, **harmful**. With the weights in a
repack buffer, `supports_op` makes the CPU backend say yes, the scheduler assigns
it the expert matmul, and **the GPU offload disappears**. Measured: default
323.24 prefill, with repack disabled 308.80, with repack active but host buffers
off **207.66, i.e. -35.8%**. Decode moves by under 2% across all three.

**Rewriting the CPU vector dot products.** Closed on arithmetic before spending a
bench. At 8 threads the CPU side reads 0.442 GB of weights per token at 33.2 GB/s
effective, which is the memory bandwidth ceiling **with the cards' DMA running**
(44.57 measured idle, 33 under load). The thread curve is flat from 8 to 12 for
that reason, and at 2 threads decode is 38% worse, because there you really are
compute-bound. Beyond 8 threads the kernel is no longer the constraint. Also, at
batch 1 the repack table is never consulted at all: the expert matmul calls the
vector dot directly.

**More threads.** 8 is the optimum here and the curve is flat above it, for the
reason just given.

## Memory and context

**`-ub 3072`.** Worth +6.6% prefill on a short prompt, +0.2% on a long one, zero
on decode, and it has no workable layer split: one overruns the constrained card
by 1341 MiB, the next does not let the other card start. See
[memory-placement.md](memory-placement.md).

**`-ub 8192`.** Closed on arithmetic before spending a bench. The indexer's score
matrix scales with `n_kv * n_tokens * 4`, which is 3.2 MiB per token of
micro-batch in f16, so the largest thing the engine will actually build is not
today's reserve but the runtime at near-full context: at `ub 8192` that is 24.6
GiB against the 24 the card has. Trimming the reserve moves *when* you pay, not
*how much*. Falsifiable prediction for anyone who wants the number anyway: `ub
8192` runs at 27674 tokens and goes out of memory at 187798. Wanting a large
micro-batch means first removing the materialization of that score matrix.

**`-ncmoe` above 28** on a 32 GB machine: 28.5 GB of experts in RAM out of 30
available, the page cache cannot hold them, and every measurement comes out cold.

**`-ncmoe` below 14**, even after freeing arena space. Either it starts on a
short prompt and dies on a long one, or it leaves the first card with 441 MiB
free and costs -17.3% prefill.

**Trimming the worst-case graph reserve (`--ctx-size-reserve`).** It works and
frees 6.5 GiB, and it is worth **1.5%**, because that memory was in host memory
rather than VRAM. It also regresses with the speculative head on. Off by default.

**More RAM as a speed upgrade.** Superseded twice, and the current answer is in
[memory-placement.md](memory-placement.md). Short version: it is not a percentage
of its own, it is what makes a +22% that already exists safe to keep.

## Model and quantization

**A second lossless compression layer on the weights.** Measured entropy **7.948
bits out of 8**, so a theoretical ceiling of 3.2%, with decode more expensive
than the saving. The underlying reason is that quantization already *is* that
scheme: an offline compressor, a 2.06-bit format, and a decoder on the GPU.

**Speculative decoding as an I/O reduction.** With 6 experts active out of 256,
two adjacent tokens share **1.2%** of their experts. Verifying K tokens costs
close to K reads and accepts fewer than K. Cost per accepted token gets **worse**.

**History-based expert prefetch between layers.** Measured coverage **exactly
zero**. Where an LRU cache already exists, a recency-based prefetcher cannot add
anything by definition: its hits would already be cache hits, and the misses are
precisely what recency does not predict. Twenty lines of C and seventy of Python
established this before anyone wrote the prefetcher.

**A RAM cache as a second level under the VRAM cache.** At 4 GiB, that is 606
slots against the VRAM level's 1611, it produced **zero exact hits**. It filled
on the same misses and lost entries before the first level did. The effective
capacity of two inclusive levels is that of the larger, not the sum.

**2-bit degeneration as a quality discriminator.** Zero literal loops, zero
out-of-alphabet characters, zero language drift. On that axis the heavily
quantized sparse model and the lightly quantized dense one are identical, so the
test discriminates nothing.

## Things the code was already doing

Worth a section because each of these was proposed, designed, and then found to
exist.

**`madvise(MADV_RANDOM)` on the large lazy-mapped table.** Already applied to
every lazy interval.

**Optimizing the tap loop of a convolution.** The metadata says it runs on **one
layer out of 48**: nine microseconds out of thirty-three milliseconds.

**Prefetching rows of that table during prefill, using indices known on the host
before the graph runs.** Re-sending the **same** prompt, which is the best
possible case for any prefetch, gives indistinguishable prefill. During prefill
those page faults are not on the critical path.

**Sampling cost over a 248320-entry vocabulary as the explanation for the fixed
term in decode.** The bench does not sample and the server does, and the bench
that does not sample is **6.1% slower**. It costs zero or less.

**Turning CUDA graphs off** to see what they are worth: 7.3% of decode, so they
stay. Note the flag is read with `getenv(...) != nullptr`, so setting it to `0`
also disables them. That idiom trap is in [method.md](method.md).

## Hardware

**A better slot for both cards at once.** There is not one. The board is Mini-ITX
with a single CPU-connected x16, and on this CPU socket no micro-ATX board splits
two x16 slots off the CPU as x8/x8. What was available was choosing **which** card
gets the good slot, and that turned out to be worth more than everything else
combined. See [two-gpus.md](two-gpus.md).

**A second SSD.** The first one is idle a third of the time.

**A second GPU, as the fix for slow prefill** (this predates the current pair).
The GPU was at 20% during prefill. Before adding hardware, measure whether what
you have is saturated.
