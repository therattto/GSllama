# Two GPUs from different vendors, one process

llama.cpp will happily load one model across a CUDA backend and a Vulkan backend
at the same time. It works. What follows is what we learned about making it fast,
most of which is about where work actually goes rather than how fast each card
is.

## The thing that surprised us most: `-ts` does not move the offloading

With `--n-cpu-moe` (`-ncmoe`), the expert weights of the first N layers stay in
host memory. They are not computed by the CPU. A GPU reads them across PCIe and
does the multiply, so the cost is **bus bandwidth, not arithmetic**.

Which GPU? In `ggml_backend_sched_backend_id_from_cur`, an operation whose
weights live in host memory is assigned by walking the GPU backends **in index
order** and taking the first that accepts the offload. That order is the order of
the names in `-dev`.

So with `-dev CUDA0,Vulkan0`, **every** layer that `-ncmoe` put in RAM is
streamed and computed by the CUDA device, no matter what `-ts` or `-ot` say.

This cost us a lot of time. Pre-swap, the CUDA card was the one on the x4 chipset
link, so all streaming went over the worst link in the machine while the card with
the CPU x16 slot sat at 9 to 13% utilization. The obvious fix, declaring the
high layers resident in RAM so the tensor split would hand them to the other
card, was measured and is **worse**: -20.8% prefill, -33.2% decode, 89 graph
splits instead of 51. It moved the layers without moving the offload, so it only
added crossings.

To change which card streams, you reverse `-dev`, not `-ts`. Note that `-ts` is
positional with respect to `-dev`, so it has to be rewritten at the same time.

## What `-ts` does control, which is the long-context ceiling

`-ts a/b` gives layers 0..a-1 to the first device and a..n-1 to the second. Since
`ncmoe` is always smaller than `a`, **every** layer on the second device is a
full-weight layer. Therefore:

- `ncmoe` lightens **only the first device**, about 950 MiB per layer here;
- the second device's peak depends **only on the second field of `-ts`**, about
  1030 MiB per layer, and `ncmoe` does not touch it.

Direct evidence: two configurations differing only in `ncmoe` (24 against 22) had
the **identical** peak on the second card, to the MiB. The cell with the lowest
`ncmoe` of the whole grid was the one that died at 180224 tokens, because it had
three more layers on the constrained card.

The practical consequence, and the reason this is written down: **to buy margin
at long context you lower the second field of `-ts` or lower `-ub`, never
`ncmoe`.** We spent two days attributing that ceiling to `-ub`. The wrong
conclusion was consistent with every measurement we had, because the grid driver
responded to a cell that failed to start by moving a layer to the other card and
retrying, so every low-`ncmoe` cell ended up on a split that survives loading and
does not survive long context.

A corollary worth stating: **an automatic fallback that shifts layers to relieve
a failure belongs nowhere near a bench.** It silently turns the parameter you are
measuring into a different one.

Also, the per-layer cost does not transfer between cards. Estimated at 950 MiB
from one card's peaks, it is over 1372 MiB on the other, which is why a split one
notch further does not start even with 1435 MiB apparently free.

## `-ub` and `-ncmoe` finance each other

On a model whose experts stream from host memory, a larger micro-batch amortises
the re-read of those weights, and it costs compute-buffer memory, which you buy
back by moving more layers to host memory. They buy and sell the same resource,
so they are the one documented exception to moving a single parameter at a time:
they have to be moved as a grid.

On a dense model that fits entirely in VRAM, none of that applies. There is no
re-read to amortise, and raising `-ub` only costs memory. Measured on a 27B dense
model in the same repository: raising it is a straight loss.

**Tunings do not transfer between models even on the same machine.** On our
sparse MoE the best split is roughly `ncmoe + 5`, dictated by which card receives
the layers that `-ncmoe` lightened. On the dense 27B the best split is nothing
like it, and for a completely different reason, below.

## The tensor split is a compute knob, not a memory knob

On the dense 27B, the inherited split gave most layers to the larger-VRAM AMD
card, which is what the VRAM numbers suggest. Measured, the opposite is right:
give the **NVIDIA** card more layers than its share of VRAM would justify,
because prefill is compute-limited and CUDA returns much more per layer here than
Vulkan does on the AMD card. Decode, which would favour the AMD card, flattens
out early anyway.

Result on a 117043-token prompt: prefill from 587.51 to 1039.13 t/s, **+76.9%**,
decode unchanged. Total time with 500 tokens generated: -17.4% at 9109 tokens,
-32.5% at 27674, -40.9% at 117043.

And the ceiling that stopped us going further was not the model. It was the
speculative draft head taking about a GiB on the CUDA device. Moving the draft
head to the other card (`-devd`) raised the usable split substantially. When a
split refuses to go further, check what else is sitting on that device before
concluding the model needs the memory.

## Where the time actually goes, in decode

Measured pre-swap, with a per-split profile. Warm decode cost 33.65 ms per token:

| part | time |
|---|---|
| 24 CPU splits for the host-resident experts | 12.52 ms (11.90 compute) |
| AMD card on Vulkan, one split, 3065 nodes | 11.84 ms |
| NVIDIA card on CUDA, 25 splits | 9.29 ms |
| split 0, embeddings and the PLE gather | 0.012 ms |

The "fixed term" of 22.16 ms that a fit over thread counts could not explain is
simply **the two GPUs**, 21.1 ms of it. It does not scale with threads because it
is not CPU work. We looked for it in the CPU, in the crossings, and in sampling,
for a long time, before profiling the splits.

Cost per graph node: **3.8 us on Vulkan against 1.5 us on CUDA**. That ratio is
the single most useful number we have for deciding what to do next on the AMD
side: the card pays per dispatch far more than it pays for the arithmetic, so the
lever is removing operations, not making them cheaper.

## The AMD card pays for barriers, and narrowing them does not help

The Vulkan decode split emits **827 pipeline barriers per token** at about 4.8 us
each, which is 3.9 ms, a third of that card's time.

We tried narrowing the access masks, including one variant that is not even
correct per spec. GPU time per graph came out **11.614 / 11.612 / 11.568 ms**
across the three, that is **0.4%**, with the barrier count stuck at 827 in all
three. The 4.8 us are not cache flushes decided by the masks, they are pipeline
drain. The only lever is emitting fewer of them.

We also checked whether the Vulkan equivalent of CUDA graphs would be worth
building. Host time on that card is **1.25 ms of 12.85, i.e. 9.7%**. Even
reducing it to zero is worth a quarter of what the barriers are worth. An earlier
note in this project claimed "almost half of that card's time is command
building" and was flatly wrong: it came from comparing a subset of perf-logger
lines against the split's wall clock.

## Vulkan against the native AMD backend

The question "why is Vulkan faster than ROCm here" has three answers at three
levels, and they are worth separating because only one of them is about speed.

1. **Availability.** At the production configuration ROCm **does not start**. It
   asks for the compute buffer as one contiguous 4199.73 MiB block where about
   1600 remain, and cannot stitch fragments. Vulkan asks for **more**, 4922.73
   MiB, and succeeds, because the kernel driver grants it host memory (GTT).
   Most of the apparent gap is not speed, it is whether the process exists.
2. **Phase.** At a micro-batch where both run, prefill is level (91 against 89
   t/s) and decode is 10% better on Vulkan (26.0 against 23.5).
3. **Operation.** On `MUL_MAT_ID` with the real quantization, Vulkan wins every
   shape from 1.4x to 5.4x, and **2.1x at n=1**, which is decode.

An earlier claim that the native backend costs 2.6 GiB of usable VRAM is false:
at rest it reports **more** free memory than Vulkan does.

## The hardware finding that beat all the software

On 12 September 2026 the two cards swapped slots. The NVIDIA card, which is first
in `-dev` and therefore does all the expert streaming, moved from the chipset x4
link to the CPU x16 slot.

At the production configuration: prefill on a short prompt from 306 to **900.18
t/s** (+194%), on a long prompt from 353 to **891.01** (+152%), decode **flat**
(-1.2% and +1.1%). Before the swap, prefill depended on prompt length; after it,
the two values coincide.

Two readings of that, both worth having:

- **Decode did not move at all.** At batch 1 the link is not the constraint, so
  the same hardware change that nearly tripled prefill is worth nothing to token
  generation. Whether a PCIe upgrade helps depends entirely on which phase you
  care about.
- **The prediction was written down first.** Prefill up, decode flat. That is the
  only reason the result is interpretable rather than a story told afterwards.

Three things moved with the cards and are easy to forget:

- The **monitor** followed the card, so the desktop session started consuming
  VRAM on the memory-constrained card and a split that used to work stopped
  starting. The fix is leaving the graphical session (the greeter alone is 222
  MiB against 1354) or moving the cable, **not** a lighter desktop: the compositor
  is only 394 of those 1350 MiB and the rest is applications.
- **Vulkan device enumeration changed**, so the launcher now pins visible devices
  explicitly and verifies the mapping before starting rather than trusting index
  order.
- **The whole tuning had to be redone from scratch.** It had been tuned against a
  link that no longer exists.
