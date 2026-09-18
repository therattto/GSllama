# Memory: VRAM, GTT, the reserve, and the page cache

Four different things present as "the model is running at half speed today", and
telling them apart is most of the work. This file is the decision tree.

## The compute buffer is the big number, not the KV cache

At 200704 tokens of configured context, on our sparse MoE:

| | size |
|---|---|
| KV cache | 909 MiB |
| compute buffer | **4922 MiB on one card plus 3989 on the other** |

Almost all of it is one `[n_kv x n_token]` f32 tensor from the sparse-attention
indexer, 1176 MiB per copy, with four copies live at once.

The part that catches people: **it grows with the context you configured, not
with the prompt you sent**, at roughly 0.0248 MiB per token of `-c` here. You pay
for the maximum context even when you never use it.

It is easy to go back and retune the layer split believing the KV cache is the
large item, and then never find the real margin.

To diagnose this on a model you do not know, turn on the allocator debug in
`ggml/src/ggml-alloc.c` (the define is there, commented out) and it prints the
live tensors every time the peak grows. That is faster and more reliable than
reading the graph construction code.

## The reserve decides shippability, not the steady state

These are two different numbers:

- the **worst-case graph reserve**, taken at load;
- the **steady-state** occupancy, after the first request replaces that reserve
  with a compute buffer sized for the actual prompt.

A configuration that cannot ship can look like the lightest of all if you measure
it on a short prompt. Measured on the constrained card, capacity 24560 MiB:

| configuration | weights | KV + states | compute reserve | free |
|---|---|---|---|---|
| `ub 2048` `ts 30/18`, shipped | 16913.90 | 795.38 | 4142.86 | 2708 |
| `ub 2048` `ts 29/19` | 17889.79 | 798.49 | 4236.59 | 1635 |
| `ub 3072` `ts 29/19` | 17889.79 | 798.49 | **7212.93** | **-1341** |

In steady state on a short prompt, `ub 3072` sits at 19929 MiB, **less** than the
21951 of the shipped configuration, because the real buffer for a 9109-token
prompt is only 1002.7 MiB. But at load it overruns by 2371 MiB into host memory.
The steady state promised and the reserve refused.

Read the reserve with the verbose load log. Do not infer shippability from what
`nvidia-smi` or the sysfs counters show while the server is answering.

Two numbers from that table worth keeping: the per-token cost of the micro-batch
was **1.187 MiB on the NVIDIA card and 2.906 on the AMD one**, a factor of 2.45,
even though the AMD card held fewer layers. A per-token model derived from one
card does not transfer to the other.

## GTT, or: the allocation that silently lands in host memory

On AMD, when device-local memory is short, the Vulkan allocation loop falls back
to host memory (GTT) **silently**. Nothing in the log says so. The model loads,
the server answers, and prefill runs at half speed.

We chased this for days as non-deterministic driver placement. It is not. Here is
the actual sequence:

1. At startup the worst-case graph is reserved, 8059 MiB at our context.
2. On the first request that reserve is freed.
3. **63 ms later** the real compute buffer is allocated, 1621 MiB in two pieces.
4. At that instant free VRAM according to the kernel is **162 MiB**, because the
   hand-back is asynchronous. Twenty ms later it is 1166, twenty more and it is
   2496.
5. So the 1021.8 MiB piece is born in GTT, and stays there for the life of the
   process.

What appeared to fix it was sending a prompt over 18000 tokens, which looked like
the driver migrating the buffer. It was not. A longer prompt grows `n_kv`, which
forces a **reallocation**, and by then the memory really was free.

The fix in this fork, on by default: before a device-local allocation above 256
MiB that does not fit in free VRAM, sample the kernel's used-VRAM counter every
20 ms and wait for it to stop growing, giving up after three stable samples.
`GGML_VK_WAIT_VRAM_MS=0` turns it off.

Result: GTT 25 MiB instead of 1047, first request 227/28.0 instead of 140/21.6,
steady state reached at the third request instead of after a long prompt, and the
steady state itself 3.7% better. Confirmed later on a six-arm palindrome: prefill
on a 9109-token prompt from 180.88 to 321.49 t/s.

**Diagnostic rule that came out of this:** GTT above 100 MiB is a fault to
investigate, not a condition to wait out. A healthy run here sits at 19 to 63
MiB.

## Removing an operation can cost memory

This one is counter-intuitive enough to be worth its own section.

The sparse-attention mask is built by filling selected cells and then adding the
original mask. That add allocates a third `[n_kv, n_batch]` tensor, 588 MiB at our
context. Writing the mask values directly into the selected cells removes the add
and, on paper, those 588 MiB.

Measured, the compute buffer **grows by 3593 MiB**.

The reason is that the add was the node at which a lifetime ended. Without it, the
chain from the fill to flash attention is all views, the allocator keeps the
parent alive to the end of the layer, and the fills of several sparse layers end
up resident simultaneously.

A related trap in the same family: `ggml_op_can_inplace()` lists the unary ops, so
a rectifier should be in place, and it was not. The allocator refuses in-place
reuse when the parent is a **view**, and a reshape immediately before had created
one. Applying the rectifier **before** the reshape instead of after is worth 526
MiB and, with speculation off, +9.5% prefill and +14.2% decode. Same arithmetic,
same numbers out, different order, different memory.

## Not all scratch is in the arena

A CUDA-pool allocation does not appear in the scheduler's reserve and grows with
the prompt. In our case a full sort inside the top-k path was allocating from the
pool: a configuration would start fine, run a short prompt fine, and die at 44% of
a long prefill with an out-of-memory inside the sort.

If your memory accounting balances and the process still dies on a longer prompt,
look for allocations that bypass the graph allocator.

## Page cache: the other reason prefill halves

Everything above is about where a buffer lives. This is a different failure with
the same symptom, and the way to tell them apart is to look at bytes read from the
device per request.

On long prompts, prefill had **two stable regimes at identical configuration**:
730 t/s with the disk idle, and 480 to 580 with 1.2 GiB read per request. Across
sixteen runs, VRAM and GTT were constant to the MiB, which is what excluded
placement. It was the page cache.

Inside the available cache, two things compete: about 19.6 GiB of expert weights
for the host-resident layers, which prefill re-reads once per chunk, and 27.5 GiB
of a hash table that is deliberately designed to live on disk, mapped lazily with
random-access advice. The generation phase is not involved.

Putting the expert weights in allocated memory instead of mapped files, so the
cache cannot reclaim them (and so the host buffer becomes pinned, which lets DMA
start directly instead of going through a staging copy), was worth:

- short prompt 756.0 to **922.6** t/s, **+22.0%**
- long prompt 730.1 to **875.1** t/s, **+19.8%**
- decode unchanged

It also removed the lottery: four warm starts out of four, against three out of
four. Against an unlucky start the gain is 71%.

Three things to state alongside that number, because it is the only change in the
project that carries a risk of the process being killed:

- **`Cached` in `/proc/meminfo` includes `Shmem`.** With the experts in shared
  memory, the real file cache is `Cached` minus those 21.9 GiB, which was **4.5
  GiB**, one sixth of the hash table, and prefill still ran at 876. So in prefill
  that table does **not** need to be cached; only the experts did. The threshold
  is around 1 GiB of file cache, not 23.
- **The margin is 2.5 GiB**, measured with balloons at 0, 2.5, 3 and 3.5 GiB. At
  2.5 the long prompt pays 10%; at 3 it collapses by 35%, bimodally, 875 or 565
  and nothing in between; at 3.5 both collapse. The **short** prompt loses
  reproducibility before it loses speed, so noisy measurements are the early
  symptom of memory pressure.
- **It changes how the machine breaks.** Those 21.9 GiB are anonymous memory: they
  cannot be dropped, only swapped, and at 4 GiB of pressure the OOM killer
  arrives. At 3.5 it kills nobody and at 3 performance was already -35%, so the
  danger is visible before it hits. We deliberately did **not** protect the server
  with an OOM score adjustment: it already has the highest score, and making it
  immune would only redirect the killer at the remote-access path, which is the
  way back in.

### The claim this replaced

For about a week this project stated that more RAM was worth "about 2% of decode
and zero on prefill". That was measured honestly: a major page fault costs 48.3 us
paired, and in steady state there are 14 to 27 faults per token, 0.71 ms of 34.98.

It is still wrong, because it was taken inside one cache regime and quoted as if it
applied across the regime boundary. The real answer is the two regimes above.
Extrapolating a slope across a boundary you have not tested for is, in our
experience, the single most productive source of confident wrong numbers.

## Diagnostics that lie

Three instruments that gave us the opposite of the truth.

**PCIe link speed at rest.** `lspci` on an idle machine showed our NVIDIA card at
2.5 GT/s (power saving) and the AMD card at 16 GT/s x16. Read straight, that says
the AMD card has the better link. It is exactly backwards: the Navi 31 cards
present an internal bridge and report it as an endpoint, so what you are reading is
the card's own switch, not the slot. **Look at the bridge upstream of the device,
not at the endpoint**, and look at it under load.

**A dead process has not freed the GPU.** Three consecutive start failures with
out-of-memory, on a configuration that had just run eighteen times, turned out to
be an orphaned server holding 8940 MiB. The bench did check, but it waited for
process death plus two seconds, and that is not when the driver hands the memory
back. Poll the used-memory counters until they drop below a threshold, and call it
before every configuration and from an exit trap. Read the **two lines above** an
allocation failure as well: one that looked like the model weights was, on the
previous line, the draft head.

**`Cached` includes `Shmem`**, as above. A cache figure that looks healthy can be
entirely your own anonymous memory.
