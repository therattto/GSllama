# Notes

Working notes from running one large MoE model across an NVIDIA and an AMD GPU
in the same llama.cpp process, on a desktop with less RAM than the model has
weights. They are the reasoning behind the patches in this fork, written down so
the patches can be argued with rather than trusted.

They are not a tutorial. Most of what is here is the shape of a problem and the
measurement that settled it, including the measurements that killed ideas we
liked.

| file | what is in it |
|---|---|
| [method.md](method.md) | how the numbers were taken, and the ways we got them wrong first |
| [two-gpus.md](two-gpus.md) | scheduling one model across two vendors' backends |
| [memory-placement.md](memory-placement.md) | VRAM, GTT, the graph reserve, and the page cache |
| [dead-ends.md](dead-ends.md) | things that were tried and closed, with the number that closed them |

## The machine

Everything here was measured on one desktop. That is a limitation and it is
stated up front, because a single machine cannot tell you which findings are
general.

- Ryzen 7 5800X3D, 8 cores, **31.7 GiB of RAM**. The board is Mini-ITX with two
  DIMM slots and a 64 GB ceiling.
- **RTX 4080 Super, 16 GB**, on CUDA.
- **RX 7900 XTX, 24 GB**, on Vulkan (RADV).
- One PCIe x16 slot wired to the CPU. The other card sits behind the chipset on
  a x4 link. There is no configuration of this board where both cards are well
  connected.
- The model is Qwen3.8-Flash-Next, a sparse MoE, quantized to about 2.06 bits
  per weight, **78.9 GB on disk** against 40 GB of total VRAM and 31.7 GiB of
  RAM. Nothing fits anywhere, which is the whole reason this fork exists.

## How to read the numbers

**Every number here has a date, and most have a topology.** On 12 September 2026
the two cards swapped slots: the NVIDIA card moved from the chipset x4 link to
the CPU x16 slot. That single change was worth more than every software change
in this repository put together, and it invalidated a large part of the earlier
measurements. Where a finding predates it the note says so, because the same
experiment run today can give a different answer.

The convention:

- **pre-swap** means the NVIDIA card was on PCIe 3.0 x4 behind the chipset and
  the AMD card had the CPU x16 slot.
- **post-swap** means the reverse, which is the current and final arrangement.

A number with no marker is one where the topology does not enter, for instance a
count of graph nodes or a shader dispatch count.

## What travels and what does not

The absolute numbers do not travel. They are one board, one pair of cards, one
model, one quantization.

What we think does travel is smaller and duller:

- the **mechanisms**, for instance why weights declared resident in host memory
  are computed by one particular device and not the one the tensor split names;
- the **diagnostics**, for instance which counter to look at to tell a memory
  placement problem from a page cache problem, when both present as "prefill is
  half speed today";
- the **failure modes of the measurements themselves**, which is what
  [method.md](method.md) is about and is probably the most portable thing here.

## Honesty about provenance

These notes are a distillation. The originals are a much larger set of working
documents in Italian, written as the work happened, and they contain claims that
were later refuted. Where a finding was reversed, the reversal is what is written
here, not the original. Two examples, so the pattern is visible:

- "More RAM is worth about 2% of decode" was measured, written down, and is
  **wrong**. It was taken in one page cache regime and applied to another. The
  corrected figure is in [memory-placement.md](memory-placement.md).
- "Almost half of the AMD card's time is command building on the host" was
  written down and is **wrong**. It came from a log parser that was reading half
  the lines it should have. The host is under 10%.

Both survived for a while because they were plausible and nobody re-measured
them. That is the normal way this goes.
