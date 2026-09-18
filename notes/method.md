# How the numbers were taken

This is the file we would keep if we had to throw away the rest. The findings in
this repository are worth what the measurements behind them are worth, and most
of what we learned was about the measurements.

## Two questions, two instruments

**"Is it faster"** and **"is it still correct"** are different questions and
neither is evidence for the other. A change ships only if it passes both.

The trap that made this a rule: this model has a speculative draft head, and
draft acceptance looks like a correctness signal. It is not. The draft and the
target share the same sparse-attention indexer, so a systematic selection error
agrees with itself and acceptance stays high while retrieval quality falls. The
correctness bench retrieves facts planted at known depths in a long context
instead.

## Paired, alternated, and throw away the first run

On identical configuration, wall-clock throughput varied by **12%** between runs.
Any single pair of numbers could have proved anything, in either direction.

Three things fixed it:

- **Paired arms, alternated**, and compare medians rather than single runs.
- **Palindrome ordering, A B B A, not A B A B.** There is usually a monotone
  drift: six alternated runs on the same short prompt gave prefill 232.89,
  242.75, 251.11, 257.28, 261.06, rising every time whatever the arm, because the
  page cache was warming on 17 GiB of expert weights. Under a monotone drift the
  simple alternation is systematically unfair to whichever arm goes first.
- **Discard the first run.** Cold runs are not averaged in, they are dropped.

Where a lower-variance proxy exists, cite that first. Bytes read from the device
and wait time on the read path are far quieter than wall clock, and they are what
a reader can check.

## Write the prediction down before looking

Two opposite outcomes are both narratable afterwards as confirmations of
whatever you believed. The only defence we found is to write the predicted
direction and rough size **before** running, and keep it next to the number.

A prediction that misses is worth more than one that lands, if you then explain
why it missed. A fair share of the switches in this fork are off by default with
the measurement that killed them in the comment above, and several of those
comments record a prediction that was wrong.

## The seven ways we already got it wrong

Each of these happened. Each now has a mechanical check, because the ones that
depend on remembering do not survive.

**1. The test did not ask the question.** The first retrieval test posted a raw
document with no chat template and no `ignore_eos`. The model stopped after
**one token**, both arms scored zero, and skimmed quickly that reads as "no
regression". Same family: a 48k-token prompt passed to `curl` as an inline
argument never reached the server at all, because a single `execve` argument caps
at 128 KiB, and the log simply ended after the warm-ups with no error line.

*Check*: the judge validates `prompt_tokens` and `completion_tokens` **before**
looking at content, and reports three outcomes rather than two: pass, fail, and
**bench broken**. A prompt under 90% of the expected length, or a reply under ten
tokens, is a broken bench, which is a fault in the instrument and not a result.

**2. The bench measured a configuration that was never shipped.** The bench did
not set an environment variable that the launcher did, and used a different
speculative draft depth. In the first case a combination that **loses the Vulkan
device outright** could not surface from either instrument alone. In the second,
two full rounds of measurement described a configuration nobody runs.

*Check*: the command is **not retyped** in the bench. The launcher has a dry-run
mode that prints the exact `argv` and the value of every relevant environment
variable, and the bench inherits it. The deliberate differences are printed in
clear on every round. Adding an environment variable that changes behaviour
without adding it to that list is exactly how an untested configuration is born.

**3. One point tested on a dimension where the code branches.** A setting was
promoted on three paired rounds at a 9109-token prompt and died at 44% of a
27674-token one. Retrieval was validated at 27820 tokens of a 200704-token
window, that is, on 14% of it.

*Check*, and this is the one we would recommend to anyone: **every literal
constant in a dispatch condition is a mandatory test point, translated into the
parameter the user actually controls.** It is not a judgement call, it is a
translation, and the translation is the step people skip. A kernel that switches
path at `ncols > 24576` is not a test at 24576 columns, it is a test above and
below **98304 tokens of prompt**, because that is what those columns are in the
user's units.

**4. No positive control.** A test that would give the same answer if it were
blind has demonstrated nothing, and failure 1 is exactly that, read optimistically.

*Check*: a sabotage cell runs with every indexer score forced equal, so block
selection becomes independent of content. With it on, deep retrieval **must**
fail. If it does not, the bench prints invalid and refuses to judge the other
cells. The same logic works in reverse: in a backend comparison, the arm that
moved by -27% is what proved the probe could see a backend change at all, and
therefore that the zero measured on the other backend was a real zero rather
than a blind instrument.

**5. The comparison was contaminated.** Either by drift, as above, or by moving
two variables at once, where a collapse gets attributed to whichever one you were
interested in.

*Check*: one parameter at a time, with the table reprinted at every step. The
single documented exception is a pair of parameters that buy and sell the same
resource, which have to be moved as a grid because moving either alone is
strictly worse.

**6. The control arm was inherited rather than pinned.** The reference cell was
the empty string, meaning "whatever the launcher decides". The day a change was
promoted into the launcher, the reference and the treatment silently became **the
same configuration**, and the bench would have gone on printing two passing rows
while comparing a thing to itself.

*Check*: the reference cell **explicitly turns off** every variable the treatment
turns on. When a change ships, the line to update is the reference, not the
treatment. Rule 2 says the treatment must follow production; this says the
control must not.

**7. A new probe reintroduced a bug already fixed elsewhere.** A readiness probe
waited for a log line the server does not write. Every configuration timed out
after four minutes and was recorded as failing to start. The existing bench
already had the correct pattern.

*Check*: a new tool inherits its patterns from the working one, and before use
you confirm the pattern actually occurs in a log the program already produced.
That is a two-second `grep -c` on an old log. It applies to readiness patterns,
error patterns, and any `grep` whose result decides an outcome.

## The bench contaminates itself

Worth stating separately because it is not obvious. Our long-context retrieval
bench hid its needles in a haystack built from the project's own documents. Then
we wrote the results of each run into those same documents. Within a few days the
haystack contained the answers.

If a bench draws on a corpus you also write to, it will eventually grade an open
book.

## Five recurring principles

Counted by how often they appear as the cause in the full lesson log, not chosen
by taste.

1. **The bench must follow production, not resemble it.** Every knob the launcher
   sets and the bench does not is a piece of configuration that was never tested.
   At least once, that gap hid a fault instead of reporting a number.
2. **Read the guard, not the option.** What decides is the branch actually
   compiled or executed: the entry in the CMake cache, the `#if` on a version, the
   systemd unit that is really wanted, the `getenv() != nullptr`. Not the name of
   the setting.
3. **Look at GTT together with VRAM, and the reserve together with the steady
   state.** Memory "freed" that was in host memory to begin with was never freed,
   and the configuration that looks lightest while running can have a reserve that
   does not fit. See [memory-placement.md](memory-placement.md).
4. **Comparisons are paired and the first run is warm-up.**
5. **Write the prediction down before looking at the measurement.**

## One environment variable idiom, chosen and stated

A smaller thing that cost real time. Across a backend, some flags were read as
`getenv(name) != nullptr` and others as `atoi(getenv(name))`. So for some of them
setting the value to `0` **turns the feature on**, because presence is what counts.

A bench whose reference arm sets a flag to `0` to disable it then differs from
the treatment arm in two things at once, and the result is uninterpretable. Worse,
it is uninterpretable in a way that looks like a clean result.

Every switch this fork adds states its idiom in the comment above it, and unset
always means the previous behaviour, character for character.
