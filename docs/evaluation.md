# Reproducible local model evaluation

The benchmark runs **the actual TypeScript `Engine` used by LocalJev**, against
either backend: a local OpenAI-compatible inference server (oMLX), or the
on-device Apple Foundation Model, for which the runner starts and stops its own
`fm serve` per model. No Python, hosted inference, LLM-as-judge, or synthetic gold
labels are used. It does not require the LocalJev HTTP server to be running.

## What it answers

Which installed model — and which answer mode — offers the best quality/latency
tradeoff? How does that change when the same evidence is buried in a longer input?
Both modes are still prompted: `probability` asks the model to report numbers,
`vote` counts constrained label samples. This is not a benchmark of direct logits,
single-pass reads, TypeSafe Jev, or OpenJev's patched-vLLM backend.

The default matrix in [`eval/default.json`](../eval/default.json) is:

- Models: Gemma 4 E2B, E4B, 26B-A4B; Qwen3.6-35B-A3B; DiffusionGemma-26B-A4B.
- All use the installed 4-bit checkpoints. Quantization recipes are **not identical**.
- 40 examples per task, balanced over gold classes: 120 examples total.
- Context: 0 or 2,048 **additional background words**, with the full target retained.
- 5 models × 120 examples × 2 input conditions = **1,200 measured requests**.
- One question per request, one request in flight. Two excluded warm-ups per model.
- Temperature 0, thinking disabled, 256 maximum output tokens, up to 2 corrective
  retries (same behavior as LocalJev), 90-second timeout per upstream attempt.
  `config.temperature` applies to `probability` mode only: a `vote` model samples
  at its own `voteTemperature` (default 1), because identical samples would make
  every vote unanimous. The report names the effective temperature per model, and
  the manifest records it under `effectiveSettings`.
- No 12B model is included: it was not installed/available in this setup.
- Ornith is not included: this experiment compares the five explicitly selected models.

A pilot with `--limit 3` exercises one example per task, **not a meaningful evaluation**.
Increase samples for a stronger result before making production choices.

### Selecting models and answer modes

Each entry of `models` is either a bare string — an oMLX model id on the `openai`
backend, in `probability` mode — or an object:

| Field | Required | Meaning |
|---|---|---|
| `backend` | yes | `"openai"` or `"apple"` |
| `model` | `openai` only | Upstream model id. The `apple` backend defaults to `"system"` |
| `answerMode` | no | `"probability"` (default) or `"vote"` |
| `voteSamples` | `vote` only | Samples per decision; default 5 |
| `voteTemperature` | `vote` only | Sampling temperature; default 1 |

Unknown keys are rejected rather than ignored, so a typo fails the run instead of
quietly measuring something else. `voteSamples` and `voteTemperature` are only
accepted alongside `answerMode: "vote"`. Labels are the model id for a string,
`apple:<model>` for the apple backend, and a `:voteN` suffix in vote mode — so
`{ "backend": "apple", "answerMode": "vote", "voteSamples": 5 }` is reported as
`apple:system:vote5` and can appear in the same config as its probability twin.

[`eval/apple.json`](../eval/apple.json) does exactly that, comparing both modes of
the on-device model over the same examples.

## Gold datasets and provenance

| Task | Source/split | LocalJev type | Gold mapping | Default sample |
|---|---|---|---|---|
| News topic | [AG News](https://huggingface.co/datasets/fancyzhx/ag_news), test (7,600 rows) | `choice` | World / Sports / Business / Sci-Tech → 0–3 | 10 per class |
| Passage-based yes/no | [BoolQ](https://huggingface.co/datasets/google/boolq), validation (3,270 rows) | `noul` | `answer: false/true` → 0/1 | 20 per class |
| Fine-grained sentiment | [SST-5](https://huggingface.co/datasets/SetFit/sst5), test (2,210 rows) | `score` | Very negative / Negative / Neutral / Positive / Very positive → 0–4 | 8 per class |

These cover three different useful skills: routing/category selection, grounded
reading, and an ordered rubric. They are deliberately small and broad, rather than
claiming to represent a particular business workflow. Public datasets may have
appeared in pretraining. The split is held out **for this evaluation**, not proven
unseen by the models.

Sources are downloaded from pinned Hugging Face commits; every file is checked
against a fixed SHA-256 in [`scripts/eval/data.ts`](../scripts/eval/data.ts).
Changed/corrupt bytes fail closed, including in the local cache. Parquet is decoded
in Bun using dev-only `hyparquet` dependencies. No dataset credential is needed.

Selection is reproducible: rank row IDs by SHA-256 of seed + ID, select equally within
each class, then deterministically shuffle/interleave tasks. Choice option order is
also shuffled per example and held fixed across models/contexts. Source row IDs,
gold labels, selected-input hashes, source commits and checksums go into the run
manifest. Gold labels, dataset names and row IDs are **never sent to the model**.

### Dataset licenses

Dataset text is downloaded to ignored `.eval-cache/`, **not vendored or re-licensed
under this repository's MIT license**:

- BoolQ's card specifies [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
  Source/paper: [Clark et al., 2019](https://arxiv.org/abs/1905.10044).
- AG News's card marks the license unknown and describes research/non-commercial
  use. See the [source card](https://huggingface.co/datasets/fancyzhx/ag_news).
  Paper: Zhang, Zhao and LeCun, *Character-level Convolutional Networks for Text
  Classification*, 2015. Do not assume unrestricted commercial redistribution.
- SST-5 consists of movie-review text. The SetFit mirror does not state a license;
  review the [Stanford source](https://nlp.stanford.edu/sentiment/) and upstream terms.
  Paper: Socher et al., *Recursive Deep Models for Semantic Compositionality Over a
  Sentiment Treebank*, 2013.

Published result artifacts contain IDs, labels, predictions and measurements, not
review/news/Wikipedia passages. All original text stays in the ignored cache.

## Context-length dimension

**Maximum context capacity is not input length.** Changing a runner's limit from
128K to 256K while sending the same short prompt is not the experiment we want.
The benchmark does not modify any oMLX context, cache or model settings.

Every state has a clearly marked `TARGET` section. Each question explicitly asks
about that section only. The original text is never truncated. In the longer
condition, half of 2,048 background words precede the target and half follow it.
Background is deterministically assembled from *BoolQ training passages*, with all
training labels/questions stripped and exact matches to selected target text
excluded. This is the same context across models; actual tokenizer counts may differ.

This is a **distraction plus prefill-length stress test**, not a genuine new
long-document benchmark. It preserves the gold answer, but tests whether a model
can focus on evidence among unrelated material. It does not test larger relevant
contexts, long reasoning chains, or target-position variation.

`backgroundWords` can be changed to `[0, 512, 2048, 8192]` in a copied config. These
are word counts, not token budgets. Actual full prompt tokens (including system
instructions, questions, schemas and chat templates) are reported per cell.
Avoid changing the target's content or truncating it just to hit a token count:
that could invalidate its label.

## Timing and cache policy

- `elapsedMs` measures a full `Engine.decide`: upstream request(s), JSON decode,
  validation, normalization and corrective retries. There is no extra LocalJev HTTP
  hop; the oMLX HTTP hop is included.
- Calls are non-streaming: **TTFT is not measured** and must not be inferred from
  the reported full-decision time or `usage.total_time`.
- Excluded warm-ups use a separate artificial, unlabeled smoke example. The first
  may include weight loading, eviction and grammar compilation; it is reported but
  is **not an isolated cold-start measurement**.
- Model order is fixed to avoid frequent loads. Context order alternates per example;
  tasks are interleaved. One machine/run can still be affected by thermal drift,
  other applications and background inference. Use AC power and keep other inference
  clients idle.
- The default `bust-prefix` policy prepends a unique, explicitly irrelevant request
  reference to the first system message, before any cacheable shared prefix. This
  makes input-prefill comparisons less dependent on previous requests. A new run
  gets a new nonce; the same nonce/reference is used across models. The cache is not
  globally cleared. Reported cached tokens should be near zero; inspect the report.
- `shared-prefix` removes this extra reference and measures the normal LocalJev
  request behavior with the runner's existing caching. It is a **different experiment**;
  cache state is not reset or guaranteed. Do not pool the two policies.
- Cache-busting adds a small prompt perturbation. Temperature 0 and a fixed seed do
  not guarantee bit-for-bit determinism across kernels, runtime versions, or runs.
- oMLX may enforce schemas through a grammar for ordinary AR models while falling
  back to prompting for diffusion; `fm serve` applies its own constrained decoding.
  Thus the requests share a logical schema, but their actual prompts/token counts
  and enforcement can differ. This is a realistic **model + runner + LocalJev**
  comparison, not an isolated architecture experiment.
- Cache behaviour is an oMLX notion. In the recorded apple runs `fm serve`
  reported neither `prompt_tokens_details.cached_tokens` nor `usage.total_time`,
  so the cached-input column is 0 and backend time is blank for apple rows
  whatever `cacheMode` says. `bust-prefix` still perturbs its prompts identically,
  which keeps the two backends comparable on wall time and token counts.

Each attempt records wall time, status, prompt/output/cached tokens, backend
`total_time` when present, finish reason, reasoning-output length, warning headers
and request/response hashes, plus the vote `sample` index and corrective `retry`
index the Engine labelled the request with. Those two labels are what keep a vote
model's N samples from being counted as N-1 retries; results recorded before the
labels existed are still read, with every request past the first treated as a
retry. No credential headers or raw prompts/completions are saved. Whatever distribution the production Engine returns is normalized before
being scored: the model's **self-reported** numbers in `probability` mode, and
**sample frequencies quantized to 1/K** in `vote` mode. These are different kinds
of estimate; compare the modes as modes, not as two calibrated numbers on one
scale. Retries' token usage and time are included; in vote mode one decision
issues K requests, so its latency and token counts cover all of them.

## Run (fish, bash, or zsh)

1. For `openai` models, load/install the selected checkpoints in oMLX. The runner
   preflights the model list and refuses missing model IDs; it will not silently
   skip a model. For `apple` models nothing has to be running: the runner starts
   its own `fm serve` per model and stops it again, but macOS 27 with Apple
   Intelligence enabled and an accepted `fm license` is required.
2. Configure your key in the ignored `.env` (or your environment), as for LocalJev.
   The apple backend needs no key. `LOCALJEV_UPSTREAM` is only consulted for
   `openai` models; apple models always use their own managed server.
3. Install dependencies and run:

```sh
bun install --frozen-lockfile
bun test
bun run typecheck

# Pilot: 30 measured calls; one example per task, not balanced at this size.
bun run eval --out eval/runs/pilot --limit 3

# Default full matrix: 1,200 measured calls plus warm-ups.
bun run eval --out eval/runs/my-bakeoff
```

No LocalJev server is required. Data is downloaded once and verified on every run.
A new output directory is required for each new run; existing results are never
overwritten implicitly. Five consecutive inference failures stop the run rather
than flooding a broken server.

### Resume an interrupted run

Ctrl-C finishes the in-flight request, writes it, generates a partial report, and
stops. This is a "complete and save the current request, then stop" policy, not a
guarantee of immediate cancellation: the signal aborts an `fm serve` that is still
starting, but a request already sent upstream runs to completion so its result is
not lost. Resume skips already completed request keys (including recorded failures):

```sh
bun run eval --resume eval/runs/my-bakeoff
```

Use the same `--config` and `--limit`, if supplied originally. Resume checks the
config, source/suite hash, sample count, inference/sampling code hashes, and the
recorded `effectiveSettings` — each model's answer mode, vote sample count and
temperature, and for apple models the `fm` binary path and the macOS build. It
refuses to mix changed experiments. Runs whose manifest predates `effectiveSettings`
skip only that check; their code hashes still pin the behaviour. New warm-ups are
recorded for remaining models; resumed timings may span different thermal, cache
and runtime conditions.

Only one process can write to a run directory. After a hard kill, a `.lock` file may
remain: verify that the recorded PID is no longer running before deleting that
lock. Do not run two benchmark writers against the same directory. A torn final
JSONL line after a machine crash needs manual recovery; malformed result files
are rejected rather than silently dropped.

### Larger samples or different dimensions

Copy `eval/default.json` to another path and edit it, then:

```sh
bun run eval --config eval/my-config.json --out eval/runs/larger
```

`samplesPerTask` must be a multiple of 20 (balanced across 2, 4 and 5 classes).
For example, 200 per task and two contexts means 6,000 measured calls over five
models. `--limit` is for debugging: taking only an initial subset destroys the
balance guarantee. Increasing the sample size selects a larger deterministic
sample, not newly invented questions.

Other dimensions worth testing separately: repeated runs, profile/model order,
4-bit vs 8-bit, more questions per request, more choice options, warm prefix caching,
and workload concurrency. None is varied in the initial matrix. Changing the
model's configured maximum context is rarely the first useful axis.

## Reports and metrics

Run directories (ignored by git by default) contain:

| File | Contents |
|---|---|
| `manifest.json` | Configuration, sources, sampled row IDs/gold, input/suite/code hashes, machine/runtime metadata, available model limits, per-model `effectiveSettings` (answer mode, vote samples/temperature, `fm` binary, macOS build), best-effort safe model-settings snapshot |
| `results.jsonl` | One durable line per measured model/example/context, normalized answer and per-attempt telemetry; no source texts |
| `warmups.jsonl` | Excluded first/warm-up calls |
| `summary.json` | Machine-readable aggregate quality/timing/confusion matrices and paired context changes |
| `report.md` | Human-readable model × context comparison |

Regenerate reports **without any inference or downloads**:

```sh
bun run eval:report eval/runs/my-bakeoff
```

Metrics:

- **Effective accuracy:** correct / attempted; request failures count as wrong.
  Also record valid-only accuracy and coverage. For `score`, accuracy is argmax
  probability, not rounding the expected score. For `noul`, `p >= 0.5` means yes.
  Choice ties retain the Engine's first-presented-option behavior.
- **Macro task accuracy:** equal mean of the three task accuracies. It is just a
  compact summary; prefer per-task results that match your workload.
- **Score MAE:** absolute error of the expected score on the 0–4 scale.
- **Macro F1 / confusion matrices:** conditional on valid outputs.
- **Brier:** binary `(p_yes-y)^2` for BoolQ; sum of squared class errors for
  multiclass tasks. Magnitudes are not comparable across different task types.
- **NLL:** negative log probability of the gold class, clipped at `1e-12`.
- **ECE:** ten equal-width confidence bins using the probability assigned to the
  predicted class. LocalJev's inverse-entropy `confidence` field is **not** an
  estimated probability of being correct and is not used for calibration.
- **Wilson 95% intervals:** indicative per-task accuracy uncertainty, not a
  significance ranking or correction for stratification/multiple comparisons.
- **Timing:** p50, p95 and mean full-decision latency, token means, retry rates,
  failures, cached fraction. Failed requests/timeouts remain in timing statistics.
- **Paired context effect:** prediction changes, correct→wrong, and wrong→correct
  for the same examples with valid results in both conditions.

For the balanced full sample, constant-class/chance accuracy is 25% on AG News,
50% on BoolQ, and 20% on SST-5. Always returning the middle sentiment score (2)
has MAE 1.2. These are useful sanity baselines, not competitive classifiers.

Forty examples/task is a screening sample: one answer changes accuracy by 2.5
percentage points. Calibration is especially noisy; do not calibrate/tune on these
same held-out labels and then report them as an unbiased evaluation. Gold SST-5
sentiment labels can also be subjective.

## Reproducibility boundaries

Pinned dataset commits, checksums, seeds and code hashes make the test inputs and
metric computations auditable. The OpenAI model-list API does **not** expose the
installed weight revision, complete effective runtime settings or tokenization
version. The manifest records IDs and reported limits, oMLX version, Bun version,
hardware, the per-model `effectiveSettings`, and an explicit allow-list from the
default local oMLX settings file when available (not a guarantee that file is the
active configuration).

The apple backend is less inspectable still: the model ships with the OS and has
no version endpoint, so the only identifier recorded is the macOS build number
(`sw_vers -buildVersion`). An OS update can change the model underneath an
otherwise identical config, which is why a resume against a different build is
refused.

For publication-grade repeatability, additionally pin the actual model files,
quantization recipe, oMLX/MLX versions and per-model settings, repeat runs with
counterbalanced model order, measure effective rather than requested sampling,
and evaluate a larger untouched sample. A strong follow-up is a private labeled
set from your real workflow; these public tasks should guide the next experiment,
not be treated as proof of production quality.
