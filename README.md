# LocalJev

A local, Jev-compatible `POST /v1/systemone` API written in TypeScript for
[Bun](https://bun.sh/), backed by an OpenAI-compatible Chat Completions endpoint.

Two inference backends are supported:

- `openai` (default) — any OpenAI-compatible server, such as oMLX serving
  DiffusionGemma;
- `apple` — the on-device Apple Foundation Model through `fm serve`, which ships
  with macOS 27 and needs no model download.

The defaults target:

- inference server: `http://127.0.0.1:8000`
- model: `diffusiongemma-26B-A4B-it-4bit`
- LocalJev API: `http://127.0.0.1:8080`

## Why a bridge is needed

[Jev](https://typesafe.ai/) uses a typed decision API rather than an OpenAI chat API.
[OpenJev](https://github.com/razorback16/openjev) implements the Jev wire protocol and
obtains probabilities with a special one-step DiffusionGemma **structured read**. Its
backend depends on unmerged vLLM request extensions such as
`diffusion_seed_canvas`, `diffusion_read_only`, and requested token logprobs.

The normal oMLX API does not expose those primitives. LocalJev therefore takes the
portable approach:

1. translate `state` and typed Jev questions into a classification prompt;
2. ask DiffusionGemma for a JSON probability scalar/vector;
3. validate the complete result and retry malformed output;
4. normalize vectors and calculate Jev-compatible choices, expected scores, and
   entropy-based confidence;
5. return the normal Jev response shape.

This is wire-compatible, but not mathematically equivalent to OpenJev's logit read.
The probabilities are generated/self-reported by the model rather than read directly
from its logits. Evaluate their calibration on your own workload before relying on
them for consequential decisions.

## Run with oMLX

Requires Bun 1.2+ and a running oMLX server.

```sh
bun install
cp .env.example .env
$EDITOR .env # replace the upstream API-key placeholder
bun run start
```

Bun loads `.env` automatically. Alternatively, set the key in your shell before
starting the server:

```fish
# fish
set -gx LOCALJEV_UPSTREAM_API_KEY 'your-local-omlx-key'
```

```sh
# bash/zsh
export LOCALJEV_UPSTREAM_API_KEY='your-local-omlx-key'
```

LocalJev listens on `http://127.0.0.1:8080`. Check that the configured model is
available:

```bash
curl http://127.0.0.1:8080/ready
```

Make a decision:

```bash
curl http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "jev-latest",
    "state": "Hi, I have been trying to connect Stripe but keep getting a 403 error.",
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
          "billing": "Payment or subscription issues",
          "technical": "Bugs or integration problems",
          "sales": "Pricing or account questions"
        }
      },
      "frustration": {
        "type": "score",
        "instructions": "How frustrated does the customer appear?",
        "criteria": ["Calm", "Frustrated but civil", "Very angry"]
      },
      "urgent": {
        "type": "noul",
        "instructions": "Does this require an immediate response?"
      }
    }
  }'
```

## Run with Apple Foundation Models

Requires macOS 27 with Apple Intelligence enabled and the on-device model
downloaded. Check the CLI that ships with the OS:

```sh
fm available      # expect: System model available
fm license        # review and accept the model license once
```

Then start LocalJev with the `apple` backend. No inference server has to be
running first: LocalJev spawns `fm serve` on a private Unix socket, waits for its
health check, and stops it again on shutdown.

```sh
LOCALJEV_BACKEND=apple bun run start
curl http://127.0.0.1:8080/ready
# {"status":"ready","backend":"apple","upstream_model":"system"}
```

The `/v1/systemone` request and response shapes are identical to the oMLX
backend, so clients and the SDK need no changes.

To reuse a `fm serve` you already run yourself, point `LOCALJEV_UPSTREAM` at it.
LocalJev then only connects, and never spawns or stops a child process:

```sh
fm serve --host 127.0.0.1 --port 1976
LOCALJEV_BACKEND=apple LOCALJEV_UPSTREAM=http://127.0.0.1:1976 bun run start
```

> **Switching an existing `.env` to `apple` means commenting out all three
> upstream lines**, not just one:
>
> - `LOCALJEV_UPSTREAM` — any value makes the apple backend connect there
>   instead of spawning `fm serve`, so LocalJev talks to an oMLX server that is
>   probably not running.
> - `LOCALJEV_UPSTREAM_MODEL` — `fm serve` only serves `system`; leaving
>   `diffusiongemma-…` in place makes `/ready` report unavailable and every
>   decision fail with an unknown-model error from the backend.
> - `LOCALJEV_UPSTREAM_API_KEY` — unused by `fm serve`; harmless but misleading.
>
> The apple defaults already supply the right values once these are unset.

### Backend defaults

With `LOCALJEV_BACKEND=apple`, these defaults change to fit the on-device model's
**4096-token context window**. Any environment variable you set explicitly still
wins.

| Setting | `openai` default | `apple` default |
|---|---|---|
| `LOCALJEV_UPSTREAM_MODEL` | `diffusiongemma-26B-A4B-it-4bit` | `system` |
| `LOCALJEV_MAX_OUTPUT_TOKENS` | `2048` | `512` |
| `LOCALJEV_QUESTIONS_PER_CALL` | `16` | `8` |
| `LOCALJEV_OUTCOMES_PER_CALL` | `128` | `32` |

Long `state` documents and large question batches can still exceed the window.
LocalJev already splits questions into several model calls, but a single oversized
`state` cannot be split and will be refused.

### Refusals return HTTP 422

When the backend declines a specific input, retrying it cannot succeed, so
LocalJev answers 422 with no `retry-after`:

```
HTTP/1.1 422 Unprocessable Entity
{"detail":{"error_type":"invalid_request_error",
           "message":"The inference backend rejected this request: The model's safety guardrails were triggered."}}
```

Two paths lead there. On **any** backend, a chat completion carrying a non-empty
`message.refusal` is a refusal. On the **apple** backend, `fm serve` instead
reports refusals as HTTP 500, so three confirmed messages are recognised by
wording: `The model's safety guardrails were triggered.`, `The session's
transcript exceeded the model's context size.` and `The model refused to
answer.`. An unrecognised 500 stays a 503, which is the safe way to be wrong.

Everything else keeps its old meaning:

| Upstream | LocalJev | Meaning |
|---|---|---|
| Non-empty `message.refusal` (any backend) | 422, no `retry-after` | The model declined this input |
| 500 with a confirmed refusal wording (`apple`) | 422, no `retry-after` | As above; `fm serve` reports refusals as 500 |
| 4xx other than 408/429 | 502 | LocalJev or the backend is misconfigured; the upstream message is included |
| 408, 429, and any other 5xx | 503 with `retry-after` | A transient condition worth retrying; the upstream message is included |

Note that a refusal turns on the **content being classified**, not only on
malicious requests: moderating or triaging user-generated text that is itself
abusive can be refused. If your workload must classify such text, the `openai`
backend with a model you control is the better fit.

### Quality caveats

The on-device model is roughly 3B parameters, and its probabilities are much less
calibrated than the larger models in the bake-off. Observed on this machine:

- probabilities saturate toward `0`, `0.5` and `1` instead of expressing graded
  belief — the README example above returns `{billing: 0, technical: 0.5,
  sales: 0.5}` rather than a peaked distribution, and the `urgent` noul comes back
  as exactly `0`;
- misclassifications occur on inputs the larger models get right: "I was charged
  twice" was answered `technical` rather than `billing`, with a saturated `[0,1,0]`
  distribution;
- **the model refuses ordinary, harmless inputs.** In the `eval/runs/apple-modes-v2`
  measurement (40 examples per task, macOS 27.0 build 26A428) it answered `The
  model refused to answer.` for **35% of BoolQ** examples and roughly **8% of
  AG News**. These are public benchmark passages, not abusive text — one was a
  BoolQ question about the horror drama *Fear the Walking Dead*. The refusal
  tracks the **topic of the input**, not how the prompt is written: removing the
  untrusted-data instruction, dropping the `<document>` tags, or passing the state
  as raw text each left 16–17 of the 17 reproduced cases still refusing. There is
  no prompt-side workaround; these arrive as HTTP 422.

Treat it as a zero-setup default for development and low-stakes routing, not as a
calibrated probability source. Measure it on your own workload first; see
`eval/apple.json` and the evaluation section below.

### Voting instead of self-reported probabilities

`LOCALJEV_ANSWER_MODE=vote` derives the distribution from the model's behaviour
rather than its introspection. Instead of asking for numbers, LocalJev makes the
model pick **one** label per question — constrained by a JSON-schema `enum`, so
the decoder only has to accept a listed outcome — and repeats that with
`LOCALJEV_VOTE_SAMPLES` different seeds. The frequency of each label becomes the
probability: 3 of 5 samples answering `yes` yields `noul: 0.6`.

This works on both backends. It is **the default for the apple backend** and off
by default for openai. It exists because the on-device model answers a "give me a
probability" prompt with saturated `0`/`1` values, and sometimes with an all-zero
array that has no valid normalization at all; picking one label is a much easier
task for a ~3B model, and disagreement between samples is what expresses
uncertainty.

```sh
LOCALJEV_BACKEND=apple bun run start                                  # votes
LOCALJEV_BACKEND=apple LOCALJEV_ANSWER_MODE=probability bun run start # 4× faster, much less accurate
```

The trade-offs are real and worth stating plainly:

- **Latency multiplies by K.** One decision becomes K upstream requests. They are
  issued concurrently, but `fm serve` largely serializes them, so the wall time of
  a 5-sample decision is roughly 5× a single one.
- **Probability granularity is 1/K.** With the default 5 samples the only
  reachable values are `0`, `0.2`, `0.4`, `0.6`, `0.8` and `1`. A vote cannot
  express "0.73", and a unanimous vote reports `1.0` — which is a small-sample
  artifact, not genuine certainty. Raise `LOCALJEV_VOTE_SAMPLES` for a finer grid,
  at proportional cost.
- **Token usage is the sum of every sample**, and `usage` in the response reflects
  that.
- Ties are broken toward the outcome listed first, matching probability mode.

The apple default was chosen by measurement, not taste. `eval/apple.json` runs
`apple:system` and `apple:system:vote5` over the same 120 labelled examples
(40 each of AG News, BoolQ and SST-5; macOS 27.0 build 26A428, M2 Max). Two
independent runs agreed:

| Mode | Macro accuracy | AG News, answered only | BoolQ, answered only | SST-5 MAE ↓ | p50 latency |
|---|---:|---:|---:|---:|---:|
| `probability` | 35.8% | 43.2% | 65.4% | 1.186 | 1.0 s |
| `vote` (5 samples) | 52.5% | 78.1% | 75.0% | 0.600 | 4.0 s |

Macro accuracy counts refused and failed requests as wrong, which is why it sits
well below the answered-only columns. Forty examples per task leaves wide
intervals, so read this as "voting is clearly better here", not as a precise
score. Re-run it on your own workload with
`bun run eval --config eval/apple.json --out eval/runs/my-apple`.

## Use the TypeSafe SDK

The SDK requires an API-key value. LocalJev accepts any value unless
`LOCALJEV_API_KEY` is configured. Set the SDK environment for your shell:

```fish
# fish
set -gx TYPESAFE_BASE_URL http://127.0.0.1:8080
set -gx TYPESAFE_API_KEY local
```

```sh
# bash/zsh
export TYPESAFE_BASE_URL=http://127.0.0.1:8080
export TYPESAFE_API_KEY=local
```

```python
from typesafe_sdk import TypeSafeClient

client = TypeSafeClient()
response = client.system_one(
    "I was charged twice this month.",
    {
        "billing": {
            "type": "noul",
            "instructions": "Is this a billing issue?",
        }
    },
)
print(response.nouls["billing"].noul)
```

`jev-latest` and `jev-preview` are accepted aliases so SDK defaults work unchanged.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `LOCALJEV_BACKEND` | `openai` | `openai` or `apple`; any other value fails at startup |
| `LOCALJEV_FM_BINARY` | `/usr/bin/fm` | Apple CLI used to spawn `fm serve` (`apple` backend only) |
| `LOCALJEV_UPSTREAM` | `http://127.0.0.1:8000` | OpenAI-compatible base URL, with or without `/v1`. On the `apple` backend, setting this connects to an existing `fm serve` instead of spawning one |
| `LOCALJEV_UPSTREAM_API_KEY` | empty | Bearer key sent to the inference server |
| `LOCALJEV_UPSTREAM_MODEL` | `diffusiongemma-26B-A4B-it-4bit` (apple: `system`) | Upstream model identifier |
| `LOCALJEV_API_KEY` | empty | Optional Bearer key required from LocalJev clients |
| `LOCALJEV_HOST` | `127.0.0.1` | Listen address |
| `LOCALJEV_PORT` | `8080` | Listen port |
| `LOCALJEV_TIMEOUT` | `180` | Upstream timeout in seconds |
| `LOCALJEV_MAX_INFLIGHT` | `2` | Concurrent calls admitted upstream |
| `LOCALJEV_MAX_QUEUE` | `64` | Admitted decisions — running plus waiting — before a new one is refused with HTTP 529. A `vote` decision counts once, however many samples it issues |
| `LOCALJEV_MALFORMED_RETRIES` | `2` | Corrective retries for invalid model JSON |
| `LOCALJEV_ANSWER_MODE` | `probability` (`vote` for apple) | `probability` (the model reports its own probabilities) or `vote` (sample labels K times and count); any other value fails at startup |
| `LOCALJEV_VOTE_SAMPLES` | `5` | Samples per question group in `vote` mode (integer ≥ 1) |
| `LOCALJEV_VOTE_TEMPERATURE` | `1` | Sampling temperature in `vote` mode; `0` makes every sample identical |
| `LOCALJEV_MAX_OUTPUT_TOKENS` | `2048` (apple: `512`) | Per-completion output ceiling |
| `LOCALJEV_QUESTIONS_PER_CALL` | `16` (apple: `8`) | Chunking limit per model call |
| `LOCALJEV_OUTCOMES_PER_CALL` | `128` (apple: `32`) | Choice/score outcomes per model call |

Bun automatically loads `.env`, so you can also copy `.env.example`, replace its
placeholder, and run the server.

## Development

```bash
bun install
bun test
bun run typecheck
bun run smoke       # live call to the configured inference server
```

Ctrl-C during `bun run smoke` stops a managed `fm serve` — including one still
starting — and exits with status 130. A request already sent to the upstream is
not cancelled by that: it keeps running on the upstream until its own timeout
ends it, because the smoke script does not pass the interruption signal to the
request itself.

## Evaluate different models

The repeatable bake-off uses public gold labels for news categorization (AG News),
yes/no reading comprehension (BoolQ), and five-level sentiment (SST-5). It runs the
same LocalJev engine against five installed models, comparing quality, calibration,
retries, and full-decision latency at two actual input lengths.

```sh
# Quick integration check (30 requests, not a meaningful quality sample)
bun run eval --out eval/runs/pilot --limit 3

# 5 models × 120 labeled examples × 2 input lengths = 1,200 requests
bun run eval --out eval/runs/my-bakeoff

# Regenerate a completed or partial report without running inference
bun run eval:report eval/runs/my-bakeoff

# Apple Foundation Models only; spawns and stops fm serve itself
bun run eval --config eval/apple.json --out eval/runs/apple --limit 3
```

A `models` entry is either an upstream model id (a string) or an object such as
`{ "backend": "apple", "model": "system" }`, reported under the label
`apple:system`. The object form also selects the answer mode — adding
`"answerMode": "vote"`, `"voteSamples"` and `"voteTemperature"` gives a second
label like `apple:system:vote5`. Unknown or misplaced keys fail the run rather
than being ignored. All forms can appear in one config, so the on-device model can
be compared against oMLX models, and both answer modes against each other, in a
single run. The full field list is in [the evaluation guide](docs/evaluation.md).

The oMLX backend requires oMLX and the upstream key in `.env`; the apple backend
requires neither. No running LocalJev HTTP server or Python is needed. See [the evaluation guide](docs/evaluation.md) for pinned data
sources, methodology, configuration, resuming runs, and limitations.

The [first completed bake-off](docs/evaluation-results-2026-09-18.md) includes
1,200 requests on an M5 Max. Gemma 4 26B-A4B and Qwen3.6 were the strongest overall
candidates in this small screening sample; the report includes per-task results,
latency, context effects, and caveats rather than claiming a definitive winner.

## Should you use LM Studio instead?

Not currently for this model. As of September 18, 2026, DiffusionGemma support is
still tracked as open in both
[`lmstudio-ai/mlx-engine#336`](https://github.com/lmstudio-ai/mlx-engine/issues/336)
and
[`lmstudio-ai/lmstudio-bug-tracker#2037`](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/2037).
The reported MLX backend fails to load `diffusion_gemma`, while the normal llama.cpp
backend reports an unknown architecture. oMLX already loads and serves your exact
checkpoint successfully, so it is the better runner for this Mac today.

Even after LM Studio adds ordinary generation support, changing runners alone will
not make the result OpenJev-equivalent. The runner must expose seeded diffusion
canvases, read-only denoising, and selected-token logits/logprobs. If LM Studio only
provides standard Chat Completions, LocalJev can use it by changing
`LOCALJEV_UPSTREAM`, but the probability path remains prompted/self-reported.

For direct model probabilities, the best paths are:

1. add the structured-read primitives to oMLX's DiffusionGemma lane and consume them
   here; or
2. run OpenJev's patched vLLM backend on a supported NVIDIA machine.
