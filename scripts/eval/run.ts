import { appendFile, mkdir, open, readFile, unlink } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import { AppleServer } from "../../src/apple";
import { apiBaseUrl, loadSettings } from "../../src/config";
import { ATTEMPT_HEADER, Engine, SAMPLE_HEADER } from "../../src/engine";
import type { Answer } from "../../src/types";
import { hashSeed, parseArgs, readConfig, resolveModel, sha256, type Attempt, type EvalConfig, type EvaluationRow, type Example, type ResolvedModel } from "./common";
import { makeState, prepareSuite } from "./data";
import { report } from "./report";

export function answerValues(answer: Answer, example: Example) {
  const probabilities = answer.type === "noul" ? [1 - answer.noul, answer.noul]
    : example.labels.map((label, index) => answer.probabilities[answer.type === "score" ? String(index) : label]!);
  if (probabilities.some((p) => !Number.isFinite(p) || p < 0 || p > 1) || Math.abs(probabilities.reduce((a, b) => a + b, 0) - 1) > 1e-6) throw new Error("Invalid returned distribution");
  const prediction = answer.type === "noul" ? Number(answer.noul >= .5)
    : answer.type === "choice" ? example.labels.indexOf(answer.choice)
    : probabilities.indexOf(Math.max(...probabilities));
  if (prediction < 0) throw new Error("Unknown returned label");
  return { probabilities, prediction, score: answer.type === "score" ? answer.score : null };
}

function command(args: string[]): string | null {
  try { const p = Bun.spawnSync(args); return p.exitCode === 0 ? p.stdout.toString().trim() : null; } catch { return null; }
}
export async function codeHashes() {
  // src/images.ts included because every decision, images or not, is walked by
  // extractImages; a resume must not mix results from two preprocessing rules.
  const files = ["src/engine.ts", "src/types.ts", "src/config.ts", "src/apple.ts", "src/images.ts", "bun.lock", "scripts/eval/common.ts", "scripts/eval/data.ts", "scripts/eval/run.ts"];
  return Object.fromEntries(await Promise.all(files.map(async (p) => [p, sha256(await Bun.file(p).bytes())])));
}
async function localModelSettings(models: string[]) {
  // Explicit allow-list; never serialize the entire oMLX configuration or local paths.
  const keys = ["temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty", "frequency_penalty", "force_sampling", "enable_thinking", "max_context_window", "max_tokens", "dflash_enabled", "mtp_enabled", "vlm_mtp_enabled", "specprefill_enabled", "reasoning_effort"];
  try {
    const config = JSON.parse(await readFile(join(homedir(), ".omlx/model_settings.json"), "utf8"));
    return Object.fromEntries(models.map((model) => [model, Object.fromEntries(keys.filter((k) => Object.hasOwn(config.models?.[model] ?? {}, k)).map((k) => [k, config.models[model][k]]))]));
  } catch { return null; }
}
const numeric = (v: unknown) => typeof v === "number" && Number.isFinite(v) && v >= 0 ? v : 0;

/** Per-decision telemetry sink; its identity is captured by each request. */
interface Collector { reference: string; attempts: Attempt[] }
const newCollector = (reference: string): Collector => ({ reference, attempts: [] });

/**
 * The settings that actually shaped a model's requests, so a resume can refuse
 * to mix a run sampled at a different temperature or against a different binary.
 */
export function effectiveSettings(models: ResolvedModel[], fmBinary: string, appleBuild: string | null) {
  return Object.fromEntries(models.map((spec) => [spec.label, {
    backend: spec.backend, model: spec.model, answerMode: spec.answerMode,
    voteSamples: spec.voteSamples, voteTemperature: spec.voteTemperature,
    ...(spec.backend === "apple" ? { fmBinary, appleBuild } : {}),
  }]));
}

export async function run(): Promise<void> {
  const args = parseArgs();
  if (args.resume && args.out) throw new Error("Choose --out or --resume, not both");
  const config = await readConfig(args.config);
  const suite = await prepareSuite(config);
  const suiteHash = sha256(JSON.stringify(suite));
  const limit = args.limit === undefined ? suite.examples.length : Number(args.limit);
  if (!Number.isSafeInteger(limit) || limit < 1 || limit > suite.examples.length) throw new Error("Invalid --limit (total examples per model/context)");
  const examples = suite.examples.slice(0, limit);
  const models = config.models.map(resolveModel);
  const openaiModels = models.filter((m) => m.backend === "openai");
  const appleModels = models.filter((m) => m.backend === "apple");
  const settings = loadSettings();
  const headers = settings.upstreamApiKey ? { authorization: `Bearer ${settings.upstreamApiKey}` } : {};
  let available: { id: string; max_model_len?: number }[] = [];
  let backendVersion: string | null = null;
  if (openaiModels.length) {
    const modelResponse = await fetch(`${apiBaseUrl(settings)}/models`, { headers, signal: AbortSignal.timeout(10_000) });
    if (!modelResponse.ok) throw new Error(`Upstream models: HTTP ${modelResponse.status}; check .env`);
    available = (await modelResponse.json() as { data: { id: string; max_model_len?: number }[] }).data;
    for (const { model } of openaiModels) if (!available.some((m) => m.id === model)) throw new Error(`Model unavailable: ${model}`);
    try {
      const r = await fetch(`${settings.upstream.replace(/\/v1\/?$/, "")}/openapi.json`, { headers, signal: AbortSignal.timeout(10_000) });
      if (r.ok) backendVersion = (await r.json()).info?.version ?? null;
    } catch { /* some runners don't publish version info */ }
  }
  // The on-device model ships with the OS, so its build number is the only
  // version identifier available; /v1/models is checked when the server starts.
  const appleBuild = appleModels.length ? command(["sw_vers", "-buildVersion"]) : null;
  // Why not a preflight probe like the oMLX branch: starting fm serve costs
  // seconds, so each model's own server checks /v1/models when the loop reaches it.
  const directory = args.resume ?? args.out ?? `eval/runs/${new Date().toISOString().replace(/[:.]/g, "-")}`;
  if (!args.resume && await Bun.file(join(directory, "manifest.json")).exists()) throw new Error("Output exists; use --resume DIRECTORY or a new directory");
  await mkdir(directory, { recursive: true });
  const lockPath = join(directory, ".lock");
  const lock = await open(lockPath, "wx");
  await lock.writeFile(String(process.pid));
  let stopping = false;
  // Aborts an fm serve still starting up; without it a signal would wait out
  // the full start timeout with the child already spawned.
  const startupAbort = new AbortController();
  const stop = () => { stopping = true; startupAbort.abort(); console.log("Stopping after the current request; resume this directory to continue."); };
  process.on("SIGINT", stop);
  process.on("SIGTERM", stop);
  try {
    const hashes = await codeHashes();
    const effective = effectiveSettings(models, settings.fmBinary, appleBuild);
    let manifest;
    if (args.resume) {
      manifest = await Bun.file(join(directory, "manifest.json")).json();
      if (JSON.stringify(manifest.config) !== JSON.stringify(config) || manifest.suiteHash !== suiteHash || manifest.examples.length !== examples.length || JSON.stringify(manifest.codeHashes) !== JSON.stringify(hashes)) {
        throw new Error("Resume refused: config, data, limit, or inference/evaluation code changed");
      }
      // Manifests written before effective settings were recorded have nothing
      // to compare against; their code hashes already pin the behaviour.
      const hasRecordedSettings = manifest.effectiveSettings !== undefined;
      if (hasRecordedSettings && JSON.stringify(manifest.effectiveSettings) !== JSON.stringify(effective)) {
        throw new Error("Resume refused: effective model settings (answer mode, vote samples/temperature, fm binary, or OS build) changed");
      }
    } else {
      manifest = {
        runId: directory.split("/").at(-1), startedAt: new Date().toISOString(), nonce: randomUUID(),
        config, suiteHash, sources: suite.sources,
        examples: examples.map(({ id, gold, labels, row, task, question, text }) => ({ id, row, task, gold, labels, inputHash: sha256(JSON.stringify([text, question])) })),
        expectedResults: examples.length * config.models.length * config.backgroundWords.length,
        environment: {
          bun: Bun.version, cpu: command(["sysctl", "-n", "machdep.cpu.brand_string"]),
          memoryGiB: Number(command(["sysctl", "-n", "hw.memsize"])) / 2 ** 30,
          os: command(["sw_vers", "-productVersion"]), architecture: process.arch,
          power: command(["pmset", "-g", "batt"])?.includes("AC Power") ? "AC" : "unknown/battery",
        },
        gitCommit: command(["git", "rev-parse", "HEAD"]), dirty: Boolean(command(["git", "status", "--porcelain"])), codeHashes: hashes,
        backendVersion, appleBuild, effectiveSettings: effective,
        models: available.filter((m) => openaiModels.some((s) => s.model === m.id)),
        inspectedLocalModelSettings: await localModelSettings(openaiModels.map((m) => m.model)),
        runnerSettingsNote: "Best-effort whitelist from default oMLX settings location; not proof of effective runtime settings. Model weight revisions are not exposed by /v1/models.",
      };
      await Bun.write(join(directory, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n");
      await Bun.write(join(directory, "results.jsonl"), "");
      await Bun.write(join(directory, "warmups.jsonl"), "");
    }
    const doneText = await Bun.file(join(directory, "results.jsonl")).text();
    const doneRows: EvaluationRow[] = doneText.trim() ? doneText.trim().split("\n").map((line) => JSON.parse(line)) : [];
    const done = new Set(doneRows.map((r) => r.key));
    if (done.size !== doneRows.length) throw new Error("Duplicate result keys");
    console.log(`Run ${directory}: ${done.size}/${manifest.expectedResults} complete. Credentials/prompts are not logged.`);
    for (const spec of models) {
      const model = spec.label;
      const jobs = examples.flatMap((example, i) => {
        // Alternate profile order; each data point gets both lengths and model order stays fixed.
        const profiles = i % 2 ? [...config.backgroundWords].reverse() : config.backgroundWords;
        return profiles.map((words) => ({ example, words, key: `${model}|${words}|${example.id}` }));
      }).filter((job) => !done.has(job.key));
      if (!jobs.length || stopping) continue;
      const isApple = spec.backend === "apple";
      const modelSettings = loadSettings({
        backend: spec.backend, managedUpstream: isApple, answerMode: spec.answerMode,
        ...(spec.voteSamples === null ? {} : { voteSamples: spec.voteSamples }),
        ...(spec.voteTemperature === null ? {} : { voteTemperature: spec.voteTemperature }),
        upstreamModel: spec.model, temperature: config.temperature, maxOutputTokens: config.maxOutputTokens,
        malformedRetries: config.malformedRetries, timeoutMs: config.timeoutSeconds * 1000, maxInflight: 1,
      });
      // One immutable collector per decision. Why not a shared mutable array:
      // a sample that completes after its decision was abandoned would append
      // itself to whatever row happened to be measuring next.
      let collector: Collector = newCollector("warmup");
      // The measuring wrapper stays plain HTTP; only the transport underneath it
      // changes, so Attempt keeps recording the same fields for both backends.
      let appleServer: AppleServer | null = null;
      if (isApple) {
        appleServer = new AppleServer(modelSettings, { signal: startupAbort.signal });
        await appleServer.start();
      }
      const transport = appleServer ? appleServer.fetch : fetch;
      const engine = new Engine(modelSettings, async (url, init) => {
        const own = collector;
        const body = JSON.parse(String(init?.body));
        if (config.cacheMode === "bust-prefix") {
          body.messages[0].content = `Request reference (not evidence): ${sha256(`${manifest.nonce}:${own.reference}:${own.attempts.length}`).slice(0, 24)}.\n` + body.messages[0].content;
        }
        const requestBody = JSON.stringify(body);
        // The Engine labels each request, so a vote sample is never mistaken
        // for a corrective retry of the previous one.
        const labels = new Headers(init?.headers);
        const attempt: Attempt = {
          sample: Number(labels.get(SAMPLE_HEADER) ?? 0),
          retry: Number(labels.get(ATTEMPT_HEADER) ?? 0),
          ms: 0, status: null, inputTokens: 0, outputTokens: 0, cachedTokens: 0,
          backendSeconds: null, finishReason: null, reasoningCharacters: 0, warning: null,
          requestHash: sha256(requestBody), responseHash: null,
        };
        const started = performance.now();
        try {
          const r = await transport(url, { ...init, body: requestBody });
          attempt.status = r.status;
          attempt.warning = r.headers.get("warning");
          const bytes = await r.arrayBuffer();
          attempt.responseHash = sha256(new Uint8Array(bytes));
          try {
            const payload = JSON.parse(new TextDecoder().decode(bytes));
            const usage = payload.usage ?? {};
            attempt.inputTokens = numeric(usage.prompt_tokens ?? usage.input_tokens);
            attempt.outputTokens = numeric(usage.completion_tokens ?? usage.output_tokens);
            attempt.cachedTokens = numeric(usage.prompt_tokens_details?.cached_tokens);
            attempt.backendSeconds = typeof usage.total_time === "number" ? usage.total_time : null;
            attempt.finishReason = payload.choices?.[0]?.finish_reason ?? null;
            attempt.reasoningCharacters = String(payload.choices?.[0]?.message?.reasoning_content ?? "").length;
          } catch { /* engine classifies invalid payloads */ }
          return new Response(bytes, { status: r.status, headers: r.headers });
        } finally {
          attempt.ms = performance.now() - started;
          own.attempts.push(attempt);
        }
      });
      try {
      if (appleServer) {
        const served = await appleServer.fetch("http://localhost/v1/models", { signal: AbortSignal.timeout(10_000) });
        if (!served.ok) throw new Error(`fm serve /v1/models: HTTP ${served.status}`);
        const ids = (await served.json() as { data: { id: string }[] }).data;
        if (!ids.some((m) => m.id === spec.model)) throw new Error(`Apple model unavailable: ${spec.model}`);
      }
      for (let i = 0; i < config.warmupRequests && !stopping; i++) {
        const own = newCollector(`${model}:warmup:${i}:${Date.now()}`);
        collector = own;
        const started = performance.now();
        let ok = false, error: string | null = null;
        try {
          await engine.decide({ answer: { type: "noul", instructions: "Does the message mention a payment problem?", criteria: null } }, "I was charged twice for my order.", config.seed + i);
          ok = true;
        } catch (e) { error = e instanceof Error ? e.message : "Inference failed"; }
        await appendFile(join(directory, "warmups.jsonl"), JSON.stringify({ model, index: i, ms: performance.now() - started, ok, error, attempts: own.attempts }) + "\n");
      }
      let failuresInARow = 0;
      for (const { example, words, key } of jobs) {
        if (stopping) break;
        const state = makeState(example, words, suite);
        const own = newCollector(`${example.id}:${words}`);
        collector = own;
        const result: EvaluationRow = {
          key, model, backgroundWords: words, task: example.task, exampleId: example.id, gold: example.gold, labels: example.labels,
          stateHash: sha256(state), stateWords: state.split(/\s+/).length,
          elapsedMs: 0, ok: false, answer: null, probabilities: null, prediction: null, score: null, error: null, attempts: own.attempts,
        };
        const started = performance.now();
        try {
          // Only state + question go upstream. Gold labels and benchmark IDs never do.
          const decision = await engine.decide({ answer: example.question }, state, hashSeed(`${config.seed}:${example.id}`));
          const answer = decision.answers.answer;
          if (!answer) throw new Error("Missing answer");
          Object.assign(result, answerValues(answer, example), { answer, ok: true });
          failuresInARow = 0;
        } catch (error) {
          result.error = error instanceof Error ? error.message : "Inference failed";
          failuresInARow++;
        }
        result.elapsedMs = performance.now() - started;
        await appendFile(join(directory, "results.jsonl"), JSON.stringify(result) + "\n");
        done.add(key);
        const retries = own.attempts.filter((a) => (a.retry ?? 0) > 0).length;
        const requestCount = spec.answerMode === "vote"
          ? `${own.attempts.length} request(s) for ${spec.voteSamples} vote(s), ${retries} retry(ies)`
          : `${own.attempts.length} attempt(s)`;
        console.log(`${done.size}/${manifest.expectedResults} ${model} +${words}w ${example.id} ${result.ok ? (result.prediction === result.gold ? "correct" : "wrong") : "FAILED"} ${(result.elapsedMs / 1000).toFixed(2)}s ${requestCount}`);
        if (failuresInARow >= 5) throw new Error("Five consecutive failures; stopping rather than flooding a broken backend. Results are resumable.");
      }
      await report(directory);
      } finally { await appleServer?.close(); }
    }
    manifest.finishedAt = new Date().toISOString();
    manifest.completedResults = done.size;
    await Bun.write(join(directory, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n");
    console.log(`Report: ${directory}/report.md`);
  } finally {
    process.off("SIGINT", stop); process.off("SIGTERM", stop);
    await lock.close(); await unlink(lockPath);
    if (await Bun.file(join(directory, "results.jsonl")).exists()) await report(directory);
  }
}

if (import.meta.main) await run();
