import { describe, expect, test } from "bun:test";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { hashOrder, quantile, readConfig, resolveModel, type EvaluationRow, type Example, type Suite } from "../scripts/eval/common";
import { balancedSample, makeState } from "../scripts/eval/data";
import { correctiveRetries, taskMetrics, timingMetrics, wilson } from "../scripts/eval/metrics";
import { answerValues, codeHashes, effectiveSettings } from "../scripts/eval/run";
import { report } from "../scripts/eval/report";

const example: Example = {
  id: "boolq:validation:0", row: 0, task: "boolq", text: "The target evidence remains intact.",
  gold: 1, labels: ["no", "yes"], question: { type: "noul", instructions: "Is this true?", criteria: null },
};
const suite: Suite = { version: 1, seed: 42, samplesPerTask: 20, sources: [], examples: [example], background: ["One unrelated passage here.", "Another passage with several irrelevant words."] };
function row(overrides: Partial<EvaluationRow> = {}): EvaluationRow {
  return {
    key: "model|0|boolq:validation:0", model: "model", backgroundWords: 0,
    task: "boolq", exampleId: example.id, gold: 1, labels: ["no", "yes"],
    stateHash: "example", stateWords: 6, elapsedMs: 1000, ok: true,
    answer: { type: "noul", noul: .8 }, probabilities: [.2, .8], prediction: 1,
    score: null, error: null,
    attempts: [{ ms: 990, status: 200, inputTokens: 100, outputTokens: 20, cachedTokens: 0, backendSeconds: .9, finishReason: "stop", reasoningCharacters: 0, warning: null, requestHash: "request", responseHash: "response" }],
    ...overrides,
  };
}

describe("evaluation data", () => {
  test("hash sampling is deterministic, stratified, and without replacement", () => {
    const population = Array.from({ length: 100 }, (_, i) => ({ ...example, id: `row-${i}`, gold: i % 2 }));
    const sample = balancedSample(population, 20, 42);
    expect(sample).toEqual(balancedSample(population, 20, 42));
    expect(sample.filter((e) => e.gold === 0)).toHaveLength(10);
    expect(new Set(sample.map((e) => e.id)).size).toBe(20);
    expect(sample).not.toEqual(balancedSample(population, 20, 43));
    expect(() => balancedSample(population, 21, 42)).toThrow();
    expect(() => balancedSample(population, 102, 42)).toThrow();
    expect(hashOrder(population, (e) => e.id, "a")).toHaveLength(100);
  });
  test("context variants preserve gold evidence, add the requested words, and omit labels", () => {
    const short = makeState(example, 0, suite);
    const long = makeState(example, 2048, suite);
    expect(short).toBe(`TARGET:\n${example.text}\nEND TARGET`);
    expect(long).toContain(short);
    expect(long.split("BACKGROUND (irrelevant):")).toHaveLength(3);
    const added = long.replace(short, "").replaceAll("BACKGROUND (irrelevant):", "").trim().split(/\s+/);
    expect(added).toHaveLength(2048);
    expect(long).toBe(makeState(example, 2048, suite));
    expect(long).not.toContain(example.id);
    expect(makeState({ ...example, gold: 0 }, 2048, suite)).toBe(long);
  });
  test("choice probabilities follow semantic labels, not shuffled presentation order", () => {
    const e: Example = { ...example, task: "ag_news", labels: ["world", "sports"], gold: 0 };
    expect(answerValues({ type: "choice", choice: "world", probabilities: { sports: .2, world: .8 }, confidence: .2 }, e)).toEqual({ probabilities: [.8, .2], prediction: 0, score: null });
  });
  test("score accuracy uses argmax, while score error uses the expected value", () => {
    const e: Example = { ...example, task: "sst5", labels: ["negative", "neutral", "positive"], gold: 0 };
    const result = answerValues({ type: "score", score: .8, probabilities: { "0": .5, "1": .2, "2": .3 }, legend: {}, confidence: 0 }, e);
    expect(result.prediction).toBe(0);
    expect(result.score).toBe(.8);
    expect(answerValues({ type: "noul", noul: .5 }, example).prediction).toBe(1);
  });
});

describe("evaluation metrics", () => {
  test("binary Brier, NLL and ECE have known values", () => {
    const m = taskMetrics([row()]);
    expect(m.effectiveAccuracy).toBe(1);
    expect(m.brier).toBeCloseTo(.04);
    expect(m.nll).toBeCloseTo(-Math.log(.8));
    expect(m.ece).toBeCloseTo(.2);
  });
  test("failures count as wrong, not as missing or uniform predictions", () => {
    const m = taskMetrics([row(), row({ key: "failed", ok: false, probabilities: null, prediction: null, answer: null, error: "timeout" })]);
    expect(m.total).toBe(2);
    expect(m.valid).toBe(1);
    expect(m.validAccuracy).toBe(1);
    expect(m.effectiveAccuracy).toBe(.5);
    expect(m.brier).toBeCloseTo(.04);
  });
  test("multiclass Brier, score MAE and zero-probability clipping", () => {
    const m = taskMetrics([row({ task: "sst5", labels: ["a", "b", "c"], gold: 0, probabilities: [.5, .2, .3], prediction: 0, score: .8 })]);
    expect(m.brier).toBeCloseTo(.38);
    expect(m.scoreMAE).toBeCloseTo(.8);
    expect(taskMetrics([row({ probabilities: [1, 0], prediction: 0 })]).nll).toBeCloseTo(-Math.log(1e-12));
  });
  test("latency includes retries and failures; tokens sum all attempts", () => {
    const r = row();
    const m = timingMetrics([r, row({ elapsedMs: 3000, ok: false, attempts: [r.attempts[0]!, r.attempts[0]!] })]);
    expect(m.latencyP50Ms).toBe(2000);
    expect(m.latencyP95Ms).toBeCloseTo(2900);
    expect(m.failures).toBe(1);
    expect(m.retriedRequests).toBe(1);
    expect(m.inputTokensMean).toBe(150);
    expect(m.outputTokensMean).toBe(30);
    expect(m.firstPassValid).toBe(1);
  });
  test("vote samples are not counted as corrective retries", () => {
    const attempt = (sample: number, retry: number) => ({ ...row().attempts[0]!, sample, retry });
    // A vote5 decision: five samples, none of which needed a retry.
    const clean = row({ attempts: [attempt(0, 0), attempt(1, 0), attempt(2, 0), attempt(3, 0), attempt(4, 0)] });
    expect(correctiveRetries(clean)).toBe(0);
    const m = timingMetrics([clean]);
    expect(m.retriedRequests).toBe(0);
    expect(m.firstPassValid).toBe(1);
    expect(m.additionalAttempts).toBe(0);
  });
  test("a corrective retry inside one vote sample is counted once", () => {
    const attempt = (sample: number, retry: number) => ({ ...row().attempts[0]!, sample, retry });
    const retried = row({ attempts: [attempt(0, 0), attempt(1, 0), attempt(1, 1)] });
    expect(correctiveRetries(retried)).toBe(1);
    const m = timingMetrics([retried]);
    expect(m.retriedRequests).toBe(1);
    expect(m.firstPassValid).toBe(0);
    expect(m.additionalAttempts).toBe(1);
  });
  test("results recorded before sample labels existed keep their old counts", () => {
    // No sample/retry fields: every request past the first is read as a retry.
    const legacy = row({ attempts: [row().attempts[0]!, row().attempts[0]!] });
    expect(correctiveRetries(legacy)).toBe(1);
    expect(timingMetrics([legacy]).retriedRequests).toBe(1);
    expect(timingMetrics([row()]).firstPassValid).toBe(1);
  });
  test("empty metrics are absent, never spuriously perfect", () => {
    expect(taskMetrics([]).effectiveAccuracy).toBeNull();
    expect(taskMetrics([]).ece).toBeNull();
    expect(timingMetrics([]).latencyMeanMs).toBeNull();
    expect(quantile([], .5)).toBeNull();
    expect(wilson(0, 0)).toBeNull();
    expect(wilson(20, 40)![0]).toBeCloseTo(.352, 2);
    expect(wilson(20, 40)![1]).toBeCloseTo(.648, 2);
  });
  test("a vote report does not present its samples as retries", async () => {
    const directory = await mkdtemp(join(tmpdir(), "localjev-vote-report-"));
    try {
      const attempt = (sample: number, retry: number) => ({ ...row().attempts[0]!, sample, retry });
      await Bun.write(join(directory, "manifest.json"), JSON.stringify({
        runId: "vote", config: {
          models: [{ backend: "apple", answerMode: "vote", voteSamples: 5, voteTemperature: 1 }],
          backgroundWords: [0], samplesPerTask: 40, seed: 1, temperature: 0,
          maxOutputTokens: 256, malformedRetries: 2, warmupRequests: 1, cacheMode: "bust-prefix",
        },
        environment: { bun: "t", cpu: "t", memoryGiB: 96 }, examples: [example], expectedResults: 1,
      }));
      await Bun.write(join(directory, "results.jsonl"), JSON.stringify(row({
        model: "apple:system:vote5",
        attempts: [attempt(0, 0), attempt(1, 0), attempt(2, 0), attempt(3, 0), attempt(4, 0)],
      })) + "\n");
      await Bun.write(join(directory, "warmups.jsonl"), "");
      await report(directory);
      const summary = await Bun.file(join(directory, "summary.json")).json();
      const timing = summary.cells[0].timing;
      // Five clean samples: no retry, and the decision is first-pass valid.
      expect(timing.retriedRequests).toBe(0);
      expect(timing.firstPassValid).toBe(1);
      expect(timing.additionalAttempts).toBe(0);
      expect(await Bun.file(join(directory, "report.md")).text()).toContain("| 0/1 |");
    } finally { await rm(directory, { recursive: true, force: true }); }
  });
  test("reports are regenerable and mark incomplete matrices", async () => {
    const directory = await mkdtemp(join(tmpdir(), "localjev-eval-test-"));
    try {
      await Bun.write(join(directory, "manifest.json"), JSON.stringify({ runId: "test", config: { models: ["model"], backgroundWords: [0, 2048], samplesPerTask: 40, seed: 42, temperature: 0, maxOutputTokens: 256, malformedRetries: 2, cacheMode: "bust-prefix" }, environment: { bun: "test", cpu: "test", memoryGiB: 64 }, examples: [example], expectedResults: 2 }));
      await Bun.write(join(directory, "results.jsonl"), JSON.stringify(row()) + "\n");
      await Bun.write(join(directory, "warmups.jsonl"), "");
      await report(directory);
      const first = await Bun.file(join(directory, "report.md")).text();
      expect(first).toContain("PARTIAL");
      await report(directory);
      expect(await Bun.file(join(directory, "report.md")).text()).toBe(first);
      expect((await Bun.file(join(directory, "summary.json")).json()).complete).toBe(false);
      await appendDuplicate(directory);
      await expect(report(directory)).rejects.toThrow("Duplicate results");
    } finally { await rm(directory, { recursive: true, force: true }); }
  });
});
describe("evaluation model specs", () => {
  const base = { seed: 1, samplesPerTask: 20, backgroundWords: [0], temperature: 0, maxOutputTokens: 256, malformedRetries: 2, timeoutSeconds: 90, warmupRequests: 1, cacheMode: "bust-prefix" };
  async function writeConfig(models: unknown): Promise<string> {
    const directory = await mkdtemp(join(tmpdir(), "localjev-config-test-"));
    const path = join(directory, "config.json");
    await Bun.write(path, JSON.stringify({ ...base, models }));
    return path;
  }
  test("a bare string names an upstream model, an object selects the apple backend", () => {
    expect(resolveModel("gemma-4")).toEqual({ label: "gemma-4", model: "gemma-4", backend: "openai", answerMode: "probability", voteSamples: null, voteTemperature: null });
    expect(resolveModel({ backend: "apple" })).toEqual({ label: "apple:system", model: "system", backend: "apple", answerMode: "probability", voteSamples: null, voteTemperature: null });
    expect(resolveModel({ backend: "apple", model: "system" }).label).toBe("apple:system");
  });
  test("both model forms load, and labels distinguish them", async () => {
    const config = await readConfig(await writeConfig(["gemma-4", { backend: "apple" }]));
    expect(config.models.map((m) => resolveModel(m).label)).toEqual(["gemma-4", "apple:system"]);
  });
  test("unknown backends, empty ids, and repeated labels are rejected", async () => {
    await expect(readConfig(await writeConfig([{ backend: "ollama" }]))).rejects.toThrow("models must be a nonempty list");
    await expect(readConfig(await writeConfig([{ backend: "apple", model: "" }]))).rejects.toThrow("models must be a nonempty list");
    await expect(readConfig(await writeConfig([""]))).rejects.toThrow("models must be a nonempty list");
    await expect(readConfig(await writeConfig([]))).rejects.toThrow("models must be a nonempty list");
    await expect(readConfig(await writeConfig([{ backend: "apple" }, { backend: "apple", model: "system" }]))).rejects.toThrow("Duplicate models");
  });
  test("a vote spec is labelled by its sample count, and only vote specs carry one", () => {
    expect(resolveModel({ backend: "apple", answerMode: "vote" }).label).toBe("apple:system:vote5");
    expect(resolveModel({ backend: "apple", answerMode: "vote", voteSamples: 8 })).toEqual({ label: "apple:system:vote8", model: "system", backend: "apple", answerMode: "vote", voteSamples: 8, voteTemperature: 1 });
    expect(resolveModel({ backend: "openai", model: "gemma-4", answerMode: "vote" }).label).toBe("gemma-4:vote5");
  });
  test("a vote spec pins its own sampling temperature, defaulting to 1", () => {
    expect(resolveModel({ backend: "apple", answerMode: "vote" }).voteTemperature).toBe(1);
    expect(resolveModel({ backend: "apple", answerMode: "vote", voteTemperature: 0.7 }).voteTemperature).toBe(0.7);
    // Probability mode has no vote temperature to report at all.
    expect(resolveModel({ backend: "apple" }).voteTemperature).toBeNull();
  });
  test("an openai spec needs an explicit model id, and vote fields must be well formed", async () => {
    expect((await readConfig(await writeConfig([{ backend: "openai", model: "gemma-4" }]))).models).toHaveLength(1);
    await expect(readConfig(await writeConfig([{ backend: "openai" }]))).rejects.toThrow("models must be a nonempty list");
    await expect(readConfig(await writeConfig([{ backend: "apple", answerMode: "majority" }]))).rejects.toThrow("models must be a nonempty list");
    await expect(readConfig(await writeConfig([{ backend: "apple", answerMode: "vote", voteSamples: 0 }]))).rejects.toThrow("models must be a nonempty list");
    await expect(readConfig(await writeConfig([{ backend: "apple", answerMode: "vote", voteSamples: 1.5 }]))).rejects.toThrow("models must be a nonempty list");
    await expect(readConfig(await writeConfig([{ backend: "apple", answerMode: "vote", voteTemperature: -1 }]))).rejects.toThrow("models must be a nonempty list");
  });
  test("a misspelled or misplaced model-spec key is rejected instead of ignored", async () => {
    await expect(readConfig(await writeConfig([{ backend: "apple", answermode: "vote" }]))).rejects.toThrow("models must be a nonempty list");
    await expect(readConfig(await writeConfig([{ backend: "apple", voteSampples: 5 }]))).rejects.toThrow("models must be a nonempty list");
    // voteSamples/voteTemperature only mean something in vote mode.
    await expect(readConfig(await writeConfig([{ backend: "apple", voteSamples: 8 }]))).rejects.toThrow("models must be a nonempty list");
    await expect(readConfig(await writeConfig([{ backend: "apple", voteTemperature: 1 }]))).rejects.toThrow("models must be a nonempty list");
  });
  test("effective settings capture what a resume must not silently change", () => {
    const models = [resolveModel({ backend: "apple" }), resolveModel({ backend: "apple", answerMode: "vote", voteSamples: 5, voteTemperature: 1 })];
    const recorded = effectiveSettings(models, "/usr/bin/fm", "26A428");
    expect(recorded["apple:system:vote5"]).toEqual({
      backend: "apple", model: "system", answerMode: "vote",
      voteSamples: 5, voteTemperature: 1, fmBinary: "/usr/bin/fm", appleBuild: "26A428",
    });
    // Each of these differences has to make the recorded settings differ.
    const hotter = effectiveSettings([resolveModel({ backend: "apple", answerMode: "vote", voteSamples: 5, voteTemperature: 0.5 })], "/usr/bin/fm", "26A428");
    const otherBinary = effectiveSettings(models, "/opt/fm", "26A428");
    const otherBuild = effectiveSettings(models, "/usr/bin/fm", "26B100");
    expect(JSON.stringify(hotter)).not.toBe(JSON.stringify(recorded));
    expect(JSON.stringify(otherBinary)).not.toBe(JSON.stringify(recorded));
    expect(JSON.stringify(otherBuild)).not.toBe(JSON.stringify(recorded));
  });
  test("an openai model records no apple-only fields to compare against", () => {
    const recorded = effectiveSettings([resolveModel("gemma-4")], "/usr/bin/fm", null);
    expect(recorded["gemma-4"]).toEqual({
      backend: "openai", model: "gemma-4", answerMode: "probability",
      voteSamples: null, voteTemperature: null,
    });
  });
  test("the shipped apple config compares probability against vote on the same backend", async () => {
    const config = await readConfig("eval/apple.json");
    const resolved = config.models.map((m) => resolveModel(m));
    expect(resolved.map((m) => m.backend)).toEqual(["apple", "apple"]);
    expect(resolved.map((m) => m.label)).toEqual(["apple:system", "apple:system:vote5"]);
    // Pinned in the file so a resume is not at the mercy of the environment.
    expect(resolved.map((m) => m.voteTemperature)).toEqual([null, 1]);
    expect(config.samplesPerTask).toBe(40);
    expect(config.backgroundWords).toEqual([0]);
  });
});

async function appendDuplicate(directory: string) {
  await Bun.write(join(directory, "results.jsonl"), [row(), row()].map((r) => JSON.stringify(r)).join("\n") + "\n");
}

describe("run provenance", () => {
  test("every source file a decision depends on is hashed, images included", async () => {
    const hashes = await codeHashes();
    // extractImages runs for every decision, images or not, so a resume that
    // ignored it could mix results from two different preprocessing rules.
    for (const file of ["src/engine.ts", "src/images.ts", "src/types.ts", "src/config.ts"]) {
      expect(Object.keys(hashes)).toContain(file);
      expect(hashes[file]).toMatch(/^[0-9a-f]{64}$/);
    }
  });
});
