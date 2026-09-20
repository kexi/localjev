import { afterEach, describe, expect, test } from "bun:test";

import { loadSettings } from "../src/config";

const touched = [
  "LOCALJEV_BACKEND",
  "LOCALJEV_UPSTREAM",
  "LOCALJEV_UPSTREAM_MODEL",
  "LOCALJEV_MAX_OUTPUT_TOKENS",
  "LOCALJEV_QUESTIONS_PER_CALL",
  "LOCALJEV_OUTCOMES_PER_CALL",
  "LOCALJEV_FM_BINARY",
  "LOCALJEV_ANSWER_MODE",
  "LOCALJEV_VOTE_SAMPLES",
  "LOCALJEV_VOTE_TEMPERATURE",
] as const;

afterEach(() => {
  for (const name of touched) delete process.env[name];
});

describe("backend settings", () => {
  test("defaults to the OpenAI-compatible upstream and its larger batches", () => {
    const settings = loadSettings();
    expect(settings.backend).toBe("openai");
    expect(settings.managedUpstream).toBe(false);
    expect(settings.upstreamModel).toBe("diffusiongemma-26B-A4B-it-4bit");
    expect(settings.maxOutputTokens).toBe(2_048);
    expect(settings.questionsPerCall).toBe(16);
    expect(settings.outcomesPerCall).toBe(128);
  });

  test("apple defaults fit the on-device 4096-token context and manage fm serve", () => {
    process.env.LOCALJEV_BACKEND = "apple";
    const settings = loadSettings();
    expect(settings.backend).toBe("apple");
    expect(settings.managedUpstream).toBe(true);
    expect(settings.fmBinary).toBe("/usr/bin/fm");
    expect(settings.upstreamModel).toBe("system");
    expect(settings.maxOutputTokens).toBe(512);
    expect(settings.questionsPerCall).toBe(8);
    expect(settings.outcomesPerCall).toBe(32);
  });

  test("an explicit upstream means LocalJev connects instead of spawning fm serve", () => {
    process.env.LOCALJEV_BACKEND = "apple";
    process.env.LOCALJEV_UPSTREAM = "http://127.0.0.1:18976";
    const settings = loadSettings();
    expect(settings.managedUpstream).toBe(false);
    expect(settings.upstream).toBe("http://127.0.0.1:18976");
  });

  test("environment variables override every apple default", () => {
    process.env.LOCALJEV_BACKEND = "apple";
    process.env.LOCALJEV_UPSTREAM_MODEL = "custom";
    process.env.LOCALJEV_MAX_OUTPUT_TOKENS = "99";
    process.env.LOCALJEV_QUESTIONS_PER_CALL = "3";
    process.env.LOCALJEV_OUTCOMES_PER_CALL = "7";
    process.env.LOCALJEV_FM_BINARY = "/opt/fm";
    const settings = loadSettings();
    expect(settings.upstreamModel).toBe("custom");
    expect(settings.maxOutputTokens).toBe(99);
    expect(settings.questionsPerCall).toBe(3);
    expect(settings.outcomesPerCall).toBe(7);
    expect(settings.fmBinary).toBe("/opt/fm");
  });

  test("an unknown backend name fails at startup rather than silently", () => {
    process.env.LOCALJEV_BACKEND = "ollama";
    expect(() => loadSettings()).toThrow("LOCALJEV_BACKEND must be 'openai' or 'apple'");
  });

  test("selecting apple by override applies its defaults, not the openai ones", () => {
    const settings = loadSettings({ backend: "apple" });
    expect(settings.backend).toBe("apple");
    expect(settings.upstreamModel).toBe("system");
    expect(settings.maxOutputTokens).toBe(512);
    expect(settings.questionsPerCall).toBe(8);
    expect(settings.outcomesPerCall).toBe(32);
  });

  test("selecting openai by override keeps its defaults even when the env says apple", () => {
    process.env.LOCALJEV_BACKEND = "apple";
    const settings = loadSettings({ backend: "openai" });
    expect(settings.upstreamModel).toBe("diffusiongemma-26B-A4B-it-4bit");
    expect(settings.maxOutputTokens).toBe(2_048);
    expect(settings.questionsPerCall).toBe(16);
  });

  test("an explicit override still beats the backend default it derives from", () => {
    const settings = loadSettings({ backend: "apple", maxOutputTokens: 77 });
    expect(settings.maxOutputTokens).toBe(77);
    expect(settings.questionsPerCall).toBe(8);
  });

  test("an environment variable still beats an override-selected backend default", () => {
    process.env.LOCALJEV_QUESTIONS_PER_CALL = "4";
    const settings = loadSettings({ backend: "apple" });
    expect(settings.questionsPerCall).toBe(4);
    expect(settings.outcomesPerCall).toBe(32);
  });

  test("an upstream named by override connects instead of spawning fm serve", () => {
    const settings = loadSettings({
      backend: "apple",
      upstream: "http://localhost:1976",
    });
    expect(settings.managedUpstream).toBe(false);
    expect(settings.upstream).toBe("http://localhost:1976");
  });

  test("apple without any upstream still manages its own fm serve", () => {
    expect(loadSettings({ backend: "apple" }).managedUpstream).toBe(true);
  });

  test("an explicit managedUpstream override wins over the upstream it was given", () => {
    // The evaluation runner relies on this to manage a server per model.
    const settings = loadSettings({
      backend: "apple",
      upstream: "http://localhost:1976",
      managedUpstream: true,
    });
    expect(settings.managedUpstream).toBe(true);
  });
});

describe("answer mode settings", () => {
  test("openai asks the model for probabilities while apple votes, unless told otherwise", () => {
    expect(loadSettings().answerMode).toBe("probability");
    expect(loadSettings({ backend: "apple" }).answerMode).toBe("vote");
    process.env.LOCALJEV_BACKEND = "apple";
    expect(loadSettings().answerMode).toBe("vote");
  });

  test("an explicit answer mode beats the apple voting default", () => {
    process.env.LOCALJEV_BACKEND = "apple";
    process.env.LOCALJEV_ANSWER_MODE = "probability";
    expect(loadSettings().answerMode).toBe("probability");
  });

  test("voting defaults to five samples at temperature 1 so samples can differ", () => {
    const settings = loadSettings();
    expect(settings.voteSamples).toBe(5);
    expect(settings.voteTemperature).toBe(1);
  });

  test("environment variables select voting and its sampling parameters", () => {
    process.env.LOCALJEV_ANSWER_MODE = "vote";
    process.env.LOCALJEV_VOTE_SAMPLES = "9";
    process.env.LOCALJEV_VOTE_TEMPERATURE = "0.7";
    const settings = loadSettings();
    expect(settings.answerMode).toBe("vote");
    expect(settings.voteSamples).toBe(9);
    expect(settings.voteTemperature).toBe(0.7);
  });

  test("a zero vote temperature is allowed, but fewer than one sample is not", () => {
    process.env.LOCALJEV_VOTE_TEMPERATURE = "0";
    expect(loadSettings().voteTemperature).toBe(0);
    process.env.LOCALJEV_VOTE_SAMPLES = "0";
    expect(() => loadSettings()).toThrow("LOCALJEV_VOTE_SAMPLES");
  });

  test("a non-integer sample count and a negative temperature fail at startup", () => {
    process.env.LOCALJEV_VOTE_SAMPLES = "2.5";
    expect(() => loadSettings()).toThrow("LOCALJEV_VOTE_SAMPLES must be an integer");
    delete process.env.LOCALJEV_VOTE_SAMPLES;
    process.env.LOCALJEV_VOTE_TEMPERATURE = "-1";
    expect(() => loadSettings()).toThrow("LOCALJEV_VOTE_TEMPERATURE");
  });

  test("an unknown answer mode fails at startup rather than silently", () => {
    process.env.LOCALJEV_ANSWER_MODE = "majority";
    expect(() => loadSettings()).toThrow(
      "LOCALJEV_ANSWER_MODE must be 'probability' or 'vote'",
    );
  });
});
