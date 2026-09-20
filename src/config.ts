export type Backend = "openai" | "apple";
export type AnswerMode = "probability" | "vote";

export interface Settings {
  backend: Backend;
  answerMode: AnswerMode;
  voteSamples: number;
  voteTemperature: number;
  fmBinary: string;
  managedUpstream: boolean;
  upstream: string;
  upstreamApiKey: string;
  upstreamModel: string;
  apiKey: string;
  host: string;
  port: number;
  timeoutMs: number;
  maxOutputTokens: number;
  malformedRetries: number;
  temperature: number;
  maxInflight: number;
  maxQueue: number;
  questionsPerCall: number;
  outcomesPerCall: number;
}

function numberSetting(name: string, fallback: number, minimum: number): number {
  const raw = process.env[name];
  const value = raw === undefined ? fallback : Number(raw);
  if (!Number.isFinite(value) || value < minimum) {
    throw new Error(`${name} must be a number greater than or equal to ${minimum}`);
  }
  return value;
}

function integerSetting(name: string, fallback: number, minimum: number): number {
  const value = numberSetting(name, fallback, minimum);
  if (!Number.isInteger(value)) {
    throw new Error(`${name} must be an integer`);
  }
  return value;
}

function backendSetting(): Backend {
  const raw = process.env.LOCALJEV_BACKEND ?? "openai";
  if (raw !== "openai" && raw !== "apple") {
    throw new Error("LOCALJEV_BACKEND must be 'openai' or 'apple'");
  }
  return raw;
}

function answerModeSetting(fallback: AnswerMode): AnswerMode {
  const raw = process.env.LOCALJEV_ANSWER_MODE ?? fallback;
  if (raw !== "probability" && raw !== "vote") {
    throw new Error("LOCALJEV_ANSWER_MODE must be 'probability' or 'vote'");
  }
  return raw;
}

// The on-device model has a 4096-token context, so the apple backend asks for
// smaller batches than the oMLX defaults. Why not one shared default: a single
// conservative number would needlessly slow the roomier oMLX upstream down.
// Why not probability mode for apple too: measured on 120 labelled examples, the
// on-device model saturates self-reported probabilities to 0/1 and sometimes
// returns all zeros (35.8% macro accuracy against 52.5% for five-sample voting).
const APPLE_DEFAULTS = {
  answerMode: "vote",
  upstreamModel: "system",
  maxOutputTokens: 512,
  questionsPerCall: 8,
  outcomesPerCall: 32,
} as const;

export function loadSettings(
  overrides: Partial<Settings> = {},
): Settings {
  // The backend is resolved first because every derived default below depends
  // on it; an override that only names the backend must still get its defaults.
  const backend = overrides.backend ?? backendSetting();
  const isApple = backend === "apple";
  // Resolved before managedUpstream for the same reason as the backend: naming
  // an upstream explicitly means "connect there", however it was named.
  const upstream = overrides.upstream ?? process.env.LOCALJEV_UPSTREAM;
  return {
    backend,
    answerMode: answerModeSetting(
      isApple ? APPLE_DEFAULTS.answerMode : "probability",
    ),
    voteSamples: integerSetting("LOCALJEV_VOTE_SAMPLES", 5, 1),
    // Why not reuse LOCALJEV_TEMPERATURE: it defaults to 0, and identical
    // samples would make every vote unanimous regardless of real uncertainty.
    voteTemperature: numberSetting("LOCALJEV_VOTE_TEMPERATURE", 1, 0),
    fmBinary: process.env.LOCALJEV_FM_BINARY ?? "/usr/bin/fm",
    managedUpstream: isApple && upstream === undefined,
    upstream: upstream ?? "http://127.0.0.1:8000",
    upstreamApiKey: process.env.LOCALJEV_UPSTREAM_API_KEY ?? "",
    upstreamModel:
      process.env.LOCALJEV_UPSTREAM_MODEL ??
      (isApple ? APPLE_DEFAULTS.upstreamModel : "diffusiongemma-26B-A4B-it-4bit"),
    apiKey: process.env.LOCALJEV_API_KEY ?? "",
    host: process.env.LOCALJEV_HOST ?? "127.0.0.1",
    port: integerSetting("LOCALJEV_PORT", 8080, 1),
    timeoutMs: numberSetting("LOCALJEV_TIMEOUT", 180, 0.001) * 1_000,
    maxOutputTokens: integerSetting(
      "LOCALJEV_MAX_OUTPUT_TOKENS",
      isApple ? APPLE_DEFAULTS.maxOutputTokens : 2_048,
      1,
    ),
    malformedRetries: integerSetting("LOCALJEV_MALFORMED_RETRIES", 2, 0),
    temperature: numberSetting("LOCALJEV_TEMPERATURE", 0, 0),
    maxInflight: integerSetting("LOCALJEV_MAX_INFLIGHT", 2, 1),
    maxQueue: integerSetting("LOCALJEV_MAX_QUEUE", 64, 1),
    questionsPerCall: integerSetting(
      "LOCALJEV_QUESTIONS_PER_CALL",
      isApple ? APPLE_DEFAULTS.questionsPerCall : 16,
      1,
    ),
    outcomesPerCall: integerSetting(
      "LOCALJEV_OUTCOMES_PER_CALL",
      isApple ? APPLE_DEFAULTS.outcomesPerCall : 128,
      1,
    ),
    ...overrides,
  };
}

export function apiBaseUrl(settings: Settings): string {
  const base = settings.upstream.replace(/\/+$/, "");
  return base.endsWith("/v1") ? base : `${base}/v1`;
}

export const MODEL_VERSION = "localjev-0.2";
export const MODEL_ALIASES = new Set([
  MODEL_VERSION,
  "localjev-latest",
  "jev-latest",
  "jev-preview",
]);
export const MODELS = [
  {
    name: "localjev-latest",
    description:
      "Alias for LocalJev 0.2, backed by a local inference backend.",
    release_date: "2026-09-18",
  },
  {
    name: MODEL_VERSION,
    description:
      "Jev-compatible prompted probability inference with a local model.",
    release_date: "2026-09-18",
  },
] as const;
