import { createHash } from "node:crypto";
import type { Answer, Question } from "../../src/types";

export const TASKS = ["ag_news", "boolq", "sst5"] as const;
export type Task = (typeof TASKS)[number];
/** A plain string is an upstream model id; the object form selects a backend. */
export type ModelSpec =
  | string
  | {
      backend: "apple" | "openai";
      model?: string;
      answerMode?: "probability" | "vote";
      voteSamples?: number;
      voteTemperature?: number;
    };
export interface EvalConfig {
  seed: number;
  samplesPerTask: number;
  models: ModelSpec[];
  backgroundWords: number[];
  temperature: number;
  maxOutputTokens: number;
  malformedRetries: number;
  timeoutSeconds: number;
  warmupRequests: number;
  cacheMode: "bust-prefix" | "shared-prefix";
}
export interface Example {
  id: string;
  task: Task;
  row: number;
  text: string;
  question: Question;
  gold: number;
  labels: string[];
}
export interface Suite {
  version: number;
  seed: number;
  samplesPerTask: number;
  sources: Source[];
  examples: Example[];
  background: string[];
}
export interface Source {
  name: string;
  dataset: string;
  revision: string;
  split: string;
  url: string;
  sha256: string;
  format: "parquet" | "jsonl";
}
export interface Attempt {
  /** Vote sample this request belongs to; 0 in probability mode. Absent in runs recorded before vote mode existed. */
  sample?: number;
  /** Corrective retry index within that sample; 0 is the first try. */
  retry?: number;
  ms: number;
  status: number | null;
  inputTokens: number;
  outputTokens: number;
  cachedTokens: number;
  backendSeconds: number | null;
  finishReason: string | null;
  reasoningCharacters: number;
  warning: string | null;
  requestHash: string;
  responseHash: string | null;
}
export interface EvaluationRow {
  key: string;
  model: string;
  backgroundWords: number;
  task: Task;
  exampleId: string;
  gold: number;
  labels: string[];
  stateHash: string;
  stateWords: number;
  elapsedMs: number;
  ok: boolean;
  answer: Answer | null;
  probabilities: number[] | null;
  prediction: number | null;
  score: number | null;
  error: string | null;
  attempts: Attempt[];
}
export interface ResolvedModel {
  label: string;
  model: string;
  backend: "openai" | "apple";
  answerMode: "probability" | "vote";
  voteSamples: number | null;
  /** Sampling temperature for vote mode; null outside it. */
  voteTemperature: number | null;
}
export const DEFAULT_VOTE_SAMPLES = 5;
export const DEFAULT_VOTE_TEMPERATURE = 1;
export function resolveModel(spec: ModelSpec): ResolvedModel {
  if (typeof spec === "string") return { label: spec, model: spec, backend: "openai", answerMode: "probability", voteSamples: null, voteTemperature: null };
  const isApple = spec.backend === "apple";
  const model = spec.model ?? (isApple ? "system" : "");
  const answerMode = spec.answerMode ?? "probability";
  const isVote = answerMode === "vote";
  const voteSamples = isVote ? spec.voteSamples ?? DEFAULT_VOTE_SAMPLES : null;
  // Pinned in the config rather than read from the environment, so a resumed
  // run cannot silently sample at a different temperature than the first pass.
  const voteTemperature = isVote ? spec.voteTemperature ?? DEFAULT_VOTE_TEMPERATURE : null;
  // Only a vote suffix distinguishes labels, so an unannotated spec keeps the old name.
  const base = isApple ? `apple:${model}` : model;
  return { label: voteSamples === null ? base : `${base}:vote${voteSamples}`, model, backend: spec.backend, answerMode, voteSamples, voteTemperature };
}
const MODEL_SPEC_KEYS = ["backend", "model", "answerMode", "voteSamples", "voteTemperature"] as const;
function validModelSpec(spec: unknown): spec is ModelSpec {
  if (typeof spec === "string") return Boolean(spec);
  if (typeof spec !== "object" || spec === null || Array.isArray(spec)) return false;
  const entry = spec as Record<string, unknown>;
  // A typo like "answermode" would otherwise be dropped and the run would
  // measure a mode nobody asked for.
  if (Object.keys(entry).some((key) => !MODEL_SPEC_KEYS.includes(key as (typeof MODEL_SPEC_KEYS)[number]))) return false;
  const isApple = entry.backend === "apple";
  if (!isApple && entry.backend !== "openai") return false;
  const hasModelName = typeof entry.model === "string" && Boolean(entry.model);
  // An openai spec has no default model id to fall back on.
  if (!(hasModelName || (isApple && entry.model === undefined))) return false;
  const hasValidMode = entry.answerMode === undefined || entry.answerMode === "probability" || entry.answerMode === "vote";
  const hasValidSamples = entry.voteSamples === undefined || (Number.isSafeInteger(entry.voteSamples) && (entry.voteSamples as number) >= 1);
  const hasValidTemperature = entry.voteTemperature === undefined || (typeof entry.voteTemperature === "number" && Number.isFinite(entry.voteTemperature) && entry.voteTemperature >= 0);
  // Vote-only fields on a probability spec are a mistake, not a harmless extra.
  const isVote = entry.answerMode === "vote";
  const voteFieldsBelong = isVote || (entry.voteSamples === undefined && entry.voteTemperature === undefined);
  return hasValidMode && hasValidSamples && hasValidTemperature && voteFieldsBelong;
}
export function sha256(input: string | Uint8Array): string {
  return createHash("sha256").update(input).digest("hex");
}
export function hashSeed(input: string): number {
  return Number.parseInt(sha256(input).slice(0, 8), 16);
}
export function hashOrder<T>(items: T[], key: (item: T) => string, seed: string): T[] {
  return items.map((item) => ({ item, hash: sha256(`${seed}:${key(item)}`) }))
    .sort((a, b) => a.hash < b.hash ? -1 : a.hash > b.hash ? 1 : 0)
    .map(({ item }) => item);
}
export function mean(values: number[]): number | null {
  return values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
}
export function quantile(values: number[], p: number): number | null {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const index = (sorted.length - 1) * p;
  const low = Math.floor(index);
  return sorted[low]! + (sorted[Math.ceil(index)]! - sorted[low]!) * (index - low);
}
export function parseArgs(): Record<string, string> {
  const args = process.argv.slice(2);
  const out: Record<string, string> = {};
  for (let i = 0; i < args.length; i += 2) {
    if (!args[i]?.startsWith("--") || !args[i + 1] || args[i + 1]!.startsWith("--")) {
      throw new Error("Use --config PATH, --out DIRECTORY, --limit N, or --resume DIRECTORY");
    }
    const name = args[i]!.slice(2);
    if (!["config", "out", "limit", "resume"].includes(name)) throw new Error(`Unknown option: ${name}`);
    out[name] = args[i + 1]!;
  }
  return out;
}
export async function readConfig(path = "eval/default.json"): Promise<EvalConfig> {
  const c = await Bun.file(path).json() as EvalConfig;
  for (const key of ["seed", "samplesPerTask", "maxOutputTokens", "malformedRetries", "warmupRequests"] as const) {
    if (!Number.isSafeInteger(c[key]) || c[key] < 0) throw new Error(`Invalid ${key}`);
  }
  if (!c.samplesPerTask || c.samplesPerTask % 20) throw new Error("samplesPerTask must be a positive multiple of 20 (balanced 2/4/5-class tasks)");
  if (!c.maxOutputTokens || !Number.isFinite(c.timeoutSeconds) || c.timeoutSeconds <= 0) throw new Error("Invalid token limit or timeout");
  if (!Number.isFinite(c.temperature) || c.temperature < 0) throw new Error("Invalid temperature");
  if (!Array.isArray(c.models) || !c.models.length || c.models.some((m) => !validModelSpec(m))) throw new Error("models must be a nonempty list of model ids or { backend: 'apple', model?: string }");
  const labels = c.models.map((m) => resolveModel(m).label);
  if (new Set(labels).size !== labels.length) throw new Error("Duplicate models");
  if (!Array.isArray(c.backgroundWords) || !c.backgroundWords.length || c.backgroundWords.some((n) => !Number.isSafeInteger(n) || n < 0 || n > 16384)) throw new Error("backgroundWords must contain integers from 0 to 16384");
  if (new Set(c.backgroundWords).size !== c.backgroundWords.length) throw new Error("Duplicate context profiles");
  if (!["bust-prefix", "shared-prefix"].includes(c.cacheMode)) throw new Error("Invalid cacheMode");
  return c;
}
