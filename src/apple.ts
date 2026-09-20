import { randomBytes } from "node:crypto";
import { unlink } from "node:fs/promises";

import type { Settings } from "./config";
import { BackendUnavailableError, Engine } from "./engine";

// macOS caps a Unix socket path at 104 bytes including the terminator; fm serve
// exceeds it silently, creating no socket and never exiting. Why not os.tmpdir():
// it resolves to /var/folders/<...>/T/, which already spends ~50 bytes.
const SOCKET_DIRECTORY = "/tmp";
const SOCKET_PATH_LIMIT = 104;

export type Fetch = (
  input: string | URL | Request,
  init?: RequestInit,
) => Promise<Response>;

export interface Spawned {
  readonly exited: Promise<number>;
  readonly exitCode: number | null;
  readonly stderr?: ReadableStream<Uint8Array> | number | undefined;
  kill(signal?: number | NodeJS.Signals): void;
}

export type Spawn = (command: string[]) => Spawned;

export interface AppleServerOptions {
  spawn?: Spawn;
  fetchImpl?: Fetch;
  startTimeoutMs?: number;
  pollIntervalMs?: number;
  killGraceMs?: number;
  socketPath?: string;
  /** Aborts a start in progress and stops the child it already spawned. */
  signal?: AbortSignal;
  /**
   * Called immediately after the child is spawned, before it is ready. Lets the
   * caller take a last-resort reference to a process that is still starting.
   */
  onSpawn?: (server: AppleServer) => void;
}

type State = "idle" | "starting" | "ready" | "exited" | "closing" | "closed";

function log(event: string, fields: Record<string, unknown> = {}): void {
  console.log(JSON.stringify({ event, component: "apple", ...fields }));
}

/**
 * Drains `fm serve`'s stderr into one JSON log line per output line. Why not
 * stderr: "ignore": a piped stream nobody reads fills its buffer and blocks the
 * child, and discarding it would also lose the only diagnostics fm emits.
 */
async function drainStderr(stream: ReadableStream<Uint8Array>): Promise<void> {
  const decoder = new TextDecoder();
  const reader = stream.getReader();
  let pending = "";
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      pending += decoder.decode(value, { stream: true });
      const lines = pending.split("\n");
      pending = lines.pop() ?? "";
      for (const line of lines) {
        if (line.trim()) log("apple_server_stderr", { line: line.trimEnd() });
      }
    }
    if (pending.trim()) log("apple_server_stderr", { line: pending.trimEnd() });
  } catch {
    /* the child exited mid-read; its exit is reported separately */
  }
}

export function socketPathFor(directory = SOCKET_DIRECTORY): string {
  const path = `${directory}/localjev-${process.pid}-${randomBytes(4).toString("hex")}.sock`;
  const isTooLong = Buffer.byteLength(path) >= SOCKET_PATH_LIMIT;
  if (isTooLong) {
    throw new BackendUnavailableError(
      `Unix socket path ${JSON.stringify(path)} is ${Buffer.byteLength(path)} bytes; macOS allows fewer than ${SOCKET_PATH_LIMIT}`,
    );
  }
  return path;
}

/**
 * Owns a `fm serve --socket` child process: starts it, waits for /health, and
 * exposes a fetch that reaches it over the Unix socket.
 */
export class AppleServer {
  readonly socketPath: string;
  private readonly spawn: Spawn;
  private readonly fetchImpl: Fetch;
  private readonly startTimeoutMs: number;
  private readonly pollIntervalMs: number;
  private readonly killGraceMs: number;
  private readonly signal: AbortSignal | undefined;
  private readonly onSpawn: ((server: AppleServer) => void) | undefined;
  private state: State = "idle";
  private child: Spawned | null = null;
  // One shared promise so concurrent close() callers await the same shutdown
  // rather than each sending their own SIGTERM to a process that may be gone.
  private closePromise: Promise<void> | null = null;
  // Concurrent start() callers join the first attempt instead of racing to
  // spawn a second child on the same socket.
  private startPromise: Promise<void> | null = null;

  constructor(
    private readonly settings: Settings,
    options: AppleServerOptions = {},
  ) {
    this.socketPath = options.socketPath ?? socketPathFor();
    this.spawn =
      options.spawn ??
      ((command) => Bun.spawn(command, { stdout: "ignore", stderr: "pipe" }));
    this.fetchImpl = options.fetchImpl ?? ((input, init) => fetch(input, init));
    this.startTimeoutMs = options.startTimeoutMs ?? 30_000;
    this.pollIntervalMs = options.pollIntervalMs ?? 100;
    this.killGraceMs = options.killGraceMs ?? 2_000;
    this.signal = options.signal;
    this.onSpawn = options.onSpawn;
  }

  /** False once the child has exited or been stopped; the engine then fails fast. */
  running(): boolean {
    return this.state === "ready";
  }

  start(): Promise<void> {
    if (this.state === "ready") return Promise.resolve();
    // A concurrent caller joins the attempt already running rather than
    // spawning a rival child on the same socket. Once the server has reached a
    // terminal state that attempt is history, and starting again is an error.
    const isStarting = this.state === "starting";
    if (isStarting && this.startPromise) return this.startPromise;
    const isSpent = this.state !== "idle";
    if (isSpent) {
      return Promise.reject(
        new BackendUnavailableError(
          `${this.settings.fmBinary} serve was already stopped; create a new AppleServer`,
        ),
      );
    }
    this.startPromise = this.startOnce();
    return this.startPromise;
  }

  private async startOnce(): Promise<void> {
    if (this.signal?.aborted) {
      // Never spawned, so there is nothing to stop; just refuse.
      this.state = "closed";
      throw new BackendUnavailableError(
        `${this.settings.fmBinary} serve was stopped while starting`,
      );
    }
    this.state = "starting";
    // A local reference: close() may null the field out mid-startup, and every
    // failure path below must still act on the child it actually spawned.
    let child: Spawned;
    try {
      child = this.spawn([
        this.settings.fmBinary,
        "serve",
        "--socket",
        this.socketPath,
      ]);
    } catch (error) {
      // A missing or unexecutable fm binary: there is no child to clean up, but
      // the server must still end up in a terminal state.
      this.state = "closed";
      this.closePromise = Promise.resolve();
      throw new BackendUnavailableError(
        `${this.settings.fmBinary} serve could not be started: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
    this.child = child;
    log("apple_server_spawned", {
      binary: this.settings.fmBinary,
      socket: this.socketPath,
    });
    const errors = child.stderr;
    if (errors instanceof ReadableStream) void drainStderr(errors);

    let exitCode: number | null = null;
    let exitedEarly = false;
    void child.exited.then((code) => {
      exitedEarly = true;
      exitCode = code;
    });
    const onAbort = () => void this.close();
    this.signal?.addEventListener("abort", onAbort, { once: true });

    const deadline = Date.now() + this.startTimeoutMs;
    try {
      // Why inside the try: a caller's hook that throws must still stop the
      // child it was told about, not leave it running behind a rejected start.
      this.onSpawn?.(this);
      for (;;) {
        // Re-checked after every await: close() may have run while the health
        // probe was in flight, and answering 200 afterwards must not revive it.
        const wasClosed = this.state !== "starting";
        if (wasClosed) {
          throw new BackendUnavailableError(
            `${this.settings.fmBinary} serve was stopped while starting`,
          );
        }
        if (exitedEarly) {
          throw new BackendUnavailableError(
            `${this.settings.fmBinary} serve exited with code ${exitCode ?? child.exitCode} before becoming ready`,
          );
        }
        const remainingMs = deadline - Date.now();
        if (remainingMs <= 0) {
          throw new BackendUnavailableError(
            `${this.settings.fmBinary} serve did not become ready within ${this.startTimeoutMs}ms`,
          );
        }
        const isHealthy = await this.healthy(remainingMs);
        const closedMeanwhile = this.state !== "starting";
        if (closedMeanwhile) {
          throw new BackendUnavailableError(
            `${this.settings.fmBinary} serve was stopped while starting`,
          );
        }
        if (isHealthy && !exitedEarly) break;
        await Bun.sleep(Math.min(this.pollIntervalMs, Math.max(1, remainingMs)));
      }
    } catch (error) {
      // Why route through close(): a failed start and a concurrent close()
      // would otherwise each call stopChild() and send SIGTERM twice.
      await this.close();
      throw error;
    } finally {
      this.signal?.removeEventListener("abort", onAbort);
    }
    this.state = "ready";
    log("apple_server_ready", { socket: this.socketPath });
    this.watchExit(child);
  }

  /**
   * Notes an unexpected exit after startup. Why not restart automatically: a
   * crash loop hidden behind retries is harder to diagnose than a server that
   * reports itself unavailable and lets the process manager decide.
   */
  private watchExit(child: Spawned): void {
    void child.exited.then((code) => {
      const isExpected = this.state === "closing" || this.state === "closed";
      if (isExpected) return;
      this.state = "exited";
      this.child = null;
      log("apple_server_exited", { socket: this.socketPath, exitCode: code });
    });
  }

  /**
   * Probes /health with the time the caller has left. Why two signals: a socket
   * that accepts the connection and then never answers would otherwise hang
   * past the start deadline, and a shutdown must not wait out the probe either.
   */
  private async healthy(remainingMs: number): Promise<boolean> {
    const timeout = AbortSignal.timeout(
      Math.max(1, Math.min(remainingMs, 5_000)),
    );
    const signal = this.signal
      ? AbortSignal.any([timeout, this.signal])
      : timeout;
    try {
      const response = await this.fetch("http://localhost/health", { signal });
      return response.ok;
    } catch {
      return false;
    }
  }

  /**
   * Kills the child immediately, skipping the SIGTERM grace period. Only for a
   * shutdown that has run out of time: the process is leaving regardless, and
   * an orphaned `fm serve` is worse than an ungraceful stop.
   */
  kill(): void {
    const child = this.child;
    if (!child) return;
    // Reaches a child that close() is still waiting out, which is the case this
    // exists for. `this.child` stays set so a concurrent shutdown can finish
    // its own bookkeeping; a second SIGKILL would be harmless anyway.
    try {
      child.kill("SIGKILL");
      log("apple_server_killed", { socket: this.socketPath });
    } catch {
      /* already gone */
    }
  }

  fetch: Fetch = (input, init) =>
    this.fetchImpl(input, { ...init, unix: this.socketPath } as RequestInit);

  close(): Promise<void> {
    this.closePromise ??= this.shutdown();
    return this.closePromise;
  }

  private async shutdown(): Promise<void> {
    const child = this.child;
    this.state = "closing";
    // Why not clear `this.child` here: a shutdown deadline can expire while the
    // SIGTERM grace period is still running, and kill() must still reach the
    // child. It is cleared only once the child is known to be gone.
    await this.stopChild(child);
    this.child = null;
    this.state = "closed";
  }

  private async stopChild(child: Spawned | null): Promise<void> {
    if (child) {
      child.kill("SIGTERM");
      const timer = Bun.sleep(this.killGraceMs).then(() => "timeout" as const);
      const outcome = await Promise.race([
        child.exited.then(() => "exited" as const),
        timer,
      ]);
      if (outcome === "timeout") {
        child.kill("SIGKILL");
        // Bounded: a child that survives SIGKILL is unkillable (uninterruptible
        // wait), and blocking here forever would hold the whole shutdown.
        await Promise.race([child.exited, Bun.sleep(this.killGraceMs)]);
      }
      log("apple_server_stopped", { socket: this.socketPath, outcome });
    }
    // fm removes the socket itself on a clean exit; this covers SIGKILL and
    // startup failures so a stale file never blocks the next run.
    try {
      await unlink(this.socketPath);
    } catch {
      /* already gone */
    }
  }
}

export interface CreatedEngine {
  engine: Engine;
  appleServer: AppleServer | null;
}

/**
 * Builds the Engine for the configured backend, starting a managed `fm serve`
 * when the apple backend has no explicit upstream to connect to.
 */
export async function createEngine(
  settings: Settings,
  options: AppleServerOptions = {},
): Promise<CreatedEngine> {
  if (!settings.managedUpstream) {
    return { engine: new Engine(settings), appleServer: null };
  }
  const appleServer = new AppleServer(settings, options);
  await appleServer.start();
  const engine = new Engine(settings, appleServer.fetch, {
    onClose: () => appleServer.close(),
    isAvailable: () => appleServer.running(),
  });
  return { engine, appleServer };
}
