import { createEngine } from "./apple";
import type { Settings } from "./config";
import type { DecisionEngine } from "./engine";
import {
  INTERRUPTED_EXIT_CODE,
  type Lifecycle,
  type LifecycleDependencies,
  defaultLifecycleDependencies,
  logEvent,
  withLifecycle,
} from "./lifecycle";
import { LocalJevApp } from "./server";

export interface RunningServer {
  url: { toString(): string };
  stop(closeActiveConnections?: boolean): Promise<void> | void;
}

export interface ServeOptions {
  hostname: string;
  port: number;
  idleTimeout: number;
  fetch(request: Request): Promise<Response>;
}

export interface CreatedBackend {
  engine: DecisionEngine & { close?(): Promise<void> };
  /** Last resort when the shutdown deadline expires mid-cleanup. */
  kill?(): void;
}

export interface RunDependencies extends LifecycleDependencies {
  serve(options: ServeOptions): RunningServer;
  /**
   * `signal` aborts a startup still in progress, child processes included.
   * `onSpawn` is called as soon as a child exists, before the backend is ready,
   * so a deadline expiring mid-startup can still reach it.
   */
  createBackend(
    settings: Settings,
    signal: AbortSignal,
    onSpawn: (kill: () => void) => void,
  ): Promise<CreatedBackend>;
}

export const defaultDependencies: RunDependencies = {
  ...defaultLifecycleDependencies,
  serve: (options) => Bun.serve(options) as unknown as RunningServer,
  createBackend: async (settings, signal, onSpawn) => {
    const { engine, appleServer } = await createEngine(settings, {
      signal,
      onSpawn: (server) => onSpawn(() => server.kill()),
    });
    return { engine, ...(appleServer ? { kill: () => appleServer.kill() } : {}) };
  },
};

/**
 * Brings up the backend under `lifecycle` and registers its teardown. Returns
 * null when a signal arrived before the backend was usable, in which case the
 * caller should simply stop.
 */
async function startBackend(
  settings: Settings,
  lifecycle: Lifecycle,
  dependencies: RunDependencies,
): Promise<LocalJevApp | null> {
  if (lifecycle.interrupted()) return null;
  let created: CreatedBackend;
  try {
    created = await dependencies.createBackend(
      settings,
      lifecycle.signal,
      // Registered the moment the child exists: a deadline that expires before
      // the backend is ready must still be able to SIGKILL it.
      (kill) => lifecycle.onDeadline(kill, "backend kill (starting)"),
    );
  } catch (error) {
    // An aborted startup is the signal doing its job, not a failure to report.
    if (lifecycle.interrupted()) return null;
    throw error;
  }
  const app = new LocalJevApp(settings, created.engine);
  lifecycle.onCleanup(() => app.close(), "backend close");
  if (created.kill) lifecycle.onDeadline(created.kill, "backend kill");
  // The signal landed while the backend was starting; the child exists now, so
  // the cleanup registered just above is what stops it.
  if (lifecycle.interrupted()) return null;
  return app;
}

/**
 * Starts the HTTP server and keeps the managed backend's lifetime tied to the
 * process. Everything about signals, deadlines and teardown lives in
 * `withLifecycle`; this only describes what to bring up and what to wait for.
 */
export async function run(
  settings: Settings,
  dependencies: RunDependencies = defaultDependencies,
): Promise<void> {
  let scope: Lifecycle | null = null;
  await withLifecycle(async (lifecycle) => {
    scope = lifecycle;
    const app = await startBackend(settings, lifecycle, dependencies);
    // Interrupted before the server existed: cleanup still runs on the way out.
    if (!app) return;

    const server = dependencies.serve({
      hostname: settings.host,
      port: settings.port,
      idleTimeout: 255,
      fetch: (request) => app.fetch(request),
    });
    // stop(true) closes active connections: waiting for an in-flight decision
    // would routinely outlast the signal and leave the child behind.
    lifecycle.onCleanup(() => server.stop(true), "server stop");
    dependencies.log(`LocalJev listening on ${server.url.toString()}`);

    await lifecycle.whenSignalled();
  }, dependencies);
  // Routed through the lifecycle so a deadline that already exited with 130
  // is not overwritten by this 0 when cleanup finishes afterwards.
  const lifecycle = scope as Lifecycle | null;
  if (lifecycle) lifecycle.exit(0);
  else dependencies.exit(0);
}

export interface OwnedBackendOptions extends Partial<LifecycleDependencies> {}

/**
 * Runs `body` with a managed backend whose child process is stopped on the way
 * out, including when a signal arrives mid-startup or mid-inference. Shared by
 * the smoke script so it does not reimplement the teardown `run()` already has.
 *
 * `body` receives the lifecycle's `signal`: passing it to an upstream request
 * is what lets a signal interrupt inference against a backend LocalJev does not
 * own, such as a hung OpenAI-compatible server. A `body` that ignores the
 * signal still lets this helper return on interruption, but its in-flight
 * upstream request keeps running until the upstream's own timeout ends it.
 */
export async function withManagedBackend<T>(
  settings: Settings,
  body: (engine: CreatedBackend["engine"], signal: AbortSignal) => Promise<T>,
  options: OwnedBackendOptions = {},
): Promise<T> {
  const dependencies: LifecycleDependencies = {
    ...defaultLifecycleDependencies,
    ...options,
  };
  return withLifecycle(async (lifecycle) => {
    let created: Awaited<ReturnType<typeof createEngine>>;
    try {
      created = await createEngine(settings, {
        signal: lifecycle.signal,
        onSpawn: (server) =>
          lifecycle.onDeadline(() => server.kill(), "backend kill (starting)"),
      });
    } catch (error) {
      // A start the signal cut short is an interruption, not a startup failure:
      // the caller turns that into the same exit code as every other one.
      if (lifecycle.interrupted()) {
        logEvent(dependencies.log, "startup_interrupted", { reason: "signal" });
        throw new InterruptedError("Interrupted while starting the backend");
      }
      throw error;
    }
    lifecycle.onCleanup(() => created.engine.close(), "backend close");
    if (created.appleServer) {
      lifecycle.onDeadline(() => created.appleServer?.kill(), "backend kill");
    }
    if (lifecycle.interrupted()) {
      logEvent(dependencies.log, "startup_interrupted", { reason: "signal" });
      throw new InterruptedError("Interrupted before the first request");
    }
    // Whichever finishes first: a signal must not wait out an inference against
    // an upstream that may never answer.
    const outcome = await Promise.race([
      body(created.engine, lifecycle.signal).then((value) => ({ value })),
      lifecycle.whenSignalled().then(() => ({ interrupted: true as const })),
    ]);
    if ("interrupted" in outcome) {
      logEvent(dependencies.log, "run_interrupted", { reason: "signal" });
      throw new InterruptedError("Interrupted by a signal");
    }
    return outcome.value;
  }, dependencies);
}

/**
 * A run cut short by a signal, raised however far the run had got: while the
 * backend was still starting, or with a request already in flight. A one-shot
 * script turns this into `INTERRUPTED_EXIT_CODE`, because the work it was asked
 * to do did not happen.
 *
 * Why `run()` exits 0 for the same signal: stopping a server on SIGTERM is the
 * requested outcome, not a failure to produce a result.
 */
export class InterruptedError extends Error {}

export { INTERRUPTED_EXIT_CODE };
