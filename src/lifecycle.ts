export type SignalName = "SIGINT" | "SIGTERM";

export interface LifecycleDependencies {
  onSignal(signal: SignalName, handler: () => void): void;
  offSignal(signal: SignalName, handler: () => void): void;
  exit(code: number): void;
  log(message: string): void;
  /** Injected so tests drive the deadline without waiting real time. */
  setTimer(callback: () => void, ms: number): unknown;
  clearTimer(handle: unknown): void;
  /** How long the whole interrupted shutdown may take before the process leaves. */
  shutdownTimeoutMs?: number;
}

export const defaultLifecycleDependencies: LifecycleDependencies = {
  onSignal: (signal, handler) => {
    process.on(signal, handler);
  },
  offSignal: (signal, handler) => {
    process.off(signal, handler);
  },
  exit: (code) => process.exit(code),
  log: (message) => console.log(message),
  setTimer: (callback, ms) => {
    const handle = setTimeout(callback, ms);
    // unref keeps a pending deadline from holding an otherwise finished process
    // open; it is not a cancellation, so clearTimer still has to clear it.
    handle.unref?.();
    return handle;
  },
  clearTimer: (handle) => clearTimeout(handle as Parameters<typeof clearTimeout>[0]),
};

export const SHUTDOWN_TIMEOUT_MS = 10_000;
/** Exit status for a run cut short by a signal, distinct from a clean stop. */
export const INTERRUPTED_EXIT_CODE = 130;

export function logEvent(
  log: (message: string) => void,
  event: string,
  fields: Record<string, unknown> = {},
): void {
  log(JSON.stringify({ event, component: "lifecycle", ...fields }));
}

/**
 * What a lifecycle body is given: a signal that fires the moment a shutdown is
 * requested, and a way to register the teardown that must run before exit.
 */
export interface Lifecycle {
  /** Aborts on the first signal — during startup and during the body alike. */
  readonly signal: AbortSignal;
  /** True once a signal has been seen. */
  interrupted(): boolean;
  /** Resolves on the first signal; never rejects. A server body awaits this. */
  whenSignalled(): Promise<void>;
  /** True once the shutdown deadline has fired. */
  expired(): boolean;
  /**
   * Exits the process at most once for this lifetime. A deadline that already
   * exited wins: a later normal-path exit is dropped rather than overriding
   * the interrupted status.
   */
  exit(code: number): void;
  /**
   * Registers a teardown step, innermost last. Steps run in reverse order once,
   * whatever ends the run, and a step that throws is logged, not propagated.
   */
  onCleanup(step: () => Promise<void> | void, name: string): void;
  /**
   * Registers a last resort for when the deadline expires with cleanup still
   * running: the process is leaving either way, so this is the final chance to
   * stop a child (SIGKILL) rather than orphan it.
   */
  onDeadline(step: () => void, name: string): void;
}

/** The mutable state behind a `Lifecycle`; `withLifecycle` adapts it. */
class LifecycleScope {
  private readonly controller = new AbortController();
  private readonly steps: { run: () => Promise<void> | void; name: string }[] = [];
  private readonly lastResorts: { run: () => void; name: string }[] = [];
  private readonly handlers: (() => void)[] = [];
  private timer: unknown = null;
  private cleanupPromise: Promise<void> | null = null;
  private sawSignal = false;
  private expired = false;
  private released = false;
  private exited = false;

  constructor(
    private readonly dependencies: LifecycleDependencies,
    private readonly timeoutMs: number,
  ) {}

  get signal(): AbortSignal {
    return this.controller.signal;
  }

  interrupted(): boolean {
    return this.sawSignal;
  }

  onCleanup(step: () => Promise<void> | void, name: string): void {
    this.steps.push({ run: step, name });
  }

  onDeadline(step: () => void, name: string): void {
    this.lastResorts.push({ run: step, name });
  }

  /** Attaches the signal handlers. Called before anything can be spawned. */
  listen(onFirstSignal: () => void): void {
    const handle = () => {
      const isRepeat = this.sawSignal;
      this.sawSignal = true;
      this.controller.abort();
      if (isRepeat) return;
      // The deadline covers everything a signal sets in motion — aborting a
      // startup as well as unwinding a running server — because the caller's
      // own grace period is already ticking.
      this.startDeadline();
      onFirstSignal();
    };
    for (const name of ["SIGINT", "SIGTERM"] as const) {
      this.dependencies.onSignal(name, handle);
      this.handlers.push(() => this.dependencies.offSignal(name, handle));
    }
  }

  private startDeadline(): void {
    this.timer = this.dependencies.setTimer(() => {
      // Belt and braces alongside clearTimer: a timer implementation that
      // cannot cancel must still not fire into a scope that already finished.
      if (this.released) return;
      this.expired = true;
      logEvent(this.dependencies.log, "shutdown_deadline_expired", {
        timeoutMs: this.timeoutMs,
      });
      // Cleanup is still running and out of time. Take the child down by force
      // rather than let the process exit around it. Every hook gets its turn
      // even if an earlier one throws.
      for (const { run, name } of this.lastResorts) {
        try {
          run();
        } catch (error) {
          logEvent(this.dependencies.log, "shutdown_last_resort_failed", {
            step: name,
            error: error instanceof Error ? error.message : String(error),
          });
        }
      }
      this.exit(INTERRUPTED_EXIT_CODE);
    }, this.timeoutMs);
  }

  /** Exits at most once, so a late normal path cannot override the deadline. */
  exit(code: number): void {
    if (this.exited) return;
    this.exited = true;
    this.dependencies.exit(code);
  }

  hasExited(): boolean {
    return this.exited;
  }

  /** Runs every registered step once, innermost first. Never rejects. */
  cleanup(): Promise<void> {
    this.cleanupPromise ??= (async () => {
      for (const { run, name } of [...this.steps].reverse()) {
        try {
          await run();
        } catch (error) {
          logEvent(this.dependencies.log, "shutdown_step_failed", {
            step: name,
            error: error instanceof Error ? error.message : String(error),
          });
        }
      }
    })();
    return this.cleanupPromise;
  }

  release(): void {
    this.released = true;
    for (const off of this.handlers) off();
    this.handlers.length = 0;
    if (this.timer !== null) this.dependencies.clearTimer(this.timer);
    this.timer = null;
  }

  deadlineExpired(): boolean {
    return this.expired;
  }
}

/**
 * Runs `body` inside one managed lifetime: signals are caught from before the
 * body starts, the first one aborts `lifecycle.signal` and starts the shutdown
 * deadline, registered cleanup runs exactly once however the body ends, and the
 * handlers and timer are always released afterwards.
 *
 * Why one component for both entry points: `run()` and the smoke script kept
 * growing their own flags for the same four concerns, and each round of review
 * found a different hole in one copy or the other.
 */
export async function withLifecycle<T>(
  body: (lifecycle: Lifecycle) => Promise<T>,
  dependencies: LifecycleDependencies = defaultLifecycleDependencies,
): Promise<T> {
  const timeoutMs = dependencies.shutdownTimeoutMs ?? SHUTDOWN_TIMEOUT_MS;
  const scope = new LifecycleScope(dependencies, timeoutMs);
  let signalled: () => void = () => {};
  const arrived = new Promise<void>((resolve) => {
    signalled = resolve;
  });
  scope.listen(() => signalled());
  const lifecycle: Lifecycle = {
    signal: scope.signal,
    interrupted: () => scope.interrupted(),
    whenSignalled: () => arrived,
    expired: () => scope.deadlineExpired(),
    exit: (code) => scope.exit(code),
    onCleanup: (step, name) => scope.onCleanup(step, name),
    onDeadline: (step, name) => scope.onDeadline(step, name),
  };

  try {
    return await body(lifecycle);
  } finally {
    // Runs on every path — success, throw, or signal — and only once, because
    // the scope memoizes it. Handlers stay attached until it has finished.
    await scope.cleanup();
    scope.release();
  }
}
