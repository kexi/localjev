import { describe, expect, test } from "bun:test";

import { loadSettings } from "../src/config";
import type { DecisionEngine } from "../src/engine";
import {
  INTERRUPTED_EXIT_CODE,
  defaultLifecycleDependencies,
} from "../src/lifecycle";
import {
  InterruptedError,
  maxRequestBodySize,
  run,
  withManagedBackend,
  type RunDependencies,
  type RunningServer,
} from "../src/main";

type SignalName = "SIGINT" | "SIGTERM";

/** A promise whose settlement the test controls, so ordering never uses sleep. */
function deferred<T = void>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

interface Harness {
  dependencies: RunDependencies;
  closes: number;
  kills: number;
  exits: number[];
  stops: (boolean | undefined)[];
  logs: string[];
  startAborted: boolean;
  /** Resolves once createBackend has been entered. */
  backendStarting: Promise<void>;
  raise(signal: SignalName): void;
  handlerCount(): number;
  /** Fires the pending shutdown deadline, as the real timer would. */
  fireDeadline(): void;
  pendingTimers(): number;
}

interface HarnessOptions {
  serveThrows?: Error;
  createBackendThrows?: Error;
  /** Held open to keep createBackend in flight. */
  startGate?: Promise<unknown>;
  closeGate?: Promise<unknown>;
  closeThrows?: Error;
  stopGate?: Promise<unknown>;
  shutdownTimeoutMs?: number;
}

function harness(options: HarnessOptions = {}): Harness {
  const handlers = new Map<SignalName, Set<() => void>>();
  const timers: (() => void)[] = [];
  const entered = deferred();
  const state = {
    closes: 0,
    kills: 0,
    exits: [] as number[],
    stops: [] as (boolean | undefined)[],
    logs: [] as string[],
    startAborted: false,
  };

  const engine: DecisionEngine = {
    async decide() {
      throw new Error("not used");
    },
    async close() {
      if (options.closeGate) await options.closeGate;
      state.closes += 1;
      if (options.closeThrows) throw options.closeThrows;
    },
  };

  const server: RunningServer = {
    url: { toString: () => "http://127.0.0.1:8080/" },
    async stop(closeActiveConnections) {
      state.stops.push(closeActiveConnections);
      if (options.stopGate) await options.stopGate;
    },
  };

  const result: Harness = {
    get closes() {
      return state.closes;
    },
    get kills() {
      return state.kills;
    },
    get exits() {
      return state.exits;
    },
    get stops() {
      return state.stops;
    },
    get logs() {
      return state.logs;
    },
    get startAborted() {
      return state.startAborted;
    },
    backendStarting: entered.promise,
    raise(signal) {
      for (const handler of [...(handlers.get(signal) ?? [])]) handler();
    },
    handlerCount: () =>
      [...handlers.values()].reduce((sum, set) => sum + set.size, 0),
    fireDeadline() {
      const fire = timers.shift();
      if (!fire) throw new Error("no shutdown deadline is pending");
      fire();
    },
    pendingTimers: () => timers.length,
    dependencies: {
      serve: () => {
        if (options.serveThrows) throw options.serveThrows;
        return server;
      },
      createBackend: async (_settings, signal) => {
        entered.resolve();
        signal.addEventListener("abort", () => {
          state.startAborted = true;
        });
        if (options.createBackendThrows) throw options.createBackendThrows;
        if (options.startGate) await options.startGate;
        return { engine, kill: () => { state.kills += 1; } };
      },
      onSignal: (signal, handler) => {
        const set = handlers.get(signal) ?? new Set();
        set.add(handler);
        handlers.set(signal, set);
      },
      offSignal: (signal, handler) => {
        handlers.get(signal)?.delete(handler);
      },
      exit: (code) => {
        state.exits.push(code);
      },
      log: (message) => {
        state.logs.push(message);
      },
      setTimer: (callback) => {
        timers.push(callback);
        return timers.length;
      },
      clearTimer: () => {
        timers.length = 0;
      },
      ...(options.shutdownTimeoutMs === undefined
        ? {}
        : { shutdownTimeoutMs: options.shutdownTimeoutMs }),
    },
  };
  return result;
}

const settings = () => loadSettings({ backend: "openai" });

describe("process startup and shutdown", () => {
  test("a port already in use closes the backend instead of orphaning it", async () => {
    const h = harness({ serveThrows: new Error("EADDRINUSE: port 8081 in use") });
    await expect(run(settings(), h.dependencies)).rejects.toThrow("EADDRINUSE");
    expect(h.closes).toBe(1);
    expect(h.handlerCount()).toBe(0);
  });

  test("a signal during backend startup aborts the start rather than waiting it out", async () => {
    const start = deferred();
    const h = harness({ startGate: start.promise });
    const running = run(settings(), h.dependencies);
    await h.backendStarting;
    h.raise("SIGTERM");
    // The abort is what stops fm serve; the start only finishes afterwards.
    expect(h.startAborted).toBe(true);
    start.resolve();
    await running;
    expect(h.closes).toBe(1);
    expect(h.exits).toEqual([0]);
    expect(h.stops).toEqual([]);
    expect(h.handlerCount()).toBe(0);
  });

  test("a startup that never settles still exits when the deadline expires", async () => {
    // createBackend ignores its abort, as a wedged child process would.
    const h = harness({ startGate: deferred().promise, shutdownTimeoutMs: 10 });
    void run(settings(), h.dependencies);
    await h.backendStarting;
    h.raise("SIGTERM");
    // The deadline is armed from the first signal, not from serve().
    expect(h.pendingTimers()).toBe(1);
    h.fireDeadline();
    expect(h.exits).toEqual([INTERRUPTED_EXIT_CODE]);
    expect(h.logs.join(" ")).toContain("shutdown_deadline_expired");
  });

  test("an expired deadline kills the child rather than orphaning it", async () => {
    const h = harness({ closeGate: deferred().promise, shutdownTimeoutMs: 10 });
    const running = run(settings(), h.dependencies);
    await h.backendStarting;
    await Promise.resolve();
    h.raise("SIGTERM");
    await Promise.resolve();
    h.fireDeadline();
    // close() is still hanging, so the last resort is the only thing that can
    // stop the child before the process leaves.
    expect(h.kills).toBe(1);
    expect(h.exits).toEqual([INTERRUPTED_EXIT_CODE]);
    void running;
  });

  test("a startup aborted by a signal exits cleanly instead of reporting a failure", async () => {
    const h = harness({ createBackendThrows: new Error("serve was stopped while starting") });
    const dependencies: RunDependencies = {
      ...h.dependencies,
      createBackend: async (_settings, signal) => {
        h.raise("SIGTERM");
        expect(signal.aborted).toBe(true);
        throw new Error("serve was stopped while starting");
      },
    };
    await run(settings(), dependencies);
    expect(h.exits).toEqual([0]);
    expect(h.handlerCount()).toBe(0);
  });

  test("a backend that fails to start leaves no signal handler behind", async () => {
    const h = harness({ createBackendThrows: new Error("fm serve did not become ready") });
    await expect(run(settings(), h.dependencies)).rejects.toThrow("did not become ready");
    expect(h.closes).toBe(0);
    expect(h.handlerCount()).toBe(0);
  });

  test("a second signal during cleanup does not interrupt it", async () => {
    const close = deferred();
    const h = harness({ closeGate: close.promise });
    const running = run(settings(), h.dependencies);
    await h.backendStarting;
    await Promise.resolve();
    h.raise("SIGTERM");
    // Lands while close() is still in flight.
    await Promise.resolve();
    h.raise("SIGTERM");
    h.raise("SIGINT");
    close.resolve();
    await running;
    expect(h.closes).toBe(1);
    expect(h.stops).toHaveLength(1);
    expect(h.exits).toEqual([0]);
  });

  test("handlers stay attached until cleanup has finished", async () => {
    const close = deferred();
    const h = harness({ closeGate: close.promise });
    const running = run(settings(), h.dependencies);
    await h.backendStarting;
    await Promise.resolve();
    h.raise("SIGTERM");
    await Promise.resolve();
    // A signal arriving mid-cleanup must still find a handler, or the default
    // action would kill the parent before the child is stopped.
    expect(h.handlerCount()).toBe(2);
    close.resolve();
    await running;
    expect(h.handlerCount()).toBe(0);
  });

  test("SIGTERM closes active connections rather than waiting for them", async () => {
    const h = harness();
    const running = run(settings(), h.dependencies);
    await h.backendStarting;
    await Promise.resolve();
    h.raise("SIGTERM");
    await running;
    expect(h.stops).toEqual([true]);
    expect(h.closes).toBe(1);
    expect(h.exits).toEqual([0]);
  });

  test("repeated signals shut down once, not once per signal", async () => {
    const h = harness();
    const running = run(settings(), h.dependencies);
    await h.backendStarting;
    await Promise.resolve();
    h.raise("SIGTERM");
    h.raise("SIGINT");
    h.raise("SIGTERM");
    await running;
    expect(h.stops).toHaveLength(1);
    expect(h.closes).toBe(1);
    expect(h.exits).toEqual([0]);
    // Only the first signal arms a deadline.
    expect(h.pendingTimers()).toBe(0);
  });

  test("a backend whose close rejects is logged and still lets the process exit", async () => {
    const h = harness({ closeThrows: new Error("fm refused to die") });
    const running = run(settings(), h.dependencies);
    await h.backendStarting;
    await Promise.resolve();
    h.raise("SIGTERM");
    await running;
    expect(h.exits).toEqual([0]);
    expect(h.logs.join(" ")).toContain("shutdown_step_failed");
    expect(h.logs.join(" ")).toContain("fm refused to die");
    expect(h.handlerCount()).toBe(0);
  });

  test("the real timer is cancelled on release, so no deadline fires afterwards", async () => {
    // Deliberately uses the default setTimer/clearTimer rather than the fakes:
    // an unref'd timer is not a cancelled one, and only the real pair shows it.
    const h = harness({ shutdownTimeoutMs: 20 });
    const { setTimer, clearTimer, ...rest } = h.dependencies;
    const dependencies: RunDependencies = {
      ...rest,
      setTimer: defaultLifecycleDependencies.setTimer,
      clearTimer: defaultLifecycleDependencies.clearTimer,
    };
    const running = run(settings(), dependencies);
    await h.backendStarting;
    await Promise.resolve();
    h.raise("SIGTERM");
    await running;
    expect(h.exits).toEqual([0]);
    // Well past the 20ms deadline: a surviving timer would fire in here.
    await Bun.sleep(120);
    expect(h.exits).toEqual([0]);
    expect(h.kills).toBe(0);
    expect(h.logs.join(" ")).not.toContain("shutdown_deadline_expired");
  });

  test("the default clearTimer really cancels what the default setTimer armed", async () => {
    // Checked apart from the lifecycle: its released guard would hide a
    // clearTimer that silently does nothing.
    let fired = false;
    const handle = defaultLifecycleDependencies.setTimer(() => {
      fired = true;
    }, 10);
    defaultLifecycleDependencies.clearTimer(handle);
    await Bun.sleep(60);
    expect(fired).toBe(false);
  });

  test("an expired deadline exits once, and a later clean exit cannot override it", async () => {
    const close = deferred();
    const h = harness({ closeGate: close.promise, shutdownTimeoutMs: 10 });
    const running = run(settings(), h.dependencies);
    await h.backendStarting;
    await Promise.resolve();
    h.raise("SIGTERM");
    await Promise.resolve();
    h.fireDeadline();
    expect(h.exits).toEqual([INTERRUPTED_EXIT_CODE]);
    // Cleanup finishes afterwards; the normal path must not add a second exit.
    close.resolve();
    await running;
    expect(h.exits).toEqual([INTERRUPTED_EXIT_CODE]);
  });

  test("a deadline hook that throws does not stop the others or the exit", async () => {
    const h = harness({ closeGate: deferred().promise, shutdownTimeoutMs: 10 });
    const order: string[] = [];
    const dependencies: RunDependencies = {
      ...h.dependencies,
      createBackend: async (settingsArg, signal, onSpawn) => {
        onSpawn(() => {
          order.push("first");
          throw new Error("kill failed");
        });
        const created = await h.dependencies.createBackend(
          settingsArg,
          signal,
          () => {},
        );
        return { ...created, kill: () => order.push("second") };
      },
    };
    void run(settings(), dependencies);
    await h.backendStarting;
    await Promise.resolve();
    h.raise("SIGTERM");
    await Promise.resolve();
    h.fireDeadline();
    // Registered innermost-last, so both run despite the first throwing.
    expect(order).toEqual(["first", "second"]);
    expect(h.exits).toEqual([INTERRUPTED_EXIT_CODE]);
    expect(h.logs.join(" ")).toContain("shutdown_last_resort_failed");
  });

  test("a deadline expiring mid-startup can still kill the child", async () => {
    // createBackend never settles, as a wedged fm serve start would not.
    const h = harness({ startGate: deferred().promise, shutdownTimeoutMs: 10 });
    const killed: string[] = [];
    const dependencies: RunDependencies = {
      ...h.dependencies,
      createBackend: async (settingsArg, signal, onSpawn) => {
        // The child exists well before the backend is ready.
        onSpawn(() => killed.push("starting child"));
        return h.dependencies.createBackend(settingsArg, signal, () => {});
      },
    };
    void run(settings(), dependencies);
    await h.backendStarting;
    h.raise("SIGTERM");
    h.fireDeadline();
    expect(killed).toEqual(["starting child"]);
    expect(h.exits).toEqual([INTERRUPTED_EXIT_CODE]);
  });

  test("a clean shutdown leaves no timer pending", async () => {
    const h = harness();
    const running = run(settings(), h.dependencies);
    await h.backendStarting;
    await Promise.resolve();
    h.raise("SIGTERM");
    await running;
    // Why it matters: an un-cleared deadline would keep the process alive.
    expect(h.pendingTimers()).toBe(0);
  });
});

describe("scripts that own a backend directly", () => {
  function signalHarness() {
    const handlers = new Map<SignalName, Set<() => void>>();
    const logs: string[] = [];
    const timers: (() => void)[] = [];
    return {
      logs,
      raise(signal: SignalName) {
        for (const handler of [...(handlers.get(signal) ?? [])]) handler();
      },
      count: () => [...handlers.values()].reduce((sum, set) => sum + set.size, 0),
      pendingTimers: () => timers.length,
      options: {
        onSignal: (signal: SignalName, handler: () => void) => {
          const set = handlers.get(signal) ?? new Set();
          set.add(handler);
          handlers.set(signal, set);
        },
        offSignal: (signal: SignalName, handler: () => void) => {
          handlers.get(signal)?.delete(handler);
        },
        log: (message: string) => logs.push(message),
        setTimer: (callback: () => void) => {
          timers.push(callback);
          return timers.length;
        },
        clearTimer: () => {
          timers.length = 0;
        },
      },
    };
  }

  test("the smoke path closes its backend even when the body throws", async () => {
    const signals = signalHarness();
    await expect(
      withManagedBackend(
        settings(),
        async () => {
          throw new Error("inference failed");
        },
        signals.options,
      ),
    ).rejects.toThrow("inference failed");
    expect(signals.count()).toBe(0);
  });

  test("the smoke path registers signal handlers and releases them on success", async () => {
    const signals = signalHarness();
    let sawHandlers = 0;
    const result = await withManagedBackend(
      settings(),
      async () => {
        sawHandlers = signals.count();
        return "done";
      },
      signals.options,
    );
    expect(result).toBe("done");
    expect(sawHandlers).toBe(2);
    expect(signals.count()).toBe(0);
  });

  test("a signal interrupts inference instead of waiting for an upstream that never answers", async () => {
    const signals = signalHarness();
    const started = deferred();
    const failed = withManagedBackend(
      settings(),
      // Never resolves on its own, like a request to a hung upstream.
      async () => {
        started.resolve();
        return new Promise<string>(() => {});
      },
      signals.options,
    );
    await started.promise;
    signals.raise("SIGINT");
    await expect(failed).rejects.toBeInstanceOf(InterruptedError);
    expect(signals.count()).toBe(0);
  });

  test("the interrupted body is told to stop through the lifecycle signal", async () => {
    const signals = signalHarness();
    const started = deferred();
    let observed = false;
    const failed = withManagedBackend(
      settings(),
      async (_engine, signal) => {
        started.resolve();
        return new Promise<string>((_resolve, reject) => {
          signal.addEventListener("abort", () => {
            observed = true;
            reject(new Error("upstream request aborted"));
          });
        });
      },
      signals.options,
    );
    await started.promise;
    signals.raise("SIGTERM");
    await expect(failed).rejects.toThrow();
    // The body can pass this signal to fetch and cancel the real request.
    expect(observed).toBe(true);
  });

  test("a second signal during the smoke teardown does not strand it", async () => {
    const signals = signalHarness();
    const started = deferred();
    const failed = withManagedBackend(
      settings(),
      async () => {
        started.resolve();
        return new Promise<string>(() => {});
      },
      signals.options,
    );
    await started.promise;
    signals.raise("SIGINT");
    signals.raise("SIGINT");
    await expect(failed).rejects.toBeInstanceOf(InterruptedError);
    expect(signals.count()).toBe(0);
    expect(signals.pendingTimers()).toBe(0);
  });

  test("an interruption while the backend starts is normalized, not a generic error", async () => {
    const signals = signalHarness();
    // An apple start aborted by a signal rejects with its own wording; the
    // helper has to recognise it so the script exits 130 rather than 1.
    await expect(
      withManagedBackend(
        loadSettings({
          backend: "apple",
          managedUpstream: true,
          fmBinary: "/nonexistent/fm",
        }),
        async () => "never reached",
        {
          ...signals.options,
          onSignal: (signal, handler) => {
            signals.options.onSignal(signal, handler);
            // Fires while createEngine is still bringing the backend up.
            if (signal === "SIGTERM") queueMicrotask(() => handler());
          },
        },
      ),
    ).rejects.toBeInstanceOf(InterruptedError);
    expect(signals.count()).toBe(0);
  });

  test("a genuine startup failure keeps its own error rather than looking interrupted", async () => {
    const signals = signalHarness();
    const failed = withManagedBackend(
      loadSettings({
        backend: "apple",
        managedUpstream: true,
        fmBinary: "/nonexistent/fm",
      }),
      async () => "never reached",
      signals.options,
    );
    await expect(failed).rejects.not.toBeInstanceOf(InterruptedError);
    expect(signals.count()).toBe(0);
  });

  test("a successful smoke run arms no deadline and leaves no timer behind", async () => {
    const signals = signalHarness();
    await withManagedBackend(settings(), async () => "ok", signals.options);
    // Why it matters: a stray 10s timer used to delay the script's exit.
    expect(signals.pendingTimers()).toBe(0);
  });
});

describe("request body ceiling", () => {
  test("a text-only server keeps a fixed allowance", () => {
    expect(maxRequestBodySize(loadSettings())).toBe(16 * 1024 * 1024);
  });

  test("enabling images raises it by the base64 worst case the limits allow", () => {
    const settings = loadSettings({
      extensions: new Set(["images" as const]),
      maxImages: 4,
      maxImageBytes: 5_000_000,
    });
    // 4 images x the padded base64 length of 5_000_000 bytes, plus the text allowance.
    expect(maxRequestBodySize(settings)).toBe(
      4 * (4 * Math.ceil(5_000_000 / 3)) + 16 * 1024 * 1024,
    );
  });

  test("the per-image allowance is the padded base64 length, not a rounded ratio", () => {
    const allowance = (maxImageBytes: number) =>
      maxRequestBodySize(
        loadSettings({
          extensions: new Set(["images" as const]),
          maxImages: 1,
          maxImageBytes,
        }),
      ) -
      16 * 1024 * 1024;
    // 1, 2 and 3 bytes all encode to one 4-character group; 4 bytes needs two.
    expect([1, 2, 3, 4].map(allowance)).toEqual([4, 4, 4, 8]);
  });

  test("the configured ceiling reaches the server it starts", async () => {
    const settings = loadSettings({
      extensions: new Set(["images" as const]),
      maxImages: 1,
    });
    const test = harness();
    let seen = -1;
    const started = run(settings, {
      ...test.dependencies,
      serve: (options) => {
        seen = options.maxRequestBodySize;
        return test.dependencies.serve(options);
      },
    });
    await test.backendStarting;
    test.raise("SIGINT");
    await started;
    expect(seen).toBe(maxRequestBodySize(settings));
  });
});
