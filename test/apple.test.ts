import { describe, expect, test } from "bun:test";

import { AppleServer, socketPathFor, type Spawned } from "../src/apple";
import { loadSettings } from "../src/config";
import { BackendUnavailableError } from "../src/engine";

const appleSettings = () =>
  loadSettings({ backend: "apple", managedUpstream: true, upstreamModel: "system" });

interface FakeChild extends Spawned {
  signals: (number | NodeJS.Signals)[];
}

function fakeChild(options: { exitAfterMs?: number; exitCode?: number; ignoreSigterm?: boolean } = {}): FakeChild {
  const signals: (number | NodeJS.Signals)[] = [];
  let settle!: (code: number) => void;
  const exited = new Promise<number>((resolve) => {
    settle = resolve;
  });
  const child: FakeChild = {
    signals,
    exited,
    exitCode: null,
    kill(signal = "SIGTERM") {
      signals.push(signal);
      const stays = options.ignoreSigterm && signal === "SIGTERM";
      if (!stays) settle(0);
    },
  };
  if (options.exitAfterMs !== undefined) {
    setTimeout(() => {
      Object.assign(child, { exitCode: options.exitCode ?? 1 });
      settle(options.exitCode ?? 1);
    }, options.exitAfterMs);
  }
  return child;
}

describe("Apple foundation-models server", () => {
  test("rejects a socket path that exceeds the macOS 104-byte limit", () => {
    expect(() => socketPathFor(`/tmp/${"d".repeat(120)}`)).toThrow(BackendUnavailableError);
    expect(() => socketPathFor(`/tmp/${"d".repeat(120)}`)).toThrow("macOS allows fewer than 104");
    expect(Buffer.byteLength(socketPathFor())).toBeLessThan(104);
  });

  test("spawns fm serve on its socket and reports ready once /health answers", async () => {
    const commands: string[][] = [];
    const child = fakeChild();
    let probes = 0;
    const server = new AppleServer(appleSettings(), {
      socketPath: "/tmp/localjev-test.sock",
      spawn: (command) => {
        commands.push(command);
        return child;
      },
      fetchImpl: async () => {
        probes += 1;
        // The first probe fails the way a not-yet-listening socket does.
        if (probes === 1) throw new Error("ECONNREFUSED");
        return Response.json({ status: "fm serve is running" });
      },
      pollIntervalMs: 1,
    });

    await server.start();
    expect(commands).toEqual([["/usr/bin/fm", "serve", "--socket", "/tmp/localjev-test.sock"]]);
    expect(probes).toBe(2);
  });

  test("routes requests through the Unix socket rather than TCP", async () => {
    const seen: RequestInit[] = [];
    const server = new AppleServer(appleSettings(), {
      socketPath: "/tmp/localjev-unix.sock",
      spawn: () => fakeChild(),
      fetchImpl: async (_input, init) => {
        seen.push(init ?? {});
        return Response.json({ ok: true });
      },
      pollIntervalMs: 1,
    });
    await server.start();
    await server.fetch("http://localhost/v1/chat/completions", { method: "POST" });

    const last = seen.at(-1) as RequestInit & { unix?: string };
    expect(last.unix).toBe("/tmp/localjev-unix.sock");
    expect(last.method).toBe("POST");
  });

  test("fails fast when the child exits before serving", async () => {
    const server = new AppleServer(appleSettings(), {
      socketPath: "/tmp/localjev-dead.sock",
      spawn: () => fakeChild({ exitAfterMs: 5, exitCode: 2 }),
      fetchImpl: async () => {
        throw new Error("ECONNREFUSED");
      },
      pollIntervalMs: 1,
      startTimeoutMs: 5_000,
    });
    await expect(server.start()).rejects.toThrow("exited with code 2");
  });

  test("gives up when /health never answers within the start timeout", async () => {
    const server = new AppleServer(appleSettings(), {
      socketPath: "/tmp/localjev-slow.sock",
      spawn: () => fakeChild(),
      fetchImpl: async () => {
        throw new Error("ECONNREFUSED");
      },
      pollIntervalMs: 1,
      startTimeoutMs: 30,
    });
    await expect(server.start()).rejects.toThrow("did not become ready within 30ms");
  });

  test("close terminates the child and escalates to SIGKILL when it ignores SIGTERM", async () => {
    const stubborn = fakeChild({ ignoreSigterm: true });
    const server = new AppleServer(appleSettings(), {
      socketPath: "/tmp/localjev-stubborn.sock",
      spawn: () => stubborn,
      fetchImpl: async () => Response.json({ status: "ok" }),
      pollIntervalMs: 1,
      killGraceMs: 10,
    });
    await server.start();
    await server.close();
    expect(stubborn.signals).toEqual(["SIGTERM", "SIGKILL"]);

    // A second close is a no-op, so shutdown paths can call it more than once.
    await server.close();
    expect(stubborn.signals).toEqual(["SIGTERM", "SIGKILL"]);
  });

  test("stderr is consumed line by line so a chatty fm serve cannot block on a full pipe", async () => {
    const child = fakeChild();
    const lines = ["first warning", "second warning"];
    Object.assign(child, {
      stderr: new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(new TextEncoder().encode("first warning\nsecond "));
          controller.enqueue(new TextEncoder().encode("warning\n"));
          controller.close();
        },
      }),
    });
    const logged: string[] = [];
    const original = console.log;
    console.log = (line: string) => {
      const entry = JSON.parse(line);
      if (entry.event === "apple_server_stderr") logged.push(entry.line);
    };
    try {
      const server = new AppleServer(appleSettings(), {
        socketPath: "/tmp/localjev-stderr.sock",
        spawn: () => child,
        fetchImpl: async () => Response.json({ status: "ok" }),
        pollIntervalMs: 1,
      });
      await server.start();
      await server.close();
    } finally {
      console.log = original;
    }
    expect(logged).toEqual(lines);
  });

  test("close stops a cooperative child with SIGTERM alone", async () => {
    const child = fakeChild();
    const server = new AppleServer(appleSettings(), {
      socketPath: "/tmp/localjev-clean.sock",
      spawn: () => child,
      fetchImpl: async () => Response.json({ status: "ok" }),
      pollIntervalMs: 1,
      killGraceMs: 50,
    });
    await server.start();
    await server.close();
    expect(child.signals).toEqual(["SIGTERM"]);
  });

  test("a health probe that hangs cannot outlive the start deadline", async () => {
    const child = fakeChild();
    const server = new AppleServer(appleSettings(), {
      socketPath: "/tmp/localjev-hang.sock",
      spawn: () => child,
      // A socket that accepts the connection and then never answers.
      fetchImpl: (_input, init) =>
        new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => reject(new Error("aborted")));
        }),
      pollIntervalMs: 1,
      startTimeoutMs: 40,
    });
    await expect(server.start()).rejects.toThrow("did not become ready within 40ms");
    // The child that never became ready must not be left behind.
    expect(child.signals).toEqual(["SIGTERM"]);
  });

  test("a start that fails still stops the child it spawned", async () => {
    const child = fakeChild();
    const server = new AppleServer(appleSettings(), {
      socketPath: "/tmp/localjev-failstart.sock",
      spawn: () => child,
      fetchImpl: async () => {
        throw new Error("ECONNREFUSED");
      },
      pollIntervalMs: 1,
      startTimeoutMs: 20,
    });
    await expect(server.start()).rejects.toThrow(BackendUnavailableError);
    expect(child.signals).toEqual(["SIGTERM"]);
    expect(server.running()).toBe(false);
  });

  test("closing during startup stops the child instead of leaving it running", async () => {
    const child = fakeChild();
    let probes = 0;
    const server = new AppleServer(appleSettings(), {
      socketPath: "/tmp/localjev-racing.sock",
      spawn: () => child,
      fetchImpl: async () => {
        probes += 1;
        // Close lands between the first failed probe and the next attempt.
        if (probes === 1) void server.close();
        throw new Error("ECONNREFUSED");
      },
      pollIntervalMs: 1,
      startTimeoutMs: 1_000,
    });
    await expect(server.start()).rejects.toThrow("stopped while starting");
    expect(child.signals).toContain("SIGTERM");
    expect(server.running()).toBe(false);
  });

  test("a start after close is an error rather than a silently dead server", async () => {
    const server = new AppleServer(appleSettings(), {
      socketPath: "/tmp/localjev-reuse.sock",
      spawn: () => fakeChild(),
      fetchImpl: async () => Response.json({ status: "ok" }),
      pollIntervalMs: 1,
    });
    await server.start();
    await server.close();
    await expect(server.start()).rejects.toThrow("already stopped");
  });

  test("concurrent closes send one SIGTERM and share a single shutdown", async () => {
    const child = fakeChild();
    const server = new AppleServer(appleSettings(), {
      socketPath: "/tmp/localjev-concurrent.sock",
      spawn: () => child,
      fetchImpl: async () => Response.json({ status: "ok" }),
      pollIntervalMs: 1,
    });
    await server.start();
    await Promise.all([server.close(), server.close(), server.close()]);
    expect(child.signals).toEqual(["SIGTERM"]);
  });

  test("a health response arriving after close does not revive the server", async () => {
    const child = fakeChild();
    const held: ((response: Response) => void)[] = [];
    const server = new AppleServer(appleSettings(), {
      socketPath: "/tmp/localjev-late-health.sock",
      spawn: () => child,
      // The probe is held open until the test closes the server.
      fetchImpl: () => new Promise<Response>((resolve) => held.push(resolve)),
      pollIntervalMs: 1,
      startTimeoutMs: 5_000,
    });
    const starting = server.start();
    await Bun.sleep(5);
    const closed = server.close();
    for (const resolve of held) resolve(Response.json({ status: "fm serve is running" }));
    await expect(starting).rejects.toThrow("stopped while starting");
    await closed;
    expect(server.running()).toBe(false);
    // The late 200 must not have promoted the server back to ready.
    await expect(server.start()).rejects.toThrow("already stopped");
  });

  test("a start racing a close sends exactly one SIGTERM", async () => {
    const child = fakeChild();
    let probes = 0;
    const server = new AppleServer(appleSettings(), {
      socketPath: "/tmp/localjev-one-term.sock",
      spawn: () => child,
      fetchImpl: async () => {
        probes += 1;
        if (probes === 1) void server.close();
        throw new Error("ECONNREFUSED");
      },
      pollIntervalMs: 1,
      startTimeoutMs: 1_000,
    });
    await expect(server.start()).rejects.toThrow("stopped while starting");
    // Both the failed start and the concurrent close must share one shutdown.
    expect(child.signals).toEqual(["SIGTERM"]);
  });

  test("a failed start is terminal: the next start reports it rather than succeeding", async () => {
    const server = new AppleServer(appleSettings(), {
      socketPath: "/tmp/localjev-terminal.sock",
      spawn: () => fakeChild(),
      fetchImpl: async () => {
        throw new Error("ECONNREFUSED");
      },
      pollIntervalMs: 1,
      startTimeoutMs: 15,
    });
    await expect(server.start()).rejects.toThrow("did not become ready");
    await expect(server.start()).rejects.toThrow("already stopped");
    expect(server.running()).toBe(false);
  });

  test("concurrent starts share one child rather than racing for the socket", async () => {
    let spawns = 0;
    const server = new AppleServer(appleSettings(), {
      socketPath: "/tmp/localjev-shared-start.sock",
      spawn: () => {
        spawns += 1;
        return fakeChild();
      },
      fetchImpl: async () => Response.json({ status: "ok" }),
      pollIntervalMs: 1,
    });
    await Promise.all([server.start(), server.start(), server.start()]);
    expect(spawns).toBe(1);
    expect(server.running()).toBe(true);
    await server.close();
  });

  test("an abort signal stops a start in progress without waiting out its timeout", async () => {
    const child = fakeChild();
    const abort = new AbortController();
    const server = new AppleServer(appleSettings(), {
      socketPath: "/tmp/localjev-signal.sock",
      spawn: () => child,
      fetchImpl: async () => {
        // Fires once the child is spawned and the first probe is under way.
        abort.abort();
        throw new Error("ECONNREFUSED");
      },
      pollIntervalMs: 1,
      // Long enough that only the abort can end this start promptly.
      startTimeoutMs: 60_000,
      signal: abort.signal,
    });
    const started = Date.now();
    await expect(server.start()).rejects.toThrow("stopped while starting");
    expect(Date.now() - started).toBeLessThan(5_000);
    expect(child.signals).toEqual(["SIGTERM"]);
  });

  test("a health probe already in flight is cancelled by the startup signal", async () => {
    const child = fakeChild();
    const abort = new AbortController();
    let probing!: () => void;
    const probeStarted = new Promise<void>((resolve) => { probing = resolve; });
    let probeAborted = false;
    const server = new AppleServer(appleSettings(), {
      socketPath: "/tmp/localjev-probe-abort.sock",
      spawn: () => child,
      // Accepts the connection and never answers, as a wedged fm serve would.
      fetchImpl: (_input, init) =>
        new Promise<Response>((_resolve, reject) => {
          probing();
          init?.signal?.addEventListener("abort", () => {
            probeAborted = true;
            reject(new Error("aborted"));
          });
        }),
      pollIntervalMs: 1,
      // Neither the start deadline nor the probe's own 5s cap may be what ends
      // this: only the startup signal composed into the probe can.
      startTimeoutMs: 600_000,
      signal: abort.signal,
    });
    const starting = server.start();
    await probeStarted;
    abort.abort();
    await expect(starting).rejects.toThrow("stopped while starting");
    expect(probeAborted).toBe(true);
    expect(child.signals).toEqual(["SIGTERM"]);
  });

  test("kill stops the child at once for a shutdown that ran out of time", async () => {
    const child = fakeChild({ ignoreSigterm: true });
    const server = new AppleServer(appleSettings(), {
      socketPath: "/tmp/localjev-kill.sock",
      spawn: () => child,
      fetchImpl: async () => Response.json({ status: "ok" }),
      pollIntervalMs: 1,
    });
    await server.start();
    server.kill();
    // No SIGTERM grace period: the process is leaving either way.
    expect(child.signals).toEqual(["SIGKILL"]);
  });

  test("kill reaches a child that close is still waiting out", async () => {
    // The case the deadline exists for: SIGTERM ignored, grace period running,
    // and the shutdown budget runs out before the child gives up.
    const child = fakeChild({ ignoreSigterm: true });
    const server = new AppleServer(appleSettings(), {
      socketPath: "/tmp/localjev-kill-during-close.sock",
      spawn: () => child,
      fetchImpl: async () => Response.json({ status: "ok" }),
      pollIntervalMs: 1,
      killGraceMs: 10_000,
    });
    await server.start();
    const closing = server.close();
    await Bun.sleep(5);
    expect(child.signals).toEqual(["SIGTERM"]);
    server.kill();
    expect(child.signals).toEqual(["SIGTERM", "SIGKILL"]);
    await closing;
  });

  test("kill reaches a child that is still starting up", async () => {
    const child = fakeChild();
    const killers: (() => void)[] = [];
    const server = new AppleServer(appleSettings(), {
      socketPath: "/tmp/localjev-kill-starting.sock",
      spawn: () => child,
      // Never becomes ready, so start() is still running when kill arrives.
      fetchImpl: async () => {
        throw new Error("ECONNREFUSED");
      },
      pollIntervalMs: 1,
      startTimeoutMs: 60_000,
      onSpawn: (spawned) => killers.push(() => spawned.kill()),
    });
    const starting = server.start();
    await Bun.sleep(5);
    // onSpawn fired before the server was ready, which is the point.
    expect(killers).toHaveLength(1);
    for (const kill of killers) kill();
    expect(child.signals).toContain("SIGKILL");
    await server.close();
    await expect(starting).rejects.toThrow();
  });

  test("a spawn hook that throws still stops the child it was told about", async () => {
    const child = fakeChild();
    const server = new AppleServer(appleSettings(), {
      socketPath: "/tmp/localjev-onspawn-throws.sock",
      spawn: () => child,
      fetchImpl: async () => Response.json({ status: "ok" }),
      onSpawn: () => {
        throw new Error("hook failed");
      },
    });
    await expect(server.start()).rejects.toThrow("hook failed");
    expect(child.signals).toContain("SIGTERM");
    expect(server.running()).toBe(false);
  });

  test("a child that survives SIGKILL does not block the shutdown forever", async () => {
    // Ignores every signal, like a process stuck in an uninterruptible wait.
    const child: FakeChild = {
      signals: [],
      exited: new Promise<number>(() => {}),
      exitCode: null,
      kill(signal = "SIGTERM") {
        this.signals.push(signal);
      },
    };
    const server = new AppleServer(appleSettings(), {
      socketPath: "/tmp/localjev-unkillable.sock",
      spawn: () => child,
      fetchImpl: async () => Response.json({ status: "ok" }),
      pollIntervalMs: 1,
      killGraceMs: 5,
    });
    await server.start();
    await server.close();
    expect(child.signals).toEqual(["SIGTERM", "SIGKILL"]);
  });

  test("a signal that already fired means no child is ever spawned", async () => {
    let spawns = 0;
    const abort = new AbortController();
    abort.abort();
    const server = new AppleServer(appleSettings(), {
      socketPath: "/tmp/localjev-preaborted.sock",
      spawn: () => {
        spawns += 1;
        return fakeChild();
      },
      fetchImpl: async () => Response.json({ status: "ok" }),
      pollIntervalMs: 1,
      signal: abort.signal,
    });
    await expect(server.start()).rejects.toThrow("stopped while starting");
    expect(spawns).toBe(0);
  });

  test("a binary that cannot be spawned fails without leaving a startable server", async () => {
    const server = new AppleServer(appleSettings(), {
      socketPath: "/tmp/localjev-nobinary.sock",
      spawn: () => {
        throw new Error("ENOENT: no such file or directory");
      },
      fetchImpl: async () => Response.json({ status: "ok" }),
      pollIntervalMs: 1,
    });
    await expect(server.start()).rejects.toThrow("could not be started");
    await expect(server.start()).rejects.toThrow("already stopped");
    // close() on a server that never spawned must still resolve.
    await server.close();
  });

  test("an unexpected exit after ready is logged and makes the server unavailable", async () => {
    const child = fakeChild();
    const logged: Record<string, unknown>[] = [];
    const original = console.log;
    console.log = (line: string) => {
      logged.push(JSON.parse(line));
    };
    try {
      const server = new AppleServer(appleSettings(), {
        socketPath: "/tmp/localjev-crash.sock",
        spawn: () => child,
        fetchImpl: async () => Response.json({ status: "ok" }),
        pollIntervalMs: 1,
      });
      await server.start();
      expect(server.running()).toBe(true);
      // fm serve dies on its own; kill() settles the fake child's exit promise.
      child.kill("SIGKILL");
      await child.exited;
      await Bun.sleep(1);
      expect(server.running()).toBe(false);
      await server.close();
    } finally {
      console.log = original;
    }
    const exit = logged.find((entry) => entry.event === "apple_server_exited");
    expect(exit).toMatchObject({ component: "apple", exitCode: 0 });
  });
});
