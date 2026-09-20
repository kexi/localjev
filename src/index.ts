import { loadSettings } from "./config";
import { run } from "./main";

export { AppleServer, createEngine } from "./apple";
export { loadSettings } from "./config";
export { Engine } from "./engine";
export { run, withManagedBackend } from "./main";
export { LocalJevApp } from "./server";
export * from "./types";

if (import.meta.main) {
  await run(loadSettings());
}
