import test from "node:test";
import assert from "node:assert/strict";

import {
  HostError,
  hostFromPage,
  hostFromShell,
  resolveHost,
} from "../src/host_handle.js";


test("shell and page adapters return only the host handle", () => {
  const shell = { port: 4312, token: "shell-token", ignored: true };
  const page = {
    __symphonai: { port: 4313, token: "page-token", ignored: true },
  };

  assert.deepEqual(hostFromShell(shell), { port: 4312, token: "shell-token" });
  assert.deepEqual(hostFromPage(page), { port: 4313, token: "page-token" });
});


test("resolveHost prefers the shell bridge and falls back to the page", () => {
  const both = {
    __symphonaiShell: { port: 5001, token: "from-shell" },
    __symphonai: { port: 5002, token: "from-page" },
  };
  assert.deepEqual(resolveHost(both), { port: 5001, token: "from-shell" });
  assert.deepEqual(
    resolveHost({ __symphonai: both.__symphonai }),
    { port: 5002, token: "from-page" },
  );
});


test("missing and malformed handles raise HostError without exposing tokens", () => {
  const secret = "must-not-leak";
  const cases = [
    {},
    { __symphonaiShell: { port: 0, token: secret } },
    { __symphonai: { port: 70000, token: secret } },
    { __symphonaiShell: { port: 4312, token: "" } },
  ];
  for (const env of cases) {
    assert.throws(
      () => resolveHost(env),
      (error) => error instanceof HostError && !error.message.includes(secret),
    );
  }
});
