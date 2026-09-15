import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import { DEFAULT_ROUTE, formatRoute, PAGES, parseRoute } from "../src/route.js";

test("routes parse, reject junk, and round-trip", () => {
  assert.deepEqual(parseRoute("#/settings"), { page: "settings", section: "" });
  assert.deepEqual(parseRoute("#/settings/mcp"), { page: "settings", section: "mcp" });
  assert.deepEqual(parseRoute("#/roadmap"), { page: "roadmap", section: "" });
  assert.deepEqual(parseRoute("#/chat"), { page: "chat", section: "" });
  for (const fragment of ["", "#", "#/nope", "#//", "junk"]) {
    assert.deepEqual(parseRoute(fragment), DEFAULT_ROUTE);
  }

  const routes = [
    ...PAGES.map((page) => ({ page, section: "" })),
    { page: "settings", section: "mcp" },
  ];
  for (const route of routes) {
    assert.deepEqual(parseRoute(formatRoute(route)), route);
  }
  assert.deepEqual(PAGES, ["chat", "settings", "roadmap"]);
  assert.ok(Object.isFrozen(PAGES));
});

test("the route module is pure", async () => {
  const source = await readFile(new URL("../src/route.js", import.meta.url), "utf8");

  for (const name of ["document", "localStorage", "globalThis", "window"]) {
    assert.ok(!source.includes(name));
  }
});
