import test from "node:test";
import assert from "node:assert/strict";

import { layoutPhase } from "../src/roadmap_graph.js";

const spec = (id, after = undefined, fields = {}) => ({
  title: id,
  spec: `specs/39/${id}-task.md`,
  ...(after === undefined ? {} : { after }),
  ...fields,
});

test("lays out splits, follow-ups, and a merge in dependency rows", () => {
  const result = layoutPhase([
    spec("a"), spec("b", ["a"]), spec("c", ["a"]),
    spec("d", ["b"]), spec("e", ["c", "d"]),
  ], {
    followUps: { b: [{ id: "bF", title: "fix goal" }] },
  });
  assert.deepEqual(result.rows.map((row) => row.map((node) => node.id)), [["a"], ["b", "c"], ["d"], ["e"]]);
  assert.deepEqual(result.rows[1][0].followUps.map(({ id }) => id), ["bF"]);
  assert.deepEqual(result.edges, [
    { from: "a", to: "b" }, { from: "a", to: "c" },
    { from: "b", to: "d" }, { from: "c", to: "e" }, { from: "d", to: "e" },
  ]);
});

test("groups four parallel children and attaches their incoming edges to the group", () => {
  const result = layoutPhase([spec("a"), ...["d", "b", "c", "e"].map((id) => spec(id, ["a"]))]);
  assert.equal(result.rows[1].length, 1);
  assert.deepEqual(result.rows[1][0].group.map(({ id }) => id), ["b", "c", "d", "e"]);
  assert.deepEqual(result.edges, [{ from: "a", to: "group-1" }]);
});

test("tags known cross-phase links, and reports cycles and unknown links", () => {
  const cross = layoutPhase([spec("z", ["37eF"])], { titles: { "37eF": "follow-up" } });
  assert.deepEqual(cross.rows[0][0].tags, ["← 37eF"]);
  assert.deepEqual(cross.edges, []);
  assert.match(layoutPhase([spec("x", ["y"]), spec("y", ["x"])]).error, /cycle.*x|cycle.*y/);
  assert.match(layoutPhase([spec("x", ["missing"])]).error, /missing/);
});

test("tags a long edge blocked by a node in its source column", () => {
  const result = layoutPhase([spec("a"), spec("x", ["a"]), spec("d", ["a", "x"])]);
  assert.deepEqual(result.rows.map((row) => row.map((node) => node.id)), [["a"], ["x"], ["d"]]);
  assert.ok(result.rows[2][0].tags.includes("← a"));
  assert.ok(!result.edges.some((edge) => edge.from === "a" && edge.to === "d"));
});

test("collapses leading done rows and drops their outgoing edges", () => {
  const result = layoutPhase([
    spec("a", undefined, { done: true }),
    spec("b", ["a"], { done: true }),
    spec("c", ["b"]),
  ]);
  assert.equal(result.summary, "✓ a b");
  assert.equal(result.rows[0][0].id, "c");
  assert.equal(result.rows[0][0].row, 0);
  assert.deepEqual(result.edges, []);
});

test("keeps the full graph when every node is done", () => {
  const result = layoutPhase([
    spec("a", undefined, { done: true }),
    spec("b", ["a"], { done: true }),
    spec("c", ["b"], { done: true }),
  ]);
  assert.equal(result.summary, "");
  assert.deepEqual(result.rows.map((row) => row.map((node) => node.id)), [["a"], ["b"], ["c"]]);
});
