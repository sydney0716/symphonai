import test from "node:test";
import assert from "node:assert/strict";

import { splitDiff } from "../src/diff_split.js";

test("splitDiff pairs removals and additions and leaves unmatched cells blank", () => {
  const rows = splitDiff("@@ -2,2 +2,3 @@\n-old one\n-old two\n+new one\n+new two\n+new three");
  assert.deepEqual(rows, [
    { kind: "hunk", text: "@@ -2,2 +2,3 @@" },
    { kind: "change", old: "-old one", new: "+new one" },
    { kind: "change", old: "-old two", new: "+new two" },
    { kind: "change", old: "", new: "+new three" },
  ]);
});

test("splitDiff duplicates context exactly and preserves line text", () => {
  const rows = splitDiff("--- a/file\n+++ b/file\n@@ -1,2 +1,2 @@\n same context \n-old text  \n+new text  ");
  assert.deepEqual(rows, [
    { kind: "header", text: "--- a/file" },
    { kind: "header", text: "+++ b/file" },
    { kind: "hunk", text: "@@ -1,2 +1,2 @@" },
    { kind: "context", old: " same context ", new: " same context " },
    { kind: "change", old: "-old text  ", new: "+new text  " },
  ]);
});
