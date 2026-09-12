import test from "node:test";
import assert from "node:assert/strict";

import { createClient } from "../src/client.js";
import { createInitFlow } from "../src/init_flow.js";


const SURVEY = {
  root: ".",
  languages: [[".py", 3], [".js", 1]],
  by_directory: [["src", [[".py", 3]]], ["web", [[".js", 1]]]],
  entry_points: ["pyproject.toml"],
  docs: ["README.md", "docs/design.md"],
  tests: ["tests"],
  tree_summary: ".\n  docs/\n    design.md\n  src/\n    main.py",
  truncated_directories: ["src"],
  stopped: true,
  file_count: 4,
};


function boundary({ roadmapExists = false } = {}) {
  const calls = { survey: 0, prompt: [], file: [] };
  return {
    calls,
    async survey() {
      calls.survey += 1;
      return { survey: SURVEY };
    },
    async prompt(text) {
      calls.prompt.push(text);
      return { accepted: true, run_id: "run-1" };
    },
    async file(path) {
      calls.file.push(path);
      if (!roadmapExists) {
        const error = new Error("host file request failed with status 404");
        error.status = 404;
        throw error;
      }
      return { path, text: "{}" };
    },
  };
}


test("init surveys the client and sends one conversational opening", async () => {
  const client = boundary();
  const flow = createInitFlow({ client });
  const decoy = { ...SURVEY, languages: [[".rb", 99]] };
  const started = await flow.start(decoy);

  assert.equal(client.calls.survey, 1);
  assert.equal(client.calls.prompt.length, 1);
  const [prompt] = client.calls.prompt;
  for (const expected of [
    ".py: 3 files",
    "Files surveyed: 4 (truncated)",
    "src — .py: at least 3",
    "web — .js: 1",
    "pyproject.toml",
    "docs/design.md",
    "Propose a roadmap",
    "ask me what the survey is missing",
  ]) {
    assert.ok(prompt.includes(expected), `prompt omitted ${expected}`);
  }
  assert.ok(!prompt.includes(".rb: 99 files"));
  assert.deepEqual(started.survey, SURVEY);
});


test("completion re-reads the roadmap instead of trusting conversation text", async () => {
  const missingClient = boundary();
  const missing = createInitFlow({ client: missingClient });
  assert.equal(
    await missing.isComplete("The roadmap was written to docs/roadmap.json."),
    false,
  );
  assert.deepEqual(missingClient.calls.file, ["docs/roadmap.json"]);

  const presentClient = boundary({ roadmapExists: true });
  assert.equal(await createInitFlow({ client: presentClient }).isComplete(), true);
  assert.deepEqual(presentClient.calls.file, ["docs/roadmap.json"]);
});


test("client survey uses one authenticated GET request", async () => {
  const requests = [];
  const client = createClient({
    port: 4312,
    token: "survey-token",
    fetch: async (url, options) => {
      requests.push({ url, options });
      return {
        status: 200,
        async json() {
          return { survey: SURVEY };
        },
      };
    },
  });

  assert.deepEqual(await client.survey(), { survey: SURVEY });
  assert.equal(requests.length, 1);
  assert.equal(requests[0].url, "http://127.0.0.1:4312/survey");
  assert.equal(requests[0].options.method, "GET");
  assert.equal(requests[0].options.headers.Authorization, "Bearer survey-token");
});
