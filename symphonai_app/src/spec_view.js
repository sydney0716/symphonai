import { reportPathFor, specPaths } from "./roadmap.js";

class SpecViewError extends Error {
  constructor(message) {
    super(message);
    this.name = "SpecViewError";
  }
}

function primarySpec(item) {
  return specPaths(item)[0] ?? null;
}

function fileFrom(reply, fallbackPath) {
  if (
    reply === null ||
    typeof reply !== "object" ||
    reply.status !== 200 ||
    typeof reply.text !== "string"
  ) {
    return null;
  }
  return {
    path: typeof reply.path === "string" ? reply.path : fallbackPath,
    text: reply.text,
  };
}

export function createSpecView({ client }) {
  if (!client || typeof client.file !== "function") {
    throw new TypeError("spec view requires a file client");
  }

  async function open(item) {
    const path = primarySpec(item);
    if (path === null) {
      return {
        spec: null,
        error: new SpecViewError("item has no spec binding"),
      };
    }

    const specReply = await client.file(path);
    const spec = fileFrom(specReply, path);
    if (spec === null) {
      return {
        spec: null,
        error: new SpecViewError(`spec unavailable: ${path}`),
      };
    }

    const reportPath = reportPathFor(path);
    if (reportPath === null) {
      return { spec, report: null };
    }
    const reportReply = await client.file(reportPath);
    if (reportReply.status === 404) {
      return { spec, report: null };
    }
    const report = fileFrom(reportReply, reportPath);
    if (report === null) {
      throw new SpecViewError(`report unavailable: ${reportPath}`);
    }
    return { spec, report };
  }

  function askIntent(item, question) {
    const path = primarySpec(item);
    return `Regarding ${path ?? "this unbound roadmap item"}: ${question}`;
  }

  function reviewIntent(item) {
    const path = primarySpec(item);
    if (path === null) {
      return "Review this unbound roadmap item.";
    }
    const reportPath = reportPathFor(path);
    return reportPath === null
      ? `Review ${path}.`
      : `Review ${path} together with its report at ${reportPath}.`;
  }

  return { open, askIntent, reviewIntent };
}
