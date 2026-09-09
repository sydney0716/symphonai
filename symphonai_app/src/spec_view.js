import { followUpsFor, reportPathFor, specPaths } from "./roadmap.js";

function primarySpec(item) {
  return specPaths(item)[0] ?? null;
}

function fileFrom(reply, fallbackPath) {
  if (
    reply === null ||
    typeof reply !== "object" ||
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

  async function open(item, { specPaths: allSpecPaths = [] } = {}) {
    const paths = specPaths(item);
    const path = paths[0] ?? null;
    const result = {
      specs: [],
      report: null,
      followUps: [],
      error: null,
    };
    if (path === null) {
      result.error = "item has no spec binding";
      return result;
    }

    result.followUps = followUpsFor(path, allSpecPaths).map((followUpPath) => ({
      path: followUpPath,
      text: null,
    }));

    const errors = [];
    for (const specPath of paths) {
      let reply;
      try {
        reply = await client.file(specPath);
      } catch {
        errors.push(`spec unavailable: ${specPath}`);
        continue;
      }
      const spec = fileFrom(reply, specPath);
      if (spec === null) {
        errors.push(`spec unavailable: ${specPath}`);
        continue;
      }
      result.specs.push(spec);
    }

    // The first binding is the item's primary spec and owns its report.
    const reportPath = reportPathFor(path);
    if (reportPath !== null) {
      try {
        const report = fileFrom(await client.file(reportPath), reportPath);
        if (report === null) {
          errors.push(`report unavailable: ${reportPath}`);
        } else {
          result.report = report;
        }
      } catch (error) {
        if (error?.status !== 404) {
          errors.push(`report unavailable: ${reportPath}`);
        }
      }
    }

    if (errors.length > 0) {
      result.error = errors.join("; ");
    }
    return result;
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
