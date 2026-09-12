function isObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function renderList(title, values) {
  const rows = Array.isArray(values) && values.length > 0
    ? values.map((value) => `- ${value}`)
    : ["- none found"];
  return `${title}:\n${rows.join("\n")}`;
}

function renderSurvey(survey) {
  if (!isObject(survey) || typeof survey.tree_summary !== "string") {
    throw new TypeError("repository survey is malformed");
  }
  const languages = Array.isArray(survey.languages)
    ? survey.languages.map((item) => (
      Array.isArray(item) && item.length === 2
        ? `${item[0]}: ${item[1]} files`
        : String(item)
    ))
    : [];
  const truncatedDirectories = new Set(
    Array.isArray(survey.truncated_directories)
      ? survey.truncated_directories.map(String)
      : [],
  );
  const byDirectory = Array.isArray(survey.by_directory)
    ? survey.by_directory.map((item) => {
      if (!Array.isArray(item) || item.length !== 2 || !Array.isArray(item[1])) {
        return String(item);
      }
      const lowerBound = truncatedDirectories.has(String(item[0]));
      const counts = item[1].map((count) => {
        if (!Array.isArray(count)) {
          return String(count);
        }
        return `${count[0]}: ${lowerBound ? "at least " : ""}${count[1]}`;
      });
      return `${item[0]} — ${counts.length > 0 ? counts.join(", ") : "no extensions"}`;
    })
    : [];
  return [
    `Repository: ${String(survey.root ?? ".")}`,
    `Files surveyed: ${String(survey.file_count ?? "unknown")}${survey.stopped ? " (truncated)" : ""}`,
    renderList("Languages", languages),
    renderList("Languages by top-level directory", byDirectory),
    renderList("Entry points", survey.entry_points),
    renderList("Documentation", survey.docs),
    renderList("Test directories", survey.tests),
    `Tree (two levels):\n${survey.tree_summary}`,
  ].join("\n\n");
}

function isNotFound(error) {
  return error?.status === 404 || /\b404\b/.test(String(error?.message ?? ""));
}

export function createInitFlow({ client }) {
  if (
    !client
    || typeof client.survey !== "function"
    || typeof client.prompt !== "function"
    || typeof client.file !== "function"
  ) {
    throw new TypeError("init flow requires survey, prompt, and file clients");
  }

  async function start() {
    const reply = await client.survey();
    const survey = reply?.survey;
    const surveyText = renderSurvey(survey);
    const openingPrompt = [
      "Here is a bounded survey of this repository:",
      surveyText,
      "Propose a roadmap for this repository and ask me what the survey is missing. Continue as a conversation; do not write docs/roadmap.json until we have agreed on the roadmap.",
    ].join("\n\n");
    await client.prompt(openingPrompt);
    return { survey, surveyText, openingPrompt };
  }

  async function isComplete() {
    try {
      await client.file("docs/roadmap.json");
      return true;
    } catch (error) {
      if (isNotFound(error)) {
        return false;
      }
      throw error;
    }
  }

  return { start, isComplete };
}
