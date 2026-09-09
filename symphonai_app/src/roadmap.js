export class RoadmapError extends Error {
  constructor(message) {
    super(message);
    this.name = "RoadmapError";
  }
}

function isObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function phaseLabel(phase, index) {
  return typeof phase?.id === "string" ? `phase ${phase.id}` : `phase ${index}`;
}

export function itemsOf(phase) {
  const label = typeof phase?.id === "string" ? `phase ${phase.id}` : "phase";
  if (!isObject(phase) || !Array.isArray(phase.items)) {
    throw new RoadmapError(`${label} items must be an array`);
  }
  return phase.items.map((item, index) => {
    if (typeof item === "string") {
      return { title: item, desc: "", spec: [] };
    }
    if (!isObject(item)) {
      throw new RoadmapError(`${label} item ${index} must be a string or object`);
    }
    if (typeof item.title !== "string") {
      throw new RoadmapError(`${label} item ${index} title must be a string`);
    }
    if (item.desc !== undefined && typeof item.desc !== "string") {
      throw new RoadmapError(`${label} item ${index} desc must be a string`);
    }
    const specs = item.spec === undefined || item.spec === null
      ? []
      : typeof item.spec === "string"
        ? [item.spec]
        : item.spec;
    if (
      !Array.isArray(specs) ||
      specs.some((path) => typeof path !== "string")
    ) {
      throw new RoadmapError(
        `${label} item ${index} spec must be a string or string array`,
      );
    }
    return {
      ...item,
      title: item.title,
      desc: item.desc ?? "",
      spec: [...specs],
    };
  });
}

export function parseRoadmap(json) {
  let value;
  try {
    value = JSON.parse(json);
  } catch {
    throw new RoadmapError("roadmap is not valid JSON");
  }
  if (!isObject(value) || typeof value.goal !== "string") {
    throw new RoadmapError("roadmap goal must be a string");
  }
  if (!Array.isArray(value.phases)) {
    throw new RoadmapError("roadmap phases must be an array");
  }
  const phases = value.phases.map((phase, index) => {
    const label = phaseLabel(phase, index);
    if (!isObject(phase)) {
      throw new RoadmapError(`${label} must be an object`);
    }
    for (const field of ["id", "name", "status"]) {
      if (typeof phase[field] !== "string") {
        throw new RoadmapError(`${label} ${field} must be a string`);
      }
    }
    return { ...phase, items: itemsOf(phase) };
  });
  return { goal: value.goal, phases };
}

export function specPaths(item) {
  if (!isObject(item)) {
    return [];
  }
  if (typeof item.spec === "string") {
    return [item.spec];
  }
  return Array.isArray(item.spec) &&
    item.spec.every((path) => typeof path === "string")
    ? [...item.spec]
    : [];
}

export function followUpsFor(path, allSpecPaths) {
  if (typeof path !== "string" || !Array.isArray(allSpecPaths)) {
    return [];
  }
  const parent = /^specs\/([0-9]+)\/([0-9]+[A-Za-z])-[^/]+\.md$/.exec(path);
  if (parent === null) {
    return [];
  }
  const prefix = `${parent[2]}F`;
  const followUp = new RegExp(
    `^specs/${parent[1]}/(${prefix}(?:[0-9]+)?)-[^/]+\\.md$`,
  );
  return allSpecPaths
    .map((candidate) => ({ candidate, match: followUp.exec(candidate) }))
    .filter(({ match }) => match !== null)
    .sort((left, right) => {
      const leftOrder = left.match[1] === prefix
        ? 1
        : Number(left.match[1].slice(prefix.length));
      const rightOrder = right.match[1] === prefix
        ? 1
        : Number(right.match[1].slice(prefix.length));
      return leftOrder - rightOrder || left.candidate.localeCompare(right.candidate);
    })
    .map(({ candidate }) => candidate);
}

export function reportPathFor(path) {
  if (typeof path !== "string") {
    return null;
  }
  const match = /^specs\/([^/]+)\/([^/]+)\.md$/.exec(path);
  if (match === null) {
    return null;
  }
  return `specs/report/${match[1]}/${match[2]}-report.md`;
}

export function phaseProgress(phase) {
  const items = itemsOf(phase);
  return {
    done: phase.status === "done"
      ? items.length
      : items.filter((item) => item.done === true).length,
    total: items.length,
  };
}

export function renderRoadmap(state) {
  if (!isObject(state) || !Array.isArray(state.phases)) {
    throw new RoadmapError("render state must contain phases");
  }
  return {
    goal: state.goal,
    phases: state.phases.map((phase) => ({
      id: phase.id,
      name: phase.name,
      status: phase.status,
      progress: phaseProgress(phase),
      items: itemsOf(phase),
    })),
  };
}
