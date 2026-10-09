export function createAgentBoard() {
  const rows = [];
  const subagents = new Set();
  const calls = new Map();

  function clear() {
    rows.length = 0;
    subagents.clear();
    calls.clear();
  }

  function apply(frame) {
    if (frame?.kind !== "event" || !frame.payload) return;
    const { type, agent_id, agent_name, subagent_agent_id, subagent_name, tool_name, tool_call_id, target } = frame.payload;
    if (type === "RunStarted") {
      calls.delete(agent_id);
      const row = rows.find((entry) => entry.agentId === agent_id);
      if (row) {
        Object.assign(row, { name: agent_name || "agent", state: "running", tool: "", target: "" });
      } else {
        if (!subagents.has(agent_id)) {
          clear();
        }
        rows.unshift({ agentId: agent_id, parentAgentId: null, name: agent_name || "agent", state: "running", tool: "", target: "", last: null });
      }
      return;
    }
    if (type === "SubagentSpawned") {
      subagents.add(subagent_agent_id);
      rows.push({ agentId: subagent_agent_id, parentAgentId: agent_id ?? null, name: subagent_name, state: "running", tool: "", target: "", last: null });
      return;
    }
    const id = type === "SubagentStopped" ? subagent_agent_id : agent_id;
    const row = rows.find((entry) => entry.agentId === id);
    if (!row) return;
    if (type === "ToolCallStarted") {
      row.tool = tool_name;
      row.target = target ?? "";
      row.state = "running";
      if (!calls.has(id)) calls.set(id, new Map());
      calls.get(id).set(tool_call_id, { tool: row.tool, target: row.target });
    } else if (type === "ToolCallFinished" || type === "ToolCallFailed") {
      const pending = calls.get(id);
      const finished = pending?.get(tool_call_id);
      if (finished) row.last = finished;
      pending?.delete(tool_call_id);
      const current = pending ? [...pending.values()].at(-1) : null;
      row.tool = current?.tool ?? "";
      row.target = current?.target ?? "";
    } else if (type === "PermissionRequested") {
      row.state = "waiting";
      row.tool = tool_name;
      row.target = calls.get(id)?.get(tool_call_id)?.target ?? "";
    } else if (type === "PermissionDenied") {
      row.state = "running";
    } else if (type === "SubagentStopped" || type === "RunFinished" || type === "RunFailed") {
      row.state = "done";
      row.tool = "";
      row.target = "";
      calls.delete(id);
    }
  }

  return { rows, apply, clear };
}
