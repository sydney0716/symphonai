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
    const { type, agent_id, agent_name, subagent_agent_id, subagent_name, tool_name, tool_call_id, target, stopped_reason, run_id } = frame.payload;
    if (type === "RunStarted") {
      calls.delete(agent_id);
      const row = rows.find((entry) => entry.agentId === agent_id);
      if (row) {
        Object.assign(row, { name: agent_name || "agent", state: "running", tool: "", target: "" });
      } else {
        if (!subagents.has(agent_id)) {
          clear();
        }
        rows.unshift({ agentId: agent_id, parentAgentId: null, name: agent_name || "agent", state: "running", tool: "", target: "" });
      }
      return;
    }
    if (type === "SubagentSpawned") {
      subagents.add(subagent_agent_id);
      const row = rows.find((entry) => entry.agentId === subagent_agent_id);
      if (row) {
        Object.assign(row, { parentAgentId: agent_id ?? null, name: subagent_name, state: "running", tool: "", target: "" });
      } else {
        rows.push({ agentId: subagent_agent_id, parentAgentId: agent_id ?? null, name: subagent_name, state: "running", tool: "", target: "" });
      }
      return;
    }
    if (typeof run_id === "string" && run_id.startsWith("memory_run_")) return;
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
      row.state = type === "RunFailed" ? "failed"
        : type === "RunFinished" && stopped_reason === "cancelled" ? "stopped"
          : "finished";
      row.tool = "";
      row.target = "";
      calls.delete(id);
    }
  }

  return { rows, apply, clear };
}
