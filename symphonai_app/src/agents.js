export function createAgentBoard() {
  const rows = [];

  function apply(frame) {
    if (frame?.kind !== "event" || !frame.payload) return;
    const { type, agent_id, agent_name, subagent_agent_id, subagent_name, tool_name } = frame.payload;
    if (type === "RunStarted") {
      const row = rows.find((entry) => entry.agentId === agent_id);
      if (row) {
        Object.assign(row, { name: agent_name || "agent", state: "running", tool: "" });
      } else {
        rows.unshift({ agentId: agent_id, name: agent_name || "agent", state: "running", tool: "" });
      }
      return;
    }
    if (type === "SubagentSpawned") {
      rows.push({ agentId: subagent_agent_id, name: subagent_name, state: "running", tool: "" });
      return;
    }
    const id = type === "SubagentStopped" ? subagent_agent_id : agent_id;
    const row = rows.find((entry) => entry.agentId === id);
    if (!row) return;
    if (type === "ToolCallStarted") {
      row.tool = tool_name;
      row.state = "running";
    } else if (type === "ToolCallFinished" || type === "ToolCallFailed") {
      row.tool = "";
    } else if (type === "PermissionRequested") {
      row.state = "waiting";
      row.tool = tool_name;
    } else if (type === "PermissionDenied") {
      row.state = "running";
    } else if (type === "SubagentStopped" || type === "RunFinished" || type === "RunFailed") {
      row.state = "done";
      row.tool = "";
    }
  }

  return { rows, apply };
}
