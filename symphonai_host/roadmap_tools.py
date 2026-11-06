"""Leader tools for the app's project spec-loop roadmap."""

import json

from symphonai_api.models import ToolResult
from symphonai_api.tools.base import LocalTool
from symphonai_api.tools.metadata import ToolMetadata
from symphonai_host.roadmap import RoadmapStore


_DESCRIPTIONS = {
    "list_roadmap": "List this project's spec-loop goals in order with their loop states.",
    "add_roadmap_goal": "Add a spec-loop goal to the project's roadmap, only when the person asks.",
    "start_roadmap_goal": "Start a roadmap goal when asked. Its loop stops for the person to approve the plan.",
    "remove_roadmap_goal": "Remove a roadmap goal when asked, leaving its loop and files intact.",
}


class RoadmapTool(LocalTool):
    def __init__(self, host, name):
        self.host = host
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return _DESCRIPTIONS[self.name]

    @property
    def parameters(self):
        key = "goal" if self.name == "add_roadmap_goal" else "goal_id"
        properties = {} if self.name == "list_roadmap" else {key: {"type": "string", "minLength": 1}}
        return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}

    def validate(self, arguments):
        if any(not value.strip() for value in arguments.values()):
            return "goal or goal_id must not be blank"
        return None

    def metadata(self, arguments):
        listing = self.name == 'list_roadmap'
        return ToolMetadata(concurrency_safe=listing)

    def _execute(self, tool_call, policy, cancel=None):
        from symphonai_host.run import LoopChanged

        try:
            with self.host._lock:
                store = RoadmapStore(self.host._repo_root)
                if self.name == "start_roadmap_goal":
                    reply, loop = self.host.loops.start_roadmap_goal(tool_call.arguments["goal_id"])
                    reply["loop_id"] = loop.id
                else:
                    roadmap = store.load()
                    if self.name == "add_roadmap_goal":
                        store.add(roadmap, tool_call.arguments["goal"].strip())
                    elif self.name == "remove_roadmap_goal":
                        goal = store.find(roadmap, tool_call.arguments["goal_id"])
                        if goal is None:
                            raise KeyError(tool_call.arguments["goal_id"])
                        roadmap["goals"].remove(goal)
                    if self.name != "list_roadmap":
                        store.save(roadmap)
                        self.host._broker.publish(LoopChanged(agent_id="", run_id="", change="roadmap"))
                    reply = store.reply(roadmap)
            return ToolResult(tool_call.id, True, content=json.dumps(reply, ensure_ascii=False))
        except KeyError:
            return ToolResult(tool_call.id, False, error=f"Unknown roadmap goal: {tool_call.arguments.get('goal_id', '')}")
        except (OSError, ValueError, RuntimeError) as exc:
            return ToolResult(tool_call.id, False, error=str(exc))


def roadmap_tools(host):
    return {name: RoadmapTool(host, name) for name in _DESCRIPTIONS}
