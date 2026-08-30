
from dataclasses import asdict, dataclass, field

from typing import Any, Dict, List, Optional

@dataclass
class EvalCase:
    id: str
    title: str
    category: str
    prompt: str
    tools: List[Dict[str, Any]]
    expected_tool: Optional[str]
    expected_args: Dict[str, Any] = field(default_factory=dict)
    must_not_call: bool = False
    rubric: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def _tool(name: str, description: str, properties: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required,
                           "additionalProperties": False}}}


BASE_CASE = EvalCase("exact", "精确调用与参数类型", "基础选择", "turn brightness to 0 in room 1",
             [_tool("set_lights", "set brightness",
                    {"room": {"type": "string"},
                     "brightness": {"type": "integer", "minimum": 0, "maximum": 100}},
                    ["room", "brightness"])],
             "set_lights", {"room": "1", "brightness": 0}, rubric="应调用 set_lights 并正确填写房间与亮度参数。")