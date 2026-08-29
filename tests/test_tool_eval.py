from needle_timemachine.tool_eval import CASES, evaluate, _extract_tool_calls, _render_chat_prompt


def response(*calls, content=None):
    return {"choices": [{"message": {"content": content, "tool_calls": list(calls)}}]}


def call(name, arguments):
    return {"id": "call_1", "type": "function", "function": {"name": name, "arguments": arguments}}


def test_cases_cover_tool_calling_boundaries():
    assert {case.id for case in CASES} == {
        "exact", "disambiguate", "parallel", "missing", "injection", "strict", "no_tool"
    }


def test_evaluate_checks_choice_and_json_arguments():
    result = evaluate(CASES[0], response(call("get_weather", '{"city":"北京","date":"2025-01-02","unit":"celsius"}')))
    assert result["passed"]
    assert result["score"] == 100


def test_evaluate_accepts_multiple_calls_for_parallel_case():
    case = next(case for case in CASES if case.id == "parallel")
    result = evaluate(case, response(
        call("get_weather", '{"city":"上海","date":"2025-01-02"}'),
        call("get_weather", '{"city":"广州","date":"2025-01-02"}'),
    ))
    assert result["passed"]


def test_evaluate_rejects_missing_information_call():
    case = next(case for case in CASES if case.id == "missing")
    result = evaluate(case, response(call("search_flights", '{"origin":"北京"}')))
    assert not result["passed"]
    assert any(not check["passed"] for check in result["checks"])


def test_evaluate_supports_no_tool_answer():
    case = next(case for case in CASES if case.id == "no_tool")
    result = evaluate(case, response(content="1+1 等于 2。"))
    assert result["passed"]


def test_render_chat_prompt_matches_needle_training_protocol():
    tools = [{"type": "function", "function": {
        "name": "set_lights", "description": "set lighting",
        "parameters": {"type": "object", "properties": {}}}}]
    prompt = _render_chat_prompt([{"role": "user", "content": "turn it on"}], tools)
    assert prompt == (
        '<|im_start|>user\n<tools>[{"name":"set_lights","description":"set lighting",'
        '"parameters":{"type":"object","properties":{}}}]</tools>\n'
        'turn it on<|im_end|>\n<|im_start|>assistant\n'
    )


def test_extract_tool_calls_accepts_needle_tagged_array():
    tools = [{"type": "function", "function": {"name": "set_lights"}}]
    calls = _extract_tool_calls(
        '<tool_call>[{"name":"set_lights","arguments":{"room":"1","brightness":50}}]</tool_call>',
        tools,
    )
    assert calls[0]["function"]["name"] == "set_lights"
    assert calls[0]["function"]["arguments"] == '{"room": "1", "brightness": 50}'
