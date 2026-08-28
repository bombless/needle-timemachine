"""Interactive experiments for the Needle Time Machine UI."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from .trace import Tracer


TEMPLATES = [
    {
        "id": "thermostat",
        "name": "Thermostat tool call",
        "description": "A canonical Needle tool-calling example.",
        "system": "You control a smart home. Use tools when they match the user's request.",
        "prompt": "Make it 21 degrees and cool the room.",
        "tools": [
            {"name": "set_thermostat", "description": "Set the room thermostat.", "parameters": {"type": "object", "properties": {"temperature": {"type": "integer", "description": "Target temperature in Celsius."}, "mode": {"type": "string", "enum": ["heat", "cool", "auto"]}}, "required": ["temperature"]}},
        ],
    },
    {
        "id": "weather",
        "name": "Weather lookup",
        "description": "See how a natural-language request selects a simple tool.",
        "system": "You are a concise assistant. Call a tool when the user asks for live information.",
        "prompt": "What's the weather in Tokyo?",
        "tools": [
            {"name": "get_weather", "description": "Get current weather for a city.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}},
            {"name": "get_time", "description": "Get the current local time for a city.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}},
        ],
    },
    {
        "id": "extraction",
        "name": "Structured extraction",
        "description": "Extract fields from a passage using a record schema.",
        "system": "Extract only values that are explicitly present in the user's text.",
        "prompt": "Order from GreenMart: 2 oat milks at $3.50 each. Total $7.00, paid by Visa.",
        "tools": [
            {"name": "receipt", "description": "Extract receipt information from text.", "parameters": {"type": "object", "properties": {"store": {"type": "string"}, "total": {"type": "number"}, "payment_method": {"type": "string"}}, "required": ["store", "total"]}},
        ],
    },
]


def render_needle_prompt(query: str, tools: list[dict[str, Any]], system: str = "") -> str:
    """Match the prompt format used by Needle's training renderer."""
    try:
        from needle.model.finetune import render_example
        prompt, _ = render_example({"query": query, "tools": tools, "system": system})
        return prompt
    except ImportError:
        tools_json = json.dumps(tools, separators=(",", ":"), ensure_ascii=False)
        prefix = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
        return prefix + f"<|im_start|>user\n<tools>{tools_json}</tools>\n{query}<|im_end|>\n<|im_start|>assistant\n"


def _softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def _token_info(tokenizer: Any, token_id: int) -> dict[str, Any]:
    try:
        text = tokenizer.decode([token_id])
    except Exception:
        text = ""
    raw = text.encode("utf-8", errors="replace")
    return {"token_id": int(token_id), "token_text": text, "token_bytes_hex": raw.hex(" ").upper()}


def run_experiment(runtime: Any, query: str, tools: list[dict[str, Any]], system: str = "", *, max_new_tokens: int = 96, top_k: int = 5) -> dict[str, Any]:
    """Run greedy Needle generation while recording per-token Top-K probabilities."""
    if not query.strip():
        raise ValueError("Prompt cannot be empty")
    prompt = render_needle_prompt(query, tools, system)
    ids = [runtime.tokenizer.bos_token_id] + runtime.tokenizer.encode(prompt)
    max_len = int(runtime.config.max_seq_len)
    if len(ids) + 1 > max_len:
        raise ValueError(f"Prompt is too long ({len(ids)} tokens); max_seq_len={max_len}")

    events: list[dict[str, Any]] = []
    generated: list[int] = []
    for step in range(max_new_tokens):
        tokens = np.asarray([ids + [0] * (max_len - len(ids))], dtype=np.int32)
        output = runtime.model.apply({"params": runtime.params}, tokens)
        logits = output[0, len(ids) - 1] if hasattr(output, "shape") and output.ndim == 3 else np.asarray(output)[0, len(ids) - 1]
        probs = _softmax(np.asarray(logits))
        k = min(max(1, int(top_k)), probs.size)
        indices = np.argpartition(probs, -k)[-k:]
        indices = indices[np.argsort(probs[indices])[::-1]]
        top = []
        for index in indices:
            item = _token_info(runtime.tokenizer, int(index))
            item["probability"] = float(probs[index])
            top.append(item)
        chosen = int(indices[0])
        chosen_info = _token_info(runtime.tokenizer, chosen)
        chosen_info["probability"] = float(probs[chosen])
        events.append({"step": step, "chosen": chosen_info, "top_k": top})
        generated.append(chosen)
        ids.append(chosen)
        if chosen == getattr(runtime.tokenizer, "eos_token_id", -1):
            break
        if len(ids) >= max_len:
            break

    return {
        "query": query,
        "system": system,
        "tools": tools,
        "rendered_prompt": prompt,
        "generated_token_ids": generated,
        "generated_text": runtime.tokenizer.decode(generated),
        "events": events,
        "top_k": top_k,
    }
