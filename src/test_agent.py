import json
import time
import ollama

MODELS = {"mist": "qwen2.5:0.5b", "fog": "llama3.2:1b", "cloud": "llama3.1:8b"}

PROMPT = """You are a resource agent at the {tier} layer.
Node state: CPU free 40%, memory free 512 MB, bandwidth 20 Mbps, latency 15 ms.
Task: needs 25% CPU, 300 MB memory.
Decide whether to execute locally or offload.
Reply ONLY with JSON: {{"decision": "local" or "offload", "reason": "<short>"}}"""

for tier, model in MODELS.items():
    try:
        start = time.perf_counter()
        resp = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": PROMPT.format(tier=tier)}],
            format="json",
            options={"temperature": 0},
        )
        elapsed = time.perf_counter() - start
        text = resp["message"]["content"]
        try:
            parsed = json.loads(text)
            valid = True
        except json.JSONDecodeError:
            parsed, valid = None, False
        print(f"[{tier}] model={model} time={elapsed:.2f}s valid_json={valid} output={text.strip()}")
    except Exception as e:
        print(f"[{tier}] model={model} ERROR: {e}")
