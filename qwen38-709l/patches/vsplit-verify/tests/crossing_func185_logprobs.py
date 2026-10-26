"""crossing_func185_logprobs.py: the next-token distribution where bench/crossing.py went wrong (func_185, about 3,840 tokens
into the reply), from a prefilled assistant turn, plus two controls. usage: BASE_URL=... MODEL=... python crossing_func185_logprobs.py"""
import json, os, urllib.request, math
BASE = os.environ["BASE_URL"]; MODEL = os.environ["MODEL"]
seed = "\n".join(f"def func_{i}(x):\n    return x * {i} + {i % 7}\n" for i in range(6))
prompt = ("Continue this Python file with the same pattern up to and including func_399 "
          "(func_i returns x * i + (i % 7)). Output the whole file, code only, no commentary, no code fences.\n\n" + seed)


def body(n):
    return "\n".join(f"def func_{i}(x):\n    return x * {i} + {i % 7}\n" for i in range(n))


CASES = [
    ("after func_184, name 18?", body(185) + "\ndef func_18"),
    ("func_185 body 18?", body(185) + "\ndef func_185(x):\n    return x * 18"),
    ("after func_183, name 18? (control)", body(184) + "\ndef func_18"),
    ("after func_104, name 10? (control)", body(105) + "\ndef func_10"),
]
for label, pre in CASES:
    req = {"model": MODEL, "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": pre}],
           "continue_final_message": True, "add_generation_prompt": False, "max_tokens": 1, "temperature": 1.0,
           "logprobs": True, "top_logprobs": 8, "chat_template_kwargs": {"enable_thinking": False}}
    r = json.load(urllib.request.urlopen(urllib.request.Request(BASE + "/chat/completions", json.dumps(req).encode(),
                                                                {"Content-Type": "application/json"}), timeout=300))
    c = r["choices"][0]; u = r["usage"]
    tops = c["logprobs"]["content"][0]["top_logprobs"]
    print(f"{label}: prompt {u['prompt_tokens']} tok | " + "  ".join(f"{t['token']!r} {math.exp(t['logprob']):.4f}" for t in tops), flush=True)
