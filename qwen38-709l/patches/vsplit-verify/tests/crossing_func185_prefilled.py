"""crossing_func185_prefilled.py: reproduce the crossing.py slip (func_185 -> 186) with short sampled continuations.
Each request: the crossing prompt, the assistant turn pre-filled with func_0..func_{n-1} (continue_final_message),
then up to 60 sampled tokens (temperature 1.0, the model's top-k/top-p). A continuation is wrong when its funcs are
not exactly func_n, func_n+1, ... with the right bodies. pad = extra user-prompt tokens to move the text across the
960-token page boundary (control). usage: python crossing_func185_prefilled.py <n> <reps> <pad_words>"""
import json, os, re, sys, urllib.request
BASE = os.environ["BASE_URL"]; MODEL = os.environ["MODEL"]
n, reps, pad = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
seed = "\n".join(f"def func_{i}(x):\n    return x * {i} + {i % 7}\n" for i in range(6))
prompt = ("Continue this Python file with the same pattern up to and including func_399 "
          "(func_i returns x * i + (i % 7)). Output the whole file, code only, no commentary, no code fences.\n\n" + seed)
if pad:
    prompt = " ".join(["note"] * pad) + "\n\n" + prompt
pre = "\n".join(f"def func_{i}(x):\n    return x * {i} + {i % 7}\n" for i in range(n)) + "\n"
bad = 0; ptok = None; examples = []
for r in range(reps):
    req = {"model": MODEL, "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": pre}],
           "continue_final_message": True, "add_generation_prompt": False, "max_tokens": 60, "temperature": 1.0,
           "chat_template_kwargs": {"enable_thinking": False}}
    j = json.load(urllib.request.urlopen(urllib.request.Request(BASE + "/chat/completions", json.dumps(req).encode(),
                                                                {"Content-Type": "application/json"}), timeout=300))
    t = j["choices"][0]["message"]["content"] or ""; ptok = j["usage"]["prompt_tokens"]
    want = n; ok = True
    for m in re.finditer(r"def func_(\d+)\(x\):\n    return (.*)\n", t):
        if int(m.group(1)) != want or m.group(2).strip() != f"x * {want} + {want % 7}":
            ok = False; break
        want += 1
    if want == n:
        ok = False
    if not ok:
        bad += 1; examples.append(t[:90])
print(f"n={n} pad={pad} prompt_tokens={ptok} (page edge 3840 at +{3840 - ptok}): {bad}/{reps} wrong", flush=True)
for e in examples[:3]:
    print("   ", repr(e))
