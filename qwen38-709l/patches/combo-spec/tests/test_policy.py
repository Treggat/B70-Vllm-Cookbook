"""Offline check of b70_combo_policy: synthetic requests that copy (with edits) vs write fresh tokens."""
import importlib.util, random, sys, pathlib
P = pathlib.Path(__file__).resolve().parents[2] / "patches" / "combo" / "b70_combo_policy.py"
spec = importlib.util.spec_from_file_location("pol", P); pol = importlib.util.module_from_spec(spec); spec.loader.exec_module(pol)

class Req:
    def __init__(s, rid, prompt): s.request_id = rid; s._all_token_ids = list(prompt); s.np = len(prompt); s.is_prefill_chunk = False
    @property
    def num_output_tokens(s): return len(s._all_token_ids) - s.np
class Sched:
    def __init__(s): s.requests = {}; s.num_spec_tokens = 15

random.seed(1)
src = [random.randrange(1000, 60000) for _ in range(3000)]           # a "file" in the prompt
prompt = [random.randrange(1000, 60000) for _ in range(500)] + src + [7, 8, 9]
def run(output_stream, label, lag=2, accept_fn=None):
    sch = Sched(); r = Req(label, prompt); sch.requests[label] = r
    pol._state.clear(); ks = []
    pending = []  # async lag: tokens the scheduler cannot see yet
    i = 0
    while i < len(output_stream):
        k = pol.choose_k(sch, {label: 1}, 5); ks.append(k)
        # simulate a step: accepted tokens = all copied-or-not; yield depends on mode
        y = accept_fn(i, k) if accept_fn else (12 if k == 15 else 5)
        new = output_stream[i:i + y]; i += y
        pending.append(new)
        if len(pending) > lag: r._all_token_ids += pending.pop(0)
        pol.observe(r, k, y - 1, 15)
    return ks
# 1) pure copy of the file with an edit every 400 tokens
copy = []
for j in range(0, 3000, 400):
    copy += src[j:j + 397] + [42, 43, 44]
ks = run(copy, "copy")
print("copy-with-edits: steps", len(ks), "copy-K share %.0f%%" % (100 * ks.count(15) / len(ks)))
# 2) fresh code: random tokens (no copying) -> must stay MTP
fresh = [random.randrange(1000, 60000) for _ in range(3000)]
ks = run(fresh, "fresh")
print("fresh: copy-K steps", ks.count(15), "of", len(ks))
# 3) false trigger: boilerplate repeats a 4-gram but continuation differs -> back-off
bp = []
for j in range(200):
    bp += src[100:104] + [random.randrange(1000, 60000) for _ in range(11)]
ks = run(bp, "boiler", accept_fn=lambda i, k: 1 if k == 15 else 5)
print("boilerplate w/ low copy yield: copy-K steps", ks.count(15), "of", len(ks))
# 3b) copy of a REPETITIVE source (code-like: 4-grams repeat inside the file), edit every 400 tokens
rep_src = [random.choice([11, 12, 13, 14, 15, 16]) if random.random() < 0.7 else random.randrange(1000, 60000) for _ in range(3000)]
prompt_save = prompt
prompt = [random.randrange(1000, 60000) for _ in range(500)] + rep_src + [7, 8, 9]
rc = []
for j in range(0, 3000, 400):
    rc += rep_src[j:j + 397] + [42, 43, 44]
ks = run(rc, "repcopy")
print("repetitive-source copy: copy-K share %.0f%%" % (100 * ks.count(15) / len(ks)))
prompt = prompt_save
# 4) long context -> never copy
pol.MAX_CTX = 2000
ks = run(copy, "long"); print("ctx > MAX_CTX: copy-K steps", ks.count(15))
