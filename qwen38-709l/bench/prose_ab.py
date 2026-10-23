# prose_ab.py <tag>: fixed-prompt prose speed and acceptance bench for one server.
# Endpoint settings as in common.py (BASE_URL, MODEL, API_KEY, METRICS_URL = the server's own vLLM /metrics, which must
# see no other client while this runs). Compare two runs with prose_compare.py <tag_a> <tag_b>.
# Three modes over the same 10 prose prompts, single stream, streamed:
#   G  greedy (temperature 0), thinking off, 600 tokens            paired across arms: near-identical text
#   S  the model's default sampling (generation_config: temperature 1.0, top_k 20, top_p 0.95), thinking off,
#      seeds 1 and 2, 500 tokens
#   T  thinking on, reasoning_effort medium, default sampling, seed 1, 1000 tokens (closest to an agent's prose turn)
# Per request: spec counters before/after (drafts, draft tokens, accepted), completion tokens, decode time (first token
# to last), written as JSON lines to prose_<tag>.jsonl next to this script (gitignored); a summary per mode is printed
# at the end. No generated text is stored, only its sha1 and length.
import hashlib, json, os, re, sys, time, urllib.request

from common import BASE, HDR, METRICS as MET, MODEL
URL = BASE + "/chat/completions"
tag = sys.argv[1]
ONLY = set(os.environ.get("PROSE_MODES", "G,S,T").split(","))
NP = int(os.environ.get("PROSE_N", "10"))   # prompts per mode (smoke tests use fewer)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, f"prose_{tag}.jsonl")
PROMPTS = [
    "Write about 400 words of plain prose on why canal towns in the Netherlands kept their 17th-century street plans while most European cities rebuilt theirs.",
    "In plain prose, explain to a curious teenager how a refrigerator moves heat from inside the box to the room, with no equations and no lists.",
    "Write the opening of a short story: a lighthouse keeper on a remote island finds a stranger's boot on the rocks after a storm. About 400 words, plain prose.",
    "Write a warm, professional email to a colleague who is leaving the team after six years, thanking them and recalling two projects you worked on together.",
    "Describe a slow morning walk through a mountain village market in autumn, in vivid but plain prose, about 400 words.",
    "Argue, in about 400 words of plain prose, whether small towns should invest in public libraries or in faster broadband first.",
    "Explain in plain prose how the Roman road network was planned, built and maintained, and why so much of it outlasted the empire.",
    "You are the narrator of a fantasy tavern scene: a bard, a tired soldier and a merchant argue about a rumour of a dragon. Write about 400 words of narration and dialogue.",
    "Write a short, honest product description for a handmade oak cutting board, in flowing prose, about 250 words, no bullet points.",
    "In about 400 words of plain prose, explain what makes a good code review comment and what makes a bad one, with no lists.",
]
KEYS = {"drafts": "vllm:spec_decode_num_drafts_total", "dtok": "vllm:spec_decode_num_draft_tokens_total",
        "acc": "vllm:spec_decode_num_accepted_tokens_total"}


def snap():
    t = urllib.request.urlopen(MET, timeout=20).read().decode()
    return {k: float(re.search(r"^" + re.escape(n) + r"\{[^}]*\} ([0-9.e+]+)$", t, re.M).group(1)) for k, n in KEYS.items()}


def run(prompt, mode, seed, mt):
    body = {"model": MODEL, "stream": True, "max_tokens": mt, "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": prompt}]}
    if mode == "G":
        body.update(temperature=0, chat_template_kwargs={"enable_thinking": False}, reasoning_effort="none")
    elif mode == "S":
        body.update(seed=seed, chat_template_kwargs={"enable_thinking": False}, reasoning_effort="none")
    else:
        body.update(seed=seed, chat_template_kwargs={"enable_thinking": True}, reasoning_effort="medium")
    req = urllib.request.Request(URL, json.dumps(body).encode(), HDR)
    a = snap(); t0 = time.time(); first = None; usage = None; text = []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            j = json.loads(line[6:])
            if j.get("usage"):
                usage = j["usage"]
            for c in j.get("choices", []):
                d = c.get("delta", {})
                piece = (d.get("reasoning") or d.get("reasoning_content") or "") + (d.get("content") or "")
                if piece:
                    first = first or time.time(); text.append(piece)
    t1 = time.time(); b = snap()
    n = (usage or {}).get("completion_tokens", 0)
    row = {"tag": tag, "mode": mode, "seed": seed, "prompt": PROMPTS.index(prompt), "tokens": n,
           "decode_s": round(t1 - (first or t0), 3), "ttft_s": round((first or t1) - t0, 3),
           "drafts": b["drafts"] - a["drafts"], "draft_tokens": b["dtok"] - a["dtok"], "accepted": b["acc"] - a["acc"],
           "text_sha": hashlib.sha1("".join(text).encode()).hexdigest()[:12]}
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return row


plan = []
if "G" in ONLY:
    plan += [(p, "G", 0, 600) for p in PROMPTS[:NP]]
if "S" in ONLY:
    plan += [(p, "S", s, 500) for s in (1, 2) for p in PROMPTS[:NP]]
if "T" in ONLY:
    plan += [(p, "T", 1, 1000) for p in PROMPTS[:NP]]
rows = []
for p, mode, seed, mt in plan:
    r = run(p, mode, seed, mt); rows.append(r)
    tps = r["tokens"] / max(r["decode_s"], 1e-6); ts = 1 + r["accepted"] / max(r["drafts"], 1)
    print(f"{tag} {mode} seed {seed} p{r['prompt']}: {r['tokens']} tok, {tps:6.1f} t/s, {ts:4.2f} tok/step", flush=True)
for mode in ("G", "S", "T"):
    rs = [r for r in rows if r["mode"] == mode]
    if not rs:
        continue
    dr, acc, tok, dec = (sum(r[k] for r in rs) for k in ("drafts", "accepted", "tokens", "decode_s"))
    print(f"SUMMARY {tag} {mode}: {len(rs)} requests, {tok} tokens, tok/step {1 + acc / dr:.3f}, "
          f"accepted/draft-token {acc / max(sum(r['draft_tokens'] for r in rs), 1):.4f}, decode {tok / dec:.1f} t/s", flush=True)
