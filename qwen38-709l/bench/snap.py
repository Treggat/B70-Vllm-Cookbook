# snap.py save NAME   -> store current 709L seat counters as NAME
# snap.py diff A B    -> per-turn numbers between two snapshots
import json, re, sys, urllib.request, os
D = os.path.dirname(os.path.abspath(__file__))
URL = "http://192.168.1.218:8080/upstream/VLLM-Qwen3.8-Fable-709L/metrics"
KEYS = {
    "gen": "vllm:generation_tokens_total",
    "prompt": "vllm:prompt_tokens_total",
    "dec_s": "vllm:request_decode_time_seconds_sum",
    "reqs": "vllm:request_decode_time_seconds_count",
    "drafts": "vllm:spec_decode_num_drafts_total",
    "dtok": "vllm:spec_decode_num_draft_tokens_total",
    "acc": "vllm:spec_decode_num_accepted_tokens_total",
    "running": "vllm:num_requests_running",
}


def scrape():
    # only safe while this seat is already loaded (llama-swap would load it otherwise)
    run = json.load(urllib.request.urlopen("http://192.168.1.218:8080/running", timeout=10))
    assert any(r["model"] == "VLLM-Qwen3.8-Fable-709L" and r["state"] == "ready" for r in run["running"]), run
    txt = urllib.request.urlopen(URL, timeout=10).read().decode()
    out = {}
    for k, name in KEYS.items():
        m = re.search(r"^" + re.escape(name) + r"\{[^}]*\} ([0-9.e+]+)$", txt, re.M)
        out[k] = float(m.group(1)) if m else 0.0
    return out


if sys.argv[1] == "save":
    s = scrape()
    json.dump(s, open(os.path.join(D, f"snap_{sys.argv[2]}.json"), "w"))
    print(sys.argv[2], s)
else:
    a = json.load(open(os.path.join(D, f"snap_{sys.argv[2]}.json")))
    b = json.load(open(os.path.join(D, f"snap_{sys.argv[3]}.json")))
    d = {k: b[k] - a[k] for k in KEYS}
    steps = d["drafts"]
    copy = (d["dtok"] - 5 * steps) / 10 if steps else 0
    print(f"requests {d['reqs']:.0f}, prompt tok {d['prompt']:.0f}, gen tok {d['gen']:.0f}, "
          f"decode t/s {d['gen'] / d['dec_s']:.1f}" if d["dec_s"] else "no decode",
          f"| steps {steps:.0f}, tok/step {(d['acc'] + steps) / steps:.2f}, drafts/step {d['dtok'] / steps:.2f}" if steps else "",
          f"| copy steps {copy:.0f} ({100 * copy / steps:.0f}%)" if steps else "",
          f"| accept {100 * d['acc'] / d['dtok']:.1f}%" if d["dtok"] else "")
