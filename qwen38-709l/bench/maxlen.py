"""Max-model-len boundary test: the case the MTP boundary patch covers (patches/patch_mtp_boundary_032.py). A prompt of
about (max_model_len - 2k) tokens asks for a reply longer than the space left, so generation runs into the window and
the last MTP step is a partial speculative group at exactly max-model-len. PASS = the request ends cleanly (HTTP 200,
finish_reason "length", prompt + completion == the window), the reply has no "!!!!" run or replacement characters, and
the server answers a follow-up. Send it to the server DIRECTLY: a router that estimates token counts (ours does) refuses
prompts this close to the window, and the test needs the exact edge.
Usage: python maxlen.py [max_model_len=131072] [prompt_tokens=max_model_len-2000]   (endpoint/model from env, see common.py)"""
import json, os, re, sys, time
from common import complete, stdlib_mix

WINDOW = int(sys.argv[1]) if len(sys.argv) > 1 else 131072
TOK = int(sys.argv[2]) if len(sys.argv) > 2 else WINDOW - 2000
SRC = stdlib_mix(["typing.py", "argparse.py", "inspect.py", "subprocess.py", "dataclasses.py", "functools.py", "logging/__init__.py",
                  "shutil.py", "tarfile.py", "zipfile/__init__.py", "email/_header_value_parser.py", "asyncio/base_events.py",
                  "collections/__init__.py", "enum.py", "json/decoder.py", "http/client.py", "urllib/request.py", "unittest/mock.py",
                  "pydoc.py", "configparser.py", "datetime.py", "_pydecimal.py", "pickle.py", "ast.py", "doctest.py", "difflib.py",
                  "smtplib.py", "imaplib.py", "ftplib.py", "turtle.py", "statistics.py", "fractions.py", "threading.py", "socket.py"])
ASK = "\n\nNow write the numbers from 1 to 20000 in words, one per line, without stopping or commenting."
head = f"BOUNDARY-TEST {time.time():.0f}\n"
chars = int(TOK * 3.4)
for _ in range(4):   # calibrate the prompt to TOK real tokens with max_tokens=1 requests (the prefix cache makes repeats cheap)
    msg = head + SRC[:chars] + ASK
    _, u = complete(msg, 1, temp=0)
    got = u["prompt_tokens"]
    print("calibrate", chars, "chars ->", got, "tokens", flush=True)
    if abs(got - TOK) < 300:
        break
    chars = int(chars * TOK / got)
t = time.time()
c, u = complete(msg, 8000, temp=0.2)
text = c["message"]["content"] or ""
res = {"window": WINDOW, "prompt_tokens": u["prompt_tokens"], "completion_tokens": u["completion_tokens"],
       "total": u["prompt_tokens"] + u["completion_tokens"], "finish": c["finish_reason"], "secs": round(time.time() - t, 1),
       "bang_runs": len(re.findall(r"!{4,}", text)), "replacement": text.count("�"), "tail": text[-160:]}
f, _ = complete("Capital of France? One word.", 10, temp=0)
res["followup"] = f["message"]["content"]
res["pass"] = (res["finish"] == "length" and res["total"] == WINDOW and res["bang_runs"] == 0 and res["replacement"] == 0
               and "Paris" in (res["followup"] or ""))
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "maxlen.json"), "w"), indent=1)
print("PASS" if res["pass"] else "FAIL")
