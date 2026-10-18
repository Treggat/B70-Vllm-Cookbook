"""PTC (programmatic tool calling) reliability probe: does a server emit a VALID run_code call when the Python it must
write is full of awkward strings? Streaming, like an agent harness. Read-only: nothing is executed anywhere.

Usage: python ptc_probe.py BASE MODEL [--parallel 4] [--extra '{"reasoning_effort":"low"}'] [--rounds 1]
  e.g. python ptc_probe.py http://localhost:8000 gpt-oss --rounds 6 --extra '{"reasoning_effort":"medium"}'
--extra keys set to null are removed from the request (e.g. temperature null = no sampler sent).
Set OPENAI_API_KEY in the environment to send a bearer key.
--dump FILE appends one JSON line per FAILED call (task, verdict, raw tool calls, content) for diagnosis.
"""
import ast, json, os, sys, time, textwrap, threading, urllib.request

BASE = sys.argv[1].rstrip('/'); MODEL = sys.argv[2]; ARGS = sys.argv[3:]
def opt(n, d): return ARGS[ARGS.index(n) + 1] if n in ARGS else d
PAR = int(opt('--parallel', '4')); ROUNDS = int(opt('--rounds', '1')); EXTRA = json.loads(opt('--extra', '{}'))
DUMP = opt('--dump', ''); DUMP_LOCK = threading.Lock()

SYSTEM = """You are a coding agent. `run_code` is the ONLY tool you can call. Its `code` argument is the BODY of an
async PYTHON function (top-level await and return work). Inside it, call tools as `await tools.name({...})` with a
plain dict. Available: tools.bash({"command": str}), tools.read({"file_path": str}), tools.write({"file_path": str,
"content": str}), tools.edit({"file_path": str, "old_string": str, "new_string": str}), tools.grep({"pattern": str,
"path": str}). Print what you want to see with print(). Do exactly what the user asks in ONE run_code call. Do not
explain, do not ask questions."""

TOOLS = [{"type": "function", "function": {"name": "run_code", "description": "Run the body of an async Python function with `tools` bound.",
          "parameters": {"type": "object", "additionalProperties": False, "required": ["code", "description"],
                         "properties": {"code": {"type": "string", "description": "Body of an async Python function."},
                                        "description": {"type": "string", "description": "Short summary."}}}}}]

TASKS = [
    'Write the file /tmp/greet.py. It must contain a function greet(name) with a TRIPLE-QUOTED docstring saying Return a greeting. and it returns f"Hello, {name}!".',
    'In /tmp/a.py replace the exact text print("hi") with print("""hello "world" """) using tools.edit.',
    r'Search /app for the regex \bclass\s+\w+\(Protocol\): using tools.grep and print the result.',
    'Write /tmp/cfg.json containing this exact JSON: {"name": "a \\"quoted\\" value", "path": "C:\\\\Users\\\\me\\\\x", "list": [1, 2, {"k": "v"}]}',
    "Run this shell pipeline with tools.bash and print the output: grep -rn \"def restrict\" /app/packages | awk -F: '{print $1\":\"$2}' | head -5",
    'Write /tmp/notes.md whose content has three lines: a heading "# Notes", a line with a backtick code span `x = "y"`, and a line containing a single backslash \\ and a tab character.',
    "Read /app/package.json, parse it with json, and print the version field and the number of keys in scripts, using an f-string that contains both single and double quotes.",
    'Write /tmp/sql.py that defines QUERY as a triple-quoted string containing: SELECT * FROM t WHERE name = \'O\'\'Brien\' AND note LIKE "%50\\%%";',
    "Use tools.bash to run a heredoc: cat <<'EOF' > /tmp/h.txt with two lines, line one and line \"two\", then EOF; then read the file back and print it.",
    'In /tmp/b.py replace the text old = re.compile(r"\\d+\\.\\d+") with new = re.compile(r"(?P<major>\\d+)\\.(?P<minor>\\d+)") using tools.edit.',
    'Write /tmp/t.yaml with keys: title: "He said \\"hi\\"", path: \'C:\\temp\\new\', and a block scalar key body: | with two indented lines.',
    "Grep /app/packages/core/tools/src/index.ts for the pattern restrict\\( and print only the first 3 matching lines.",
]

def check(args_text):
    try: a = json.loads(args_text)
    except Exception as e: return 'INVALID JSON args: ' + str(e)[:80]
    if not isinstance(a, dict) or not isinstance(a.get('code'), str): return 'args lack a string `code`'
    try: ast.parse('async def __f():\n' + textwrap.indent(a['code'], '    ') + '\n')
    except SyntaxError as e: return f'code is not valid Python: {e.msg} (line {e.lineno})'
    if 'tools.' not in a['code']: return 'code never calls tools'
    return None

def one(i, task, out):
    body = {'model': MODEL, 'stream': True, 'max_tokens': 3000, 'temperature': 0.6, 'tools': TOOLS, 'tool_choice': 'auto',
            'stream_options': {'include_usage': True},
            'messages': [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': task}]}
    body.update(EXTRA)
    body = {k: v for k, v in body.items() if v is not None}  # --extra '{"temperature": null}' = send none, like many agent harnesses
    headers = {'content-type': 'application/json'}
    if os.environ.get('OPENAI_API_KEY'): headers['authorization'] = 'Bearer ' + os.environ['OPENAI_API_KEY']
    req = urllib.request.Request(BASE + '/v1/chat/completions', data=json.dumps(body).encode(), headers=headers)
    t0 = time.time(); calls = {}; content = ''; reasoning = 0; finish = None; usage = None; err = None
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            for raw in resp:
                line = raw.decode('utf-8', 'replace').strip()
                if not line.startswith('data:'): continue
                d = line[5:].strip()
                if d == '[DONE]': continue  # read to EOF: closing early makes some proxies log the request as aborted
                try: o = json.loads(d)
                except Exception: continue
                if o.get('usage'): usage = o['usage']
                for ch in o.get('choices') or []:
                    dl = ch.get('delta') or {}
                    content += dl.get('content') or ''
                    reasoning += len(dl.get('reasoning_content') or dl.get('reasoning') or '')
                    for tc in dl.get('tool_calls') or []:
                        c = calls.setdefault(tc.get('index', 0), {'name': '', 'args': ''})
                        fn = tc.get('function') or {}
                        c['name'] += fn.get('name') or ''; c['args'] += fn.get('arguments') or ''
                    if ch.get('finish_reason'): finish = ch['finish_reason']
    except Exception as e:
        err = 'HTTP/stream error: ' + str(e)[:120]
    verdict = err
    if not verdict:
        if not calls: verdict = f'NO tool call (finish={finish}, said: {content[:70]!r})'
        elif len(calls) > 1: verdict = f'{len(calls)} tool calls instead of 1'
        else:
            c = list(calls.values())[0]
            verdict = f'wrong tool name {c["name"]!r}' if c['name'] != 'run_code' else check(c['args'])
    out[i] = {'ok': verdict is None, 'why': verdict, 'leak': '<|' in content, 'secs': time.time() - t0, 'reasoning_chars': reasoning,
              'out_tokens': (usage or {}).get('completion_tokens')}
    if verdict is not None and DUMP:
        with DUMP_LOCK, open(DUMP, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'task': i % len(TASKS) + 1, 'why': verdict, 'finish': finish,
                                'calls': list(calls.values()), 'content': content}) + '\n')

jobs = [(r * len(TASKS) + i, t) for r in range(ROUNDS) for i, t in enumerate(TASKS)]
out = {}; t0 = time.time(); sem = threading.Semaphore(PAR); threads = []
def runner(i, t):
    with sem: one(i, t, out)
for i, t in jobs:
    th = threading.Thread(target=runner, args=(i, t)); th.start(); threads.append(th)
for th in threads: th.join()
ok = sum(1 for v in out.values() if v['ok']); secs = sorted(v['secs'] for v in out.values())
toks = [v['out_tokens'] for v in out.values() if v['out_tokens']]
print(f"{MODEL}: {ok}/{len(out)} valid run_code calls | per call: median {secs[len(secs)//2]:.1f}s, slowest {secs[-1]:.1f}s | "
      f"avg output {sum(toks)/max(1,len(toks)):.0f} tok, avg reasoning {sum(v['reasoning_chars'] for v in out.values())/len(out):.0f} chars | "
      f"{PAR} at once, whole probe {time.time()-t0:.0f}s | leaked control text {sum(v['leak'] for v in out.values())}")
for i in sorted(out):
    if not out[i]['ok']: print(f"   task {i % len(TASKS) + 1}: {out[i]['why']}")
