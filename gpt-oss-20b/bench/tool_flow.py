"""Streaming multi-turn tool-call flow, coding-agent style (OpenCode-like tools), for a harmony (gpt-oss) server. Nothing is executed:
tool results are simulated, so the test only judges what the SERVER emits across turns.

Why: an earlier leak (tool name `write<|channel|>commentary`, harmony control tokens in the recipient) only
showed up in real multi-turn streaming flows, never in single-turn curl tests.

Usage: python tool_flow.py BASE MODEL [--sessions 6] [--turns 8] [--extra '{"reasoning_effort":"medium"}']
  e.g. python tool_flow.py http://localhost:8000 gpt-oss --sessions 16 --turns 8
Set OPENAI_API_KEY if the server wants a bearer key. Per session it prints the tool calls made and every problem found; the last line sums up.
"""
import json, os, re, sys, threading, time, urllib.error, urllib.request

BASE = sys.argv[1].rstrip('/'); MODEL = sys.argv[2]; ARGS = sys.argv[3:]
def opt(n, d): return ARGS[ARGS.index(n) + 1] if n in ARGS else d
SESSIONS = int(opt('--sessions', '6')); TURNS = int(opt('--turns', '8')); EXTRA = json.loads(opt('--extra', '{}'))
CTRL = re.compile(r'<\|(channel|constrain|message|start|end|call|return|endoftext)\|>')

def fn(name, desc, props, req):
    return {'type': 'function', 'function': {'name': name, 'description': desc, 'strict': False,
            'parameters': {'type': 'object', 'properties': props, 'required': req}}}
S = {'type': 'string'}
TOOLS = [fn('read', 'Read a file.', {'filePath': S}, ['filePath']),
         fn('write', 'Write a whole file.', {'filePath': S, 'content': S}, ['filePath', 'content']),
         fn('edit', 'Replace exact text in a file.', {'filePath': S, 'oldString': S, 'newString': S}, ['filePath', 'oldString', 'newString']),
         fn('bash', 'Run a shell command.', {'command': S, 'description': S}, ['command', 'description']),
         fn('grep', 'Search files with a regex.', {'pattern': S, 'path': S}, ['pattern'])]
NAMES = {t['function']['name'] for t in TOOLS}
SYSTEM = ("You are a coding agent working in /work. Use the tools to do the task step by step: look first, then write, "
          "then run the tests with bash, then fix anything that fails. When the task is done, reply with one short "
          "sentence and no tool call.")
TASKS = [
    'Create /work/greet.py with greet(name) returning f"Hello, {name}!" and /work/test_greet.py with two pytest tests, then run pytest.',
    'Read /work/app.py, then add a function slugify(s) that lowercases and replaces spaces with "-", and run python -c to test it.',
    'Find every TODO in /work with grep, then write /work/TODO.md listing them as a markdown checklist.',
    'Write /work/cfg.json with {"name": "a \\"q\\" b", "path": "C:\\\\x"} then read it back with bash cat and confirm it parses with python -m json.tool.',
    'Create /work/bucket.py with a TokenBucket class (capacity, refill rate, take(n) -> bool) and a test file, then run pytest.',
    'Edit /work/app.py to rename function old_name to new_name everywhere, then grep to confirm no old_name is left.',
]

def fake_result(name, args, turn):
    if name == 'read':
        return 'def old_name(x):\n    return x  # TODO: validate\n\nprint(old_name(1))\n'
    if name == 'grep':
        return '/work/app.py:2:    return x  # TODO: validate'
    if name == 'bash':
        cmd = args.get('command', '')
        if 'pytest' in cmd:
            return '2 passed in 0.03s' if turn > 3 else 'FAILED test_x.py::test_a - AssertionError\n1 failed, 1 passed'
        return 'ok'
    return 'Wrote file successfully.' if name == 'write' else 'Edit applied.'

def stream(messages):
    body = {'model': MODEL, 'stream': True, 'messages': messages, 'tools': TOOLS, 'tool_choice': 'auto',
            'max_tokens': 4000, 'stream_options': {'include_usage': True}}
    body.update(EXTRA); body = {k: v for k, v in body.items() if v is not None}
    h = {'content-type': 'application/json'}
    if os.environ.get('OPENAI_API_KEY'): h['authorization'] = 'Bearer ' + os.environ['OPENAI_API_KEY']
    req = urllib.request.Request(BASE + '/v1/chat/completions', data=json.dumps(body).encode(), headers=h)
    content, reasoning, calls, finish = '', '', {}, None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode('utf-8', 'replace').strip()
            if not line.startswith('data:') or line == 'data: [DONE]': continue
            ev = json.loads(line[5:])
            if 'error' in ev: raise RuntimeError('stream error event: ' + json.dumps(ev['error'])[:200])
            for ch in ev.get('choices', []):
                d = ch.get('delta') or {}
                content += d.get('content') or ''
                reasoning += d.get('reasoning') or d.get('reasoning_content') or ''
                for tc in d.get('tool_calls') or []:
                    c = calls.setdefault(tc.get('index', 0), {'id': '', 'name': '', 'args': ''})
                    c['id'] = tc.get('id') or c['id']
                    f = tc.get('function') or {}
                    c['name'] += f.get('name') or ''; c['args'] += f.get('arguments') or ''
                finish = ch.get('finish_reason') or finish
    return content, reasoning, [calls[k] for k in sorted(calls)], finish

totals = {'sessions': 0, 'turns': 0, 'calls': 0, 'bad_name': 0, 'ctrl_leak': 0, 'bad_json': 0, 'http_err': 0,
          'bang': 0, 'done': 0}
LOCK = threading.Lock()

def session(i):
    task = TASKS[i % len(TASKS)]
    msgs = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': task}]
    log, prob = [], []
    t0 = time.time()
    for turn in range(1, TURNS + 1):
        try:
            content, reasoning, calls, finish = stream(msgs)
        except urllib.error.HTTPError as e:
            prob.append(f't{turn} HTTP {e.code}: {e.read()[:160]!r}'); bump('http_err'); break
        except Exception as e:  # noqa: BLE001
            prob.append(f't{turn} {type(e).__name__}: {str(e)[:160]}'); bump('http_err'); break
        bump('turns')
        if content.count('!') > 50: prob.append(f't{turn} "!" run in content'); bump('bang')
        for where, text in (('content', content), ('reasoning', reasoning)):
            if where == 'content' and CTRL.search(text): prob.append(f't{turn} control token in {where}: {CTRL.search(text).group(0)}'); bump('ctrl_leak')
        if not calls:
            log.append(f't{turn}:final({finish})'); bump('done'); break
        tool_calls = []
        for c in calls:
            bump('calls')
            if c['name'] not in NAMES:
                prob.append(f't{turn} tool name {c["name"]!r} not offered'); bump('bad_name')
            if CTRL.search(c['name'] + c['args']):
                prob.append(f't{turn} control token in call {c["name"]!r}'); bump('ctrl_leak')
            try:
                args = json.loads(c['args'] or '{}')
            except Exception:  # noqa: BLE001
                prob.append(f't{turn} invalid JSON args for {c["name"]!r}: {c["args"][:80]!r}'); bump('bad_json'); args = {}
            tool_calls.append({'id': c['id'] or f'call_{turn}_{len(tool_calls)}', 'type': 'function',
                               'function': {'name': c['name'], 'arguments': c['args'] or '{}'}})
            log.append(f't{turn}:{c["name"]}')
        msgs.append({'role': 'assistant', 'content': content or None, 'tool_calls': tool_calls})
        for tc in tool_calls:
            try: a = json.loads(tc['function']['arguments'])
            except Exception: a = {}  # noqa: BLE001
            msgs.append({'role': 'tool', 'tool_call_id': tc['id'], 'content': fake_result(tc['function']['name'], a, turn)})
    with LOCK:
        totals['sessions'] += 1
        print(f'[s{i}] {time.time() - t0:5.1f}s {" ".join(log)}' + ''.join('\n      PROBLEM ' + p for p in prob), flush=True)

def bump(k):
    with LOCK: totals[k] += 1

th = [threading.Thread(target=session, args=(i,)) for i in range(SESSIONS)]
[t.start() for t in th]; [t.join() for t in th]
print(f'TOTAL {MODEL} {json.dumps(EXTRA)}: ' + ', '.join(f'{k} {v}' for k, v in totals.items()))
