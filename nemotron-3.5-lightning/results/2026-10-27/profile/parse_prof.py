import re, sys
def ms(s):
    m = re.match(r"([\d.]+)(us|ms|s)$", s); v = float(m.group(1)); return v / 1000 if m.group(2) == "us" else v * 1000 if m.group(2) == "s" else v
for f in sys.argv[1:]:
    rows = []; steps = None
    for line in open(f):
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) < 11 or not re.match(r"[\d.]+(us|ms|s)$", parts[6] if len(parts) > 6 else ""): continue
        name, selfxpu, calls = parts[0], ms(parts[6]), int(parts[10])
        if name.startswith("execute_context") and steps is None: steps = calls
        rows.append((name, selfxpu, calls))
    kern = [r for r in rows if not r[0].startswith(("execute_context", "ze", "ur", "aten::", "_xpu_C::", "_C::", "_moe_C::", "vllm::mamba", "_vllm_fa2"))]
    tot = sum(r[1] for r in kern)
    print(f"== {f}: {steps} steps, kernel time {tot/steps:.2f} ms/step")
    for n, t, c in sorted(kern, key=lambda r: -r[1])[:14]:
        print(f"  {t/steps:6.2f} ms/step {100*t/tot:5.1f}%  {c/steps:5.1f}/step  {n[:70]}")
