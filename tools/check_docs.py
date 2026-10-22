#!/usr/bin/env python3
"""Checks the cookbook against the rules in AGENTS.md. Run from anywhere: python tools/check_docs.py

- every relative Markdown link (and #anchor) resolves
- every recipe folder has its row on the front page and the standard pages
- every file in a recipe is covered by its README's Layout table (docs/ is linked from the page itself)
- no private hosts, local paths, internal names, secrets, .bak files or big binaries

Prints one line per problem and exits 1, or prints OK.
"""
import os
import re
import subprocess
import sys
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEXT_EXT = {".md", ".py", ".sh", ".json", ".jsonl", ".csv", ".out", ".log", ".txt", ".conf", ".yaml", ".yml",
            ".patch", ".example", ".toml", ".cfg"}
MAX_BYTES = 1_000_000
RECIPE_PAGES = ["README.md", "docs/RECIPE.md", "docs/PITFALLS.md"]

# (pattern, why). Matched against every text file, case-sensitive unless (?i).
BANNED = [
    (r"\b192\.168\.\d+\.\d+|\b10\.\d+\.\d+\.\d+\b", "private IP address"),
    (r"(?i)\b[A-Z]:\\(Users|AI)\\|/mnt/user/|/pc/workspace|/root/\.dsh", "local path"),
    (r"(?i)\bAI\.Tower\b|\bBIGBOI\b", "private host name"),
    (r"(?i)\bhearth\b", "internal tool name (say 'our router', or link the project and explain it)"),
    (r"\bDSH\b", "internal tool name (say 'our agent harness')"),
    (r"(?i)\bcard[- ][AB]\b", "internal card name (say 'the first card' / 'the second card')"),
    (r"(?i)\bour scratch dir", "a file the reader can't get (ship it in bench/ or drop the reference)"),
    (r"\bhf_[A-Za-z0-9]{30,}|\bsk-[A-Za-z0-9_-]{20,}|\bgh[pousr]_[A-Za-z0-9]{30,}", "token or API key"),
]


def files():
    out = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                                  cwd=ROOT, text=True)
    return sorted({f for f in out.split("\n") if f and os.path.exists(os.path.join(ROOT, f))})


def strip_code(md):
    md = re.sub(r"^```.*?^```", "", md, flags=re.S | re.M)
    return re.sub(r"`[^`\n]*`", "", md)


def slugs(md):
    seen, out = Counter(), set()
    for line in strip_code(md).splitlines():
        m = re.match(r"^#{1,6}\s+(.*?)\s*#*\s*$", line)
        if not m:
            continue
        s = re.sub(r"[^\w\- ]", "", m.group(1).lower().replace("`", "")).replace(" ", "-")
        out.add(s if not seen[s] else f"{s}-{seen[s]}")
        seen[s] += 1
    # headings with inline code: slug keeps the code text, so add those too
    for line in md.splitlines():
        m = re.match(r"^#{1,6}\s+(.*?)\s*$", line)
        if m:
            out.add(re.sub(r"[^\w\- ]", "", m.group(1).lower()).replace(" ", "-"))
    return out


def check_links(path, text, problems):
    for m in re.finditer(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)", strip_code(text)):
        url = m.group(1)
        if re.match(r"[a-z]+:", url):
            if "github.com/Treggat/B70-Vllm-Cookbook/blob" in url:
                problems.append(f"{path}: absolute link to this repo, use a relative one: {url}")
            continue
        target, _, anchor = url.partition("#")
        dest = os.path.normpath(os.path.join(os.path.dirname(path), target)) if target else path
        full = os.path.join(ROOT, dest)
        if not os.path.exists(full):
            problems.append(f"{path}: broken link {url}")
        elif anchor and dest.endswith(".md"):
            if anchor not in slugs(open(full, encoding="utf-8").read()):
                problems.append(f"{path}: no heading for #{anchor} in {dest}")


def layout_paths(readme):
    """Backticked paths in the first column of the '## Layout' table."""
    m = re.search(r"^## Layout\s*\n(.*?)(?=^## |\Z)", readme, flags=re.S | re.M)
    if not m:
        return None
    paths = []
    for row in m.group(1).splitlines():
        cell = row.strip().strip("|").split("|")[0] if row.startswith("|") else ""
        paths += re.findall(r"`([^`]+)`", cell)
    return paths


def main():
    problems = []
    all_files = files()
    front = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    recipes = sorted({f.split("/")[0] for f in all_files if f.count("/") == 1 and f.endswith("/README.md")})

    for f in all_files:
        full = os.path.join(ROOT, f)
        if os.path.getsize(full) > MAX_BYTES:
            problems.append(f"{f}: over {MAX_BYTES // 1000} kB, link to it instead of committing it")
        if re.search(r"\.(bak|orig|rej|so|safetensors|bin|gguf|pt)$|\.bak", f):
            problems.append(f"{f}: backup or binary file in the tree")
        ext = os.path.splitext(f)[1]
        if ext not in TEXT_EXT and not f.endswith((".gitignore", "LICENSE")) and "LICENSE" not in f \
                and "dist-info" not in f:
            continue
        try:
            text = open(full, encoding="utf-8").read()
        except UnicodeDecodeError:
            continue
        if f in ("tools/check_docs.py", "AGENTS.md", "CLAUDE.md"):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            for pat, why in BANNED:
                if re.search(pat, line):
                    problems.append(f"{f}:{n}: {why}: {line.strip()[:100]}")
        if ext == ".md":
            check_links(f, text, problems)

    for r in recipes:
        if f"]({r}/README.md)" not in front:
            problems.append(f"README.md: no front-page row linking {r}/README.md")
        for page in RECIPE_PAGES:
            if not os.path.exists(os.path.join(ROOT, r, page)):
                problems.append(f"{r}: missing {page}")
        readme = open(os.path.join(ROOT, r, "README.md"), encoding="utf-8").read()
        listed = layout_paths(readme)
        if listed is None:
            problems.append(f"{r}/README.md: no '## Layout' table")
            continue
        for p in listed:
            if not os.path.exists(os.path.join(ROOT, r, p.rstrip("/"))):
                problems.append(f"{r}/README.md: Layout lists {p}, which does not exist")
        covered = [p.rstrip("/") for p in listed] + ["README.md", "docs", ".gitignore"]
        for f in all_files:
            if not f.startswith(r + "/"):
                continue
            rel = f[len(r) + 1:]
            if not any(rel == c or rel.startswith(c + "/") for c in covered):
                problems.append(f"{r}/README.md: Layout table does not cover {rel}")

    for p in problems:
        print(p)
    print(f"{len(problems)} problem(s)" if problems else "OK")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
