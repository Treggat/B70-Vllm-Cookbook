"""vLLM 0.30-0.32: accept a video sent the llama.cpp way. llama.cpp-style clients (our agent harness among them) put a clip on the
wire as an `image_url` part whose URL is `data:video/mp4;base64,...`; vLLM opens that as an image and answers
400 "cannot identify image file". Route such URLs to vLLM's own video parser. Idempotent; aborts if the anchor moved.
0.32 moved chat_utils.py from vllm/entrypoints/ to vllm/renderers/ (anchor text unchanged); both paths are tried.
Target tree: VLLM_ROOT (the vllm/ package dir of the install you are building) or argv[1]."""
import os
import sys
ROOT = os.environ.get("VLLM_ROOT") or (sys.argv[1] if len(sys.argv) > 1 else None)
if not ROOT or not os.path.isfile(os.path.join(ROOT, "__init__.py")):
    sys.exit("usage: VLLM_ROOT=/path/to/vllm python " + os.path.basename(__file__) + "  (the vllm package dir)")
CANDIDATES = [os.path.join(ROOT, "renderers/chat_utils.py"), os.path.join(ROOT, "entrypoints/chat_utils.py")]
MARK = "B70_IMAGE_URL_VIDEO"
old = ('    elif part_type in ("image_url", "input_image"):\n'
       '        str_content = cast(str, content)\n'
       '        mm_parser.parse_image(str_content, uuid)\n'
       '        modality = "image"\n')
new = ('    elif part_type in ("image_url", "input_image"):\n'
       '        str_content = cast(str, content)\n'
       '        if isinstance(str_content, str) and str_content.startswith("data:video/"):\n'
       '            # ' + MARK + ': a clip sent the llama.cpp way (image_url carrying a video data URL)\n'
       '            mm_parser.parse_video(str_content, uuid)\n'
       '            modality = "video"\n'
       '        else:\n'
       '            mm_parser.parse_image(str_content, uuid)\n'
       '            modality = "image"\n')
hits = []
for p in CANDIDATES:
    if os.path.isfile(p):
        t = open(p).read()
        if MARK in t or old in t:
            hits.append(p)
if not hits:
    sys.exit("image_url anchor not found in renderers/chat_utils.py or entrypoints/chat_utils.py; not patched")
P = hits[0]
t = open(P).read()
if MARK in t:
    print("image_url->video patch already applied to", P)
    sys.exit(0)
if t.count(old) != 1:
    sys.exit("image_url anchor count != 1 in " + P + "; not patched")
t = t.replace(old, new, 1)
compile(t, P, "exec")
open(P, "w").write(t)
print("image_url->video patch applied to", P)
