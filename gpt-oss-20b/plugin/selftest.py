"""Offline self-test for the gptoss031 vLLM plugin (vLLM 0.31 and 0.32). No model load, no GPU.

Run it with the python of the vLLM venv that will serve the model (it needs that venv's vllm, xgrammar and
openai_harmony, plus the gpt-oss tokenizer), with this folder on PYTHONPATH:
    PYTHONPATH=$PWD GPTOSS_MODEL_DIR=/path/to/gpt-oss-20b $VENV/venv/bin/python selftest.py [name-filter]
GPTOSS_MODEL_DIR may also be a Hugging Face id (default openai/gpt-oss-20b).

Group A checks the grammar itself against the real tokenizer, token by token, built the way vLLM's
xgrammar backend builds it. Group B checks the hooks into vLLM.
"""
import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
MODEL_DIR = os.environ.get("GPTOSS_MODEL_DIR", "openai/gpt-oss-20b")

# An agent harness's run_code tool as it arrives on the wire: strict false, no additionalProperties.
RUN_CODE = {
    "name": "run_code",
    "strict": False,
    "parameters": {
        "type": "object",
        "required": ["code", "description"],
        "properties": {"code": {"type": "string"}, "description": {"type": "string"}},
    },
}
READ = {
    "name": "read",
    "strict": False,
    "parameters": {"type": "object", "required": ["file_path"], "properties": {"file_path": {"type": "string"}}},
}
GOOD_BODY = '{"code":"print(1)","description":"d"}'
CHANNEL_FIRST = "<|channel|>commentary to=functions.run_code <|constrain|>json<|message|>"
CALL, RETURN, END = "<|call|>", "<|return|>", "<|end|>"
STOP_IDS = [200002, 199999, 200012]  # gpt-oss generation_config eos_token_id

_cache = {}


def tokenizer():
    if "tok" not in _cache:
        from transformers import AutoTokenizer

        _cache["tok"] = AutoTokenizer.from_pretrained(MODEL_DIR)
    return _cache["tok"]


def vocab_size():
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(MODEL_DIR).vocab_size


def compiler():
    """Same construction as vllm/v1/structured_output/backend_xgrammar.py for HF tokenizers."""
    if "compiler" not in _cache:
        import xgrammar as xgr

        info = xgr.TokenizerInfo.from_huggingface(tokenizer(), vocab_size=vocab_size())
        _cache["compiler"] = xgr.GrammarCompiler(info, max_threads=4)
    return _cache["compiler"]


def matcher(tag):
    import xgrammar as xgr

    # vLLM 0.31 backend_xgrammar: GrammarMatcher(ctx, override_stop_tokens=<request stop ids>)
    return xgr.GrammarMatcher(compiler().compile_structural_tag(json.dumps(tag.model_dump())),
                              override_stop_tokens=STOP_IDS)


def ids_of(text):
    return tokenizer().encode(text, add_special_tokens=False)


def feed(m, text):
    """Feed text as real token ids. Returns None if every token was accepted, else the rejected piece."""
    for token in ids_of(text):
        if not m.accept_token(token):
            return tokenizer().decode([token])
    return None


def allowed(m, token_text):
    import xgrammar as xgr

    (token,) = ids_of(token_text)
    bitmask = xgr.allocate_token_bitmask(1, vocab_size())
    m.fill_next_token_bitmask(bitmask, 0)
    return bool((int(bitmask[0, token // 32]) >> (token % 32)) & 1)


def tool_tag(tools=(RUN_CODE, READ), tool_choice="auto", level="parameter"):
    from gptoss031.tags import build_tool_call_tag

    return build_tool_call_tag(list(tools), tool_choice=tool_choice, level=level)


# ---------------------------------------------------------------- group A: the grammar


def test_tag_compiles_for_a_run_code_schema():
    assert matcher(tool_tag()) is not None


def test_accepts_channel_first_header_with_constrain_token():
    m = matcher(tool_tag())
    assert feed(m, CHANNEL_FIRST + GOOD_BODY) is None


def test_accepts_channel_first_header_with_plain_json_marker():
    m = matcher(tool_tag())
    assert feed(m, "<|channel|>commentary to=functions.run_code json<|message|>" + GOOD_BODY) is None


def test_accepts_recipient_first_header():
    m = matcher(tool_tag())
    assert feed(m, " to=functions.run_code<|channel|>commentary <|constrain|>json<|message|>" + GOOD_BODY) is None


def test_accepts_reasoning_before_the_tool_call():
    m = matcher(tool_tag())
    text = "<|channel|>analysis<|message|>I should run code.<|end|><|start|>assistant" + CHANNEL_FIRST + GOOD_BODY
    assert feed(m, text) is None


def test_accepts_code_with_escaped_quotes_newlines_and_backslashes():
    code = 'import re\nprint("""hello "world" """)\npath = "C:\\\\Users\\\\x"\nm = re.compile(r"\\d+\\.\\d+")\n'
    body = json.dumps({"code": code, "description": "tricky strings"})
    m = matcher(tool_tag())
    assert feed(m, CHANNEL_FIRST + body) is None


def test_rejects_unescaped_quote_inside_the_code_string():
    m = matcher(tool_tag())
    assert feed(m, CHANNEL_FIRST) is None
    assert feed(m, '{"code":"print("hi")","description":"d"}') is not None


def test_rejects_raw_newline_inside_the_code_string():
    m = matcher(tool_tag())
    assert feed(m, CHANNEL_FIRST) is None
    assert feed(m, '{"code":"a = 1\nb = 2","description":"d"}') is not None


def test_rejects_a_tool_name_that_was_not_offered():
    m = matcher(tool_tag())
    assert feed(m, "<|channel|>commentary to=functions.bogus <|constrain|>json<|message|>") is not None


def test_parameter_level_rejects_a_body_missing_a_required_field():
    m = matcher(tool_tag(level="parameter"))
    assert feed(m, CHANNEL_FIRST) is None
    assert feed(m, '{"description":"d"}') is not None


def test_function_level_accepts_any_json_object_for_a_nonstrict_tool():
    m = matcher(tool_tag(level="function"))
    assert feed(m, CHANNEL_FIRST + '{"anything":[1,2,{"k":"v"}]}') is None


def test_function_level_still_pins_the_schema_of_a_strict_tool():
    strict_tool = dict(RUN_CODE, strict=True)
    m = matcher(tool_tag(tools=(strict_tool,), level="function"))
    assert feed(m, CHANNEL_FIRST) is None
    assert feed(m, '{"description":"d"}') is not None


def test_a_stop_token_is_allowed_once_the_call_body_is_complete():
    m = matcher(tool_tag())
    assert feed(m, CHANNEL_FIRST + GOOD_BODY) is None
    usable = [t for t in (CALL, RETURN) if allowed(m, t)]
    print("      stop tokens allowed after a complete call:", usable)
    assert usable, "neither <|call|> nor <|return|> is allowed: the model could never stop"


def test_end_token_is_not_allowed_after_a_complete_call_body():
    # "<|end|>" here left only the EOS <|return|> legal next, and vLLM's harmony parser
    # streamed that stray <|return|> as message text (the "<|return|>" leak).
    m = matcher(tool_tag())
    assert feed(m, CHANNEL_FIRST + GOOD_BODY) is None
    assert not allowed(m, END), "<|end|> after a call body brings back the <|return|> text leak"


def test_end_token_is_not_allowed_after_a_final_answer():
    m = matcher(tool_tag(tool_choice="auto"))
    assert feed(m, "<|channel|>final<|message|>Hello there, no tool needed.") is None
    print("      <|end|> allowed inside a final answer's text:", allowed(m, END))
    assert allowed(m, RETURN) or allowed(m, CALL)


def test_no_stop_token_is_allowed_in_the_middle_of_the_body():
    m = matcher(tool_tag())
    assert feed(m, CHANNEL_FIRST + '{"code":"print(1)"') is None
    assert not allowed(m, CALL) and not allowed(m, RETURN)


def test_auto_allows_a_plain_final_answer():
    m = matcher(tool_tag(tool_choice="auto"))
    assert feed(m, "<|channel|>final<|message|>Hello there, no tool needed.") is None
    assert allowed(m, RETURN) or allowed(m, CALL)


def test_required_rejects_a_plain_final_answer():
    m = matcher(tool_tag(tool_choice="required"))
    assert feed(m, "<|channel|>final<|message|>") is not None


def test_required_still_accepts_a_tool_call():
    m = matcher(tool_tag(tool_choice="required"))
    assert feed(m, CHANNEL_FIRST + GOOD_BODY) is None


def test_final_only_tag_constrains_the_answer_to_the_json_schema():
    from xgrammar.structural_tag import JSONSchemaFormat

    from gptoss031.tags import build_final_tag

    schema = {"type": "object", "required": ["n"], "properties": {"n": {"type": "integer"}}}
    m = matcher(build_final_tag(JSONSchemaFormat(json_schema=schema)))
    text = '<|channel|>analysis<|message|>thinking<|end|><|start|>assistant<|channel|>final<|message|>{"n": 3}'
    assert feed(m, text) is None
    m2 = matcher(build_final_tag(JSONSchemaFormat(json_schema=schema)))
    assert feed(m2, '<|channel|>final<|message|>{"n": "three"}') is not None




def test_call_can_end_on_call_token_and_not_on_end_token_under_031_stop_ids():
    """Why the port exists: 0.31's own tag ends calls on ["<|end|>", ""]; ours on the stop token only."""
    m = matcher(tool_tag())
    assert feed(m, CHANNEL_FIRST + GOOD_BODY) is None
    assert allowed(m, CALL), "<|call|> must be allowed to end a call under vLLM 0.31's stop tokens"
    assert not allowed(m, END)


# ---------------------------------------------------------------- group B: the vLLM 0.31 hooks


def chat_request(**overrides):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    def wire(tool):
        return {"type": "function", "function": {"name": tool["name"], "description": "x",
                                                 "parameters": tool["parameters"], "strict": tool["strict"]}}

    body = {"model": "gpt-oss", "messages": [{"role": "user", "content": "hi"}],
            "tools": [wire(RUN_CODE), wire(READ)], "tool_choice": "auto"}
    body.update(overrides)
    return ChatCompletionRequest(**body)


class FakeRenderer:
    seen = []

    def adjust_harmony_request(self, request):
        FakeRenderer.seen.append(("adjust", request.structured_outputs))

    def _make_request_with_harmony(self, request, should_include_tools=True):
        FakeRenderer.seen.append(("render", request.reasoning_effort))
        return "ok"


def test_hook_points_exist_in_031():
    from openai_harmony import StreamableParser
    from vllm.renderers.online_renderer import OnlineRenderer
    for name in ("adjust_harmony_request", "_make_request_with_harmony"):
        assert callable(getattr(OnlineRenderer, name, None)), name
    assert callable(StreamableParser.process)


def test_adjust_hook_adds_our_compilable_tag_for_nonstrict_tools():
    from gptoss031.vllm_plugin import install, uninstall
    install("function", "medium", guard=False, renderer=FakeRenderer)
    try:
        req = chat_request()
        FakeRenderer().adjust_harmony_request(req)
        so = req.structured_outputs
        assert so is not None and so.structural_tag, so
        compiler().compile_structural_tag(so.structural_tag)
    finally:
        uninstall()


def test_tag_survives_to_sampling_params():
    from gptoss031.vllm_plugin import constrain_request
    req = constrain_request(chat_request(), "function")
    sp = req.to_sampling_params(max_tokens=10, default_sampling_params={})
    assert sp.structured_outputs is not None and sp.structured_outputs.structural_tag


def test_no_tools_and_tool_choice_none_are_left_alone():
    from gptoss031.vllm_plugin import constrain_request
    assert constrain_request(chat_request(tools=None, tool_choice=None), "function").structured_outputs is None
    assert constrain_request(chat_request(tool_choice="none"), "function").structured_outputs is None


def test_level_off_adds_nothing():
    from gptoss031.vllm_plugin import constrain_request
    assert constrain_request(chat_request(), "off").structured_outputs is None


def test_named_tool_choice_offers_only_that_tool():
    import xgrammar as xgr

    from gptoss031.vllm_plugin import constrain_request
    req = constrain_request(chat_request(tool_choice={"type": "function", "function": {"name": "read"}}), "function")
    m = xgr.GrammarMatcher(compiler().compile_structural_tag(req.structured_outputs.structural_tag),
                           override_stop_tokens=STOP_IDS)
    assert feed(m, "<|channel|>commentary to=functions.run_code <|constrain|>json<|message|>") is not None


def test_render_hook_maps_and_caps_effort():
    from gptoss031.vllm_plugin import install, uninstall
    for cap, sent, want in (("high", "none", "low"), ("high", "high", "high"), ("medium", "high", "medium"),
                            ("medium", "xhigh", "medium"), ("medium", "low", "low")):
        install("function", cap, guard=False, renderer=FakeRenderer)
        try:
            FakeRenderer()._make_request_with_harmony(chat_request(reasoning_effort=sent))
            assert FakeRenderer.seen[-1] == ("render", want), (cap, sent, FakeRenderer.seen[-1])
        finally:
            uninstall()


def test_harmony_guard_turns_a_header_error_into_a_skipped_token():
    from openai_harmony import HarmonyEncodingName, HarmonyError, Role, StreamableParser, load_harmony_encoding

    from gptoss031.vllm_plugin import _installed, install, uninstall
    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    # the shape seen live on 0.31 without a grammar: a header that never closes properly
    bad = ("<|start|>assistant<|channel|>commentary to=functions.read need to actually check file contents"
           "<|end|><|start|>assistant<|channel|>commentary<|message|>ok<|end|>")
    toks = ids_of(bad)

    def run():
        p = StreamableParser(enc, role=None)
        for t in toks:
            p.process(t)

    try:
        run()
        unguarded_raises = False
    except HarmonyError:
        unguarded_raises = True
    install("function", "high", guard=True, renderer=FakeRenderer)
    try:
        run()  # must not raise
        print("      unguarded parser raised:", unguarded_raises, "| tokens skipped by the guard:",
              _installed.get("guarded", 0))
    finally:
        uninstall()


def test_install_on_real_classes_and_uninstall_restores():
    from openai_harmony import StreamableParser
    from vllm.renderers.online_renderer import OnlineRenderer

    from gptoss031.vllm_plugin import install, uninstall
    before = (OnlineRenderer.adjust_harmony_request, OnlineRenderer._make_request_with_harmony,
              StreamableParser.process)
    names = install("function", "medium", guard=True)
    assert len(names) == 3, names
    assert install("function", "medium", guard=True) == names
    uninstall()
    after = (OnlineRenderer.adjust_harmony_request, OnlineRenderer._make_request_with_harmony,
             StreamableParser.process)
    assert after == before


def test_entry_point_is_discoverable():
    from importlib.metadata import entry_points
    eps = [ep for ep in entry_points(group="vllm.general_plugins") if ep.name == "gptoss031"]
    assert eps and eps[0].value == "gptoss031.vllm_plugin:register", eps


if __name__ == "__main__":
    flt = sys.argv[1] if len(sys.argv) > 1 else ""
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_") and flt in n]
    passed = 0
    for n, f in tests:
        try:
            f()
            passed += 1
            print("PASS", n)
        except Exception:  # noqa: BLE001
            print("FAIL", n)
            traceback.print_exc(limit=4)
    print(f"{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
