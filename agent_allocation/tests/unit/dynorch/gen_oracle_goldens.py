"""Generate ``tests/fixtures/dynorch/oracle_goldens.json`` with the transformers renderer (contract §4 oracle).

Run ONCE, offline, under the independent environment (the package under test is never imported here)::

    /orcd/home/002/mabdel03/conda_envs/asys_env/bin/python3.11 tests/unit/dynorch/gen_oracle_goldens.py

(transformers 5.9.0, jinja2 3.1.6). It reads ``tests/fixtures/dynorch/oracle_cases.json`` (template-facing message
dicts and OpenAI tool objects), renders every case with
``transformers.utils.chat_template_utils.render_jinja_template`` over the vendored template
``src/agent_allocation/api/templates/qwen3.8-27b.chat_template.jinja`` and writes the goldens (with the template's
sha256 and the library versions). ``test_template.py`` asserts ``dynorch.template.render_chat`` reproduces every
golden byte for byte and that ``\\u003c`` (HTML-escaped ``<``) never appears.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
PROJECT = HERE.parents[3]
TEMPLATE = PROJECT / "src" / "agent_allocation" / "api" / "templates" / "qwen3.8-27b.chat_template.jinja"
CASES = PROJECT / "tests" / "fixtures" / "dynorch" / "oracle_cases.json"
GOLDENS = PROJECT / "tests" / "fixtures" / "dynorch" / "oracle_goldens.json"
PINNED_SHA256 = "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"


def main() -> int:
    import jinja2
    import transformers
    from transformers.utils.chat_template_utils import render_jinja_template

    source = TEMPLATE.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest != PINNED_SHA256:
        print(f"template sha256 {digest} != pinned {PINNED_SHA256}; refusing to generate goldens", file=sys.stderr)
        return 2
    cases = json.loads(CASES.read_text(encoding="utf-8"))["cases"]
    rendered_cases = []
    for case in cases:
        kwargs = {"reasoning_effort": case["reasoning_effort"]}
        if case["enable_thinking"] is False:
            kwargs["enable_thinking"] = False
        text = render_jinja_template([case["messages"]], tools=case["tools"], chat_template=source.decode("utf-8"),
                                     add_generation_prompt=True, **kwargs)
        # transformers 5.x returns (rendered_per_conversation, generation_indices); unwrap to the one string
        while isinstance(text, (list, tuple)):
            text = text[0]
        if not isinstance(text, str):
            raise TypeError(f"unexpected render result type {type(text).__name__}")
        rendered_cases.append({"name": case["name"], "rendered": text})
    GOLDENS.write_text(json.dumps({
        "template_sha256": digest,
        "renderer": f"transformers {transformers.__version__} render_jinja_template, jinja2 {jinja2.__version__}, "
                    f"python {sys.version.split()[0]}",
        "cases_file": CASES.name,
        "cases": rendered_cases,
    }, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {GOLDENS} ({len(rendered_cases)} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
