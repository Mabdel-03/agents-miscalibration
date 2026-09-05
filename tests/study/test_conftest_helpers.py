"""The conftest helpers themselves: registry entry shape, stub tokenizer, run-root skeleton."""

from __future__ import annotations

import json
from pathlib import Path

from agents_scaling.serving.registry import ServerEntry, entry_matches_profile, list_servers
from tests.study.conftest import RUN_ROOT_SUBDIRS


def test_tmp_run_root_skeleton(tmp_run_root: Path):
    assert all((tmp_run_root / name).is_dir() for name in RUN_ROOT_SUBDIRS)


def test_register_fake_server_matches_profile(tmp_run_root: Path, register_fake_server):
    path = register_fake_server(tmp_run_root, "32B-long", "node3904", 8001)
    assert path == tmp_run_root / "servers" / "32B-long" / "node3904_8001.json"
    entry = ServerEntry(**json.loads(path.read_text()))
    assert entry_matches_profile(entry, "32B-long")
    assert not entry_matches_profile(entry, "14B-long")
    assert entry.slurm_job_id is None and entry.tp_size == 2 and entry.max_model_len == 40960
    assert [e.host for e in list_servers(tmp_run_root, "32B-long")] == ["node3904"]
    small = register_fake_server(tmp_run_root, "4B-long", port=8002)
    assert entry_matches_profile(ServerEntry(**json.loads(small.read_text())), "4B-long")


def test_stub_tokenizer(stub_tokenizer):
    ids = stub_tokenizer.encode("a b a")
    assert len(ids) == 3 and ids[0] == ids[2] != ids[1]
    messages = [{"role": "user", "content": "hello world"}]
    thinking_on = stub_tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                     enable_thinking=True, return_dict=False)
    thinking_off = stub_tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                      enable_thinking=False, return_dict=False)
    assert isinstance(thinking_on, list) and len(thinking_off) > len(thinking_on)
    as_dict = stub_tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    assert as_dict["input_ids"] == stub_tokenizer.apply_chat_template(messages, return_dict=False)
    text = stub_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    assert text.startswith("<|im_start|>user\nhello world<|im_end|>")


def test_real_tokenizer_matches_stub_interface(real_tokenizer_32b):
    messages = [{"role": "user", "content": "2+2?"}]
    ids = real_tokenizer_32b.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False
    )
    assert isinstance(ids, list) and len(ids) > 5
    # transformers 5.x default: a dict with input_ids (the stub mirrors this)
    as_dict = real_tokenizer_32b.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                     enable_thinking=False)
    assert list(as_dict["input_ids"]) == ids
    # enable_thinking=False appends the empty think block <think>\n\n</think>\n\n
    text = real_tokenizer_32b.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)
    assert text.endswith("<think>\n\n</think>\n\n")
    assert real_tokenizer_32b.encode("hello world", add_special_tokens=False)
