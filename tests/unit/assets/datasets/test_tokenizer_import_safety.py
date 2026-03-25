from __future__ import annotations

import builtins

import pytest

from gage_eval.assets.datasets.utils.tokenizers import get_or_load_tokenizer


@pytest.mark.fast
def test_get_or_load_tokenizer_skips_transformers_auto_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        normalized = str(name or "")
        if normalized == "transformers" or normalized.startswith("transformers."):
            raise AssertionError("tokenizer loading must not auto-import transformers")
        if normalized == "torch" or normalized.startswith("torch."):
            raise AssertionError("tokenizer loading must not auto-import torch")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)

    tokenizer = get_or_load_tokenizer({"tokenizer_path": "dummy_tok"})

    assert tokenizer is None
