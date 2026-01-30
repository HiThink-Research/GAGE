import importlib

import pytest


@pytest.mark.fast
def test_role_sandbox_entry_removed():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("gage_eval.role.sandbox.runtime")
