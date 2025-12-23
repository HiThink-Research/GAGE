Legacy vLLM Backend Bugfix Report
================================

Scope
-----
- Bug 1: Multimodal dedup should preserve repeated use of the same image inside messages, while deduping cross-field duplicates.
- Bug 2: messages == [] should not trigger chat_template exceptions (list index out of range) in _render_with_processor.

Changes
-------
- legacy_vllm_backend.py: _merge_image_sources now dedups only cross-field repeats; message repeats are preserved.
- legacy_vllm_backend.py: _render_with_processor returns early on empty messages; empty list render falls back to prompt.
- Tests updated/added:
  - tests/backends/test_legacy_vllm_backend_multimodal.py
  - tests/integration/compatibility/backends/test_legacy_vllm_backend_multimodal.py
  - tests/backends/test_multimodal_processor_chat_template.py

Test Log
--------
Command:
PYTHONPATH=src .venv/bin/python -m unittest tests/backends/test_legacy_vllm_backend_multimodal.py tests/integration/compatibility/backends/test_legacy_vllm_backend_multimodal.py tests/backends/test_multimodal_processor_chat_template.py

Output:
/Users/shuo/code/GAGE/src/gage_eval/registry/manager.py:210: RuntimeWarning: [registry] Failed to import gage_eval.role.model.backends.faiss_backend: No module named 'numpy'
/Users/shuo/code/GAGE/src/gage_eval/registry/manager.py:210: RuntimeWarning: [registry] Failed to import gage_eval.role.model.backends.multi_provider_http_backend: No module named 'huggingface_hub'
/Users/shuo/code/GAGE/src/gage_eval/registry/manager.py:210: RuntimeWarning: [registry] Failed to import gage_eval.role.model.backends.nanotron_backend: No module named 'yaml'
/Users/shuo/code/GAGE/src/gage_eval/registry/manager.py:210: RuntimeWarning: [registry] Failed to import gage_eval.assets.datasets.loaders.hf_hub_loader: No module named 'huggingface_hub'
/Users/shuo/code/GAGE/src/gage_eval/__init__.py:30: RuntimeWarning: Failed to auto-discover 'model_hubs' assets from gage_eval.assets.models.hubs: No module named 'huggingface_hub'
...........
----------------------------------------------------------------------
Ran 11 tests in 0.007s

OK
