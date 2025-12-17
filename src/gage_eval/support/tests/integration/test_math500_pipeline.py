import os
import sys
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import yaml
import datasets

# Add src to path
SRC_ROOT = Path(__file__).resolve().parents[4]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gage_eval.config.pipeline_config import PipelineConfig
from gage_eval.evaluation.runtime_builder import build_runtime
from gage_eval.config import build_default_registry
from gage_eval.role.resource_profile import NodeResource, ResourceProfile
from gage_eval.observability.trace import ObservabilityTrace

@pytest.mark.not_gpu
@pytest.mark.not_network
def test_math500_pipeline_mocked(tmp_path):
    """Integration test for Math500 pipeline with mocked backend and loader."""
    
    # 1. Load Config
    config_path = SRC_ROOT.parent / "config/custom/math500_openai.yaml"
    if not config_path.exists():
        # Try finding it relative to current execution
        config_path = Path("gage-eval-main/config/custom/math500_openai.yaml")
        if not config_path.exists():
            pytest.skip(f"Config file not found at {config_path}")
    
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    # 2. Modify Config Dict
    if config_dict.get("tasks"):
        config_dict["tasks"][0]["max_samples"] = 2
    
    if config_dict.get("backends"):
        backend_cfg = config_dict["backends"][0].get("config", {})
        backend_cfg["api_key"] = "mock-key"
        backend_cfg["base_url"] = "http://mock-url"
        backend_cfg["model"] = "mock-model"
        config_dict["backends"][0]["config"] = backend_cfg
    
    config = PipelineConfig.from_dict(config_dict)
    
    # Set output dir
    save_dir = tmp_path / "runs"
    os.environ["GAGE_EVAL_SAVE_DIR"] = str(save_dir)
    os.environ["GAGE_EVAL_THREADS"] = "1"
    
    # 3. Mocks
    mock_data = [
        {
            "problem": "Calculate 1+1.",
            "answer": "2",
            "solution": "1+1=2",
            "subject": "math",
            "level": 1,
            "unique_id": "test-1"
        },
        {
            "problem": "Calculate 2*2.",
            "answer": "4",
            "solution": "2*2=4",
            "subject": "math",
            "level": 1,
            "unique_id": "test-2"
        }
    ]
    
    def side_effect(inputs, **kwargs):
        choices = []
        for inp in inputs:
            user_msg = ""
            # Handle various input types (str or list of messages)
            if isinstance(inp, str):
                user_msg = inp
            elif isinstance(inp, list):
                # If list of strings
                if len(inp) > 0 and isinstance(inp[0], str):
                    user_msg = " ".join(inp)
                else:
                    # List of messages (objects or dicts)
                    for msg in inp:
                        role = getattr(msg, 'role', None) or (msg.get('role') if isinstance(msg, dict) else None)
                        if role == 'user':
                             content = getattr(msg, 'content', None) or (msg.get('content') if isinstance(msg, dict) else None)
                             if isinstance(content, list):
                                 # List of MessageContent
                                 item = content[0]
                                 if hasattr(item, 'text'):
                                     user_msg = item.text
                                 elif isinstance(item, dict):
                                     user_msg = item.get('text', '')
                                 else:
                                     user_msg = str(item)
                             else:
                                 user_msg = str(content)
                             break
            
            text = ""
            if "1+1" in user_msg:
                text = "The answer is \boxed{2}."
            elif "2*2" in user_msg:
                text = "The answer is \boxed{4}."
            else:
                text = "\boxed{0}."
            
            # Mimic OpenAI Choice structure or similar that backend returns
            # And wrapped in a dict
            choices.append({"message": {"role": "assistant", "content": text}, "index": 0})
        
        # Return a dict because BaseBackend calls result.setdefault
        return {"choices": choices, "model_output": choices}

    with patch("gage_eval.assets.datasets.loaders.hf_hub_loader._load_or_cache_dataset") as mock_fetch, \
         patch("gage_eval.role.model.backends.openai_http_backend.OpenAICompatibleHTTPBackend.generate") as mock_generate:
        
        mock_fetch.return_value = (datasets.Dataset.from_list(mock_data), "/tmp/mock_cache")
        mock_generate.side_effect = side_effect
        
        # 4. Run
        registry = build_default_registry()
        profile = ResourceProfile(nodes=[NodeResource(node_id="local", gpus=0, cpus=1)])
        trace = ObservabilityTrace(run_id="test_run_math500")
        
        runtime = build_runtime(
            config=config,
            registry=registry,
            resource_profile=profile,
            trace=trace
        )
        
        runtime.run()
        
    # 5. Verify
    assert mock_fetch.called
    assert mock_generate.called
    assert mock_generate.call_count >= 1
