#!/usr/bin/env python3
"""
Batch test PettingZoo Atari configurations with mocked LLM backend.
Verifies config loading, environment instantiation (ROMs), and game loop execution.
"""

import os
import glob
import sys
import yaml
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add src to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Imports from src
from loguru import logger
import gage_eval # trigger auto discovery
from gage_eval.evaluation.runtime_builder import build_runtime
from gage_eval.role.arena.types import ArenaAction 
from gage_eval.config.pipeline_config import PipelineConfig
from gage_eval.config import build_default_registry
from gage_eval.role.resource_profile import ResourceProfile, NodeResource

# Try importing pettingzoo, mock if missing (for CI/no-ROM envs)
try:
    import pettingzoo.atari
except (ImportError, ModuleNotFoundError):
    print("⚠️  PettingZoo Atari not found. Using Mock environment for verification.")
    from unittest.mock import MagicMock
    import sys
    
    # We also need a MockEnv that behaves like an AEC env
    class MockAtariEnv:
        def __init__(self, render_mode=None, **kwargs):
            self.possible_agents = ["player_0", "player_1"]
            self.agents = self.possible_agents[:]
            self.metadata = {"render_modes": ["human", "rgb_array"]}
            self._cumulative_rewards = {a: 0 for a in self.agents}
            self.rewards = {a: 0 for a in self.agents}
            self.terminations = {a: False for a in self.agents}
            self.truncations = {a: False for a in self.agents}
            self.infos = {a: {} for a in self.agents}
            self.agent_selection = "player_0"
            self.num_agents = len(self.agents)
        
        def action_space(self, agent):
            # Return a mock object with 'n' attribute (Discrete)
            # Atari usually has 18 actions
            space = MagicMock()
            space.n = 18
            # Also support contains
            space.contains.return_value = True
            return space

        def reset(self, seed=None, options=None):
            self.agents = self.possible_agents[:]
            self.agent_selection = self.agents[0]
            self.rewards = {a: 0 for a in self.agents}
            self.terminations = {a: False for a in self.agents}
            self.truncations = {a: False for a in self.agents}
            self.infos = {a: {} for a in self.agents}
        
        def step(self, action):
            if not self.agents:
                return
            idx = self.agents.index(self.agent_selection)
            self.agent_selection = self.agents[(idx + 1) % len(self.agents)]
            
        def observe(self, agent):
            # Return dummy observation compatible with existing parser
            # Assuming parser expects some structure, but usually raw is fine
            return {"observation": [[0]]} 
            
        def last(self):
            return (self.observe(self.agent_selection), 0, False, False, {})
            
        def close(self):
            pass

    def mock_env_creator(**kwargs):
        return MockAtariEnv(**kwargs)

    # Mock pettingzoo package
    m_top = MagicMock()
    sys.modules["pettingzoo"] = m_top
    
    m_atari = MagicMock()
    sys.modules["pettingzoo.atari"] = m_atari
    
    # Scan configs to know which modules to mock
    # We do this lazily or upfront. Upfront is better.
    # config_dir_scan = os.path.join(os.path.dirname(__file__), "../config/custom/pettingzoo")
    config_dir_scan = REPO_ROOT / "config/custom/pettingzoo"
    if os.path.exists(config_dir_scan):
        for fname in os.listdir(config_dir_scan):
            if fname.endswith(".yaml"):
                # We can't easily parse yaml here without loading it, but we can guess or just parse
                try:
                    with open(os.path.join(config_dir_scan, fname), "r") as f:
                        # Simple grep for env_id
                        for line in f:
                            if "env_id:" in line:
                                env_id = line.split("env_id:")[1].strip()
                                # Mock this module
                                mod_mock = MagicMock()
                                mod_mock.env = mock_env_creator
                                sys.modules[env_id] = mod_mock
                                print(f"   Mocked {env_id}")
                                break
                except Exception:
                    pass

# configure logger
logger.remove()
logger.add(sys.stderr, level="WARNING")

CONFIG_DIR = str(REPO_ROOT / "config/custom/pettingzoo")

def mock_think(self, observation):
    """Mock thinking process that returns a random legal move."""
    import random
    # Need to handle case where legal_moves might be None or empty
    moves = observation.legal_moves
    if not moves:
        move = "0"
    else:
        move = str(random.choice(moves))
    return ArenaAction(player=self.name, move=move, raw=move)

def test_config(config_path):
    print(f"Testing {os.path.basename(config_path)} ... ", end="", flush=True)
    
    # Inject dummy api key for backend init check (since we mock execution anyway)
    os.environ["OPENAI_API_KEY"] = "sk-dummy-key-for-testing"

    try:
        # Load config manually
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
        
        # Patch raw dict data BEFORE creating frozen config object
        # 1. Force render_mode = None 
        # 2. Reduce max turns = 2
        
        for adapter in data.get("role_adapters", []):
            if adapter.get("role_type") == "arena":
                # Ensure params dict exists
                if "params" not in adapter:
                    adapter["params"] = {}
                params = adapter["params"]

                # Patch environment
                if "environment" in params:
                    env = params["environment"]
                    if "env_kwargs" not in env:
                        env["env_kwargs"] = {}
                    env["env_kwargs"]["render_mode"] = None
                    # Also clear legacy field just in case
                    if "render_mode" in env:
                        env["render_mode"] = None


                
                # Patch scheduler
                if "scheduler" in params:
                    params["scheduler"]["max_turns"] = 2

        # 3. Reduce dataset samples
        # Need to patch both "tasks" list in config
        for task in data.get("tasks", []):
            task["max_samples"] = 1

        pipeline_config = PipelineConfig.from_dict(data)
        
        # Build runtime
        registry = build_default_registry()
        # Ensure adapters are registered (they should be via auto-discovery)
        
        resource_profile = ResourceProfile(nodes=[NodeResource(node_id="local")])
        builder = build_runtime(pipeline_config, registry=registry, resource_profile=resource_profile)
        runtime = builder 
        
        # Patch LLMPlayer.think globally for this runtime execution
        from gage_eval.role.arena.players.llm_player import LLMPlayer
        
        with patch.object(LLMPlayer, 'think', side_effect=mock_think, autospec=True):
            # Run the pipeline
            runtime.run()
            
        print("✅ PASS")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: {str(e)}")
        # import traceback
        # traceback.print_exc()
        return False

def main():
    if not os.path.exists(CONFIG_DIR):
        print(f"Config dir {CONFIG_DIR} not found.")
        return

    configs = sorted(glob.glob(os.path.join(CONFIG_DIR, "*_ai.yaml")))
    print(f"Found {len(configs)} configurations.")
    
    passed = 0
    failed = 0
    
    for config_path in configs:
        if test_config(config_path):
            passed += 1
        else:
            failed += 1
            
    print("\n--- Test Summary ---")
    print(f"Total: {len(configs)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
