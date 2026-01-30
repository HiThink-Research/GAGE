#!/usr/bin/env python3
"""
Replay PettingZoo game from Arena move log.

Usage:
    python scripts/replay_pettingzoo.py <sample_path> [delay_ms]

Example:
    python scripts/replay_pettingzoo.py \
      runs/boxing_ai/samples/task_pettingzoo_boxing_ai/sample.json 100
"""
import json
import time
import sys
import importlib
from pathlib import Path


def replay_game(sample_path: str, delay_ms: int = 500):
    """Replay a PettingZoo game from an Arena sample file.
    
    Args:
        sample_path: Path to the Arena sample JSON file
        delay_ms: Delay between moves in milliseconds (default: 500)
    """
    # Load sample
    sample_file = Path(sample_path)
    if not sample_file.exists():
        print(f"Error: Sample file not found: {sample_path}")
        sys.exit(1)
    
    with open(sample_file) as f:
        sample = json.load(f)
    
    # Extract game result from sample structure
    # Structure: sample -> sample.predict_result[0] -> game_log
    sample_data = sample.get("sample", {})
    predict_results = sample_data.get("predict_result", [])
    
    if not predict_results or not isinstance(predict_results, list):
        print("Error: No predict_result array found in sample.sample")
        sys.exit(1)
    
    game_result = predict_results[0]
    game_log = game_result.get("game_log", [])
    
    # Extract env_id and task_id
    metadata = sample_data.get("metadata", {})
    env_id_from_metadata = metadata.get("game_id")
    task_id = sample.get("task_id", "")
    
    # IMPORTANT: The test data's metadata.game_id is often stale/wrong.
    # Infer the correct game from task_id instead (e.g., "pettingzoo_boxing_ai" -> "boxing")
    env_id = None
    if task_id and task_id.startswith("pettingzoo_"):
        # Extract game name from task_id: "pettingzoo_boxing_ai" -> "boxing"
        parts = task_id.replace("pettingzoo_", "")
        for suffix in ("_ai", "_dummy", "_litellm"):
            parts = parts.replace(suffix, "")
        game_name = parts
        
        # Look up the correct version from known games
        GAME_VERSIONS = {
            "basketball_pong": "v3", "boxing": "v2", "combat_plane": "v2", 
            "combat_tank": "v2", "double_dunk": "v3", "entombed_competitive": "v3", 
            "entombed_cooperative": "v3", "flag_capture": "v2", "foozpong": "v3", 
            "ice_hockey": "v2", "joust": "v3", "mario_bros": "v3", "maze_craze": "v3", 
            "othello": "v3", "pong": "v3", "space_invaders": "v2", "space_war": "v2", 
            "surround": "v2", "tennis": "v3", "video_checkers": "v4", 
            "volleyball_pong": "v3", "wizard_of_wor": "v3"
        }
        version = GAME_VERSIONS.get(game_name, "v2")
        env_id = f"pettingzoo_atari_{game_name}_{version}"
        print(f"Inferred game from task_id: {game_name} (version {version})")
    
    # Fallback to metadata if task_id inference failed
    if not env_id:
        env_id = env_id_from_metadata
    
    if not env_id:
        print("Error: Could not determine game from task_id or metadata")
        sys.exit(1)
    
    if not game_log:
        print("Warning: Game log is empty")
        return
    
    print(f"Replaying: {env_id}")
    print(f"Total moves: {len(game_log)}")
    print(f"Delay: {delay_ms}ms per step")
    print("-" * 50)
    
    # Parse env_id (e.g., "pettingzoo_atari_space_invaders_v2" or "pettingzoo.atari.boxing_v2")
    try:
        # Handle both formats: pettingzoo_atari_game_v2 and pettingzoo.atari.game_v2
        if "_v" in env_id:
            # Format: pettingzoo_atari_boxing_v2
            parts = env_id.rsplit("_v", 1)
            game_name = parts[0].replace("pettingzoo_atari_", "")
            version = "v" + parts[1]
            module_path = f"pettingzoo.atari.{game_name}_{version}"
        else:
            # Format: pettingzoo.atari.boxing_v2
            parts = env_id.rsplit("_", 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid env_id format: {env_id}")
            module_path, version = parts[0], parts[1]
        
        # Import and create environment with human rendering
        module = importlib.import_module(module_path)
        env = module.env(render_mode="human")
        env.reset()
        
    except Exception as e:
        print(f"Error initializing environment: {e}")
        sys.exit(1)
    
    # Replay moves
    try:
        for idx, entry in enumerate(game_log, 1):
            action_id = entry.get("action_id")
            player = entry.get("player", "?")
            move_str = entry.get("move", "?")
            
            if action_id is not None:
                print(f"[{idx}/{len(game_log)}] {player}: {move_str} (action_id={action_id})")
                env.step(action_id)
                env.render()
                time.sleep(delay_ms / 1000.0)
            else:
                print(f"[{idx}/{len(game_log)}] {player}: Skipped (action_id=None)")
        
        print("-" * 50)
        print("Replay completed!")
        input("Press Enter to close...")
        
    except KeyboardInterrupt:
        print("\nReplay interrupted by user")
    except Exception as e:
        print(f"\nError during replay: {e}")
    finally:
        try:
            env.close()
        except:
            pass


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nError: Missing required argument <sample_path>")
        sys.exit(1)
    
    sample_path = sys.argv[1]
    delay_ms = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    
    if delay_ms < 0:
        print("Error: delay_ms must be >= 0")
        sys.exit(1)
    
    replay_game(sample_path, delay_ms)


if __name__ == "__main__":
    main()
